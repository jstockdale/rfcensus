/*
 * lora_demod.c — main entry point: frame sync state machine, lifecycle,
 *                streaming input plumbing.
 *
 * Copyright (c) 2026, Off by One. BSD-3-Clause.
 *
 * The frame sync state machine — the heart of the receiver — is
 * implemented as a co-routine driven by lora_demod_process_cf(). Each
 * call advances the state machine across whatever symbols have been
 * fed in:
 *
 *   DETECT       : while not yet 8 consecutive matching upchirps,
 *                  step one sample at a time looking for the start
 *                  of a preamble. Once we see N_PREAMBLE upchirps in
 *                  a row landing in the same FFT bin (within ±1),
 *                  we've found a likely preamble.
 *
 *   SYNC_NETID   : the next 2 symbols carry the network ID (sync
 *                  word, 0x2B for Meshtastic). If those match within
 *                  tolerance we commit. If not, return to DETECT.
 *
 *   SYNC_DOWN    : 2.25 downchirps mark the end of the preamble.
 *                  We use these to refine the integer + fractional
 *                  CFO and STO (sample-time offset).
 *
 *   DEMOD        : pure symbol-by-symbol decode. After enough symbols
 *                  for the header we know the payload length+CR, then
 *                  read remaining symbols, then call codec layer.
 *
 *  Tolerance and safety nets in this state machine matter a LOT.
 *  Real-world IQ is messy: AGC ringing, partial preambles when the
 *  capture starts mid-packet, false detects on noise. The state
 *  machine has to bail back to DETECT cleanly on every failure mode.
 */

#include "lora_internal.h"
#include "lora_dechirp_simd.h"   /* v0.7.11: vectorized complex multiply */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* LoRa preamble is 8 upchirps for "standard" preamble length.
 * Meshtastic uses 16 (longer = more time for sync). We accept either:
 * any 8 consecutive upchirps in the same bin counts as a preamble. */
#define N_PREAMBLE_DETECT 8

/* Bin tolerance during preamble detection. With low SNR, CFO drift,
 * and small STO error, the peak may wander between consecutive
 * symbols. ±2 bins gives reasonable robustness without accepting too
 * much noise. */
#define PREAMBLE_BIN_TOL  4

/* Energy gate: a "chirp" is only counted if the FFT peak magnitude
 * exceeds noise by this ratio. Conservative — false negatives are
 * cheap (we just keep searching), false positives waste cycles
 * sliding into SYNC and bailing out. */
#define DETECT_PEAK_RATIO 3.0f

/* DETECT search step. When we have NO candidate (last_bin < 0),
 * step by N/4 — fine enough that we're guaranteed to land within
 * one symbol of any real preamble start. When we DO have a candidate
 * but the next symbol breaks alignment, we slide back by N/4 and
 * retry — this lets us recover from ½-symbol misalignment without
 * abandoning a real preamble. */
#define DETECT_STEP_COARSE_DIV 4   /* step = N / 4 when searching */

/* ────────────────────────────────────────────────────────────────── */

void lora_ring_read(const struct lora_demod *d, uint64_t off, cf_t *dst) {
    /* Logical offset → physical ring position. The ring is power-of-2
     * sized so we can mask. */
    uint32_t N = d->N;
    uint32_t base = (uint32_t)(off & d->ring_mask);
    if (base + N <= d->ring_size) {
        memcpy(dst, &d->ring[base], N * sizeof(cf_t));
    } else {
        uint32_t first = d->ring_size - base;
        memcpy(dst, &d->ring[base], first * sizeof(cf_t));
        memcpy(&dst[first], &d->ring[0], (N - first) * sizeof(cf_t));
    }
}

/* Round up to the next power of 2 (32-bit). */
static uint32_t next_pow2(uint32_t v) {
    if (v <= 1) return 1;
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

/* ────────────────────────────────────────────────────────────────────
 * Lifecycle
 * ────────────────────────────────────────────────────────────────── */

lora_demod_t *lora_demod_new(const lora_config_t *cfg,
                             lora_decode_cb_t cb,
                             void *userdata) {
    if (!cfg || cfg->sf < LORA_MIN_SF || cfg->sf > LORA_MAX_SF) return NULL;
    if (cfg->bandwidth != LORA_BW_125 &&
        cfg->bandwidth != LORA_BW_250 &&
        cfg->bandwidth != LORA_BW_500) return NULL;
    if (cfg->sample_rate_hz < cfg->bandwidth) {
        /* Need at least 1 sample per BW period to recover the chirp. */
        return NULL;
    }

    struct lora_demod *d = (struct lora_demod *)calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->cfg = *cfg;
    d->cb = cb;
    d->userdata = userdata;
    d->N = 1u << cfg->sf;
    /* Integer decimation factor. If sample_rate is not an exact
     * multiple of bandwidth we lose a small amount of precision in
     * the dechirp reference (the FFT bins land slightly off the LoRa
     * symbol grid). For Meshtastic-style bandwidths (125/250/500 kHz),
     * use sample rates that ARE integer multiples (1 Msps, 2 Msps,
     * etc.) to avoid this. The decoder will work otherwise but with
     * reduced sensitivity. */
    d->os_factor = cfg->sample_rate_hz / cfg->bandwidth;
    if (d->os_factor < 1) d->os_factor = 1;

    /* Mix oscillator init. Per the docstring, mix_freq_hz =
     * (capture_freq - signal_freq), so positive means the signal sits
     * BELOW the capture center and we need to shift it UP to land at
     * baseband. Shift-up by f Hz = multiply by exp(+j·2π·f·n/Fs).
     * Phasor starts at 1+0j; renormalization every 8192 samples
     * corrects accumulated float drift. */
    d->mix_enabled = (cfg->mix_freq_hz != 0);
    if (d->mix_enabled) {
        double w = +2.0 * M_PI * (double)cfg->mix_freq_hz
                   / (double)cfg->sample_rate_hz;
        d->mix_step = (float)cos(w) + (float)sin(w) * I;
        d->mix_phasor = 1.0f + 0.0f * I;
    } else {
        d->mix_step = 1.0f + 0.0f * I;
        d->mix_phasor = 1.0f + 0.0f * I;
    }
    d->mix_norm_counter = 0;

    /* Resampler init. Step = input_rate / bw. For clean integer
     * oversample (e.g. 1M / 250k = 4) this gives integer step;
     * fractional step (2.4M / 250k = 9.6) handled by linear interp.
     * Start position at 0; first output emitted when we've consumed
     * at least 1 input sample (so we have prev to interpolate from). */
    d->resamp_step = (double)cfg->sample_rate_hz / (double)cfg->bandwidth;
    d->resamp_pos = 0.0;
    d->resamp_have_prev = false;

    d->upchirp   = (cf_t *)calloc(d->N, sizeof(cf_t));
    d->downchirp = (cf_t *)calloc(d->N, sizeof(cf_t));
    /* v0.7.9: FFT in/out buffers must be 16-byte aligned for the
     * pffft SIMD path. lora_fft_aligned_alloc returns suitable
     * memory regardless of which backend is compiled in. */
    d->fft_in    = (lora_fft_cpx_t *)lora_fft_aligned_alloc(
                       (size_t)d->N * sizeof(lora_fft_cpx_t));
    d->fft_out   = (lora_fft_cpx_t *)lora_fft_aligned_alloc(
                       (size_t)d->N * sizeof(lora_fft_cpx_t));
    if (!d->upchirp || !d->downchirp || !d->fft_in || !d->fft_out) {
        lora_demod_free(d); return NULL;
    }
    /* Zero-init the FFT buffers — lora_fft_aligned_alloc doesn't
     * (unlike calloc), and stale memory in unused tail bins would
     * pollute later transforms if N rounds up to a SIMD multiple. */
    memset(d->fft_in,  0, (size_t)d->N * sizeof(lora_fft_cpx_t));
    memset(d->fft_out, 0, (size_t)d->N * sizeof(lora_fft_cpx_t));
    d->fft_ctx = lora_fft_new(d->N);
    if (!d->fft_ctx) { lora_demod_free(d); return NULL; }
    lora_build_chirps(d->N, d->upchirp, d->downchirp);

    /* Ring sized to hold ≥ 16 symbols of look-back (preamble + sync
     * word + downchirps + safety margin), all at the BW-rate after
     * decimation. */
    uint32_t want = d->N * 32;
    d->ring_size = next_pow2(want);
    d->ring_mask = d->ring_size - 1;
    d->ring = (cf_t *)calloc(d->ring_size, sizeof(cf_t));
    if (!d->ring) { lora_demod_free(d); return NULL; }

    /* Symbol buffer: max packet at SF12 = ~140 symbols, round up. */
    d->demod_symbols = (uint16_t *)calloc(512, sizeof(uint16_t));
    if (!d->demod_symbols) { lora_demod_free(d); return NULL; }

    d->state = LORA_STATE_DETECT;
    return d;
}

void lora_demod_free(lora_demod_t *d) {
    if (!d) return;
    free(d->upchirp);
    free(d->downchirp);
    /* v0.7.9: FFT buffers came from lora_fft_aligned_alloc — must
     * use lora_fft_aligned_free, not plain free(). pffft uses an
     * over-allocation trick to align so plain free() corrupts the
     * heap. */
    if (d->fft_in)  lora_fft_aligned_free(d->fft_in);
    if (d->fft_out) lora_fft_aligned_free(d->fft_out);
    if (d->fft_ctx) lora_fft_destroy(d->fft_ctx);
    free(d->ring);
    free(d->demod_symbols);
    free(d);
}

void lora_demod_reset(lora_demod_t *d) {
    if (!d) return;
    d->state = LORA_STATE_DETECT;
    d->consec_upchirps = 0;
    d->last_bin = -1;
    d->cfo_int = 0;
    d->cfo_frac = 0.0f;
    d->symbols_collected = 0;
    d->symbols_needed = 0;
    d->header_decoded = false;
}

void lora_demod_get_stats(const lora_demod_t *d, lora_demod_stats_t *out) {
    if (d && out) *out = d->stats;
}

/* ────────────────────────────────────────────────────────────────────
 * Sample input — handles oversample decimation, ring write, then
 * invokes the state machine pump.
 * ────────────────────────────────────────────────────────────────── */

/* Decimate from input rate to BW rate, with mix applied first.
 * Returns the number of BW-rate samples produced.
 *
 * Pipeline per input sample:
 *   1. Cast cu8/cf to complex float
 *   2. If mix enabled: multiply by mix_phasor, advance phasor
 *   3. Track sample index. Emit interpolated output samples whenever
 *      our fractional position resamp_pos is between the previous and
 *      current input sample.
 *
 * Fractional resampling: emit output sample[k] = lerp(input[i],
 * input[i+1], frac) where i = floor(k * step) and frac = k*step - i.
 * This is implemented incrementally: we walk input samples one at a
 * time, and each iteration we may emit zero or more output samples
 * (depending on how big step is — for 9.6× decimation we emit ~once
 * per 10 inputs).
 *
 * The mix MUST be applied at the input rate (before resampling) — if
 * we resampled first, anything outside [-bw/2, +bw/2] would alias and
 * the mix would translate aliased noise into our band.
 */
static uint32_t ingest_samples_cf(struct lora_demod *d,
                                  const float *iq, size_t n_complex) {
    uint32_t produced = 0;
    for (size_t i = 0; i < n_complex; i++) {
        cf_t s = iq[2*i] + iq[2*i + 1] * I;
        if (d->mix_enabled) {
            s = s * d->mix_phasor;
            d->mix_phasor = d->mix_phasor * d->mix_step;
            d->mix_norm_counter++;
            if ((d->mix_norm_counter & 8191) == 0) {
                float mag = sqrtf(crealf(d->mix_phasor) * crealf(d->mix_phasor)
                                + cimagf(d->mix_phasor) * cimagf(d->mix_phasor));
                if (mag > 1e-6f) d->mix_phasor /= mag;
            }
        }

        /* Resampler: we just received the (input_index)th input sample.
         * Emit any output samples whose position falls in
         * [input_index-1, input_index]. The position resamp_pos
         * tracks the next output sample's input-coordinate; advance it
         * by step after each emission.
         *
         * On the very first input sample we have no prev to
         * interpolate against — just stash it and continue.
         */
        if (!d->resamp_have_prev) {
            d->resamp_prev = s;
            d->resamp_have_prev = true;
            /* resamp_pos starts at 0; the first output sample IS
             * input[0]. Emit it directly. */
            d->ring[d->ring_w & d->ring_mask] = s;
            d->ring_w++;
            produced++;
            d->resamp_pos = d->resamp_step;  /* next output position */
            continue;
        }

        /* The current input sample has integer position
         * (input_count - 1) in input-sample coordinates, where
         * input_count is the total samples processed so far.
         * (We don't track input_count explicitly; we know prev was at
         * pos = ceil(resamp_pos - step) - 1 and current is at the
         * next integer.) Track our position implicitly: every input
         * sample increments our notion of "current input pos" by 1.
         *
         * Implementation: maintain resamp_pos in units of "samples
         * since prev". When prev was just-now consumed, resamp_pos is
         * how far past prev we want our next output sample. If <= 1,
         * the next output is between prev and current (interpolate).
         * Then advance resamp_pos by step (still measured from the
         * SAME prev) and check again — could emit multiple outputs
         * per input if step < 1 (upsampling, not our case).
         *
         * After processing all outputs for this input pair, we
         * subtract 1 from resamp_pos (current becomes the new prev,
         * shifting our origin forward by 1).
         */
        while (d->resamp_pos <= 1.0) {
            float frac = (float)d->resamp_pos;
            cf_t out = d->resamp_prev * (1.0f - frac) + s * frac;
            d->ring[d->ring_w & d->ring_mask] = out;
            d->ring_w++;
            produced++;
            d->resamp_pos += d->resamp_step;
        }
        d->resamp_pos -= 1.0;
        d->resamp_prev = s;
    }
    return produced;
}

/* ────────────────────────────────────────────────────────────────────
 * State machine
 * ────────────────────────────────────────────────────────────────── */

/* Try to decode the current packet from the symbols collected. Called
 * once we've buffered enough symbols to cover header+payload. */
static void try_decode_packet(struct lora_demod *d) {
    uint8_t cr_out = 0, has_crc_out = 0;
    bool crc_ok = false;
    uint8_t buf[LORA_MAX_PAYLOAD];
    int n = lora_decode_payload(d->demod_symbols, d->symbols_collected,
                                d->cfg.sf, /*cr placeholder*/4,
                                d->cfg.ldro != 0,
                                buf, sizeof(buf),
                                &cr_out, &has_crc_out, &crc_ok);
    if (n < 0) {
        d->stats.headers_failed++;
        return;
    }
    d->stats.headers_decoded++;
    if (crc_ok) d->stats.packets_decoded++;
    else        d->stats.packets_crc_failed++;

    if (d->cb) {
        lora_decoded_t out = {0};
        memcpy(out.payload, buf, n);
        out.payload_len = (uint16_t)n;
        out.cr = cr_out;
        out.has_crc = has_crc_out;
        out.crc_ok = crc_ok ? 1 : 0;
        out.rssi_db = d->rssi_est;
        out.snr_db = d->snr_est;
        out.cfo_hz = (float)d->cfo_int * (float)d->cfg.bandwidth / (float)d->N
                   + d->cfo_frac;
        out.sample_offset = d->preamble_start;
        d->cb(&out, d->userdata);
    }
}

/* Advance the state machine using whatever's in the ring. Returns
 * number of packets decoded during this pump. */
static int pump_state_machine(struct lora_demod *d) {
    int pkts = 0;
    uint32_t N = d->N;
    /* We need at least N samples of look-back to demod a symbol. */
    while (d->ring_w >= d->read_cursor + N) {
        switch (d->state) {

        case LORA_STATE_DETECT: {
            /* Try a chirp at read_cursor. Track bin stability.
             *
             * Strategy: when we have no candidate, step by N/4 to
             * sweep the input quickly while still being guaranteed to
             * land within ½ symbol of any real preamble. When we have
             * a partial run going and the next symbol breaks alignment,
             * we DON'T immediately reset — we slide back by N/4 and
             * try again, in case we're slightly misaligned with the
             * symbol grid. Only after several failed retries do we
             * abandon the partial run and resume coarse search.
             */
            float mag = 0;
            uint16_t bin = lora_detect_chirp(d, d->read_cursor, &mag);
            d->stats.detect_attempts++;
            if (mag > d->stats.detect_peak_mag_max) d->stats.detect_peak_mag_max = mag;
            uint32_t coarse_step = N / DETECT_STEP_COARSE_DIV;
            if (coarse_step < 1) coarse_step = 1;

            /* Energy gate: peak must clear N × 0.04 in normalized
             * units. With cu8 input scaled to [-1,+1], a real LoRa
             * preamble at -20 dBFS produces FFT peaks ≈ 0.1 × N. The
             * 0.04 floor catches signals down to ~-28 dBFS while
             * mostly rejecting white noise (whose FFT peaks scatter
             * randomly with expected max ≈ √N for N=512..2048 = 22..45,
             * well below 0.04 × N = 20..82). */
            if (mag < (float)N * 0.04f) {
                /* Likely noise. Reset partial run and slide ahead. */
                d->read_cursor += coarse_step;
                d->consec_upchirps = 0;
                d->last_bin = -1;
                break;
            }
            d->stats.detect_above_gate++;
            if (d->last_bin < 0) {
                /* First candidate. Lock its bin as the reference and
                 * step by exactly N to land on the next expected
                 * chirp boundary. */
                d->last_bin = bin;
                d->consec_upchirps = 1;
                d->preamble_start = d->read_cursor;
                d->preamble_phase[0] = mag;
                d->read_cursor += N;
                break;
            }
            int bin_diff = (int)bin - d->last_bin;
            if (bin_diff > (int)N/2) bin_diff -= N;
            if (bin_diff < -(int)N/2) bin_diff += N;
            if (abs(bin_diff) <= PREAMBLE_BIN_TOL) {
                /* Same bin (within tolerance). Extend the run. */
                d->consec_upchirps++;
                if (d->consec_upchirps > d->stats.detect_max_run) {
                    d->stats.detect_max_run = d->consec_upchirps;
                }
                if (d->consec_upchirps - 1 < 8) {
                    d->preamble_phase[d->consec_upchirps - 1] = mag;
                }
                d->last_bin = bin;
                d->read_cursor += N;
                if (d->consec_upchirps >= N_PREAMBLE_DETECT) {
                    /* Got it. The CFO estimate is just the bin value
                     * itself (the consistent offset across all 8
                     * chirps tells us how far off the receiver is
                     * from the transmitter, modulo N). */
                    d->cfo_int = (int32_t)d->last_bin;
                    if (d->cfo_int > (int)N/2) d->cfo_int -= N;
                    d->stats.preambles_found++;
                    d->state = LORA_STATE_SYNC_NETID;
                    d->netid_match[0] = 0;
                    d->netid_match[1] = 0;
                    /* v0.7.7: compute RSSI + SNR from preamble. The
                     * preamble_phase[] array holds the dechirp peak
                     * magnitude for each of the 8 preamble symbols
                     * (filled in above as each chirp was confirmed).
                     * Average them for a stable RSSI estimate.
                     *
                     * RSSI: the FFT of an N-sample windowed unit-
                     * amplitude tone produces a peak of magnitude N.
                     * We normalize by N to get a 0-to-1 magnitude
                     * scale, then convert to dB. cu8 input already
                     * scaled to ±1 in the demod path, so 0 dB ≈
                     * full-scale RSSI; typical Meshtastic packets
                     * land between -25 and -5 dBFS.
                     *
                     * SNR: peak power vs the OFF-peak energy in the
                     * same FFT. We do one extra dechirp+FFT at the
                     * preamble_start position and compute
                     * peak_mag² / mean(other_bins²). This costs one
                     * extra FFT per detected packet — negligible
                     * given packets are rare and FFTs are fast. */
                    {
                        float sum_mag = 0.0f;
                        for (int pi = 0; pi < 8; pi++) {
                            sum_mag += d->preamble_phase[pi];
                        }
                        float avg_mag = sum_mag / 8.0f;
                        d->rssi_est = 20.0f * log10f(
                            (avg_mag / (float)N) + 1e-30f);

                        /* SNR: dechirp at preamble_start, compute
                         * peak² and total² minus peak². The bin
                         * count for noise = N-1. */
                        cf_t snr_buf[4096];
                        if (N <= 4096) {
                            lora_ring_read(d, d->preamble_start,
                                           snr_buf);
                            /* v0.7.11: vectorized dechirp */
                            lora_dechirp_mul(
                                (const float*)(const void*)snr_buf,
                                (const float*)(const void*)d->downchirp,
                                (float*)(void*)d->fft_in,
                                N
                            );
                            lora_fft_forward(d->fft_ctx, d->fft_in,
                                              d->fft_out);
                            float total_p = 0.0f;
                            float peak_p = 0.0f;
                            for (uint32_t k = 0; k < N; k++) {
                                float r = d->fft_out[k].r;
                                float ii = d->fft_out[k].i;
                                float p = r * r + ii * ii;
                                total_p += p;
                                if (p > peak_p) peak_p = p;
                            }
                            float noise_p = total_p - peak_p;
                            if (noise_p > 0.0f && peak_p > 0.0f) {
                                /* Per-bin noise power = noise_p/(N-1)
                                 * SNR = peak_p / (per-bin noise) */
                                float noise_per_bin = noise_p
                                                    / (float)(N - 1);
                                d->snr_est = 10.0f * log10f(
                                    peak_p / noise_per_bin);
                            } else {
                                d->snr_est = 0.0f;
                            }
                        }
                    }
                }
            } else {
                /* Bin jumped. Either: (a) noise that happened to
                 * exceed the gate, (b) we lost symbol-time alignment
                 * partway through a real preamble. We can't tell
                 * which, so abandon this run and resume the coarse
                 * search ONE coarse step past where the run started
                 * (not past where we currently are — we may have
                 * advanced N samples per detected chirp, so the real
                 * preamble could be slightly behind us). */
                uint64_t restart = d->preamble_start + coarse_step;
                if (restart <= d->read_cursor) {
                    d->read_cursor = restart;
                } else {
                    d->read_cursor += coarse_step;
                }
                d->consec_upchirps = 0;
                d->last_bin = -1;
            }
            break;
        }

        case LORA_STATE_SYNC_NETID: {
            /* The next two symbols encode the LoRa sync word. The
             * sync word is an 8-bit value split into two 4-bit nibbles;
             * each nibble is encoded as a SHIFT from the preamble bin
             * value, scaled by 8. So expected symbol values are:
             *   sym1 bin = (sync_word >> 4) * 8
             *   sym2 bin = (sync_word & 0xF) * 8
             * (modulo CFO, which is already in d->cfo_int.)
             *
             * v0.6.16 critical fix: Meshtastic uses a 16-chirp preamble
             * (vs LoRa's standard 8). Our DETECT only requires 8 in a
             * row, so when we transition to SYNC_NETID we may still be
             * INSIDE the preamble. We need to slide forward one symbol
             * at a time until we see a non-preamble bin (sync_word_1
             * has a SHIFTED bin value, not the preamble's bin-0-after-
             * CFO value). gr-lora handles this in its NET_ID1 case at
             * frame_sync_impl.cc:568-587.
             *
             * Algorithm: read sym at cursor. If bin is near 0 (within
             * a few of preamble's CFO-corrected position), it's more
             * preamble — slide forward by N. If bin is shifted (= sync
             * word), pin it as sym1 and read the next as sym2.
             *
             * Cap the slide at 16 extra symbols (covers Meshtastic's
             * preamble plus margin). After that, abandon as a false
             * preamble detection.
             */
            uint8_t expected_hi = (uint8_t)((d->cfg.sync_word >> 4) & 0xF);
            uint8_t expected_lo = (uint8_t)((d->cfg.sync_word     ) & 0xF);
            if (d->ring_w < d->read_cursor + N) return pkts;

            cf_t buf1[4096];
            lora_ring_read(d, d->read_cursor, buf1);
            float m1;
            uint16_t b1 = lora_symbol_demod(d, buf1, &m1);

            /* "preamble continuation" check: bin near 0 (within ±2)
             * means this symbol is another preamble upchirp, not the
             * sync word. Slide forward one symbol and keep looking.
             * The preamble_phase counter caps the slide. */
            int b1_signed = (int)b1;
            if (b1_signed > (int)N/2) b1_signed -= N;
            if (abs(b1_signed) <= 2) {
                /* Still in preamble. Slide forward by N. Use the
                 * netid_match[0] field as a slide counter (overloading
                 * an unused field). Cap at 16 extra symbols. */
                d->netid_match[0]++;
                if (d->netid_match[0] > 16) {
                    /* Way past where any reasonable preamble should
                     * end. False detection — bail to DETECT. */
                    d->state = LORA_STATE_DETECT;
                    d->consec_upchirps = 0;
                    d->last_bin = -1;
                    d->read_cursor += N;
                    break;
                }
                d->read_cursor += N;
                break;  /* stay in SYNC_NETID, try the next symbol */
            }

            /* Got a non-preamble bin. Read the second sync symbol.
             *
             * v0.6.16 empirical finding: the second sync symbol does
             * NOT start at cursor+N. Probing real Meshtastic captures
             * showed sym2 actually lives at cursor+N+N/4 — there's a
             * quarter-symbol gap between sync_word_1 and sync_word_2.
             * The probe data was conclusive: reading at cursor+N+0
             * gave bin (true_value - N/4); reading at cursor+N+N/4
             * gave bin = true_value exactly. Pattern reproduced across
             * 9 packets at SF9.
             *
             * Why the gap? Suspected cause: Meshtastic / SX127x adds
             * a quarter-symbol downchirp suffix to sync_word_1 (similar
             * to the QUARTER_DOWN at end of preamble) that gr-lora
             * handles in its preamble state machine but isn't called
             * out in the basic spec. Investigating further is a TODO
             * but the empirical fix is solid against this fixture. */
            if (d->ring_w < d->read_cursor + 2 * N + N / 4) return pkts;
            cf_t buf2[4096];
            lora_ring_read(d, d->read_cursor + N + N / 4, buf2);
            float m2;
            uint16_t b2 = lora_symbol_demod(d, buf2, &m2);

            /* v0.6.17: multi-sync candidate matching.
             *
             * The user runs both Meshtastic packets (sync 0x2B, public)
             * and may have other LoRa devices on the same band using
             * different sync words (0x12 private network is common,
             * 0x34 LoRaWAN public, etc). Refusing to decode anything
             * but our configured sync byte misses real packets.
             *
             * New approach: derive the on-air sync byte from the
             * recovered bins. Each bin should land near a multiple
             * of 8 (a nibble × 8 position). If both bins are within
             * tolerance of a valid nibble position, we synthesize the
             * sync byte = (nibble_hi << 4) | nibble_lo and proceed
             * to header decode regardless of whether it matches our
             * config's sync_word. Header CRC is the final arbiter:
             * if the packet wasn't really LoRa or used a different
             * format, the CRC will fail and we discard it.
             *
             * The configured cfg.sync_word still matters for stats
             * (we track which sync was matched vs NOT-matched) and
             * for higher-level routing later, but it no longer gates
             * decode attempts.
             */
            int e1 = (int)expected_hi * 8;
            int e2 = (int)expected_lo * 8;
            int diff1 = ((int)b1 - e1 + (int)N) % (int)N;
            int diff2 = ((int)b2 - e2 + (int)N) % (int)N;
            if (diff1 > (int)N/2) diff1 -= N;
            if (diff2 > (int)N/2) diff2 -= N;

            /* Round each bin to the nearest nibble*8 position to
             * derive what nibble it represents. With ±4 bin tolerance
             * we accept any bin within ±4 of a valid nibble*8 anchor. */
            int nib1 = ((int)b1 + 4) / 8;  /* round to nearest /8 */
            int nib2 = ((int)b2 + 4) / 8;
            int residual1 = (int)b1 - nib1 * 8;
            int residual2 = (int)b2 - nib2 * 8;
            int observed_sync = ((nib1 & 0xF) << 4) | (nib2 & 0xF);
            int observed_ok = (
                nib1 >= 0 && nib1 < 16 &&
                nib2 >= 0 && nib2 < 16 &&
                abs(residual1) <= 4 && abs(residual2) <= 4
            );

            /* Diagnostic: log first 8 sync attempts. Gated behind
             * LORA_DECODE_DEBUG=1 so production runs stay quiet. */
            static int sync_dbg = -1;
            if (sync_dbg < 0) {
                const char *env = getenv("LORA_DECODE_DEBUG");
                sync_dbg = (env && env[0] == '1') ? 1 : 0;
            }
            if (sync_dbg && d->stats.preambles_found <= 8) {
                fprintf(stderr,
                    "  [sync@%llu cur=%llu] bins=(%u,%u) cfg=(%d,%d) "
                    "diff=(%+d,%+d) cfo=%d mag=(%.0f,%.0f) slid=%d "
                    "observed=0x%02X residual=(%+d,%+d)\n",
                    (unsigned long long)d->preamble_start,
                    (unsigned long long)d->read_cursor,
                    b1, b2, e1, e2, diff1, diff2, d->cfo_int, m1, m2,
                    d->netid_match[0], observed_sync,
                    residual1, residual2);
            }

            int sync_ok = observed_ok;
            /* Lax-sync mode keeps the magnitude-only fallback for
             * deepest-debugging where bins are scrambled but signal
             * is present. */
            static int lax_sync = -1;
            if (lax_sync < 0) {
                const char *env = getenv("LORA_LAX_SYNC");
                lax_sync = (env && env[0] == '1') ? 1 : 0;
            }
            if (lax_sync && m1 > (float)N * 0.05f && m2 > (float)N * 0.05f) {
                sync_ok = 1;
            }

            if (sync_ok) {
                d->stats.syncwords_matched++;
                /* Stash observed sync byte so the consumer can decide
                 * what to do with it (e.g. "0x2B → meshtastic public",
                 * "0x12 → some other private network"). */
                d->observed_sync_word = (uint8_t)observed_sync;
                d->state = LORA_STATE_SYNC_DOWN;
                /* v0.6.18: advance by 2N (NOT 2N + N/4). The +N/4
                 * fix that v0.6.16 added was an empirical hack to
                 * compensate for what we now know is just STO
                 * (sample-time offset) — sym2's "apparent" position
                 * shifts when STO is non-zero. With proper STO
                 * refinement in SYNC_DOWN (joint solve from
                 * up_val/down_val), the cursor gets rewound by sto
                 * samples, restoring true alignment.
                 *
                 * Keep the +N/4 in the SYM2 READ (above) because we
                 * haven't computed STO yet at that point — the +N/4
                 * read shift was empirically the right amount to
                 * make sym2's bin land at expected_lo.
                 *
                 * Tracing: cursor at sync match = sym1_pos = T+16N+sto.
                 * +2N → T+18N+sto = downchirp1_start + sto. SYNC_DOWN
                 * reads downchirp here (still off by sto). Computes
                 * sto, rewinds → T+18N (= true downchirp1_start).
                 * +2.25N → T+20.25N (= true payload_start). DEMOD
                 * reads exactly aligned. */
                d->read_cursor += 2 * N;
                d->netid_match[0] = 0;  /* reset slide counter */
            } else {
                /* Bins don't round to valid nibble positions. Either
                 * noise or a malformed packet. Bail. */
                d->state = LORA_STATE_DETECT;
                d->consec_upchirps = 0;
                d->last_bin = -1;
                d->read_cursor += N;
                d->netid_match[0] = 0;
            }
            break;
        }

        case LORA_STATE_SYNC_DOWN: {
            /* End of preamble: 2.25 downchirps follow the sync word.
             *
             * STO/CFO refinement (per gr-lora frame_sync_impl.cc lines
             * 605-622, paraphrased):
             *
             * If we dechirp a DOWNCHIRP against the UPCHIRP reference
             * (instead of the downchirp reference we use for normal
             * symbol demod), the result is a tone whose bin position
             * encodes a combination of CFO_int and 2× the residual
             * symbol-time offset. For an aligned pure downchirp at
             * zero CFO, this would give bin 0. The deviation tells us
             * how to refine cfo_int:
             *
             *   if down_val < N/2:   cfo_int_refined = down_val / 2
             *   else:                cfo_int_refined = (down_val - N) / 2
             *
             * (Wrap convention: bins above N/2 represent negative
             * frequency offsets.)
             *
             * We only need ONE downchirp to do this refinement. The
             * other 1.25 downchirps we just skip past, but we use them
             * as a buffer in case the alignment is off by up to ¼
             * symbol either way.
             */
            if (d->ring_w < d->read_cursor + 2 * N) return pkts;
            cf_t buf[4096];
            lora_ring_read(d, d->read_cursor, buf);
            /* Dechirp by multiplying by UPCHIRP (= conj of downchirp).
             * This is the opposite of normal symbol demod (which
             * multiplies by downchirp). We do this inline since it's
             * a one-shot calculation. */
            for (uint32_t i = 0; i < N; i++) {
                cf_t dc = buf[i] * d->upchirp[i];
                d->fft_in[i].r = crealf(dc);
                d->fft_in[i].i = cimagf(dc);
            }
            lora_fft_forward(d->fft_ctx, d->fft_in, d->fft_out);
            float best = -1; uint16_t down_val = 0;
            for (uint32_t k = 0; k < N; k++) {
                float r = d->fft_out[k].r, ii = d->fft_out[k].i;
                float m2 = r*r + ii*ii;
                if (m2 > best) { best = m2; down_val = (uint16_t)k; }
            }
            float down_mag = sqrtf(best);

            /* v0.6.18: joint CFO + STO refinement.
             *
             * Math (derived in lora_demod.c.notes; see also gr-lora's
             * frame_sync_impl.cc:615 which only refines CFO):
             *
             *   up_val   = (sto + cfo_bins) mod N    (preamble dechirp)
             *   down_val = (cfo_bins - sto) mod N    (downchirp dechirp)
             *
             *   cfo_bins = (up_val + down_val) / 2
             *   sto      = (up_val - down_val) / 2
             *
             * Where:
             *   sto = sample-time offset (positive = our cursor is
             *         that many samples PAST true symbol start)
             *   cfo_bins = true CFO in units of FFT bins
             *
             * Both up_val and down_val are wrapped to (-N/2, +N/2] for
             * the arithmetic so the half-N ambiguity resolves cleanly.
             *
             * Why this matters: gr-lora ALSO does this implicitly via
             * the items_to_consume = N - k_hat step at preamble end,
             * which re-aligns the sample stream by k_hat samples
             * (where k_hat = the preamble's measured bin = sto+cfo
             * before refinement). Doing the joint solve here gives
             * us the same alignment without needing to track items_
             * to_consume separately.
             */
            int up_val = d->cfo_int;  /* the preamble bin we locked */
            int dv = (int)down_val;
            if (dv > (int)N / 2) dv -= N;        /* signed wrap */
            int uv = up_val;
            if (uv > (int)N / 2) uv -= N;
            if (uv < -(int)N / 2) uv += N;
            int cfo_bins = (uv + dv) / 2;
            int sto_samples = (uv - dv) / 2;

            /* v0.6.18 diagnostic: log downchirp peak vs N. If much
             * less than N, our downchirp position is misaligned (which
             * means subsequent payload reads will also be off). */
            static int sd_dbg = -1;
            if (sd_dbg < 0) {
                const char *env = getenv("LORA_DECODE_DEBUG");
                sd_dbg = (env && env[0] == '1') ? 1 : 0;
            }
            if (sd_dbg && d->stats.preambles_found <= 4) {
                fprintf(stderr,
                    "  [sync_down cur=%llu] down_val=%u (signed=%+d) "
                    "mag=%.0f (max=%u) up=%+d cfo=%+d sto=%+d\n",
                    (unsigned long long)d->read_cursor,
                    down_val, dv, down_mag, N, uv,
                    cfo_bins, sto_samples);
            }

            /* v0.6.18: estimate fractional CFO via Bernier's algorithm.
             *
             * Per gr-lora frame_sync_impl.cc:194-252: dechirp each of
             * the 8 preamble upchirps, FFT, look at the complex value
             * at the strongest bin (= preamble peak bin, NOT the
             * refined cfo_int — preamble samples haven't been STO-
             * corrected, so their FFT peak is still at sto+true_cfo).
             * The phase of this value rotates by 2π * cfo_frac per
             * symbol. Sum the complex products fft[i] * conj(fft[i+1])
             * across all 8 preamble symbols — argument of the sum
             * gives the average per-symbol phase rotation.
             *
             *   cfo_frac = -arg(four_cum) / (2π)
             *
             * cfo_frac is in units of bins (e.g. 0.4 = 40% of one
             * bin's worth of CFO). Range is (-0.5, 0.5] since cfo_int
             * captured the integer part.
             *
             * We compute Bernier BEFORE applying cfo_int_refined so
             * we still have the preamble peak bin (= up_val) on hand.
             */
            uint64_t pstart = d->preamble_start;
            uint32_t target_bin = (uint32_t)((up_val + (int)N) % (int)N);
            d->preamble_peak_bin = target_bin;
            cf_t four_cum = 0.0f + 0.0f * I;
            cf_t prev_fft = lora_dechirp_bin(d, pstart, target_bin);
            d->preamble_fft_at_peak[0] = prev_fft;
            for (int i = 1; i < 8; i++) {
                cf_t cur_fft = lora_dechirp_bin(
                    d, pstart + (uint64_t)i * N, target_bin);
                d->preamble_fft_at_peak[i] = cur_fft;
                /* fft[i] * conj(fft[i+1]) accumulates for one less
                 * iteration than the number of symbols (7 products
                 * for 8 symbols). gr-lora's convention is the same. */
                four_cum += prev_fft * conjf(cur_fft);
                prev_fft = cur_fft;
            }
            float cfo_frac_raw = -atan2f(cimagf(four_cum),
                                          crealf(four_cum)) /
                                  (2.0f * (float)M_PI);
            d->cfo_frac = cfo_frac_raw;

            static int bn_dbg = -1;
            if (bn_dbg < 0) {
                const char *env = getenv("LORA_DECODE_DEBUG");
                bn_dbg = (env && env[0] == '1') ? 1 : 0;
            }
            if (bn_dbg && d->stats.preambles_found <= 4) {
                fprintf(stderr,
                    "  [bernier pstart=%llu target_bin=%u] "
                    "four_cum=(%.1f,%.1f) cfo_frac=%.3f bins\n",
                    (unsigned long long)pstart, target_bin,
                    crealf(four_cum), cimagf(four_cum),
                    cfo_frac_raw);
            }

            /* Apply joint CFO + STO refinement.
             *
             * v0.6.18: removed the delta gate that v0.6.17 had. The
             * delta was always large because cfo_int_preamble = sto +
             * true_cfo while cfo_int_refined = true_cfo only. The
             * difference IS sto, which can easily exceed N/4 — exactly
             * the magnitude that the gate rejected. With the gate in
             * place, refinement was almost never applied, defeating
             * the whole point.
             *
             * Sanity check is now ONLY on downchirp magnitude (must be
             * > 0.4 N). A weak downchirp means we're not aligned with
             * a real downchirp; refinement would be garbage.
             */
            if (down_mag > (float)N * 0.4f &&
                abs(sto_samples) <= (int)N / 2) {
                d->cfo_int = cfo_bins;
                /* sto positive means cursor is LATE → rewind by sto. */
                if (sto_samples > 0 &&
                    (uint64_t)sto_samples <= d->read_cursor) {
                    d->read_cursor -= (uint64_t)sto_samples;
                } else if (sto_samples < 0) {
                    d->read_cursor += (uint64_t)(-sto_samples);
                }
            }

            /* Skip past the full 2.25 downchirps to land at DEMOD
             * start. The 0.25 quarter is a guard region — gr-lora's
             * state machine includes a QUARTER_DOWN state for fine
             * STO correction we don't yet implement. We absorb the
             * quarter into the skip and accept up to ±N/4 alignment
             * error, which the dechirp+FFT tolerates since each
             * symbol's energy spreads over ~N/4 bins under that much
             * STO error. Sensitivity loss: ~3 dB worst case; for the
             * Meshtastic LongFast preset (SF11, ~10 dB SNR margin)
             * this is acceptable. Full QUARTER_DOWN STO correction
             * is tracked for v0.6.17. */
            d->read_cursor += (uint32_t)(2.25f * (float)N);
            d->state = LORA_STATE_DEMOD;
            d->symbols_collected = 0;
            d->symbols_needed = 0;
            d->header_decoded = false;
            break;
        }

        case LORA_STATE_DEMOD: {
            /* Read one symbol.
             *
             * v0.6.18: removed the +N/4 read offset. With proper STO
             * refinement in SYNC_DOWN (joint solve from up_val/down_val),
             * the cursor is now aligned with true symbol boundaries —
             * no manual offset needed.
             *
             * v0.6.18: apply CFO_frac correction (Bernier-derived in
             * SYNC_DOWN). Multiply each sample by exp(-j 2π cfo_frac n / N)
             * to rotate out the residual fractional CFO. n is the
             * sample index WITHIN the current symbol (0..N-1). The
             * accumulated phase across samples-since-correction-was-
             * estimated would technically need to be tracked, but in
             * practice CFO is constant over the packet duration so a
             * per-symbol n=0..N-1 phase ramp is sufficient — gr-lora
             * does the same thing per symbol.
             */
            cf_t buf[4096];
            lora_ring_read(d, d->read_cursor, buf);
            if (d->cfo_frac != 0.0f) {
                float two_pi_cfo_frac_over_N = 2.0f * (float)M_PI *
                                                d->cfo_frac / (float)N;
                for (uint32_t n = 0; n < N; n++) {
                    float ang = -two_pi_cfo_frac_over_N * (float)n;
                    cf_t rot = cosf(ang) + sinf(ang) * I;
                    buf[n] *= rot;
                }
            }
            float mag;
            uint16_t bin = lora_symbol_demod(d, buf, &mag);
            d->demod_symbols[d->symbols_collected++] = bin;
            d->read_cursor += N;
            /* v0.6.18 diagnostic: log first 16 symbols of first
             * packet to verify alignment. Mag should approach N
             * (~512 at SF9) when properly aligned. Gated behind
             * LORA_DECODE_DEBUG=1. */
            static int demod_dbg = -1;
            if (demod_dbg < 0) {
                const char *env = getenv("LORA_DECODE_DEBUG");
                demod_dbg = (env && env[0] == '1') ? 1 : 0;
            }
            if (demod_dbg && d->stats.preambles_found == 1 &&
                d->symbols_collected <= 16) {
                fprintf(stderr,
                    "    [demod sym%d] bin=%u mag=%.0f\n",
                    d->symbols_collected, bin, mag);
            }

            /* v0.7.2: After collecting 8 (header) symbols, decode the
             * header to learn the EXACT total symbol count needed for
             * this packet. Without this, we'd wait for the worst-case
             * MAX_SYMS=320 symbols before attempting decode — adding
             * up to 1.3 seconds of latency for SF9 short packets and
             * even more for high SF.
             *
             * If the header decodes cleanly, set symbols_needed to the
             * exact count and emit when we hit it.
             *
             * If the header CRC fails (not a real packet — preamble was
             * a false positive on noise or a non-LoRa burst), abandon
             * immediately. The DETECT path will pick up any actual
             * preamble that comes later.
             *
             * This early-exit is critical for the lazy multi-decoder
             * pipeline: when the passband detector deactivates a slot
             * after a few hundred ms of silence, we need decoded packets
             * to have been emitted BEFORE teardown — not buffered
             * pending more samples that will never arrive. */
            if (d->symbols_needed == 0
                && d->symbols_collected == 8) {
                int needed = lora_compute_symbols_needed(
                    d->demod_symbols, 8,
                    d->cfg.sf, d->cfg.ldro != 0);
                if (needed < 0) {
                    /* Header CRC failed — bail. */
                    d->stats.headers_failed++;
                    d->symbols_collected = 0;
                    d->state = LORA_STATE_DETECT;
                    d->consec_upchirps = 0;
                    d->last_bin = -1;
                    break;
                }
                /* Clamp to our allocated buffer size (320) just in
                 * case the header reports something exotic. We never
                 * accept a value smaller than 8 (header size) either. */
                if (needed < 8) needed = 8;
                if (needed > 320) needed = 320;
                d->symbols_needed = (uint32_t)needed;
            }

            uint32_t target = d->symbols_needed
                              ? d->symbols_needed
                              : 320;
            if (d->symbols_collected >= target) {
                try_decode_packet(d);
                pkts++;
                d->state = LORA_STATE_DETECT;
                d->consec_upchirps = 0;
                d->last_bin = -1;
                d->symbols_needed = 0;
            }
            break;
        }

        default:
            /* Should never happen; reset for safety. */
            d->state = LORA_STATE_DETECT;
            d->consec_upchirps = 0;
            d->last_bin = -1;
            break;
        }
    }
    return pkts;
}

/* ────────────────────────────────────────────────────────────────── */

int lora_demod_process_cf(lora_demod_t *d, const float *iq, size_t n) {
    if (!d || !iq) return 0;
    d->stats.samples_processed += n;
    /* Initialize read_cursor on first ever call so it follows ring_w
     * from the start. */
    if (d->ring_w == 0 && d->read_cursor == 0) {
        /* nothing yet */
    }
    ingest_samples_cf(d, iq, n);
    return pump_state_machine(d);
}

int lora_demod_process_cu8(lora_demod_t *d, const uint8_t *iq, size_t n) {
    if (!d || !iq) return 0;
    /* Convert chunks of cu8 to float on the stack and feed in.
     * 8KB stack buffer = 4096 complex samples = generous chunk size
     * that keeps the FFT scratch hot in cache. */
    enum { CHUNK = 4096 };
    float fbuf[CHUNK * 2];
    int total = 0;
    while (n > 0) {
        size_t this_n = n > CHUNK ? CHUNK : n;
        for (size_t i = 0; i < this_n; i++) {
            /* cu8 from rtl_sdr is centered on 127.5. Normalize to
             * roughly [-1, 1] so the FFT magnitudes are comparable
             * across captures regardless of front-end gain. */
            fbuf[2*i]     = ((float)iq[2*i]     - 127.5f) / 127.5f;
            fbuf[2*i + 1] = ((float)iq[2*i + 1] - 127.5f) / 127.5f;
        }
        total += lora_demod_process_cf(d, fbuf, this_n);
        iq += this_n * 2;
        n  -= this_n;
    }
    return total;
}

/* v0.7.11: feed already-channelized baseband samples. The caller has
 * mixed the slot to DC and decimated to bandwidth rate; we just
 * write them straight to the ring (skipping the per-decoder mix
 * oscillator AND the linear-interp resampler in ingest_samples_cf)
 * and pump the state machine.
 *
 * This is the foundation for v0.7.11 channel filter sharing: instead
 * of N decoders at the same slot frequency each doing their own
 * mix + resample (N×O(samples)), upstream code does it once and
 * fans out the result. For BW=250 in Meshtastic that's 5 SF
 * decoders sharing one channelization, dropping 80% of the
 * channelization cost.
 *
 * Caller invariants:
 *   • Decoder constructed with mix_freq_hz=0 (no further mixing).
 *   • Decoder constructed with sample_rate_hz == bandwidth (resampler
 *     is a no-op since input rate == output rate).
 *   • Input samples are already at the slot's center frequency
 *     mixed to DC.
 */
int lora_demod_feed_baseband(lora_demod_t *d, const float *iq, size_t n) {
    if (!d || !iq) return 0;
    d->stats.samples_processed += n;
    /* Write directly to the ring buffer. No mix, no resample —
     * caller has already done both. */
    for (size_t i = 0; i < n; i++) {
        cf_t s = iq[2*i] + iq[2*i + 1] * I;
        d->ring[d->ring_w & d->ring_mask] = s;
        d->ring_w++;
    }
    return pump_state_machine(d);
}
