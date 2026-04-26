/*
 * lora_chirp.c — reference chirp generation and per-symbol FFT demod.
 *
 * Copyright (c) 2026, Off by One. BSD-3-Clause.
 *
 * The LoRa modulation is "chirp spread spectrum": each symbol is a
 * frequency sweep (a chirp) from -BW/2 to +BW/2, with a STARTING
 * offset that encodes the symbol value. With 2^SF distinct starting
 * offsets, each symbol carries SF bits.
 *
 * To demodulate: multiply the received signal by the conjugate of a
 * pure baseline upchirp ("dechirp"), which collapses the swept-
 * frequency signal down to a constant tone whose frequency is
 * proportional to the symbol value. An N-point FFT recovers the tone
 * frequency as a peak bin in 0..N-1, where N = 2^SF.
 *
 * This file implements:
 *   • lora_build_chirps()   — precompute up/downchirp reference (one shot)
 *   • lora_symbol_demod()   — per-symbol FFT-based bin recovery
 *   • lora_detect_chirp()   — same algorithm but used during preamble
 *                             detection to find any chirp regardless
 *                             of bin
 */

#include "lora_internal.h"
#include "lora_dechirp_simd.h"   /* v0.7.11: vectorized complex multiply */
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ────────────────────────────────────────────────────────────────────
 * Reference chirp generation
 * ────────────────────────────────────────────────────────────────────
 * A LoRa upchirp is a complex exponential whose phase is a quadratic
 * function of time, sweeping from -BW/2 to +BW/2 over the symbol
 * duration T_sym = N/BW (assuming sample_rate == BW, i.e. 1×
 * oversample after decimation).
 *
 *   upchirp[n]   = exp(j * 2π * ( -BW/2 * (n/BW) + (BW^2)/(2N) * (n/BW)^2 ))
 *                = exp(j * 2π * ( -n/2 + n^2 / (2N) ))    // simplified
 *
 * The downchirp is the complex conjugate.
 *
 * We compute these once per demod instance at construction time. N can
 * be up to 4096 (SF12) so the buffers are at most 32 KB each.
 */
void lora_build_chirps(uint32_t N, cf_t *upchirp, cf_t *downchirp) {
    /* Per Tapparel et al. and gr-lora's utilities.h::build_upchirp
     * with id=0 (the reference upchirp), the phase at sample n is:
     *
     *     phase[n] = 2π × (n² / (2N) - n/2)
     *              = π × (n²/N - n)
     *
     * Note this is MATHEMATICALLY DIFFERENT from a phase-accumulation
     * approach with dphase[0] = -π and ddphase = 2π/N — that produces
     * an extra -πn/N term equivalent to a half-bin frequency offset
     * which causes off-by-one errors in symbol demod (caught by
     * test_synth in Phase A). We compute directly from the formula
     * for correctness; double precision keeps the n² term accurate
     * at SF12 (n² up to 16M, well within double's ~16-digit mantissa).
     *
     * Downchirp is the complex conjugate.
     */
    double Nd = (double)N;
    for (uint32_t n = 0; n < N; n++) {
        double nd = (double)n;
        double phase = M_PI * (nd * nd / Nd - nd);
        float c = (float)cos(phase);
        float s = (float)sin(phase);
        upchirp[n]   = c + s * I;
        downchirp[n] = c - s * I;
    }
}

/* ────────────────────────────────────────────────────────────────────
 * Per-symbol demod
 * ────────────────────────────────────────────────────────────────────
 * Given N samples of a received chirp:
 *   1. Multiply pointwise by the reference downchirp (dechirp)
 *   2. N-point FFT
 *   3. argmax(|FFT|) gives the symbol value (0..N-1)
 *   4. Apply integer-CFO correction (subtract cfo_int mod N from the
 *      bin)
 *
 * The peak magnitude is also returned so the caller can use it for
 * SNR estimation and for filtering "no signal here" symbols during
 * preamble search.
 *
 * Assumes os_factor==1 (i.e. samples have already been decimated to
 * BW rate). The caller in lora_demod.c handles oversample decimation.
 */
uint16_t lora_symbol_demod(struct lora_demod *d,
                           const cf_t *samples,
                           float *peak_mag_out) {
    uint32_t N = d->N;
    /* v0.7.11: vectorized dechirp using NEON / SSE3 / scalar fallback.
     * cf_t is layout-compatible with interleaved float pairs (see
     * lora_internal.h note); the SIMD helper writes directly into
     * fft_in. Multiplies samples × downchirp pointwise. */
    lora_dechirp_mul(
        (const float*)(const void*)samples,
        (const float*)(const void*)d->downchirp,
        (float*)(void*)d->fft_in,
        N
    );
    lora_fft_forward(d->fft_ctx, d->fft_in, d->fft_out);

    /* Find peak bin by magnitude-squared (no need for sqrt). */
    float best_mag2 = -1.0f;
    uint16_t best_bin = 0;
    for (uint32_t k = 0; k < N; k++) {
        float r = d->fft_out[k].r;
        float i = d->fft_out[k].i;
        float m2 = r * r + i * i;
        if (m2 > best_mag2) {
            best_mag2 = m2;
            best_bin = (uint16_t)k;
        }
    }
    /* Integer-CFO correction: shift by cfo_int. The sign convention
     * here: positive cfo_int means received signal is HIGHER in
     * frequency than reference, so the dechirped tone lands in a
     * HIGHER bin than the true symbol value. Subtract to recover. */
    int32_t corrected = (int32_t)best_bin - d->cfo_int;
    /* Wrap modulo N (positive result) */
    corrected = ((corrected % (int32_t)N) + (int32_t)N) % (int32_t)N;
    if (peak_mag_out) *peak_mag_out = sqrtf(best_mag2);
    return (uint16_t)corrected;
}

/* ────────────────────────────────────────────────────────────────────
 * Detect a chirp during preamble search
 * ────────────────────────────────────────────────────────────────────
 * Same FFT-based dechirp used in symbol demod, but called from the
 * frame-sync detect loop. We don't apply CFO correction here (we're
 * trying to ESTABLISH the CFO). Returns the raw peak bin and magnitude
 * so the state machine can:
 *   • Track bin stability (preamble = 8 chirps that all land in the
 *     same bin within ±1, modulo CFO)
 *   • Use the bin offset itself as the CFO estimate
 *   • Use the magnitude for energy gating (skip bin spam from noise)
 */
uint16_t lora_detect_chirp(struct lora_demod *d,
                           uint64_t sample_offset,
                           float *peak_mag_out) {
    uint32_t N = d->N;
    /* Read N samples out of the ring at the given logical offset */
    cf_t buf[4096];   /* SF12 max */
    if (N > 4096) {
        /* Shouldn't happen — guarded at construction */
        if (peak_mag_out) *peak_mag_out = 0.0f;
        return 0;
    }
    lora_ring_read(d, sample_offset, buf);
    /* v0.7.11: vectorized dechirp + FFT */
    lora_dechirp_mul(
        (const float*)(const void*)buf,
        (const float*)(const void*)d->downchirp,
        (float*)(void*)d->fft_in,
        N
    );
    lora_fft_forward(d->fft_ctx, d->fft_in, d->fft_out);
    float best_mag2 = -1.0f;
    uint16_t best_bin = 0;
    for (uint32_t k = 0; k < N; k++) {
        float r = d->fft_out[k].r;
        float i = d->fft_out[k].i;
        float m2 = r * r + i * i;
        if (m2 > best_mag2) {
            best_mag2 = m2;
            best_bin = (uint16_t)k;
        }
    }
    if (peak_mag_out) *peak_mag_out = sqrtf(best_mag2);
    return best_bin;
}

/* v0.6.18: dechirp + FFT, return the complex value at a specific
 * bin. Used by Bernier's CFO_frac estimator which needs the complex
 * FFT bin (not just magnitude) to compute phase progression across
 * preamble symbols.
 *
 * Equivalent to lora_detect_chirp but returns a complex value at the
 * caller-specified bin. We re-do the full FFT here rather than trying
 * to share state with detect_chirp because the caller (SYNC_DOWN
 * Bernier loop) wants 8 separate FFTs for 8 separate preamble
 * positions. Cheaper to expose the helper than to thread a callback
 * into detect_chirp.
 */
cf_t lora_dechirp_bin(struct lora_demod *d,
                       uint64_t sample_offset,
                       uint32_t target_bin) {
    uint32_t N = d->N;
    cf_t buf[4096];
    if (N > 4096) return 0.0f + 0.0f * I;
    lora_ring_read(d, sample_offset, buf);
    /* v0.7.11: vectorized dechirp + FFT */
    lora_dechirp_mul(
        (const float*)(const void*)buf,
        (const float*)(const void*)d->downchirp,
        (float*)(void*)d->fft_in,
        N
    );
    lora_fft_forward(d->fft_ctx, d->fft_in, d->fft_out);
    if (target_bin >= N) return 0.0f + 0.0f * I;
    return d->fft_out[target_bin].r + d->fft_out[target_bin].i * I;
}

/* v0.7.11: backend name as a public symbol for Python diagnostics. */
const char* lora_dechirp_backend_name(void) {
    return lora_dechirp_backend();
}
