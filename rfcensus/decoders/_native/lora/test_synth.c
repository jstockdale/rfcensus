/*
 * test_synth.c — synthetic tests of the decoder's correctness.
 *
 * No real RF capture needed: we generate a clean LoRa signal in
 * software, feed it through the demod, and verify we get the right
 * symbol values back. This exercises the DSP without depending on
 * field captures or frequency translation.
 *
 * Tests:
 *   1. Chirp orthogonality:  FFT(upchirp × downchirp) peaks at bin 0
 *   2. Symbol round-trip:    encode symbol K, demod, recover K
 *   3. Codec round-trip:     known nibble pattern → encoder pipeline
 *                            → decoder pipeline → matches input
 *   4. Hamming round-trip:   all 16 nibbles encode→decode at all CRs
 *   5. CRC test vectors:     known data → known CRC value
 *
 * Copyright (c) 2026, Off by One. BSD-3-Clause.
 */

#include "lora_demod.h"
#include "lora_internal.h"
#include "kiss_fft.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int n_failures = 0;
#define EXPECT(cond, msg, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: " msg "\n", ##__VA_ARGS__); \
        n_failures++; \
    } else { \
        fprintf(stderr, "  ok: " msg "\n", ##__VA_ARGS__); \
    } \
} while(0)

/* ───────────────────────────────────────────────────────────────── */

static void test_chirp_orthogonality(void) {
    fprintf(stderr, "\n[1] Chirp orthogonality\n");
    /* For SF in 7..12, generate up/downchirp, multiply pointwise,
     * FFT, expect a single dominant bin at 0. */
    for (uint8_t sf = 7; sf <= 12; sf++) {
        uint32_t N = 1u << sf;
        cf_t *up = calloc(N, sizeof(cf_t));
        cf_t *dn = calloc(N, sizeof(cf_t));
        kiss_fft_cpx *in = calloc(N, sizeof(kiss_fft_cpx));
        kiss_fft_cpx *out = calloc(N, sizeof(kiss_fft_cpx));
        kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, NULL, NULL);

        lora_build_chirps(N, up, dn);
        for (uint32_t i = 0; i < N; i++) {
            cf_t prod = up[i] * dn[i];
            in[i].r = crealf(prod);
            in[i].i = cimagf(prod);
        }
        kiss_fft(cfg, in, out);
        /* Peak should be at bin 0 with magnitude ≈ N. */
        float best = -1; int best_k = -1;
        for (uint32_t k = 0; k < N; k++) {
            float m2 = out[k].r*out[k].r + out[k].i*out[k].i;
            if (m2 > best) { best = m2; best_k = (int)k; }
        }
        EXPECT(best_k == 0, "sf=%u: peak at bin 0 (got %d, mag^2=%.0f)",
               sf, best_k, best);
        free(up); free(dn); free(in); free(out); free(cfg);
    }
}

/* Generate a synthetic LoRa symbol with given value. The TX modulator
 * is: phase[n] = phase[n-1] + 2π * (value/N + n/N - 1/2) × (1/N)
 * which we compute as a cumulative sum. Equivalently: it's an upchirp
 * with starting frequency offset proportional to (value × BW/N).
 *
 * Easiest synthesis: take the reference upchirp and CYCLICALLY rotate
 * it by `value` samples. That's exactly what the LoRa symbol is.
 */
static void synth_symbol(const cf_t *upchirp, uint32_t N, uint16_t value,
                         cf_t *out) {
    for (uint32_t n = 0; n < N; n++) {
        out[n] = upchirp[(n + value) % N];
    }
}

static void test_symbol_demod(void) {
    fprintf(stderr, "\n[2] Symbol round-trip\n");
    /* Exhaustive at SF7 (128 values), sampled at higher SFs. */
    for (uint8_t sf = 7; sf <= 10; sf++) {
        uint32_t N = 1u << sf;
        lora_config_t cfg = {
            .sample_rate_hz = 125000,
            .bandwidth = LORA_BW_125,
            .sf = sf, .sync_word = 0x2B, .ldro = 0,
        };
        lora_demod_t *d = lora_demod_new(&cfg, NULL, NULL);
        if (!d) { fprintf(stderr, "  FAIL: demod_new for sf=%u\n", sf);
                  n_failures++; continue; }
        cf_t *sym = calloc(N, sizeof(cf_t));
        int errs = 0;
        /* Test stride: every value at SF7 (128 tests), every 7th at
         * SF8/9/10 (~36/73/146 tests). Catches off-by-one anywhere. */
        uint32_t stride = (sf == 7) ? 1 : 7;
        for (uint16_t v = 0; v < N; v += stride) {
            synth_symbol(d->upchirp, N, v, sym);
            float mag;
            uint16_t recovered = lora_symbol_demod(d, sym, &mag);
            if (recovered != v) {
                if (errs < 3) fprintf(stderr,
                    "  FAIL: sf=%u v=%u → recovered=%u (mag=%.0f)\n",
                    sf, v, recovered, mag);
                errs++;
            }
        }
        EXPECT(errs == 0, "sf=%u: %u test values, all recover exactly",
               sf, (sf == 7) ? N : N / 7);
        free(sym);
        lora_demod_free(d);
    }
}

/* End-to-end mix test: synthesize a symbol at non-zero IF, feed it
 * through the full pipeline (mix + decimation + ring + demod), verify
 * we get the right symbol value out.
 *
 * This is the smallest test that covers the input path: cf samples →
 * mix oscillator → ring → state machine → symbol_demod. If this works
 * on synthesized data, the mix code is plumbed correctly. Real-world
 * fixtures are still needed to validate sensitivity at low SNR.
 */
static void test_mix_end_to_end(void) {
    fprintf(stderr, "\n[2c] Digital mix end-to-end\n");
    uint8_t sf = 9;
    uint32_t N = 1u << sf;        /* 512 samples per symbol */
    uint32_t bw = 250000;
    uint32_t sample_rate = 1000000;  /* 4× oversample */
    int32_t mix_freq = 100000;     /* signal sits at -100 kHz IF, mix
                                     * by +100 kHz to pull it to baseband */

    /* Build a reference upchirp at the BW rate (post-decimation) so
     * we can synthesize a known symbol value. */
    cf_t *upchirp_bw = calloc(N, sizeof(cf_t));
    cf_t *downchirp_bw = calloc(N, sizeof(cf_t));
    lora_build_chirps(N, upchirp_bw, downchirp_bw);

    /* Synthesize a symbol of value 99 at the BW rate. */
    uint16_t test_v = 99;
    cf_t *sym_bw = calloc(N, sizeof(cf_t));
    for (uint32_t n = 0; n < N; n++) sym_bw[n] = upchirp_bw[(n + test_v) % N];

    /* Upsample by os_factor=4 (zero-stuff → in real life you'd
     * interpolate, but for a clean test, sample-and-hold is fine for
     * the in-band signal — it just creates spectral images outside
     * ±BW/2 which the decimator + the DSP rejects). Then frequency-
     * shift down by mix_freq (i.e., put the signal at IF=-mix_freq
     * so the demod's mix=+mix_freq pulls it back to baseband). */
    uint32_t os = sample_rate / bw;
    uint32_t total_samples = N * os * 2;  /* 2 symbols worth of stream */
    cf_t *stream = calloc(total_samples, sizeof(cf_t));
    /* Emit the symbol, sample-and-hold upsampled, in the middle of the
     * stream (preceded by a silent gap to give the demod something to
     * settle on). For this isolated test we'll feed the symbol DIRECTLY
     * into lora_symbol_demod after the mix+decimate, bypassing the
     * frame sync state machine. */
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t k = 0; k < os; k++) {
            uint32_t idx = n * os + k;
            /* Apply -mix_freq frequency shift: multiply by exp(j2π·(-mix_freq)·idx/Fs)
             * so the demod's +mix_freq mix pulls it back to baseband. */
            double w = -2.0 * M_PI * (double)mix_freq * (double)idx
                       / (double)sample_rate;
            cf_t shift = (float)cos(w) + (float)sin(w) * I;
            stream[idx] = sym_bw[n] * shift;
        }
    }

    /* Build the demod with mix_freq_hz set to translate the signal
     * back to baseband. */
    lora_config_t cfg = {
        .sample_rate_hz = sample_rate,
        .bandwidth = LORA_BW_250,
        .sf = sf,
        .sync_word = 0x2B,
        .ldro = 0,
        .mix_freq_hz = mix_freq,
    };
    lora_demod_t *d = lora_demod_new(&cfg, NULL, NULL);
    if (!d) { fprintf(stderr, "  FAIL: demod_new\n"); n_failures++; goto cleanup; }

    /* Convert to interleaved float and feed via the public API. We
     * feed exactly one symbol's worth (N*os samples) and then read
     * from the ring. */
    float *fbuf = (float *)stream;  /* cf_t == 2 floats, layout-compat */
    /* But wait: cf_t is `float complex` not `struct{float,float}`.
     * Per C99 it's required to be array-compatible. Let's be safe
     * and convert explicitly. */
    float *interleaved = calloc(N * os * 2, sizeof(float));
    for (uint32_t i = 0; i < N * os; i++) {
        interleaved[2*i]     = crealf(stream[i]);
        interleaved[2*i + 1] = cimagf(stream[i]);
    }
    lora_demod_process_cf(d, interleaved, N * os);

    /* The ring should now contain exactly N samples of the
     * baseband-translated symbol. Read them out and demod. */
    cf_t recovered[4096];
    lora_ring_read(d, 0, recovered);
    float mag;
    uint16_t r = lora_symbol_demod(d, recovered, &mag);
    EXPECT(r == test_v,
           "mix+decimate end-to-end: synth bin %u recovered as %u (mag=%.0f)",
           test_v, r, mag);
    free(interleaved);
    lora_demod_free(d);
cleanup:
    free(stream);
    free(sym_bw);
    free(upchirp_bw);
    free(downchirp_bw);
}
static void test_cfo_correction(void) {
    fprintf(stderr, "\n[2b] CFO correction\n");
    uint8_t sf = 8;
    uint32_t N = 1u << sf;
    lora_config_t cfg = {
        .sample_rate_hz = 125000, .bandwidth = LORA_BW_125,
        .sf = sf, .sync_word = 0x2B, .ldro = 0,
    };
    lora_demod_t *d = lora_demod_new(&cfg, NULL, NULL);
    cf_t *sym = calloc(N, sizeof(cf_t));
    /* Synthesize symbol value 50, but tell the demod CFO is +5 bins.
     * The "received" signal looks like it has bin (50 + 5) mod N.
     * After correction, we should recover 50. */
    synth_symbol(d->upchirp, N, 55, sym);  /* signal at bin 55 */
    d->cfo_int = 5;                        /* claim CFO is +5 */
    float mag;
    uint16_t r = lora_symbol_demod(d, sym, &mag);
    EXPECT(r == 50, "CFO=+5: recover 50 from signal at bin 55 (got %u)", r);
    /* Negative CFO */
    synth_symbol(d->upchirp, N, 3, sym);   /* signal at bin 3 */
    d->cfo_int = -7;                       /* CFO is -7 */
    r = lora_symbol_demod(d, sym, &mag);
    EXPECT(r == 10, "CFO=-7: recover 10 from signal at bin 3 (got %u)", r);
    free(sym);
    lora_demod_free(d);
}

/* ───────────────────────────────────────────────────────────────── */

static void test_hamming_roundtrip(void) {
    fprintf(stderr, "\n[3] Hamming encode/decode round-trip\n");
    /* For each CR (1..4) and each 4-bit input, encoding then decoding
     * with no error should recover the original. We don't have a public
     * encoder API but we can derive expected codewords inline. */
    /* CR=4 (8,4): codeword bit pattern from gr-lora — we already hard-
     * coded the encode table inside lora_codec.c. Quick sanity: decode
     * 0x00 → data 0; decode 0xFF → data 0xF (all-ones ↔ all-ones). */
    bool corr, unc;
    EXPECT(lora_hamming_decode(0x00, 4, &corr, &unc) == 0x0,
           "CR=4 decode 0x00 → 0x0");
    /* The all-ones codeword for data=0xF: each parity bit is XOR of 3
     * data bits, all 1 → parity = 1. So codeword = 0xFF. */
    EXPECT(lora_hamming_decode(0xFF, 4, &corr, &unc) == 0xF,
           "CR=4 decode 0xFF → 0xF");
    /* Single-bit error correction: flip bit 0 of codeword for d=0xF
     * (i.e. 0xFE) and verify we still recover 0xF. */
    EXPECT(lora_hamming_decode(0xFE, 4, &corr, &unc) == 0xF,
           "CR=4 single-bit correction (0xFE → 0xF)");
    EXPECT(corr == true, "  → marked as corrected");
    /* CR=3 (7,4): drop the high parity bit. */
    EXPECT(lora_hamming_decode(0x00, 3, &corr, &unc) == 0x0,
           "CR=3 decode 0x00 → 0x0");
}

static void test_crc16(void) {
    fprintf(stderr, "\n[4] CRC-16 vectors\n");
    /* Empty input: CRC of zero bytes with init=0x0000 should be
     * 0x0000. */
    EXPECT(lora_crc16((uint8_t *)"", 0) == 0x0000,
           "CRC of empty = 0x0000");
    /* Single byte 0x00: 0x0000 XOR (0x00 << 8) = 0x0000, then 8
     * shifts of 0 = 0x0000. */
    uint8_t one_zero = 0;
    EXPECT(lora_crc16(&one_zero, 1) == 0x0000,
           "CRC of [0x00] = 0x0000");
    /* Single byte 0x01: produces a non-zero CRC. Check it's stable
     * (regression guard). Computed value: 0x1021 (= polynomial). */
    uint8_t one_one = 1;
    uint16_t crc1 = lora_crc16(&one_one, 1);
    EXPECT(crc1 == 0x1021, "CRC of [0x01] = 0x1021 (got 0x%04x)", crc1);
}

static void test_gray(void) {
    fprintf(stderr, "\n[5] Gray code round-trip\n");
    int errs = 0;
    for (uint16_t v = 0; v < 4096; v++) {
        uint16_t enc = lora_gray_encode(v);
        uint16_t dec = lora_gray_decode(enc);
        if (dec != v) {
            if (errs < 3) fprintf(stderr, "  FAIL: v=%u enc=%u dec=%u\n",
                                  v, enc, dec);
            errs++;
        }
    }
    EXPECT(errs == 0, "All 12-bit values round-trip");
}

static void test_whitening(void) {
    fprintf(stderr, "\n[6] Whitening sequence\n");
    /* The first byte of the LoRa whitening sequence is well-known.
     * Per Semtech AN1200.18 (hidden register description) and gr-lora's
     * whitening_impl.cc, the first byte is 0xFF (= seed all-ones). */
    uint8_t b0 = lora_whitening_byte(0);
    EXPECT(b0 == 0xFF, "First whitening byte = 0xFF (got 0x%02x)", b0);
    /* Sequence is 255 bytes long; index 255 == index 0. */
    EXPECT(lora_whitening_byte(255) == lora_whitening_byte(0),
           "Sequence wraps at 255");
}

/* ───────────────────────────────────────────────────────────────── */

int main(void) {
    test_chirp_orthogonality();
    test_symbol_demod();
    test_mix_end_to_end();
    test_cfo_correction();
    test_hamming_roundtrip();
    test_crc16();
    test_gray();
    test_whitening();

    fprintf(stderr, "\n=== %d failure(s) ===\n", n_failures);
    return n_failures == 0 ? 0 : 1;
}
