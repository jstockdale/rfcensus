/* lora_probe.c — multi-SF blind preamble detector implementation.
 *
 * For each candidate SF, we keep:
 *   • A reference downchirp of length N = 2^SF * oversample
 *   • An aligned input buffer for the dechirp result (FFT input)
 *   • An aligned output buffer (FFT result)
 *   • A pre-allocated FFT context
 *
 * Probe scan steps for one SF:
 *   1. Multiply input samples by reference downchirp (in-place into
 *      fft_in). After this, a coherent upchirp at the input becomes
 *      a constant-frequency tone whose frequency = symbol value.
 *   2. Forward FFT — concentrates the tone's energy into one bin.
 *   3. Find the peak bin's magnitude.
 *   4. Compute mean magnitude of all OTHER bins as the noise floor.
 *   5. SNR = 20*log10(peak / noise_floor).
 *
 * For a clean preamble at SNR > -5 dB (which any real-world receivable
 * signal exceeds), the post-correlation SNR is typically +30 dB or
 * more. Our threshold of +10 dB has plenty of margin for noise alone
 * (which produces a flat spectrum, peak/mean ≈ sqrt(N) ≈ 30 for
 * N=2048, giving ~10 dB ratio that just barely fails).
 *
 * Note we run on ONLY THE FIRST N samples for each SF — one symbol's
 * worth. A LoRa preamble is 8 upchirps, so any random window inside
 * the preamble will give a strong dechirp. If the caller suspects
 * the window doesn't perfectly align with a symbol boundary, they
 * can call scan() multiple times with shifted starting positions
 * and take the max — but a single scan with reasonably-aligned
 * lookback usually suffices, because the energy in any 1-symbol
 * window of the preamble is identical regardless of phase.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#include "lora_probe.h"
#include "lora_dechirp_simd.h"   /* v0.7.11 vectorized complex multiply */

/* Internal complex type matching cf_t conventions used elsewhere. */
typedef float _Complex cf_t;

#define MAX_SF_COUNT 8

struct lora_probe_sf {
    uint32_t sf;
    uint32_t N;                  /* FFT size = 2^sf * oversample */
    cf_t *downchirp;             /* N samples — reference downchirp */
    lora_fft_ctx_t *fft_ctx;
    lora_fft_cpx_t *fft_in;      /* N samples, 16-byte aligned */
    lora_fft_cpx_t *fft_out;     /* N samples, 16-byte aligned */
};

struct lora_probe {
    uint32_t sf_count;
    float snr_threshold_db;
    struct lora_probe_sf sfs[MAX_SF_COUNT];
    size_t total_bytes;          /* for memory_usage() reporting */
};

/* Build the reference downchirp for a given N. The downchirp is the
 * complex conjugate of the upchirp; multiplying the input by the
 * downchirp ("dechirping") converts a coherent upchirp into a
 * constant-frequency tone whose frequency depends on the symbol
 * value.
 *
 * Phase formula: phase[n] = π · (n²/N - n), matching
 * lora_build_chirps in lora_chirp.c (closed-form per Tapparel et al.
 * and gr-lora). Downchirp is the complex conjugate of upchirp:
 * conj(cos+i·sin) = cos-i·sin = exp(-iφ).
 *
 * NOTE: an earlier draft used incremental phase accumulation
 * (`phase += 2π·inst_freq`) which produced a chirp slope wrong by
 * the oversample factor — the probe would then "detect" SF8 in a
 * signal that was really SF9 (and similarly mislabel every SF). The
 * closed-form formula here matches the decoder bit-for-bit so probe
 * SFs and decoder SFs always agree.
 */
static void build_downchirp(uint32_t N, cf_t *out) {
    double Nd = (double)N;
    for (uint32_t n = 0; n < N; n++) {
        double nd = (double)n;
        double phase = 3.14159265358979323846 * (nd * nd / Nd - nd);
        float c = (float)cos(phase);
        float s = (float)sin(phase);
        /* Downchirp = complex conjugate of upchirp */
        out[n] = c + (-s) * I;
    }
}

static int init_sf(struct lora_probe_sf *s, uint32_t sf, uint32_t oversample) {
    s->sf = sf;
    s->N = ((uint32_t)1 << sf) * oversample;
    s->downchirp = (cf_t*)calloc(s->N, sizeof(cf_t));
    if (!s->downchirp) return -1;
    s->fft_in = (lora_fft_cpx_t*)lora_fft_aligned_alloc(
        (size_t)s->N * sizeof(lora_fft_cpx_t));
    s->fft_out = (lora_fft_cpx_t*)lora_fft_aligned_alloc(
        (size_t)s->N * sizeof(lora_fft_cpx_t));
    if (!s->fft_in || !s->fft_out) return -1;
    memset(s->fft_in,  0, (size_t)s->N * sizeof(lora_fft_cpx_t));
    memset(s->fft_out, 0, (size_t)s->N * sizeof(lora_fft_cpx_t));
    s->fft_ctx = lora_fft_new(s->N);
    if (!s->fft_ctx) return -1;
    build_downchirp(s->N, s->downchirp);
    return 0;
}

static void free_sf(struct lora_probe_sf *s) {
    if (s->downchirp) { free(s->downchirp); s->downchirp = NULL; }
    if (s->fft_in)  { lora_fft_aligned_free(s->fft_in); s->fft_in = NULL; }
    if (s->fft_out) { lora_fft_aligned_free(s->fft_out); s->fft_out = NULL; }
    if (s->fft_ctx) { lora_fft_destroy(s->fft_ctx); s->fft_ctx = NULL; }
}

lora_probe_t* lora_probe_create(
    const uint32_t *sfs,
    uint32_t sf_count,
    uint32_t oversample,
    float snr_threshold_db
) {
    if (sf_count == 0 || sf_count > MAX_SF_COUNT) return NULL;
    if (oversample == 0) return NULL;
    /* Validate each SF is in the LoRa-supported range 6..12 */
    for (uint32_t i = 0; i < sf_count; i++) {
        if (sfs[i] < 6 || sfs[i] > 12) return NULL;
    }
    lora_probe_t *p = (lora_probe_t*)calloc(1, sizeof(lora_probe_t));
    if (!p) return NULL;
    p->sf_count = sf_count;
    p->snr_threshold_db = snr_threshold_db;
    p->total_bytes = sizeof(lora_probe_t);
    for (uint32_t i = 0; i < sf_count; i++) {
        if (init_sf(&p->sfs[i], sfs[i], oversample) != 0) {
            lora_probe_destroy(p);
            return NULL;
        }
        p->total_bytes += sizeof(struct lora_probe_sf)
                       + p->sfs[i].N * (sizeof(cf_t)
                                        + 2 * sizeof(lora_fft_cpx_t));
    }
    return p;
}

void lora_probe_destroy(lora_probe_t *p) {
    if (!p) return;
    for (uint32_t i = 0; i < p->sf_count; i++) {
        free_sf(&p->sfs[i]);
    }
    free(p);
}

/* Single-SF scan: dechirp + FFT, find peak, compute SNR. Internal
 * helper for scan(). */
static void scan_one_sf(struct lora_probe_sf *s,
                        const lora_fft_cpx_t *samples,
                        lora_probe_result_t *out) {
    uint32_t N = s->N;
    /* v0.7.11: vectorized dechirp using NEON / SSE3 / scalar fallback.
     * lora_fft_cpx_t is layout-compatible with interleaved float pairs
     * AND with cf_t (struct {float r; float i;}); the SIMD helper
     * writes directly into fft_in. */
    lora_dechirp_mul(
        (const float*)(const void*)samples,
        (const float*)(const void*)s->downchirp,
        (float*)(void*)s->fft_in,
        N
    );
    lora_fft_forward(s->fft_ctx, s->fft_in, s->fft_out);
    /* Find peak magnitude bin and accumulate sum of magnitudes for
     * noise floor estimate. We skip the peak bin from the noise sum
     * AFTER the loop to avoid double-pass. */
    float best_mag2 = -1.0f;
    uint16_t best_bin = 0;
    float total_mag2 = 0.0f;
    for (uint32_t k = 0; k < N; k++) {
        float r = s->fft_out[k].r;
        float i = s->fft_out[k].i;
        float m2 = r * r + i * i;
        total_mag2 += m2;
        if (m2 > best_mag2) {
            best_mag2 = m2;
            best_bin = (uint16_t)k;
        }
    }
    float peak_mag = sqrtf(best_mag2);
    /* Noise floor: mean of non-peak bins. Use power (mag^2) for the
     * mean to avoid the bias of mean-of-sqrt vs sqrt-of-mean — then
     * take sqrt at the end. */
    float noise_mag2 = (total_mag2 - best_mag2) / (float)(N - 1);
    float noise_mag = sqrtf(noise_mag2);
    float snr_db = (noise_mag > 0.0f)
        ? 20.0f * log10f(peak_mag / noise_mag)
        : 1000.0f;    /* effectively infinite SNR if noise is zero */
    out->sf = s->sf;
    out->peak_mag = peak_mag;
    out->noise_floor = noise_mag;
    out->snr_db = snr_db;
    out->peak_bin = best_bin;
    out->detected = false;    /* threshold check done by caller */
}

uint32_t lora_probe_scan(
    lora_probe_t *p,
    const lora_fft_cpx_t *samples,
    uint32_t n_samples,
    lora_probe_result_t *results
) {
    if (!p || !samples || !results) return 0;
    uint32_t n_detected = 0;
    for (uint32_t i = 0; i < p->sf_count; i++) {
        struct lora_probe_sf *s = &p->sfs[i];
        if (n_samples < s->N) {
            /* Not enough samples for this SF — fail it cleanly. */
            results[i].sf = s->sf;
            results[i].peak_mag = 0.0f;
            results[i].noise_floor = 0.0f;
            results[i].snr_db = -1000.0f;
            results[i].peak_bin = 0;
            results[i].detected = false;
            continue;
        }
        scan_one_sf(s, samples, &results[i]);
        if (results[i].snr_db >= p->snr_threshold_db) {
            results[i].detected = true;
            n_detected++;
        }
    }
    return n_detected;
}

size_t lora_probe_memory_usage(const lora_probe_t *p) {
    return p ? p->total_bytes : 0;
}
