/* lora_probe.h — multi-SF blind preamble detector.
 *
 * Purpose: given a buffer of decimated baseband samples (at BW rate
 * after channelization), identify which spreading factor(s) have a
 * preamble present. Replaces the v0.7.x "spawn 5 SF decoders per slot
 * activation, then race" approach with "look once, spawn matching SF".
 *
 * For 5 candidate SFs at SF7..SF11 (BW=250kHz), the blind probe
 * costs ~5 × (dechirp + FFT) ≈ 200µs per activation, vs the racing
 * approach which lets 4 wrong-SF decoders run for ~5-50ms before
 * being killed.
 *
 * Multi-system support: when two transmitters at the same slot
 * frequency are simultaneously active on different SFs (rare but
 * possible — mixed mesh networks, neighbor-mesh interference),
 * the probe returns ALL detected SFs, not just the strongest. The
 * caller spawns full decoders for every detected SF.
 *
 * The probe operates on ALREADY-DECIMATED baseband (BW-rate complex
 * samples), so a single channelization pass feeds N concurrent SF
 * probes. This is the foundation for v0.8 channel filter sharing.
 */
#ifndef LORA_PROBE_H
#define LORA_PROBE_H

#include <stdint.h>
#include <stdbool.h>

#include "lora_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-SF probe state. One per candidate SF. Allocated by
 * lora_probe_create. Holds the SF-specific reference downchirp +
 * FFT context. The caller never touches this directly. */
typedef struct lora_probe_sf lora_probe_sf_t;

/* Top-level probe. Owns N per-SF probes and provides the public API. */
typedef struct lora_probe lora_probe_t;

/* Result of one probe scan. */
typedef struct {
    uint32_t sf;             /* which SF (e.g. 7..12) */
    float    peak_mag;       /* magnitude of strongest FFT bin */
    float    noise_floor;    /* mean magnitude of non-peak bins */
    float    snr_db;         /* 20*log10(peak/noise) */
    uint16_t peak_bin;       /* which bin had the peak (0..N-1) */
    bool     detected;       /* snr_db >= threshold */
} lora_probe_result_t;

/* Create a multi-SF probe. The caller supplies the candidate SF list
 * (e.g. {7, 8, 9, 10, 11} for BW=250 in Meshtastic). The bandwidth +
 * oversample determine the FFT size N = (1 << SF) * oversample for
 * each SF.
 *
 * sf_count must be <= 8.
 * snr_threshold_db: minimum peak-to-noise ratio to count as detected.
 *   Typical: 10.0 dB (preamble correlator gain is ~30 dB at 0 dB SNR
 *   so we have plenty of margin).
 *
 * Returns NULL on alloc failure. */
lora_probe_t* lora_probe_create(
    const uint32_t *sfs,
    uint32_t sf_count,
    uint32_t oversample,
    float snr_threshold_db
);

void lora_probe_destroy(lora_probe_t *p);

/* Run the probe on a buffer of decimated baseband samples. The
 * buffer must contain at least max(N) samples (one symbol at the
 * largest SF). For SF11 with oversample=2, that's 4096 samples.
 *
 * Each per-SF probe takes the FIRST N samples from `samples` (where
 * N = 2^sf * oversample for that SF), does dechirp + FFT, finds
 * the peak bin and computes peak/noise ratio.
 *
 * Results array must have at least sf_count entries (the order
 * matches the sfs[] passed at create-time).
 *
 * Returns the number of SFs marked as detected (= sum of
 * results[i].detected). */
uint32_t lora_probe_scan(
    lora_probe_t *p,
    const lora_fft_cpx_t *samples,
    uint32_t n_samples,
    lora_probe_result_t *results
);

/* Diagnostic: how many bytes does this probe occupy on the heap?
 * Used by tests to confirm we're not silently leaking. */
size_t lora_probe_memory_usage(const lora_probe_t *p);

#ifdef __cplusplus
}
#endif

#endif /* LORA_PROBE_H */
