/*
 * lora_demod.h — Clean-room LoRa physical layer decoder (RX only)
 *
 * Copyright (c) 2026, Off by One. BSD-3-Clause.
 *
 * Implements the LoRa PHY based on Semtech AN1200.22 ("LoRa Modulation
 * Basics"), the SX127x datasheet packet format section, and the public
 * algorithm description in Tapparel et al. "An Open-Source LoRa Physical
 * Layer Prototype on GNU Radio" (arXiv:2002.08208).
 *
 * Designed for headless server use:
 *   • No GNU Radio runtime, no external DSP framework
 *   • One callback per decoded packet, with raw decrypted-payload bytes
 *     and metadata (RSSI, SNR, CFO, sample-offset)
 *   • Streaming input: caller pumps complex-int8 (cu8) or complex-float
 *     samples in chunks of arbitrary size; we maintain a sliding ring
 *   • Per-channel state — one demod instance per (freq, BW, SF) target
 *
 * Scope (RX only):
 *   ✓ Preamble + sync-word detection
 *   ✓ CFO/STO estimation and correction
 *   ✓ FFT-based symbol demod (per-symbol dechirp + N-point FFT, peak bin)
 *   ✓ Gray demap, block deinterleaver
 *   ✓ Hamming(4,5..8) decoder (hard-decision; soft TBD)
 *   ✓ Whitening (LFSR XOR with the standard LoRa sequence)
 *   ✓ Explicit-header parsing (length, CR, has_CRC) — Meshtastic always
 *     uses explicit header, so implicit-header mode is not implemented
 *   ✓ CRC-16 verification
 *   ✗ Transmit: out of scope; this is a passive monitor.
 *
 * What this is NOT:
 *   • Not a Meshtastic protocol stack — it produces raw LoRa payload
 *     bytes. Caller (Python via meshtastic-lite) handles AES-CTR
 *     decrypt, channel hash, protobuf decode, etc.
 *   • Not multi-SF: caller specifies SF at instance creation. Use the
 *     existing rfcensus chirp_analysis to detect SF first, then spin up
 *     a demod for that SF.
 */
#ifndef LORA_DEMOD_H
#define LORA_DEMOD_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum payload length per LoRa spec (1-byte length field). */
#define LORA_MAX_PAYLOAD 255

/* Maximum spreading factor we support. SF7..SF12 covers everything
 * Meshtastic uses across all presets (SHORT_TURBO=SF7 through
 * VERY_LONG_SLOW=SF12). LoRa spec also defines SF5/6 but Meshtastic
 * doesn't use them. */
#define LORA_MIN_SF 7
#define LORA_MAX_SF 12

/* LoRa bandwidth options (Hz). Meshtastic uses 125, 250, 500. */
typedef enum {
    LORA_BW_125 = 125000,
    LORA_BW_250 = 250000,
    LORA_BW_500 = 500000,
} lora_bw_t;

/* Coding rate (Hamming code). LoRa uses CR 1..4 → rate 4/5..4/8.
 * Meshtastic always uses CR=5 (4/5 rate, the lightest FEC). */
typedef enum {
    LORA_CR_4_5 = 1,
    LORA_CR_4_6 = 2,
    LORA_CR_4_7 = 3,
    LORA_CR_4_8 = 4,
} lora_cr_t;

/* Configuration passed at demod creation. Most fields are channel
 * properties that don't change during the life of the instance. */
typedef struct {
    uint32_t  sample_rate_hz;   /* IQ stream sample rate (e.g. 250000 for
                                 * 1× oversample at BW=250 kHz, or 500000
                                 * for 2× oversample) */
    lora_bw_t bandwidth;        /* LoRa channel bandwidth */
    uint8_t   sf;               /* Spreading factor 7..12 */
    uint8_t   sync_word;        /* LoRa sync word (Meshtastic: 0x2B) */
    uint8_t   has_crc_default;  /* Used only in implicit header mode (UNUSED for
                                 * Meshtastic — explicit header carries this) */
    uint8_t   ldro;             /* Low Data Rate Optimization. Required when
                                 * symbol_time_ms > 16ms. Set explicitly per
                                 * preset (LongSlow + VeryLongSlow at BW=125k
                                 * have it on). */
    int32_t   mix_freq_hz;      /* Digital down-mix frequency. The decoder
                                 * multiplies each input sample by
                                 * exp(-j·2π·mix_freq·n/sample_rate) to
                                 * translate the LoRa signal from its IF
                                 * to baseband (IF=0). Set to (capture_freq -
                                 * lora_signal_freq). E.g. capture at 913.5
                                 * MHz with LoRa at 913.125 MHz: mix_freq_hz =
                                 * +375000 (the LoRa signal sits 375 kHz below
                                 * the capture center, so we shift up by 375k
                                 * to land it at baseband). 0 = no mixing. */
} lora_config_t;

/* Result of a successful packet decode, passed to the user callback. */
typedef struct {
    uint8_t  payload[LORA_MAX_PAYLOAD];
    uint16_t payload_len;
    uint8_t  cr;                /* coding rate from header (1..4) */
    uint8_t  has_crc;           /* CRC was present in packet */
    uint8_t  crc_ok;            /* CRC matched (1) or failed (0). When 0 the
                                 * payload bytes are still provided so the
                                 * caller can decide what to do. */
    /* Signal quality */
    float    rssi_db;           /* Estimated RSSI of preamble (relative dB) */
    float    snr_db;            /* Estimated SNR */
    float    cfo_hz;            /* Carrier frequency offset estimate */
    /* Stream position — useful for debugging + for relating decodes
     * back to the IQ capture they came from. */
    uint64_t sample_offset;     /* Sample index of the start of the preamble */
} lora_decoded_t;

/* Packet decode callback. Called once per successful demod run. The
 * lora_decoded_t pointer is owned by the demod and only valid for the
 * duration of the call — copy out anything you want to keep. */
typedef void (*lora_decode_cb_t)(const lora_decoded_t *pkt, void *userdata);

/* Opaque demod instance. */
typedef struct lora_demod lora_demod_t;

/* ─── Lifecycle ──────────────────────────────────────────────────── */

/* Create a demod instance. Returns NULL on error (bad config, OOM,
 * unsupported SF, etc.). Caller must lora_demod_free() to release. */
lora_demod_t *lora_demod_new(const lora_config_t *cfg,
                             lora_decode_cb_t cb,
                             void *userdata);

void lora_demod_free(lora_demod_t *d);

/* ─── Streaming input ────────────────────────────────────────────── */

/* Feed n complex samples (interleaved I,Q as float pairs).
 * Caller can hand in any chunk size; the demod buffers internally.
 * Returns the number of complete packets decoded during this call
 * (callback fires once per packet before the function returns). */
int lora_demod_process_cf(lora_demod_t *d, const float *iq, size_t n);

/* Feed n complex samples in cu8 format (I,Q as uint8_t pairs centered
 * on 127.5, the rtl_sdr / rtl_tcp default). Convenience wrapper that
 * converts inline; for high-throughput integrations, batch-convert
 * upstream and use process_cf directly. */
int lora_demod_process_cu8(lora_demod_t *d, const uint8_t *iq, size_t n);

/* v0.7.11: Feed n complex samples that are ALREADY at bandwidth rate
 * AND already mixed to baseband (DC = slot center frequency). Bypasses
 * the per-decoder mix + resampler — used for channel filter sharing
 * where one upstream channelization pass feeds N concurrent SF
 * decoders at the same slot frequency.
 *
 * Caller MUST construct the decoder with mix_freq_hz=0 and
 * sample_rate_hz=bandwidth, otherwise the mix oscillator would
 * still be advancing and the resampler would still be running and
 * the output would be wrong.
 *
 * Returns the number of complete packets decoded during this call. */
int lora_demod_feed_baseband(lora_demod_t *d, const float *iq, size_t n);

/* Reset frame-sync state (e.g. when retuning, after silence detection).
 * Doesn't free the instance. */
void lora_demod_reset(lora_demod_t *d);

/* ─── Diagnostics ────────────────────────────────────────────────── */

typedef struct {
    uint64_t samples_processed;
    uint32_t preambles_found;
    uint32_t syncwords_matched;     /* Preambles where sync word matched */
    uint32_t headers_decoded;       /* Header CRC passed */
    uint32_t headers_failed;        /* Header CRC failed (signal too weak / CFO drift) */
    uint32_t packets_decoded;       /* Full payload + CRC passed */
    uint32_t packets_crc_failed;    /* Full payload but CRC failed */
    /* Diagnostics for tuning the energy gate */
    uint64_t detect_attempts;       /* Total DETECT-state symbol evaluations */
    uint64_t detect_above_gate;     /* DETECT symbols above the energy gate */
    uint32_t detect_max_run;        /* Longest run of same-bin chirps seen
                                     * (informs whether N_PREAMBLE_DETECT is
                                     * the right threshold). */
    float    detect_peak_mag_max;   /* Largest peak magnitude observed */
} lora_demod_stats_t;

void lora_demod_get_stats(const lora_demod_t *d, lora_demod_stats_t *out);

/* v0.7.16: query whether the decoder is currently sitting in the
 * DETECT state (= idle, looking for a new preamble) or in the middle
 * of decoding a packet (any of SYNC_NETID / SYNC_DOWN / QUARTER_DOWN
 * / DEMOD).
 *
 * Returns:
 *   1 — decoder is idle (safe to tear down with no packet loss)
 *   0 — decoder is actively decoding (tearing down would lose the
 *       in-flight packet)
 *
 * Used by lazy_pipeline to defer reap-while-decoding. The internal
 * lora_state_t enum stays private; callers only need the binary
 * idle/busy distinction. */
int lora_demod_is_idle(const lora_demod_t *d);

#ifdef __cplusplus
}
#endif

#endif /* LORA_DEMOD_H */
