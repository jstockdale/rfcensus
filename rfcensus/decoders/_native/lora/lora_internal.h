/*
 * lora_internal.h — private types/helpers shared across the decoder
 * implementation files. Not part of the public API.
 *
 * Copyright (c) 2026, Off by One. BSD-3-Clause.
 */
#ifndef LORA_INTERNAL_H
#define LORA_INTERNAL_H

#include "lora_demod.h"
#include "lora_fft.h"     /* v0.7.9: pluggable FFT backend (pffft/kiss) */
#include <complex.h>
#include <stdint.h>
#include <stdbool.h>

/* Single complex sample type used everywhere in the demod. We use C99
 * float complex internally; lora_fft_cpx_t is layout-compatible
 * (struct of two floats) so we cast freely between cf_t and
 * lora_fft_cpx_t pointers. */
typedef float _Complex cf_t;

/* Frame-sync state machine. The DETECT → SYNC → DEMOD sequence mirrors
 * gr-lora's state machine as described in the EPFL paper (sec. III.B).
 *
 *   DETECT     : looking for N consecutive upchirps (preamble)
 *   SYNC_NETID : confirming the network ID (sync word) symbols
 *   SYNC_DOWN  : confirming the two downchirps that mark end of preamble
 *   DEMOD      : extracting payload symbols
 */
typedef enum {
    LORA_STATE_DETECT = 0,
    LORA_STATE_SYNC_NETID,
    LORA_STATE_SYNC_DOWN,
    LORA_STATE_QUARTER_DOWN,
    LORA_STATE_DEMOD,
} lora_state_t;

/* Internal demod instance. Field comments document state ownership and
 * update cadence; this struct is HOT (touched per-symbol), so layout
 * matters for cache. We group by access pattern: configuration first,
 * then sliding-window buffers, then per-frame state, then stats. */
struct lora_demod {
    /* ── Configuration (read-only after creation) ─────────────────── */
    lora_config_t   cfg;
    lora_decode_cb_t cb;
    void           *userdata;

    /* Derived constants */
    uint32_t        N;              /* Samples per LoRa symbol = 2^SF (assumes
                                     * sample_rate == bandwidth, i.e. 1×
                                     * oversample. For higher oversample we
                                     * decimate before demod.) */
    uint32_t        os_factor;      /* sample_rate / bandwidth (must be ≥1) */

    /* ── Reference chirps (precomputed once) ──────────────────────── */
    cf_t           *upchirp;        /* N samples, base upchirp at SF/BW */
    cf_t           *downchirp;      /* N samples, conjugate of upchirp */

    /* ── Digital down-mix (applied at input rate, before resampling) ─
     * To translate a signal sitting at IF=mix_freq_hz down to IF=0 we
     * multiply each input sample by mix_phasor, then advance the phasor
     * by mix_step. Phasor is renormalized periodically to prevent drift
     * over long captures. */
    cf_t            mix_phasor;
    cf_t            mix_step;
    bool            mix_enabled;
    uint64_t        mix_norm_counter;

    /* ── Resampler state ──────────────────────────────────────────── */
    /* We resample from the input rate (cfg.sample_rate_hz) down to the
     * BW rate using fractional-delay linear interpolation. State:
     *
     *   resamp_step   = sample_rate / bandwidth (fractional, e.g. 9.6
     *                   for 2.4 Msps capture and BW=250 kHz)
     *   resamp_pos    = next position (in input samples) where we
     *                   want to emit an output sample
     *   resamp_prev   = the most recently consumed input sample, kept
     *                   so we can interpolate across chunk boundaries
     *
     * Linear interpolation is poor at rejecting out-of-band noise
     * compared to a polyphase FIR, but for clean LoRa signals in mostly
     * empty spectrum it works fine for initial bring-up. A polyphase
     * implementation is tracked for v0.6.17+ as a sensitivity improver.
     */
    double          resamp_step;
    double          resamp_pos;
    cf_t            resamp_prev;
    bool            resamp_have_prev;

    /* ── FFT scratch (v0.7.9: pluggable backend, pffft default) ──── */
    lora_fft_ctx_t    *fft_ctx;       /* N-point forward FFT context */
    lora_fft_cpx_t    *fft_in;        /* N samples — dechirped symbol
                                          (16-byte aligned for SIMD) */
    lora_fft_cpx_t    *fft_out;       /* N samples — FFT result
                                          (16-byte aligned for SIMD) */

    /* ── Sliding input ring ───────────────────────────────────────── */
    /* The frame sync needs to look back ≥12.25 symbols to confirm a
     * preamble (8 detect + 2 netid + 2.25 downchirp). We maintain a
     * ring of at least RING_SYMS symbols of history. */
    cf_t           *ring;           /* Power-of-two sized ring buffer */
    uint32_t        ring_size;      /* In SAMPLES, power of 2 */
    uint32_t        ring_mask;      /* ring_size - 1 */
    uint64_t        ring_w;         /* Total samples written (monotonic) */
    /* Read cursor for the state machine — chases ring_w by the amount
     * of buffered look-back the current state needs. */
    uint64_t        read_cursor;

    /* ── Frame-sync state ─────────────────────────────────────────── */
    lora_state_t    state;
    uint8_t         consec_upchirps;    /* How many upchirps in a row found
                                         * during DETECT. Resets on miss. */
    int32_t         last_bin;           /* Peak bin from last DETECT symbol;
                                         * used to confirm successive symbols
                                         * land in the SAME bin (stable
                                         * preamble) within ±1 bin tolerance. */
    int32_t         cfo_int;            /* Integer-bin CFO estimate from preamble.
                                         * Applied to all subsequent demods. */
    float           cfo_frac;           /* Fractional CFO (sub-bin) — offset
                                         * inside the dechirp reference. */
    uint64_t        preamble_start;     /* Sample index where the preamble began
                                         * (passed to user in lora_decoded_t). */
    int8_t          netid_match[2];     /* 1 = matched expected sync word symbols,
                                         * 0 = pending, -1 = mismatch */
    uint8_t         observed_sync_word; /* v0.6.17: the sync byte derived from
                                         * the recovered sym1/sym2 bins, regardless
                                         * of whether it matches cfg.sync_word.
                                         * Lets the consumer route packets by
                                         * actual on-air sync (Meshtastic public
                                         * 0x2B vs other private networks). */
    /* CFO/STO refinement: collected across the 8 detect symbols so we
     * can compute fractional CFO from phase progression. */
    float           preamble_phase[8];
    /* v0.6.18: Bernier-algorithm CFO_frac estimation. Save the
     * complex value at the strongest bin from each preamble dechirp
     * so we can compute four_cum = sum_i(fft_val[k0,i] * conj(fft_val[k0,i+1]))
     * across all 8 preamble symbols. arg(four_cum) gives the
     * per-symbol phase rotation, which is 2π * cfo_frac. The
     * algorithm is robust to noise because it uses 8 phase
     * differences and averages them in the complex sum.
     *
     * Stored as flat array of 8 complex bins (the value at idx_max
     * across all 8 preamble symbols). idx_max chosen as the bin
     * with largest cumulative magnitude across the preamble. */
    cf_t            preamble_fft_at_peak[8];
    uint32_t        preamble_peak_bin;  /* Which bin we stored values
                                         * for (= argmax of preamble
                                         * dechirp magnitudes) */

    /* ── Demodulation state (during DEMOD) ────────────────────────── */
    uint16_t       *demod_symbols;      /* Buffer for one packet's symbol values
                                         * before deinterleave. Sized for max
                                         * packet at SF12. */
    uint32_t        symbols_collected;  /* How many filled in current packet */
    uint32_t        symbols_needed;     /* Total symbols to read for this packet
                                         * (0 = unknown until header decoded) */
    bool            header_decoded;     /* Header passed; we know length+CR */
    uint8_t         pkt_cr;             /* From header */
    bool            pkt_has_crc;        /* From header */
    uint32_t        pkt_len;            /* From header */

    /* ── Signal quality (per-frame) ───────────────────────────────── */
    float           rssi_est;
    float           snr_est;

    /* ── Stats (lifetime of instance) ─────────────────────────────── */
    lora_demod_stats_t stats;
};

/* ─── Internal helpers (impls in lora_chirp.c / lora_codec.c) ────── */

/* Build the reference upchirp/downchirp for given (sf, bw, sample_rate). */
void lora_build_chirps(uint32_t N, cf_t *upchirp, cf_t *downchirp);

/* Run the per-symbol demod: dechirp samples (multiply by downchirp),
 * FFT, return the peak bin index (0..N-1) and optionally the peak
 * magnitude. samples must have N consecutive cf_t. cfo_int is added
 * to the result mod N. */
uint16_t lora_symbol_demod(struct lora_demod *d,
                           const cf_t *samples,
                           float *peak_mag_out);

/* Cross-correlate against reference upchirp at the current ring
 * cursor; returns the peak bin and magnitude. Used for preamble detect
 * and CFO estimation. */
uint16_t lora_detect_chirp(struct lora_demod *d,
                           uint64_t sample_offset,
                           float *peak_mag_out);

/* v0.6.18: dechirp + FFT, return the COMPLEX value at a specific bin
 * position. Used by Bernier's CFO_frac estimator. */
cf_t lora_dechirp_bin(struct lora_demod *d,
                       uint64_t sample_offset,
                       uint32_t target_bin);

/* Read N samples out of the ring starting at logical offset `off`,
 * unwrapping the wrap-around. Caller-provided dst must have N slots. */
void lora_ring_read(const struct lora_demod *d, uint64_t off, cf_t *dst);

/* Codec layer: takes a vector of demodulated symbol values + CR + LDRO
 * and produces decoded bytes. Returns number of payload bytes written
 * to `out`, or -1 on header CRC failure. */
int lora_decode_payload(const uint16_t *symbols, uint32_t n_symbols,
                        uint8_t sf, uint8_t cr, bool ldro,
                        uint8_t *out, uint32_t out_cap,
                        uint8_t *out_cr, uint8_t *out_has_crc,
                        bool *out_crc_ok);

/* Compute the total number of demodulated symbols required to fully
 * decode a packet, given the first 8 (header block) symbols.
 *
 * Returns:
 *   > 0 → total symbols needed (8 + payload_blocks × cw_len_pld).
 *         Caller keeps demodulating until reaching this count, then
 *         calls lora_decode_payload().
 *   < 0 → header CRC failed; this is not a real packet, abandon.
 *   = 0 → not enough symbols supplied (need ≥ 8).
 *
 * Used by the demod state machine to early-emit short packets without
 * waiting for the worst-case symbol count. Without this, decoding a
 * 49-byte SF9 packet would wait for 320 symbols (~1.3 sec at 1MS/s)
 * instead of the ~30 actually needed (~120 ms). */
int lora_compute_symbols_needed(const uint16_t *symbols,
                                uint32_t n_symbols,
                                uint8_t sf, bool ldro);

/* Gray code helpers — encode and decode a value of given bit-width.
 * LoRa uses Gray coding to reduce error probability between adjacent
 * symbols. encode = v ^ (v >> 1); RX-side LoRa uses this (NOT gray
 * decode) — gr-lora's gray_mapping_impl.cc:70. */
static inline uint16_t lora_gray_encode(uint16_t v) { return v ^ (v >> 1); }
uint16_t lora_gray_decode(uint16_t v);

/* CRC-16 used by LoRa explicit-header frames. Polynomial 0x1021, init
 * 0x0000, no reflection (the LoRa flavor is non-standard CCITT). */
uint16_t lora_crc16(const uint8_t *data, uint32_t len);

/* Whitening LFSR sequence — a fixed 255-byte sequence per LoRa spec.
 * Returns the byte at offset i (mod 255). */
uint8_t lora_whitening_byte(uint32_t i);

/* Hamming(4,k+4) decoder: input a single k+4-bit codeword (k=1..4
 * giving CR 4/5..4/8), output a 4-bit nibble. May correct 1 bit (only
 * for CR≥2 i.e. 4/6+) or detect a 2-bit error (CR=4 only). */
uint8_t lora_hamming_decode(uint8_t codeword, uint8_t cr,
                            bool *corrected, bool *uncorrectable);

#endif /* LORA_INTERNAL_H */
