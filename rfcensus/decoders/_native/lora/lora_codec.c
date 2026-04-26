/*
 * lora_codec.c — post-demod codec: Hamming, Gray, deinterleave, dewhiten, CRC.
 *
 * Copyright (c) 2026, Off by One. BSD-3-Clause.
 *
 * Algorithms from:
 *   • Semtech SX1276/77/78/79 datasheet, section "Frequency-shift Chirp
 *     Modulation" and "Packet engine"
 *   • Tapparel et al. arXiv:2002.08208 sections III.C-III.E (codec
 *     pipeline description)
 *
 * All functions here are stateless and deterministic — easy to unit
 * test against published vectors.
 */

#include "lora_internal.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>      /* v0.7.9: was transitively pulled via kiss_fft.h
                           before; lora_fft.h doesn't include it. */

/* ────────────────────────────────────────────────────────────────────
 * Gray code
 * ────────────────────────────────────────────────────────────────────
 * LoRa uses standard binary-reflected Gray coding to reduce the
 * probability that a frequency-bin error produces a multi-bit symbol
 * error. Encode: v ^ (v >> 1). Decode: cumulative XOR of all higher
 * bits.
 */
/* Standard gray DECODE: cumulative XOR of all higher bits. Inverse
 * of gray_encode (single XOR). Kept for compatibility with any
 * tests that use it. */
uint16_t lora_gray_decode(uint16_t v) {
    uint16_t r = v;
    for (int s = 1; s < 16; s <<= 1) {
        r ^= r >> s;
    }
    return r;
}

/* v0.6.18: lora_gray_encode is defined inline in lora_internal.h. */

/* ────────────────────────────────────────────────────────────────────
 * Whitening sequence
 * ────────────────────────────────────────────────────────────────────
 * The standard 255-byte LoRa whitening sequence as defined by the
 * Semtech LoRa specification (also documented in the SX127x datasheet
 * App Note AN1200.18 "PN9 sequence"). This is the byte-wise output of
 * a fixed PRNG and the SAME 255 values appear in every open-source
 * LoRa implementation (gr-lora_sdr, lora-sdr, sx12xx-driver, etc) —
 * they are physical-layer constants, not a creative work.
 *
 * v0.6.18: replaced the LFSR generator (which had the wrong bit-order
 * and produced garbage) with the hard-coded table. Future contributor:
 * if you change the table, run test_decode against
 * /tmp/meshtastic_30s_913_5mhz_1msps.cu8 and verify packet CRC passes.
 */
static const uint8_t whitening_seq[255] = {
    0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE1, 0xC2, 0x85,
    0x0B, 0x17, 0x2F, 0x5E, 0xBC, 0x78, 0xF1, 0xE3,
    0xC6, 0x8D, 0x1A, 0x34, 0x68, 0xD0, 0xA0, 0x40,
    0x80, 0x01, 0x02, 0x04, 0x08, 0x11, 0x23, 0x47,
    0x8E, 0x1C, 0x38, 0x71, 0xE2, 0xC4, 0x89, 0x12,
    0x25, 0x4B, 0x97, 0x2E, 0x5C, 0xB8, 0x70, 0xE0,
    0xC0, 0x81, 0x03, 0x06, 0x0C, 0x19, 0x32, 0x64,
    0xC9, 0x92, 0x24, 0x49, 0x93, 0x26, 0x4D, 0x9B,
    0x37, 0x6E, 0xDC, 0xB9, 0x72, 0xE4, 0xC8, 0x90,
    0x20, 0x41, 0x82, 0x05, 0x0A, 0x15, 0x2B, 0x56,
    0xAD, 0x5B, 0xB6, 0x6D, 0xDA, 0xB5, 0x6B, 0xD6,
    0xAC, 0x59, 0xB2, 0x65, 0xCB, 0x96, 0x2C, 0x58,
    0xB0, 0x61, 0xC3, 0x87, 0x0F, 0x1F, 0x3E, 0x7D,
    0xFB, 0xF6, 0xED, 0xDB, 0xB7, 0x6F, 0xDE, 0xBD,
    0x7A, 0xF5, 0xEB, 0xD7, 0xAE, 0x5D, 0xBA, 0x74,
    0xE8, 0xD1, 0xA2, 0x44, 0x88, 0x10, 0x21, 0x43,
    0x86, 0x0D, 0x1B, 0x36, 0x6C, 0xD8, 0xB1, 0x63,
    0xC7, 0x8F, 0x1E, 0x3C, 0x79, 0xF3, 0xE7, 0xCE,
    0x9C, 0x39, 0x73, 0xE6, 0xCC, 0x98, 0x31, 0x62,
    0xC5, 0x8B, 0x16, 0x2D, 0x5A, 0xB4, 0x69, 0xD2,
    0xA4, 0x48, 0x91, 0x22, 0x45, 0x8A, 0x14, 0x29,
    0x52, 0xA5, 0x4A, 0x95, 0x2A, 0x54, 0xA9, 0x53,
    0xA7, 0x4E, 0x9D, 0x3B, 0x77, 0xEE, 0xDD, 0xBB,
    0x76, 0xEC, 0xD9, 0xB3, 0x67, 0xCF, 0x9E, 0x3D,
    0x7B, 0xF7, 0xEF, 0xDF, 0xBF, 0x7E, 0xFD, 0xFA,
    0xF4, 0xE9, 0xD3, 0xA6, 0x4C, 0x99, 0x33, 0x66,
    0xCD, 0x9A, 0x35, 0x6A, 0xD4, 0xA8, 0x51, 0xA3,
    0x46, 0x8C, 0x18, 0x30, 0x60, 0xC1, 0x83, 0x07,
    0x0E, 0x1D, 0x3A, 0x75, 0xEA, 0xD5, 0xAA, 0x55,
    0xAB, 0x57, 0xAF, 0x5F, 0xBE, 0x7C, 0xF9, 0xF2,
    0xE5, 0xCA, 0x94, 0x28, 0x50, 0xA1, 0x42, 0x84,
    0x09, 0x13, 0x27, 0x4F, 0x9F, 0x3F, 0x7F,
};

uint8_t lora_whitening_byte(uint32_t i) {
    return whitening_seq[i % 255];
}

/* ────────────────────────────────────────────────────────────────────
 * CRC-16 (LoRa flavor)
 * ────────────────────────────────────────────────────────────────────
 * LoRa uses CRC-16/CCITT-FALSE: polynomial 0x1021, init 0xFFFF, no
 * reflection, no XOR-out. The CRC is appended to the payload before
 * whitening; on RX we de-whiten the payload + CRC, then compute over
 * just the payload and compare to the trailing 2 bytes.
 *
 * Note: The Semtech spec actually documents this as "CRC-16/IBM" in
 * one place and "CRC-CCITT" in another, with confusing diagrams. The
 * value we compute below matches what real SX127x radios produce on
 * the wire and what gr-lora's crc_verif_impl.cc computes. Lock in
 * a published test vector in the unit tests so this can never silently
 * regress to a different polynomial.
 */
uint16_t lora_crc16(const uint8_t *data, uint32_t len) {
    uint16_t crc = 0x0000;  /* LoRa init is 0x0000, NOT 0xFFFF. (See
                             * SX127x datasheet rev 7 section 4.2.13.6
                             * which corrects an earlier rev's error.) */
    for (uint32_t i = 0; i < len; i++) {
        crc ^= ((uint16_t)data[i]) << 8;
        for (int b = 0; b < 8; b++) {
            crc = (crc & 0x8000) ? ((crc << 1) ^ 0x1021) : (crc << 1);
        }
    }
    return crc;
}

/* ────────────────────────────────────────────────────────────────────
 * Hamming decoder for LoRa coding rates 4/5..4/8
 * ────────────────────────────────────────────────────────────────────
 * LoRa "coding rate" CR=1..4 maps to Hamming(5,4)/(6,4)/(7,4)/(8,4):
 *   CR=1  4/5  parity-only (single bit, just XOR — no error correct)
 *   CR=2  4/6  single parity over different subset (still no correct,
 *              just two parity bits = error detection)
 *   CR=3  4/7  Hamming(7,4) — corrects 1 bit, detects 2
 *   CR=4  4/8  Hamming(8,4) — corrects 1 bit, detects 2
 *
 * The header is ALWAYS coded at CR=4 (4/8) regardless of payload CR
 * to give the most robust decoding for the length+CR+CRC fields.
 *
 * For CR=3,4 we use a syndrome lookup table (16 entries each) generated
 * at startup. Hard-decision only; soft decoding from FFT magnitudes is
 * a future enhancement.
 */

/* Hamming(8,4) generator matrix. Each codeword is 4 data bits + 4
 * parity bits arranged as p0 p1 d0 p2 d1 d2 d3 p3 (per gr-lora's
 * convention which matches Semtech's actual wire format). */
static uint8_t hamming84_encode_table[16];
static uint8_t hamming84_decode_table[256];   /* codeword → 4-bit data */
static uint8_t hamming74_decode_table[128];   /* codeword → 4-bit data */
static bool hamming_initialized = false;

/* Compute parity of the bits whose mask is set in `mask`. */
static inline uint8_t parity_bits(uint8_t v, uint8_t mask) {
    v &= mask;
    v ^= v >> 4;
    v ^= v >> 2;
    v ^= v >> 1;
    return v & 1;
}

static void init_hamming_tables(void) {
    if (hamming_initialized) return;
    /* v0.6.18: rewritten to match gr-lora's hamming_enc_impl.cc:116
     * exactly. The previous version had data bits in positions 0-3
     * (LSB) and parity in 4-7 (MSB) — that worked for header (CR=4)
     * because the bit-reversal of my LSB-first deinterleave canceled
     * with the bit-reversal of my LSB-first Hamming. For payload at
     * CR=1/2/3, the cancellation breaks (cw_len ≠ sf_app+1).
     *
     * gr-lora's encoding (line 116):
     *   out = (d_bin[3] << 7 | d_bin[2] << 6 | d_bin[1] << 5 | d_bin[0] << 4
     *          | p0 << 3 | p1 << 2 | p2 << 1 | p3) >> (4 - cr_app)
     *
     * Where d_bin[0] is MSB of nibble (= data bit 3), d_bin[3] is LSB
     * (= data bit 0). So the encoded codeword has:
     *   bit 7 = data bit 0 (LSB of nibble)
     *   bit 6 = data bit 1
     *   bit 5 = data bit 2
     *   bit 4 = data bit 3 (MSB of nibble)
     *   bit 3 = p0
     *   bit 2 = p1
     *   bit 1 = p2
     *   bit 0 = p3
     *
     * For CR<4, the result is right-shifted by (4-cr_app), discarding
     * the lowest parity bits. So for CR=3 (cw_len=7), we shift right
     * by 1 → drops p3. For CR=1 (cw_len=5), we shift by 3 → drops
     * p1, p2, p3, keeping only p0.
     */
    for (uint8_t d = 0; d < 16; d++) {
        /* MSB-first data bits per gr-lora's int2bool */
        uint8_t d_bin0 = (d >> 3) & 1;  /* MSB of nibble */
        uint8_t d_bin1 = (d >> 2) & 1;
        uint8_t d_bin2 = (d >> 1) & 1;
        uint8_t d_bin3 = (d >> 0) & 1;  /* LSB of nibble */
        uint8_t p0 = d_bin3 ^ d_bin2 ^ d_bin1;
        uint8_t p1 = d_bin2 ^ d_bin1 ^ d_bin0;
        uint8_t p2 = d_bin3 ^ d_bin2 ^ d_bin0;
        uint8_t p3 = d_bin3 ^ d_bin1 ^ d_bin0;
        /* Full 8-bit codeword (CR=4). Lower CRs shift this right. */
        uint8_t cw84 = (uint8_t)(
            (d_bin3 << 7) | (d_bin2 << 6) | (d_bin1 << 5) | (d_bin0 << 4) |
            (p0 << 3)     | (p1 << 2)     | (p2 << 1)     |  p3);
        hamming84_encode_table[d] = cw84;
    }
    /* Build decode table for CR=4 by exhaustive nearest-neighbor
     * search. Distance >2 → uncorrectable. */
    for (int rx = 0; rx < 256; rx++) {
        int best_dist = 99, best_data = 0;
        for (int d = 0; d < 16; d++) {
            int dist = __builtin_popcount(rx ^ hamming84_encode_table[d]);
            if (dist < best_dist) {
                best_dist = dist;
                best_data = d;
            }
        }
        hamming84_decode_table[rx] = (uint8_t)(best_data | (best_dist << 4));
    }
    /* (7,4) — gr-lora shifts CR=4 codeword right by 1, dropping p3.
     * That gives a 7-bit codeword with bits:
     *   bit 6 = data bit 0, bit 5 = bit 1, bit 4 = bit 2, bit 3 = bit 3,
     *   bit 2 = p0, bit 1 = p1, bit 0 = p2. */
    for (int rx = 0; rx < 128; rx++) {
        int best_dist = 99, best_data = 0;
        for (int d = 0; d < 16; d++) {
            int cw7 = hamming84_encode_table[d] >> 1;  /* drop p3 */
            int dist = __builtin_popcount(rx ^ cw7);
            if (dist < best_dist) {
                best_dist = dist;
                best_data = d;
            }
        }
        hamming74_decode_table[rx] = (uint8_t)(best_data | (best_dist << 4));
    }
    hamming_initialized = true;
}

uint8_t lora_hamming_decode(uint8_t codeword, uint8_t cr,
                            bool *corrected, bool *uncorrectable) {
    if (!hamming_initialized) init_hamming_tables();
    uint8_t entry;
    if (cr == 4) {
        entry = hamming84_decode_table[codeword & 0xFF];
    } else if (cr == 3) {
        entry = hamming74_decode_table[codeword & 0x7F];
    } else if (cr == 2) {
        /* (6,4) — gr-lora shifts CR=4 cw right by 2, dropping p3 + p2.
         * Bits: 5=d0, 4=d1, 3=d2, 2=d3, 1=p0, 0=p1.
         * Take data: bits 5..2 → reverse to nibble bit 0..3.
         * No correction (only 2 parity bits = 1 bit error detect). */
        uint8_t d0 = (codeword >> 5) & 1;
        uint8_t d1 = (codeword >> 4) & 1;
        uint8_t d2 = (codeword >> 3) & 1;
        uint8_t d3 = (codeword >> 2) & 1;
        entry = (uint8_t)((d3 << 3) | (d2 << 2) | (d1 << 1) | d0);
    } else {
        /* CR=1: (5,4) — single parity. gr-lora shifts CR=4 cw right
         * by 3, dropping p3 + p2 + p1. Bits: 4=d0, 3=d1, 2=d2, 1=d3, 0=p0.
         * Take data: bits 4..1 → nibble bit 0..3 (with reversal). */
        uint8_t d0 = (codeword >> 4) & 1;
        uint8_t d1 = (codeword >> 3) & 1;
        uint8_t d2 = (codeword >> 2) & 1;
        uint8_t d3 = (codeword >> 1) & 1;
        entry = (uint8_t)((d3 << 3) | (d2 << 2) | (d1 << 1) | d0);
    }
    uint8_t data = entry & 0x0F;
    uint8_t err = entry >> 4;
    if (corrected)     *corrected     = (err == 1);
    if (uncorrectable) *uncorrectable = (err >= 2);
    return data;
}

/* ────────────────────────────────────────────────────────────────────
 * Block deinterleaver
 * ────────────────────────────────────────────────────────────────────
 * LoRa interleaves blocks of sf_app symbols × cw_len bits using a
 * diagonal interleaver. RX-side direct port of gr-lora's
 * deinterleaver_impl.cc:108-133:
 *
 *   for i in [0, cw_len):
 *     inter_bin[i] = int2bool(in[i], sf_app)   // MSB-first bits
 *   for i in [0, cw_len):
 *     for j in [0, sf_app):
 *       deinter_bin[(i - j - 1) mod sf_app][i] = inter_bin[i][j]
 *   for i in [0, sf_app):
 *     out[i] = bool2int(deinter_bin[i])         // MSB-first
 *
 * Where:
 *   • inter_bin[i][j]: bit at position (sf_app-1-j) of in[i]
 *     (MSB at index 0, LSB at index sf_app-1)
 *   • deinter_bin[c]: an array of cw_len bits, MSB-first
 *   • out[c] bit (cw_len-1-r) = deinter_bin[c][r]
 *
 * v0.6.18: rewrite. Previous version computed bit_idx as
 * `(c + cw_len - r - 1) mod sf_app` and packed LSB-first.
 * That formula HAPPENED to give the same data nibbles as gr-lora
 * for the SPECIAL CASE of header (cw_len=8, sf_app=7) — because
 * the bit-reversal of the LSB-packing canceled with the bit-reversal
 * of my old Hamming decoder which read data bits from positions 0-3.
 * For payload at any other (cw_len, sf_app), the cancellation
 * fails and decoded nibbles are wrong (random-looking dewhitened
 * payload, payload CRC fails). Fixing both deinterleave AND Hamming
 * to match gr-lora's MSB-first convention makes all CR values work.
 */
static void deinterleave_block(const uint16_t *symbols_in,
                               uint8_t sf_app, uint8_t cw_len,
                               uint8_t *codewords_out) {
    for (uint8_t c = 0; c < sf_app; c++) {
        uint8_t cw = 0;
        for (uint8_t r = 0; r < cw_len; r++) {
            /* j = (r - c - 1) mod sf_app — gr-lora's index. */
            int j = ((int)r - (int)c - 1);
            j = ((j % (int)sf_app) + (int)sf_app) % (int)sf_app;
            /* inter_bin[r][j] = bit (sf_app-1-j) of sym[r] */
            uint8_t in_bit_idx = (uint8_t)(sf_app - 1 - j);
            uint8_t bit = (uint8_t)((symbols_in[r] >> in_bit_idx) & 1);
            /* MSB-first packing: deinter_bin[c][r] -> output bit
             * (cw_len-1-r) of cw. */
            uint8_t out_bit_idx = (uint8_t)(cw_len - 1 - r);
            cw |= (uint8_t)(bit << out_bit_idx);
        }
        codewords_out[c] = cw;
    }
}

/* ────────────────────────────────────────────────────────────────────
 * Header decode
 * ────────────────────────────────────────────────────────────────────
 * The first SF-2 nibbles after deinterleave/Hamming form the header:
 *   nibble 0     : payload length (high 4 bits)
 *   nibble 1     : payload length (low 4 bits) — combined gives 1..255
 *   nibble 2 b0  : CRC present (1 = yes)
 *   nibble 2 b1-3: payload coding rate (1..4)
 *   nibble 3-5   : header CRC (12 bits) over the previous bits
 *
 * Header is ALWAYS encoded at CR=4 (4/8). The header CRC is a
 * 5-checksum-bit pattern over the length and CR fields (see SX127x
 * datasheet section 4.1.1.6 — the "header info" CRC).
 *
 * Returns 0 on success, -1 on header CRC failure.
 */
static int decode_header(const uint8_t *nibbles,
                         uint32_t *out_len,
                         uint8_t  *out_cr,
                         bool     *out_has_crc) {
    uint8_t pl_hi = nibbles[0] & 0x0F;
    uint8_t pl_lo = nibbles[1] & 0x0F;
    uint32_t pay_len = ((uint32_t)pl_hi << 4) | pl_lo;
    uint8_t info = nibbles[2] & 0x0F;
    uint8_t cr = (info >> 1) & 0x07;
    bool has_crc = (info & 1) != 0;

    /* v0.6.18: header CRC formula rewritten to match gr-lora's
     * header_decoder_impl.cc:141-145 exactly. The previous version
     * had three bugs:
     *   • c2 used (info & 1) where gr-lora uses (info & 2) >> 1
     *   • c1 had spurious (info & 8) >> 3 prefix not in gr-lora
     *   • c0 had spurious (pl_hi & 1) ^ (pl_lo & 2) >> 1 prefix
     *
     * AND the received-CRC extraction was completely wrong:
     *   old: ((nibbles[3] & 0x0F) << 1) | ((nibbles[4] & 0x08) >> 3)
     *        — extracts low 4 bits of n3 + bit 3 of n4 into a 5-bit
     *        value, putting bit 4 = bit 3 of n3 and so on.
     *   new (gr-lora): ((nibbles[3] & 1) << 4) | nibbles[4]
     *        — extracts bit 0 of n3 + all 4 bits of n4 into a 5-bit
     *        value. Total of 5 CRC bits packed across two nibbles.
     *
     * Variables renamed for clarity matching gr-lora's
     * `in[0]`, `in[1]`, `in[2]` = nibbles 0, 1, 2 (length-hi,
     * length-lo, info).
     */
    uint8_t in0 = pl_hi;
    uint8_t in1 = pl_lo;
    uint8_t in2 = info;
    uint8_t c4 =
        ((in0 & 0x8) >> 3) ^ ((in0 & 0x4) >> 2) ^
        ((in0 & 0x2) >> 1) ^  (in0 & 0x1);
    uint8_t c3 =
        ((in0 & 0x8) >> 3) ^ ((in1 & 0x8) >> 3) ^
        ((in1 & 0x4) >> 2) ^ ((in1 & 0x2) >> 1) ^
         (in2 & 0x1);
    uint8_t c2 =
        ((in0 & 0x4) >> 2) ^ ((in1 & 0x8) >> 3) ^
         (in1 & 0x1)       ^ ((in2 & 0x8) >> 3) ^
        ((in2 & 0x2) >> 1);
    uint8_t c1 =
        ((in0 & 0x2) >> 1) ^ ((in1 & 0x4) >> 2) ^
         (in1 & 0x1)       ^ ((in2 & 0x4) >> 2) ^
        ((in2 & 0x2) >> 1) ^  (in2 & 0x1);
    uint8_t c0 =
         (in0 & 0x1)       ^ ((in1 & 0x2) >> 1) ^
        ((in2 & 0x8) >> 3) ^ ((in2 & 0x4) >> 2) ^
        ((in2 & 0x2) >> 1) ^  (in2 & 0x1);
    uint8_t computed = (uint8_t)((c4 << 4) | (c3 << 3) |
                                 (c2 << 2) | (c1 << 1) | c0);
    /* gr-lora line 138: header_chk = ((in[3] & 1) << 4) + in[4]; */
    uint8_t received = (uint8_t)(((nibbles[3] & 0x1) << 4) |
                                  (nibbles[4] & 0x0F));
    if (computed != received) {
        return -1;  /* header CRC fail */
    }
    if (cr < 1 || cr > 4) return -1;
    if (pay_len < 1) return -1;
    *out_len = pay_len;
    *out_cr = cr;
    *out_has_crc = has_crc;
    return 0;
}

/* ────────────────────────────────────────────────────────────────────
 * End-to-end payload decode (called from frame sync once all symbols
 * are demodulated)
 * ────────────────────────────────────────────────────────────────────
 * Pipeline (per gr-lora Fig. 2):
 *   raw symbols (FFT bins)
 *     → Gray decode
 *     → block deinterleave  (cw_len = CR + 4, sf_app = SF-2 for header,
 *                             sf_app = SF-2*ldro for payload)
 *     → Hamming decode      (4-bit nibbles)
 *     → dewhitening         (XOR with whitening LFSR)
 *     → CRC verify          (last 2 bytes)
 *
 * The TRICKY part is that the first block (header + first byte or so)
 * is decoded at CR=4 and sf_app=SF-2 unconditionally. Subsequent blocks
 * use the CR + LDRO from the header.
 */
int lora_decode_payload(const uint16_t *symbols, uint32_t n_symbols,
                        uint8_t sf, uint8_t cr_unused, bool ldro,
                        uint8_t *out, uint32_t out_cap,
                        uint8_t *out_cr, uint8_t *out_has_crc,
                        bool *out_crc_ok) {
    (void)cr_unused;  /* CR comes from the header, not the caller */
    if (sf < LORA_MIN_SF || sf > LORA_MAX_SF) return -1;

    /* ── Block 1: header at CR=4, sf_app = SF-2 (header always uses
     *    -2 LDRO regardless of actual LDRO setting) ─────────────────── */
    uint8_t sf_app_hdr = sf - 2;
    uint8_t cw_len_hdr = 4 + 4;   /* header is always 4/8 */
    if (n_symbols < cw_len_hdr) return -1;

    /* Gray-decode all symbols up front (the TX maps gray-coded values
     * into the chirp; we reverse it before deinterleave). For LDRO and
     * header mode we also drop the bottom bits (which are zero on TX
     * since the modulator can't represent them reliably at slow data
     * rates). */
    uint16_t gray[512];   /* Max packet ≈ 270 syms at SF11/CR1; 512 is
                           * a safe ceiling without alloca. */
    if (n_symbols > 512) return -1;
    uint32_t N = 1u << sf;

    /* v0.6.18 deep diagnostic: dump the header decode pipeline once
     * to localize the failure. Set LORA_DECODE_DEBUG=1 in env to
     * enable. */
    static int dbg_count = 0;
    static int dbg_enabled = -1;
    if (dbg_enabled < 0) {
        const char *env = getenv("LORA_DECODE_DEBUG");
        dbg_enabled = (env && env[0] == '1') ? 1 : 0;
    }
    int dbg = (dbg_enabled && dbg_count < 1) ? 1 : 0;
    if (dbg) {
        dbg_count++;
        fprintf(stderr, "  [pipeline] sf=%u sf_app_hdr=%u cw_len_hdr=%u "
                "n_symbols=%u\n", sf, sf_app_hdr, cw_len_hdr, n_symbols);
        fprintf(stderr, "    raw[0..7]:");
        for (int i = 0; i < 8 && i < (int)n_symbols; i++)
            fprintf(stderr, " %3u", symbols[i]);
        fprintf(stderr, "\n");
    }
    for (uint32_t i = 0; i < n_symbols; i++) {
        uint16_t v = symbols[i];
        /* v0.6.18: per gr-lora fft_demod_impl.cc:313 the formula is
         *
         *   mod(get_symbol_val(in) - 1, (1<<sf)) / ((header||ldro) ? 4 : 1)
         *
         * The -1 subtraction (with mod-N wrap for v=0) compensates
         * for a 1-bin offset that's intrinsic to the LoRa modulator's
         * gray-mapping.
         *
         * Then divide by 4 for header/LDRO modes (drops the bottom 2
         * bits of the chirp value, which the modulator couldn't
         * represent reliably at slow data rates).
         */
        v = (uint16_t)((v + N - 1) % N);
        uint16_t sf_app_used;
        if (i < cw_len_hdr) {
            v = (uint16_t)((v / 4) % (1u << sf_app_hdr));
            sf_app_used = sf_app_hdr;
        } else {
            uint8_t sf_app_pld = (uint8_t)(sf - (ldro ? 2 : 0));
            uint16_t shift = (uint16_t)(ldro ? 4 : 1);
            v = (uint16_t)((v / shift) % (1u << sf_app_pld));
            sf_app_used = sf_app_pld;
        }
        (void)sf_app_used;  /* may be unused in some configurations */
        /* v0.6.18: gr-lora's RX-side gray_mapping_impl.cc uses
         * SINGLE-XOR gray ENCODE (v ^ (v >> 1)) — NOT the standard
         * multi-XOR gray decode. The naming "gray_mapping" is
         * misleading; check the flowgraph: gray_mapping is on RX
         * (between fft_demod and deinterleaver), gray_demap is on TX.
         *
         * Why ENCODE on RX? The TX modulator applies the standard
         * gray DECODE (multi-XOR) + 1 to data nibbles. To invert on
         * RX: subtract 1 (in fft_demod), then apply the inverse of
         * gray_decode = gray_encode (single XOR). Using the wrong
         * direction scrambles every header symbol in a way that
         * looks plausible at first (Hamming decode "succeeds" with
         * borderline distances) but never passes header CRC. */
        gray[i] = lora_gray_encode(v);
    }

    if (dbg) {
        fprintf(stderr, "    after_div:");
        for (int i = 0; i < 8 && i < (int)n_symbols; i++) {
            uint16_t v = symbols[i];
            v = (uint16_t)((v + N - 1) % N);
            v = (uint16_t)((v / 4) % (1u << sf_app_hdr));
            fprintf(stderr, " %3u", v);
        }
        fprintf(stderr, "\n    gray[0-7]:");
        for (int i = 0; i < 8; i++) fprintf(stderr, " %3u", gray[i]);
        fprintf(stderr, "\n");
    }

    /* Deinterleave header block */
    uint8_t hdr_cw[12];   /* sf_app_hdr is at most SF-2 = 10 for SF12 */
    deinterleave_block(gray, sf_app_hdr, cw_len_hdr, hdr_cw);

    if (dbg) {
        fprintf(stderr, "    hdr_cw[0..%u]:", sf_app_hdr - 1);
        for (int i = 0; i < sf_app_hdr; i++)
            fprintf(stderr, " 0x%02X", hdr_cw[i]);
        fprintf(stderr, "\n");
    }

    /* Hamming-decode header nibbles */
    uint8_t hdr_nib[12];
    for (uint8_t i = 0; i < sf_app_hdr; i++) {
        bool corrected, uncorrectable;
        hdr_nib[i] = lora_hamming_decode(hdr_cw[i], 4, &corrected, &uncorrectable);
    }

    if (dbg) {
        fprintf(stderr, "    hdr_nib[0..%u]:", sf_app_hdr - 1);
        for (int i = 0; i < sf_app_hdr; i++)
            fprintf(stderr, " 0x%X", hdr_nib[i] & 0xF);
        fprintf(stderr, "\n");
    }

    /* Parse header. If CRC fails, abort. */
    uint32_t pay_len; uint8_t pay_cr; bool has_crc;
    if (decode_header(hdr_nib, &pay_len, &pay_cr, &has_crc) != 0) {
        return -1;
    }
    *out_cr = pay_cr;
    *out_has_crc = has_crc ? 1 : 0;

    /* Header block also carries the first (sf_app_hdr - 5) data nibbles
     * after the 5-nibble header itself. Pack them into bytes. */
    uint32_t hdr_data_nibs = sf_app_hdr - 5;
    uint8_t pkt_buf[LORA_MAX_PAYLOAD + 2];   /* +2 for CRC */
    memset(pkt_buf, 0, sizeof(pkt_buf));
    uint32_t nib_pos = 0;
    for (uint32_t i = 0; i < hdr_data_nibs; i++) {
        if (nib_pos / 2 >= sizeof(pkt_buf)) break;
        if (nib_pos % 2 == 0) {
            pkt_buf[nib_pos / 2] = (uint8_t)(hdr_nib[5 + i] & 0x0F);
        } else {
            pkt_buf[nib_pos / 2] |= (uint8_t)((hdr_nib[5 + i] & 0x0F) << 4);
        }
        nib_pos++;
    }

    /* ── Subsequent blocks: CR=pay_cr, sf_app = SF - 2*ldro ─────────
     * Each block is sf_app_pld nibbles output from cw_len_pld symbols.
     * We process blocks until we've collected enough nibbles for
     * pay_len + (has_crc ? 2 : 0) bytes. */
    uint8_t sf_app_pld = (uint8_t)(sf - (ldro ? 2 : 0));
    uint8_t cw_len_pld = (uint8_t)(4 + pay_cr);
    uint32_t total_bytes = pay_len + (has_crc ? 2 : 0);
    uint32_t total_nibs_needed = total_bytes * 2;

    uint32_t sym_pos = cw_len_hdr;
    while (nib_pos < total_nibs_needed && sym_pos + cw_len_pld <= n_symbols) {
        uint8_t pld_cw[12];
        deinterleave_block(&gray[sym_pos], sf_app_pld, cw_len_pld, pld_cw);
        sym_pos += cw_len_pld;
        for (uint8_t i = 0; i < sf_app_pld; i++) {
            bool corrected, uncorrectable;
            uint8_t nib = lora_hamming_decode(pld_cw[i], pay_cr,
                                              &corrected, &uncorrectable);
            if (nib_pos / 2 >= sizeof(pkt_buf)) break;
            if (nib_pos % 2 == 0) {
                pkt_buf[nib_pos / 2] = (uint8_t)(nib & 0x0F);
            } else {
                pkt_buf[nib_pos / 2] |= (uint8_t)((nib & 0x0F) << 4);
            }
            nib_pos++;
            if (nib_pos >= total_nibs_needed) break;
        }
    }
    if (nib_pos < total_nibs_needed) return -1;  /* truncated */

    /* Dewhiten the payload (NOT the CRC bytes — the CRC is computed
     * over the WHITENED payload, not the plaintext, per Semtech spec).
     * Note: there's confusion in some online docs about whether CRC
     * is over plaintext or whitened bytes. Real radios CRC the
     * UN-whitened payload bytes per SX127x datasheet rev 7 §4.2.13.6.
     * gr-lora's crc_verif_impl.cc agrees. */
    for (uint32_t i = 0; i < pay_len; i++) {
        pkt_buf[i] ^= lora_whitening_byte(i);
    }

    if (dbg) {
        fprintf(stderr, "    payload (after dewhiten, %u bytes):", pay_len);
        for (uint32_t i = 0; i < pay_len && i < 64; i++) {
            if (i % 16 == 0) fprintf(stderr, "\n      ");
            fprintf(stderr, "%02X ", pkt_buf[i]);
        }
        fprintf(stderr, "\n    crc bytes (raw, NOT dewhitened): %02X %02X\n",
                pkt_buf[pay_len], pkt_buf[pay_len + 1]);
    }

    /* Verify CRC if present
     *
     * v0.6.18: gr-lora's crc_verif_impl.cc uses an unusual CRC verify:
     *   1. Compute CRC-16 over the FIRST (pay_len - 2) bytes
     *   2. XOR the result with the last 2 payload bytes:
     *        crc = crc XOR pay[len-1] XOR (pay[len-2] << 8)
     *   3. Compare to received CRC bytes pkt_buf[pay_len], pkt_buf[pay_len+1]
     *
     * Why? The CRC algorithm is implicitly initialized with the LAST
     * 2 bytes of the payload — equivalent to using them as a non-zero
     * init value applied at the END instead of the beginning. The math
     * works out the same as CRCing over all pay_len bytes with init
     * 0x0000 IF the CRC poly properties allow it, but in practice gr-lora
     * radios produce CRCs that only match this specific algorithm.
     *
     * Pre-fix this code did `lora_crc16(pkt_buf, pay_len)` and got
     * computed=0xC97F received=0x6C3B — pure mismatch on every packet.
     */
    bool crc_ok = true;
    if (has_crc && pay_len >= 2) {
        uint16_t computed = lora_crc16(pkt_buf, pay_len - 2);
        computed ^= (uint16_t)pkt_buf[pay_len - 1];
        computed ^= ((uint16_t)pkt_buf[pay_len - 2]) << 8;
        uint16_t received = (uint16_t)(pkt_buf[pay_len] |
                                       ((uint16_t)pkt_buf[pay_len + 1] << 8));
        if (dbg) {
            fprintf(stderr, "    crc16 computed=0x%04X received=0x%04X %s\n",
                    computed, received,
                    (computed == received) ? "MATCH ✓" : "MISMATCH");
        }
        crc_ok = (computed == received);
    }
    *out_crc_ok = crc_ok;

    /* Copy payload (without CRC) to caller buffer */
    uint32_t copy_len = pay_len;
    if (copy_len > out_cap) copy_len = out_cap;
    memcpy(out, pkt_buf, copy_len);
    return (int)copy_len;
}


/* Compute the total number of symbols needed to fully decode a packet,
 * given the first 8 demodulated symbols (the header block at CR=4).
 *
 * Returns:
 *   > 0 → total symbols needed (8 + payload_blocks × cw_len_pld).
 *         Caller should keep collecting symbols until reaching this
 *         count, then call lora_decode_payload().
 *   < 0 → header CRC failed (this is not a real packet, give up).
 *   = 0 → not enough symbols supplied (need at least 8).
 *
 * Use this in the demod state machine to early-exit when the packet
 * is short. Without this, the demod must wait for MAX_SYMS=320
 * symbols regardless of the actual packet length, adding hundreds of
 * milliseconds of latency for typical short Meshtastic packets.
 */
int lora_compute_symbols_needed(const uint16_t *symbols,
                                uint32_t n_symbols,
                                uint8_t sf, bool ldro) {
    if (sf < LORA_MIN_SF || sf > LORA_MAX_SF) return -1;
    if (n_symbols < 8) return 0;

    /* Replicate the SAME gray-decode + deinterleave + hamming-decode
     * pipeline as lora_decode_payload() for the first 8 (header) symbols.
     * If anything diverges, this fast-path will mis-estimate the
     * symbol count and the slow path will recover (just less efficiently). */
    uint8_t sf_app_hdr = sf - 2;
    uint8_t cw_len_hdr = 4 + 4;
    uint32_t N = 1u << sf;

    uint16_t gray[8];
    for (uint32_t i = 0; i < cw_len_hdr; i++) {
        uint16_t v = symbols[i];
        v = (uint16_t)((v + N - 1) % N);
        v = (uint16_t)((v / 4) % (1u << sf_app_hdr));
        gray[i] = lora_gray_encode(v);
    }

    uint8_t hdr_cw[12];
    deinterleave_block(gray, sf_app_hdr, cw_len_hdr, hdr_cw);

    uint8_t hdr_nib[12];
    for (uint8_t i = 0; i < sf_app_hdr; i++) {
        bool corrected, uncorrectable;
        hdr_nib[i] = lora_hamming_decode(hdr_cw[i], 4, &corrected, &uncorrectable);
    }

    uint32_t pay_len; uint8_t pay_cr; bool has_crc;
    if (decode_header(hdr_nib, &pay_len, &pay_cr, &has_crc) != 0) {
        return -1;   /* header CRC fail */
    }

    /* Header block carries (sf_app_hdr - 5) data nibbles, the rest of
     * the payload comes from subsequent blocks at sf_app_pld nibbles
     * per block. */
    uint32_t total_bytes = pay_len + (has_crc ? 2 : 0);
    uint32_t total_nibs_needed = total_bytes * 2;
    uint32_t hdr_data_nibs = sf_app_hdr - 5;
    if (total_nibs_needed <= hdr_data_nibs) {
        /* Whole packet fits in the header block. We're already done. */
        return (int)cw_len_hdr;
    }
    uint32_t remaining_nibs = total_nibs_needed - hdr_data_nibs;
    uint8_t sf_app_pld = (uint8_t)(sf - (ldro ? 2 : 0));
    uint8_t cw_len_pld = (uint8_t)(4 + pay_cr);
    /* Each block produces sf_app_pld nibbles from cw_len_pld symbols. */
    uint32_t blocks_needed = (remaining_nibs + sf_app_pld - 1) / sf_app_pld;
    uint32_t total_syms = cw_len_hdr + blocks_needed * cw_len_pld;
    return (int)total_syms;
}

