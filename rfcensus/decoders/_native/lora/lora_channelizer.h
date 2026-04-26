/* lora_channelizer.h — shared mix + resample for one slot frequency.
 *
 * Purpose: bit-exact factor-out of the per-decoder mix + linear-interp
 * resampler that lives inside lora_demod.c. When N SF decoders are
 * watching the same slot frequency (typically 5 at BW=250), running
 * one channelizer + N feed_baseband calls is N× cheaper than running
 * N copies of the same mix+resamp inside each decoder.
 *
 * Bit-exactness: this implementation copies the EXACT formulas from
 * ingest_samples_cf in lora_demod.c (mix oscillator, periodic
 * renormalize, linear-interp resampler) so a decoder fed via
 * feed_baseband + this channelizer produces identical output to
 * a decoder fed via feed_cu8.
 *
 * Output format: float pairs (interleaved I/Q) ready to hand to
 * lora_demod_feed_baseband. The caller owns the output buffer and
 * sizes it for the worst-case number of output samples per call:
 *
 *   max_outputs = ceil(input_samples / decimation_ratio) + 1
 *
 * For 65536-sample input chunks at 9.6× decimation (2.4MS/s →
 * 250kHz), max_outputs ≈ 6826. Allocate 8192 floats × 2 = 16KB and
 * you're safe.
 */
#ifndef LORA_CHANNELIZER_H
#define LORA_CHANNELIZER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lora_channelizer lora_channelizer_t;

/* Construct a channelizer for one slot.
 *
 *   sample_rate_hz: input sample rate (e.g. 2_400_000)
 *   bandwidth_hz:   slot bandwidth = output sample rate (e.g. 250_000)
 *   mix_freq_hz:    frequency to shift TO DC, in Hz. Typically
 *                   center_freq_hz - slot_freq_hz. Sign convention
 *                   matches lora_demod: the input is multiplied by
 *                   exp(+j·2π·mix·t).
 *
 * Returns NULL on alloc failure or invalid params.
 */
lora_channelizer_t* lora_channelizer_new(
    uint32_t sample_rate_hz,
    uint32_t bandwidth_hz,
    int32_t  mix_freq_hz
);

void lora_channelizer_free(lora_channelizer_t *c);

/* Feed cu8 samples; emit channelized baseband float-pairs.
 *
 *   in: pointer to interleaved cu8 (I,Q each 0..255 centered on 127.5)
 *   n_in: number of complex input samples (= bytes/2)
 *   out: pre-allocated output buffer for interleaved float32 I/Q
 *   max_out_complex: capacity of `out` in COMPLEX samples (= floats/2).
 *                    Must be >= ceil(n_in / decim_ratio) + 1 to be safe.
 *
 * Returns the number of complex samples written to `out`.
 */
size_t lora_channelizer_feed_cu8(
    lora_channelizer_t *c,
    const uint8_t *in,
    size_t n_in,
    float *out,
    size_t max_out_complex
);

/* Same as feed_cu8 but for already-converted float input. */
size_t lora_channelizer_feed_cf(
    lora_channelizer_t *c,
    const float *in,
    size_t n_in,
    float *out,
    size_t max_out_complex
);

/* Diagnostic: total input samples consumed since construction. */
uint64_t lora_channelizer_samples_in(const lora_channelizer_t *c);
uint64_t lora_channelizer_samples_out(const lora_channelizer_t *c);

#ifdef __cplusplus
}
#endif

#endif /* LORA_CHANNELIZER_H */
