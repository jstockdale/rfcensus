/* lora_fft.h — pluggable FFT backend for the LoRa decoder.
 *
 * Two implementations live in this directory:
 *
 *   • lora_fft_kiss.c  — uses kiss_fft (scalar C, portable, baseline).
 *   • lora_fft_pffft.c — uses pffft (Pretty Fast FFT by Pommier; uses
 *                         SSE on x86, NEON on aarch64; ~2-3× faster
 *                         than kiss_fft on Pi 5 / Cortex-A76).
 *
 * Build-time selection: `LORA_FFT_BACKEND=pffft` (default) or
 * `=kiss` on the make command line. Both backends expose this
 * identical API so the caller (lora_demod.c, lora_chirp.c) stays
 * agnostic.
 *
 * The complex-number type ``lora_fft_cpx_t`` is layout-compatible
 * with ``kiss_fft_cpx`` (struct of two floats) AND with the
 * interleaved float pair format pffft expects (re,im,re,im,...) —
 * a contiguous array of N cpx elements has the same memory layout
 * as a 2N-length array of float. So both backends can take the
 * same caller buffers without copying. Buffers must be 16-byte
 * aligned (use ``lora_fft_aligned_alloc`` to be safe — required
 * for SIMD on x86 and ARM).
 *
 * Forward FFT only. Sign convention: out[k] = Σ in[n] · e^(-2πikn/N).
 * No normalization (matches both backends' default and our existing
 * usage).
 */
#ifndef LORA_FFT_H
#define LORA_FFT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float r;
    float i;
} lora_fft_cpx_t;

typedef struct lora_fft_ctx lora_fft_ctx_t;

/* Allocate FFT context for an N-point complex forward transform.
 * Returns NULL on alloc failure. N must be a power of 2; pffft
 * additionally requires N >= 32 (smaller is silently rejected by
 * pffft and we return NULL too). For our LoRa use case N ranges
 * from 256 (SF7 × 2 oversample) to 32768 (SF12 × 8 oversample),
 * always satisfying both constraints. */
lora_fft_ctx_t* lora_fft_new(uint32_t N);

void lora_fft_destroy(lora_fft_ctx_t* ctx);

/* Forward complex FFT, in-place safe (in == out is allowed).
 * Both buffers must be 16-byte aligned (use lora_fft_aligned_alloc).
 * The pffft backend uses an internal pre-allocated work buffer
 * stored in the context to avoid per-call mallocs. */
void lora_fft_forward(lora_fft_ctx_t* ctx,
                       const lora_fft_cpx_t* in,
                       lora_fft_cpx_t* out);

/* SIMD-friendly aligned allocator. Returns 16-byte-aligned memory
 * (or larger alignment if the backend requires it). Free with
 * lora_fft_aligned_free, NOT plain free(). */
void* lora_fft_aligned_alloc(size_t bytes);
void  lora_fft_aligned_free(void* ptr);

/* Diagnostic: which backend was compiled in? Returns "pffft" or
 * "kiss". Used by tests + benchmark output. */
const char* lora_fft_backend_name(void);

#ifdef __cplusplus
}
#endif

#endif /* LORA_FFT_H */
