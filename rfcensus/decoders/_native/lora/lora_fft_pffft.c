/* lora_fft_pffft.c — pffft-backed implementation of the lora_fft
 * API. Uses SSE on x86, NEON on aarch64; falls back to scalar on
 * unsupported architectures (pffft handles this internally).
 *
 * Performance notes:
 *   • pffft's SIMD path needs 16-byte-aligned input/output, which
 *     we satisfy via pffft_aligned_malloc.
 *   • The "work" buffer (a scratch area pffft uses internally) is
 *     allocated once per context and reused on every transform —
 *     avoiding per-call malloc overhead.
 *   • pffft_transform_ordered produces output in natural order
 *     (out[0] = DC, out[1] = bin 1, ...), matching what kiss_fft
 *     produces and what our chirp/dechirp code expects. Do NOT
 *     use pffft_transform (without _ordered) — that produces
 *     output in pffft's internal "z" reordering for chained
 *     transforms.
 *
 * Memory layout: pffft expects interleaved real/imag floats
 * (re,im,re,im,...). Our lora_fft_cpx_t is {float r; float i;}
 * which is layout-compatible with that interleaved format — so
 * we cast the pointer directly without copying.
 */
#include <stdint.h>
#include <stdlib.h>

#include "pffft.h"
#include "lora_fft.h"

struct lora_fft_ctx {
    PFFFT_Setup *setup;
    float       *work;       /* persistent scratch buffer */
    uint32_t     N;          /* transform size in complex samples */
};

lora_fft_ctx_t* lora_fft_new(uint32_t N) {
    /* pffft requires N to be a multiple of pffft_simd_size() (=4 for
     * SSE/NEON, 1 for scalar fallback) AND a "good" composite of
     * 2/3/5. For our power-of-2 LoRa sizes this is always satisfied
     * for N >= 32. We require N >= 32 explicitly so we don't get
     * silent NULL returns from pffft. */
    if (N < 32 || (N & (N - 1)) != 0) {
        return NULL;
    }
    lora_fft_ctx_t *ctx = (lora_fft_ctx_t*)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->N = N;
    ctx->setup = pffft_new_setup((int)N, PFFFT_COMPLEX);
    if (!ctx->setup) {
        free(ctx);
        return NULL;
    }
    /* pffft work buffer: 2*N floats for complex transforms, must be
     * 16-byte aligned. pffft will allocate transparently if we pass
     * NULL on each call, but that's a per-call malloc — we'd rather
     * pay the cost once. */
    ctx->work = (float*)pffft_aligned_malloc(
        (size_t)N * 2 * sizeof(float)
    );
    if (!ctx->work) {
        pffft_destroy_setup(ctx->setup);
        free(ctx);
        return NULL;
    }
    return ctx;
}

void lora_fft_destroy(lora_fft_ctx_t* ctx) {
    if (!ctx) return;
    if (ctx->work) pffft_aligned_free(ctx->work);
    if (ctx->setup) pffft_destroy_setup(ctx->setup);
    free(ctx);
}

void lora_fft_forward(lora_fft_ctx_t* ctx,
                       const lora_fft_cpx_t* in,
                       lora_fft_cpx_t* out) {
    /* Cast lora_fft_cpx_t* (struct {float r; float i;}) to float* —
     * layout-compatible interleaved format that pffft expects. The
     * input buffer is read-only from pffft's perspective even when
     * the API takes a non-const pointer; cast through (void*) to
     * silence the discard-const warning. */
    pffft_transform_ordered(
        ctx->setup,
        (const float*)(const void*)in,
        (float*)out,
        ctx->work,
        PFFFT_FORWARD
    );
}

void* lora_fft_aligned_alloc(size_t bytes) {
    return pffft_aligned_malloc(bytes);
}

void  lora_fft_aligned_free(void* ptr) {
    pffft_aligned_free(ptr);
}

const char* lora_fft_backend_name(void) {
    return "pffft";
}
