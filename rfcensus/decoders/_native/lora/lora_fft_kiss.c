/* lora_fft_kiss.c — kiss_fft-backed implementation of the lora_fft
 * API. Baseline / fallback. Used when LORA_FFT_BACKEND=kiss is
 * passed to make.
 *
 * No SIMD; pure scalar C. Slower than pffft on platforms with
 * working SIMD (Pi 5 / x86_64 with SSE), but maximally portable —
 * works on architectures pffft doesn't support without hand-tuning.
 *
 * Memory layout: kiss_fft_cpx is {float r; float i;}, identical to
 * lora_fft_cpx_t. We can cast the pointer directly without copying.
 * No explicit alignment requirement (kiss_fft doesn't use SIMD),
 * but lora_fft_aligned_alloc still returns 16-byte alignment so
 * callers can swap backends without code changes.
 */
#include <stdint.h>
#include <stdlib.h>

#include "kiss_fft.h"
#include "lora_fft.h"

struct lora_fft_ctx {
    kiss_fft_cfg cfg;
    uint32_t     N;
};

lora_fft_ctx_t* lora_fft_new(uint32_t N) {
    if (N < 2 || (N & (N - 1)) != 0) return NULL;
    lora_fft_ctx_t *ctx = (lora_fft_ctx_t*)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->N = N;
    /* kiss_fft_alloc(N, inverse_fft, mem, lenmem):
     *   inverse_fft = 0 for forward
     *   mem/lenmem  = NULL/NULL → kiss_fft mallocs internally. */
    ctx->cfg = kiss_fft_alloc((int)N, 0, NULL, NULL);
    if (!ctx->cfg) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void lora_fft_destroy(lora_fft_ctx_t* ctx) {
    if (!ctx) return;
    if (ctx->cfg) kiss_fft_free(ctx->cfg);
    free(ctx);
}

void lora_fft_forward(lora_fft_ctx_t* ctx,
                       const lora_fft_cpx_t* in,
                       lora_fft_cpx_t* out) {
    /* lora_fft_cpx_t and kiss_fft_cpx are layout-compatible
     * (both are {float r; float i;}). Cast the pointer directly. */
    kiss_fft(
        ctx->cfg,
        (const kiss_fft_cpx*)in,
        (kiss_fft_cpx*)out
    );
}

/* No SIMD requirement, but keep the alignment contract for backend
 * portability — code written against this API must work whether
 * the active backend is pffft (needs alignment) or kiss (doesn't). */
void* lora_fft_aligned_alloc(size_t bytes) {
    void *p = NULL;
    /* posix_memalign returns 0 on success. 16-byte alignment matches
     * SSE/NEON requirements that pffft would need; kiss doesn't care. */
    if (posix_memalign(&p, 16, bytes) != 0) {
        return NULL;
    }
    return p;
}

void  lora_fft_aligned_free(void* ptr) {
    free(ptr);
}

const char* lora_fft_backend_name(void) {
    return "kiss";
}
