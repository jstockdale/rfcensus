/* lora_dechirp_simd.h — vectorized complex pointwise multiply for
 * the LoRa dechirp hot loop.
 *
 * Operation: out[i] = samples[i] * downchirp[i], for i in [0, N).
 * All three buffers are interleaved float pairs (r, i, r, i, ...).
 *
 * Backends:
 *   • LORA_DECHIRP_FORCE_SCALAR — scalar fallback (testing override)
 *   • __ARM_NEON                 → NEON 4-wide complex multiply (Pi 5)
 *   • __SSE3__                   → SSE3 4-wide complex multiply (x86_64)
 *   • else                       → scalar fallback (semantics-defining)
 *
 * Header-only inline so the call sites get full inlining. */
#ifndef LORA_DECHIRP_SIMD_H
#define LORA_DECHIRP_SIMD_H

#include <stdint.h>
#include <stddef.h>
#include <complex.h>

#if defined(LORA_DECHIRP_FORCE_SCALAR)
#  define LORA_DECHIRP_BACKEND "scalar"
#  define LORA_DECHIRP_USE_NEON 0
#  define LORA_DECHIRP_USE_SSE3 0
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define LORA_DECHIRP_BACKEND "neon"
#  define LORA_DECHIRP_USE_NEON 1
#  define LORA_DECHIRP_USE_SSE3 0
#elif defined(__SSE3__)
#  include <pmmintrin.h>
#  define LORA_DECHIRP_BACKEND "sse3"
#  define LORA_DECHIRP_USE_NEON 0
#  define LORA_DECHIRP_USE_SSE3 1
#else
#  define LORA_DECHIRP_BACKEND "scalar"
#  define LORA_DECHIRP_USE_NEON 0
#  define LORA_DECHIRP_USE_SSE3 0
#endif

typedef float _Complex lora_cf_t;

/* Vectorized complex pointwise multiply: out[i] = a[i] * b[i].
 * a, b, out are interleaved float pairs. n is the number of COMPLEX
 * elements. out may alias a or b. */
static inline void lora_dechirp_mul(
    const float * __restrict__ a,
    const float * __restrict__ b,
    float       * __restrict__ out,
    size_t n
) {
#if LORA_DECHIRP_USE_NEON
    /* NEON: 4 complex multiplies per iteration. */
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t va = vld2q_f32(a + 2 * i);
        float32x4x2_t vb = vld2q_f32(b + 2 * i);
        float32x4_t ar = va.val[0], ai = va.val[1];
        float32x4_t br = vb.val[0], bi = vb.val[1];
        /* (a+bi)(c+di) = (ac-bd) + (ad+bc)i */
        float32x4_t real = vmlsq_f32(vmulq_f32(ar, br), ai, bi);
        float32x4_t imag = vmlaq_f32(vmulq_f32(ar, bi), ai, br);
        float32x4x2_t res = { { real, imag } };
        vst2q_f32(out + 2 * i, res);
    }
    for (; i < n; i++) {
        float ar = a[2*i], ai = a[2*i + 1];
        float br = b[2*i], bi = b[2*i + 1];
        out[2*i]     = ar * br - ai * bi;
        out[2*i + 1] = ar * bi + ai * br;
    }
#elif LORA_DECHIRP_USE_SSE3
    /* SSE3: 2 complex multiplies per iteration. Standard idiom. */
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128 va = _mm_loadu_ps(a + 2 * i);
        __m128 vb = _mm_loadu_ps(b + 2 * i);
        __m128 vb_re = _mm_moveldup_ps(vb);
        __m128 vb_im = _mm_movehdup_ps(vb);
        __m128 va_sw = _mm_shuffle_ps(va, va, 0xB1);
        __m128 t1 = _mm_mul_ps(va, vb_re);
        __m128 t2 = _mm_mul_ps(va_sw, vb_im);
        __m128 res = _mm_addsub_ps(t1, t2);
        _mm_storeu_ps(out + 2 * i, res);
    }
    for (; i < n; i++) {
        float ar = a[2*i], ai = a[2*i + 1];
        float br = b[2*i], bi = b[2*i + 1];
        out[2*i]     = ar * br - ai * bi;
        out[2*i + 1] = ar * bi + ai * br;
    }
#else
    /* Scalar reference. */
    for (size_t i = 0; i < n; i++) {
        float ar = a[2*i], ai = a[2*i + 1];
        float br = b[2*i], bi = b[2*i + 1];
        out[2*i]     = ar * br - ai * bi;
        out[2*i + 1] = ar * bi + ai * br;
    }
#endif
}

static inline const char* lora_dechirp_backend(void) {
    return LORA_DECHIRP_BACKEND;
}

const char* lora_dechirp_backend_name(void);

#endif /* LORA_DECHIRP_SIMD_H */
