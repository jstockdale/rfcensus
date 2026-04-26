/* lora_channelizer.c — bit-exact factor-out of ingest_samples_cf
 * from lora_demod.c. See lora_channelizer.h for the design. */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#include "lora_channelizer.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef float _Complex cf_t;

struct lora_channelizer {
    /* Mix oscillator — must produce SAME phasor sequence as
     * ingest_samples_cf in lora_demod.c so a decoder fed via
     * feed_baseband from this channelizer is bit-exact to one
     * fed via feed_cu8. */
    cf_t      mix_phasor;
    cf_t      mix_step;
    uint32_t  mix_norm_counter;
    int       mix_enabled;

    /* Resampler — linear-interp at fractional step. */
    double    resamp_step;     /* sample_rate / bandwidth */
    double    resamp_pos;
    cf_t      resamp_prev;
    int       resamp_have_prev;

    /* Diagnostics */
    uint64_t  samples_in;
    uint64_t  samples_out;
};

lora_channelizer_t* lora_channelizer_new(
    uint32_t sample_rate_hz,
    uint32_t bandwidth_hz,
    int32_t  mix_freq_hz
) {
    if (sample_rate_hz == 0 || bandwidth_hz == 0) return NULL;
    if (bandwidth_hz > sample_rate_hz) return NULL;    /* would upsample */
    lora_channelizer_t *c = (lora_channelizer_t*)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->mix_enabled = (mix_freq_hz != 0);
    if (c->mix_enabled) {
        double w = +2.0 * M_PI * (double)mix_freq_hz
                   / (double)sample_rate_hz;
        c->mix_step = (float)cos(w) + (float)sin(w) * I;
        c->mix_phasor = 1.0f + 0.0f * I;
    } else {
        c->mix_step = 1.0f + 0.0f * I;
        c->mix_phasor = 1.0f + 0.0f * I;
    }
    c->mix_norm_counter = 0;
    c->resamp_step = (double)sample_rate_hz / (double)bandwidth_hz;
    c->resamp_pos = 0.0;
    c->resamp_have_prev = 0;
    c->samples_in = 0;
    c->samples_out = 0;
    return c;
}

void lora_channelizer_free(lora_channelizer_t *c) {
    if (c) free(c);
}

/* The hot loop. cf_t input version — cu8 wrapper below converts.
 * Returns number of complex output samples written. */
static size_t feed_cf_inner(
    lora_channelizer_t *c,
    const float *iq,
    size_t n_in,
    float *out,
    size_t max_out_complex
) {
    size_t produced = 0;
    for (size_t i = 0; i < n_in; i++) {
        cf_t s = iq[2*i] + iq[2*i + 1] * I;
        if (c->mix_enabled) {
            s = s * c->mix_phasor;
            c->mix_phasor = c->mix_phasor * c->mix_step;
            c->mix_norm_counter++;
            if ((c->mix_norm_counter & 8191) == 0) {
                float mag = sqrtf(crealf(c->mix_phasor) * crealf(c->mix_phasor)
                                + cimagf(c->mix_phasor) * cimagf(c->mix_phasor));
                if (mag > 1e-6f) c->mix_phasor /= mag;
            }
        }

        if (!c->resamp_have_prev) {
            c->resamp_prev = s;
            c->resamp_have_prev = 1;
            /* First output IS input[0] — emit immediately. */
            if (produced < max_out_complex) {
                out[2*produced]     = crealf(s);
                out[2*produced + 1] = cimagf(s);
                produced++;
            }
            c->resamp_pos = c->resamp_step;
            continue;
        }

        while (c->resamp_pos <= 1.0) {
            float frac = (float)c->resamp_pos;
            cf_t outsample = c->resamp_prev * (1.0f - frac) + s * frac;
            if (produced < max_out_complex) {
                out[2*produced]     = crealf(outsample);
                out[2*produced + 1] = cimagf(outsample);
                produced++;
            }
            c->resamp_pos += c->resamp_step;
        }
        c->resamp_pos -= 1.0;
        c->resamp_prev = s;
    }
    c->samples_in += n_in;
    c->samples_out += produced;
    return produced;
}

size_t lora_channelizer_feed_cf(
    lora_channelizer_t *c,
    const float *in,
    size_t n_in,
    float *out,
    size_t max_out_complex
) {
    if (!c || !in || !out) return 0;
    return feed_cf_inner(c, in, n_in, out, max_out_complex);
}

size_t lora_channelizer_feed_cu8(
    lora_channelizer_t *c,
    const uint8_t *in,
    size_t n_in,
    float *out,
    size_t max_out_complex
) {
    if (!c || !in || !out) return 0;
    /* Convert in chunks to keep stack reasonable + cache-friendly.
     * Same chunk size as lora_demod_process_cu8 (4096 complex
     * samples = 8KB stack). */
    enum { CHUNK = 4096 };
    float fbuf[CHUNK * 2];
    size_t total_out = 0;
    while (n_in > 0) {
        size_t this_n = n_in > CHUNK ? CHUNK : n_in;
        for (size_t i = 0; i < this_n; i++) {
            fbuf[2*i]     = ((float)in[2*i]     - 127.5f) / 127.5f;
            fbuf[2*i + 1] = ((float)in[2*i + 1] - 127.5f) / 127.5f;
        }
        /* Capacity check: feed_cf_inner gates writes against
         * max_out_complex anyway, but we want to pass the REMAINING
         * capacity each iteration. */
        size_t remaining_cap = (max_out_complex > total_out)
            ? (max_out_complex - total_out) : 0;
        size_t n_written = feed_cf_inner(
            c, fbuf, this_n,
            out + 2 * total_out,
            remaining_cap
        );
        total_out += n_written;
        in += this_n * 2;
        n_in -= this_n;
    }
    return total_out;
}

uint64_t lora_channelizer_samples_in(const lora_channelizer_t *c) {
    return c ? c->samples_in : 0;
}

uint64_t lora_channelizer_samples_out(const lora_channelizer_t *c) {
    return c ? c->samples_out : 0;
}
