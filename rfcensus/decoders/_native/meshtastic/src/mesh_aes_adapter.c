/**
 * mesh_aes_adapter.c — Bridge between tiny-AES-c and meshtastic-lite.
 *
 * meshtastic-lite (in software-AES mode) declares:
 *
 *   extern void mesh_aes_block_encrypt(const uint8_t *key, int key_bits,
 *                                       const uint8_t in[16], uint8_t out[16]);
 *
 * and expects the embedder to provide an implementation. This file
 * provides one by dispatching to mesh_aes128_block.c or mesh_aes256_block.c
 * based on the runtime key_bits argument.
 *
 * The split into two files is required because tiny-AES-c selects
 * key/round sizes via COMPILE-TIME macros — a single compilation
 * unit can only support one key size. To get both AES-128 and AES-256
 * in the same shared library we compile two object files (each with
 * exactly one of AES128/AES256 enabled) and link both.
 *
 * Performance: tiny-AES-c is software-only (~30-50 MB/s on Pi 5).
 * For Meshtastic, irrelevant — payloads are <240 bytes, packet rates
 * cap at tens per second per channel. If you ever need GB/s decrypt
 * throughput, swap this for libmbedcrypto with ARMv8 crypto extensions.
 */
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Provided by mesh_aes128_block.c and mesh_aes256_block.c respectively. */
void mesh_aes128_block_encrypt(const uint8_t *key, const uint8_t in[16], uint8_t out[16]);
void mesh_aes256_block_encrypt(const uint8_t *key, const uint8_t in[16], uint8_t out[16]);

void mesh_aes_block_encrypt(const uint8_t *key, int key_bits,
                            const uint8_t in[16], uint8_t out[16])
{
    if (key_bits == 128) {
        mesh_aes128_block_encrypt(key, in, out);
    } else if (key_bits == 256) {
        mesh_aes256_block_encrypt(key, in, out);
    } else {
        /* AES-192 isn't used by Meshtastic; reject by zeroing the
         * output (downstream sanity checks will fail loudly). */
        memset(out, 0, 16);
    }
}

#ifdef __cplusplus
}
#endif

