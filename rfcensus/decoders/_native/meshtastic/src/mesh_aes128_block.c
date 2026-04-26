/**
 * mesh_aes128_block.c — Thin AES-128 single-block encrypt wrapper.
 *
 * Compiled as a SEPARATE object file with ONLY AES128 enabled in
 * tiny-AES-c. This is required because tiny-AES-c selects key/round
 * sizes via compile-time macros (#define AES128 vs AES256), so we
 * can't have both modes in the same compilation unit.
 *
 * The dispatcher in mesh_aes_adapter.c picks between this and the
 * matching mesh_aes256_block.c based on the runtime key_bits arg.
 */
#include <stdint.h>
#include <string.h>

/* Force AES-128 only for this compilation unit. We #include the
 * tiny-AES-c .c file directly — NOT aes.h — so the implementation
 * is compiled with these macros active. Each key-size variant must
 * have its own .o file (mesh_aes128_block.o, mesh_aes256_block.o)
 * because the macros bake into the round-key tables and round count. */
#undef AES128
#undef AES256
#undef AES192
#define AES128 1
#define ECB    1
#define CBC    0
#define CTR    0

/* Make tiny-AES-c's globals static so we don't get duplicate-symbol
 * link errors with the AES-256 variant (both files would otherwise
 * export the same symbol names). */
#define AES_TYPE_STATIC

/* Rename the public functions so they don't collide with the AES-256
 * compile unit. */
#define AES_init_ctx           AES128_init_ctx
#define AES_init_ctx_iv        AES128_init_ctx_iv
#define AES_ctx_set_iv         AES128_ctx_set_iv
#define AES_ECB_encrypt        AES128_ECB_encrypt
#define AES_ECB_decrypt        AES128_ECB_decrypt
#define AES_CBC_encrypt_buffer AES128_CBC_encrypt_buffer
#define AES_CBC_decrypt_buffer AES128_CBC_decrypt_buffer
#define AES_CTR_xcrypt_buffer  AES128_CTR_xcrypt_buffer

#include "../third_party/tiny-AES-c/aes.c"

void mesh_aes128_block_encrypt(const uint8_t *key,
                               const uint8_t in[16],
                               uint8_t out[16])
{
    struct AES_ctx ctx;
    AES128_init_ctx(&ctx, key);
    memcpy(out, in, 16);
    AES128_ECB_encrypt(&ctx, out);
}
