/**
 * mesh_aes256_block.c — Thin AES-256 single-block encrypt wrapper.
 * See mesh_aes128_block.c for why this is a separate compile unit.
 */
#include <stdint.h>
#include <string.h>

#undef AES128
#undef AES256
#undef AES192
#define AES256 1
#define ECB    1
#define CBC    0
#define CTR    0

#define AES_init_ctx           AES256_init_ctx
#define AES_init_ctx_iv        AES256_init_ctx_iv
#define AES_ctx_set_iv         AES256_ctx_set_iv
#define AES_ECB_encrypt        AES256_ECB_encrypt
#define AES_ECB_decrypt        AES256_ECB_decrypt
#define AES_CBC_encrypt_buffer AES256_CBC_encrypt_buffer
#define AES_CBC_decrypt_buffer AES256_CBC_decrypt_buffer
#define AES_CTR_xcrypt_buffer  AES256_CTR_xcrypt_buffer

#include "../third_party/tiny-AES-c/aes.c"

void mesh_aes256_block_encrypt(const uint8_t *key,
                               const uint8_t in[16],
                               uint8_t out[16])
{
    struct AES_ctx ctx;
    AES256_init_ctx(&ctx, key);
    memcpy(out, in, 16);
    AES256_ECB_encrypt(&ctx, out);
}
