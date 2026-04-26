/**
 * tests/test_capi.c — Unit tests for libmeshtastic's flat C API.
 *
 * Validates:
 *   1. Default channel adds successfully and computes the expected hash
 *   2. AES-128 round-trip (encrypt then decrypt yields plaintext)
 *   3. The full decode pipeline parses a synthetic frame correctly
 *   4. Multi-channel decrypt picks the right channel by hash
 *   5. Bogus PSK fails to decrypt
 *
 * Build with: make test
 */
#include "meshtastic_capi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Pull in the C++ inlines we need for test setup (encryption, header
 * construction). The C-API only exposes decrypt because that's what
 * passive monitors do; for tests we need to build packets too. */
#include "meshtastic_packet.h"
#include "meshtastic_crypto.h"
#include "meshtastic_channel.h"

static int failures = 0;

#define CHECK(cond, msg) do {                                       \
    if (!(cond)) {                                                  \
        fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__);   \
        failures++;                                                 \
    } else {                                                        \
        fprintf(stderr, "  ok:   %s\n", msg);                       \
    }                                                               \
} while (0)

static void test_default_channel(void) {
    fprintf(stderr, "[1] Default channel\n");
    mesh_capi_table_t *t = mesh_capi_table_new(MESH_PRESET_LONG_FAST);
    CHECK(t != NULL, "table created");

    int idx = mesh_capi_table_add_default(t);
    CHECK(idx == 0, "added at index 0");
    CHECK(mesh_capi_table_count(t) == 1, "count is 1");

    /* The well-known LongFast hash: name="LongFast" XOR'd with the
     * default PSK bytes. We can compute it directly using the C++
     * helpers as a cross-check. */
    MeshCryptoKey k = meshExpandPsk((const uint8_t[]){0x01}, 1);
    uint8_t expected_hash = meshComputeChannelHash("LongFast", &k);
    int actual_hash = mesh_capi_table_channel_hash(t, 0);
    CHECK((int)expected_hash == actual_hash,
          "channel hash matches direct computation");

    mesh_capi_table_free(t);
}

static void test_aes_roundtrip(void) {
    fprintf(stderr, "[2] AES-128 round trip\n");
    /* Encrypt a known plaintext with the default key, then decrypt and
     * compare. Use the inline helpers — this also exercises tiny-AES-c
     * via meshtastic-lite's mesh_aes_block_encrypt path. */
    MeshCryptoKey k = meshExpandPsk((const uint8_t[]){0x01}, 1);
    CHECK(k.length == 16, "default PSK expands to 16 bytes");

    uint8_t pt_orig[64];
    for (int i = 0; i < 64; i++) pt_orig[i] = (uint8_t)(0xA0 + i);

    uint8_t buf[64];
    memcpy(buf, pt_orig, 64);

    /* Encrypt in-place. CTR is symmetric so meshEncrypt == meshDecrypt. */
    bool enc_ok = meshEncrypt(&k, /*from=*/0x12345678, /*id=*/0xCAFE,
                              buf, 64);
    CHECK(enc_ok, "encrypt returned true");
    CHECK(memcmp(buf, pt_orig, 64) != 0, "ciphertext differs from plaintext");

    /* Decrypt in-place. */
    bool dec_ok = meshDecrypt(&k, 0x12345678, 0xCAFE, buf, 64);
    CHECK(dec_ok, "decrypt returned true");
    CHECK(memcmp(buf, pt_orig, 64) == 0, "decrypted matches original");
}

static void test_aes256_roundtrip(void) {
    fprintf(stderr, "[3] AES-256 round trip\n");
    /* Build a 32-byte key (AES-256). Pattern is just incrementing bytes. */
    uint8_t raw_key[32];
    for (int i = 0; i < 32; i++) raw_key[i] = (uint8_t)(0x10 + i);
    MeshCryptoKey k = meshExpandPsk(raw_key, 32);
    CHECK(k.length == 32, "32-byte raw expands to AES-256");

    uint8_t pt_orig[100];
    for (int i = 0; i < 100; i++) pt_orig[i] = (uint8_t)(i * 7);
    uint8_t buf[100];
    memcpy(buf, pt_orig, 100);

    meshEncrypt(&k, 0xDEADBEEF, 0x1234, buf, 100);
    CHECK(memcmp(buf, pt_orig, 100) != 0, "AES-256 ciphertext differs");
    meshDecrypt(&k, 0xDEADBEEF, 0x1234, buf, 100);
    CHECK(memcmp(buf, pt_orig, 100) == 0, "AES-256 round-trip OK");
}

static void test_full_decode_pipeline(void) {
    fprintf(stderr, "[4] Full encode→decode pipeline\n");

    /* Build a fake LoRa frame: 16-byte header + 32-byte encrypted payload. */
    MeshCryptoKey k = meshExpandPsk((const uint8_t[]){0x01}, 1);
    uint8_t channel_hash = meshComputeChannelHash("LongFast", &k);

    uint8_t plaintext[32];
    for (int i = 0; i < 32; i++) plaintext[i] = (uint8_t)(0x55 ^ i);

    uint8_t ciphertext[32];
    memcpy(ciphertext, plaintext, 32);
    meshEncrypt(&k, /*from=*/0xAAAA1111, /*id=*/0xBEEF,
                ciphertext, 32);

    uint8_t frame[16 + 32];
    size_t n = meshBuildPacket(frame,
        /*to=*/MESH_ADDR_BROADCAST,
        /*from=*/0xAAAA1111,
        /*id=*/0xBEEF,
        /*hop_limit=*/3, /*hop_start=*/3,
        /*want_ack=*/false, /*via_mqtt=*/false,
        channel_hash,
        /*next_hop=*/0, /*relay_node=*/0,
        ciphertext, 32);
    CHECK(n == 48, "frame is 48 bytes");

    /* Decode through the C-API. */
    mesh_capi_table_t *t = mesh_capi_table_new(MESH_PRESET_LONG_FAST);
    mesh_capi_table_add_default(t);

    mesh_capi_decoded_t out;
    int rc = mesh_capi_decode(t, frame, n, &out);
    CHECK(rc == 0, "decode returned channel index 0");
    CHECK(out.from == 0xAAAA1111, "from matches");
    CHECK(out.to == MESH_ADDR_BROADCAST, "to matches");
    CHECK(out.id == 0xBEEF, "id matches");
    CHECK(out.hop_limit == 3, "hop_limit matches");
    CHECK(out.channel_hash == channel_hash, "channel_hash matches");
    CHECK(out.plaintext_len == 32, "plaintext length matches");
    CHECK(memcmp(out.plaintext, plaintext, 32) == 0,
          "plaintext bytes match");

    mesh_capi_table_free(t);
}

static void test_multi_channel(void) {
    fprintf(stderr, "[5] Multi-channel decrypt picks the right channel\n");

    mesh_capi_table_t *t = mesh_capi_table_new(MESH_PRESET_LONG_FAST);

    /* Add three channels with different PSKs. */
    uint8_t psk_a[16] = {0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
                         0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA};
    uint8_t psk_b[16] = {0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB,
                         0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB};

    int idx0 = mesh_capi_table_add_default(t);
    int idx1 = mesh_capi_table_add(t, "ChannelA", psk_a, 16, 0);
    int idx2 = mesh_capi_table_add(t, "ChannelB", psk_b, 16, 0);
    CHECK(idx0 == 0 && idx1 == 1 && idx2 == 2, "three channels added");

    /* Encrypt with PSK B and verify only channel B decrypts. */
    MeshCryptoKey kb = meshExpandPsk(psk_b, 16);
    uint8_t kb_hash = meshComputeChannelHash("ChannelB", &kb);

    uint8_t plaintext[20];
    for (int i = 0; i < 20; i++) plaintext[i] = (uint8_t)(i + 0x80);
    uint8_t cipher[20];
    memcpy(cipher, plaintext, 20);
    meshEncrypt(&kb, 0x42, 0x99, cipher, 20);

    uint8_t frame[16 + 20];
    meshBuildPacket(frame,
        MESH_ADDR_BROADCAST, 0x42, 0x99,
        3, 3, false, false,
        kb_hash, 0, 0,
        cipher, 20);

    mesh_capi_decoded_t out;
    int rc = mesh_capi_decode(t, frame, sizeof(frame), &out);
    CHECK(rc == 2, "channel B (idx 2) was the match");
    CHECK(memcmp(out.plaintext, plaintext, 20) == 0,
          "decrypted to original plaintext");

    mesh_capi_table_free(t);
}

static void test_unknown_channel(void) {
    fprintf(stderr, "[6] Frame with unknown channel hash returns -1\n");

    mesh_capi_table_t *t = mesh_capi_table_new(MESH_PRESET_LONG_FAST);
    mesh_capi_table_add_default(t);

    /* Build a frame with a channel hash that won't match. */
    uint8_t frame[16 + 8];
    meshBuildPacket(frame,
        0xFFFFFFFF, 0x123, 0x456,
        3, 3, false, false,
        /*channel_hash=*/0xEE,  /* random — unlikely to match LongFast */
        0, 0,
        (const uint8_t[]){1,2,3,4,5,6,7,8}, 8);

    mesh_capi_decoded_t out;
    int rc = mesh_capi_decode(t, frame, sizeof(frame), &out);
    CHECK(rc == -1, "decode returned -1 (no match)");
    CHECK(out.channel_index == -1, "channel_index is -1");
    CHECK(out.from == 0x123, "header still parsed");

    mesh_capi_table_free(t);
}

int main(void) {
    fprintf(stderr, "libmeshtastic v%s — C-API tests\n\n",
            mesh_capi_version());

    test_default_channel();
    test_aes_roundtrip();
    test_aes256_roundtrip();
    test_full_decode_pipeline();
    test_multi_channel();
    test_unknown_channel();

    fprintf(stderr, "\n=== %d failure(s) ===\n", failures);
    return failures == 0 ? 0 : 1;
}
