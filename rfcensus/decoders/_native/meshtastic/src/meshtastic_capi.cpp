/**
 * meshtastic_capi.cpp — Implementation of the flat C API.
 *
 * Wraps the C++ header-only library (meshtastic_channel.h etc.) in
 * extern-C functions safe for ctypes / FFI consumers. Allocations
 * use new/delete (the only C++ runtime requirement).
 *
 * License: BSD-3-Clause.
 */
#include "meshtastic_capi.h"
#include "meshtastic_packet.h"
#include "meshtastic_crypto.h"
#include "meshtastic_channel.h"
#include "meshtastic_config.h"

#include <cstring>
#include <new>

/* The opaque table type is just a thin shell around MeshChannelTable
 * so we can pass it back and forth as void* through C ABI. */
struct mesh_capi_table {
    MeshChannelTable inner;
};

/* Map the C-side preset enum to the C++ enum. They have the same
 * numeric values by construction (see capi.h comment), so this is a
 * static_cast — but keep it explicit so a future renumber gets caught
 * at compile time. */
static MeshModemPreset to_cpp_preset(mesh_capi_preset_t p) {
    switch (p) {
        case MESH_PRESET_LONG_FAST:      return MODEM_LONG_FAST;
        case MESH_PRESET_LONG_SLOW:      return MODEM_LONG_SLOW;
        case MESH_PRESET_LONG_MODERATE:  return MODEM_LONG_MODERATE;
        case MESH_PRESET_LONG_TURBO:     return MODEM_LONG_TURBO;
        case MESH_PRESET_MEDIUM_FAST:    return MODEM_MEDIUM_FAST;
        case MESH_PRESET_MEDIUM_SLOW:    return MODEM_MEDIUM_SLOW;
        case MESH_PRESET_SHORT_FAST:     return MODEM_SHORT_FAST;
        case MESH_PRESET_SHORT_SLOW:     return MODEM_SHORT_SLOW;
        case MESH_PRESET_SHORT_TURBO:    return MODEM_SHORT_TURBO;
    }
    return MODEM_LONG_FAST;  /* default fallback */
}

extern "C" {

mesh_capi_table_t *mesh_capi_table_new(mesh_capi_preset_t preset) {
    mesh_capi_table_t *t = new (std::nothrow) mesh_capi_table_t;
    if (!t) return nullptr;
    t->inner.init(to_cpp_preset(preset));
    return t;
}

void mesh_capi_table_free(mesh_capi_table_t *t) {
    delete t;
}

int mesh_capi_table_add(mesh_capi_table_t *t,
                        const char *name,
                        const uint8_t *psk, uint8_t psk_len,
                        int is_primary)
{
    if (!t) return -1;
    return t->inner.addChannel(name ? name : "",
                                psk, psk_len,
                                is_primary != 0);
}

int mesh_capi_table_add_default(mesh_capi_table_t *t) {
    if (!t) return -1;
    return t->inner.addDefaultChannel();
}

int mesh_capi_table_count(const mesh_capi_table_t *t) {
    if (!t) return 0;
    return (int)t->inner.count;
}

int mesh_capi_table_channel_hash(const mesh_capi_table_t *t, int idx) {
    if (!t) return -1;
    if (idx < 0 || idx >= (int)t->inner.count) return -1;
    return (int)t->inner.channels[idx].hash;
}

int mesh_capi_decode(const mesh_capi_table_t *t,
                     const uint8_t *raw, size_t raw_len,
                     mesh_capi_decoded_t *out)
{
    if (!t || !raw || !out) return -3;
    if (raw_len < sizeof(MeshPacketHeader)) return -3;

    /* Step 1: parse the on-wire header into a MeshRxPacket struct. */
    MeshRxPacket pkt;
    if (!meshParsePacket(raw, raw_len, &pkt))
        return -3;

    /* Copy header fields to the C-side struct unconditionally — even
     * if decryption fails the caller still wants to see who/what. */
    out->to            = pkt.to;
    out->from          = pkt.from;
    out->id            = pkt.id;
    out->hop_limit     = pkt.hop_limit;
    out->hop_start     = pkt.hop_start;
    out->want_ack      = pkt.want_ack ? 1 : 0;
    out->via_mqtt      = pkt.via_mqtt ? 1 : 0;
    out->channel_hash  = pkt.channel_hash;
    out->next_hop      = pkt.next_hop;
    out->relay_node    = pkt.relay_node;
    out->channel_index = -1;
    out->plaintext_len = (uint16_t)pkt.payload_len;

    if (pkt.payload_len > sizeof(out->plaintext))
        out->plaintext_len = (uint16_t)sizeof(out->plaintext);

    /* Always copy the raw payload first — if no PSK matches we still
     * give the caller the ciphertext to log / forward. */
    std::memcpy(out->plaintext, pkt.payload, out->plaintext_len);

    /* Step 2: try every configured channel that matches the wire hash.
     * tryDecrypt copies the ciphertext into the output buffer per
     * attempt, so we use a temp buffer and only commit on success. */
    if (t->inner.count == 0)
        return -1;

    uint8_t scratch[MESH_MAX_PAYLOAD];
    MeshCryptoKey matched_key;
    int idx = t->inner.tryDecrypt(pkt.payload, pkt.payload_len,
                                   pkt.channel_hash,
                                   pkt.from, pkt.id,
                                   scratch, &matched_key,
                                   /*validate_fn=*/nullptr);
    if (idx < 0)
        return -1;  /* no channel matched */

    /* Commit the decrypted plaintext to the output. */
    std::memcpy(out->plaintext, scratch, out->plaintext_len);
    out->channel_index = (int8_t)idx;
    return idx;
}

const char *mesh_capi_version(void) {
    return "0.7.0";
}

}  /* extern "C" */
