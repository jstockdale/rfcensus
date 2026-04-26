/**
 * meshtastic_capi.h — Flat C-extern API for the meshtastic-lite library.
 *
 * This header exposes the operations needed by external callers
 * (Python via ctypes, other languages, or just C consumers who don't
 * want to pull in the full C++ header set). Internally it wraps the
 * existing meshtastic_channel.h / meshtastic_packet.h / meshtastic_crypto.h
 * inlines without modifying them.
 *
 * Memory model:
 *   • The channel table is allocated by mesh_capi_table_new() and must
 *     be released with mesh_capi_table_free(). All channels added via
 *     mesh_capi_table_add() are owned by the table.
 *   • Decode results are filled into caller-provided MeshDecodedPacket
 *     structs; no internal buffers escape.
 *   • All functions are thread-safe with respect to DIFFERENT table
 *     instances. Concurrent calls on the SAME table are NOT safe (no
 *     internal locking).
 *
 * Error model:
 *   • Lifecycle functions return NULL or 0 / negative on failure.
 *   • Decode functions return >= 0 on success (channel index that
 *     decrypted), -1 if no channel matched the wire hash, -2 if all
 *     matching channels failed to decrypt (bad PSK), -3 on bad input.
 *
 * License: BSD-3-Clause (matches meshtastic-lite).
 */
#ifndef MESHTASTIC_CAPI_H
#define MESHTASTIC_CAPI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Modem preset enum mirrored to C. Must match MeshModemPreset in
 * meshtastic_config.h. The numeric values are part of our wire-level
 * compatibility surface — DO NOT renumber. */
typedef enum {
    MESH_PRESET_LONG_FAST       = 0,
    MESH_PRESET_LONG_SLOW       = 1,
    MESH_PRESET_LONG_MODERATE   = 2,
    MESH_PRESET_LONG_TURBO      = 3,
    MESH_PRESET_MEDIUM_FAST     = 4,
    MESH_PRESET_MEDIUM_SLOW     = 5,
    MESH_PRESET_SHORT_FAST      = 6,
    MESH_PRESET_SHORT_SLOW      = 7,
    MESH_PRESET_SHORT_TURBO     = 8,
} mesh_capi_preset_t;

/* Result of decoding a single LoRa frame. Caller allocates; mesh_capi
 * fills. */
typedef struct {
    uint32_t to;                /* destination NodeNum */
    uint32_t from;              /* source NodeNum */
    uint32_t id;                /* packet ID (lower 32 bits) */
    uint8_t  hop_limit;         /* 0..7 — remaining hops */
    uint8_t  hop_start;         /* 0..7 — initial hops at TX */
    uint8_t  want_ack;          /* 1 if want_ack flag set */
    uint8_t  via_mqtt;          /* 1 if relayed via MQTT */
    uint8_t  channel_hash;      /* wire-side channel hash */
    uint8_t  next_hop;          /* low byte of next-hop NodeNum */
    uint8_t  relay_node;        /* low byte of relaying NodeNum */
    int8_t   channel_index;     /* index in the channel table that
                                 * decrypted this packet, or -1 */
    uint16_t plaintext_len;     /* length of plaintext (after header,
                                 * without the 16-byte header) */
    uint8_t  plaintext[239];    /* MESH_MAX_PAYLOAD - MESH_HEADER_LEN
                                 * = 255 - 16 = 239 bytes max */
} mesh_capi_decoded_t;

/* Opaque channel table handle. */
typedef struct mesh_capi_table mesh_capi_table_t;

/* ─── Lifecycle ──────────────────────────────────────────────────── */

/* Allocate a channel table for a given preset. The preset is needed
 * for resolving empty channel names to "LongFast" / "MediumFast" etc.
 * during hash computation.
 *
 * Returns NULL on alloc failure or invalid preset. */
mesh_capi_table_t *mesh_capi_table_new(mesh_capi_preset_t preset);

/* Release a channel table and all channels it owns. */
void mesh_capi_table_free(mesh_capi_table_t *t);

/* Add a channel to the table. Returns the channel index on success,
 * or -1 if the table is full (MESH_MAX_CHANNELS = 8 by default).
 *
 * `name`: channel name (NULL or "" → use the table's preset name).
 * `psk`: raw PSK bytes (may be NULL with psk_len=0 for unencrypted).
 * `psk_len`: 0 (no encrypt), 1 (short index — PSK index 1 = default
 *   key, 2..N are derived), 16 (AES-128), or 32 (AES-256).
 * `is_primary`: 1 if this is the primary channel (used for inheritance
 *   when secondary channels have no PSK). */
int mesh_capi_table_add(mesh_capi_table_t *t,
                        const char *name,
                        const uint8_t *psk, uint8_t psk_len,
                        int is_primary);

/* Convenience: add the default LongFast channel (preset's default name,
 * PSK index 1 = MESH_DEFAULT_PSK). Equivalent to mesh_capi_table_add
 * with name="" and psk=[0x01]. Returns the channel index or -1. */
int mesh_capi_table_add_default(mesh_capi_table_t *t);

/* Number of channels currently in the table. */
int mesh_capi_table_count(const mesh_capi_table_t *t);

/* Get the wire hash for a channel by index, or -1 if out of range.
 * Useful for diagnostics: "did the packet's channel hash 0x68 match
 * any of our configured channels?" */
int mesh_capi_table_channel_hash(const mesh_capi_table_t *t, int idx);

/* ─── Decode + decrypt ───────────────────────────────────────────── */

/* Parse + decrypt a raw LoRa frame.
 *
 * `raw`: bytes from the LoRa PHY decoder (after dewhitening, after
 *   payload-CRC validation if available). Must be >= 16 bytes (header).
 * `raw_len`: total frame length including header.
 * `out`: caller-allocated decoded packet struct.
 *
 * Returns:
 *   >= 0: matched and decrypted; value is the channel index used.
 *         out->channel_index is also set to this value.
 *         out->plaintext / plaintext_len contain the decrypted payload.
 *   -1: no channel in the table had a matching wire hash. The header
 *       is still parsed and copied into `out`; plaintext is the raw
 *       (still-encrypted) ciphertext and channel_index is -1.
 *   -2: a channel matched the hash but decryption produced gibberish
 *       (validate_fn rejected it). Currently we don't apply a
 *       validator so this won't happen in practice.
 *   -3: bad input (NULL, too short).
 */
int mesh_capi_decode(const mesh_capi_table_t *t,
                     const uint8_t *raw, size_t raw_len,
                     mesh_capi_decoded_t *out);

/* ─── Library version ────────────────────────────────────────────── */

/* Returns a NUL-terminated version string like "0.7.0". */
const char *mesh_capi_version(void);

#ifdef __cplusplus
}
#endif

#endif /* MESHTASTIC_CAPI_H */
