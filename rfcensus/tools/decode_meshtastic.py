"""decode_meshtastic — Standalone Meshtastic-from-SDR decoder.

Reads IQ from any of three sources (cu8 file / rtl_sdr subprocess /
rtl_tcp connection), runs the LoRa physical-layer decoder, attempts
AES-CTR decryption against configured channel PSKs, and writes results
to stdout / PCAP / JSONL.

By default ("auto" preset selection), every Meshtastic preset whose
default frequency slot falls within the dongle's passband is decoded
in parallel — one ``LoraDecoder`` instance per preset, all sharing
the same IQ stream and the same channel-PSK table.

Examples::

    # Decode a saved capture
    rfcensus-meshtastic-decode capture.cu8 \\
        --frequency 913500000 --sample-rate 1000000

    # Live decode from dongle 0, all presets in passband, write PCAP
    rfcensus-meshtastic-decode --rtl-sdr --device 0 \\
        --frequency 915000000 --sample-rate 2400000 \\
        --pcap mesh.pcap

    # Live from dongle by serial, only MediumFast
    rfcensus-meshtastic-decode --rtl-sdr --device 00000003 \\
        --frequency 913125000 --sample-rate 1000000 \\
        --preset MEDIUM_FAST

    # Share a dongle through rtl_tcp + use a custom channel PSK
    rfcensus-meshtastic-decode --rtl-tcp 127.0.0.1:1234 \\
        --frequency 913125000 --sample-rate 1000000 \\
        --psk MyPrivate:01020304050607080900aabbccddeeff

    # See what preset slots exist in the US band
    rfcensus-meshtastic-decode --list-slots --region US
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from rfcensus.decoders.meshtastic_native import (
    MeshtasticDecoder, PRESETS as MESH_PRESETS,
)
from rfcensus.decoders.meshtastic_pipeline import (
    MultiPresetPipeline, PipelinePacket,
)
from rfcensus.decoders.lazy_pipeline import LazyMultiPresetPipeline
from rfcensus.utils.iq_source import (
    DEFAULT_CHUNK_SIZE,
    FileIQSource,
    IQSource,
    RtlSdrConfig,
    RtlSdrSubprocess,
    RtlTcpSource,
    find_device_index_by_serial,
)
from rfcensus.utils.iq_reader import (
    IqReader,
    DEFAULT_RING_CAPACITY_SECS,
)
from rfcensus.utils.meshtastic_region import (
    PRESETS, REGIONS,
    all_default_slots, default_slot,
    enumerate_all_slots_in_passband, slots_in_passband,
)
from rfcensus.utils.pcap_loratap import PcapLoraTapWriter


# ─────────────────────────────────────────────────────────────────────
# Minimal protobuf parsing — just enough for portnum + payload
# ─────────────────────────────────────────────────────────────────────

_PORTNUMS = {
    0:  "UNKNOWN_APP",        1: "TEXT_MESSAGE_APP",
    2:  "REMOTE_HARDWARE_APP", 3: "POSITION_APP",
    4:  "NODEINFO_APP",        5: "ROUTING_APP",
    6:  "ADMIN_APP",           7: "TEXT_MESSAGE_COMPRESSED_APP",
    8:  "WAYPOINT_APP",        9: "AUDIO_APP",
    10: "DETECTION_SENSOR_APP", 32: "REPLY_APP",
    33: "IP_TUNNEL_APP",       34: "PAXCOUNTER_APP",
    64: "SERIAL_APP",          65: "STORE_FORWARD_APP",
    66: "RANGE_TEST_APP",      67: "TELEMETRY_APP",
    68: "ZPS_APP",             69: "SIMULATOR_APP",
    70: "TRACEROUTE_APP",      71: "NEIGHBORINFO_APP",
    72: "ATAK_PLUGIN",         73: "MAP_REPORT_APP",
    74: "POWERSTRESS_APP",
}


def _parse_data_envelope(plaintext: bytes) -> Optional[tuple[int, bytes]]:
    """Pull (portnum, payload) from the outer Meshtastic Data protobuf."""
    if len(plaintext) < 4 or plaintext[0] != 0x08:
        return None
    if plaintext[1] & 0x80:
        return None
    portnum = plaintext[1]
    if plaintext[2] != 0x12:
        return None
    paylen = plaintext[3]
    if 4 + paylen > len(plaintext):
        return None
    return portnum, plaintext[4 : 4 + paylen]


def _try_decode_position(payload: bytes) -> Optional[tuple[float, float]]:
    """Pull lat/lon from a POSITION_APP payload (fixed32 × 1e7 encoding)."""
    if len(payload) < 10 or payload[0] != 0x0D or payload[5] != 0x15:
        return None
    lat = struct.unpack("<i", payload[1:5])[0] / 1e7
    lon = struct.unpack("<i", payload[6:10])[0] / 1e7
    return lat, lon


# v0.7.6: minimal protobuf wire-format reader. We don't pull in the
# generated .pb2.py from upstream Meshtastic — too much code and a
# protoc dependency for a passive observer. Instead we hand-decode
# the few message shapes we care about (Telemetry.device_metrics +
# User), using the fact that protobuf wire format is just
# (field_no << 3) | wire_type tag bytes followed by varint / fixed
# / length-prefixed values. This gets us battery/voltage/utilization
# from telemetry and short_name/long_name/hw_model from nodeinfo
# without dragging in the protobuf runtime.

def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Decode one varint starting at pos. Returns (value, next_pos)."""
    result = 0
    shift = 0
    while pos < len(buf):
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift > 63:    # malformed
            raise ValueError("varint too long")
    raise ValueError("truncated varint")


def _read_pb_fields(buf: bytes) -> dict[int, list]:
    """Walk a protobuf message and return {field_no: [values...]}.

    Each value is a tuple (wire_type, raw) where raw is the bytes
    (for length-prefixed) or the decoded int (for varint/fixed).
    Best-effort — returns {} on malformed input rather than raising.
    """
    out: dict[int, list] = {}
    pos = 0
    try:
        while pos < len(buf):
            tag, pos = _read_varint(buf, pos)
            field_no = tag >> 3
            wire_type = tag & 0x7
            if wire_type == 0:    # varint
                val, pos = _read_varint(buf, pos)
                out.setdefault(field_no, []).append((0, val))
            elif wire_type == 1:    # 64-bit fixed
                if pos + 8 > len(buf):
                    break
                out.setdefault(field_no, []).append((1, buf[pos:pos+8]))
                pos += 8
            elif wire_type == 2:    # length-prefixed
                length, pos = _read_varint(buf, pos)
                if pos + length > len(buf):
                    break
                out.setdefault(field_no, []).append(
                    (2, buf[pos:pos+length])
                )
                pos += length
            elif wire_type == 5:    # 32-bit fixed
                if pos + 4 > len(buf):
                    break
                out.setdefault(field_no, []).append((5, buf[pos:pos+4]))
                pos += 4
            else:
                # Group / unknown — bail out, what we have is enough.
                break
    except (ValueError, IndexError):
        pass
    return out


def _try_decode_telemetry(payload: bytes) -> Optional[dict]:
    """Decode a Telemetry message. Returns a flat dict of the fields
    we care about (battery_pct, voltage, channel_util, air_util_tx,
    uptime_s) or None if nothing recognizable. Only handles the
    DeviceMetrics oneof (Telemetry.field 2) — environment_metrics
    and the others are rarer on a typical mesh."""
    fields = _read_pb_fields(payload)
    # Telemetry.time = field 1 (fixed32 epoch seconds)
    # Telemetry.device_metrics = field 2 (Message)
    dm_entries = fields.get(2, [])
    if not dm_entries:
        return None
    _, dm_buf = dm_entries[0]
    dm_fields = _read_pb_fields(dm_buf)
    out: dict = {}
    # DeviceMetrics field numbers per Meshtastic protobufs:
    # 1 = battery_level (uint32, 0-101)
    # 2 = voltage (float)
    # 3 = channel_utilization (float)
    # 4 = air_util_tx (float)
    # 5 = uptime_seconds (uint32)
    for fn, key, kind in [
        (1, "battery_pct", "varint"),
        (2, "voltage_v", "f32"),
        (3, "channel_util", "f32"),
        (4, "air_util_tx", "f32"),
        (5, "uptime_s", "varint"),
    ]:
        entries = dm_fields.get(fn)
        if not entries:
            continue
        wt, raw = entries[0]
        if kind == "varint" and wt == 0:
            out[key] = raw
        elif kind == "f32" and wt == 5 and len(raw) == 4:
            val = struct.unpack("<f", raw)[0]
            # v0.7.10: range-validate. Some firmware variants emit
            # garbage at this field position (sentinel for "no
            # measurement", or a different field repurposed) which
            # would otherwise render as e.g. 16520712290304.00V.
            # Per-field plausible ranges:
            #   voltage_v:    0.1 .. 15.0 V  (any sane Meshtastic node)
            #   channel_util: 0.0 .. 100.0 % (it's a percentage)
            #   air_util_tx:  0.0 .. 100.0 % (also a percentage)
            # NaN and Inf are also rejected (math.isfinite catches both).
            import math
            if not math.isfinite(val):
                continue
            if key == "voltage_v" and not (0.1 <= val <= 15.0):
                continue
            if key in ("channel_util", "air_util_tx") and not (
                0.0 <= val <= 100.0
            ):
                continue
            out[key] = val
    return out or None


def _looks_like_user_text(b: bytes, max_len: int) -> bool:
    """v0.7.8 / v0.7.10: heuristic — does this byte sequence look like a
    legitimate Meshtastic name/id field?

    v0.7.10 tightened the check after a real-RF case where 21 bytes
    of binary data passed v0.7.8's looser "≥80% printable" rule
    because it happened to contain a run of `0x55` ('U') bytes.
    The new rules:
      • Length 1..max_len bytes
      • No NULL bytes
      • Bytes form valid UTF-8 (strict — random binary nearly never does)
      • All decoded characters are printable (excludes control chars)
      • Some character variety (≥2 distinct chars when length ≥ 4 — real
        names aren't `UUUUUUUU`)

    Real names like "TestNode", "Robin's 🌲", "Hi 👋 Mike", and "李明"
    all pass. Random 21-byte binary chunks like `\\xd1\\xe6\\xab\\xcd` +
    `\\x55` × 17 fail because they aren't valid UTF-8.
    """
    if not b or len(b) > max_len:
        return False
    if 0 in b:
        return False
    try:
        s = b.decode("utf-8")    # strict — no errors='replace'
    except UnicodeDecodeError:
        return False
    if not all(c.isprintable() for c in s):
        return False
    # Variety: real names are not just one character repeated.
    if len(s) >= 4 and len(set(s)) < 2:
        return False
    return True


def _try_decode_nodeinfo(payload: bytes) -> Optional[dict]:
    """Decode a User message (NODEINFO_APP payload). Returns a dict
    with id (string), long_name, short_name, hw_model (int),
    public_key (32 bytes hex) — all optional. None if the buffer
    doesn't look like a User message.

    v0.7.8: defensive validation against PKI-style envelopes where
    field 2 is binary instead of a long_name string.

    v0.7.9: extract the public key when present. Two paths:
      • Standard: field 8 (public_key, bytes) — newer Meshtastic
        firmware following the documented schema.
      • PKI envelope: when field 2 carries 32 bytes of high-entropy
        binary (looks like x25519 key material), treat THAT as the
        public key. This handles the non-standard envelope variant
        we documented in v0.7.8 — at least the user can now see
        which node has which key, even if the wrapping format
        isn't standard. Other 32-byte length-delim fields are
        scanned the same way.
    """
    fields = _read_pb_fields(payload)
    # User field numbers per Meshtastic protobufs:
    # 1 = id (string, e.g. "!a1b2c3d4")
    # 2 = long_name (string, 1-40 chars) OR (PKI variant) public_key bytes
    # 3 = short_name (string, 1-4 chars)
    # 4 = macaddr (deprecated bytes)
    # 5 = hw_model (HardwareModel enum, varint)
    # 8 = public_key (bytes, 32) — newer firmware, standard PKI
    out: dict = {}
    # Per-field max-length sanity bounds matching the protobuf spec.
    for fn, key, max_len in [
        (1, "id",         16),    # "!" + 8 hex chars + slack
        (2, "long_name",  40),
        (3, "short_name",  4),
    ]:
        entries = fields.get(fn)
        if not entries or entries[0][0] != 2:
            continue
        raw = entries[0][1]
        if not _looks_like_user_text(raw, max_len):
            continue
        try:
            out[key] = raw.decode("utf-8", errors="replace")
        except Exception:
            pass
    hw_entries = fields.get(5)
    if hw_entries and hw_entries[0][0] == 0:
        out["hw_model"] = hw_entries[0][1]
    # v0.7.10: PKI public-key extraction. Three strategies, in order
    # of confidence (best first):
    #
    #   (a) STANDARD: parsed protobuf field 8 with 32-byte content.
    #       This is the documented Meshtastic User.public_key location
    #       and is what newer firmware following spec emits.
    #
    #   (b) TARGETED: scan the payload for a "wire-type-2 tag + length
    #       0x20 + 32 bytes" pattern. Catches non-standard field
    #       numbers (e.g. some firmware variants put the key at field
    #       2 or 3 instead of 8) while still respecting protobuf
    #       framing — we always extract the bytes that come AFTER a
    #       valid tag+length, never bytes that include the framing.
    #       This was the v0.7.9 bug: "pki:1215d1e6…" had `12 15`
    #       framing leak into the displayed key.
    #
    #   (c) BOGUS-LENGTH ENVELOPE: the documented variant where field
    #       2 has a varint length that overflows the buffer (e.g.
    #       0x83 0x17 = 2947) followed directly by 32 key bytes.
    #       Distinct from (b) because the length isn't 0x20.
    #
    # If none of these find a key, we DO NOT fall back to a generic
    # entropy scan — that produced the framing-byte leak. A missing
    # key is better than a wrong one.
    if "public_key" not in out:
        # (a) Standard field 8.
        pk_entries = fields.get(8)
        if pk_entries and pk_entries[0][0] == 2:
            pk = pk_entries[0][1]
            if len(pk) == 32 and _looks_like_key_material(pk):
                out["public_key"] = pk.hex()
    if "public_key" not in out:
        # (b) Targeted scan for "tag + 0x20 + 32 bytes" pattern at
        # any position in the payload. The byte at off must be a
        # valid wire-type-2 tag (low 3 bits = 010 = 2). The byte at
        # off+1 must be exactly 0x20 (length 32 as single varint
        # byte). The 32 bytes at off+2 must pass key-material check.
        for off in range(len(payload) - 33):
            tag_byte = payload[off]
            if (tag_byte & 0x7) != 2:    # not wire-type 2
                continue
            if payload[off + 1] != 0x20:    # not length 32
                continue
            candidate = payload[off + 2:off + 34]
            if _looks_like_key_material(candidate):
                out["public_key"] = candidate.hex()
                break
    if "public_key" not in out and len(payload) >= 32:
        # (c) Bogus-length envelope: field-2 tag + varint length that
        # overflows the buffer. Documented in v0.7.8 forensics.
        scan_start = 0
        id_entries = fields.get(1)
        if id_entries and id_entries[0][0] == 2:
            scan_start = 2 + len(id_entries[0][1])
        if (scan_start < len(payload) - 32
                and payload[scan_start] == 0x12):
            try:
                claimed_len, hdr_end = _read_varint(payload, scan_start + 1)
                if (claimed_len > len(payload) - hdr_end
                        and hdr_end + 32 <= len(payload)):
                    candidate = payload[hdr_end:hdr_end + 32]
                    if _looks_like_key_material(candidate):
                        out["public_key"] = candidate.hex()
            except Exception:
                pass
    return out or None


def _entropy_bits_per_byte(b: bytes) -> float:
    """Shannon entropy of a byte sequence in bits/byte. Used for
    public-key candidate scoring."""
    import math
    if not b:
        return 0.0
    counts = [0] * 256
    for c in b:
        counts[c] += 1
    total = len(b)
    return -sum((c / total) * math.log2(c / total)
                for c in counts if c > 0)


def _looks_like_key_material(b: bytes) -> bool:
    """v0.7.9: heuristic for x25519 / Ed25519 public-key bytes.

    A 32-byte Curve25519 public key is uniformly distributed pseudo-
    random bytes: high entropy, no structure, no NULL runs. This
    heuristic rejects:
      • Anything that's not exactly 32 bytes
      • Anything with > 4 NULL bytes (legitimate keys don't have
        these — Curve25519 clears specific bits but doesn't zero
        whole bytes)
      • Anything whose entropy is below ~4 bits/byte (text and
        structured data have far less variation)

    Real x25519 keys typically score 4.7-5.0 bits/byte on a 32-byte
    sample. Random ASCII text scores 4.1-4.3. Repeating patterns
    score below 3. The 4.0 threshold catches keys reliably without
    false-positive on names that happen to be exactly 32 chars long.
    """
    if len(b) != 32:
        return False
    if b.count(0) > 4:
        return False
    return _entropy_bits_per_byte(b) >= 4.0


# Subset of the most commonly-seen Meshtastic HardwareModel enum
# values. Full list lives in upstream proto/mesh.proto; we keep the
# "what hardware are people running" picks here. Anything not in the
# table renders as "hw=N" so users can look it up themselves.
_HW_MODELS = {
    1: "TLORA_V2", 2: "TLORA_V1", 3: "TLORA_V2_1_1P6", 4: "TBEAM",
    5: "HELTEC_V2_0", 6: "TBEAM_V0P7", 7: "T_ECHO", 8: "TLORA_V1_1P3",
    9: "RAK4631", 10: "HELTEC_V2_1", 11: "HELTEC_V1", 12: "LILYGO_TBEAM_S3_CORE",
    13: "RAK11200", 14: "NANO_G1", 15: "TLORA_V2_1_1P8",
    16: "TLORA_T3_S3", 17: "NANO_G1_EXPLORER", 18: "NANO_G2_ULTRA",
    19: "LORA_TYPE", 25: "STATION_G1", 32: "RAK11310",
    39: "HELTEC_WIRELESS_PAPER", 40: "T_DECK", 41: "T_WATCH_S3",
    42: "PICOMPUTER_S3", 43: "HELTEC_HT62", 50: "DIY_V1",
    52: "STATION_G2", 70: "HELTEC_WIRELESS_TRACKER",
}


# ─────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DecodedRecord:
    sample_offset: int
    payload_len: int
    cr: int
    crc_ok: bool
    cfo_hz: float
    preset: str
    freq_hz: int
    # v0.7.7: signal quality from the C decoder. RSSI in dBFS
    # (negative for sub-full-scale signals; typical Meshtastic
    # range -1 to -25). SNR in dB above noise floor (typical
    # +5 to +25 for valid packets). Both populated from the
    # preamble dechirp at SYNC_NETID transition. Previously
    # always 0.0 because the C side never wrote them.
    rssi_db: float = 0.0
    snr_db: float = 0.0
    src: Optional[int] = None
    dst: Optional[int] = None
    pkt_id: Optional[int] = None
    hop_limit: Optional[int] = None
    hop_start: Optional[int] = None
    channel_hash: Optional[int] = None
    decrypted: bool = False
    channel_name: Optional[str] = None
    portnum: Optional[int] = None
    portnum_label: Optional[str] = None
    text: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    # v0.7.6: structured payload for TELEMETRY_APP and NODEINFO_APP.
    # telemetry is a flat dict keyed by battery_pct/voltage_v/
    # channel_util/air_util_tx/uptime_s; nodeinfo carries id +
    # long_name + short_name + hw_model. Either may be None.
    telemetry: Optional[dict] = None
    nodeinfo: Optional[dict] = None


def _to_record(pp: PipelinePacket, channel_names: list[str]) -> DecodedRecord:
    rec = DecodedRecord(
        sample_offset=pp.lora.sample_offset,
        payload_len=pp.lora.payload_len,
        cr=pp.lora.cr,
        crc_ok=pp.lora.crc_ok,
        cfo_hz=pp.lora.cfo_hz,
        preset=pp.slot.preset.key,
        freq_hz=pp.slot.freq_hz,
        # v0.7.7: signal quality
        rssi_db=pp.lora.rssi_db,
        snr_db=pp.lora.snr_db,
    )
    if pp.mesh is None:
        return rec
    m = pp.mesh
    rec.src = m.from_node
    rec.dst = m.to
    rec.pkt_id = m.id
    rec.hop_limit = m.hop_limit
    rec.hop_start = m.hop_start
    rec.channel_hash = m.channel_hash
    if m.decrypted:
        rec.decrypted = True
        rec.channel_name = channel_names[m.channel_index]
        env = _parse_data_envelope(m.plaintext)
        if env:
            portnum, payload = env
            rec.portnum = portnum
            rec.portnum_label = _PORTNUMS.get(portnum, f"port_{portnum}")
            if portnum == 1:
                try:
                    rec.text = payload.decode("utf-8", errors="replace")
                except Exception:
                    pass
            elif portnum == 3:
                pos = _try_decode_position(payload)
                if pos:
                    rec.lat, rec.lon = pos
            elif portnum == 4:
                # NODEINFO_APP — User message
                rec.nodeinfo = _try_decode_nodeinfo(payload)
            elif portnum == 67:
                # TELEMETRY_APP — Telemetry message
                rec.telemetry = _try_decode_telemetry(payload)
    return rec


def _print_human(rec: DecodedRecord) -> None:
    """One-line stdout summary."""
    crc_tag = "✓" if rec.crc_ok else "✗"
    # v0.7.9: include slot center frequency next to preset name so
    # the user can tell which decoder caught what under --slots all.
    # Format: "mediumfast@913.125" (preset@MHz, 3-decimal precision
    # to distinguish adjacent slots which are typically 250 kHz
    # apart at the high end of US 915 band). Compact enough to fit
    # in the existing column without expanding the line.
    preset_name = rec.preset.replace("_", "")[:10].lower()
    preset = f"{preset_name}@{rec.freq_hz/1e6:.3f}"
    sigq = f"rssi={rec.rssi_db:+5.1f}dB snr={rec.snr_db:+4.1f}dB"
    if not rec.crc_ok:
        # v0.7.8: classify the CRC failure by SNR so the user can
        # tell what's happening on the air. A weak signal (SNR < 5)
        # failing CRC is "edge of range" — expected behavior, the
        # packet was barely above noise. A strong signal (SNR ≥ 15)
        # failing CRC is "interference" — the signal was clean but
        # something corrupted the bytes, which usually means a
        # colliding transmission overlapping the same airtime, or
        # an in-band burst (radar pulse, garage door opener, etc).
        # Between those, behavior is ambiguous so we don't
        # speculate.
        if rec.snr_db < 5.0:
            reason = "weak signal"
        elif rec.snr_db >= 15.0:
            reason = "interference / collision"
        else:
            reason = ""
        suffix = f"(CRC fail — {reason})" if reason else "(CRC fail)"
        print(f"@{rec.sample_offset:>10}  {preset:<20}  len={rec.payload_len:3d}  "
              f"crc={crc_tag}  {sigq}  cfo={rec.cfo_hz:+5.0f}Hz  {suffix}")
        return
    src = f"0x{rec.src:08X}" if rec.src is not None else "?"
    dst_marker = "→bcast" if rec.dst == 0xFFFFFFFF else f"→0x{rec.dst:08X}"
    hop = f"{rec.hop_limit}/{rec.hop_start}" if rec.hop_limit is not None else "?/?"
    crypt = (f"  ch={rec.channel_name!r}({rec.portnum_label})"
             if rec.decrypted
             else f"  hash=0x{rec.channel_hash:02X} (no PSK)")
    extra = ""
    if rec.text is not None:
        # v0.7.15: emit the full text body. Earlier versions truncated
        # to 48 chars with "..." which made longer messages (e.g. forum
        # debate threads on the air) unreadable in the log. Python's
        # repr() will already escape control chars and limit line
        # damage from CR/LF in the body.
        extra = f"  text={rec.text!r}"
    elif rec.lat is not None and rec.lon is not None:
        extra = f"  pos=({rec.lat:.4f},{rec.lon:.4f})"
    elif rec.telemetry:
        # v0.7.6: render the device-metrics fields users actually
        # care about. Dropping any field that wasn't in the payload
        # so we don't render zeros for things the device didn't send.
        bits = []
        t = rec.telemetry
        if "battery_pct" in t:
            # 101 means "plugged in / not on battery" in Meshtastic
            if t["battery_pct"] == 101:
                bits.append("plugged-in")
            else:
                bits.append(f"bat={t['battery_pct']}%")
        if "voltage_v" in t:
            bits.append(f"{t['voltage_v']:.2f}V")
        if "channel_util" in t:
            bits.append(f"chutil={t['channel_util']:.1f}%")
        if "air_util_tx" in t:
            bits.append(f"airtx={t['air_util_tx']:.2f}%")
        if "uptime_s" in t:
            up = t["uptime_s"]
            if up >= 86400:
                bits.append(f"up={up // 86400}d{(up % 86400) // 3600}h")
            elif up >= 3600:
                bits.append(f"up={up // 3600}h{(up % 3600) // 60}m")
            else:
                bits.append(f"up={up // 60}m{up % 60}s")
        if bits:
            extra = "  " + " ".join(bits)
    elif rec.nodeinfo:
        # v0.7.6: render the human-meaningful identification fields.
        bits = []
        n = rec.nodeinfo
        if "long_name" in n and "short_name" in n:
            bits.append(f"'{n['long_name']}' [{n['short_name']}]")
        elif "long_name" in n:
            bits.append(f"'{n['long_name']}'")
        elif "short_name" in n:
            bits.append(f"[{n['short_name']}]")
        if "id" in n:
            bits.append(n["id"])
        if "hw_model" in n:
            hw = _HW_MODELS.get(n["hw_model"], f"hw={n['hw_model']}")
            bits.append(hw)
        # v0.7.10: full 64-char public key in the per-packet line.
        # Previously we abbreviated as "pki:1234abcd…ef89" but that
        # made independent verification harder — the user wants to
        # be able to copy the whole key for cross-referencing
        # against their own node directory or to use as a unique
        # node identity. The line gets longer but the value is real.
        if "public_key" in n:
            bits.append(f"pki:{n['public_key']}")
        if bits:
            extra = "  " + " ".join(bits)
    print(f"@{rec.sample_offset:>10}  {preset:<20}  len={rec.payload_len:3d}  "
          f"crc={crc_tag}  {sigq}  cfo={rec.cfo_hz:+5.0f}Hz  "
          f"{src} {dst_marker} hop={hop}{crypt}{extra}")


# ─────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────

def _parse_psk_arg(arg: str) -> tuple[str, bytes]:
    if ":" not in arg:
        raise argparse.ArgumentTypeError(
            f"--psk value {arg!r} must be NAME:HEX (e.g. 'MyChan:aabbcc...')"
        )
    name, hex_str = arg.split(":", 1)
    try:
        psk = bytes.fromhex(hex_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--psk hex for {name!r} not parseable: {e}"
        )
    if len(psk) not in (1, 16, 32):
        raise argparse.ArgumentTypeError(
            f"--psk for {name!r}: expected 1 / 16 / 32 bytes, got {len(psk)}"
        )
    return name, psk


def _parse_preset_arg(arg: str):
    arg = arg.strip()
    if arg in ("auto", "all"):
        return arg
    presets = [p.strip().upper() for p in arg.split(",")]
    for p in presets:
        if p not in PRESETS:
            raise argparse.ArgumentTypeError(
                f"unknown preset {p!r}; valid: {', '.join(PRESETS)}"
            )
    return presets


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rfcensus.tools.decode_meshtastic",
        description=(
            "Decode + decrypt Meshtastic LoRa packets from a cu8 file, "
            "live RTL-SDR, or rtl_tcp server. By default decodes every "
            "Meshtastic preset whose frequency slot falls within the "
            "dongle's passband, in parallel."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("capture", type=Path, nargs="?", default=None,
        help="Path to cu8 IQ file (file mode)")
    src.add_argument("--rtl-sdr", action="store_true",
        help="Use rtl_sdr subprocess (live mode, exclusive dongle)")
    src.add_argument("--rtl-tcp", metavar="HOST:PORT",
        help="Connect to rtl_tcp server (live mode, can share dongle)")

    p.add_argument("--device", default="0",
        help="Dongle index (0,1,...) or serial number (live modes)")
    p.add_argument("--gain", type=float, default=-1,
        help="Tuner gain in dB (-1 = AGC). Try 30-40 for site survey.")
    p.add_argument("--ppm", type=int, default=0,
        help="Frequency correction in PPM")

    p.add_argument("--frequency", type=int,
        help="Tuner center frequency in Hz")
    p.add_argument("--sample-rate", type=int,
        help="IQ sample rate in Hz (default: 2400000 live, 1000000 file)")
    p.add_argument("--region", default="US", choices=list(REGIONS),
        help="Meshtastic region for slot computation")

    p.add_argument("--preset", type=_parse_preset_arg, default="auto",
        metavar="auto|all|CSV",
        help="Which Meshtastic presets to decode. 'auto' = presets "
             "whose default channel hash falls in the passband; 'all' "
             "= every preset; or comma-separated list like "
             "'LONG_FAST,MEDIUM_FAST'")
    p.add_argument("--slots", choices=["default", "all"], default="default",
        help="'default' = only each preset's default-channel slot "
             "(catches public Meshtastic traffic only, ~9 decoders). "
             "'all' = every (preset, frequency-slot) pair in the "
             "passband (catches custom-named channels too — up to ~80 "
             "decoders for US at 2.4 MS/s). With --slots all the "
             "lazy pipeline (--lazy) is enabled by default; pass "
             "--no-lazy to force eager spawning.")
    p.add_argument("--lazy", dest="lazy", action="store_true", default=None,
        help="Use the coarse-FFT lazy pipeline: a wide-FFT detector "
             "watches the whole passband and spawns LoRa decoders on "
             "demand for slots showing energy. Cheap baseline (~10%% "
             "of one core) regardless of slot count, vs. eager spawn "
             "which scales linearly with slots and quickly hits CPU "
             "limits at --slots all. Default: enabled when --slots "
             "all; disabled for --slots default (where the small "
             "decoder count makes eager faster).")
    p.add_argument("--no-lazy", dest="lazy", action="store_false",
        help="Force eager pipeline (one decoder per (preset, slot) "
             "kept always-on). Useful for benchmarks or low-slot-count "
             "configs where the lazy detector overhead exceeds "
             "savings.")
    p.add_argument("--hop", action="store_true",
        help="Cycle dongle tuning to cover every preset slot in the "
             "region. Use for one-dongle full-region coverage.")
    p.add_argument("--hop-dwell", type=float, default=30.0,
        help="Seconds to dwell on each tuning before hopping")

    p.add_argument("--no-default", action="store_true",
        help="Don't auto-add the public default channel PSK")
    p.add_argument("--default-channel-preset", default=None,
        help="Which preset's name to use for the default channel hash "
             "(default: pick first preset in slots list)")
    p.add_argument("--psk", action="append", default=[], type=_parse_psk_arg,
        metavar="NAME:HEX",
        help="Add a custom channel PSK (repeatable)")

    p.add_argument("--pcap", type=Path,
        help="Write Wireshark PCAP with DLT_LORATAP framing")
    p.add_argument("--jsonl", type=Path,
        help="Write per-packet JSON Lines log")
    p.add_argument("--quiet", action="store_true",
        help="Suppress per-packet stdout")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="IQ chunk size in bytes")
    p.add_argument("--reader-buffer-secs", type=float,
        default=DEFAULT_RING_CAPACITY_SECS,
        help=f"v0.7.16: decouple-ring capacity in seconds for live "
             f"modes (default {DEFAULT_RING_CAPACITY_SECS}s = ~9.6 MB at "
             f"2.4 MS/s). A reader thread fills this ring from rtl_tcp "
             f"or rtl_sdr while the decode pipeline drains it; absorbs "
             f"GC pauses + transient processing stalls without dropping "
             f"samples. Set to 0 to disable (revert to v0.7.15 single-"
             f"threaded behavior — only useful for debugging).")
    p.add_argument("--max-runtime", type=float,
        help="Stop after this many seconds (live modes only)")

    p.add_argument("--list-slots", action="store_true",
        help="Print every preset's frequency slot in the configured "
             "region (or in the configured passband if --frequency + "
             "--sample-rate given) and exit")
    p.add_argument("--list-presets", action="store_true",
        help="Print every preset's parameters and exit")
    return p


def _list_presets(region_code: str) -> None:
    print(f"# Meshtastic presets in region {region_code}:")
    print(f"  {'preset':<14}  {'BW':<6}  {'SF':<3}  {'CR':<4}  "
          f"{'slot':<10}  {'freq (MHz)':<10}")
    print("  " + "-" * 60)
    for slot in all_default_slots(region_code):
        p = slot.preset
        print(f"  {p.key:<14}  {p.bandwidth_hz//1000:>3}kHz  "
              f"{p.sf:<3}  4/{p.cr:<2}  {slot.slot:>4}/{slot.num_slots:<5}  "
              f"{slot.freq_hz/1e6:>8.3f}")


def _list_slots(region_code: str, center_hz: Optional[int],
                 sample_rate_hz: Optional[int]) -> None:
    if center_hz is None or sample_rate_hz is None:
        print("# (no --frequency / --sample-rate given; showing ALL slots)")
        _list_presets(region_code)
        return
    slots = slots_in_passband(region_code, center_hz, sample_rate_hz)
    print(f"# Passband: {center_hz/1e6:.3f} MHz ± "
          f"{sample_rate_hz/2/1e6:.3f} MHz @ {sample_rate_hz/1e6:.3f} MS/s "
          f"({region_code})")
    if not slots:
        print(f"#   no Meshtastic preset slots fit in this passband")
        print(f"#   try --list-presets to see all slots in {region_code}")
        return
    print(f"#   {len(slots)} slot(s) in passband:")
    for s in slots:
        offset = s.freq_hz - center_hz
        print(f"     {s.preset.key:<14}  {s.freq_hz/1e6:>8.3f} MHz  "
              f"({offset:+d} Hz from center)  "
              f"BW={s.preset.bandwidth_hz//1000}kHz SF{s.preset.sf}")


# ─────────────────────────────────────────────────────────────────────
# Source construction
# ─────────────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> int:
    if device.isdigit():
        return int(device)
    idx = find_device_index_by_serial(device)
    if idx is None:
        raise SystemExit(
            f"error: device {device!r} not found. Use index (0,1,...) "
            f"or serial. Run `rtl_test -t` to enumerate dongles."
        )
    return idx


def _build_source(args: argparse.Namespace) -> tuple[IQSource, int, int]:
    """Returns (source, center_hz, sample_rate_hz).

    v0.7.16: live sources (RtlSdrSubprocess, RtlTcpSource) are wrapped
    in an IqReader by default — a reader thread fills a 2-second ring
    buffer while the decode pipeline drains it. Absorbs GC pauses and
    transient processing stalls without dropping samples. File mode
    (FileIQSource) is NOT wrapped: the file is already faster than
    real-time, so the threading layer would only add overhead with no
    benefit. Set --reader-buffer-secs 0 to disable wrapping for live
    modes (debugging only).
    """
    if args.capture:
        if not args.capture.exists():
            raise SystemExit(f"error: {args.capture} not found")
        if args.frequency is None:
            raise SystemExit(
                "error: --frequency required for file mode (cu8 has "
                "no embedded tuner metadata)"
            )
        sample_rate = args.sample_rate or 1_000_000
        return (FileIQSource(args.capture, chunk_size=args.chunk_size),
                args.frequency, sample_rate)

    if args.frequency is None:
        raise SystemExit("error: --frequency is required for live modes")
    sample_rate = args.sample_rate or 2_400_000
    cfg = RtlSdrConfig(
        freq_hz=args.frequency,
        sample_rate_hz=sample_rate,
        device_index=_resolve_device(args.device),
        gain_tenths_db=int(args.gain * 10) if args.gain >= 0 else -1,
        ppm=args.ppm,
    )
    if args.rtl_tcp:
        host, _, port_str = args.rtl_tcp.partition(":")
        port = int(port_str) if port_str else 1234
        raw = RtlTcpSource(host, port, cfg, chunk_size=args.chunk_size)
    else:
        raw = RtlSdrSubprocess(cfg, chunk_size=args.chunk_size)

    # v0.7.16: wrap with IqReader unless explicitly disabled.
    if args.reader_buffer_secs > 0:
        reader = IqReader(
            raw,
            sample_rate_hz=sample_rate,
            ring_capacity_secs=args.reader_buffer_secs,
            chunk_size=args.chunk_size,
        )
        reader.start()
        return (reader, args.frequency, sample_rate)
    return (raw, args.frequency, sample_rate)


# ─────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list_presets:
        _list_presets(args.region)
        return 0
    if args.list_slots:
        _list_slots(args.region, args.frequency, args.sample_rate)
        return 0
    if args.hop and args.capture:
        print("error: --hop only makes sense for live modes "
              "(--rtl-sdr or --rtl-tcp), not file mode",
              file=sys.stderr)
        return 2

    # ── Resolve preset list / hop plan ────────────────────────────
    hop_plan: list = []   # populated when --hop is set
    slots: list = []      # populated when --hop is NOT set
    source = None
    center_hz = args.frequency
    sample_rate_hz = args.sample_rate or (
        1_000_000 if args.capture else 2_400_000
    )

    if args.hop:
        # Decide which presets to cover, then compute the hop plan,
        # then construct the source pointed at the first tuning.
        if args.preset == "auto" or args.preset == "all":
            target_presets = list(PRESETS.keys())
        else:
            target_presets = list(args.preset)

        from rfcensus.utils.meshtastic_hop import plan_hop
        hop_plan = plan_hop(args.region, sample_rate_hz,
                             presets=target_presets)
        if not hop_plan:
            print(f"error: could not build a hop plan covering "
                  f"{target_presets}. Sample rate may be too low.",
                  file=sys.stderr)
            return 2

        # Override args.frequency with the first tuning's center so
        # _build_source picks it up. (The center will get overridden
        # again on each subsequent hop via source.retune().)
        args.frequency = hop_plan[0].center_freq_hz
        source, center_hz, sample_rate_hz = _build_source(args)

        if not hasattr(source, "retune"):
            print(f"error: source type {type(source).__name__} does not "
                  f"support retune; --hop requires --rtl-sdr or "
                  f"--rtl-tcp", file=sys.stderr)
            return 2

    else:
        # Non-hop: build source first, then resolve preset slots.
        source, center_hz, sample_rate_hz = _build_source(args)

        # Decide which preset KEYS to monitor.
        if args.preset == "auto" or args.preset == "all":
            target_presets = list(PRESETS.keys())
        else:
            target_presets = list(args.preset)

        # Decide whether to monitor only DEFAULT slots or ALL slots.
        if args.slots == "all":
            slots = enumerate_all_slots_in_passband(
                args.region, center_hz, sample_rate_hz,
                presets=target_presets,
            )
            if not slots:
                print(f"error: no slots fit in passband "
                      f"({center_hz/1e6:.3f} MHz ± "
                      f"{sample_rate_hz/2/1e6:.3f} MHz)",
                      file=sys.stderr)
                return 2
        elif args.preset not in ("auto", "all"):
            # Explicit preset list — use each preset's default slot.
            slots = [default_slot(args.region, k) for k in target_presets]
        else:
            # auto / all with --slots default
            in_band = slots_in_passband(args.region, center_hz,
                                          sample_rate_hz,
                                          presets=target_presets)
            if args.preset == "all":
                all_slots = all_default_slots(args.region)
                missed = [s.preset.key for s in all_slots
                          if s.freq_hz not in {x.freq_hz for x in in_band}
                          and s.preset.key in target_presets]
                if missed:
                    print(f"warning: --preset all without --hop misses "
                          f"{len(missed)} preset(s) outside the dongle's "
                          f"passband: {', '.join(missed)}.\n"
                          f"Add --hop to cycle through tunings, or run "
                          f"another instance per missed preset.",
                          file=sys.stderr)
            slots = in_band
            if not slots:
                print(f"error: no Meshtastic preset DEFAULT slots fit "
                      f"in passband ({center_hz/1e6:.3f} MHz ± "
                      f"{sample_rate_hz/2/1e6:.3f} MHz). "
                      f"Try --slots all to scan every (preset, slot) "
                      f"pair, or --hop to cycle tunings.",
                      file=sys.stderr)
                return 2

    # ── Build channel table ───────────────────────────────────────
    # The "default channel" for any Meshtastic preset uses the preset's
    # display name (e.g. "MediumFast") as the channel name, which feeds
    # into the DJB2 hash → channel slot. Each preset therefore has its
    # OWN default channel hash. When monitoring multiple presets, we
    # need to add one default channel per preset so each preset's
    # default-PSK traffic decrypts.
    #
    # In hop mode, we add channels for EVERY preset that any tuning
    # in the hop plan will visit, not just the current tuning's slots.
    # That way the channel table doesn't need rebuilding per hop.
    if args.hop:
        all_visited_presets: set[str] = set()
        for tuning in hop_plan:
            for s in tuning.slots:
                all_visited_presets.add(s.preset.key)
        default_preset_keys = sorted(all_visited_presets)
    else:
        default_preset_keys = [s.preset.key for s in slots]

    if args.default_channel_preset:
        if args.default_channel_preset not in MESH_PRESETS:
            print(f"error: --default-channel-preset "
                  f"{args.default_channel_preset!r} not recognized by "
                  f"libmeshtastic", file=sys.stderr)
            return 2
        init_preset = args.default_channel_preset
        default_presets_to_add = [init_preset] if not args.no_default else []
    else:
        init_preset = (default_preset_keys[0]
                        if default_preset_keys else "LONG_FAST")
        default_presets_to_add = (
            list(default_preset_keys) if not args.no_default else []
        )

    mesh = MeshtasticDecoder(init_preset)
    channel_names: list[str] = []

    # PSK = b"\x01" is libmeshtastic's short-index for the default PSK
    # (MESH_DEFAULT_PSK), which is what every default-channel
    # Meshtastic device uses.
    for preset_key in default_presets_to_add:
        display_name = PRESETS[preset_key].display_name
        if display_name in channel_names:
            continue
        mesh.add_channel(display_name, psk=b"\x01", is_primary=False)
        channel_names.append(display_name)

    for name, psk in args.psk:
        mesh.add_channel(name, psk=psk, is_primary=False)
        channel_names.append(name)

    if not args.quiet:
        print(f"# region={args.region}  rate={sample_rate_hz/1e6:.3f}MS/s",
              file=sys.stderr)
        if args.hop:
            print(f"# hop plan ({len(hop_plan)} tuning(s), "
                  f"dwell {args.hop_dwell:.0f}s each):", file=sys.stderr)
            for i, t in enumerate(hop_plan):
                ks = ",".join(s.preset.key for s in t.slots)
                print(f"#   [{i}] {t.center_freq_hz/1e6:>8.3f}MHz  →  "
                      f"{ks}", file=sys.stderr)
        else:
            print(f"# center={center_hz/1e6:.3f}MHz", file=sys.stderr)
            print(f"# decoders ({len(slots)} preset slot(s) in passband):",
                  file=sys.stderr)
            for s in slots:
                mix_freq = center_hz - s.freq_hz
                print(f"#   {s.preset.key:<14}  {s.freq_hz/1e6:>8.3f}MHz  "
                      f"mix={mix_freq:+d}Hz  "
                      f"BW={s.preset.bandwidth_hz//1000}kHz "
                      f"SF{s.preset.sf}", file=sys.stderr)
        print(f"# channels ({len(channel_names)}):", file=sys.stderr)
        for i, name in enumerate(channel_names):
            print(f"#   [{i}] {name!r}  hash=0x{mesh.channel_hash(i):02X}",
                  file=sys.stderr)

    pcap_writer: Optional[PcapLoraTapWriter] = None
    jsonl_fp = None
    t0 = time.time()
    n_total = n_decrypted = 0
    per_preset_decrypt: dict[str, int] = {}
    # v0.7.8: per-channel RSSI samples for distribution summary.
    # Keyed by channel name (decrypted) or "unknown(0xNN)" (un-PSK).
    per_channel_rssi: dict[str, list[float]] = {}
    # v0.7.9: per-slot hit counts. Keyed by (preset_key, freq_hz)
    # tuple. Lets the user see WHICH slot under --slots all caught
    # what — diagnoses whether a particular slot frequency is
    # contributing or dead weight.
    per_slot_hits: dict[tuple[str, int], int] = {}
    # v0.7.10: per-channel-per-slot RSSI samples. Lets the summary
    # break down "MediumFast" into its constituent slots so the user
    # can see "the real signal is at 913.125, the @914.375 hits are
    # bleed-through". Keyed by (channel_name, freq_hz).
    per_channel_slot_rssi: dict[tuple[str, int], list[float]] = {}
    # v0.7.10: rolling window of recent CRC events for adjacent-slot
    # bleed-through detection. Each entry: (sample_offset, freq_hz,
    # crc_ok). Window kept small (last ~200 events) to bound memory
    # on long runs. Used to identify CRC-fails that are likely
    # caused by single-transmitter adjacent-slot leak rather than
    # genuine collision/interference.
    from collections import deque
    crc_event_window: deque = deque(maxlen=200)
    # Aggregate counter: per (fail_freq, neighbor_freq), how many
    # CRC fails on fail_freq were within `bleed_window_samples` of
    # a CRC-pass at neighbor_freq. Surfaced in summary.
    bleed_pairs: dict[tuple[int, int], int] = {}
    # 50ms at sample_rate_hz — typical max LoRa packet duration is
    # ~280ms but the adjacent-slot lock fires within microseconds
    # of the primary lock since it's the same airwave. 50ms is
    # generous enough to catch the same-packet case while small
    # enough to not falsely link unrelated transmissions.
    bleed_window_samples = sample_rate_hz // 20
    final_stats: dict = {}
    # v0.7.7: lazy-pipeline performance counters surfaced in summary
    # so user sees CPU saturation / dropped samples / racing wins.
    # Eager pipeline doesn't fill this; only lazy.
    lazy_perf_stats: dict = {}
    interrupted = False

    # Decide: lazy or eager pipeline?
    #   • If user explicitly passed --lazy or --no-lazy, honor that.
    #   • Else default: lazy when --slots all (large decoder count
    #     would dominate CPU); eager when --slots default (small
    #     decoder count, eager has lower per-packet latency).
    if args.lazy is None:
        use_lazy = (args.slots == "all")
    else:
        use_lazy = args.lazy

    if not args.quiet:
        print(f"# pipeline:    {'lazy (coarse-FFT)' if use_lazy else 'eager'}",
              file=sys.stderr)

    def _build_pipeline(slots_for_pipe: list, center: int):
        """Construct either MultiPresetPipeline (eager) or
        LazyMultiPresetPipeline (lazy) based on the user's choice.
        Both expose ``feed_cu8`` / ``pop_packets`` / ``stats()`` /
        ``close`` so the rest of the CLI is pipeline-agnostic."""
        if use_lazy:
            return LazyMultiPresetPipeline(
                sample_rate_hz=sample_rate_hz,
                center_freq_hz=center,
                candidate_slots=slots_for_pipe,
                mesh=mesh,
            )
        return MultiPresetPipeline(
            slots=slots_for_pipe,
            sample_rate_hz=sample_rate_hz,
            center_freq_hz=center,
            mesh=mesh,
        )

    def _pump_one_pipeline(
        pipe: "MultiPresetPipeline | LazyMultiPresetPipeline",
        src: IQSource,
        until_ts: float,
        capture_t0: float,
    ) -> None:
        """Pump IQ from ``src`` through ``pipe`` until ``until_ts``.

        Mutates the closure-captured counters (n_total, n_decrypted,
        per_preset_decrypt) and writes to pcap/jsonl as packets emerge.
        Updates ``final_stats`` from this pipeline before returning.
        """
        nonlocal n_total, n_decrypted, final_stats
        try:
            for chunk in src:
                pipe.feed_cu8(chunk)
                for pp in pipe.pop_packets():
                    n_total += 1
                    if pp.decrypted:
                        n_decrypted += 1
                        per_preset_decrypt[pp.slot.preset.key] = \
                            per_preset_decrypt.get(pp.slot.preset.key, 0) + 1
                    # v0.7.8: track RSSI per channel so the summary
                    # can show signal-strength distribution. Bucket
                    # by the channel name when decrypted (= which
                    # preset's default PSK matched), or by the raw
                    # channel hash when undecrypted (so user sees
                    # signal levels even from unknown-PSK channels).
                    if pp.crc_ok:
                        rec_for_bucket = (pp.mesh.channel_index
                                          if pp.mesh and pp.mesh.decrypted
                                          else None)
                        bucket_key = (
                            channel_names[rec_for_bucket]
                            if rec_for_bucket is not None
                            else f"unknown(0x{pp.mesh.channel_hash:02X})"
                            if pp.mesh
                            else f"preset:{pp.slot.preset.key}"
                        )
                        per_channel_rssi.setdefault(
                            bucket_key, []
                        ).append(pp.lora.rssi_db)
                        # v0.7.9: per-slot decoder hit count.
                        slot_key = (pp.slot.preset.key, pp.slot.freq_hz)
                        per_slot_hits[slot_key] = (
                            per_slot_hits.get(slot_key, 0) + 1
                        )
                        # v0.7.10: per-channel-per-slot RSSI for the
                        # nested breakdown.
                        per_channel_slot_rssi.setdefault(
                            (bucket_key, pp.slot.freq_hz), []
                        ).append(pp.lora.rssi_db)
                    # v0.7.10: adjacent-slot bleed-through detection.
                    # When a CRC-fail arrives at slot F, scan the
                    # rolling window for a recent CRC-pass at slot
                    # G ≠ F within `bleed_window_samples`. When a
                    # CRC-PASS arrives, also scan back for recent
                    # CRC-fails at other slots — those failed
                    # decodes were almost certainly OUR signal
                    # leaking into the wrong bandpass. We need both
                    # directions because temporal order isn't fixed:
                    # the adjacent-slot decoder may catch the
                    # preamble before OR after the on-frequency
                    # decoder, depending on noise + which decoder's
                    # state machine fires first.
                    my_off = pp.lora.sample_offset
                    if not pp.crc_ok:
                        for past_off, past_freq, past_ok in crc_event_window:
                            if not past_ok:
                                continue
                            if past_freq == pp.slot.freq_hz:
                                continue
                            if abs(my_off - past_off) <= bleed_window_samples:
                                key = (pp.slot.freq_hz, past_freq)
                                bleed_pairs[key] = (
                                    bleed_pairs.get(key, 0) + 1
                                )
                                break
                    else:    # CRC pass: relabel earlier nearby fails
                        for past_off, past_freq, past_ok in crc_event_window:
                            if past_ok:
                                continue
                            if past_freq == pp.slot.freq_hz:
                                continue
                            if abs(my_off - past_off) <= bleed_window_samples:
                                key = (past_freq, pp.slot.freq_hz)
                                bleed_pairs[key] = (
                                    bleed_pairs.get(key, 0) + 1
                                )
                                # don't break — multiple slots may
                                # all have failed on the same packet
                    # Always record the event for FUTURE lookups.
                    crc_event_window.append((
                        pp.lora.sample_offset,
                        pp.slot.freq_hz,
                        pp.crc_ok,
                    ))
                    rec = _to_record(pp, channel_names)
                    if not args.quiet:
                        _print_human(rec)
                    if jsonl_fp:
                        jsonl_fp.write(json.dumps(asdict(rec)) + "\n")
                        jsonl_fp.flush()
                    if pcap_writer and pp.crc_ok:
                        ts = (capture_t0 +
                                 pp.lora.sample_offset / sample_rate_hz
                              if args.capture else time.time())
                        pcap_writer.write_packet(
                            pp.lora.payload,
                            frequency_hz=pp.slot.freq_hz,
                            bandwidth_hz=pp.slot.preset.bandwidth_hz,
                            sf=pp.slot.preset.sf,
                            sync_word=0x2B,
                            rssi_dbm=pp.lora.rssi_db,
                            snr_db=pp.lora.snr_db,
                            timestamp=ts,
                        )
                if time.time() >= until_ts:
                    return
        finally:
            # Merge stats — each pipeline reports per-preset counters
            # for THIS dwell only; we accumulate them across hops by
            # adding to a running dict.
            for key, s in pipe.stats().items():
                if key in final_stats:
                    prev = final_stats[key]
                    # Sum the integer fields; keep the latest float.
                    final_stats[key] = type(s)(
                        samples_processed=prev.samples_processed
                            + s.samples_processed,
                        preambles_found=prev.preambles_found
                            + s.preambles_found,
                        syncwords_matched=prev.syncwords_matched
                            + s.syncwords_matched,
                        headers_decoded=prev.headers_decoded
                            + s.headers_decoded,
                        headers_failed=prev.headers_failed
                            + s.headers_failed,
                        packets_decoded=prev.packets_decoded
                            + s.packets_decoded,
                        packets_crc_failed=prev.packets_crc_failed
                            + s.packets_crc_failed,
                        detect_attempts=prev.detect_attempts
                            + s.detect_attempts,
                        detect_above_gate=prev.detect_above_gate
                            + s.detect_above_gate,
                        detect_max_run=max(prev.detect_max_run,
                                            s.detect_max_run),
                        detect_peak_mag_max=max(prev.detect_peak_mag_max,
                                                  s.detect_peak_mag_max),
                    )
                else:
                    final_stats[key] = s
            # v0.7.7: capture lazy-pipeline-specific perf stats so
            # we can surface CPU-saturation diagnostics in the
            # summary. Eager pipeline doesn't have these attributes;
            # use getattr with defaults so the path stays generic.
            ls = getattr(pipe, "_stats", None)
            if ls is not None and hasattr(ls, "ring_overflows"):
                lazy_perf_stats["ring_overflows"] = (
                    lazy_perf_stats.get("ring_overflows", 0)
                    + ls.ring_overflows
                )
                lazy_perf_stats["samples_dropped"] = (
                    lazy_perf_stats.get("samples_dropped", 0)
                    + ls.samples_dropped
                )
                lazy_perf_stats["racing_wins"] = (
                    lazy_perf_stats.get("racing_wins", 0)
                    + ls.racing_wins
                )
                lazy_perf_stats["racing_losers_killed"] = (
                    lazy_perf_stats.get("racing_losers_killed", 0)
                    + ls.racing_losers_killed
                )
                lazy_perf_stats["racing_unresolved"] = (
                    lazy_perf_stats.get("racing_unresolved", 0)
                    + ls.racing_unresolved
                )
                lazy_perf_stats["slot_activations"] = (
                    lazy_perf_stats.get("slot_activations", 0)
                    + ls.slot_activations
                )
                # v0.7.11/12: probe stats (only present on lazy
                # pipeline; eager pipeline doesn't probe).
                for fname in ("probe_decisions", "probe_filtered",
                              "probe_kept_all", "probe_rejected",
                              "periodic_probe_scans",
                              "periodic_probe_positive",
                              "periodic_reaps", "periodic_respawns",
                              "pin_events", "unpin_events",
                              "reap_skipped_pinned",
                              # v0.7.16: reap-while-decoding deferral
                              # diagnostics. See LazyPipelineStats for
                              # field semantics.
                              "reap_deferred_busy",
                              "reap_completed_after_defer",
                              "reap_force_after_hung"):
                    if hasattr(ls, fname):
                        lazy_perf_stats[fname] = (
                            lazy_perf_stats.get(fname, 0)
                            + getattr(ls, fname)
                        )
            if hasattr(pipe, "keepup_ratio"):
                # Latest reading wins (it's already a rolling mean).
                lazy_perf_stats["keepup_ratio"] = pipe.keepup_ratio

    try:
        if args.pcap:
            pcap_writer = PcapLoraTapWriter(args.pcap)
            pcap_writer.open()
        if args.jsonl:
            jsonl_fp = args.jsonl.open("w")

        max_runtime_deadline = (t0 + args.max_runtime
                                 if args.max_runtime else None)

        if args.hop:
            # Hop mode: cycle through hop_plan, dwelling on each
            # tuning for hop_dwell seconds (or until max_runtime).
            with source as src:
                hop_idx = 0
                while True:
                    tuning = hop_plan[hop_idx % len(hop_plan)]
                    if not args.quiet:
                        ks = ",".join(s.preset.key for s in tuning.slots)
                        print(f"# [hop {hop_idx}] tune "
                              f"{tuning.center_freq_hz/1e6:.3f}MHz  →  {ks}",
                              file=sys.stderr)
                    # Retune (no-op on first iteration since we already
                    # constructed the source at hop_plan[0]).
                    if hop_idx > 0:
                        source.retune(tuning.center_freq_hz)  # type: ignore[union-attr]
                    pipe = _build_pipeline(
                        list(tuning.slots), tuning.center_freq_hz,
                    )
                    dwell_until = time.time() + args.hop_dwell
                    if max_runtime_deadline is not None:
                        dwell_until = min(dwell_until, max_runtime_deadline)
                    _pump_one_pipeline(pipe, src, dwell_until, t0)
                    pipe.close()
                    if (max_runtime_deadline is not None
                            and time.time() >= max_runtime_deadline):
                        if not args.quiet:
                            print("# max-runtime reached", file=sys.stderr)
                        break
                    hop_idx += 1
        else:
            # Single tuning: build pipeline once, pump until done.
            pipe = _build_pipeline(slots, center_hz)
            dwell_until = (max_runtime_deadline
                            if max_runtime_deadline is not None
                            else float("inf"))
            with source as src:
                _pump_one_pipeline(pipe, src, dwell_until, t0)
            pipe.close()
            if max_runtime_deadline and time.time() >= max_runtime_deadline:
                if not args.quiet:
                    print("# max-runtime reached", file=sys.stderr)
    except KeyboardInterrupt:
        interrupted = True
        if not args.quiet:
            print("\n# interrupted", file=sys.stderr)
    finally:
        if pcap_writer:
            pcap_writer.close()
        if jsonl_fp:
            jsonl_fp.close()

    elapsed = time.time() - t0
    print(file=sys.stderr)
    print(f"# elapsed:     {elapsed:.1f}s", file=sys.stderr)
    print(f"# packets:     {n_total}", file=sys.stderr)
    print(f"# decrypted:   {n_decrypted}", file=sys.stderr)
    if per_preset_decrypt:
        for key, n in sorted(per_preset_decrypt.items()):
            print(f"#   {key:<14}  {n}", file=sys.stderr)
    for key, s in final_stats.items():
        # v0.7.16: also surface sync_ok and hdr_fail to make low-SNR
        # captures diagnosable. preambles > 0 with sync = 0 means
        # the chirp matches but the syncword didn't (= not Meshtastic
        # traffic, or wrong region channel hash). sync > 0 with
        # hdr_ok = 0 means signal is too weak to get the header
        # through CRC (raise gain or move antenna).
        print(f"# {key:<14}  preambles={s.preambles_found}  "
              f"sync={s.syncwords_matched}  "
              f"hdr_ok={s.headers_decoded}  hdr_fail={s.headers_failed}  "
              f"pkt_ok={s.packets_decoded}  pkt_crc_fail={s.packets_crc_failed}",
              file=sys.stderr)
    # v0.7.10: per-channel RSSI broken down by slot frequency. The
    # flat "MediumFast: -2.8 dBFS, n=8" view didn't tell the user
    # which slot frequency the signal was actually arriving at —
    # critical when adjacent slots can pick up the same physical
    # transmission with different SNR. New format groups by channel
    # then sorts slots by hit count descending, with the strongest
    # (== closest to the real Tx frequency) slot listed first.
    if per_channel_slot_rssi:
        print(file=sys.stderr)
        print("# RSSI by channel + slot (median, p10..p90, n samples):",
              file=sys.stderr)
        # Group: { channel_name -> [(freq_hz, [rssi samples])] }
        by_channel: dict[str, list[tuple[int, list[float]]]] = {}
        for (ch, freq), samples in per_channel_slot_rssi.items():
            by_channel.setdefault(ch, []).append((freq, samples))
        for ch in sorted(by_channel):
            print(f"#   {ch}:", file=sys.stderr)
            # Sort by hit count descending — primary slot first.
            slots_sorted = sorted(
                by_channel[ch], key=lambda fs: (-len(fs[1]), fs[0]),
            )
            for freq, samples in slots_sorted:
                samples_s = sorted(samples)
                n = len(samples_s)
                median = samples_s[n // 2]
                p10 = samples_s[max(0, n // 10)]
                p90 = samples_s[min(n - 1, (n * 9) // 10)]
                print(f"#     {freq/1e6:>8.3f} MHz   "
                      f"{median:+5.1f} dBFS  ({p10:+5.1f}..{p90:+5.1f}, "
                      f"n={n})",
                      file=sys.stderr)
    # v0.7.9: per-slot decoder hit breakdown. Particularly useful
    # when running --slots all or --slots multiple — tells the
    # user which slot frequencies are productive vs which are
    # consuming detector + decoder cycles for no return. Format:
    # "MEDIUM_FAST @ 913.375 MHz   142 packets". Sorted by hit
    # count descending so the most active slots are at the top.
    if per_slot_hits:
        print(file=sys.stderr)
        print("# Per-slot decoder hits (decoded packets per slot):",
              file=sys.stderr)
        for (preset_key, freq_hz), n in sorted(
            per_slot_hits.items(),
            key=lambda kv: (-kv[1], kv[0]),
        ):
            print(f"#   {preset_key:<14} @ {freq_hz/1e6:>8.3f} MHz   "
                  f"{n:>5} packets",
                  file=sys.stderr)
    # v0.7.10: adjacent-slot bleed-through report. CRC fails that
    # were within ~50ms of a CRC-pass at a NEIGHBORING slot are
    # almost certainly the same physical transmission leaking
    # into an adjacent decoder's bandpass — single-Tx
    # adjacent-channel pickup, NOT genuine collision/interference.
    # Without this report, those CRC fails get labeled as
    # "interference / collision" by the per-packet SNR rule, which
    # is technically wrong (no actual collision happened, just
    # one Tx leaking sideways into multiple decoders).
    if bleed_pairs:
        print(file=sys.stderr)
        print("# Adjacent-slot bleed-through (single-Tx leak into "
              "neighbor decoders):",
              file=sys.stderr)
        # Sort by event count descending.
        for (fail_freq, pass_freq), n in sorted(
            bleed_pairs.items(), key=lambda kv: (-kv[1], kv[0]),
        ):
            print(f"#   {n:>4}× CRC fail @ {fail_freq/1e6:>8.3f} MHz "
                  f"caused by Tx on {pass_freq/1e6:>8.3f} MHz "
                  f"(NOT real interference)",
                  file=sys.stderr)
    # v0.7.7: lazy-pipeline performance diagnostics. Only printed
    # when actually using lazy (eager doesn't populate the dict).
    if lazy_perf_stats:
        print(file=sys.stderr)
        # Keep-up ratio first — it's the bottom-line "are we
        # processing in real-time" indicator.
        kr = lazy_perf_stats.get("keepup_ratio", 0.0)
        if kr > 0:
            if kr < 0.6:
                health = "healthy"
            elif kr < 0.85:
                health = "ok"
            elif kr < 1.0:
                health = "tight (CPU near saturation)"
            else:
                health = "OVERLOADED — samples are being dropped"
            print(f"# keep-up:     {kr*100:.0f}% of real-time "
                  f"({health})", file=sys.stderr)
        sa = lazy_perf_stats.get("slot_activations", 0)
        rw = lazy_perf_stats.get("racing_wins", 0)
        rlk = lazy_perf_stats.get("racing_losers_killed", 0)
        ru = lazy_perf_stats.get("racing_unresolved", 0)
        if sa > 0:
            print(f"# slot fires:  {sa}", file=sys.stderr)
            print(f"# SF races:    {rw} won "
                  f"({rlk} loser decoder(s) killed early)  ·  "
                  f"{ru} unresolved (no preamble locked)",
                  file=sys.stderr)
        # v0.7.11/12: probe diagnostics
        pd = lazy_perf_stats.get("probe_decisions", 0)
        if pd > 0:
            pf = lazy_perf_stats.get("probe_filtered", 0)
            pk = lazy_perf_stats.get("probe_kept_all", 0)
            pr = lazy_perf_stats.get("probe_rejected", 0)
            print(f"# probe scans: {pd}  "
                  f"({pf} decoder(s) skipped via SF-filter, "
                  f"{pr} activations rejected by gate, "
                  f"{pk} fell back to spawn-all)",
                  file=sys.stderr)
        # v0.7.13: periodic probe / reap-respawn diagnostics
        pps = lazy_perf_stats.get("periodic_probe_scans", 0)
        if pps > 0:
            ppp = lazy_perf_stats.get("periodic_probe_positive", 0)
            preap = lazy_perf_stats.get("periodic_reaps", 0)
            presp = lazy_perf_stats.get("periodic_respawns", 0)
            pct = (100.0 * ppp / pps) if pps else 0.0
            print(f"# periodic:    {pps} probe scans "
                  f"({ppp} positive = {pct:.1f}%)  ·  "
                  f"{preap} reap event(s)  ·  "
                  f"{presp} respawn event(s)",
                  file=sys.stderr)
            # v0.7.13 commit 1c: pin diagnostics. Only print when
            # pin features are showing activity (most quiet captures
            # never pin).
            pe = lazy_perf_stats.get("pin_events", 0)
            ue = lazy_perf_stats.get("unpin_events", 0)
            rsp = lazy_perf_stats.get("reap_skipped_pinned", 0)
            if pe or ue or rsp:
                net_pinned = pe - ue
                print(f"# pin:         {pe} pin event(s) "
                      f"({ue} unpin · net {net_pinned:+d})  ·  "
                      f"{rsp} reap(s) skipped (slot was pinned)",
                      file=sys.stderr)
            # v0.7.16: reap-while-decoding deferral diagnostics.
            # Only print when something happened — quiet captures
            # never trigger this.
            rdb = lazy_perf_stats.get("reap_deferred_busy", 0)
            rcad = lazy_perf_stats.get("reap_completed_after_defer", 0)
            rfh = lazy_perf_stats.get("reap_force_after_hung", 0)
            if rdb or rcad or rfh:
                msg = (f"# reap-defer: {rdb} probe-tick(s) deferred "
                       f"because decoder was busy  ·  "
                       f"{rcad} reap(s) completed after defer")
                if rfh:
                    msg += f"  ·  ⚠ {rfh} force-reap(s) (hung decoder)"
                print(msg, file=sys.stderr)
        sd = lazy_perf_stats.get("samples_dropped", 0)
        if sd > 0:
            ro = lazy_perf_stats.get("ring_overflows", 0)
            print(f"# DROPPED:     {sd:,} samples in "
                  f"{ro} ring-buffer overflow event(s) — "
                  f"these packets were lost", file=sys.stderr)
    # v0.7.16: surface IqReader (decouple ring) stats if a reader was
    # in use. Distinct from the lazy_pipeline lookback ring stats
    # above — this is the SOURCE-side decouple buffer that absorbs
    # network/processing jitter.
    if isinstance(source, IqReader):
        rs = source.stats()
        if rs.ring.samples_dropped > 0:
            print(f"# READER DROP: {rs.ring.samples_dropped:,} samples in "
                  f"{rs.ring.overflow_events} decouple-ring overflow(s) "
                  f"— processing fell behind the reader thread; consider "
                  f"raising --reader-buffer-secs", file=sys.stderr)
        elif not args.quiet:
            cap_mb = rs.ring.capacity_bytes / 1_000_000
            print(f"# reader:      {cap_mb:.1f}MB ring  ·  no drops",
                  file=sys.stderr)
    if pcap_writer:
        print(f"# wrote PCAP:  {args.pcap} "
              f"({pcap_writer.packets_written} packets)", file=sys.stderr)
    if jsonl_fp:
        print(f"# wrote JSONL: {args.jsonl}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
