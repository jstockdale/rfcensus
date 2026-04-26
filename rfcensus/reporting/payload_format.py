"""Per-protocol payload formatting for decode rows.

Turns a ``DecodeRecord`` (or just its ``protocol`` + ``payload``) into a
single human-readable line. Used by:

  • ``rfcensus list decodes`` — recent-decodes table
  • TUI ``RecentDecodesWidget`` — live rolling view
  • ``rfcensus export decodes --format text`` — text export form

Each protocol gets a small dedicated formatter that knows what fields
its decoder writes into the payload dict. Unknown protocols fall back
to a compact key=value dump.

Design notes
------------

  • Formatters take only the payload dict, not a DecodeRecord, so the
    TUI can format from a DecodeEvent directly without having to fake
    a record.
  • Output is intentionally short — wide TUI terminals are not
    guaranteed; the table that holds these strings is also fixed-
    width.
  • Encrypted-but-not-decrypted Meshtastic packets are shown as
    ``[encrypted; ch=0xNN]`` so the operator knows there is a
    custom-channel transmission they could decrypt with the right
    PSK in site config — this is the load-bearing UX hint.
"""

from __future__ import annotations

from typing import Any


def format_payload(protocol: str, payload: dict[str, Any]) -> str:
    """Return a short one-line summary of ``payload`` for ``protocol``.

    Falls back to ``_format_generic`` for protocols without a dedicated
    formatter."""
    fmt = _FORMATTERS.get(protocol, _format_generic)
    try:
        return fmt(payload)
    except Exception as exc:    # pragma: no cover - defensive
        # Never let a bad payload break a list/TUI render
        return f"[format error: {exc!r}]"


# ─────────────────────────────────────────────────────────────────────
# Meshtastic
# ─────────────────────────────────────────────────────────────────────


# Meshtastic port number → human label. Only the most common ones; for
# anything else we just show the number.
_MESH_PORTS = {
    0: "UNKNOWN",
    1: "TEXT",
    2: "REMOTE_HW",
    3: "POSITION",
    4: "NODEINFO",
    5: "ROUTING",
    6: "ADMIN",
    7: "TEXT_COMPRESSED",
    8: "WAYPOINT",
    9: "AUDIO",
    10: "DETECTION_SENSOR",
    32: "REPLY",
    33: "IP_TUNNEL",
    34: "PAXCOUNTER",
    64: "SERIAL",
    65: "STORE_FORWARD",
    66: "RANGE_TEST",
    67: "TELEMETRY",
    68: "ZPS",
    69: "SIMULATOR",
    70: "TRACEROUTE",
    71: "NEIGHBORINFO",
    72: "ATAK_PLUGIN",
    73: "MAP_REPORT",
    74: "POWERSTRESS",
}


def _format_meshtastic(payload: dict[str, Any]) -> str:
    """Render a Meshtastic decode in compact arrow form.

    Variants:
      • CRC fail:                   ``[crc fail; preset=X]``
      • CRC ok, no decrypt:         ``[encrypted; ch=0xNN preset=X]``
      • Decrypted text msg (port 1): ``0xAAAA→0xBBBB TEXT "hello"``
      • Decrypted other port:        ``0xAAAA→0xBBBB POSITION (24 B)``

    The arrow form mirrors how Meshtastic itself displays
    conversations and makes from/to relationships scan quickly."""
    if not payload.get("crc_ok", False):
        preset = payload.get("preset", "?")
        return f"[crc fail; preset={preset}]"

    decrypted = payload.get("decrypted", False)
    if not decrypted:
        ch = payload.get("channel_hash")
        ch_str = f"0x{ch:02X}" if isinstance(ch, int) else "?"
        preset = payload.get("preset", "?")
        return f"[encrypted; ch={ch_str} preset={preset}]"

    # Decrypted: from→to + port + text/length
    fr = payload.get("from_node", 0)
    to = payload.get("to_node", 0)
    fr_s = f"0x{fr:08X}" if fr != 0xFFFFFFFF else "ALL"
    # Meshtastic broadcast destination is 0xFFFFFFFF; show as BCAST
    if to == 0xFFFFFFFF:
        to_s = "BCAST"
    else:
        to_s = f"0x{to:08X}"

    # Port — derive from the second byte of plaintext (port-tag layout
    # we already parse for the ``text`` field). If we have a ``text``
    # already, we know it's port 1.
    port_label = "?"
    pt_hex = payload.get("plaintext_hex", "")
    if pt_hex:
        try:
            pt = bytes.fromhex(pt_hex)
            if len(pt) >= 2 and pt[0] == 0x08:
                port_num = pt[1]
                port_label = _MESH_PORTS.get(port_num, f"port{port_num}")
        except Exception:
            pass

    text = payload.get("text")
    if text:
        # Quote the text and trim long messages
        snippet = text if len(text) <= 60 else text[:57] + "..."
        return f"{fr_s}→{to_s} {port_label} {snippet!r}"

    plen = payload.get("plaintext_len", 0)
    return f"{fr_s}→{to_s} {port_label} ({plen} B)"


# ─────────────────────────────────────────────────────────────────────
# rtl_433 protocols
# ─────────────────────────────────────────────────────────────────────


def _format_rtl433_like(payload: dict[str, Any]) -> str:
    """Generic rtl_433 / ert formatter — the dict has a flat structure
    with 'msg_type', 'commodity', 'consumption', 'raw' keys."""
    msg = payload.get("msg_type", "?")
    parts = [str(msg)]
    if payload.get("commodity"):
        parts.append(f"commodity={payload['commodity']}")
    if payload.get("consumption") is not None:
        parts.append(f"consumption={payload['consumption']}")
    if payload.get("_device_id"):
        parts.append(f"id={payload['_device_id']}")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# Generic fallback
# ─────────────────────────────────────────────────────────────────────


def _format_generic(payload: dict[str, Any], max_len: int = 100) -> str:
    """Compact key=value dump skipping leading-underscore keys (which
    are conventionally internal — e.g. ``_device_id`` for emitter
    tracking) and any nested dicts (which are usually raw protocol
    bytes the user doesn't want in their summary view)."""
    parts: list[str] = []
    for k, v in payload.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            parts.append(f"{k}={{...{len(v)}}}")
            continue
        if isinstance(v, (list, tuple)) and len(v) > 4:
            parts.append(f"{k}=[…×{len(v)}]")
            continue
        parts.append(f"{k}={v}")
    s = " ".join(parts)
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    return s


# ─────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────


_FORMATTERS = {
    "meshtastic": _format_meshtastic,
    # rtl_433 / rtlamr family
    "tpms": _format_rtl433_like,
    "weather_station": _format_rtl433_like,
    "ert_scm": _format_rtl433_like,
    "ert_scm_plus": _format_rtl433_like,
    "ert_idm": _format_rtl433_like,
    "ert_netidm": _format_rtl433_like,
    "r900": _format_rtl433_like,
    "r900_bcd": _format_rtl433_like,
}
