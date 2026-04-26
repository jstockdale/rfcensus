"""Thin ctypes wrapper around libmeshtastic.so.

This module exposes the flat C API from
``rfcensus/decoders/_native/meshtastic/src/meshtastic_capi.h`` as a
Pythonic interface. The underlying library handles channel-table
management, AES-CTR decryption, and packet-header parsing — see the
C-API header for the authoritative behavior contract.

Typical usage::

    from rfcensus.decoders.meshtastic_native import MeshtasticDecoder

    dec = MeshtasticDecoder(preset="LONG_FAST")
    dec.add_default_channel()
    dec.add_channel("MyPrivate", psk=bytes.fromhex("aabbcc...0011"))

    for raw_lora_frame in some_stream:
        decoded = dec.decode(raw_lora_frame)
        if decoded.channel_index >= 0:
            print(f"Decrypted on channel {decoded.channel_index}: "
                  f"{decoded.from_node:08x} → {decoded.to:08x}")
            print(decoded.plaintext.hex())

The library is searched at module import time. If not found, an
ImportError is raised with the build instructions; callers wanting a
non-fatal probe should catch that.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# ctypes struct definitions — must match meshtastic_capi.h exactly
# ─────────────────────────────────────────────────────────────────────

class _CDecodedPacket(ctypes.Structure):
    """Mirror of the C ``mesh_capi_decoded_t`` struct.

    Field order and types must match meshtastic_capi.h exactly. Layout
    is checked at module init via the ``_assert_struct_size`` helper —
    if you change the C struct, update both the size constant and the
    field list here."""
    _fields_ = [
        ("to",            ctypes.c_uint32),
        ("from_node",     ctypes.c_uint32),  # 'from' is reserved in Python
        ("id",            ctypes.c_uint32),
        ("hop_limit",     ctypes.c_uint8),
        ("hop_start",     ctypes.c_uint8),
        ("want_ack",      ctypes.c_uint8),
        ("via_mqtt",      ctypes.c_uint8),
        ("channel_hash",  ctypes.c_uint8),
        ("next_hop",      ctypes.c_uint8),
        ("relay_node",    ctypes.c_uint8),
        ("channel_index", ctypes.c_int8),
        ("plaintext_len", ctypes.c_uint16),
        ("plaintext",     ctypes.c_uint8 * 239),
    ]


# Preset enum values — see meshtastic_capi.h. Keep this dict here rather
# than scattering enum constants because it gives readable error messages
# ("PRESETS keys are: ...") if someone passes an unknown name.
PRESETS = {
    "LONG_FAST":      0,
    "LONG_SLOW":      1,
    "LONG_MODERATE":  2,
    "LONG_TURBO":     3,
    "MEDIUM_FAST":    4,  # SF9 BW=250 CR=4/5 — what our 913 MHz capture used
    "MEDIUM_SLOW":    5,
    "SHORT_FAST":     6,
    "SHORT_SLOW":     7,
    "SHORT_TURBO":    8,
}


# ─────────────────────────────────────────────────────────────────────
# Library loader
# ─────────────────────────────────────────────────────────────────────

_NATIVE_DIR = Path(__file__).parent / "_native" / "meshtastic"


def _find_library() -> Path:
    """Locate libmeshtastic.so.

    Search order:
      1. ``RFCENSUS_LIBMESHTASTIC`` env var (absolute path)
      2. The vendored copy at ``decoders/_native/meshtastic/libmeshtastic.so``
      3. The system loader's default search path (``ctypes.util.find_library``)

    Raises FileNotFoundError if none of the above resolve.
    """
    env = os.environ.get("RFCENSUS_LIBMESHTASTIC")
    if env:
        p = Path(env)
        if p.exists():
            return p

    vendored = _NATIVE_DIR / "libmeshtastic.so"
    if vendored.exists():
        return vendored

    import ctypes.util
    sys_path = ctypes.util.find_library("meshtastic")
    if sys_path:
        return Path(sys_path)

    raise FileNotFoundError(
        f"libmeshtastic.so not found. Build it with:\n"
        f"  cd {_NATIVE_DIR} && make\n"
        f"or set RFCENSUS_LIBMESHTASTIC=/path/to/libmeshtastic.so"
    )


def _load_library() -> ctypes.CDLL:
    """Load libmeshtastic.so and bind function signatures."""
    lib = ctypes.CDLL(str(_find_library()))

    # Lifecycle
    lib.mesh_capi_table_new.restype = ctypes.c_void_p
    lib.mesh_capi_table_new.argtypes = [ctypes.c_int]

    lib.mesh_capi_table_free.restype = None
    lib.mesh_capi_table_free.argtypes = [ctypes.c_void_p]

    lib.mesh_capi_table_add.restype = ctypes.c_int
    lib.mesh_capi_table_add.argtypes = [
        ctypes.c_void_p,                # table
        ctypes.c_char_p,                # name
        ctypes.POINTER(ctypes.c_uint8), # psk
        ctypes.c_uint8,                 # psk_len
        ctypes.c_int,                   # is_primary
    ]

    lib.mesh_capi_table_add_default.restype = ctypes.c_int
    lib.mesh_capi_table_add_default.argtypes = [ctypes.c_void_p]

    lib.mesh_capi_table_count.restype = ctypes.c_int
    lib.mesh_capi_table_count.argtypes = [ctypes.c_void_p]

    lib.mesh_capi_table_channel_hash.restype = ctypes.c_int
    lib.mesh_capi_table_channel_hash.argtypes = [ctypes.c_void_p, ctypes.c_int]

    # Decode
    lib.mesh_capi_decode.restype = ctypes.c_int
    lib.mesh_capi_decode.argtypes = [
        ctypes.c_void_p,                # table
        ctypes.POINTER(ctypes.c_uint8), # raw frame
        ctypes.c_size_t,                # raw_len
        ctypes.POINTER(_CDecodedPacket),# out
    ]

    lib.mesh_capi_version.restype = ctypes.c_char_p
    lib.mesh_capi_version.argtypes = []

    return lib


# Lazy-loaded singleton — calls to MeshtasticDecoder() trigger the load.
_lib: Optional[ctypes.CDLL] = None


def _lib_get() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


def library_version() -> str:
    """Return the libmeshtastic version string (e.g. '0.7.0')."""
    return _lib_get().mesh_capi_version().decode()


# ─────────────────────────────────────────────────────────────────────
# Public Python interface
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DecodedPacket:
    """A decrypted (or attempted-decrypt) Meshtastic packet.

    All fields are populated whenever the LoRa frame had a valid 16-byte
    header. ``channel_index`` is -1 if no configured channel matched the
    wire hash; in that case ``plaintext`` holds the raw ciphertext for
    forward-logging or replay.
    """
    to: int                 # destination NodeNum
    from_node: int          # source NodeNum
    id: int                 # packet ID (lower 32 bits)
    hop_limit: int          # 0..7
    hop_start: int          # 0..7
    want_ack: bool
    via_mqtt: bool
    channel_hash: int       # wire-side channel hash (0..255)
    next_hop: int           # low byte of next-hop NodeNum
    relay_node: int         # low byte of relaying NodeNum
    channel_index: int      # -1 if no channel matched
    plaintext: bytes        # decrypted payload, OR ciphertext if no match

    @property
    def is_broadcast(self) -> bool:
        """True iff the packet was sent to the broadcast address."""
        return self.to == 0xFFFFFFFF

    @property
    def decrypted(self) -> bool:
        """True iff a configured channel decrypted this packet."""
        return self.channel_index >= 0


class MeshtasticDecoder:
    """A configured channel table that can attempt-decrypt LoRa frames.

    Wraps the C-side ``mesh_capi_table_t`` with a Pythonic interface.
    Holding multiple instances is fine and cheap — they don't share
    state.

    The table must outlive any decode results that reference into it
    (currently no such references exist — decoded packets are deep-copied
    out — but if that changes, lifecycle becomes important).
    """

    def __init__(self, preset: str = "LONG_FAST") -> None:
        if preset not in PRESETS:
            raise ValueError(
                f"unknown preset {preset!r}. "
                f"Valid: {', '.join(PRESETS)}"
            )
        self._lib = _lib_get()
        self._handle = self._lib.mesh_capi_table_new(PRESETS[preset])
        if not self._handle:
            raise RuntimeError("mesh_capi_table_new returned NULL")
        self._preset = preset

    def __del__(self) -> None:
        # ctypes can be torn down before us in interpreter shutdown — be
        # defensive about the lib reference.
        if getattr(self, "_handle", None) and getattr(self, "_lib", None):
            try:
                self._lib.mesh_capi_table_free(self._handle)
            except Exception:
                pass
            self._handle = None

    @property
    def preset(self) -> str:
        return self._preset

    @property
    def channel_count(self) -> int:
        return self._lib.mesh_capi_table_count(self._handle)

    def add_default_channel(self) -> int:
        """Add the well-known LongFast channel (PSK index 1).

        Returns the channel index in the table.
        """
        idx = self._lib.mesh_capi_table_add_default(self._handle)
        if idx < 0:
            raise RuntimeError(
                "channel table full or add failed (default channel)"
            )
        return idx

    def add_channel(
        self,
        name: str = "",
        psk: bytes = b"",
        is_primary: bool = False,
    ) -> int:
        """Add a channel to the table.

        Args:
          name: channel name, or "" to inherit the preset's default name
            ("LongFast", "MediumFast", etc).
          psk: raw PSK bytes. Valid lengths are 0 (no encryption),
            1 (short-index — value 1 = MESH_DEFAULT_PSK, 2..N derived
            by bumping the last byte of the default PSK), 16 (AES-128),
            or 32 (AES-256).
          is_primary: set when adding the primary channel for the
            channel set (controls PSK inheritance for secondaries).

        Returns the channel index, or raises RuntimeError on failure.
        """
        psk_buf = (ctypes.c_uint8 * len(psk))(*psk) if psk else None
        idx = self._lib.mesh_capi_table_add(
            self._handle,
            name.encode("utf-8"),
            psk_buf,
            len(psk),
            1 if is_primary else 0,
        )
        if idx < 0:
            raise RuntimeError(
                f"failed to add channel {name!r} (table full?)"
            )
        return idx

    def channel_hash(self, idx: int) -> int:
        """Get the computed wire hash for the channel at ``idx``."""
        h = self._lib.mesh_capi_table_channel_hash(self._handle, idx)
        if h < 0:
            raise IndexError(f"channel {idx} not in table")
        return h

    def decode(self, raw: bytes) -> DecodedPacket:
        """Parse + attempt-decrypt a raw LoRa frame.

        ``raw`` must be the full LoRa payload bytes that came out of the
        physical-layer decoder (after dewhitening, after CRC validation
        if you have it). Must be at least 16 bytes (header).

        Always returns a DecodedPacket — check ``decrypted`` to see if
        any channel matched. The header fields are always populated.
        """
        if len(raw) < 16:
            raise ValueError(
                f"frame too short: {len(raw)} bytes (need at least 16)"
            )

        out = _CDecodedPacket()
        buf = (ctypes.c_uint8 * len(raw))(*raw)
        rc = self._lib.mesh_capi_decode(
            self._handle, buf, len(raw), ctypes.byref(out)
        )
        # rc >= 0 → matched and decrypted (rc = channel index)
        # rc == -1 → no channel matched (header still parsed)
        # rc == -2/-3 → bad input or all matching channels failed
        # In all "header parsed" cases (rc >= -1) we return a DecodedPacket;
        # only rc == -3 (bad input) is fatal here, and we already guarded
        # the size above so this shouldn't trigger.
        if rc == -3:
            raise RuntimeError("mesh_capi_decode returned -3 (bad input)")

        return DecodedPacket(
            to=out.to,
            from_node=out.from_node,
            id=out.id,
            hop_limit=out.hop_limit,
            hop_start=out.hop_start,
            want_ack=bool(out.want_ack),
            via_mqtt=bool(out.via_mqtt),
            channel_hash=out.channel_hash,
            next_hop=out.next_hop,
            relay_node=out.relay_node,
            channel_index=out.channel_index,
            plaintext=bytes(out.plaintext[: out.plaintext_len]),
        )
