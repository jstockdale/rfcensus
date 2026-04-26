"""Thin ctypes wrapper around liblora_demod.so.

Exposes the streaming LoRa physical-layer decoder (the C library at
``rfcensus/decoders/_native/lora/``) as a Pythonic callback interface.

The C decoder is callback-driven: caller pumps IQ samples in, decoder
fires a callback per decoded packet. We mirror that with a generator-ish
adapter that buffers packets between ``feed()`` calls and yields them
out via :meth:`pop_packets`.

Typical usage::

    from rfcensus.decoders.lora_native import LoraDecoder, LoraConfig

    cfg = LoraConfig(
        sample_rate_hz=1_000_000,
        bandwidth=250_000,
        sf=9,
        sync_word=0x2B,           # Meshtastic
        mix_freq_hz=375_000,      # signal sits at capture_freq - 375 kHz
    )
    dec = LoraDecoder(cfg)

    while chunk := source.read(32_768):       # cu8 bytes
        dec.feed_cu8(chunk)
        for pkt in dec.pop_packets():
            print(f"len={pkt.payload_len} crc_ok={pkt.crc_ok} "
                  f"@sample {pkt.sample_offset}")

    print(dec.stats())
"""
from __future__ import annotations

import ctypes
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


# ─────────────────────────────────────────────────────────────────────
# C struct mirrors — see lora_demod.h
# ─────────────────────────────────────────────────────────────────────

LORA_MAX_PAYLOAD = 255

# Bandwidth enum values (Hz). Match lora_bw_t in lora_demod.h.
LORA_BW = {
    125_000: 125_000,
    250_000: 250_000,
    500_000: 500_000,
}


class _CConfig(ctypes.Structure):
    """Mirror of ``lora_config_t``."""
    _fields_ = [
        ("sample_rate_hz",    ctypes.c_uint32),
        ("bandwidth",         ctypes.c_int),     # lora_bw_t enum
        ("sf",                ctypes.c_uint8),
        ("sync_word",         ctypes.c_uint8),
        ("has_crc_default",   ctypes.c_uint8),
        ("ldro",              ctypes.c_uint8),
        ("mix_freq_hz",       ctypes.c_int32),
    ]


class _CDecoded(ctypes.Structure):
    """Mirror of ``lora_decoded_t``."""
    _fields_ = [
        ("payload",       ctypes.c_uint8 * LORA_MAX_PAYLOAD),
        ("payload_len",   ctypes.c_uint16),
        ("cr",            ctypes.c_uint8),
        ("has_crc",       ctypes.c_uint8),
        ("crc_ok",        ctypes.c_uint8),
        ("rssi_db",       ctypes.c_float),
        ("snr_db",        ctypes.c_float),
        ("cfo_hz",        ctypes.c_float),
        ("sample_offset", ctypes.c_uint64),
    ]


class _CStats(ctypes.Structure):
    """Mirror of ``lora_demod_stats_t``."""
    _fields_ = [
        ("samples_processed",    ctypes.c_uint64),
        ("preambles_found",      ctypes.c_uint32),
        ("syncwords_matched",    ctypes.c_uint32),
        ("headers_decoded",      ctypes.c_uint32),
        ("headers_failed",       ctypes.c_uint32),
        ("packets_decoded",      ctypes.c_uint32),
        ("packets_crc_failed",   ctypes.c_uint32),
        ("detect_attempts",      ctypes.c_uint64),
        ("detect_above_gate",    ctypes.c_uint64),
        ("detect_max_run",       ctypes.c_uint32),
        ("detect_peak_mag_max",  ctypes.c_float),
    ]


# Callback signature: void(const lora_decoded_t*, void*)
_CALLBACK = ctypes.CFUNCTYPE(None,
                              ctypes.POINTER(_CDecoded),
                              ctypes.c_void_p)


# ─────────────────────────────────────────────────────────────────────
# Library loading
# ─────────────────────────────────────────────────────────────────────

_NATIVE_DIR = Path(__file__).parent / "_native" / "lora"


def _find_library() -> Path:
    env = os.environ.get("RFCENSUS_LIBLORA")
    if env and Path(env).exists():
        return Path(env)
    vendored = _NATIVE_DIR / "liblora_demod.so"
    if vendored.exists():
        return vendored
    raise FileNotFoundError(
        f"liblora_demod.so not found. Build with:\n"
        f"  cd {_NATIVE_DIR} && make"
    )


def _load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_find_library()))

    lib.lora_demod_new.restype = ctypes.c_void_p
    lib.lora_demod_new.argtypes = [
        ctypes.POINTER(_CConfig),
        _CALLBACK,
        ctypes.c_void_p,
    ]
    lib.lora_demod_free.restype = None
    lib.lora_demod_free.argtypes = [ctypes.c_void_p]

    lib.lora_demod_process_cf.restype = ctypes.c_int
    lib.lora_demod_process_cf.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.lora_demod_process_cu8.restype = ctypes.c_int
    lib.lora_demod_process_cu8.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    # v0.7.11: feed_baseband — bypass mix+resamp for shared-channel use
    lib.lora_demod_feed_baseband.restype = ctypes.c_int
    lib.lora_demod_feed_baseband.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.lora_demod_reset.restype = None
    lib.lora_demod_reset.argtypes = [ctypes.c_void_p]
    lib.lora_demod_get_stats.restype = None
    lib.lora_demod_get_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_CStats),
    ]
    return lib


_lib: Optional[ctypes.CDLL] = None


def _lib_get() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


# ─────────────────────────────────────────────────────────────────────
# Public Python interface
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LoraConfig:
    """Static configuration for a LoRa decoder instance.

    All settings are channel properties — they don't change during the
    life of the instance. To switch frequency or SF, free this decoder
    and create a new one.
    """
    sample_rate_hz: int             # IQ stream rate (Hz)
    bandwidth: int                  # 125_000 / 250_000 / 500_000
    sf: int                         # 7..12
    sync_word: int = 0x2B           # 0x2B = Meshtastic, 0x12 = generic LoRa
    has_crc_default: bool = True    # only used in implicit-header mode (unused for Meshtastic)
    ldro: bool = False              # Low Data Rate Optimization
    mix_freq_hz: int = 0            # digital downmix; 0 = no mix


@dataclass
class LoraPacket:
    """One decoded LoRa packet, as fired by the C callback."""
    payload: bytes              # raw bytes (still encrypted if Meshtastic)
    payload_len: int
    cr: int                     # coding rate 1..4
    has_crc: bool
    crc_ok: bool
    rssi_db: float
    snr_db: float
    cfo_hz: float
    sample_offset: int          # IQ-stream sample at start of preamble


@dataclass
class LoraStats:
    """Cumulative decoder statistics — see lora_demod_stats_t."""
    samples_processed: int = 0
    preambles_found: int = 0
    syncwords_matched: int = 0
    headers_decoded: int = 0
    headers_failed: int = 0
    packets_decoded: int = 0
    packets_crc_failed: int = 0
    detect_attempts: int = 0
    detect_above_gate: int = 0
    detect_max_run: int = 0
    detect_peak_mag_max: float = 0.0


class LoraDecoder:
    """Streaming LoRa decoder.

    Pump IQ samples in via :meth:`feed_cu8` (uint8 from rtl_sdr) or
    :meth:`feed_cf` (interleaved float32). Decoded packets are buffered
    internally; consume them with :meth:`pop_packets` after each feed
    or in bulk at the end.

    The instance is NOT thread-safe — one feed call at a time.
    """

    def __init__(self, config: LoraConfig) -> None:
        if config.bandwidth not in LORA_BW:
            raise ValueError(
                f"unsupported bandwidth {config.bandwidth}; "
                f"valid: {sorted(LORA_BW)}"
            )
        if not (7 <= config.sf <= 12):
            raise ValueError(f"SF {config.sf} out of range 7..12")

        self._lib = _lib_get()
        self._config = config
        self._packets: deque[LoraPacket] = deque()

        cfg = _CConfig(
            sample_rate_hz=config.sample_rate_hz,
            bandwidth=LORA_BW[config.bandwidth],
            sf=config.sf,
            sync_word=config.sync_word,
            has_crc_default=int(config.has_crc_default),
            ldro=int(config.ldro),
            mix_freq_hz=config.mix_freq_hz,
        )

        # The callback bridges C → Python. We MUST hold a reference to
        # the CFUNCTYPE wrapper for the lifetime of the decoder, else
        # Python GCs it and the C side calls into freed memory.
        self._cb_holder = _CALLBACK(self._on_packet)

        self._handle = self._lib.lora_demod_new(
            ctypes.byref(cfg),
            self._cb_holder,
            None,  # userdata
        )
        if not self._handle:
            raise RuntimeError("lora_demod_new returned NULL")

    def __del__(self) -> None:
        if getattr(self, "_handle", None) and getattr(self, "_lib", None):
            try:
                self._lib.lora_demod_free(self._handle)
            except Exception:
                pass
            self._handle = None

    @property
    def config(self) -> LoraConfig:
        return self._config

    # ── Streaming input ─────────────────────────────────────────────

    def feed_cu8(self, samples: bytes) -> int:
        """Feed cu8-format IQ samples (rtl_sdr default).

        Each sample is 2 bytes: I, Q each as uint8 centered on 127.5.
        Returns the number of complete packets decoded during this call
        (those packets are also queued for ``pop_packets``).
        """
        if len(samples) % 2 != 0:
            raise ValueError("cu8 stream must be even-length (I/Q pairs)")
        n_samples = len(samples) // 2
        if n_samples == 0:
            return 0
        buf = (ctypes.c_uint8 * len(samples)).from_buffer_copy(samples)
        return self._lib.lora_demod_process_cu8(
            self._handle, buf, n_samples
        )

    def feed_cf(self, samples) -> int:
        """Feed float32 interleaved-IQ samples.

        ``samples`` may be any buffer protocol object (numpy array,
        bytes, array.array). Length is the number of *complex* samples,
        so the underlying buffer must contain 2 × that many floats.
        """
        # Accept bytes/bytearray directly; for numpy convert via .tobytes()
        if hasattr(samples, "tobytes"):
            buf_bytes = samples.tobytes()
        else:
            buf_bytes = bytes(samples)
        n_floats = len(buf_bytes) // 4
        n_samples = n_floats // 2
        if n_samples == 0:
            return 0
        buf = (ctypes.c_float * n_floats).from_buffer_copy(buf_bytes)
        return self._lib.lora_demod_process_cf(
            self._handle, buf, n_samples
        )

    def feed_baseband(self, samples) -> int:
        """v0.7.11: feed already-channelized baseband samples.

        Samples must be at the decoder's bandwidth rate AND already
        mixed to DC. The decoder MUST have been constructed with
        ``mix_freq_hz=0`` and ``sample_rate_hz=bandwidth`` — the
        on-chip mixer and resampler are bypassed.

        Used by SharedChannelGroup to fan one channelization out to
        N concurrent SF decoders at the same slot frequency.

        Same input format as ``feed_cf`` (interleaved float32 I/Q).
        """
        if hasattr(samples, "tobytes"):
            buf_bytes = samples.tobytes()
        else:
            buf_bytes = bytes(samples)
        n_floats = len(buf_bytes) // 4
        n_samples = n_floats // 2
        if n_samples == 0:
            return 0
        buf = (ctypes.c_float * n_floats).from_buffer_copy(buf_bytes)
        return self._lib.lora_demod_feed_baseband(
            self._handle, buf, n_samples
        )

    def reset(self) -> None:
        """Reset frame-sync state (e.g. after retuning)."""
        self._lib.lora_demod_reset(self._handle)

    # ── Output ──────────────────────────────────────────────────────

    def pop_packets(self) -> Iterator[LoraPacket]:
        """Drain the buffered-packet queue."""
        while self._packets:
            yield self._packets.popleft()

    def stats(self) -> LoraStats:
        """Snapshot the cumulative decoder counters."""
        s = _CStats()
        self._lib.lora_demod_get_stats(self._handle, ctypes.byref(s))
        return LoraStats(
            samples_processed=s.samples_processed,
            preambles_found=s.preambles_found,
            syncwords_matched=s.syncwords_matched,
            headers_decoded=s.headers_decoded,
            headers_failed=s.headers_failed,
            packets_decoded=s.packets_decoded,
            packets_crc_failed=s.packets_crc_failed,
            detect_attempts=s.detect_attempts,
            detect_above_gate=s.detect_above_gate,
            detect_max_run=s.detect_max_run,
            detect_peak_mag_max=s.detect_peak_mag_max,
        )

    # ── Internal callback ───────────────────────────────────────────

    def _on_packet(self, pkt_ptr, _userdata) -> None:
        """C callback — copy out and queue."""
        p = pkt_ptr.contents
        # Defensive copy: the C side reuses the buffer after this returns.
        self._packets.append(LoraPacket(
            payload=bytes(p.payload[: p.payload_len]),
            payload_len=p.payload_len,
            cr=p.cr,
            has_crc=bool(p.has_crc),
            crc_ok=bool(p.crc_ok),
            rssi_db=p.rssi_db,
            snr_db=p.snr_db,
            cfo_hz=p.cfo_hz,
            sample_offset=p.sample_offset,
        ))
