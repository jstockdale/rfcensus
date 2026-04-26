"""Shared channelizer — Python wrapper for lora_channelizer.c.

One channelizer per slot frequency. Mixes + decimates input cu8 once,
emits baseband float pairs ready for ``LoraDecoder.feed_baseband()``.
N concurrent SF decoders at the same slot share one channelizer
instance, dropping (N-1) × channelization cost.

Bit-exactness: the C implementation mirrors the exact mix + linear-
interp resampler from lora_demod's ``ingest_samples_cf``, so a decoder
fed via channelizer + feed_baseband produces identical output to a
decoder fed via feed_cu8 (with matching mix_freq + sample_rate).
"""
from __future__ import annotations

import ctypes

import numpy as np

from rfcensus.decoders.lora_native import _lib_get


_bindings_done = False


def _ensure_bindings() -> ctypes.CDLL:
    global _bindings_done
    lib = _lib_get()
    if _bindings_done:
        return lib
    lib.lora_channelizer_new.restype = ctypes.c_void_p
    lib.lora_channelizer_new.argtypes = [
        ctypes.c_uint32,    # sample_rate_hz
        ctypes.c_uint32,    # bandwidth_hz
        ctypes.c_int32,     # mix_freq_hz
    ]
    lib.lora_channelizer_free.argtypes = [ctypes.c_void_p]
    lib.lora_channelizer_feed_cu8.restype = ctypes.c_size_t
    lib.lora_channelizer_feed_cu8.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.lora_channelizer_feed_cf.restype = ctypes.c_size_t
    lib.lora_channelizer_feed_cf.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.lora_channelizer_samples_in.restype = ctypes.c_uint64
    lib.lora_channelizer_samples_in.argtypes = [ctypes.c_void_p]
    lib.lora_channelizer_samples_out.restype = ctypes.c_uint64
    lib.lora_channelizer_samples_out.argtypes = [ctypes.c_void_p]
    _bindings_done = True
    return lib


class SharedChannelizer:
    """One channelizer per (slot_freq, bandwidth) pair.

    Multiple decoders at the same slot frequency share one of these,
    each consuming the same baseband output via feed_baseband.
    """

    def __init__(
        self,
        sample_rate_hz: int,
        bandwidth_hz: int,
        mix_freq_hz: int,
    ):
        if sample_rate_hz <= 0 or bandwidth_hz <= 0:
            raise ValueError(
                f"invalid rates: sample={sample_rate_hz} bw={bandwidth_hz}"
            )
        if bandwidth_hz > sample_rate_hz:
            raise ValueError(
                f"bandwidth {bandwidth_hz} > sample_rate {sample_rate_hz} "
                f"would require upsampling"
            )
        self.sample_rate_hz = sample_rate_hz
        self.bandwidth_hz = bandwidth_hz
        self.mix_freq_hz = mix_freq_hz
        lib = _ensure_bindings()
        self._handle = lib.lora_channelizer_new(
            sample_rate_hz, bandwidth_hz, mix_freq_hz,
        )
        if not self._handle:
            raise RuntimeError(
                f"lora_channelizer_new failed (rate={sample_rate_hz} "
                f"bw={bandwidth_hz} mix={mix_freq_hz})"
            )
        # Precompute decimation ratio for output buffer sizing.
        # Max outputs per N inputs ≈ ceil(N / step) + 1. We keep one
        # reusable buffer sized for the largest input chunk we expect
        # callers to pass; grown on demand.
        self._decim_ratio = sample_rate_hz / bandwidth_hz
        self._out_buf_capacity = 0
        self._out_buf = None    # type: ignore

    def _ensure_out_buf(self, n_in: int) -> None:
        # +16 padding for the +1 ceiling slop the resampler can produce
        max_out = int(n_in / self._decim_ratio) + 16
        if max_out > self._out_buf_capacity:
            # Round up to a nice power of 2 for cache friendliness.
            cap = 1
            while cap < max_out:
                cap *= 2
            self._out_buf_capacity = cap
            self._out_buf = (ctypes.c_float * (cap * 2))()

    def feed_cu8(self, cu8: bytes) -> np.ndarray:
        """Feed cu8 bytes; return baseband float-pairs as a numpy
        complex64 array (decimated to bandwidth rate, mixed to DC).

        The returned array is a copy backed by Python memory and
        safe to pass to ``LoraDecoder.feed_baseband()`` (or to the
        BlindProbe).
        """
        n_in = len(cu8) // 2
        if n_in == 0:
            return np.zeros(0, dtype=np.complex64)
        self._ensure_out_buf(n_in)
        in_buf = (ctypes.c_uint8 * len(cu8)).from_buffer_copy(cu8)
        lib = _ensure_bindings()
        n_out = lib.lora_channelizer_feed_cu8(
            self._handle,
            in_buf, n_in,
            self._out_buf, self._out_buf_capacity,
        )
        if n_out == 0:
            return np.zeros(0, dtype=np.complex64)
        # Build numpy view of the C buffer's used prefix and copy out.
        # ctypes -> numpy via frombuffer; reshape pairs to complex64.
        flat = np.frombuffer(self._out_buf, dtype=np.float32,
                              count=2 * n_out)
        # Interleaved float -> complex64
        return flat.view(np.complex64).copy()

    def feed_cf(self, samples) -> np.ndarray:
        """Feed already-converted float samples (interleaved I/Q).

        ``samples`` may be a numpy float32 array, bytes, or any
        buffer-protocol object containing 2N floats.
        """
        if hasattr(samples, "tobytes"):
            buf_bytes = samples.tobytes()
        else:
            buf_bytes = bytes(samples)
        n_floats = len(buf_bytes) // 4
        n_in = n_floats // 2
        if n_in == 0:
            return np.zeros(0, dtype=np.complex64)
        self._ensure_out_buf(n_in)
        in_buf = (ctypes.c_float * n_floats).from_buffer_copy(buf_bytes)
        lib = _ensure_bindings()
        n_out = lib.lora_channelizer_feed_cf(
            self._handle,
            in_buf, n_in,
            self._out_buf, self._out_buf_capacity,
        )
        if n_out == 0:
            return np.zeros(0, dtype=np.complex64)
        flat = np.frombuffer(self._out_buf, dtype=np.float32,
                              count=2 * n_out)
        return flat.view(np.complex64).copy()

    @property
    def samples_in(self) -> int:
        if not self._handle:
            return 0
        return _ensure_bindings().lora_channelizer_samples_in(self._handle)

    @property
    def samples_out(self) -> int:
        if not self._handle:
            return 0
        return _ensure_bindings().lora_channelizer_samples_out(self._handle)

    def close(self) -> None:
        if self._handle:
            _ensure_bindings().lora_channelizer_free(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
