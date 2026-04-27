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
    # v0.7.15: use c_void_p instead of POINTER(...) for IQ pointers.
    # Drops ~16 µs/call of ctypes coercion overhead per pointer arg.
    # See lora_native._ensure_lib for full discussion.
    lib.lora_channelizer_feed_cu8.restype = ctypes.c_size_t
    lib.lora_channelizer_feed_cu8.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,    # const uint8_t *cu8
        ctypes.c_size_t,
        ctypes.c_void_p,    # float *out
        ctypes.c_size_t,
    ]
    lib.lora_channelizer_feed_cf.restype = ctypes.c_size_t
    lib.lora_channelizer_feed_cf.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,    # const float *cf
        ctypes.c_size_t,
        ctypes.c_void_p,    # float *out
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
        self._out_buf_addr = 0  # cached ctypes.addressof(_out_buf)

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
            # v0.7.15: cache the buffer's address as a raw int so we
            # can pass it as c_void_p (zero per-call ctypes overhead)
            # instead of paying ~16µs/call for POINTER coercion.
            self._out_buf_addr = ctypes.addressof(self._out_buf)

    def feed_cu8(self, cu8: bytes) -> np.ndarray:
        """Feed cu8 bytes; return baseband float-pairs as a numpy
        complex64 array (decimated to bandwidth rate, mixed to DC).

        The returned array is a copy backed by Python memory and
        safe to pass to ``LoraDecoder.feed_baseband()`` (or to the
        BlindProbe).

        v0.7.15: zero-copy input. Pre-v0.7.15 this method called
        ``(ctypes.c_uint8 * len(cu8)).from_buffer_copy(cu8)`` which
        copied the entire input every call. With per-chunk inputs
        of ~10s of KB at 2.4 MS/s, that copy was meaningful CPU.
        ``c_char_p(cu8)`` is a zero-copy borrow of the bytes buffer
        (bytes are immutable so the pointer is stable for the life
        of the bytes object — i.e. for the duration of this call).
        """
        n_in = len(cu8) // 2
        if n_in == 0:
            return np.zeros(0, dtype=np.complex64)
        self._ensure_out_buf(n_in)

        lib = _ensure_bindings()
        # v0.7.15: c_char_p(cu8) is a zero-copy pointer to the bytes
        # buffer. self._out_buf_addr is a cached int address.
        n_out = lib.lora_channelizer_feed_cu8(
            self._handle,
            ctypes.c_char_p(cu8), n_in,
            self._out_buf_addr, self._out_buf_capacity,
        )
        if n_out == 0:
            return np.zeros(0, dtype=np.complex64)
        # Build numpy view of the C buffer's used prefix and copy out.
        # The .copy() IS necessary here: self._out_buf is reused by
        # the next call, so a view would dangle.
        flat = np.frombuffer(self._out_buf, dtype=np.float32,
                              count=2 * n_out)
        # Interleaved float -> complex64
        return flat.view(np.complex64).copy()

    def feed_cf(self, samples) -> np.ndarray:
        """Feed already-converted float samples (interleaved I/Q).

        ``samples`` may be a numpy float32/complex64 array, bytes, or
        any buffer-protocol object containing 2N floats.

        v0.7.15: zero-copy when ``samples`` is a numpy complex64 or
        float32 C-contiguous array (the common case).
        """
        if isinstance(samples, np.ndarray):
            if samples.dtype == np.complex64:
                if not samples.flags["C_CONTIGUOUS"]:
                    samples = np.ascontiguousarray(samples)
                n_in = samples.shape[0]
            elif samples.dtype == np.float32:
                if not samples.flags["C_CONTIGUOUS"]:
                    samples = np.ascontiguousarray(samples)
                n_in = samples.shape[0] // 2
            else:
                # Wrong dtype — fall back to bytes path.
                samples = samples.tobytes()
                n_in = len(samples) // 8
            if isinstance(samples, np.ndarray):
                if n_in == 0:
                    return np.zeros(0, dtype=np.complex64)
                self._ensure_out_buf(n_in)
                in_addr = samples.ctypes.data
                lib = _ensure_bindings()
                n_out = lib.lora_channelizer_feed_cf(
                    self._handle,
                    in_addr, n_in,
                    self._out_buf_addr, self._out_buf_capacity,
                )
                if n_out == 0:
                    return np.zeros(0, dtype=np.complex64)
                flat = np.frombuffer(self._out_buf, dtype=np.float32,
                                      count=2 * n_out)
                return flat.view(np.complex64).copy()

        # Bytes / array.array / other buffer-protocol objects.
        # c_char_p borrows the pointer (zero copy).
        if not isinstance(samples, (bytes, bytearray)):
            samples = bytes(samples)
        n_in = len(samples) // 8
        if n_in == 0:
            return np.zeros(0, dtype=np.complex64)
        self._ensure_out_buf(n_in)
        lib = _ensure_bindings()
        n_out = lib.lora_channelizer_feed_cf(
            self._handle,
            ctypes.c_char_p(samples), n_in,
            self._out_buf_addr, self._out_buf_capacity,
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
