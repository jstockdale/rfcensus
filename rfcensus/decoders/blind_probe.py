"""Multi-SF blind preamble probe — Python wrapper for lora_probe.c.

Used by the lazy pipeline at activate-event time to identify which
spreading factor(s) have a preamble in the lookback IQ. Replaces
the v0.7.x "spawn 5 decoders per slot, race them" approach with
"look once, spawn matching SF only".

Cost: ~50µs per probe scan for 5 SFs at oversample=2 (vs ~750µs
just to allocate-and-free the 5 decoders, plus the wall time for
those wrong-SF decoders to give up and be killed by racing).

Multi-system support: when two transmitters at the same slot
frequency are simultaneously active on different SFs, the probe
returns ALL detected SFs. The caller spawns full decoders for
each.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from rfcensus.decoders.lora_native import _lib_get


class _FftCpx(ctypes.Structure):
    _fields_ = [("r", ctypes.c_float), ("i", ctypes.c_float)]


class _ProbeResult(ctypes.Structure):
    _fields_ = [
        ("sf",          ctypes.c_uint32),
        ("peak_mag",    ctypes.c_float),
        ("noise_floor", ctypes.c_float),
        ("snr_db",      ctypes.c_float),
        ("peak_bin",    ctypes.c_uint16),
        ("detected",    ctypes.c_bool),
    ]


# Lazy ctypes binding — happens once on first BlindProbe construction.
# Module-import time is too early because lora_native may not yet have
# located + loaded the .so (deferred until first decoder construction).
_bindings_done = False


def _ensure_bindings() -> ctypes.CDLL:
    """Bind ctypes signatures on first use. Idempotent."""
    global _bindings_done
    lib = _lib_get()
    if _bindings_done:
        return lib
    lib.lora_probe_create.restype = ctypes.c_void_p
    lib.lora_probe_create.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_float,
    ]
    lib.lora_probe_destroy.argtypes = [ctypes.c_void_p]
    lib.lora_probe_scan.restype = ctypes.c_uint32
    # v0.7.15: c_void_p instead of POINTER(...) for the pointer args
    # — saves ~3 µs/call of ctypes type-coercion overhead. Same
    # pattern as the channelizer + decoder ctypes wrappers.
    lib.lora_probe_scan.argtypes = [
        ctypes.c_void_p,    # probe handle
        ctypes.c_void_p,    # const cf_t *iq
        ctypes.c_uint32,    # n_samples
        ctypes.c_void_p,    # ProbeResult *out
    ]
    _bindings_done = True
    return lib


@dataclass
class ProbeResult:
    """Per-SF result from BlindProbe.scan.

    Fields:
      sf: spreading factor this result is for
      snr_db: 20·log10(peak_mag / noise_floor) — relative measure
        of how strong the candidate preamble is vs surrounding bins
      peak_mag: linear magnitude of the strongest FFT bin. Comparable
        across scans within the same probe (same N, same FFT scaling)
        but NOT directly in dBFS — magnitudes scale with input
        amplitude AND with sqrt(N) due to the FFT. Use for relative
        adaptive-threshold comparison ("peak vs rolling background"),
        not as an absolute dBFS reading.
      noise_floor: linear magnitude of the mean non-peak bin
        (= probe's local noise estimate). Used to compute snr_db.
      peak_bin: which FFT bin had the peak (0..N-1). For a
        coherent preamble dechirped against the matching SF, this
        will cluster around 0 (or N-1 with phase-conjugate
        convention) for a perfectly-aligned chirp.
      detected: snr_db ≥ the probe's threshold. Cheap pre-computed
        for callers that only care about the binary decision.
    """
    sf: int
    snr_db: float
    peak_bin: int
    detected: bool
    peak_mag: float = 0.0
    noise_floor: float = 0.0


class BlindProbe:
    """Multi-SF preamble probe for one slot frequency.

    Constructed with a list of candidate SFs + oversampling factor.
    The caller feeds DECIMATED baseband samples (complex64 at BW
    rate) — typically the lookback IQ for a slot, after mixing
    down + decimating from the original sample rate.

    Thread-safe: no. Each pipeline thread should own its own probe.
    """

    def __init__(
        self,
        sfs: list[int],
        oversample: int = 2,
        snr_threshold_db: float = 10.0,
    ):
        if not 1 <= len(sfs) <= 8:
            raise ValueError(f"need 1..8 SFs, got {len(sfs)}")
        if oversample < 1:
            raise ValueError(f"oversample must be >= 1, got {oversample}")
        for sf in sfs:
            if not 6 <= sf <= 12:
                raise ValueError(f"SF must be 6..12, got {sf}")
        self._sfs = list(sfs)
        self._oversample = oversample
        self._snr_threshold_db = snr_threshold_db
        sf_arr = (ctypes.c_uint32 * len(sfs))(*sfs)
        lib = _ensure_bindings()
        self._handle = lib.lora_probe_create(
            sf_arr, len(sfs), oversample, snr_threshold_db,
        )
        if not self._handle:
            raise RuntimeError(
                f"lora_probe_create failed for sfs={sfs} oversample={oversample}"
            )
        # Pre-allocate result buffer to avoid per-call alloc.
        self._results = (_ProbeResult * len(sfs))()
        # v0.7.15: cache the address as an int for fast c_void_p pass.
        self._results_addr = ctypes.addressof(self._results)
        # Compute the largest N (SF11 oversample=2 = 4096). The probe
        # needs at least max_N samples to be able to test all SFs.
        self._max_N = max(((1 << sf) * oversample) for sf in sfs)

    @property
    def min_samples_required(self) -> int:
        """Minimum number of decimated baseband samples needed to
        scan all candidate SFs. Smaller buffers will return
        ``detected=False`` for SFs whose N exceeds the buffer."""
        return self._max_N

    def scan(self, baseband: np.ndarray) -> list[ProbeResult]:
        """Run the probe on a buffer of complex64 baseband samples.

        ``baseband`` must be a 1-D numpy array of dtype complex64 at
        bandwidth rate (typically 250 kHz for Meshtastic BW=250
        slots).

        Returns one ProbeResult per candidate SF, in the order
        passed at construction time. ``detected=True`` indicates the
        SNR exceeded the threshold.

        v0.7.15 perf note: this call wraps each result in a Python
        ``ProbeResult`` dataclass for backward compatibility. Hot
        callers (lazy_pipeline.py) should prefer :meth:`scan_inplace`
        which returns the raw C struct array (no allocation, no
        dataclass overhead). For 6 SFs that's ~6 µs/call saved.
        """
        self._scan_into_results(baseband)
        return [
            ProbeResult(
                sf=r.sf,
                snr_db=r.snr_db,
                peak_bin=r.peak_bin,
                detected=r.detected,
                peak_mag=r.peak_mag,
                noise_floor=r.noise_floor,
            )
            for r in self._results
        ]

    def scan_inplace(self, baseband: np.ndarray):
        """v0.7.15: zero-allocation fast-path scan.

        Identical to :meth:`scan` but returns the in-place ctypes
        results array (with .sf / .snr_db / .peak_bin / .detected /
        .peak_mag / .noise_floor fields readable directly). The
        returned object is a view into the probe's internal buffer
        — its contents are OVERWRITTEN by the next scan_inplace or
        scan call on this probe.

        Saves the ~6 µs/call dataclass-construction cost paid by
        :meth:`scan`. Use in hot loops; otherwise use :meth:`scan`.
        """
        self._scan_into_results(baseband)
        # The ctypes array is iterable — yields each Structure with
        # the field accessors. Caller can do ``for r in scan_inplace(bb):``.
        return self._results

    def _scan_into_results(self, baseband: np.ndarray) -> int:
        """Inner: run the C scan into self._results. Returns the
        number of SFs the C side marked as detected (== sum of
        results[i].detected). Callers that need ALL results should
        iterate ``self._results`` directly; this return value is a
        convenience for "anything detected?" queries.

        v0.7.15: zero-copy ctypes path. Replaces ``data_as(POINTER)``
        with a raw int (.ctypes.data) + c_void_p argtype, saving
        ~3 µs/call of ctypes coercion overhead.
        """
        if baseband.dtype != np.complex64:
            baseband = baseband.astype(np.complex64)
        if baseband.ndim != 1:
            raise ValueError(f"baseband must be 1-D, got {baseband.shape}")
        n = len(baseband)
        if not baseband.flags["C_CONTIGUOUS"]:
            baseband = np.ascontiguousarray(baseband)
        lib = _ensure_bindings()
        # baseband.ctypes.data is the raw int address of the array's
        # data buffer; passes through as c_void_p (per the v0.7.15
        # argtype switch above). self._results_addr is cached from
        # __init__.
        return lib.lora_probe_scan(
            self._handle, baseband.ctypes.data, n, self._results_addr
        )

    def detected_sfs(self, baseband: np.ndarray) -> list[int]:
        """Convenience: return list of SFs above threshold."""
        return [r.sf for r in self.scan(baseband) if r.detected]

    def close(self) -> None:
        if self._handle:
            lib = _ensure_bindings()
            lib.lora_probe_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def channelize_cu8_to_baseband(
    cu8: bytes,
    sample_rate_hz: int,
    bandwidth_hz: int,
    mix_freq_hz: int,
) -> np.ndarray:
    """Mix + decimate raw cu8 IQ down to baseband at BW rate.

    Used to prepare lookback IQ for the BlindProbe AND as the shared
    upstream channelization for SharedChannelGroup (multiple SF
    decoders consuming the same baseband stream).

    Implementation:
      1. Convert cu8 (uint8 I, uint8 Q) → complex64 centered at 0
      2. Multiply by exp(-j·2π·mix·t) to shift slot center to DC
      3. Apply a low-pass filter at bandwidth/2 (anti-aliasing)
      4. Decimate by integer ratio sample_rate / bandwidth

    The LPF matters: without it, energy outside ±BW/2 aliases into
    the decimated output. Empirical CRC pass rate on the real
    capture: 0/12 with no filter (excessive aliasing), 4/12 with
    naive decimation (the C decoder's linear-interp resampler has
    similar aliasing characteristics so the bandwidth rate matches),
    and 7/12 with proper LPF (matches the C decoder's mix+resamp
    on cu8 input). We use the proper LPF path.

    For non-integer sample_rate/bandwidth ratios (e.g. 2.4 MS/s
    / 250 kHz = 9.6) we round to the nearest integer. The 2-3%
    sample-rate mismatch is well within the LoRa decoder's CFO
    tolerance (~BW/4 = 62.5 kHz).
    """
    # cu8 → complex64
    arr = np.frombuffer(cu8, dtype=np.uint8)
    if len(arr) % 2 != 0:
        arr = arr[:-1]    # drop trailing odd byte (defensive)
    iq = arr.reshape(-1, 2).astype(np.float32) - 127.5
    z = (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
    # Normalize to roughly [-1, 1] so magnitudes match what the C
    # decoder sees on its cu8 path (it divides by 127.5).
    z /= 127.5

    # Mix down: multiply by exp(-j·2π·mix·t)
    n = len(z)
    if mix_freq_hz != 0:
        t = np.arange(n, dtype=np.float64) / sample_rate_hz
        mix = np.exp(-2j * np.pi * mix_freq_hz * t).astype(np.complex64)
        z = z * mix

    # Anti-alias LPF + decimate. Use a windowed-sinc FIR designed for
    # the decimation ratio. 64 taps is enough for ~50 dB stopband
    # rejection, which kills aliasing without over-filtering.
    decim = max(1, round(sample_rate_hz / bandwidth_hz))
    if decim > 1:
        # Hand-rolled windowed-sinc to avoid scipy dependency.
        #
        # Cutoff design: a LoRa chirp at BW=250 kHz spans the FULL
        # 250 kHz bandwidth. Cutting too tight at BW/2 = 125 kHz
        # loses chirp energy at the band edges and tanks recall
        # (we measured: tight LPF dropped CRC-ok from 7 to 2 on
        # the real capture). Use a wider cutoff with the LPF
        # primarily preventing OUT-OF-BAND noise from aliasing
        # into the slot, not in-band shaping.
        #
        # cutoff_norm = bandwidth / sample_rate gives Nyquist of
        # the decimated rate as the cutoff. Multiply by 0.95 so
        # the transition band starts JUST below Nyquist — minimal
        # in-band attenuation, but still rejects everything past
        # decimated Nyquist.
        n_taps = 64
        cutoff_norm = (bandwidth_hz / sample_rate_hz) * 0.95
        k = np.arange(n_taps) - (n_taps - 1) / 2
        h = np.sinc(2 * cutoff_norm * k)
        h *= np.hamming(n_taps)
        h /= h.sum()
        h = h.astype(np.float32)
        z_filt = np.convolve(z.real, h, mode="same") + \
                 1j * np.convolve(z.imag, h, mode="same")
        z = z_filt.astype(np.complex64)
    return z[::decim].copy()
