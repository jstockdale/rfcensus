"""Chirp detection via instantaneous frequency analysis.

LoRa (and other chirp-spread-spectrum protocols) sweeps the
instantaneous frequency linearly through time. A "chirp" is a tone
whose frequency moves in a straight line in the time-frequency plane.

Our detector:

1. Compute instantaneous phase: `angle(samples)`
2. Compute instantaneous frequency: `diff(unwrap(phase)) * sample_rate / (2π)`
3. Look for segments where inst. frequency changes approximately linearly
4. Return confidence [0, 1] that chirp-like behavior is present

v0.5.42: also computes refinement metadata that's useful downstream:

  • refined_center_offset_hz — best estimate of where the channel
    center actually sits, relative to the IQ capture's baseband
    (caller adds capture_freq_hz to recover absolute frequency).
    Cross-validated by FFT centroid + chirp-intercept methods.
  • snr_db — signal-to-noise ratio in dB, comparing chirp-active
    samples to gap samples.
  • burst_total_duration_s, capture_duration_s, duty_cycle —
    how much of the listening window the transmitter was active.

Not a full LoRa demodulator — doesn't decode symbols, doesn't decode
payload bytes. Just "this looks chirp-shaped at this center freq with
this SNR," which is enough confirmation that a heuristic LoRa
suspicion is real, AND gives operators metadata to label the
detection accurately in reports.

Cheap: ~2-3 ms of numpy per 500 ms of IQ at 1 MHz sample rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChirpAnalysis:
    chirp_confidence: float  # 0..1
    num_chirp_segments: int
    mean_segment_length_samples: float
    mean_slope_hz_per_sec: float
    reasoning: str = ""
    # v0.5.42: refined frequency estimation. None when the analyzer
    # couldn't compute (no chirps detected, insufficient samples).
    # Caller adds this offset to the capture's tuned center frequency
    # to obtain the absolute refined center of the signal.
    refined_center_offset_hz: float | None = None
    frequency_estimate_method: str | None = None  # "fft_centroid" /
    # "chirp_mean" / "agreement" (both methods agreed) / "disagreement"
    # (methods diverged > 5 kHz; use fft_centroid as canonical).
    frequency_uncertainty_hz: float | None = None
    fft_centroid_offset_hz: float | None = None  # raw FFT method result
    chirp_centroid_offset_hz: float | None = None  # raw chirp method result
    # v0.5.42: SNR — chirp signal vs gap noise, in dB. Useful for
    # discriminating local vs distant transmitters and for confidence.
    snr_db: float | None = None
    signal_power_dbfs: float | None = None
    noise_power_dbfs: float | None = None
    # v0.5.42: burst timing — total active samples / capture samples.
    # Lets reports show e.g. "high duty cycle gateway" vs "occasional
    # sensor."
    burst_total_duration_s: float | None = None
    capture_duration_s: float | None = None
    duty_cycle: float | None = None  # 0.0–1.0


def analyze_chirps(
    samples: np.ndarray,
    sample_rate: int,
    *,
    min_segment_samples: int = 64,
    linearity_threshold: float = 0.85,
) -> ChirpAnalysis:
    """Analyze an IQ chunk for chirp-like patterns.

    v0.5.42: in addition to the chirp-confidence/slope output, this
    populates frequency-refinement, SNR, and burst-timing metadata.
    All derived from the same inst_freq + amplitude analysis we
    already do — minimal extra cost.

    The IQ should be at baseband (DDC'd) for accurate frequency
    refinement; offsets are returned relative to 0 Hz of the input.
    For a wideband non-baseband capture (e.g., the 2 MHz IQ window
    used for in-window survey), the FFT centroid will pick up the
    dominant channel within the window — useful when there's only
    one strong signal present.
    """
    capture_duration_s = float(samples.size) / float(sample_rate) if sample_rate > 0 else 0.0

    if samples.size < min_segment_samples * 2:
        return ChirpAnalysis(
            chirp_confidence=0.0,
            num_chirp_segments=0,
            mean_segment_length_samples=0.0,
            mean_slope_hz_per_sec=0.0,
            reasoning="insufficient samples",
            capture_duration_s=capture_duration_s,
        )

    phase = np.angle(samples)
    dphase = np.diff(np.unwrap(phase))
    inst_freq = dphase * sample_rate / (2 * np.pi)

    amplitude = np.abs(samples[1:])
    amp_threshold = max(1e-3, float(np.median(amplitude)) * 0.5)
    mask = amplitude > amp_threshold

    # Vectorized segment finder (was a Python loop pre-v0.5.42 — slow
    # on 1M+ samples). Find rising/falling edges via diff on the
    # boolean mask cast to int8. Pad with a leading/trailing False so
    # boundaries at the start/end are detected too.
    mask_i8 = mask.astype(np.int8, copy=False)
    padded = np.concatenate(([0], mask_i8, [0]))
    edges = np.diff(padded.astype(np.int16))
    rising = np.flatnonzero(edges == 1)
    falling = np.flatnonzero(edges == -1)
    # rising[k] = index where mask becomes True, falling[k] = first
    # index where mask becomes False after rising[k]. Both arrays
    # have the same length by construction.
    segments: list[tuple[int, int]] = [
        (int(s), int(e))
        for s, e in zip(rising, falling)
        if e - s >= min_segment_samples
    ]

    if not segments:
        return ChirpAnalysis(
            chirp_confidence=0.0,
            num_chirp_segments=0,
            mean_segment_length_samples=0.0,
            mean_slope_hz_per_sec=0.0,
            reasoning="no above-noise segments",
            capture_duration_s=capture_duration_s,
        )

    chirp_segments = 0
    slopes: list[float] = []
    lengths: list[int] = []
    chirp_segment_freqs: list[np.ndarray] = []  # for centroid

    for start, end in segments:
        freqs = inst_freq[start:end]
        if freqs.size < min_segment_samples:
            continue
        freqs = _median_filter(freqs, 5)
        t = np.arange(freqs.size, dtype=np.float32)
        a, b = np.polyfit(t, freqs, 1)
        predicted = a * t + b
        ss_res = float(np.sum((freqs - predicted) ** 2))
        ss_tot = float(np.sum((freqs - np.mean(freqs)) ** 2))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        slope_hz_per_sec = float(a * sample_rate)
        freq_span = abs(slope_hz_per_sec) * (freqs.size / sample_rate)
        if r_squared > linearity_threshold and freq_span > 5_000:
            chirp_segments += 1
            slopes.append(slope_hz_per_sec)
            lengths.append(freqs.size)
            chirp_segment_freqs.append(freqs)

    # v0.5.42: compute SNR and duty cycle from segments regardless of
    # whether they were "chirp" segments — even non-linear above-floor
    # activity counts toward burst time and signal power.
    snr_db, sig_db, noise_db = _compute_snr(samples, mask)
    burst_total_samples = int(np.sum(mask))
    burst_total_duration_s = burst_total_samples / sample_rate if sample_rate > 0 else 0.0
    duty_cycle = (
        burst_total_samples / mask.size if mask.size > 0 else 0.0
    )

    if chirp_segments == 0:
        return ChirpAnalysis(
            chirp_confidence=0.0,
            num_chirp_segments=0,
            mean_segment_length_samples=0.0,
            mean_slope_hz_per_sec=0.0,
            reasoning=f"{len(segments)} candidate segments, none were chirp-linear",
            snr_db=snr_db,
            signal_power_dbfs=sig_db,
            noise_power_dbfs=noise_db,
            burst_total_duration_s=burst_total_duration_s,
            capture_duration_s=capture_duration_s,
            duty_cycle=duty_cycle,
        )

    # v0.5.42: refined center frequency estimation
    refined_offset, method, uncertainty, fft_off, chirp_off = _refine_center(
        samples=samples,
        sample_rate=sample_rate,
        chirp_segment_freqs=chirp_segment_freqs,
    )

    mean_len = float(np.mean(lengths))
    mean_slope = float(np.mean(np.abs(slopes)))
    confidence = min(1.0, 0.4 + 0.15 * chirp_segments)
    return ChirpAnalysis(
        chirp_confidence=confidence,
        num_chirp_segments=chirp_segments,
        mean_segment_length_samples=mean_len,
        mean_slope_hz_per_sec=mean_slope,
        reasoning=(
            f"{chirp_segments} linear chirp segment(s), mean slope {mean_slope/1e3:.1f} kHz/s"
        ),
        refined_center_offset_hz=refined_offset,
        frequency_estimate_method=method,
        frequency_uncertainty_hz=uncertainty,
        fft_centroid_offset_hz=fft_off,
        chirp_centroid_offset_hz=chirp_off,
        snr_db=snr_db,
        signal_power_dbfs=sig_db,
        noise_power_dbfs=noise_db,
        burst_total_duration_s=burst_total_duration_s,
        capture_duration_s=capture_duration_s,
        duty_cycle=duty_cycle,
    )


def _median_filter(arr: np.ndarray, kernel: int) -> np.ndarray:
    """Median filter using scipy when available, Python fallback otherwise.

    The Python fallback is slow (O(N*kernel) per call); scipy.signal.medfilt
    is O(N log kernel) and vectorized. Pre-v0.5.42 we used the Python
    version unconditionally which made analyze_chirps O(seconds) on
    1M-sample inputs.
    """
    if kernel <= 1:
        return arr
    try:
        from scipy.signal import medfilt
        return medfilt(arr, kernel_size=kernel)
    except ImportError:
        pass
    half = kernel // 2
    padded = np.pad(arr, half, mode="edge")
    out = np.empty_like(arr)
    for i in range(arr.size):
        out[i] = np.median(padded[i : i + kernel])
    return out


# ---------------------------------------------------------------------
# v0.5.42: refinement helpers
# ---------------------------------------------------------------------


def _compute_snr(
    samples: np.ndarray, active_mask: np.ndarray
) -> tuple[float | None, float | None, float | None]:
    """SNR in dB, signal_power in dBFS, noise_power in dBFS.

    Signal power: median |samples[1:]|² where active_mask is True.
    Noise power:  median |samples[1:]|² where active_mask is False.

    Uses median (not mean) so a single energetic spike doesn't pull
    the noise floor up. dBFS = 10·log10(power) — relative to a
    full-scale unit-magnitude IQ sample.

    Returns (None, None, None) if either set is empty.
    """
    if active_mask.size == 0 or samples.size < 2:
        return None, None, None
    amp = np.abs(samples[1:])
    amp_sq = (amp ** 2).astype(np.float64)
    sig = amp_sq[active_mask]
    noise = amp_sq[~active_mask]
    if sig.size == 0 or noise.size == 0:
        return None, None, None
    sig_p = float(np.median(sig))
    noise_p = float(np.median(noise))
    if sig_p <= 0 or noise_p <= 0:
        return None, None, None
    sig_dbfs = 10.0 * np.log10(sig_p)
    noise_dbfs = 10.0 * np.log10(noise_p)
    snr = sig_dbfs - noise_dbfs
    return float(snr), float(sig_dbfs), float(noise_dbfs)


def _refine_center(
    *,
    samples: np.ndarray,
    sample_rate: int,
    chirp_segment_freqs: list[np.ndarray],
    fft_size: int = 4096,
    agreement_threshold_hz: float = 5_000.0,
) -> tuple[float | None, str | None, float | None, float | None, float | None]:
    """Estimate the channel's true center frequency in two independent
    ways and reconcile them.

    Method A — FFT centroid:
      Power-weighted mean frequency of the FFT of the IQ. For a
      symmetric chirp signal centered at the channel center, this
      converges to the center. Robust to amplitude variation; biased
      slightly by any nearby out-of-channel energy.

    Method B — chirp-segment frequency mean:
      For each chirp segment, the mean of inst_freq is the segment's
      midpoint frequency. For a complete chirp from -BW/2 to +BW/2,
      that midpoint = channel center. For partial chirps, the mean
      is biased depending on alignment, but averaging across many
      partial chirps with random alignments converges to the center.

    Returns (refined_offset, method, uncertainty, fft_offset, chirp_offset).
    Method is one of: "agreement" (both methods within
    agreement_threshold_hz), "disagreement" (diverged; we trust FFT),
    or "fft_centroid" / "chirp_mean" if only one is available.

    If both methods are unavailable, returns (None, None, None, None, None).
    """
    fft_off = _fft_centroid_offset(samples, sample_rate, fft_size=fft_size)

    chirp_off = None
    if chirp_segment_freqs:
        # Concatenate all chirp segment freqs and take a single mean.
        # Per-segment-then-average would weight short segments equally
        # to long ones; concat-then-mean weights by sample count, which
        # is what we want since longer segments are more informative.
        all_freqs = np.concatenate(chirp_segment_freqs)
        chirp_off = float(np.mean(all_freqs))

    if fft_off is None and chirp_off is None:
        return None, None, None, None, None
    if fft_off is None:
        return chirp_off, "chirp_mean", _estimate_uncertainty_hz(
            sample_rate, chirp_segment_freqs
        ), None, chirp_off
    if chirp_off is None:
        return fft_off, "fft_centroid", _fft_uncertainty_hz(
            sample_rate, fft_size
        ), fft_off, None

    # Both methods present — reconcile
    diff = abs(fft_off - chirp_off)
    if diff <= agreement_threshold_hz:
        # Methods agree within threshold; report the average and a
        # tighter uncertainty (combination of both estimates).
        avg = (fft_off + chirp_off) / 2.0
        # Conservative uncertainty: the larger of the two individual
        # uncertainties, or half the disagreement, whichever is larger.
        unc_chirp = _estimate_uncertainty_hz(sample_rate, chirp_segment_freqs)
        unc_fft = _fft_uncertainty_hz(sample_rate, fft_size)
        unc = max(unc_chirp or 0.0, unc_fft or 0.0, diff / 2.0)
        return float(avg), "agreement", float(unc), fft_off, chirp_off
    else:
        # Methods disagree — prefer FFT centroid (more robust to
        # partial-chirp alignment bias), but flag the discrepancy
        # in metadata so operators see the disagreement.
        return (
            float(fft_off),
            "disagreement",
            float(diff),  # uncertainty = the disagreement itself
            fft_off,
            chirp_off,
        )


def _fft_centroid_offset(
    samples: np.ndarray, sample_rate: int, *, fft_size: int = 4096
) -> float | None:
    """Power-weighted FFT frequency centroid.

    Uses scipy.signal.welch (averaged periodograms across overlapping
    windows) instead of a single FFT. Welch's method is essential here
    because a chirp signal with chirp duration > FFT window will look
    asymmetric within a single FFT (the window catches only part of
    the sweep), pulling the centroid hundreds of kHz off the true
    center. Averaging across many windows whose start positions are
    randomized within the chirp cycle washes out that bias.

    Returns None if the input is too short or has no usable power.
    """
    if samples.size < fft_size or sample_rate <= 0:
        return None
    try:
        from scipy.signal import welch
    except ImportError:
        return None

    # Welch on complex IQ: return_onesided=False so we get both
    # negative and positive frequencies (we're not real-valued audio).
    # nperseg defines the per-window FFT size; noverlap=50% is the
    # standard choice. With 1.2M samples and nperseg=4096, we get
    # ~580 windows — plenty for averaging.
    freqs, psd = welch(
        samples, fs=sample_rate,
        nperseg=fft_size, noverlap=fft_size // 2,
        return_onesided=False, scaling="spectrum",
    )
    # welch returns frequencies in [0, sr/2, -sr/2, 0) ordering.
    # Use fftshift to get a monotonic axis for clarity.
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    total_power = float(np.sum(psd))
    if total_power <= 0:
        return None
    centroid = float(np.sum(freqs * psd) / total_power)
    return centroid


def _fft_uncertainty_hz(sample_rate: int, fft_size: int) -> float | None:
    """FFT bin width in Hz — coarse upper bound on centroid uncertainty.

    Real centroid uncertainty for a clean signal is much smaller
    (sub-bin via centroid interpolation), but this is a defensible
    conservative bound for reports.
    """
    if sample_rate <= 0 or fft_size <= 0:
        return None
    return float(sample_rate) / float(fft_size)


def _estimate_uncertainty_hz(
    sample_rate: int, chirp_segment_freqs: list[np.ndarray]
) -> float | None:
    """Stddev of per-segment mean frequencies — gives a sense of how
    consistent the chirp-mean estimator is across segments.

    With only 1-2 segments this is poorly defined; use FFT bin width
    as a fallback.
    """
    if not chirp_segment_freqs:
        return None
    if len(chirp_segment_freqs) < 2:
        # Single-segment estimate has no scatter info — fall back to
        # a sample-based estimate (the segment's own freq stddev,
        # divided by sqrt(N) for a mean stddev).
        seg = chirp_segment_freqs[0]
        return float(np.std(seg) / np.sqrt(seg.size))
    per_segment_means = [float(np.mean(s)) for s in chirp_segment_freqs]
    return float(np.std(per_segment_means))
