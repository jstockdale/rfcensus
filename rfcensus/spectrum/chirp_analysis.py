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
    # v0.6.8: SF classification via reference-dechirp (gr-lora_sdr style).
    # When this analysis was run against a baseband at a known suspected
    # bandwidth, classify_sf_dechirp populates these fields. They're
    # *the* canonical SF estimate going forward — slope-fit-based
    # estimate_sf_from_slope is unreliable on real (gap-free) LoRa.
    estimated_sf: int | None = None  # 7..12 or None
    sf_confidence: float = 0.0  # ratio best/second-best dechirp score
    sf_peak_concentration: float = 0.0  # absolute best dechirp score
    sf_scores: dict[int, float] | None = None  # SF → concentration


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


# ────────────────────────────────────────────────────────────────────
# v0.6.6: SF + variant classification helpers
# (moved from rfcensus.detectors.builtin.lora when the legacy
# wide-channel LoRa detector was removed in favor of LoraSurveyTask.)
#
# These are pure functions of (slope, bandwidth) and (sf, bandwidth)
# with no detector-state coupling. LoraSurveyTask uses them to enrich
# its DetectionEvents with SF and variant labels at full parity with
# the old detector's output. They live here in chirp_analysis because
# semantically they're "interpret chirp-analysis output" — an
# extension of analyze_chirps' result, not detector logic.
# ────────────────────────────────────────────────────────────────────


def estimate_sf_from_slope(
    *,
    slope_hz_per_sec: float,
    bandwidth_hz: int,
) -> int | None:
    """Estimate LoRa spreading factor from observed chirp slope.

    For a LoRa chirp: slope = BW² / 2^SF
    Solving for SF:    SF    = log2(BW² / slope)

    Returns integer SF rounded to nearest, clamped to [5, 12], or None
    if the slope is implausibly low/high (> 1 SF outside the valid
    range, which usually means the chirp analysis picked up something
    that isn't actually LoRa).

    .. deprecated:: 0.6.8
       This function assumes the slope was measured on a single complete
       chirp segment. It produces wrong answers on real LoRa traffic
       because real packets are *contiguous* chirps with no inter-chirp
       gaps — the slope-fit segment finder either treats the whole packet
       as one segment (sawtooth instantaneous frequency, fit fails) or
       locks onto fragments (fit slope is meaningless). The caller should
       use ``classify_sf_dechirp`` instead, which models real chirp
       structure correctly via reference-dechirp + FFT (the standard
       gr-lora_sdr / NELoRa approach). This helper is kept for backward
       compatibility and for synthetic test cases where each chirp IS a
       separate segment with gaps.
    """
    import math

    # Reject zero-or-NaN slope and zero-or-negative BW. Negative slope
    # is fine — LoRa down-chirps are valid; we take magnitude below.
    if slope_hz_per_sec == 0 or bandwidth_hz <= 0:
        return None
    # Take absolute value — up-chirps and down-chirps have opposite
    # signs; we only care about magnitude.
    slope_abs = abs(slope_hz_per_sec)
    try:
        sf_float = math.log2((bandwidth_hz ** 2) / slope_abs)
    except (ValueError, ZeroDivisionError):
        return None
    sf_int = round(sf_float)
    if sf_int < 4 or sf_int > 13:
        # Implausibly outside LoRa's valid SF range
        return None
    # Clamp to the canonical range
    return max(5, min(12, sf_int))


# ────────────────────────────────────────────────────────────────────
# v0.6.8: reference-dechirp SF classifier (replaces slope-fit method)
# ────────────────────────────────────────────────────────────────────


def make_downchirp(
    *, sample_rate: int, bandwidth_hz: int, sf: int,
) -> tuple[np.ndarray, int]:
    """Build a base LoRa downchirp at the given SF/BW/sample rate.

    Returns ``(downchirp_samples, chirp_length_samples)``. The downchirp
    is the complex conjugate of the canonical upchirp, with
    instantaneous frequency sweeping linearly from +BW/2 down to -BW/2
    over one chirp period of 2^SF / BW seconds.

    Used as the reference signal for classify_sf_dechirp. Multiplying a
    received upchirp by this downchirp produces a single tone whose
    frequency identifies the chirp's cyclic shift (ie its data symbol);
    multiplying by a downchirp at the WRONG SF produces a smeared
    spectrum with no clear peak.
    """
    if sample_rate <= 0 or bandwidth_hz <= 0 or sf < 5 or sf > 13:
        raise ValueError(
            f"invalid downchirp params: sr={sample_rate}, "
            f"bw={bandwidth_hz}, sf={sf}"
        )
    chirp_dur_s = (2 ** sf) / bandwidth_hz
    chirp_samps = int(round(sample_rate * chirp_dur_s))
    if chirp_samps < 8:
        raise ValueError(
            f"chirp too short ({chirp_samps} samples) — "
            f"sample rate {sample_rate} insufficient for SF{sf}/{bandwidth_hz}"
        )
    slope = bandwidth_hz / chirp_dur_s
    t = np.arange(chirp_samps, dtype=np.float64) / sample_rate
    # Downchirp: f(t) = +BW/2 - slope*t. Phase = 2π ∫f dt.
    inst_freq = bandwidth_hz / 2.0 - slope * t
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / sample_rate
    downchirp = np.exp(1j * phase).astype(np.complex64)
    return downchirp, chirp_samps


# Minimum acceptable concentration score (peak FFT bin / total energy)
# for an SF classification to be considered valid. Tuning notes:
#   • Pure white noise gives concentration ~1/chirp_samps + statistical
#     fluctuation. At chirp_samps=128 (SF7/250kHz at 250kHz rate),
#     baseline is ~1/128 = 0.0078, but in practice noise hits 0.040-
#     0.045 across multiple seeds (random alignment lets one bin
#     accumulate above the average).
#   • Weakest legitimate true classification observed: 0.0515 (SF7 with
#     random-symbol data at 500 kHz). Most legitimate hits are ≥0.075.
#   • 0.050 sits on the boundary; bump to 0.052 to give margin against
#     pure noise while still accepting the weakest legitimate cases.
_MIN_SF_CONCENTRATION = 0.052

# Minimum confidence (best score / second-best score) to accept the
# top-scoring SF as the answer. Tuning notes:
#   • Lowest legitimate confidence with random data symbols: 1.33
#     (SF9 at 500 kHz, BW = sample rate so each chirp is just 1 cycle).
#   • Pure noise gives ~1.75 which fails the concentration floor anyway,
#     so the confidence threshold is not the noise gate.
#   • Wrong-BW scenarios (e.g. 250 kHz signal seen through 125 kHz
#     template) can produce confidence as high as 5.9 with a wrong SF —
#     this gate cannot distinguish those. The CALLER (survey_iq_window)
#     is responsible for choosing the right BW template by comparing
#     concentration scores across templates.
#   • 1.20 is safely below the legitimate floor (1.33) and well above
#     ambient noise variance.
_MIN_SF_CONFIDENCE = 1.20

# SF range to test. LoRaWAN and Meshtastic both use SF7..SF12; SF5/6
# exist in newer LoRa versions but are rare and Meshtastic doesn't
# support them. Trim if you need to broaden.
_SF_CANDIDATES: tuple[int, ...] = (7, 8, 9, 10, 11, 12)


def classify_sf_dechirp(
    samples: np.ndarray,
    sample_rate: int,
    bandwidth_hz: int,
) -> tuple[int | None, float, float, dict[int, float]]:
    """Classify LoRa spreading factor by reference-dechirp + FFT.

    The standard gr-lora_sdr / NELoRa technique. For each candidate SF:

      1. Build a reference downchirp at this SF/BW
      2. Slice the active region of `samples` into chirp-length chunks
      3. Multiply each chunk by the downchirp, FFT it, and measure
         peak energy / total energy ("concentration")
      4. Average concentration across all chunks → score for this SF

    The correct SF concentrates the multiplied signal into a single FFT
    bin (concentration → 1.0 in the noiseless limit). A wrong SF
    leaves residual chirping in the multiplied signal, smearing energy
    across many bins (concentration ~ 1/N).

    Returns ``(best_sf, confidence, peak_concentration, scores)``:
      • best_sf — int in [7, 12] or None if no SF passed both gates
      • confidence — best_score / second_best_score; ≥1.15 to accept
      • peak_concentration — best_score itself; ≥0.05 to accept
      • scores — dict mapping each candidate SF to its concentration

    `samples` should be a complex64/complex128 IQ array, ALREADY
    digitally down-converted to baseband at the suspected channel
    center, with `sample_rate` ≥ `bandwidth_hz` and ideally ≈ `bandwidth_hz`
    (Nyquist for the LoRa bandwidth — the dechirp doesn't gain anything
    from oversampling).

    Robust to:
      • Real LoRa packet structure (preamble + cyclic-shifted data
        symbols — the FFT peak just moves, total concentration is
        unchanged)
      • Low SNR — validated to -10 dB with 100% accuracy on synthetic
        data; gracefully degrades by falling below the confidence
        threshold (returning None) rather than mislabeling
      • Active-region masking — only processes above-noise samples,
        avoiding false correlation with noise tails
      • Off-by-one chirp alignment — chunks are aligned to active-region
        boundaries, not to absolute symbol boundaries (which we don't
        know without preamble detection); this works because LoRa's
        cyclic-shift property means any contiguous chirp-length window
        of an upchirp dechirps to a tone at *some* frequency (the
        symbol value)

    Edge case worth knowing about:
      • Pure-preamble inputs (every chirp is the same all-zero-symbol
        upchirp) reduce discrimination between adjacent SFs because
        the SF/2 partial-chirp window of an SF+1 chirp dechirps about
        as well as the SF chirp itself. Real packets carry varied
        payload symbols which break this degeneracy. If you only ever
        feed in pure preambles you may see SF8/SF7 confusion at high
        BW; this isn't a realistic scenario in the field.
    """
    if samples.size < 1024:
        return None, 0.0, 0.0, {}

    # 1. Find the active region (above-noise samples) so we don't
    #    score on long noise tails. The dechirp would otherwise add a
    #    big chunk of low-concentration chunks to the average and
    #    crush the score for the correct SF.
    amplitude = np.abs(samples)
    # Median-based threshold is robust to a few stray spikes; LoRa
    # bursts have a much higher amplitude than the noise floor so
    # 0.5× median sits comfortably between them when a burst is
    # present and approximately AT the typical sample when it's
    # noise-only.
    threshold = max(1e-3, float(np.median(amplitude)) * 0.5)
    mask = amplitude > threshold
    if mask.sum() < 256:
        return None, 0.0, 0.0, {}
    active_idx = np.flatnonzero(mask)
    first, last = int(active_idx[0]), int(active_idx[-1])
    sig = samples[first : last + 1]
    if sig.size < 1024:
        return None, 0.0, 0.0, {}

    # 2. Score each candidate SF
    scores: dict[int, float] = {}
    for sf in _SF_CANDIDATES:
        try:
            downchirp, chirp_samps = make_downchirp(
                sample_rate=sample_rate,
                bandwidth_hz=bandwidth_hz,
                sf=sf,
            )
        except ValueError:
            # SF/sample-rate combination doesn't yield enough samples;
            # skip rather than fail
            continue
        # Need at least 2 chirps' worth of samples to score
        n_chunks = sig.size // chirp_samps
        if n_chunks < 2:
            continue
        trimmed = sig[: n_chunks * chirp_samps]
        chunks = trimmed.reshape(n_chunks, chirp_samps)
        # Multiply each chunk by the downchirp (broadcasts over rows)
        dechirped = chunks * downchirp[np.newaxis, :]
        # FFT → power → peak / total
        spectra = np.fft.fft(dechirped, axis=1)
        magsq = np.abs(spectra) ** 2
        peak_energy = magsq.max(axis=1)
        total_energy = magsq.sum(axis=1)
        # Avoid divide-by-zero for pure-zero chunks (shouldn't happen
        # given the active-region mask, but defensive)
        ratios = np.where(
            total_energy > 0, peak_energy / total_energy, 0.0,
        )
        scores[sf] = float(np.mean(ratios))

    if not scores:
        return None, 0.0, 0.0, {}

    # 3. Pick the SF with highest concentration
    sorted_sfs = sorted(scores.items(), key=lambda kv: -kv[1])
    best_sf, best_score = sorted_sfs[0]

    if len(sorted_sfs) > 1:
        second_score = sorted_sfs[1][1]
        # Avoid divide-by-zero (shouldn't happen — even pure noise
        # gives ~1/chirp_samps for every SF — but defensive)
        confidence = (
            best_score / second_score if second_score > 0 else 10.0
        )
    else:
        # Only one SF made it into scoring; can't compute a ratio.
        # Treat as low-confidence and reject below.
        confidence = 1.0

    # 4. Apply both gates. Either failing → return None for SF, but
    #    still report the scores so callers can log diagnostics.
    if best_score < _MIN_SF_CONCENTRATION:
        return None, confidence, best_score, scores
    if confidence < _MIN_SF_CONFIDENCE:
        return None, confidence, best_score, scores
    return best_sf, confidence, best_score, scores


def label_variant(*, sf: int, bandwidth_hz: int) -> str | None:
    """Map (SF, bandwidth) to a human-readable variant label where one
    is recognizable. Otherwise None (detection is still reported as
    generic LoRa with the numeric SF in metadata).

    Meshtastic defaults (US region, LoRaPHY_MT):
      • ShortTurbo     SF7  / 500 kHz
      • ShortFast      SF7  / 250 kHz
      • ShortSlow      SF8  / 250 kHz
      • MediumFast     SF9  / 250 kHz
      • MediumSlow     SF10 / 250 kHz
      • LongFast       SF11 / 250 kHz   (default / most common)
      • LongModerate   SF11 / 125 kHz
      • LongSlow       SF12 / 125 kHz

    LoRaWAN US uplink SF7-10/125kHz is common; the bands overlap with
    Meshtastic at (SF7-10, 125kHz). We prefer the Meshtastic label when
    the SF matches a unique default (SF11/250 is unambiguously LongFast
    Meshtastic; SF9/250 is unambiguously MediumFast).
    """
    bw_khz = bandwidth_hz // 1000
    # Meshtastic-distinctive combinations (bw=250 kHz):
    if bw_khz == 250:
        if sf == 7:
            return "meshtastic_short_fast"
        if sf == 8:
            return "meshtastic_short_slow"
        if sf == 9:
            return "meshtastic_medium_fast"
        if sf == 10:
            return "meshtastic_medium_slow"
        if sf == 11:
            return "meshtastic_long_fast"
    if bw_khz == 500 and sf == 7:
        return "meshtastic_short_turbo"
    if bw_khz == 125:
        if sf == 11:
            return "meshtastic_long_moderate_or_lorawan"
        if sf == 12:
            return "meshtastic_long_slow_or_lorawan"
        # Lower SF at 125 kHz is most commonly LoRaWAN
        if 7 <= sf <= 10:
            return f"lorawan_sf{sf}"
    return None
