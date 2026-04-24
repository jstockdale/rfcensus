"""v0.5.42: in-window opportunistic survey.

When the confirmation task captures 2.4 Msps of IQ to confirm queued
LoRa detections, that capture covers ~2 MHz around the tuner center —
roughly 8 LoRaWAN US 250 kHz channels or 16 LoRaWAN US 125 kHz
channels. The aggregator only knew about the channels that lit up
during rtl_power scanning; many real LoRa channels are too sparse
(one packet per few minutes) to reliably trigger the bin-based
detector.

This module surveys the wideband IQ for additional chirp activity
that the aggregator missed:

  1. Compute the power spectrum (Welch's method, ~10 kHz resolution)
  2. Find peaks above the noise floor that are wide enough to be
     LoRa-template channels (≥ 100 kHz wide, sustained over the
     capture)
  3. For each candidate, DDC to baseband and run chirp analysis
  4. If chirps confirmed, emit a synthesized DetectionEvent flagged
     with discovery_method="in_window_iq_survey"

Caveats
=======

Surveys can produce false positives: a wideband non-LoRa signal
(WiFi, cordless phone, broadcast spillover) might span 100+ kHz and
look like a candidate. The chirp analyzer's R² > 0.85 linearity
threshold filters most of these — only chirps with linear inst-freq
sweeps pass. Required minimum 6 dB SNR above local noise floor adds
another safety margin.

We DON'T survey the entire 2 MHz — only the regions outside the
already-confirmed clusters (no point re-detecting what we just
verified). Configurable via `exclude_freqs_hz`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rfcensus.spectrum.chirp_analysis import ChirpAnalysis, analyze_chirps
from rfcensus.tools.dsp import digital_downconvert
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SurveyHit:
    """A LoRa-like signal found by the in-window survey."""

    freq_hz: int
    bandwidth_hz: int  # one of 125k / 250k / 500k
    chirp_analysis: ChirpAnalysis
    snr_db: float


# Standard LoRa template widths to check at each candidate band.
# Order matters: we try widest first so a real 500 kHz signal isn't
# mis-identified as a 125 kHz subsegment.
SURVEY_TEMPLATES_HZ: tuple[int, ...] = (500_000, 250_000, 125_000)

# Minimum SNR (in dB above local noise floor) for a peak to be worth
# DDC + chirp analysis. 6 dB = 4× signal-to-noise power ratio, weak
# but detectable.
DEFAULT_SURVEY_SNR_THRESHOLD_DB = 6.0

# Don't survey within this distance of an excluded center frequency
# (the confirmation task's primary targets). Avoids re-detecting what
# we just confirmed.
DEFAULT_EXCLUSION_RADIUS_HZ = 600_000  # 2× max template width

# Resolution of the survey power-spectrum analysis. 10 kHz bins are
# narrow enough to resolve adjacent LoRa channels, wide enough that
# the per-bin noise variance is small. With Welch averaging this is
# robust.
SURVEY_PSD_BIN_HZ = 10_000


def survey_iq_window(
    samples: np.ndarray,
    *,
    sample_rate: int,
    capture_center_hz: int,
    exclude_freqs_hz: list[int] | None = None,
    snr_threshold_db: float = DEFAULT_SURVEY_SNR_THRESHOLD_DB,
    exclusion_radius_hz: int = DEFAULT_EXCLUSION_RADIUS_HZ,
) -> list[SurveyHit]:
    """Look for LoRa-like signals in a wideband IQ capture beyond the
    already-known target frequencies.

    Returns a list of confirmed SurveyHit objects, each representing a
    new LoRa-family detection that the bin-based aggregator missed.
    Empty list if nothing new found (the common case in quiet bands).

    `exclude_freqs_hz` lists absolute frequencies to skip — typically
    the confirmation task's primary targets, since we already have IQ
    for those.
    """
    if samples.size < sample_rate // 4:  # need at least ~250 ms
        return []
    exclude_freqs_hz = exclude_freqs_hz or []

    # Step 1: power spectrum
    psd_freqs, psd_power = _compute_psd(samples, sample_rate)
    if psd_freqs is None:
        return []

    # Step 2: noise floor estimate (median of the lower half of the PSD,
    # log-scale) — robust against a few strong peaks pulling the mean.
    noise_floor_lin = float(np.median(np.sort(psd_power)[: psd_power.size // 2]))
    noise_floor_db = 10.0 * np.log10(max(noise_floor_lin, 1e-20))
    snr_threshold_lin = noise_floor_lin * (10.0 ** (snr_threshold_db / 10.0))

    # Step 3: find candidate bands. Strategy: first find the strongest
    # peak above the noise floor, then define its width as the
    # contiguous run that's within 10 dB of the peak. This avoids
    # spectral-leakage contamination that a simple "above-noise"
    # threshold would produce (e.g. a 500 kHz chirp leaks energy
    # hundreds of kHz beyond its nominal BW at the -40 dB level).
    #
    # Repeat: mask out the found peak's region, then search for the
    # next-strongest peak in what remains, until no peak exceeds the
    # base SNR threshold.
    candidates: list[tuple[float, float, float]] = []  # (low, high, peak_power_lin)
    remaining_power = psd_power.copy()
    peak_db_threshold = 10.0  # edges at -10 dB from peak
    for _ in range(12):  # cap iterations — realistic max LoRa channels in 2 MHz
        if float(np.max(remaining_power)) < snr_threshold_lin:
            break
        peak_idx = int(np.argmax(remaining_power))
        peak_lin = float(remaining_power[peak_idx])
        edge_threshold_lin = peak_lin * (10.0 ** (-peak_db_threshold / 10.0))
        # Walk left until below edge_threshold
        low_idx = peak_idx
        while low_idx > 0 and remaining_power[low_idx - 1] >= edge_threshold_lin:
            low_idx -= 1
        # Walk right
        high_idx = peak_idx
        while high_idx < remaining_power.size - 1 and remaining_power[high_idx + 1] >= edge_threshold_lin:
            high_idx += 1
        low_hz = float(psd_freqs[low_idx])
        high_hz = float(psd_freqs[high_idx])
        if high_hz - low_hz >= 100_000:
            candidates.append((low_hz, high_hz, peak_lin))
        # Zero out the found peak region + a guard band so the next
        # iteration doesn't re-find it
        guard_samples = max(1, (high_idx - low_idx) // 2)
        clear_lo = max(0, low_idx - guard_samples)
        clear_hi = min(remaining_power.size, high_idx + guard_samples + 1)
        remaining_power[clear_lo:clear_hi] = 0.0

    # Step 4: for each candidate, DDC + chirp analysis at the best-
    # matching template width
    hits: list[SurveyHit] = []
    for low_offset, high_offset, peak_lin in candidates:
        center_offset = (low_offset + high_offset) / 2.0
        absolute_center = capture_center_hz + int(center_offset)

        # Skip if this candidate is in the exclusion list (within radius
        # of a known target)
        excluded = any(
            abs(absolute_center - excl) < exclusion_radius_hz
            for excl in exclude_freqs_hz
        )
        if excluded:
            continue

        # Try templates from widest to narrowest; score each by how
        # well the observed chirp's frequency range matches the
        # template width. First-match-wins biases toward wider
        # templates since any chirp analysis on sufficient samples
        # produces linear segments; we need to pick the template
        # whose BW actually matches the signal.
        candidate_span = high_offset - low_offset
        best_hit: SurveyHit | None = None
        best_score: float = -1.0
        for template_hz in SURVEY_TEMPLATES_HZ:
            if candidate_span > template_hz * 1.5:
                continue
            if candidate_span < template_hz * 0.4:
                continue

            try:
                baseband = digital_downconvert(
                    samples,
                    source_rate=sample_rate,
                    shift_hz=center_offset,
                    target_bw_hz=template_hz,
                )
            except Exception:
                log.exception(
                    "survey: DDC failed at offset %d", int(center_offset)
                )
                continue

            if baseband.size < 1000:
                continue

            decimated_rate = int(
                round(baseband.size / (samples.size / sample_rate))
            )
            try:
                analysis = analyze_chirps(baseband, decimated_rate)
            except Exception:
                log.exception("survey: chirp analysis failed")
                continue

            if not (
                analysis.chirp_confidence > 0.5
                and analysis.num_chirp_segments >= 1
            ):
                continue

            # Score this template: how close is its width to the
            # observed chirp BW? Derive observed BW from slope ×
            # segment duration (segment length / decimated rate).
            observed_bw = abs(analysis.mean_slope_hz_per_sec) * (
                analysis.mean_segment_length_samples / decimated_rate
            )
            # Relative error: template should be within ±30% of
            # observed BW. Smaller relative error = better score.
            rel_err = abs(template_hz - observed_bw) / template_hz
            score = analysis.chirp_confidence - rel_err
            if score > best_score:
                snr_db = 10.0 * np.log10(
                    peak_lin / max(noise_floor_lin, 1e-20)
                )
                hit_freq = absolute_center
                if analysis.refined_center_offset_hz is not None:
                    hit_freq = absolute_center + int(
                        analysis.refined_center_offset_hz
                    )
                best_hit = SurveyHit(
                    freq_hz=hit_freq,
                    bandwidth_hz=template_hz,
                    chirp_analysis=analysis,
                    snr_db=snr_db,
                )
                best_score = score

        if best_hit is not None:
            hits.append(best_hit)
            log.info(
                "in-window survey hit: %.3f MHz / %d kHz "
                "(SNR %.1f dB, %d chirp segments)",
                best_hit.freq_hz / 1e6,
                best_hit.bandwidth_hz // 1000,
                best_hit.snr_db,
                best_hit.chirp_analysis.num_chirp_segments,
            )
    return hits


def _compute_psd(
    samples: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Compute power spectral density via Welch's method.

    Returns (freqs_hz, power_linear) where freqs_hz is monotonic
    -sample_rate/2 .. +sample_rate/2 and power_linear is in arbitrary
    units (only relative levels matter for thresholding).
    """
    try:
        from scipy.signal import welch
    except ImportError:
        return None, None
    if sample_rate <= 0 or samples.size == 0:
        return None, None

    nperseg = min(samples.size, max(256, sample_rate // SURVEY_PSD_BIN_HZ))
    nperseg = int(2 ** np.ceil(np.log2(nperseg)))  # round up to power of 2
    nperseg = min(nperseg, samples.size)
    freqs, psd = welch(
        samples, fs=sample_rate,
        nperseg=nperseg, noverlap=nperseg // 2,
        return_onesided=False, scaling="spectrum",
    )
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    return freqs, psd
