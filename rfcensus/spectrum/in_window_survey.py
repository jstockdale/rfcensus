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

from rfcensus.spectrum.chirp_analysis import (
    ChirpAnalysis,
    analyze_chirps,
    classify_sf_dechirp,
    label_variant,
)
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

    # Step 4: for each candidate, DDC + reference-dechirp SF
    # classification at each plausible template width. Pick the
    # template/SF pair with the highest dechirp peak concentration —
    # that's the BW × SF combination that actually fits the signal.
    # v0.6.8: this replaces the v0.6.5 path which used analyze_chirps
    # (slope-fit method) for both gating and BW selection. The slope
    # method assumes single-chirp-with-gaps segments; real LoRa packets
    # are contiguous chirps so it produced semi-random results.
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

        candidate_span = high_offset - low_offset

        # Dechirp each plausible template width and keep the one with
        # the highest peak concentration. Concentration is comparable
        # ACROSS templates because it's a fraction of total energy in
        # the dechirped spectrum's peak bin — a 250 kHz LoRa packet
        # viewed through the matching 250 kHz template gives much
        # higher concentration than the same packet viewed through a
        # 125 kHz template (which only sees half the chirp's frequency
        # excursion).
        best_hit: SurveyHit | None = None
        best_concentration: float = 0.0
        for template_hz in SURVEY_TEMPLATES_HZ:
            # Loose pre-filter: candidate span vs template width. The
            # span comes from a noisy power-spectrum peak edge so we
            # allow a wide window. The dechirp scoring is the real
            # discriminator.
            if candidate_span > template_hz * 1.8:
                continue
            if candidate_span < template_hz * 0.3:
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

            # Run dechirp classifier — the new SF discriminator
            try:
                est_sf, sf_conf, sf_peak, sf_scores = classify_sf_dechirp(
                    baseband, decimated_rate, template_hz,
                )
            except Exception:
                log.exception("survey: dechirp classifier failed")
                continue

            if est_sf is None:
                # Either no SF passed gates OR no chunks were scoreable.
                # We still want analyze_chirps for SNR/duty-cycle even
                # if no SF was confidently picked, but only if THIS
                # template's peak score is the running best.
                continue

            if sf_peak <= best_concentration:
                continue

            # This is a stronger candidate than anything we've seen.
            # Run analyze_chirps as a back-channel for SNR / duty
            # cycle / capture metadata that goes into the SurveyHit.
            try:
                analysis = analyze_chirps(baseband, decimated_rate)
            except Exception:
                log.exception("survey: chirp analysis (back-channel) failed")
                analysis = ChirpAnalysis(
                    chirp_confidence=0.0,
                    num_chirp_segments=0,
                    mean_segment_length_samples=0.0,
                    mean_slope_hz_per_sec=0.0,
                )

            # Stamp the dechirp-derived SF results onto ChirpAnalysis
            # so downstream consumers (LoraSurveyTask._emit_detection,
            # report renderer) get them via a single object.
            analysis.estimated_sf = est_sf
            analysis.sf_confidence = sf_conf
            analysis.sf_peak_concentration = sf_peak
            analysis.sf_scores = dict(sf_scores)

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
            best_concentration = sf_peak

        if best_hit is not None:
            hits.append(best_hit)
            variant = label_variant(
                sf=best_hit.chirp_analysis.estimated_sf or 0,
                bandwidth_hz=best_hit.bandwidth_hz,
            )
            log.info(
                "in-window survey hit: %.3f MHz / %d kHz / "
                "SF%s%s (SNR %.1f dB, dechirp peak %.3f, conf %.2f)",
                best_hit.freq_hz / 1e6,
                best_hit.bandwidth_hz // 1000,
                best_hit.chirp_analysis.estimated_sf,
                f" {variant}" if variant else "",
                best_hit.snr_db,
                best_hit.chirp_analysis.sf_peak_concentration,
                best_hit.chirp_analysis.sf_confidence,
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
