"""v0.5.42 tests: ChirpAnalysis enhancements.

Tests the new fields added to ChirpAnalysis:
  • refined_center_offset_hz with FFT centroid + chirp-mean
    reconciliation
  • frequency_estimate_method (agreement / disagreement / single)
  • snr_db, signal_power_dbfs, noise_power_dbfs
  • burst_total_duration_s, capture_duration_s, duty_cycle

Reuses the chirp synthesizer from test_confirmation_task.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfcensus.spectrum.chirp_analysis import analyze_chirps

from tests.unit._dsp_fixtures import _synthesize_lora_chirp


SAMPLE_RATE = 2_400_000  # production rate; vectorized analyze_chirps
DURATION_S = 0.3  # ~36 chirps at SF11/250kHz — comfortable for analyzer


class TestRefinedFrequency:
    """Refined center frequency should converge to within ~1 kHz across
    a range of synthetic chirp offsets."""

    @pytest.mark.parametrize(
        "shift_hz", [0, 1_000, 37_000, -82_000, 200_000, -150_000]
    )
    def test_refinement_recovers_offset(self, shift_hz: int):
        samples = _synthesize_lora_chirp(
            sample_rate=SAMPLE_RATE, duration_s=DURATION_S,
            bandwidth_hz=250_000, sf=11,
            center_shift_hz=float(shift_hz),
        )
        r = analyze_chirps(samples, SAMPLE_RATE)

        assert r.refined_center_offset_hz is not None
        assert r.frequency_estimate_method is not None
        assert r.frequency_uncertainty_hz is not None

        # Should converge to within 1 kHz of the true offset
        error_hz = abs(r.refined_center_offset_hz - shift_hz)
        assert error_hz < 1000, (
            f"refined offset {r.refined_center_offset_hz:.0f} Hz "
            f"too far from true {shift_hz} Hz (error {error_hz:.0f} Hz)"
        )

    def test_baseband_chirp_methods_agree(self):
        """Both FFT and chirp methods should agree on a clean chirp."""
        samples = _synthesize_lora_chirp(
            sample_rate=SAMPLE_RATE, duration_s=DURATION_S,
            bandwidth_hz=250_000, sf=11, center_shift_hz=0.0,
        )
        r = analyze_chirps(samples, SAMPLE_RATE)
        assert r.frequency_estimate_method == "agreement", (
            f"methods should agree; got {r.frequency_estimate_method}, "
            f"fft={r.fft_centroid_offset_hz:.0f} Hz "
            f"chirp={r.chirp_centroid_offset_hz:.0f} Hz"
        )

    def test_noise_only_returns_no_refinement(self):
        """Pure noise has no chirp segments, so no refinement
        possible."""
        np.random.seed(0)
        noise = (
            np.random.randn(SAMPLE_RATE // 2)
            + 1j * np.random.randn(SAMPLE_RATE // 2)
        ).astype(np.complex64) * 0.1
        r = analyze_chirps(noise, SAMPLE_RATE)
        assert r.chirp_confidence == 0.0
        assert r.refined_center_offset_hz is None
        assert r.frequency_estimate_method is None


class TestSNR:
    """SNR should reflect the actual signal/noise power ratio."""

    def test_strong_chirp_gives_high_snr(self):
        samples = _synthesize_lora_chirp(
            sample_rate=SAMPLE_RATE, duration_s=0.5,
            bandwidth_hz=250_000, sf=11, center_shift_hz=0.0,
        )
        r = analyze_chirps(samples, SAMPLE_RATE)
        assert r.snr_db is not None
        # Synthesizer produces ~30+ dB SNR (signal=1.0, noise floor=0.02)
        assert r.snr_db > 20, f"expected high SNR, got {r.snr_db:.1f} dB"

    def test_noise_floor_unmeasurable_returns_none(self):
        """All-active or all-silent input gives no noise estimate."""
        samples = np.ones(10_000, dtype=np.complex64)  # constant CW
        r = analyze_chirps(samples, SAMPLE_RATE)
        # When all samples are above the median*0.5 threshold (constant
        # amplitude → mask is all-True), noise window is empty.
        assert r.snr_db is None

    def test_snr_present_even_when_no_chirps(self):
        """SNR should be computed from amplitude analysis even when
        chirp linearity check fails (so reports of "loud but not LoRa"
        signals can still surface SNR)."""
        # Generate continuous tone (no chirp behavior) at offset
        t = np.arange(SAMPLE_RATE // 2, dtype=np.float64) / SAMPLE_RATE
        signal = np.exp(2j * np.pi * 100_000 * t).astype(np.complex64)
        np.random.seed(0)
        noise = (
            np.random.randn(t.size) + 1j * np.random.randn(t.size)
        ).astype(np.complex64) * 0.05
        # Mix signal in for half the duration to give noise floor
        # measurement opportunity
        signal[: t.size // 2] = noise[: t.size // 2]
        r = analyze_chirps(signal + noise * 0.1, SAMPLE_RATE)
        # No chirps detected (it's a tone, not a chirp)
        assert r.num_chirp_segments == 0
        # But SNR should still be computed
        assert r.snr_db is not None


class TestDutyCycle:
    """Duty cycle = active samples / total samples."""

    def test_chirp_with_gaps_has_partial_duty(self):
        samples = _synthesize_lora_chirp(
            sample_rate=SAMPLE_RATE, duration_s=0.5,
            bandwidth_hz=250_000, sf=11,
        )
        r = analyze_chirps(samples, SAMPLE_RATE)
        assert r.duty_cycle is not None
        # 20% gap between chirps → duty cycle ~83%
        assert 0.7 < r.duty_cycle < 0.95

    def test_capture_duration_matches_input(self):
        samples = _synthesize_lora_chirp(
            sample_rate=SAMPLE_RATE, duration_s=0.5,
            bandwidth_hz=250_000, sf=11,
        )
        r = analyze_chirps(samples, SAMPLE_RATE)
        assert r.capture_duration_s == pytest.approx(0.5, abs=0.01)

    def test_burst_duration_present(self):
        samples = _synthesize_lora_chirp(
            sample_rate=SAMPLE_RATE, duration_s=0.5,
            bandwidth_hz=250_000, sf=11,
        )
        r = analyze_chirps(samples, SAMPLE_RATE)
        assert r.burst_total_duration_s is not None
        # Burst duration ≤ capture duration
        assert r.burst_total_duration_s <= r.capture_duration_s


class TestBackwardCompat:
    """Old fields should still work; new fields default to None when
    missing."""

    def test_dataclass_constructible_with_legacy_fields(self):
        from rfcensus.spectrum.chirp_analysis import ChirpAnalysis
        # Old-style: positional args only
        c = ChirpAnalysis(0.5, 3, 100.0, 5000.0, "test")
        assert c.chirp_confidence == 0.5
        assert c.refined_center_offset_hz is None
        assert c.snr_db is None

    def test_insufficient_samples_keeps_capture_duration(self):
        """Even on short input, capture_duration_s is computed."""
        samples = np.zeros(50, dtype=np.complex64)
        r = analyze_chirps(samples, SAMPLE_RATE)
        assert r.reasoning == "insufficient samples"
        assert r.capture_duration_s == pytest.approx(50 / SAMPLE_RATE)
