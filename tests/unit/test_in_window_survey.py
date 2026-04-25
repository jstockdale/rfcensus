"""v0.5.42 tests: in-window opportunistic survey.

Synthesize a 2 MHz IQ window with one or more chirp signals at
various offsets, run the survey, and verify discovery.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfcensus.spectrum.in_window_survey import (
    DEFAULT_EXCLUSION_RADIUS_HZ,
    SurveyHit,
    survey_iq_window,
)

from tests.unit._dsp_fixtures import _synthesize_lora_chirp


SAMPLE_RATE = 2_400_000
DURATION_S = 0.3
CAPTURE_CENTER_HZ = 915_000_000


def _hidden_chirp_at_offset(offset_hz: float, *, sf: int = 11) -> np.ndarray:
    """Generate a chirp signal at the given offset within the capture
    window. Returns 0.3 s of complex IQ at 2.4 Msps."""
    return _synthesize_lora_chirp(
        sample_rate=SAMPLE_RATE, duration_s=DURATION_S,
        bandwidth_hz=250_000, sf=sf, center_shift_hz=offset_hz,
    )


def _noise_only(amplitude: float = 0.05) -> np.ndarray:
    np.random.seed(0)
    n = int(SAMPLE_RATE * DURATION_S)
    return (
        np.random.randn(n) + 1j * np.random.randn(n)
    ).astype(np.complex64) * np.float32(amplitude)


class TestSurveyDiscovery:
    def test_finds_hidden_chirp(self):
        """A chirp at offset +500 kHz from the capture center should
        be discovered as a SurveyHit."""
        samples = _hidden_chirp_at_offset(500_000.0)
        hits = survey_iq_window(
            samples,
            sample_rate=SAMPLE_RATE,
            capture_center_hz=CAPTURE_CENTER_HZ,
        )
        assert len(hits) >= 1
        # Should find the chirp near 915.500 MHz
        found_freqs = [h.freq_hz for h in hits]
        assert any(
            abs(f - (CAPTURE_CENTER_HZ + 500_000)) < 50_000
            for f in found_freqs
        ), f"expected hit near 915.500 MHz; got {found_freqs}"

    def test_pure_noise_yields_no_hits(self):
        """No chirps in noise → no survey hits."""
        samples = _noise_only(amplitude=0.05)
        hits = survey_iq_window(
            samples,
            sample_rate=SAMPLE_RATE,
            capture_center_hz=CAPTURE_CENTER_HZ,
        )
        assert hits == [], f"expected no hits in noise, got: {hits}"

    def test_excludes_known_targets(self):
        """A chirp at +500 kHz, with that freq excluded → no hit."""
        target_freq = CAPTURE_CENTER_HZ + 500_000
        samples = _hidden_chirp_at_offset(500_000.0)
        hits = survey_iq_window(
            samples,
            sample_rate=SAMPLE_RATE,
            capture_center_hz=CAPTURE_CENTER_HZ,
            exclude_freqs_hz=[target_freq],
        )
        assert hits == [], (
            f"expected exclusion to suppress hit at {target_freq}; "
            f"got: {[(h.freq_hz, h.bandwidth_hz) for h in hits]}"
        )

    def test_exclusion_only_affects_nearby(self):
        """Excluding a far-away freq doesn't suppress a nearby chirp."""
        samples = _hidden_chirp_at_offset(500_000.0)
        # Exclude 915.000 MHz — far from the hit at ~915.5
        hits = survey_iq_window(
            samples,
            sample_rate=SAMPLE_RATE,
            capture_center_hz=CAPTURE_CENTER_HZ,
            exclude_freqs_hz=[CAPTURE_CENTER_HZ - 800_000],
        )
        assert len(hits) >= 1

    def test_hit_includes_metadata(self):
        """SurveyHit carries chirp_analysis with refined center, SNR,
        and other metadata."""
        samples = _hidden_chirp_at_offset(500_000.0)
        hits = survey_iq_window(
            samples,
            sample_rate=SAMPLE_RATE,
            capture_center_hz=CAPTURE_CENTER_HZ,
        )
        assert len(hits) >= 1
        hit = hits[0]
        assert hit.snr_db > 0
        assert hit.chirp_analysis is not None
        assert hit.chirp_analysis.num_chirp_segments >= 1
        assert hit.chirp_analysis.snr_db is not None
        assert hit.chirp_analysis.duty_cycle is not None

    def test_short_capture_returns_empty(self):
        """Below ~250 ms of IQ, survey returns nothing (insufficient
        data for meaningful PSD averaging)."""
        short = np.zeros(SAMPLE_RATE // 10, dtype=np.complex64)
        hits = survey_iq_window(
            short,
            sample_rate=SAMPLE_RATE,
            capture_center_hz=CAPTURE_CENTER_HZ,
        )
        assert hits == []


class TestSurveyTemplateMatching:
    def test_500khz_chirp_matches_500khz_template(self):
        """A 500 kHz wide chirp should be classified as the 500 kHz
        template."""
        samples = _synthesize_lora_chirp(
            sample_rate=SAMPLE_RATE, duration_s=DURATION_S,
            bandwidth_hz=500_000, sf=10,
            center_shift_hz=500_000.0,
        )
        hits = survey_iq_window(
            samples,
            sample_rate=SAMPLE_RATE,
            capture_center_hz=CAPTURE_CENTER_HZ,
        )
        assert len(hits) >= 1
        assert hits[0].bandwidth_hz == 500_000

    def test_125khz_chirp_matches_125khz_template(self):
        """A 125 kHz wide chirp should be classified as the 125 kHz
        template."""
        samples = _synthesize_lora_chirp(
            sample_rate=SAMPLE_RATE, duration_s=DURATION_S,
            bandwidth_hz=125_000, sf=11,
            center_shift_hz=500_000.0,
        )
        hits = survey_iq_window(
            samples,
            sample_rate=SAMPLE_RATE,
            capture_center_hz=CAPTURE_CENTER_HZ,
        )
        assert len(hits) >= 1
        assert hits[0].bandwidth_hz == 125_000


class TestSurveyDoesNotCrash:
    """Defensive: malformed or empty inputs shouldn't raise."""

    def test_zero_sample_rate(self):
        hits = survey_iq_window(
            np.ones(1000, dtype=np.complex64),
            sample_rate=0,
            capture_center_hz=CAPTURE_CENTER_HZ,
        )
        assert hits == []

    def test_empty_samples(self):
        hits = survey_iq_window(
            np.array([], dtype=np.complex64),
            sample_rate=SAMPLE_RATE,
            capture_center_hz=CAPTURE_CENTER_HZ,
        )
        assert hits == []


# ────────────────────────────────────────────────────────────────────
# v0.6.8: end-to-end integration with realistic gap-free LoRa
# ────────────────────────────────────────────────────────────────────


class TestSurveyWithRealisticLora:
    """The v0.6.5/0.6.6 survey was tested only against synthesize_lora_chirp
    (chirps with 20% silent gaps), which doesn't match real LoRa packet
    structure. This class verifies the v0.6.8 dechirp-classifier path
    handles realistic gap-free LoRa correctly — the bug class that caused
    SF9 (MediumFast) to be misclassified as SF7 (ShortFast) in real
    captures."""

    def _embed_realistic_lora_at_offset(
        self, *, offset_hz: float, sf: int, bw: int, snr_db: float = 18,
    ) -> np.ndarray:
        """Build a 0.3 s 2.4 Msps capture containing a realistic LoRa
        packet (preamble + many random data symbols, no gaps) at the
        given offset from the capture center.

        Packet length scales with SF: short SFs (7) need more chirps
        for the averaged-PSD candidate finder to register the burst
        above noise (low duty cycle otherwise). The dechirp classifier
        itself works on much shorter signals — this is a survey-stage
        threshold, not a classifier limitation.
        """
        from tests.unit._dsp_fixtures import synthesize_realistic_lora

        # Aim for ~30% duty cycle. capture is 300 ms; chirp_dur = 2^SF/BW.
        # n_chirps such that n_chirps * chirp_dur ≈ 90 ms.
        chirp_dur_s = (2 ** sf) / bw
        target_active_s = 0.09
        n_chirps = max(40, int(target_active_s / chirp_dur_s))

        rng = np.random.default_rng(7 + sf * 100 + bw)
        data = rng.integers(0, 2 ** sf, size=n_chirps).tolist()
        baseband = synthesize_realistic_lora(
            sample_rate=SAMPLE_RATE, duration_s=DURATION_S,
            bandwidth_hz=bw, sf=sf, snr_db=snr_db,
            data_symbols=data, num_data_chirps=n_chirps,
        )
        # Mix to offset
        n = baseband.size
        t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
        mixer = np.exp(2j * np.pi * offset_hz * t).astype(np.complex64)
        return (baseband * mixer).astype(np.complex64)

    def test_meshtastic_medium_fast_classified_as_sf9(self):
        """Direct regression for John's v0.6.7 misclassification:
        MediumFast (SF9 / 250 kHz) traffic must produce a SurveyHit
        with estimated_sf=9, not SF7."""
        samples = self._embed_realistic_lora_at_offset(
            offset_hz=400_000.0, sf=9, bw=250_000, snr_db=20,
        )
        hits = survey_iq_window(
            samples,
            sample_rate=SAMPLE_RATE,
            capture_center_hz=CAPTURE_CENTER_HZ,
        )
        assert len(hits) >= 1, "expected to find the embedded SF9 packet"
        # Find the hit closest to our embed location
        target = CAPTURE_CENTER_HZ + 400_000
        hit = min(hits, key=lambda h: abs(h.freq_hz - target))
        assert hit.bandwidth_hz == 250_000, (
            f"expected 250 kHz template; got {hit.bandwidth_hz}. "
            f"All hit BWs: {[h.bandwidth_hz for h in hits]}"
        )
        assert hit.chirp_analysis.estimated_sf == 9, (
            f"REGRESSION: Expected SF=9, got SF="
            f"{hit.chirp_analysis.estimated_sf}. "
            f"sf_scores={hit.chirp_analysis.sf_scores}"
        )

    def test_meshtastic_short_fast_classified_as_sf7(self):
        """SF7 / 250 kHz (Meshtastic ShortFast) classifies correctly."""
        samples = self._embed_realistic_lora_at_offset(
            offset_hz=400_000.0, sf=7, bw=250_000, snr_db=20,
        )
        hits = survey_iq_window(
            samples,
            sample_rate=SAMPLE_RATE,
            capture_center_hz=CAPTURE_CENTER_HZ,
        )
        assert len(hits) >= 1
        target = CAPTURE_CENTER_HZ + 400_000
        hit = min(hits, key=lambda h: abs(h.freq_hz - target))
        assert hit.chirp_analysis.estimated_sf == 7

    def test_meshtastic_long_fast_classified_as_sf11(self):
        """SF11 / 250 kHz (Meshtastic LongFast — most common variant)
        classifies correctly."""
        samples = self._embed_realistic_lora_at_offset(
            offset_hz=400_000.0, sf=11, bw=250_000, snr_db=20,
        )
        hits = survey_iq_window(
            samples,
            sample_rate=SAMPLE_RATE,
            capture_center_hz=CAPTURE_CENTER_HZ,
        )
        assert len(hits) >= 1
        target = CAPTURE_CENTER_HZ + 400_000
        hit = min(hits, key=lambda h: abs(h.freq_hz - target))
        assert hit.chirp_analysis.estimated_sf == 11

    def test_hit_carries_dechirp_metadata(self):
        """SurveyHit's chirp_analysis must include the dechirp
        classifier outputs so downstream consumers (LoraSurveyTask
        emission, report renderer) can use them."""
        samples = self._embed_realistic_lora_at_offset(
            offset_hz=400_000.0, sf=9, bw=250_000, snr_db=20,
        )
        hits = survey_iq_window(
            samples,
            sample_rate=SAMPLE_RATE,
            capture_center_hz=CAPTURE_CENTER_HZ,
        )
        assert len(hits) >= 1
        hit = min(
            hits, key=lambda h: abs(h.freq_hz - (CAPTURE_CENTER_HZ + 400_000)),
        )
        a = hit.chirp_analysis
        assert a.estimated_sf is not None
        assert a.sf_confidence > 1.0
        assert a.sf_peak_concentration > 0.0
        assert a.sf_scores is not None
        assert len(a.sf_scores) == 6  # SF7 through SF12

