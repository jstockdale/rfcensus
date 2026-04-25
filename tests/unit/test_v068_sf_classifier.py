"""v0.6.8 — reference-dechirp SF classifier tests.

The dechirp method (multiply by reference downchirp, FFT, measure peak
concentration) is the standard gr-lora_sdr / NELoRa technique. It
replaces the v0.6.5 slope-fit estimator which was unreliable on real
LoRa traffic because it assumed each chirp was bracketed by silent
gaps, while real LoRa packets are contiguous chirps.

Test matrix:
  • SF accuracy across SF7-12 × {125, 250, 500} kHz BW (all combinations)
  • Low-SNR robustness (validated to -10 dB on synthetic data)
  • Random data symbols (real packet structure with cyclic-shifted
    payload chirps, not just preamble repeats)
  • Pure noise rejection (must return SF=None, not a confident guess)
  • Edge cases: too-short input, mismatched sample rate, invalid SF
  • make_downchirp helper roundtrip checks
"""

from __future__ import annotations

import numpy as np
import pytest

from rfcensus.spectrum.chirp_analysis import (
    classify_sf_dechirp,
    make_downchirp,
)

from tests.unit._dsp_fixtures import synthesize_realistic_lora


# ────────────────────────────────────────────────────────────────────
# make_downchirp
# ────────────────────────────────────────────────────────────────────


class TestMakeDownchirp:
    @pytest.mark.parametrize("sf", [7, 8, 9, 10, 11, 12])
    @pytest.mark.parametrize("bw", [125_000, 250_000, 500_000])
    def test_chirp_length_matches_sf_formula(self, sf, bw):
        """chirp_length_samples = round(sample_rate * 2^SF / bandwidth)."""
        sample_rate = bw  # match Nyquist for clean integer math
        downchirp, n = make_downchirp(
            sample_rate=sample_rate, bandwidth_hz=bw, sf=sf,
        )
        expected = int(round(sample_rate * (2 ** sf) / bw))
        assert n == expected
        assert downchirp.shape == (n,)
        assert downchirp.dtype == np.complex64

    def test_invalid_sf_raises(self):
        with pytest.raises(ValueError):
            make_downchirp(sample_rate=125_000, bandwidth_hz=125_000, sf=0)
        with pytest.raises(ValueError):
            make_downchirp(sample_rate=125_000, bandwidth_hz=125_000, sf=20)

    def test_invalid_bw_raises(self):
        with pytest.raises(ValueError):
            make_downchirp(sample_rate=125_000, bandwidth_hz=0, sf=7)
        with pytest.raises(ValueError):
            make_downchirp(sample_rate=125_000, bandwidth_hz=-1, sf=7)

    def test_invalid_sample_rate_raises(self):
        with pytest.raises(ValueError):
            make_downchirp(sample_rate=0, bandwidth_hz=125_000, sf=7)
        with pytest.raises(ValueError):
            make_downchirp(sample_rate=-1, bandwidth_hz=125_000, sf=7)

    def test_chirp_too_short_raises(self):
        """SF7 at 125 kHz BW with 1 kHz sample rate would produce
        only ~1 sample per chirp — useless, must reject."""
        with pytest.raises(ValueError):
            make_downchirp(sample_rate=1_000, bandwidth_hz=125_000, sf=7)

    def test_downchirp_is_unit_amplitude(self):
        """Each sample of a pure chirp should have magnitude 1.0."""
        downchirp, _ = make_downchirp(
            sample_rate=125_000, bandwidth_hz=125_000, sf=9,
        )
        magnitudes = np.abs(downchirp)
        # Allow tiny floating-point drift from cumsum + exp
        assert np.allclose(magnitudes, 1.0, atol=1e-5)

    def test_downchirp_inst_freq_sweeps_high_to_low(self):
        """A downchirp's instantaneous frequency must go from +BW/2
        to -BW/2 (negative slope), distinguishing it from an upchirp."""
        bw = 125_000
        downchirp, n = make_downchirp(
            sample_rate=bw, bandwidth_hz=bw, sf=10,
        )
        phase = np.unwrap(np.angle(downchirp))
        inst_freq = np.diff(phase) * bw / (2 * np.pi)
        # First sample's freq should be near +BW/2; last near -BW/2
        assert inst_freq[10] > 40_000  # near +BW/2 = +62500
        assert inst_freq[-10] < -40_000  # near -BW/2 = -62500


# ────────────────────────────────────────────────────────────────────
# classify_sf_dechirp — accuracy on the SF × BW grid
# ────────────────────────────────────────────────────────────────────


class TestSFAccuracyMatrix:
    """Matrix test — every (SF, BW) combination Meshtastic and LoRaWAN
    actually use, at SNR=20 dB which is realistic for nearby gateways
    and bench testing. Failures here indicate a broken classifier."""

    @pytest.mark.parametrize("sf", [7, 8, 9, 10, 11, 12])
    @pytest.mark.parametrize("bw", [125_000, 250_000, 500_000])
    def test_classifies_correctly_at_high_snr(self, sf, bw):
        # Allow capture to hold ~28 chirps + setup time
        chirp_dur = (2 ** sf) / bw
        duration = 0.2 + chirp_dur * 28
        # Use sequential data symbols rather than the default
        # all-zeros — real LoRa packets carry varied payload, and
        # all-identical chirps create an edge case where neighboring
        # SFs (e.g. SF7 vs SF8) score deceptively close because half-
        # chirp windows of an SF8 upchirp resemble SF7 chirps.
        samples = synthesize_realistic_lora(
            sample_rate=bw, duration_s=duration,
            bandwidth_hz=bw, sf=sf, snr_db=20,
            data_symbols=list(range(20)),
        )
        best_sf, conf, peak, scores = classify_sf_dechirp(
            samples, bw, bw,
        )
        assert best_sf == sf, (
            f"Wrong SF: expected {sf}, got {best_sf}. "
            f"conf={conf:.2f}, peak={peak:.4f}, scores={scores}"
        )
        # All accepted SF results must have the documented gate margins
        assert conf >= 1.20
        assert peak >= 0.052


# ────────────────────────────────────────────────────────────────────
# Low-SNR robustness
# ────────────────────────────────────────────────────────────────────


class TestLowSNRRobustness:
    """The dechirp method's processing gain (FFT energy concentration)
    is what gives LoRa its sub-noise sensitivity. Validated to -10 dB
    on synthetic data; below that the gates correctly reject (returning
    SF=None) rather than mislabeling."""

    @pytest.mark.parametrize("snr_db", [25, 15, 10, 5, 0, -5, -10])
    def test_sf9_250khz_classifies_down_to_neg10db(self, snr_db):
        samples = synthesize_realistic_lora(
            sample_rate=250_000, duration_s=0.5,
            bandwidth_hz=250_000, sf=9, snr_db=snr_db,
        )
        best_sf, conf, peak, scores = classify_sf_dechirp(
            samples, 250_000, 250_000,
        )
        assert best_sf == 9, (
            f"SF=9 misclassified at SNR={snr_db}dB: got SF{best_sf}, "
            f"conf={conf:.2f}, peak={peak:.4f}"
        )

    def test_extreme_low_snr_rejects_rather_than_mislabel(self):
        """At -25 dB the signal is well buried; the classifier must
        either get it right or return None — never produce a confident
        wrong answer."""
        samples = synthesize_realistic_lora(
            sample_rate=250_000, duration_s=0.5,
            bandwidth_hz=250_000, sf=9, snr_db=-25,
        )
        best_sf, conf, peak, scores = classify_sf_dechirp(
            samples, 250_000, 250_000,
        )
        # Either correct or None — both outcomes are acceptable.
        # Definitely NOT a different SF with high confidence.
        assert best_sf in (9, None), (
            f"At SNR=-25 dB, classifier produced wrong SF{best_sf} "
            f"(conf={conf:.2f}). Expected SF=9 or None."
        )


# ────────────────────────────────────────────────────────────────────
# Real packet structure (random data symbols)
# ────────────────────────────────────────────────────────────────────


class TestRealPacketStructure:
    """Real LoRa packets aren't just stacked preambles — the payload
    consists of cyclic-shifted upchirps where the shift is the symbol
    value (∈ [0, 2^SF)). The dechirp's invariant is: any contiguous
    chirp-length window of an upchirp dechirps to a tone, regardless
    of cyclic shift. This test verifies the classifier handles real
    packet shape, not just degenerate preamble-only signals."""

    @pytest.mark.parametrize("sf", [7, 9, 11])
    @pytest.mark.parametrize("bw", [125_000, 250_000])
    def test_classifies_with_random_payload_symbols(self, sf, bw):
        rng = np.random.default_rng(7 + sf * 100 + bw)
        data = rng.integers(0, 2 ** sf, size=20).tolist()
        chirp_dur = (2 ** sf) / bw
        duration = 0.2 + chirp_dur * 28
        samples = synthesize_realistic_lora(
            sample_rate=bw, duration_s=duration,
            bandwidth_hz=bw, sf=sf, snr_db=15,
            data_symbols=data,
        )
        best_sf, conf, peak, scores = classify_sf_dechirp(
            samples, bw, bw,
        )
        assert best_sf == sf, (
            f"Random-symbol packet misclassified: SF{sf}/{bw}, "
            f"got SF{best_sf}, conf={conf:.2f}, peak={peak:.4f}"
        )


# ────────────────────────────────────────────────────────────────────
# Pure-noise rejection
# ────────────────────────────────────────────────────────────────────


class TestNoiseRejection:
    """The two acceptance gates (concentration ≥ 0.052 AND confidence
    ≥ 1.20) are the noise-floor protection. Pure noise produces
    concentration ~0.04 (uniform spectrum gives 1/N per bin plus
    statistical fluctuation lifting the max); without the gates we'd
    confidently mislabel any band-of-noise as some SF."""

    @pytest.mark.parametrize("seed", [1, 2, 3, 42, 100])
    def test_pure_noise_returns_none(self, seed):
        rng = np.random.default_rng(seed)
        n_samples = 125_000  # 0.5 s at 250 kHz
        noise = (
            rng.standard_normal(n_samples)
            + 1j * rng.standard_normal(n_samples)
        ).astype(np.complex64) * np.float32(0.5)
        best_sf, conf, peak, scores = classify_sf_dechirp(
            noise, 250_000, 250_000,
        )
        assert best_sf is None, (
            f"Pure noise (seed {seed}) was classified as SF{best_sf}: "
            f"conf={conf:.2f}, peak={peak:.4f} — gates failed to "
            f"reject."
        )

    def test_zero_input_returns_none(self):
        zero = np.zeros(125_000, dtype=np.complex64)
        best_sf, conf, peak, scores = classify_sf_dechirp(
            zero, 250_000, 250_000,
        )
        assert best_sf is None


# ────────────────────────────────────────────────────────────────────
# Edge cases & input validation
# ────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_too_few_samples_returns_none(self):
        """Input below 1024 samples can't be reliably classified."""
        small = np.ones(500, dtype=np.complex64)
        best_sf, conf, peak, scores = classify_sf_dechirp(
            small, 250_000, 250_000,
        )
        assert best_sf is None

    def test_no_above_noise_activity_returns_none(self):
        """All samples below the noise floor → no active region →
        cannot classify."""
        # All-zero signal has median amplitude 0; mask threshold is
        # max(1e-3, 0) = 1e-3; nothing is above that → no active region
        signal = np.zeros(125_000, dtype=np.complex64)
        # Add tiny noise well below 1e-3
        signal += np.float32(1e-6) * (
            np.random.default_rng(0).standard_normal(125_000)
        ).astype(np.complex64)
        best_sf, conf, peak, scores = classify_sf_dechirp(
            signal, 250_000, 250_000,
        )
        assert best_sf is None

    def test_returns_scores_dict_even_when_rejecting(self):
        """For diagnostics, callers can inspect per-SF scores even
        when no SF was confidently picked."""
        rng = np.random.default_rng(1)
        noise = (
            rng.standard_normal(125_000)
            + 1j * rng.standard_normal(125_000)
        ).astype(np.complex64) * np.float32(0.5)
        best_sf, conf, peak, scores = classify_sf_dechirp(
            noise, 250_000, 250_000,
        )
        assert best_sf is None  # rejected
        assert isinstance(scores, dict)
        # Every SF in the candidate range got scored
        assert set(scores.keys()) == {7, 8, 9, 10, 11, 12}
        # All scores are valid floats
        for s, v in scores.items():
            assert isinstance(v, float)
            assert v >= 0.0


# ────────────────────────────────────────────────────────────────────
# Integration: verify the v0.6.7 misclassification scenario is fixed
# ────────────────────────────────────────────────────────────────────


class TestRegressionAgainstV067Bug:
    """Specific regression test for the bug John reported: sending
    Meshtastic MediumFast (SF=9 / 250 kHz) traffic, the v0.6.5/0.6.6
    slope-fit estimator labeled it as ShortFast (SF=7) — exactly 2 SF
    off because the slope fitter was locking onto fragments instead of
    full chirps.

    The dechirp method must classify SF=9 correctly even with realistic
    packet structure (mix of preamble + cyclic-shifted data symbols).
    """

    def test_meshtastic_medium_fast_classifies_as_sf9(self):
        """Reproduces the v0.6.7 misclassification scenario with
        the v0.6.8 classifier — must produce SF=9, not SF=7."""
        rng = np.random.default_rng(2024)
        data = rng.integers(0, 2 ** 9, size=20).tolist()
        # SF9, 250 kHz, ~25 ms packet, capture 0.5 s, SNR 18 dB
        # (matches the SNR observed in John's real-world detection)
        samples = synthesize_realistic_lora(
            sample_rate=250_000, duration_s=0.5,
            bandwidth_hz=250_000, sf=9, snr_db=18,
            data_symbols=data,
        )
        best_sf, conf, peak, scores = classify_sf_dechirp(
            samples, 250_000, 250_000,
        )
        assert best_sf == 9, (
            f"REGRESSION: Meshtastic MediumFast (SF=9) misclassified "
            f"as SF{best_sf}. The bug v0.6.8 was meant to fix is back. "
            f"conf={conf:.2f}, peak={peak:.4f}, all scores={scores}"
        )
        # The SF7 score (the v0.6.7 wrong answer) should be far below
        # the SF9 score
        assert scores[9] > scores[7] * 1.5, (
            f"SF9 score ({scores[9]:.4f}) should dominate SF7 "
            f"({scores[7]:.4f}) — discrimination is weak"
        )
