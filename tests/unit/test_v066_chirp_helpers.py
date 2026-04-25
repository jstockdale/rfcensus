"""v0.6.6 — SF/variant classification helpers in chirp_analysis.

Moved from rfcensus.detectors.builtin.lora when the legacy LoRa
detector was removed in favor of LoraSurveyTask. These are pure
functions of (slope, bandwidth) and (sf, bandwidth) — no detector-
state coupling — so testing them in isolation here, separate from
both the survey task tests and the broader chirp-analysis tests,
documents them as the public stable API for chirp interpretation.
"""

from __future__ import annotations

import pytest

from rfcensus.spectrum.chirp_analysis import (
    estimate_sf_from_slope,
    label_variant,
)


# ────────────────────────────────────────────────────────────────────
# estimate_sf_from_slope
# ────────────────────────────────────────────────────────────────────


class TestEstimateSF:
    """For a LoRa chirp, slope = BW² / 2^SF.
    So SF = log2(BW² / slope), then rounded and clamped to [5, 12]."""

    @pytest.mark.parametrize("sf", [7, 8, 9, 10, 11, 12])
    def test_round_trip_at_125khz(self, sf):
        bw = 125_000
        slope = (bw ** 2) / (2 ** sf)
        result = estimate_sf_from_slope(
            slope_hz_per_sec=slope, bandwidth_hz=bw,
        )
        assert result == sf

    @pytest.mark.parametrize("sf", [7, 8, 9, 10, 11, 12])
    def test_round_trip_at_250khz(self, sf):
        bw = 250_000
        slope = (bw ** 2) / (2 ** sf)
        result = estimate_sf_from_slope(
            slope_hz_per_sec=slope, bandwidth_hz=bw,
        )
        assert result == sf

    @pytest.mark.parametrize("sf", [7, 8, 9, 10])
    def test_round_trip_at_500khz(self, sf):
        bw = 500_000
        slope = (bw ** 2) / (2 ** sf)
        result = estimate_sf_from_slope(
            slope_hz_per_sec=slope, bandwidth_hz=bw,
        )
        assert result == sf

    def test_negative_slope_treated_as_magnitude(self):
        """LoRa up-chirps and down-chirps have opposite slope signs;
        only magnitude matters for SF."""
        bw = 125_000
        slope = (bw ** 2) / (2 ** 9)
        # Same magnitude, opposite sign
        assert (
            estimate_sf_from_slope(slope_hz_per_sec=slope, bandwidth_hz=bw)
            == estimate_sf_from_slope(slope_hz_per_sec=-slope, bandwidth_hz=bw)
        )

    def test_zero_or_negative_bw_returns_none(self):
        assert estimate_sf_from_slope(
            slope_hz_per_sec=1e6, bandwidth_hz=0,
        ) is None
        assert estimate_sf_from_slope(
            slope_hz_per_sec=1e6, bandwidth_hz=-1,
        ) is None

    def test_zero_slope_returns_none(self):
        """A non-chirp (e.g., a CW carrier) has zero slope — SF
        estimation isn't meaningful."""
        assert estimate_sf_from_slope(
            slope_hz_per_sec=0.0, bandwidth_hz=125_000,
        ) is None

    def test_implausibly_high_slope_returns_none(self):
        """Slope so high it implies SF<4 — almost certainly not LoRa."""
        bw = 125_000
        # SF=2 gives slope = bw²/4 = a huge number
        wild_slope = (bw ** 2) / 4
        assert estimate_sf_from_slope(
            slope_hz_per_sec=wild_slope, bandwidth_hz=bw,
        ) is None

    def test_implausibly_low_slope_returns_none(self):
        """Slope so low it implies SF>13 — too slow to be a real LoRa
        chirp under any standard configuration."""
        bw = 125_000
        # SF=14 gives slope = bw² / 16384
        too_slow = (bw ** 2) / 16384
        assert estimate_sf_from_slope(
            slope_hz_per_sec=too_slow, bandwidth_hz=bw,
        ) is None

    def test_clamps_to_canonical_range(self):
        """Slopes that round to SF<5 (after passing the implausibility
        check at SF=4) get clamped to 5; SF>12 to 12."""
        # SF=5.4 should round to 5, then stay at 5 (clamp lower bound)
        bw = 125_000
        slope_for_sf5 = (bw ** 2) / (2 ** 5)
        # Bump up slightly so it rounds to 5 not 6
        slope_just_below_5 = slope_for_sf5 * 1.3
        result = estimate_sf_from_slope(
            slope_hz_per_sec=slope_just_below_5, bandwidth_hz=bw,
        )
        # Should be either 5 (clamp) or None (out of range); both acceptable
        assert result in (5, None)


# ────────────────────────────────────────────────────────────────────
# label_variant
# ────────────────────────────────────────────────────────────────────


class TestLabelVariant:
    """Maps (SF, BW) → human variant. Meshtastic-distinctive
    combinations win when unique; LoRaWAN claimed at 125 kHz with
    SF7-10."""

    def test_meshtastic_long_fast(self):
        """SF11 / 250 kHz is the Meshtastic default."""
        assert label_variant(sf=11, bandwidth_hz=250_000) == "meshtastic_long_fast"

    def test_meshtastic_short_fast(self):
        assert label_variant(sf=7, bandwidth_hz=250_000) == "meshtastic_short_fast"

    def test_meshtastic_short_slow(self):
        assert label_variant(sf=8, bandwidth_hz=250_000) == "meshtastic_short_slow"

    def test_meshtastic_medium_fast(self):
        assert label_variant(sf=9, bandwidth_hz=250_000) == "meshtastic_medium_fast"

    def test_meshtastic_medium_slow(self):
        assert label_variant(sf=10, bandwidth_hz=250_000) == "meshtastic_medium_slow"

    def test_meshtastic_short_turbo(self):
        assert label_variant(sf=7, bandwidth_hz=500_000) == "meshtastic_short_turbo"

    def test_lorawan_at_125khz(self):
        for sf in (7, 8, 9, 10):
            assert label_variant(sf=sf, bandwidth_hz=125_000) == f"lorawan_sf{sf}"

    def test_meshtastic_long_moderate_overlaps_lorawan(self):
        """At SF11/125 kHz, both Meshtastic LongModerate and (theoretical)
        LoRaWAN SF11 overlap — variant label flags the ambiguity."""
        result = label_variant(sf=11, bandwidth_hz=125_000)
        assert result == "meshtastic_long_moderate_or_lorawan"

    def test_meshtastic_long_slow_overlaps_lorawan(self):
        result = label_variant(sf=12, bandwidth_hz=125_000)
        assert result == "meshtastic_long_slow_or_lorawan"

    def test_unrecognized_combination_returns_none(self):
        """SF/BW combinations that don't match a known profile."""
        # 62.5 kHz isn't a standard LoRa BW
        assert label_variant(sf=10, bandwidth_hz=62_500) is None

    def test_rounds_bw_to_khz_for_matching(self):
        """BW values that round to the same kHz should classify the same
        way — minor frequency-estimation jitter shouldn't swing labels."""
        # 125_000 and 125_500 both round-down to 125 kHz under //1000
        # but 125_500 // 1000 = 125 still, so identical handling
        assert (
            label_variant(sf=8, bandwidth_hz=125_000)
            == label_variant(sf=8, bandwidth_hz=125_500)
        )
