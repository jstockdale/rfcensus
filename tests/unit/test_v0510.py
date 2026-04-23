"""Tests for v0.5.10 — antenna physics fixes.

Two real-world bugs:
  1. marine_vhf had usable_range = [150, 175] MHz, so a Marine VHF
     antenna couldn't even attempt 144 MHz amateur — but physically
     a quarter-wave VHF whip works fine across 144-180 MHz.
  2. Telescopic whips were collapsing into purpose-built library
     antennas (e.g. picking 915 MHz → assigned whip_915), losing the
     wider physical bandwidth a generic telescopic actually has.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ──────────────────────────────────────────────────────────────────
# Library marine_vhf widening
# ──────────────────────────────────────────────────────────────────


class TestMarineVhfBandwidth:
    """The library marine_vhf antenna should be matchable to 144 MHz
    amateur band, not just 156-162 marine."""

    def test_marine_vhf_covers_144_mhz(self):
        from rfcensus.config.loader import load_config

        config = load_config()
        marine = next(
            (a for a in config.antennas if a.id == "marine_vhf"), None,
        )
        assert marine is not None, "marine_vhf antenna missing from library"
        # 144 MHz must be within usable_range
        low, high = marine.usable_range
        assert low <= 144_000_000 <= high, (
            f"marine_vhf usable_range {low/1e6:.0f}-{high/1e6:.0f} MHz "
            f"doesn't include 144 MHz amateur band"
        )

    def test_marine_vhf_still_optimal_at_marine_band(self):
        """Widening shouldn't hurt the resonant region — score at 156 MHz
        should still be the maximum (perfect match)."""
        from rfcensus.config.loader import load_config
        from rfcensus.hardware.antenna import Antenna

        config = load_config()
        cfg = next(a for a in config.antennas if a.id == "marine_vhf")
        ant = Antenna.from_config(cfg)
        # At the resonant frequency, suitability should be 1.0
        assert ant.suitability(156_800_000) == 1.0
        # At 144 MHz (~8% off-resonance), suitability should be in the
        # 0.7-0.8 range — well above the 0.7 "well covered" threshold
        score_144 = ant.suitability(144_000_000)
        assert score_144 >= 0.7, (
            f"144 MHz score {score_144:.2f} below 0.7 threshold — "
            f"the matcher will still mark this band as uncovered"
        )

    def test_144_mhz_band_covered_when_marine_vhf_assigned(self):
        """End-to-end: a dongle with marine_vhf should cover the 144 MHz
        APRS band per compute_coverage()."""
        from rfcensus.config.loader import load_config
        from rfcensus.config.schema import BandConfig
        from rfcensus.engine.coverage import compute_coverage
        from rfcensus.hardware.antenna import Antenna
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
        )

        config = load_config()
        marine = Antenna.from_config(
            next(a for a in config.antennas if a.id == "marine_vhf")
        )
        caps = DongleCapabilities(
            freq_range_hz=(24_000_000, 1_700_000_000),
            max_sample_rate=2_400_000, bits_per_sample=8,
            bias_tee_capable=False, tcxo_ppm=10.0,
        )
        d = Dongle(
            id="rtl-0", serial="X", model="rtlsdr_generic",
            driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
            driver_index=0,
        )
        d.antenna = marine

        aprs = BandConfig(
            id="aprs_2m", name="APRS",
            freq_low=144_380_000, freq_high=144_400_000,
        )
        report = compute_coverage([aprs], [d])
        assert len(report.matched) == 1, (
            f"Expected aprs_2m to be matched; got missing={[m.band.id for m in report.missing]}"
        )


# ──────────────────────────────────────────────────────────────────
# Telescopic always creates custom (no library collapse)
# ──────────────────────────────────────────────────────────────────


class TestTelescopicAlwaysCustom:
    """Telescopic whips have wider physical bandwidth than purpose-built
    library antennas. Always creating a custom stanza preserves that.
    """

    def _make_state(self):
        from rfcensus.commands.setup import _WizardState
        s = _WizardState(detected=[])
        s.library_antennas = [
            {"id": "whip_915", "name": "915 MHz tuned whip"},
            {"id": "whip_433", "name": "433 MHz tuned whip"},
            {"id": "marine_vhf", "name": "Marine VHF"},
        ]
        return s

    def _make_dongle(self):
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
        )
        caps = DongleCapabilities(
            freq_range_hz=(24_000_000, 1_700_000_000),
            max_sample_rate=2_400_000, bits_per_sample=8,
            bias_tee_capable=False, tcxo_ppm=10.0,
        )
        return Dongle(
            id="rtl-0", serial="X", model="rtlsdr_generic",
            driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
            driver_index=0,
        )

    def test_915_telescopic_creates_custom_not_whip_915(self):
        from rfcensus.commands.setup import _flow_telescopic
        state = self._make_state()
        dongle = self._make_dongle()

        # Pick 915 MHz then accept default length
        responses = iter(["11", ""])
        with patch("click.prompt", side_effect=lambda *a, **kw: next(responses)):
            result = _flow_telescopic(dongle, state)

        # Must NOT be the library whip_915 — even though it's available
        assert result != "whip_915"
        assert result.startswith("whip_telescopic_")
        # Custom stanzas use ±50% range (v0.5.14 widening for survey use;
        # earlier versions were ±15% which was too tight for real-world
        # detection work — users' quarter-wave whips detect signals well
        # outside theoretical VSWR-2:1 range).
        custom = state.custom_antennas[0]
        low, high = custom["usable_range"]
        assert low == int(915_000_000 * 0.5)
        assert high == int(915_000_000 * 1.5)

    def test_162_telescopic_creates_custom_not_marine_vhf(self):
        """The original bug: picking 162 MHz with telescopic gave you
        marine_vhf, whose narrow bandwidth excluded 144 MHz coverage."""
        from rfcensus.commands.setup import _flow_telescopic
        state = self._make_state()
        dongle = self._make_dongle()

        # _pick_frequency for 162 MHz — option 3 in the COMMON_FREQUENCIES list
        # Then press Enter for default length
        responses = iter(["3", ""])
        with patch("click.prompt", side_effect=lambda *a, **kw: next(responses)):
            result = _flow_telescopic(dongle, state)

        assert result != "marine_vhf"
        assert result.startswith("whip_telescopic_")
        custom = state.custom_antennas[0]
        # Should cover 144 MHz at minimum (the original complaint)
        low, high = custom["usable_range"]
        assert low <= 144_000_000, (
            f"telescopic at 162 MHz should cover 144 MHz; "
            f"got usable_range {low/1e6:.0f}-{high/1e6:.0f} MHz"
        )
