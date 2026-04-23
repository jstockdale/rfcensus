"""Tests for v0.5.5 — fleet-aware antenna suggestion + optimizer.

Covers:
  • suggest_for_new_dongle: returns generic when fleet already covers
    everything, returns catalog match when one fits, returns quarter-
    wave fallback when nothing in catalog matches a real gap
  • optimize_fleet: assigns the right antennas to maximize coverage,
    surfaces uncovered bands, generates shopping suggestions
  • diff_against_current correctly identifies changes vs unchanged
"""

from __future__ import annotations

from dataclasses import replace

import pytest


def _band(band_id, freq_mhz, span_pct=0.02):
    """Synthetic BandConfig at a given center frequency."""
    from rfcensus.config.schema import BandConfig
    f = int(freq_mhz * 1_000_000)
    return BandConfig(
        id=band_id, name=band_id,
        freq_low=int(f * (1 - span_pct)),
        freq_high=int(f * (1 + span_pct)),
    )


def _antenna(ant_id, resonant_mhz, usable_low_mhz=None, usable_high_mhz=None):
    """Synthetic Antenna for testing."""
    from rfcensus.hardware.antenna import Antenna
    res_hz = int(resonant_mhz * 1_000_000) if resonant_mhz else None
    low = int((usable_low_mhz or resonant_mhz * 0.7) * 1_000_000)
    high = int((usable_high_mhz or resonant_mhz * 1.3) * 1_000_000)
    return Antenna(
        id=ant_id, name=ant_id, antenna_type="whip",
        resonant_freq_hz=res_hz, usable_range=(low, high),
        gain_dbi=2.0, polarization="vertical",
        requires_bias_power=False, notes="",
    )


def _wideband_antenna(ant_id, low_mhz, high_mhz):
    """Wideband antenna with no resonant frequency."""
    from rfcensus.hardware.antenna import Antenna
    return Antenna(
        id=ant_id, name=ant_id, antenna_type="discone",
        resonant_freq_hz=None,
        usable_range=(int(low_mhz * 1_000_000), int(high_mhz * 1_000_000)),
        gain_dbi=2.0, polarization="vertical",
        requires_bias_power=False, notes="",
    )


def _dongle(idx, serial, antenna=None):
    """Synthetic RTL-SDR-class dongle covering 24 MHz to 1.7 GHz."""
    from rfcensus.hardware.dongle import Dongle, DongleCapabilities, DongleStatus
    caps = DongleCapabilities(
        freq_range_hz=(24_000_000, 1_700_000_000),
        max_sample_rate=2_400_000, bits_per_sample=8,
        bias_tee_capable=False, tcxo_ppm=10.0,
    )
    d = Dongle(
        id=f"rtl-{idx}", serial=serial, model="rtlsdr_generic",
        driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
        driver_index=idx,
    )
    d.antenna = antenna
    return d


# ──────────────────────────────────────────────────────────────────
# suggest_for_new_dongle
# ──────────────────────────────────────────────────────────────────


class TestSuggestForNewDongle:
    def test_returns_generic_when_fleet_covers_everything(self):
        """All enabled bands well-covered by other dongles → suggest
        generic small whip with a "we're good" rationale."""
        from rfcensus.hardware.antenna_suggestion import suggest_for_new_dongle

        whip_915 = _antenna("whip_915", 915)
        whip_433 = _antenna("whip_433", 433)
        catalog = [_antenna("whip_315", 315), whip_433, whip_915]

        # Two existing dongles cover 915 and 433
        existing = [
            _dongle(0, "00000001", antenna=whip_915),
            _dongle(1, "00000002", antenna=whip_433),
        ]
        # New dongle to assign
        new_d = _dongle(2, "00000003")
        # Enabled bands: only 915 and 433
        bands = [_band("ism_915", 915), _band("ism_433", 433)]

        s = suggest_for_new_dongle(new_d, existing, bands, catalog)
        assert s.antenna_id == "whip_generic_small"
        assert "already cover" in s.rationale.lower()
        assert s.bands_covered == []

    def test_returns_catalog_match_for_real_gap(self):
        """Fleet has 433 and 315 covered, new dongle could fill the
        915 MHz gap with whip_915 from catalog."""
        from rfcensus.hardware.antenna_suggestion import suggest_for_new_dongle

        whip_315 = _antenna("whip_315", 315)
        whip_433 = _antenna("whip_433", 433)
        whip_915 = _antenna("whip_915", 915)
        catalog = [whip_315, whip_433, whip_915]

        existing = [
            _dongle(0, "00000001", antenna=whip_315),
            _dongle(1, "00000002", antenna=whip_433),
        ]
        new_d = _dongle(2, "00000003")
        bands = [
            _band("tpms_315", 315),
            _band("ism_433", 433),
            _band("ism_915", 915),
        ]

        s = suggest_for_new_dongle(new_d, existing, bands, catalog)
        assert s.antenna_id == "whip_915"
        assert "ism_915" in s.bands_covered

    def test_quarter_wave_fallback_when_no_catalog_match(self):
        """If catalog doesn't have an antenna for the gap, recommend a
        quarter-wave whip with the calculated length."""
        from rfcensus.hardware.antenna_suggestion import suggest_for_new_dongle

        whip_315 = _antenna("whip_315", 315)
        # Catalog has only 315 MHz whip — no 1090 MHz dipole
        catalog = [whip_315]
        existing = [_dongle(0, "00000001", antenna=whip_315)]
        new_d = _dongle(1, "00000002")
        # Only enable a 1090 band — nothing in catalog fits
        bands = [_band("adsb_1090", 1090)]

        s = suggest_for_new_dongle(new_d, existing, bands, catalog)
        assert s.antenna_id is None
        assert s.is_quarter_wave_fallback
        assert s.fallback_freq_mhz == pytest.approx(1090, abs=1)
        # Quarter wave at 1090 MHz: c/(4f) = 6.88 cm
        assert s.fallback_length_cm == pytest.approx(6.88, abs=0.5)
        assert s.buy_suggestion is not None
        assert "1090" in s.buy_suggestion

    def test_no_other_dongles_returns_no_suggestion(self):
        """If there are no other dongles to be aware of, the function
        should not pretend to know what to do — it returns the
        same-fleet-coverage suggestion (generic small whip)."""
        from rfcensus.hardware.antenna_suggestion import suggest_for_new_dongle

        catalog = [_antenna("whip_915", 915)]
        new_d = _dongle(0, "00000001")
        bands = [_band("ism_915", 915)]

        # No other dongles
        s = suggest_for_new_dongle(new_d, [], bands, catalog)
        # Fleet is empty so all bands are "uncovered" — should suggest
        # the actual matching antenna
        assert s.antenna_id == "whip_915"

    def test_skips_bands_dongle_cant_reach(self):
        """If a band's frequency isn't in the dongle's capability range,
        it shouldn't be considered."""
        from rfcensus.hardware.antenna_suggestion import suggest_for_new_dongle
        from rfcensus.hardware.dongle import DongleCapabilities, Dongle, DongleStatus

        # Dongle limited to 100-200 MHz
        caps = DongleCapabilities(
            freq_range_hz=(100_000_000, 200_000_000),
            max_sample_rate=2_400_000, bits_per_sample=8,
            bias_tee_capable=False, tcxo_ppm=10.0,
        )
        new_d = Dongle(
            id="limited", serial="X", model="test",
            driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
            driver_index=0,
        )
        bands = [_band("ism_915", 915)]  # outside dongle range
        catalog = [_antenna("whip_915", 915)]
        existing = []

        s = suggest_for_new_dongle(new_d, existing, bands, catalog)
        # No bands the dongle can help with → "fleet already covers" path
        assert s.antenna_id == "whip_generic_small"


# ──────────────────────────────────────────────────────────────────
# optimize_fleet
# ──────────────────────────────────────────────────────────────────


class TestOptimizeFleet:
    def test_assigns_each_dongle_its_best_antenna(self):
        """Three dongles, three bands, three perfectly-tuned antennas —
        the optimizer should assign each dongle the matching antenna."""
        from rfcensus.hardware.fleet_optimizer import optimize_fleet

        dongles = [
            _dongle(0, "00000001"),
            _dongle(1, "00000002"),
            _dongle(2, "00000003"),
        ]
        catalog = [
            _antenna("whip_315", 315),
            _antenna("whip_433", 433),
            _antenna("whip_915", 915),
        ]
        bands = [
            _band("tpms_315", 315),
            _band("ism_433", 433),
            _band("ism_915", 915),
        ]

        plan = optimize_fleet(dongles, bands, catalog)
        # All 3 bands well-covered
        assert plan.well_covered_count == 3
        assert len(plan.uncovered_bands) == 0
        # Each dongle gets some matching antenna
        assert set(plan.assignments.values()) == {"whip_315", "whip_433", "whip_915"}

    def test_surfaces_uncovered_bands(self):
        """If catalog can't cover a band well, it appears in
        uncovered_bands and as a shopping suggestion."""
        from rfcensus.hardware.fleet_optimizer import optimize_fleet

        dongles = [_dongle(0, "00000001")]
        # Only have a 915 antenna; ADS-B (1090) isn't covered
        catalog = [_antenna("whip_915", 915)]
        bands = [
            _band("ism_915", 915),
            _band("adsb_1090", 1090),
        ]

        plan = optimize_fleet(dongles, bands, catalog)
        assert plan.well_covered_count == 1  # only ism_915
        assert "adsb_1090" in plan.uncovered_bands
        # Shopping suggestion for 1090
        assert len(plan.shopping_suggestions) >= 1
        assert plan.shopping_suggestions[0].target_freq_mhz == pytest.approx(1090, abs=10)

    def test_assigns_none_when_no_useful_antenna(self):
        """If no antenna in catalog helps a particular dongle for any
        band, the optimizer can leave it unassigned (None)."""
        from rfcensus.hardware.fleet_optimizer import optimize_fleet

        dongles = [_dongle(0, "00000001")]
        # Catalog has only an antenna for 1090 MHz, but only 915 band is enabled
        catalog = [_antenna("dipole_1090", 1090, usable_low_mhz=1080, usable_high_mhz=1100)]
        bands = [_band("ism_915", 915)]

        plan = optimize_fleet(dongles, bands, catalog)
        # No useful antenna → assignment is None
        assert plan.assignments["rtl-0"] is None
        assert "ism_915" in plan.uncovered_bands

    def test_handles_empty_inputs_gracefully(self):
        from rfcensus.hardware.fleet_optimizer import optimize_fleet
        plan = optimize_fleet([], [], [])
        assert plan.assignments == {}
        assert plan.well_covered_count == 0


class TestDiffAgainstCurrent:
    def test_identifies_changes_and_unchanged(self):
        from rfcensus.hardware.fleet_optimizer import (
            FleetPlan, diff_against_current,
        )
        plan = FleetPlan(assignments={
            "rtl-0": "whip_915",
            "rtl-1": "whip_433",
            "rtl-2": None,
        })
        current = {
            "rtl-0": "whip_915",   # unchanged
            "rtl-1": "whip_315",   # changed
            "rtl-2": "whip_915",   # changed (to None)
        }
        diff = diff_against_current(plan, current)
        assert diff.unchanged == ["rtl-0"]
        assert ("rtl-1", "whip_315", "whip_433") in diff.changes
        assert ("rtl-2", "whip_915", None) in diff.changes


# ──────────────────────────────────────────────────────────────────
# CLI surface — just confirm the command is wired up
# ──────────────────────────────────────────────────────────────────


class TestSuggestCommand:
    def test_command_registered(self):
        from rfcensus.cli import main
        cmd_names = list(main.commands.keys())
        assert "suggest" in cmd_names

    def test_antennas_subcommand_registered(self):
        from rfcensus.commands.suggest import cli
        sub_names = list(cli.commands.keys())
        assert "antennas" in sub_names

    def test_apply_and_yes_are_aliases(self):
        from rfcensus.commands.suggest import cli
        antennas = cli.commands["antennas"]
        opts = {p.name for p in antennas.params}
        assert "apply" in opts
        assert "yes" in opts
