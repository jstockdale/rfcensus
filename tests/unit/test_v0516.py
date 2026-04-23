"""Tests for v0.5.16 — sidecar-aware wave packing.

Before this version, the scheduler's wave packing only tracked each
band's PRIMARY dongle. Sidecars (rtl_power spectrum scan, rtlamr via
rtl_tcp) were allocated at runtime and could starve other bands that
needed the same antenna-suitable dongles.

Now the scheduler:
  • Estimates total dongle demand per band via _estimate_dongle_needs
  • Reserves sidecar slots from the band's antenna-suitable pool
  • Bumps bands to new waves when the current wave can't host sidecars
"""

from __future__ import annotations

import pytest


def _make_dongle(idx, antenna=None, freq_hz_range=(24_000_000, 1_700_000_000)):
    from rfcensus.hardware.dongle import (
        Dongle, DongleCapabilities, DongleStatus,
    )
    caps = DongleCapabilities(
        freq_range_hz=freq_hz_range,
        max_sample_rate=2_400_000, bits_per_sample=8,
        bias_tee_capable=False, tcxo_ppm=10.0,
        can_share_via_rtl_tcp=True,
    )
    d = Dongle(
        id=f"rtl-{idx}", serial=f"S{idx}", model="rtlsdr_generic",
        driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
        driver_index=idx,
    )
    d.antenna = antenna
    return d


def _make_antenna(aid, mhz, low_mhz=None, high_mhz=None):
    from rfcensus.hardware.antenna import Antenna
    if low_mhz is None: low_mhz = mhz * 0.85
    if high_mhz is None: high_mhz = mhz * 1.15
    return Antenna(
        id=aid, name=aid, antenna_type="whip",
        resonant_freq_hz=int(mhz * 1_000_000),
        usable_range=(int(low_mhz * 1_000_000), int(high_mhz * 1_000_000)),
        gain_dbi=2.15, polarization="vertical",
        requires_bias_power=False, notes="",
    )


def _build_plan(dongles, band_defs):
    from rfcensus.config.schema import SiteConfig
    from rfcensus.engine.scheduler import Scheduler
    from rfcensus.events import EventBus
    from rfcensus.hardware.broker import DongleBroker
    from rfcensus.hardware.registry import HardwareRegistry

    config = SiteConfig.model_validate({
        "site": {"name": "test"},
        "antennas": [],
        "bands": {"enabled": [b["id"] for b in band_defs]},
        "band_definitions": band_defs,
        "dongles": [],
    })
    broker = DongleBroker(HardwareRegistry(dongles=dongles), EventBus())
    return Scheduler(config, broker).plan(config.band_definitions)


# ──────────────────────────────────────────────────────────────────
# _estimate_dongle_needs
# ──────────────────────────────────────────────────────────────────


class TestEstimateDongleNeeds:
    def test_single_decoder_no_sidecar_needs_one(self):
        from rfcensus.config.schema import BandConfig
        from rfcensus.engine.scheduler import _estimate_dongle_needs

        band = BandConfig(
            id="b", name="b",
            freq_low=162_000_000, freq_high=162_100_000,
            suggested_decoders=["rtl_ais"],
        )
        assert _estimate_dongle_needs(band) == 1

    def test_rtlamr_adds_shared_sidecar(self):
        """A band with rtlamr among decoders needs +1 shared dongle."""
        from rfcensus.config.schema import BandConfig
        from rfcensus.engine.scheduler import _estimate_dongle_needs

        band = BandConfig(
            id="b", name="b",
            freq_low=902_000_000, freq_high=928_000_000,
            suggested_decoders=["rtl_433", "rtlamr"],
        )
        # rtl_433 exclusive + rtlamr shared → 2 total, plus power_scan
        # for wide band (26 MHz > 5 MHz threshold) → 3
        assert _estimate_dongle_needs(band) == 3

    def test_wide_band_triggers_power_scan_slot(self):
        """A band >= 5 MHz wide automatically triggers rtl_power (+1)."""
        from rfcensus.config.schema import BandConfig
        from rfcensus.engine.scheduler import _estimate_dongle_needs

        band = BandConfig(
            id="wide", name="wide",
            freq_low=450_000_000, freq_high=470_000_000,  # 20 MHz
            suggested_decoders=["rtl_433"],
        )
        assert _estimate_dongle_needs(band) == 2  # primary + power scan

    def test_power_scan_parallel_flag_triggers_sidecar(self):
        from rfcensus.config.schema import BandConfig
        from rfcensus.engine.scheduler import _estimate_dongle_needs

        band = BandConfig(
            id="b", name="b",
            freq_low=433_000_000, freq_high=433_500_000,  # narrow
            suggested_decoders=["rtl_433"],
            power_scan_parallel=True,  # explicit
        )
        assert _estimate_dongle_needs(band) == 2

    def test_empty_decoders_minimum_one(self):
        """A band with no decoders still gets a minimum of 1 slot."""
        from rfcensus.config.schema import BandConfig
        from rfcensus.engine.scheduler import _estimate_dongle_needs

        band = BandConfig(
            id="b", name="b",
            freq_low=100_000_000, freq_high=100_100_000,
            suggested_decoders=[],
        )
        assert _estimate_dongle_needs(band) == 1


# ──────────────────────────────────────────────────────────────────
# Wave packing respects sidecars
# ──────────────────────────────────────────────────────────────────


class TestSidecarAwareWavePacking:
    def test_two_whip_915_bands_with_sidecars_bump_to_separate_waves(self):
        """The user's case: 915_ism needs 3 dongles (rtl_433 + rtlamr +
        power_scan) and there are only 2 whip_915 dongles. pocsag_929
        wants the same dongles. They can't share a wave."""
        dongles = [
            _make_dongle(1, antenna=_make_antenna("whip_915", 915, 594, 1235)),
            _make_dongle(3, antenna=_make_antenna("whip_915", 915, 594, 1235)),
        ]
        band_defs = [
            {
                "id": "ism_915", "name": "915",
                "freq_low": 902_000_000, "freq_high": 928_000_000,
                "suggested_decoders": ["rtl_433", "rtlamr"],
                "power_scan_parallel": True,
            },
            {
                "id": "pocsag_929", "name": "POCSAG",
                "freq_low": 929_000_000, "freq_high": 932_000_000,
                "suggested_decoders": ["multimon"],
            },
        ]
        plan = _build_plan(dongles, band_defs)
        # 915_ism alone in wave 0 (consuming both whip_915 dongles as
        # primary + sidecar reservation), pocsag_929 in wave 1.
        assert len(plan.waves) == 2
        wave_ids = [[t.band.id for t in w.tasks] for w in plan.waves]
        assert "ism_915" in wave_ids[0]
        assert "pocsag_929" in wave_ids[1]

    def test_bands_with_different_antennas_pack_into_same_wave(self):
        """Different-antenna bands (no shared whip pool) should pack
        into the same wave even if one has sidecars."""
        dongles = [
            _make_dongle(1, antenna=_make_antenna("whip_915", 915, 594, 1235)),
            _make_dongle(2, antenna=_make_antenna("whip_433", 433, 282, 586)),
            _make_dongle(3, antenna=_make_antenna("whip_915", 915, 594, 1235)),
        ]
        band_defs = [
            {
                "id": "ism_915", "name": "915",
                "freq_low": 902_000_000, "freq_high": 928_000_000,
                "suggested_decoders": ["rtl_433", "rtlamr"],
                "power_scan_parallel": True,
            },
            {
                "id": "ism_433", "name": "433",
                "freq_low": 433_000_000, "freq_high": 434_000_000,
                "suggested_decoders": ["rtl_433"],
            },
        ]
        plan = _build_plan(dongles, band_defs)
        # Both bands in wave 0 — 915 consumes both whip_915 dongles for
        # its sidecars, 433 uses its own whip_433 dongle (disjoint pool).
        assert len(plan.waves) == 1
        assert len(plan.waves[0].tasks) == 2

    def test_single_dongle_band_packs_normally(self):
        """Simple case: bands with no sidecars and distinct primaries
        pack into a single wave (existing behavior preserved)."""
        dongles = [
            _make_dongle(1, antenna=_make_antenna("whip_433", 433, 282, 586)),
            _make_dongle(2, antenna=_make_antenna("whip_315", 315, 204, 425)),
        ]
        band_defs = [
            {
                "id": "ism_433", "name": "433",
                "freq_low": 433_000_000, "freq_high": 434_000_000,
                "suggested_decoders": ["rtl_433"],
            },
            {
                "id": "tpms_315", "name": "315",
                "freq_low": 314_900_000, "freq_high": 315_100_000,
                "suggested_decoders": ["rtl_433"],
            },
        ]
        plan = _build_plan(dongles, band_defs)
        assert len(plan.waves) == 1
        assert len(plan.waves[0].tasks) == 2

    def test_task_records_dongles_needed(self):
        """ScheduleTask.dongles_needed is set by the planner so downstream
        code (diagnostics, degradation warnings) can see the requirement."""
        dongles = [
            _make_dongle(1, antenna=_make_antenna("whip_915", 915, 594, 1235)),
        ]
        band_defs = [{
            "id": "ism_915", "name": "915",
            "freq_low": 902_000_000, "freq_high": 928_000_000,
            "suggested_decoders": ["rtl_433", "rtlamr"],
            "power_scan_parallel": True,
        }]
        plan = _build_plan(dongles, band_defs)
        assert plan.tasks[0].dongles_needed == 3

    def test_fleet_shortage_places_band_anyway_with_best_effort(self):
        """If a band wants more dongles than exist, place it in a wave
        and reserve what's available. Runtime handles the shortage by
        skipping some sidecars (v0.5.14 deferral logic)."""
        # Fleet has 1 whip_915 dongle. 915_ism wants 3 but we only have 1.
        dongles = [
            _make_dongle(1, antenna=_make_antenna("whip_915", 915, 594, 1235)),
        ]
        band_defs = [{
            "id": "ism_915", "name": "915",
            "freq_low": 902_000_000, "freq_high": 928_000_000,
            "suggested_decoders": ["rtl_433", "rtlamr"],
            "power_scan_parallel": True,
        }]
        plan = _build_plan(dongles, band_defs)
        # Should still produce a valid plan with the band placed
        assert len(plan.waves) == 1
        assert plan.waves[0].tasks[0].band.id == "ism_915"
        # dongles_needed reflects the strategy's demand even if fleet is short
        assert plan.waves[0].tasks[0].dongles_needed == 3


# ──────────────────────────────────────────────────────────────────
# End-to-end: user's 5-dongle scenario
# ──────────────────────────────────────────────────────────────────


class TestUsersSceneario:
    def test_pocsag_bumps_to_wave_1_when_915_needs_sidecars(self):
        """The user's actual scenario: 915_ism with rtlamr+rtl_power
        should correctly take both whip_915 dongles for its wave, and
        pocsag_929 should go to wave 1 to use the now-free whip_915."""
        dongles = [
            _make_dongle(3, antenna=_make_antenna("whip_915", 915, 594, 1235)),
            _make_dongle(2, antenna=_make_antenna("whip_433", 433, 282, 586)),
            _make_dongle(1, antenna=_make_antenna("whip_915", 915, 594, 1235)),
            _make_dongle(43, antenna=_make_antenna("whip_315", 315, 204, 425)),
            _make_dongle(454, antenna=_make_antenna("marine_vhf", 156.8, 137, 180)),
        ]
        band_defs = [
            {
                "id": "ism_915", "name": "915",
                "freq_low": 902_000_000, "freq_high": 928_000_000,
                "suggested_decoders": ["rtl_433", "rtlamr"],
                "power_scan_parallel": True,
            },
            {
                "id": "pocsag_929", "name": "POCSAG",
                "freq_low": 929_000_000, "freq_high": 932_000_000,
                "suggested_decoders": ["multimon"],
            },
            {
                "id": "ism_433", "name": "433",
                "freq_low": 433_000_000, "freq_high": 434_000_000,
                "suggested_decoders": ["rtl_433"],
            },
            {
                "id": "ais", "name": "AIS",
                "freq_low": 161_900_000, "freq_high": 162_100_000,
                "suggested_decoders": ["rtl_ais"],
            },
            {
                "id": "tpms_315", "name": "315",
                "freq_low": 314_900_000, "freq_high": 315_100_000,
                "suggested_decoders": ["rtl_433"],
            },
        ]
        plan = _build_plan(dongles, band_defs)

        # Find which wave has each band
        band_to_wave = {}
        for i, w in enumerate(plan.waves):
            for t in w.tasks:
                band_to_wave[t.band.id] = i

        # ism_915 + pocsag_929 must NOT be in the same wave (they share
        # the 2-dongle whip_915 pool, and 915_ism claims both)
        assert band_to_wave["ism_915"] != band_to_wave["pocsag_929"], (
            f"915_ism and pocsag_929 in same wave; "
            f"would cause sidecar conflict. Plan: {band_to_wave}"
        )
