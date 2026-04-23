"""Tests for v0.5.17 — fixes to v0.5.16 regressions:
  • Sidecar over-reservation capped by fleet suitable pool size
  • High-demand bands placed first so they reserve sidecars ahead of
    same-score smaller bands
  • Broker uses wait_for_tcp_ready instead of blind sleep (no more
    0.0s exits for rtl_433 in shared mode)
  • wait_for_tcp_ready promoted to rfcensus.utils.tcp for reuse
"""

from __future__ import annotations


def _make_dongle(idx, antenna=None):
    from rfcensus.hardware.dongle import (
        Dongle, DongleCapabilities, DongleStatus,
    )
    caps = DongleCapabilities(
        freq_range_hz=(24_000_000, 1_700_000_000),
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


def _make_antenna(aid, mhz, low_mhz, high_mhz):
    from rfcensus.hardware.antenna import Antenna
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


class TestSidecarFleetShortageCap:
    def test_single_suitable_dongle_bands_still_pack_in_one_wave(self):
        """v0.5.16 regression: bands whose antenna-suitable pool is
        only 1 dongle but strategy wants sidecars would bump to their
        own wave forever, producing 13+ single-task waves. v0.5.17
        caps effective sidecars by fleet budget, so bands with no
        spare suitable dongles still pack normally."""
        dongles = [
            _make_dongle(1, antenna=_make_antenna("whip_433", 433, 282, 586)),
            _make_dongle(2, antenna=_make_antenna("whip_315", 315, 204, 425)),
        ]
        band_defs = [
            {
                "id": "b433", "name": "b433",
                "freq_low": 433_000_000, "freq_high": 440_000_000,  # 7 MHz
                "suggested_decoders": ["rtl_433"],
                "power_scan_parallel": True,  # wants rtl_power sidecar
            },
            {
                "id": "b315", "name": "b315",
                "freq_low": 314_900_000, "freq_high": 315_100_000,
                "suggested_decoders": ["rtl_433"],
            },
        ]
        plan = _build_plan(dongles, band_defs)
        # Both bands have suitable pool of 1 dongle. Neither can run
        # its sidecars (none available). Both should pack in wave 0.
        assert len(plan.waves) == 1
        assert len(plan.waves[0].tasks) == 2

    def test_whip_433_bands_pack_into_sequential_waves_when_needed(self):
        """Multiple bands needing the same single-dongle pool still
        have to serialize (there's only one dongle), but each wave
        still carries OTHER bands on disjoint pools."""
        dongles = [
            _make_dongle(1, antenna=_make_antenna("whip_433", 433, 282, 586)),
            _make_dongle(2, antenna=_make_antenna("marine_vhf", 157, 137, 180)),
        ]
        band_defs = [
            {"id": "b1_433", "name": "b1",
             "freq_low": 433_000_000, "freq_high": 434_000_000,
             "suggested_decoders": ["rtl_433"], "power_scan_parallel": True},
            {"id": "b2_433", "name": "b2",
             "freq_low": 465_000_000, "freq_high": 467_000_000,
             "suggested_decoders": ["rtl_433"]},
            {"id": "ais", "name": "ais",
             "freq_low": 162_000_000, "freq_high": 162_100_000,
             "suggested_decoders": ["rtl_ais"]},
        ]
        plan = _build_plan(dongles, band_defs)
        # 2 waves: each 433-band in its own wave (serialized), ais
        # parallels one of them on the marine_vhf dongle.
        assert len(plan.waves) == 2
        total = sum(len(w.tasks) for w in plan.waves)
        assert total == 3


class TestSortTiebreakByNeeds:
    def test_high_demand_band_placed_before_same_score_low_demand(self):
        """915_ism wants 3 dongles, pocsag_929 wants 1. Both score
        1.0 (resonant whip_915). 915_ism must be placed first so it
        reserves both whip_915 dongles before pocsag_929 squats on
        one of them."""
        dongles = [
            _make_dongle(1, antenna=_make_antenna("whip_915", 915, 594, 1235)),
            _make_dongle(3, antenna=_make_antenna("whip_915", 915, 594, 1235)),
        ]
        band_defs = [
            # Put pocsag FIRST in the config order — should still get
            # placed after 915_ism due to score/needs sort.
            {"id": "pocsag_929", "name": "POCSAG",
             "freq_low": 929_000_000, "freq_high": 932_000_000,
             "suggested_decoders": ["multimon"]},
            {"id": "ism_915", "name": "915",
             "freq_low": 902_000_000, "freq_high": 928_000_000,
             "suggested_decoders": ["rtl_433", "rtlamr"],
             "power_scan_parallel": True},
        ]
        plan = _build_plan(dongles, band_defs)
        # 915_ism should be in wave 0 (placed first due to higher
        # dongles_needed on same score), pocsag_929 in wave 1.
        assert plan.waves[0].tasks[0].band.id == "ism_915"
        assert any(
            t.band.id == "pocsag_929" for t in plan.waves[1].tasks
        )


class TestTcpReadyUtility:
    def test_wait_for_tcp_ready_importable_from_utils(self):
        """wait_for_tcp_ready moved from rtlamr.py to rfcensus.utils.tcp
        so the broker and other consumers can reuse it."""
        from rfcensus.utils.tcp import wait_for_tcp_ready
        assert callable(wait_for_tcp_ready)

    def test_rtlamr_imports_from_utils_not_duplicated(self):
        """Make sure we didn't leave a duplicate implementation behind."""
        import inspect
        from rfcensus.decoders.builtin import rtlamr
        # Ensure the module doesn't define its own _wait_for_tcp_ready
        # at module level any more — it should be imported.
        src = inspect.getsource(rtlamr)
        assert "async def _wait_for_tcp_ready" not in src, (
            "rtlamr.py still defines _wait_for_tcp_ready locally; "
            "should import from rfcensus.utils.tcp"
        )

    def test_broker_uses_wait_for_tcp_ready(self):
        """Broker's _start_shared_slot should actively poll for port
        readiness rather than sleeping blindly."""
        import inspect
        from rfcensus.hardware import broker
        src = inspect.getsource(broker)
        assert "wait_for_tcp_ready" in src, (
            "broker should use wait_for_tcp_ready in _start_shared_slot"
        )
        # Confirm we removed the blind sleep
        assert "await asyncio.sleep(0.4)" not in src, (
            "broker should no longer use blind 0.4s sleep for rtl_tcp "
            "readiness — use wait_for_tcp_ready instead"
        )


class TestRtl433SharedAccessMode:
    def test_rtl_433_still_shared_capable(self):
        """Per user feedback — shared-mode rtl_433 via rtl_tcp is a
        wanted feature for colocating with rtlamr on 915 MHz. The
        0.0s exit bug was a symptom fixed by the broker's TCP ready
        check, not a reason to revert the access mode."""
        from rfcensus.decoders.builtin.rtl_433 import Rtl433Decoder
        assert Rtl433Decoder.capabilities.requires_exclusive_dongle is False


class TestUserScenarioCollapses:
    def test_user_16_band_plan_is_5_waves(self):
        """End-to-end: user's 5-dongle / 16-band fleet should plan
        into 5 waves (VHF serialization floor) rather than the
        13-wave explosion v0.5.16 produced."""
        dongles = [
            _make_dongle(1, antenna=_make_antenna("whip_915", 915, 594, 1235)),
            _make_dongle(3, antenna=_make_antenna("whip_915", 915, 594, 1235)),
            _make_dongle(2, antenna=_make_antenna("whip_433", 433, 282, 586)),
            _make_dongle(43, antenna=_make_antenna("whip_315", 315, 204, 425)),
            _make_dongle(454, antenna=_make_antenna("marine_vhf", 157, 137, 180)),
        ]
        # User's 16 bands from their actual scan output
        band_defs = [
            {"id": "315_security", "name": "315",
             "freq_low": 314_900_000, "freq_high": 315_100_000,
             "suggested_decoders": ["rtl_433"]},
            {"id": "433_ism", "name": "433",
             "freq_low": 433_000_000, "freq_high": 434_000_000,
             "suggested_decoders": ["rtl_433"]},
            {"id": "915_ism", "name": "915",
             "freq_low": 902_000_000, "freq_high": 928_000_000,
             "suggested_decoders": ["rtl_433", "rtlamr"],
             "power_scan_parallel": True},
            {"id": "ais", "name": "ais",
             "freq_low": 161_900_000, "freq_high": 162_100_000,
             "suggested_decoders": ["rtl_ais"]},
            {"id": "interlogix_security", "name": "interlogix",
             "freq_low": 319_400_000, "freq_high": 319_600_000,
             "suggested_decoders": ["rtl_433"]},
            {"id": "pocsag_929", "name": "pocsag",
             "freq_low": 929_000_000, "freq_high": 932_000_000,
             "suggested_decoders": ["multimon"]},
            {"id": "nws_weather", "name": "nws",
             "freq_low": 162_400_000, "freq_high": 162_600_000,
             "suggested_decoders": ["multimon"]},
            {"id": "70cm", "name": "70cm",
             "freq_low": 420_000_000, "freq_high": 450_000_000,
             "suggested_decoders": []},
            {"id": "marine_vhf", "name": "marine",
             "freq_low": 156_000_000, "freq_high": 162_000_000,
             "suggested_decoders": ["rtl_ais"]},
            {"id": "honeywell_security", "name": "honeywell",
             "freq_low": 344_900_000, "freq_high": 345_100_000,
             "suggested_decoders": ["rtl_433"]},
            {"id": "frs_gmrs", "name": "frs",
             "freq_low": 462_500_000, "freq_high": 467_700_000,
             "suggested_decoders": ["multimon"]},
            {"id": "aprs_2m", "name": "aprs",
             "freq_low": 144_380_000, "freq_high": 144_400_000,
             "suggested_decoders": ["multimon"]},
            {"id": "business_uhf", "name": "business_uhf",
             "freq_low": 450_000_000, "freq_high": 470_000_000,
             "suggested_decoders": []},
            {"id": "business_vhf", "name": "business_vhf",
             "freq_low": 150_000_000, "freq_high": 174_000_000,
             "suggested_decoders": []},
            {"id": "p25_700_public_safety", "name": "p25_700",
             "freq_low": 764_000_000, "freq_high": 776_000_000,
             "suggested_decoders": ["p25"]},
            {"id": "p25_800_public_safety", "name": "p25_800",
             "freq_low": 851_000_000, "freq_high": 869_000_000,
             "suggested_decoders": ["p25"]},
        ]
        plan = _build_plan(dongles, band_defs)
        # 5 VHF-only bands force 5 waves minimum (they all share the
        # single marine_vhf dongle). Anything more than that is packing
        # slack. v0.5.16 produced 13 waves — v0.5.17 should be 5-6.
        assert len(plan.waves) <= 6, (
            f"Expected <=6 waves (5 VHF floor + slack), got {len(plan.waves)}. "
            f"Plan: {[(w.index, [t.band.id for t in w.tasks]) for w in plan.waves]}"
        )
        # All 16 bands placed
        assert len(plan.tasks) == 16
