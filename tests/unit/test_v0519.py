"""Tests for v0.5.19 — strategy-aware _estimate_dongle_needs.

v0.5.16's estimator over-counted for two common cases:

1. power_primary bands (marine_vhf, 70cm_amateur, business_uhf) — run
   ONLY rtl_power by design, no decoders. But the estimator's
   max(n_exclusive, 1) floor and empty-suggested_decoders handling
   produced needs=3 for 70cm_amateur when the truth is 1.

2. decoder_primary bands listing a DETECTOR name in suggested_decoders
   (p25_700 and p25_800 list "p25" which is a detector, not a decoder).
   No actual decoder matches, runtime only spawns rtl_power.

v0.5.19 makes the estimator strategy-aware so planning matches runtime.
"""

from __future__ import annotations


def _band(**kwargs):
    from rfcensus.config.schema import BandConfig, StrategyKind
    defaults = {
        "id": "b", "name": "b",
        "freq_low": 433_000_000, "freq_high": 434_000_000,
        "suggested_decoders": [],
        "strategy": StrategyKind.DECODER_PRIMARY,
        "power_scan_parallel": False,
    }
    defaults.update(kwargs)
    return BandConfig(**defaults)


class TestPowerPrimaryIsOneDongle:
    def test_power_primary_with_no_decoders_needs_one(self):
        from rfcensus.config.schema import StrategyKind
        from rfcensus.engine.scheduler import _estimate_dongle_needs
        from rfcensus.decoders.registry import get_registry

        band = _band(
            strategy=StrategyKind.POWER_PRIMARY,
            suggested_decoders=[],
            freq_low=420_000_000, freq_high=450_000_000,
        )
        assert _estimate_dongle_needs(band, get_registry()) == 1

    def test_power_primary_with_suggested_decoders_still_needs_one(self):
        from rfcensus.config.schema import StrategyKind
        from rfcensus.engine.scheduler import _estimate_dongle_needs
        from rfcensus.decoders.registry import get_registry

        band = _band(
            strategy=StrategyKind.POWER_PRIMARY,
            suggested_decoders=["multimon"],
            freq_low=150_000_000, freq_high=174_000_000,
        )
        assert _estimate_dongle_needs(band, get_registry()) == 1

    def test_exploration_strategy_needs_one(self):
        from rfcensus.config.schema import StrategyKind
        from rfcensus.engine.scheduler import _estimate_dongle_needs
        from rfcensus.decoders.registry import get_registry

        band = _band(strategy=StrategyKind.EXPLORATION)
        assert _estimate_dongle_needs(band, get_registry()) == 1


class TestDetectorMisconfigFallsBackToOne:
    def test_p25_band_with_only_detector_name_needs_one(self):
        from rfcensus.config.schema import StrategyKind
        from rfcensus.engine.scheduler import _estimate_dongle_needs
        from rfcensus.decoders.registry import get_registry

        band = _band(
            strategy=StrategyKind.DECODER_PRIMARY,
            suggested_decoders=["p25"],
            freq_low=764_000_000, freq_high=776_000_000,
        )
        assert _estimate_dongle_needs(band, get_registry()) == 1

    def test_decoder_primary_with_no_real_decoders_needs_one(self):
        from rfcensus.config.schema import StrategyKind
        from rfcensus.engine.scheduler import _estimate_dongle_needs
        from rfcensus.decoders.registry import get_registry

        band = _band(
            strategy=StrategyKind.DECODER_PRIMARY,
            suggested_decoders=["nonexistent_decoder_xyz"],
            freq_low=902_000_000, freq_high=928_000_000,
        )
        assert _estimate_dongle_needs(band, get_registry()) == 1


class TestDecoderPrimaryStillCountsCorrectly:
    def test_915_ism_with_rtl_433_and_rtlamr_and_power_scan(self):
        """Both rtl_433 and rtlamr are SHARED — they co-tenant on one
        shared dongle. Plus rtl_power exclusive. Total: 2 dongles."""
        from rfcensus.config.schema import StrategyKind
        from rfcensus.engine.scheduler import _estimate_dongle_needs
        from rfcensus.decoders.registry import get_registry

        band = _band(
            strategy=StrategyKind.DECODER_PRIMARY,
            suggested_decoders=["rtl_433", "rtlamr"],
            freq_low=902_000_000, freq_high=928_000_000,
            power_scan_parallel=True,
        )
        assert _estimate_dongle_needs(band, get_registry()) == 2

    def test_decoder_only_rtl_ais_needs_one(self):
        from rfcensus.config.schema import StrategyKind
        from rfcensus.engine.scheduler import _estimate_dongle_needs
        from rfcensus.decoders.registry import get_registry

        band = _band(
            strategy=StrategyKind.DECODER_ONLY,
            suggested_decoders=["rtl_ais"],
            freq_low=161_900_000, freq_high=162_100_000,
        )
        assert _estimate_dongle_needs(band, get_registry()) == 1

    def test_narrow_decoder_primary_no_power_scan_sidecar(self):
        from rfcensus.config.schema import StrategyKind
        from rfcensus.engine.scheduler import _estimate_dongle_needs
        from rfcensus.decoders.registry import get_registry

        band = _band(
            strategy=StrategyKind.DECODER_PRIMARY,
            suggested_decoders=["rtl_433"],
            freq_low=433_000_000, freq_high=434_000_000,
        )
        assert _estimate_dongle_needs(band, get_registry()) == 1


class TestUserScenarioCollapsesTo5Waves:
    def test_full_plan_stays_5_waves_with_17_bands(self):
        """User's real fleet + builtin bands should plan into 5 waves
        (VHF serialization floor), not 7 as in v0.5.18.

        With v0.5.32's addition of 915_ism_r900 as a second pass at
        912.6 MHz, the plan grew from 16 tasks to 17 — but wave count
        stays at 5 because the new task slots into existing idle time
        on the second whip_915 dongle. This is the expected behavior
        that motivated Option A (separate band, not intrinsic
        multi-pass): free coverage from idle fleet capacity.
        """
        from rfcensus.config.loader import load_config
        from rfcensus.decoders.registry import get_registry
        from rfcensus.engine.scheduler import Scheduler
        from rfcensus.events import EventBus
        from rfcensus.hardware.antenna import Antenna
        from rfcensus.hardware.broker import DongleBroker
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
        )
        from rfcensus.hardware.registry import HardwareRegistry

        def ant(aid, mhz, low, high):
            return Antenna(
                id=aid, name=aid, antenna_type="whip",
                resonant_freq_hz=int(mhz * 1e6),
                usable_range=(int(low * 1e6), int(high * 1e6)),
                gain_dbi=2.15, polarization="vertical",
                requires_bias_power=False, notes="",
            )

        def d(serial, a, model="rtlsdr_generic"):
            caps = DongleCapabilities(
                freq_range_hz=(24_000_000, 1_700_000_000),
                max_sample_rate=2_400_000, bits_per_sample=8,
                bias_tee_capable=False, tcxo_ppm=10.0,
                can_share_via_rtl_tcp=True,
            )
            dd = Dongle(
                id=f"rtlsdr-{serial}", serial=serial, model=model,
                driver="rtlsdr", capabilities=caps,
                status=DongleStatus.HEALTHY, driver_index=0,
            )
            dd.antenna = a
            return dd

        dongles = [
            d("00000003", ant("whip_915", 915, 594, 1235)),
            d("00000002", ant("whip_433", 433, 282, 586)),
            d("00000001", ant("whip_915", 915, 594, 1235), model="rtlsdr_v4"),
            d("00000043", ant("whip_315", 315, 204, 425)),
            d("07262454", ant("marine_vhf", 156.8, 137, 180),
              model="nesdr_smart_v5"),
        ]

        config = load_config()
        broker = DongleBroker(HardwareRegistry(dongles=dongles), EventBus())
        plan = Scheduler(
            config, broker, decoder_registry=get_registry(),
        ).plan(config.enabled_bands())

        # Wave count is the key invariant — it determines wall-time.
        # VHF bands (marine_vhf, nws_weather, ais, business_vhf,
        # aprs_2m) all need the same dongle (whip_156) and serialize
        # into 5 waves. Adding more bands at other frequencies
        # should slot into idle time without growing wave count.
        assert len(plan.waves) == 5, (
            f"expected 5 waves (VHF serialization floor), got "
            f"{len(plan.waves)}. New bands should fit in idle "
            f"dongle time, not extend the plan."
        )
        # 17 tasks = 16 original + 915_ism_r900 added in v0.5.32
        assert len(plan.tasks) == 17
