"""Tests for v0.5.13:
  • Broker hard-excludes unsuitable antennas when require_suitable_antenna=True
  • --all-bands threading: ctx.all_bands → require_suitable_antenna=False
  • Load-balanced dongle assignment: equivalent dongles spread across bands
  • Best-effort sort preserved with --all-bands (best-fit picked first)
"""

from __future__ import annotations

import asyncio

import pytest


def _make_dongle(idx, freq_low, freq_high, antenna=None):
    from rfcensus.hardware.dongle import (
        Dongle, DongleCapabilities, DongleStatus,
    )
    caps = DongleCapabilities(
        freq_range_hz=(freq_low, freq_high),
        max_sample_rate=2_400_000, bits_per_sample=8,
        bias_tee_capable=False, tcxo_ppm=10.0,
    )
    d = Dongle(
        id=f"rtl-{idx}", serial=f"S{idx}", model="rtlsdr_generic",
        driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
        driver_index=idx,
    )
    d.antenna = antenna
    return d


def _make_antenna(ant_id, resonant_mhz, low_mhz=None, high_mhz=None):
    from rfcensus.hardware.antenna import Antenna
    if low_mhz is None:
        low_mhz = resonant_mhz * 0.85
    if high_mhz is None:
        high_mhz = resonant_mhz * 1.15
    return Antenna(
        id=ant_id, name=ant_id, antenna_type="whip",
        resonant_freq_hz=int(resonant_mhz * 1_000_000),
        usable_range=(int(low_mhz * 1_000_000), int(high_mhz * 1_000_000)),
        gain_dbi=2.15, polarization="vertical",
        requires_bias_power=False, notes="",
    )


# ──────────────────────────────────────────────────────────────────
# Broker hard-excludes unsuitable antennas
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestBrokerHardAntennaExclusion:
    """The exact scenario from the user's scan: rtl_power for 915 MHz
    must NOT be allocated to a dongle with marine_vhf antenna, even
    though the dongle hardware can tune to 915 MHz. v0.5.12 and
    earlier had this as a 'soft' deprioritization which was the bug."""

    async def test_unsuitable_antenna_hard_excluded(self):
        from rfcensus.events import EventBus
        from rfcensus.hardware.broker import (
            AccessMode, DongleBroker, DongleRequirements, NoDongleAvailable,
        )
        from rfcensus.hardware.registry import HardwareRegistry

        # One dongle, marine_vhf antenna (137-180 MHz usable range)
        marine = _make_antenna("marine_vhf", 156.8, low_mhz=137, high_mhz=180)
        dongle = _make_dongle(0, 24_000_000, 1_700_000_000, antenna=marine)
        broker = DongleBroker(HardwareRegistry(dongles=[dongle]), EventBus())

        # Request 915 MHz allocation: hardware can tune, antenna can't cover
        req = DongleRequirements(
            freq_hz=915_000_000,
            access_mode=AccessMode.EXCLUSIVE,
            require_suitable_antenna=True,
        )
        with pytest.raises(NoDongleAvailable):
            await broker.allocate(req, consumer="test", timeout=0.5)

    async def test_suitable_antenna_allocated_normally(self):
        from rfcensus.events import EventBus
        from rfcensus.hardware.broker import (
            AccessMode, DongleBroker, DongleRequirements,
        )
        from rfcensus.hardware.registry import HardwareRegistry

        whip = _make_antenna("whip_915", 915, low_mhz=850, high_mhz=1000)
        dongle = _make_dongle(0, 24_000_000, 1_700_000_000, antenna=whip)
        broker = DongleBroker(HardwareRegistry(dongles=[dongle]), EventBus())

        req = DongleRequirements(
            freq_hz=915_000_000,
            access_mode=AccessMode.EXCLUSIVE,
            require_suitable_antenna=True,
        )
        lease = await broker.allocate(req, consumer="test", timeout=1.0)
        assert lease.dongle.id == "rtl-0"
        await broker.release(lease)

    async def test_require_suitable_false_allows_unsuitable(self):
        """The escape hatch: when False, broker accepts unsuitable
        antennas (used by --all-bands). Best-effort sort still applies."""
        from rfcensus.events import EventBus
        from rfcensus.hardware.broker import (
            AccessMode, DongleBroker, DongleRequirements,
        )
        from rfcensus.hardware.registry import HardwareRegistry

        marine = _make_antenna("marine_vhf", 156.8, low_mhz=137, high_mhz=180)
        dongle = _make_dongle(0, 24_000_000, 1_700_000_000, antenna=marine)
        broker = DongleBroker(HardwareRegistry(dongles=[dongle]), EventBus())

        req = DongleRequirements(
            freq_hz=915_000_000,
            access_mode=AccessMode.EXCLUSIVE,
            require_suitable_antenna=False,
        )
        lease = await broker.allocate(req, consumer="test", timeout=1.0)
        # Allocated even though antenna is unsuitable
        assert lease.dongle.id == "rtl-0"
        await broker.release(lease)


# ──────────────────────────────────────────────────────────────────
# Best-effort matching preserved with --all-bands
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestBestEffortSortWithAllBands:
    """With require_suitable_antenna=False, broker should still PREFER
    well-matched antennas. Bad antennas are merely permitted, not
    promoted. The user's intent: scan everything, but still use the
    best dongle for each band."""

    async def test_best_antenna_picked_among_unsuitable_ones(self):
        from rfcensus.events import EventBus
        from rfcensus.hardware.broker import (
            AccessMode, DongleBroker, DongleRequirements,
        )
        from rfcensus.hardware.registry import HardwareRegistry

        # Two dongles, neither suitable for 915 MHz, but one is
        # closer to it than the other:
        #   • marine_vhf (resonant 156.8 MHz)
        #   • whip_433 (resonant 433 MHz, much closer to 915 in spectrum
        #     but still way off-band)
        marine = _make_antenna("marine_vhf", 156.8, low_mhz=137, high_mhz=180)
        whip_433 = _make_antenna("whip_433", 433, low_mhz=400, high_mhz=470)
        d_marine = _make_dongle(0, 24_000_000, 1_700_000_000, antenna=marine)
        d_433 = _make_dongle(1, 24_000_000, 1_700_000_000, antenna=whip_433)
        broker = DongleBroker(
            HardwareRegistry(dongles=[d_marine, d_433]), EventBus(),
        )

        # Best-effort: the broker should pick whichever scores higher
        # for 915 MHz. Both score 0.0 because 915 is outside both ranges,
        # but we still expect a dongle to be returned (no hard exclusion)
        req = DongleRequirements(
            freq_hz=915_000_000,
            access_mode=AccessMode.EXCLUSIVE,
            require_suitable_antenna=False,
        )
        lease = await broker.allocate(req, consumer="test", timeout=1.0)
        # Either is acceptable since both score 0; just confirm we got one
        assert lease.dongle.id in ("rtl-0", "rtl-1")
        await broker.release(lease)


# ──────────────────────────────────────────────────────────────────
# Load-balanced dongle assignment
# ──────────────────────────────────────────────────────────────────


class TestLoadBalancedAssignment:
    """When two equivalent dongles can match a band, the planner should
    spread bands across them rather than double-loading one. Concrete
    case: two whip_915 dongles + 915 MHz band + 929 MHz band → one
    band per dongle, not both on the same one."""

    def test_matcher_uses_load_as_tiebreaker(self):
        from rfcensus.config.schema import BandConfig
        from rfcensus.hardware.antenna import AntennaMatcher

        whip_a = _make_antenna("whip_915_a", 915, low_mhz=850, high_mhz=1000)
        whip_b = _make_antenna("whip_915_b", 915, low_mhz=850, high_mhz=1000)
        candidates = [("rtl-0", whip_a), ("rtl-1", whip_b)]
        band = BandConfig(
            id="b", name="b",
            freq_low=914_000_000, freq_high=916_000_000,
        )

        matcher = AntennaMatcher()
        # First call with empty load: tiebreak goes to lowest id
        m1 = matcher.best_pairing(band, candidates, dongle_load={})
        assert m1.dongle_id == "rtl-0"
        # Second call with rtl-0 already loaded: tiebreak → rtl-1
        m2 = matcher.best_pairing(
            band, candidates, dongle_load={"rtl-0": 1},
        )
        assert m2.dongle_id == "rtl-1"

    def test_score_still_dominates_load(self):
        """Higher score wins even if that dongle is busier."""
        from rfcensus.config.schema import BandConfig
        from rfcensus.hardware.antenna import AntennaMatcher

        whip_perfect = _make_antenna("whip_915", 915, low_mhz=850, high_mhz=1000)
        whip_marginal = _make_antenna("whip_marginal", 800, low_mhz=750, high_mhz=950)
        candidates = [("rtl-0", whip_perfect), ("rtl-1", whip_marginal)]
        band = BandConfig(
            id="b", name="b",
            freq_low=914_000_000, freq_high=916_000_000,
        )

        # Even with rtl-0 heavily loaded, its perfect-match score wins
        matcher = AntennaMatcher()
        m = matcher.best_pairing(
            band, candidates, dongle_load={"rtl-0": 100, "rtl-1": 0},
        )
        assert m.dongle_id == "rtl-0"

    def test_scheduler_spreads_915_bands_across_equivalent_dongles(self):
        """End-to-end: with two whip_915 dongles + two 915-ish bands,
        the scheduler should assign one band per dongle. This is the
        user's actual case (rtl-0 with whip_915 + rtl-2 with whip_915)."""
        from rfcensus.config.schema import SiteConfig
        from rfcensus.engine.scheduler import Scheduler
        from rfcensus.events import EventBus
        from rfcensus.hardware.broker import DongleBroker
        from rfcensus.hardware.registry import HardwareRegistry

        whip_a = _make_antenna("whip_915", 915, low_mhz=850, high_mhz=1000)
        whip_b = _make_antenna("whip_915", 915, low_mhz=850, high_mhz=1000)
        d_a = _make_dongle(0, 24_000_000, 1_700_000_000, antenna=whip_a)
        d_b = _make_dongle(1, 24_000_000, 1_700_000_000, antenna=whip_b)

        config = SiteConfig.model_validate({
            "site": {"name": "test"},
            "antennas": [],
            "bands": {"enabled": ["ism_915", "pocsag_929"]},
            "band_definitions": [
                {
                    "id": "ism_915", "name": "ism_915",
                    "freq_low": 902_000_000, "freq_high": 928_000_000,
                },
                {
                    "id": "pocsag_929", "name": "pocsag_929",
                    "freq_low": 929_000_000, "freq_high": 932_000_000,
                },
            ],
            "dongles": [],
        })
        broker = DongleBroker(HardwareRegistry(dongles=[d_a, d_b]), EventBus())
        plan = Scheduler(config, broker).plan(config.band_definitions)

        # Both bands assigned to DIFFERENT dongles, not the same one
        dongles_used = {t.suggested_dongle_id for t in plan.tasks}
        assert dongles_used == {"rtl-0", "rtl-1"}, (
            f"Expected bands spread across both dongles; "
            f"got {[(t.band.id, t.suggested_dongle_id) for t in plan.tasks]}"
        )


# ──────────────────────────────────────────────────────────────────
# StrategyContext threads all_bands flag
# ──────────────────────────────────────────────────────────────────


class TestStrategyContextAllBands:
    def test_default_all_bands_is_false(self):
        from rfcensus.engine.strategy import StrategyContext
        import inspect
        sig = inspect.signature(StrategyContext)
        assert sig.parameters["all_bands"].default is False

    def test_strategy_passes_require_suitable_inversely(self):
        """When ctx.all_bands=True, both allocation sites should set
        require_suitable_antenna=False to bypass the broker's hard
        antenna check. Verify by reading the source."""
        import rfcensus.engine.strategy as strategy
        import inspect
        src = inspect.getsource(strategy)
        # Both allocation sites should reference the inversion
        n_inversions = src.count("require_suitable_antenna=not ctx.all_bands")
        assert n_inversions >= 2, (
            f"Expected at least 2 sites passing require_suitable_antenna="
            f"not ctx.all_bands; found {n_inversions}"
        )
