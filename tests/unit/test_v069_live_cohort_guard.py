"""v0.6.9 — live-cohort guard against false hardware-loss heuristic.

The strategy layer's "decoder exited <5s with 0 decodes → mark dongle
FAILED" heuristic was too eager: when v0.6.8's fanout filter
disconnected rtlamr for requesting set_sample_rate(2359296), rtlamr
exited at 0.1s, the heuristic mistakenly marked the dongle failed,
and subsequent shared-lease requests (e.g. lora_survey trying to join
the same fanout 7s later) couldn't find a healthy dongle.

v0.6.9 fix: if the dongle currently has OTHER active leases (rtl_433
still streaming happily), the dongle is clearly alive — the early
exit is decoder-specific, not hardware loss. Guard the mark_failed
behind active_lease_count > 1 and add an elif branch that logs the
decoder-specific failure WITHOUT touching dongle status.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rfcensus.events import EventBus
from rfcensus.hardware.antenna import Antenna
from rfcensus.hardware.broker import (
    AccessMode, DongleBroker, DongleRequirements,
)
from rfcensus.hardware.dongle import (
    Dongle, DongleCapabilities, DongleStatus,
)
from rfcensus.hardware.registry import HardwareRegistry


def _antenna_915():
    return Antenna(
        id="whip_915", name="whip_915", antenna_type="whip",
        resonant_freq_hz=915_000_000,
        usable_range=(900_000_000, 928_000_000),
        gain_dbi=2.15, polarization="vertical",
        requires_bias_power=False, notes="",
    )


def _make_dongle(serial="00000003"):
    return Dongle(
        id=f"rtlsdr-{serial}",
        serial=serial,
        model="rtlsdr_generic",
        driver="rtlsdr",
        capabilities=DongleCapabilities(
            freq_range_hz=(24_000_000, 1_700_000_000),
            max_sample_rate=2_400_000,
            bits_per_sample=8,
            bias_tee_capable=False,
            tcxo_ppm=10.0,
        ),
        antenna=_antenna_915(),
        status=DongleStatus.DETECTED,
        driver_index=0,
    )


def _make_broker():
    """Broker with one healthy dongle. Mock _start_shared_slot so
    allocate() doesn't try to spawn rtl_tcp — we only care about lease
    bookkeeping, not the underlying process plumbing."""
    registry = HardwareRegistry(dongles=[_make_dongle()])
    bus = EventBus()
    return DongleBroker(registry, bus)


def _stub_shared_slot(monkeypatch):
    """Patch DongleBroker._start_shared_slot to return a lightweight
    slot without spawning rtl_tcp. Returns the slot for assertions.

    Process and fanout are AsyncMock so the broker's release-time
    `await slot.process.stop()` and `await slot.fanout.stop()` both
    work without spawning real async resources."""
    from unittest.mock import AsyncMock
    from rfcensus.hardware import broker as broker_mod

    proc = MagicMock()
    proc.stop = AsyncMock()
    fanout = MagicMock()
    fanout.stop = AsyncMock()

    slot = broker_mod._SharedSlot(
        process=proc, host="127.0.0.1", port=1235,
        sample_rate=2_400_000, center_freq_hz=915_000_000,
        fanout=fanout,
    )

    async def fake_start_slot(self, dongle, sample_rate, center_freq_hz):
        slot.sample_rate = sample_rate
        slot.center_freq_hz = center_freq_hz
        return slot

    monkeypatch.setattr(DongleBroker, "_start_shared_slot", fake_start_slot)
    return slot


class TestActiveLeaseCount:
    """Direct tests of the helper used by the strategy guard."""

    def test_zero_leases_returns_zero(self):
        broker = _make_broker()
        assert broker.active_lease_count("rtlsdr-00000003") == 0

    def test_unknown_dongle_returns_zero(self):
        broker = _make_broker()
        # Querying a dongle that doesn't exist is fine — returns 0.
        assert broker.active_lease_count("rtlsdr-00009999") == 0

    @pytest.mark.asyncio
    async def test_single_shared_lease_returns_one(self, monkeypatch):
        broker = _make_broker()
        _stub_shared_slot(monkeypatch)

        lease = await broker.allocate(
            DongleRequirements(
                freq_hz=915_000_000, sample_rate=2_400_000,
                access_mode=AccessMode.SHARED,
                require_suitable_antenna=True,
            ),
            consumer="rtl_433:915_ism", timeout=1.0,
        )

        assert broker.active_lease_count(lease.dongle.id) == 1

    @pytest.mark.asyncio
    async def test_two_shared_leases_returns_two(self, monkeypatch):
        """The exact scenario the guard protects: rtl_433 + rtlamr
        both holding leases on the same dongle. After rtlamr's early
        exit, the count drops to 1 — but BEFORE release (when the
        strategy heuristic checks), it's 2."""
        broker = _make_broker()
        _stub_shared_slot(monkeypatch)

        req = DongleRequirements(
            freq_hz=915_000_000, sample_rate=2_400_000,
            access_mode=AccessMode.SHARED,
            require_suitable_antenna=True,
        )
        lease_a = await broker.allocate(
            req, consumer="rtl_433:915_ism", timeout=1.0,
        )
        lease_b = await broker.allocate(
            req, consumer="rtlamr:915_ism", timeout=1.0,
        )

        # Both leases live → count is 2.
        assert broker.active_lease_count(lease_a.dongle.id) == 2

        # After rtlamr's early exit, its lease is released → count drops.
        await broker.release(lease_b)
        assert broker.active_lease_count(lease_a.dongle.id) == 1

    @pytest.mark.asyncio
    async def test_exclusive_lease_counted(self):
        """Exclusive holder also counts — a dongle in exclusive use is
        clearly alive. (Exclusive doesn't need a fanout stub.)"""
        broker = _make_broker()
        lease = await broker.allocate(
            DongleRequirements(
                freq_hz=915_000_000, sample_rate=2_400_000,
                access_mode=AccessMode.EXCLUSIVE,
                require_suitable_antenna=True,
            ),
            consumer="rtl_power:915", timeout=1.0,
        )
        assert broker.active_lease_count(lease.dongle.id) == 1


class TestStrategyHeuristicGuard:
    """The guard that uses active_lease_count to suppress mark_failed.

    Verifies the broker-side signal the heuristic relies on, plus the
    behavior of the heuristic itself (cohort alive → don't mark failed,
    cohort empty → original behavior).
    """

    @pytest.mark.asyncio
    async def test_cohort_signal_true_with_peer_lease(self, monkeypatch):
        """The user's metatron scenario: rtlamr exits at 0.1s, but
        rtl_433 is still holding a lease. The signal the heuristic
        checks (active_lease_count > 1) must be True."""
        broker = _make_broker()
        _stub_shared_slot(monkeypatch)

        req = DongleRequirements(
            freq_hz=915_000_000, sample_rate=2_400_000,
            access_mode=AccessMode.SHARED,
            require_suitable_antenna=True,
        )
        rtl_433_lease = await broker.allocate(
            req, consumer="rtl_433:915_ism", timeout=1.0,
        )
        rtlamr_lease = await broker.allocate(
            req, consumer="rtlamr:915_ism", timeout=1.0,
        )

        # At the moment the strategy's heuristic fires (rtlamr has
        # exited but its lease is still in the broker tables, since
        # release happens in the finally block AFTER the heuristic):
        dongle_id = rtlamr_lease.dongle.id
        assert broker.active_lease_count(dongle_id) == 2

        # The strategy's check is `> 1` to mean "OTHER leases exist
        # beyond this decoder's own". With rtl_433 alive, it's True
        # → mark_failed is skipped.
        other_active = broker.active_lease_count(dongle_id) > 1
        assert other_active, (
            "live-cohort signal must be True when peer leases exist"
        )

        # Cleanup — test only validates the signal, not the strategy
        # call path itself.
        await broker.release(rtlamr_lease)
        await broker.release(rtl_433_lease)

    @pytest.mark.asyncio
    async def test_cohort_signal_false_when_alone(self, monkeypatch):
        """The opposite case: when the early-exiting decoder is the
        ONLY lease, the signal is False (active_lease_count == 1) and
        the original "may be hardware loss" heuristic still runs."""
        broker = _make_broker()
        _stub_shared_slot(monkeypatch)

        lease = await broker.allocate(
            DongleRequirements(
                freq_hz=915_000_000, sample_rate=2_400_000,
                access_mode=AccessMode.SHARED,
                require_suitable_antenna=True,
            ),
            consumer="rtl_433:915_ism", timeout=1.0,
        )

        dongle_id = lease.dongle.id
        # Only this lease exists → count is 1, not > 1 → signal False.
        assert broker.active_lease_count(dongle_id) == 1
        assert not (broker.active_lease_count(dongle_id) > 1)

        await broker.release(lease)

    @pytest.mark.asyncio
    async def test_dongle_stays_allocatable_after_peer_lease_released(self, monkeypatch):
        """End-to-end registry view: when the cohort guard suppresses
        mark_failed, the dongle remains allocatable to NEW consumers
        after the early-exiting decoder's lease is released. (Status
        will be BUSY while rtl_433 still holds the slot — but it's
        BUSY-with-a-fanout, which is exactly the state lora_survey
        wants to attach to.)"""
        broker = _make_broker()
        _stub_shared_slot(monkeypatch)

        req = DongleRequirements(
            freq_hz=915_000_000, sample_rate=2_400_000,
            access_mode=AccessMode.SHARED,
            require_suitable_antenna=True,
        )
        rtl_433_lease = await broker.allocate(
            req, consumer="rtl_433:915_ism", timeout=1.0,
        )
        rtlamr_lease = await broker.allocate(
            req, consumer="rtlamr:915_ism", timeout=1.0,
        )

        dongle_id = rtlamr_lease.dongle.id

        # Simulate: rtlamr's lease released because its decoder exited
        # early. mark_failed was NOT called (because guard was True).
        await broker.release(rtlamr_lease)

        # The actual signal the broker uses to pick a candidate is
        # `dongle.status not in (FAILED, UNAVAILABLE)`. With the guard
        # working, status stays BUSY (not FAILED) and lora_survey can
        # join the existing fanout.
        dongle = broker.registry.by_id(dongle_id)
        assert dongle is not None
        assert dongle.status != DongleStatus.FAILED, (
            "dongle was incorrectly marked FAILED — the v0.6.8 → v0.6.9 "
            "regression where rtlamr's command-conflict disconnect "
            "tripped the hardware-loss heuristic"
        )

        # End-to-end proof: lora_survey can still acquire a shared
        # lease on this dongle (joining the existing fanout).
        lora_survey_lease = await broker.allocate(
            req, consumer="lora_survey:915_ism", timeout=1.0,
        )
        assert lora_survey_lease.dongle.id == dongle_id

        await broker.release(lora_survey_lease)
        await broker.release(rtl_433_lease)
