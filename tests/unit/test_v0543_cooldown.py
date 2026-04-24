"""v0.5.43 regression: the scan cooldown prevents event-loop starvation
during rtl_power sweeps.

Field bug (v0.5.42 log 18:13:50-18:13:51): WideChannelAggregator.observe
ran a full O(n·k) template scan on every above-floor sample. One
rtl_power sweep over 26 MHz at 25 kHz bins produces 1040 samples that
arrive in a burst. With 3 templates (125/250/500 kHz), that's 3120
synchronous scans back-to-back. Each scan takes 0.1-1 ms; cumulatively
they block the asyncio event loop for hundreds of milliseconds.

During that block, decoder fanout writer tasks can't drain their
IQ output queues to the downstream rtl_433 clients. The clients'
kernel socket buffers fill up; rtl_433 sees its upstream stall,
times out, and exits. Result: three rtl_433 decoders on three
different dongles disconnect at the same wall-clock moment with
`ended_by=both_simultaneously` — which is the exact signature
observed in the field.

Fix: v0.5.43 adds scan_interval_s. Bin activity updates on every
observe() (cheap dict update), but the full scan runs at most every
scan_interval_s seconds. Production wiring uses 0.2 s → 5 scans per
sweep instead of 3120.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from rfcensus.events import EventBus
from rfcensus.spectrum.wide_channel_aggregator import WideChannelAggregator


@pytest.mark.asyncio
async def test_cooldown_limits_scan_frequency():
    """With a 200 ms cooldown, feeding 100 samples in rapid succession
    should trigger at most ~1 scan (all within the first 200 ms
    window)."""
    scans: list[datetime] = []
    orig_scan = WideChannelAggregator._scan_and_emit

    async def counting_scan(self, now):
        scans.append(now)
        return await orig_scan(self, now)

    bus = EventBus()
    agg = WideChannelAggregator(
        event_bus=bus, session_id=1, scan_interval_s=0.2,
    )
    # Monkey-patch for this one instance
    agg._scan_and_emit = counting_scan.__get__(agg, WideChannelAggregator)

    # Fire 100 samples in rapid succession at different freqs
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(100):
        await agg.observe(
            freq_hz=902_000_000 + i * 25_000,
            bin_width_hz=25_000,
            power_dbm=-40.0,
            noise_floor_dbm=-90.0,
            now=start + timedelta(microseconds=i),
            dongle_id="rtlsdr-test",
        )

    # With scan_interval_s=0.2 and no real time passing (all in same
    # event loop turn), at most 1 scan should have fired.
    assert len(scans) <= 2, (
        f"expected at most 2 scans with 200 ms cooldown; "
        f"got {len(scans)}"
    )


@pytest.mark.asyncio
async def test_cooldown_zero_preserves_v0_5_42_behavior():
    """With scan_interval_s=0 (the default), every observation
    triggers a scan — same as pre-v0.5.43 behavior."""
    scans: list[datetime] = []
    orig_scan = WideChannelAggregator._scan_and_emit

    async def counting_scan(self, now):
        scans.append(now)
        return await orig_scan(self, now)

    bus = EventBus()
    agg = WideChannelAggregator(event_bus=bus, session_id=1)  # default = 0
    agg._scan_and_emit = counting_scan.__get__(agg, WideChannelAggregator)

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(10):
        await agg.observe(
            freq_hz=902_000_000 + i * 25_000,
            bin_width_hz=25_000,
            power_dbm=-40.0,
            noise_floor_dbm=-90.0,
            now=start + timedelta(microseconds=i),
            dongle_id="rtlsdr-test",
        )
    # All 10 observations should have triggered a scan
    assert len(scans) == 10


@pytest.mark.asyncio
async def test_cooldown_still_catches_real_bursts():
    """A real burst produces ≥cooldown_s of observations. Even with
    throttling, at least one scan runs within that window and sees
    all the bins — so the detection still fires. Simulates the burst
    by adding real time sleeps between batches (as would happen with
    rtl_power sweeps ~1 Hz apart)."""
    from rfcensus.events import WideChannelEvent
    detections: list[WideChannelEvent] = []

    async def capture(evt):
        detections.append(evt)

    bus = EventBus()
    bus.subscribe(WideChannelEvent, capture)
    agg = WideChannelAggregator(
        event_bus=bus, session_id=1, scan_interval_s=0.1,
    )

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # First sweep: feed 10 adjacent bins
    for i in range(10):
        await agg.observe(
            freq_hz=906_000_000 + i * 25_000,
            bin_width_hz=25_000,
            power_dbm=-40.0,
            noise_floor_dbm=-90.0,
            now=start,
            dongle_id="rtlsdr-test",
        )
    # Wait past the cooldown, feed one more observation → triggers
    # scan, which now sees all 10 bins from this sweep
    await asyncio.sleep(0.15)
    await agg.observe(
        freq_hz=906_000_000,  # same bin as first — refresh its state
        bin_width_hz=25_000,
        power_dbm=-40.0,
        noise_floor_dbm=-90.0,
        now=start,
        dongle_id="rtlsdr-test",
    )
    await bus.drain()
    assert len(detections) >= 1, (
        "after cooldown expires, next observation must scan and "
        "emit the pending composite"
    )


@pytest.mark.asyncio
async def test_bin_state_updates_even_during_cooldown():
    """Cooldown skips scans but NOT bin activity updates. A burst that
    arrives during cooldown still gets recorded; the next scan (after
    cooldown expires) sees the full picture."""
    bus = EventBus()
    agg = WideChannelAggregator(
        event_bus=bus, session_id=1, scan_interval_s=1.0,  # long cooldown
    )

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # First observation — triggers scan, updates bin
    await agg.observe(
        freq_hz=906_000_000, bin_width_hz=25_000,
        power_dbm=-40.0, noise_floor_dbm=-90.0,
        now=start, dongle_id="rtlsdr-test",
    )
    assert 906_000_000 in agg._bins

    # Subsequent observations during cooldown still update bins
    for i in range(1, 10):
        await agg.observe(
            freq_hz=906_000_000 + i * 25_000,
            bin_width_hz=25_000,
            power_dbm=-40.0,
            noise_floor_dbm=-90.0,
            now=start + timedelta(microseconds=i),
            dongle_id="rtlsdr-test",
        )
    # All 10 bins should be recorded
    assert len(agg._bins) == 10
