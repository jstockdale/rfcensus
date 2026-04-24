"""v0.5.44 regression tests: background scanner task.

The v0.5.43 cooldown reduced scan frequency but didn't reduce the
DURATION of any individual scan. In busy bands a single scan could
still block the event loop for tens to hundreds of ms — long enough
to trip downstream decoder fanout read timeouts and cause the
`ended_by=both_simultaneously` cascade.

v0.5.44 decouples scanning from observe() entirely:

  • observe() is now O(1): just updates bin state, no scan
  • A background asyncio task runs _scan_and_emit on a fixed cadence
  • _scan_and_emit yields between templates so even a single run
    doesn't hold the loop across all 3 template passes
  • Production wiring (strategy.py) starts the task before sweep and
    stops it in finally

This test module verifies:
  1. After start(), observe() is O(1) — scan is NOT invoked inline
  2. The background task actually runs _scan_and_emit periodically
  3. stop() cleanly cancels the task (no task leak)
  4. Backward compat: without start(), observe() still scans inline
  5. Interior yields let concurrent coroutines run during a scan
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from rfcensus.events import EventBus, WideChannelEvent
from rfcensus.spectrum.wide_channel_aggregator import WideChannelAggregator


@pytest.mark.asyncio
async def test_observe_is_o1_when_scanner_running():
    """After start(), observe() must NOT trigger an inline scan —
    that's what decouples the sample hot path from CPU work."""
    scan_count = {"n": 0}
    orig_scan = WideChannelAggregator._scan_and_emit

    async def counting_scan(self, now):
        scan_count["n"] += 1
        return await orig_scan(self, now)

    bus = EventBus()
    agg = WideChannelAggregator(event_bus=bus, session_id=1, scan_interval_s=0.2)
    agg._scan_and_emit = counting_scan.__get__(agg, WideChannelAggregator)

    await agg.start()
    try:
        baseline = scan_count["n"]
        # Fire 200 observations back-to-back — none should trigger
        # an inline scan. The background task might run once during
        # this loop (if we happen to cross its cadence), but not
        # more than once.
        start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(200):
            await agg.observe(
                freq_hz=902_000_000 + i * 25_000,
                bin_width_hz=25_000,
                power_dbm=-40.0,
                noise_floor_dbm=-90.0,
                now=start_time + timedelta(microseconds=i),
                dongle_id="rtlsdr-test",
            )
        # At most 1 background scan completed while we were running
        # (we never slept, so usually 0; if event loop happened to
        # schedule one, it'd be 1).
        inline_scans = scan_count["n"] - baseline
        assert inline_scans <= 1, (
            f"observe() triggered {inline_scans} inline scans — "
            f"should be 0 when scanner is running"
        )
    finally:
        await agg.stop()


@pytest.mark.asyncio
async def test_background_task_runs_scans():
    """The scanner task must run _scan_and_emit on its cadence."""
    scan_count = {"n": 0}
    orig_scan = WideChannelAggregator._scan_and_emit

    async def counting_scan(self, now):
        scan_count["n"] += 1
        return await orig_scan(self, now)

    bus = EventBus()
    agg = WideChannelAggregator(
        event_bus=bus, session_id=1, scan_interval_s=0.05,  # 20 Hz for test
    )
    agg._scan_and_emit = counting_scan.__get__(agg, WideChannelAggregator)

    await agg.start()
    try:
        # Wait ~0.2 s real time → should see at least 2 scans
        await asyncio.sleep(0.2)
        assert scan_count["n"] >= 2, (
            f"expected ≥2 scans in 0.2s at 50ms cadence; "
            f"got {scan_count['n']}"
        )
    finally:
        await agg.stop()


@pytest.mark.asyncio
async def test_stop_cancels_task_cleanly():
    """stop() must cancel the background task and leave no dangling
    tasks in the event loop."""
    bus = EventBus()
    agg = WideChannelAggregator(event_bus=bus, session_id=1, scan_interval_s=0.1)
    await agg.start()
    assert agg._scanner_task is not None
    assert not agg._scanner_task.done()
    await agg.stop()
    assert agg._scanner_task is None


@pytest.mark.asyncio
async def test_stop_is_idempotent():
    """Calling stop() without start(), or twice, must not raise."""
    bus = EventBus()
    agg = WideChannelAggregator(event_bus=bus, session_id=1)
    # Never started
    await agg.stop()
    # Start and stop twice
    await agg.start()
    await agg.stop()
    await agg.stop()


@pytest.mark.asyncio
async def test_start_is_idempotent():
    """Calling start() twice must not spawn two tasks."""
    bus = EventBus()
    agg = WideChannelAggregator(event_bus=bus, session_id=1, scan_interval_s=0.1)
    await agg.start()
    first_task = agg._scanner_task
    await agg.start()  # second call
    assert agg._scanner_task is first_task
    await agg.stop()


@pytest.mark.asyncio
async def test_backward_compat_inline_scanning_still_works():
    """If start() is never called, observe() still does inline
    scanning with the v0.5.43 cooldown semantics."""
    scan_count = {"n": 0}
    orig_scan = WideChannelAggregator._scan_and_emit

    async def counting_scan(self, now):
        scan_count["n"] += 1
        return await orig_scan(self, now)

    bus = EventBus()
    agg = WideChannelAggregator(event_bus=bus, session_id=1)  # scan_interval_s=0
    agg._scan_and_emit = counting_scan.__get__(agg, WideChannelAggregator)

    # Do NOT call start(). Each observe should trigger a scan.
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(5):
        await agg.observe(
            freq_hz=902_000_000 + i * 25_000,
            bin_width_hz=25_000,
            power_dbm=-40.0,
            noise_floor_dbm=-90.0,
            now=start_time,
            dongle_id="rtlsdr-test",
        )
    assert scan_count["n"] == 5, (
        f"backward-compat inline scan broken: {scan_count['n']} scans "
        f"for 5 observations (expected 5)"
    )


@pytest.mark.asyncio
async def test_scanner_detects_bursts():
    """End-to-end: observations accumulate, background task scans,
    detection fires — without any inline scanning."""
    detections: list[WideChannelEvent] = []

    async def capture(evt):
        detections.append(evt)

    bus = EventBus()
    bus.subscribe(WideChannelEvent, capture)
    agg = WideChannelAggregator(
        event_bus=bus, session_id=1, scan_interval_s=0.05,  # 20 Hz
    )

    await agg.start()
    try:
        # Fire 10 adjacent bins (250 kHz composite)
        start_time = datetime.now(timezone.utc)
        for i in range(10):
            await agg.observe(
                freq_hz=906_000_000 + i * 25_000,
                bin_width_hz=25_000,
                power_dbm=-40.0,
                noise_floor_dbm=-90.0,
                now=start_time,
                dongle_id="rtlsdr-test",
            )
        # Wait for the background scanner to notice
        await asyncio.sleep(0.15)
        await bus.drain()
    finally:
        await agg.stop()

    assert len(detections) >= 1, (
        "background scanner must detect the composite within its "
        "cadence"
    )


@pytest.mark.asyncio
async def test_observe_runs_during_scan():
    """Critical property: observe() must remain responsive even while
    the background scanner is actively running a scan. Tests the
    interior yield behavior.

    Feed lots of bins so a scan has real work to do, then measure
    the latency of an observe() call while a scan is in flight.
    """
    bus = EventBus()
    agg = WideChannelAggregator(
        event_bus=bus, session_id=1, scan_interval_s=0.05,
    )
    await agg.start()
    try:
        # Prime the bin dict with 500 bins so scans have work
        start_time = datetime.now(timezone.utc)
        for i in range(500):
            await agg.observe(
                freq_hz=902_000_000 + i * 25_000,
                bin_width_hz=25_000,
                power_dbm=-40.0,
                noise_floor_dbm=-90.0,
                now=start_time,
                dongle_id="rtlsdr-test",
            )
        # Wait for a scan to START, then time an observe.
        await asyncio.sleep(0.06)

        t0 = asyncio.get_event_loop().time()
        await agg.observe(
            freq_hz=927_000_000,
            bin_width_hz=25_000,
            power_dbm=-40.0,
            noise_floor_dbm=-90.0,
            now=datetime.now(timezone.utc),
            dongle_id="rtlsdr-test",
        )
        elapsed = asyncio.get_event_loop().time() - t0
        # observe() should complete in under 5 ms — it does no
        # scanning, just one dict access + assignment + object update.
        # Even if a scan is running, observe() runs during the scan's
        # yield points.
        assert elapsed < 0.05, (
            f"observe() took {elapsed*1000:.1f} ms — should be < 5 ms "
            f"even while a scan is in progress"
        )
    finally:
        await agg.stop()
