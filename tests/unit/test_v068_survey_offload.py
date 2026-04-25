"""v0.6.8 — survey_iq_window must not block the asyncio event loop.

Background: in v0.6.7 and earlier, LoraSurveyTask called
survey_iq_window directly on the event loop. survey_iq_window does
heavy DSP — Welch PSD, multiple DDC operations, multiple chirp /
dechirp passes — taking several seconds of CPU on a representative
2.4 MHz × 1 second IQ window. Because the SDR fanouts run on the
SAME event loop, the loop stalled long enough that fanout I/O
couldn't be serviced and rtl_433's 3-second async-read watchdog
fired, killing every shared rtl_433 client with exit code 3 every
time the survey emitted a detection.

Fix: wrap the call in asyncio.to_thread so DSP runs on a worker
thread and the loop stays free for I/O. This test guards the
invariant: a heartbeat coroutine running concurrently with the
analysis call should still tick at its expected rate.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.mark.asyncio
class TestSurveyOffload:
    """The cooperative-scheduling invariant: while
    LoraSurveyTask._analyze_window is running, an unrelated coroutine
    on the same loop must continue making progress on time. If
    survey_iq_window were called directly (no to_thread), the
    heartbeat would be starved for the full duration of the DSP."""

    def _make_task(self):
        from rfcensus.engine.lora_survey_task import LoraSurveyTask
        from rfcensus.events import EventBus
        band = MagicMock()
        band.id = "915_ism"
        bus = EventBus()
        return LoraSurveyTask(
            broker=MagicMock(), event_bus=bus,
            band=band, duration_s=1.0,
        )

    async def test_analysis_does_not_starve_concurrent_coroutine(
        self, monkeypatch
    ):
        """Run a 'heartbeat' coroutine that sleeps 50ms in a loop
        and counts ticks. Concurrently invoke _analyze_window with
        a survey_iq_window stub that simulates 800ms of synchronous
        CPU work (well above any acceptable loop-blocking budget).
        After the analysis returns, the heartbeat should have ticked
        at least 8 times — meaning the DSP did NOT block the loop.
        """
        from rfcensus.engine import lora_survey_task as mod
        from rfcensus.engine.lora_survey_task import LoraSurveyStats

        # Stub: simulate heavy CPU work (time.sleep blocks the calling
        # thread, exactly like real DSP). If asyncio.to_thread is
        # NOT used, this would block the event loop for 800ms.
        analysis_called = []

        def fake_survey(samples, **kwargs):
            analysis_called.append(time.monotonic())
            time.sleep(0.8)  # synchronous "DSP" work
            return []  # no hits

        monkeypatch.setattr(mod, "survey_iq_window", fake_survey)

        task = self._make_task()
        stats = LoraSurveyStats()
        # Small but realistic window — its content doesn't matter
        # because we stubbed the analysis function.
        window = np.zeros(2_400_000, dtype=np.complex64)

        # Heartbeat coroutine: sleep 50ms in a loop, count ticks.
        ticks: list[float] = []
        stop = asyncio.Event()

        async def heartbeat():
            while not stop.is_set():
                ticks.append(time.monotonic())
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass

        hb_task = asyncio.create_task(heartbeat())

        # Run the analysis. With asyncio.to_thread it should yield
        # repeatedly to the loop while waiting on the worker thread,
        # letting the heartbeat tick.
        t0 = time.monotonic()
        await task._analyze_window(window, stats)
        t_elapsed = time.monotonic() - t0

        stop.set()
        await hb_task

        # Sanity: the analysis ran (the stub was called) and took
        # roughly its 800 ms of CPU.
        assert len(analysis_called) == 1
        assert t_elapsed >= 0.7, (
            f"analysis returned suspiciously fast ({t_elapsed:.2f}s) "
            f"— the stub should have taken ~0.8s"
        )

        # Critical invariant: the heartbeat ticked many times during
        # the 800 ms analysis. With to_thread it should hit ~16 ticks
        # (800 ms / 50 ms). If the loop was blocked it'd hit ~1 (the
        # initial tick before the block, then nothing until the end).
        # We use a conservative threshold: ≥8 ticks means at least
        # half the expected rate, well above the "blocked" failure mode.
        assert len(ticks) >= 8, (
            f"heartbeat only ticked {len(ticks)} times in {t_elapsed:.2f}s "
            f"— the event loop appears to have been blocked. "
            f"Expected ≥8 ticks (50 ms cadence)."
        )

    async def test_analysis_propagates_to_thread_exceptions(
        self, monkeypatch
    ):
        """If the offloaded survey_iq_window raises, the exception
        gets caught + logged inside _analyze_window (so a single bad
        window doesn't kill the survey task)."""
        from rfcensus.engine import lora_survey_task as mod
        from rfcensus.engine.lora_survey_task import LoraSurveyStats

        def boom(*args, **kwargs):
            raise RuntimeError("synthetic DSP failure")

        monkeypatch.setattr(mod, "survey_iq_window", boom)

        task = self._make_task()
        stats = LoraSurveyStats()
        window = np.zeros(1_000_000, dtype=np.complex64)

        # Should NOT raise — exception is logged + swallowed.
        await task._analyze_window(window, stats)

        # Stats still increment (analysis was attempted).
        assert stats.analyses_performed == 1
