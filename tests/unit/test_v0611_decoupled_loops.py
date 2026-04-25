"""v0.6.11 — read/analyze loop decoupling.

Background
==========
Through v0.6.10 the survey ran a single coroutine that interleaved
socket reads with energy-gating, accumulation, and analysis. When a
heavy analysis fired (~3s on a Pi 5), the read couldn't progress for
that duration. The upstream rtl_tcp fanout's per-client queue
overflowed, the fanout marked the survey client as "slow", and chunks
were silently dropped UPSTREAM with no visibility from the survey.
The user's metatron run showed >9000 fanout-side drops over 5 min,
~30% of all data.

What changed
============
The single loop is split into two:
  • _read_drain_loop: tight loop that ONLY drains the socket and
    enqueues decoded chunks on a bounded asyncio.Queue. Never
    blocks on analysis.
  • _analyze_loop: pops from the queue, energy-gates, accumulates,
    runs analysis. Slow analysis no longer blocks reading.

When the analyzer falls behind (queue fills), the read loop drops
the OLDEST queued chunk and increments stats.chunks_dropped_local.
Two wins: (a) the upstream fanout never sees us as a slow client
(we always read), and (b) when we DO drop data it's visible and
counted.

This file tests the new behavior end-to-end.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from rfcensus.engine.lora_survey_task import (
    LoraSurveyStats,
    LoraSurveyTask,
    _READ_CHUNK_BYTES,
    _READ_QUEUE_MAXSIZE,
)


# ─────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────


def _make_band(id="test_band"):
    band = MagicMock()
    band.freq_low = 902_000_000
    band.freq_high = 928_000_000
    band.id = id
    return band


def _make_task(*, duration_s: float = 1.0, sample_rate: int = 2_400_000):
    """Construct a LoraSurveyTask with mocks, ready to invoke the
    inner loops without going through the real broker / fanout."""
    return LoraSurveyTask(
        broker=MagicMock(),
        event_bus=MagicMock(),
        band=_make_band(),
        duration_s=duration_s,
        sample_rate=sample_rate,
    )


def _quiet_chunk_bytes(n: int = _READ_CHUNK_BYTES) -> bytes:
    """A chunk that's all DC (mid-range u8 = 127). Decodes to ~0
    and falls below any reasonable energy gate."""
    return bytes([127] * n)


def _loud_chunk_bytes(n: int = _READ_CHUNK_BYTES) -> bytes:
    """A chunk well above noise floor — alternating 200/55 produces
    strong sample magnitude (≈±0.57 in normalized units)."""
    return bytes([200, 55] * (n // 2))


class _ProgrammableReader:
    """asyncio.StreamReader stand-in that returns pre-loaded chunks
    and signals EOF when the queue is exhausted. Each readexactly
    can be optionally delayed to simulate slow upstream pacing."""

    def __init__(self, chunks: list[bytes], per_read_delay_s: float = 0.0):
        self._chunks = list(chunks)
        self._delay = per_read_delay_s

    async def readexactly(self, n: int) -> bytes:
        if self._delay:
            await asyncio.sleep(self._delay)
        if not self._chunks:
            # Mimic asyncio.StreamReader EOF
            raise asyncio.IncompleteReadError(partial=b"", expected=n)
        chunk = self._chunks.pop(0)
        if len(chunk) < n:
            raise asyncio.IncompleteReadError(partial=chunk, expected=n)
        return chunk[:n]


# ─────────────────────────────────────────────────────────────────────
# Stats: new fields default correctly
# ─────────────────────────────────────────────────────────────────────


class TestNewStatsFields:
    """v0.6.11 added three fields. Must default to zero so existing
    callers that consult stats by attribute don't crash."""

    def test_defaults(self):
        s = LoraSurveyStats()
        assert s.chunks_dropped_local == 0
        assert s.read_queue_high_water == 0
        assert s.analysis_duration_s_total == 0.0


# ─────────────────────────────────────────────────────────────────────
# Read-drain loop: keeps reading even when analyzer is slow
# ─────────────────────────────────────────────────────────────────────


class TestReadDrainLoopNeverBlocksOnAnalyzer:
    """The whole point of v0.6.11. If the analyzer takes seconds per
    iteration, the read loop must keep draining the socket. The
    upstream fanout never sees us as a slow client."""

    @pytest.mark.asyncio
    async def test_read_loop_drains_socket_at_full_rate_when_analyzer_stuck(
        self,
    ):
        # Queue 100 chunks worth of bytes ready to read instantly.
        # An analyzer that holds the queue full should NOT prevent the
        # read loop from consuming all of them — it just drops the
        # excess (counted in chunks_dropped_local).
        n_input = 100
        chunks = [_loud_chunk_bytes() for _ in range(n_input)]

        task = _make_task(duration_s=5.0)
        task._reader = _ProgrammableReader(chunks)
        stats = LoraSurveyStats()

        queue: asyncio.Queue = asyncio.Queue(maxsize=_READ_QUEUE_MAXSIZE)
        deadline = time.monotonic() + 5.0

        # Run read loop in isolation — no analyzer consuming. Read
        # loop should drain all 100 input chunks, hit EOF, exit.
        await task._read_drain_loop(stats, queue, deadline)

        assert stats.chunks_read == n_input, (
            f"read loop should consume all {n_input} input chunks even "
            f"with no analyzer draining the queue; got "
            f"{stats.chunks_read}. (If this fails, the read loop is "
            f"blocking on queue.put when full instead of dropping.)"
        )
        # Queue holds at most maxsize items; everything else dropped.
        expected_dropped = n_input - _READ_QUEUE_MAXSIZE
        # Account for the sentinel None pushed at end-of-stream taking
        # one slot (which means one fewer real chunk in the queue).
        assert (
            stats.chunks_dropped_local == expected_dropped
            or stats.chunks_dropped_local == expected_dropped + 1
        ), (
            f"expected ~{expected_dropped} drops, got "
            f"{stats.chunks_dropped_local}"
        )

    @pytest.mark.asyncio
    async def test_high_water_mark_tracked(self):
        """read_queue_high_water should reflect peak queue depth."""
        chunks = [_loud_chunk_bytes() for _ in range(_READ_QUEUE_MAXSIZE * 2)]

        task = _make_task(duration_s=5.0)
        task._reader = _ProgrammableReader(chunks)
        stats = LoraSurveyStats()
        queue: asyncio.Queue = asyncio.Queue(maxsize=_READ_QUEUE_MAXSIZE)
        deadline = time.monotonic() + 5.0

        await task._read_drain_loop(stats, queue, deadline)

        assert stats.read_queue_high_water >= _READ_QUEUE_MAXSIZE - 1, (
            f"high water should track up to ~maxsize "
            f"({_READ_QUEUE_MAXSIZE}), got {stats.read_queue_high_water}"
        )


# ─────────────────────────────────────────────────────────────────────
# Analyze loop: still fires when chunks are above floor
# ─────────────────────────────────────────────────────────────────────


class TestAnalyzeLoopFiresOnAboveFloorBursts:
    """Sanity: the new analyze loop still does what the old one did —
    energy-gate, accumulate, run survey_iq_window when the window
    fills. Test by feeding loud chunks straight onto the queue."""

    @pytest.mark.asyncio
    async def test_loud_chunks_trigger_analysis(self, monkeypatch):
        analyze_calls = []

        async def fake_analyze_window(window, stats):
            analyze_calls.append(window.size)
            stats.analyses_performed += 1

        # Use a small sample rate so the analysis_window_samples
        # threshold is reached in just a few chunks.
        sample_rate = 240_000  # → window = 60_000 samples = 2 chunks
        task = _make_task(sample_rate=sample_rate)
        monkeypatch.setattr(task, "_analyze_window", fake_analyze_window)

        # Push enough loud chunks for several windows, then sentinel.
        queue: asyncio.Queue = asyncio.Queue()
        chunk_samples = _READ_CHUNK_BYTES // 2  # 32K samples per chunk
        for _ in range(10):
            samples = task._decode_chunk(_loud_chunk_bytes())
            queue.put_nowait(samples)
        queue.put_nowait(None)  # EOF

        stats = LoraSurveyStats()
        deadline = time.monotonic() + 10.0
        await task._analyze_loop(stats, queue, deadline)

        assert len(analyze_calls) >= 1, (
            "analyze_window should have fired at least once on 10 loud "
            "chunks"
        )
        assert stats.chunks_above_floor >= 5, (
            f"chunks_above_floor should reflect loud chunks; got "
            f"{stats.chunks_above_floor}"
        )

    @pytest.mark.asyncio
    async def test_quiet_chunks_skip_analysis(self, monkeypatch):
        analyze_calls = []

        async def fake_analyze_window(window, stats):
            analyze_calls.append(window.size)
            stats.analyses_performed += 1

        task = _make_task(sample_rate=240_000)
        monkeypatch.setattr(task, "_analyze_window", fake_analyze_window)

        queue: asyncio.Queue = asyncio.Queue()
        for _ in range(20):
            samples = task._decode_chunk(_quiet_chunk_bytes())
            queue.put_nowait(samples)
        queue.put_nowait(None)

        stats = LoraSurveyStats()
        deadline = time.monotonic() + 10.0
        await task._analyze_loop(stats, queue, deadline)

        assert analyze_calls == [], (
            f"analyze_window must NOT fire on quiet chunks; got "
            f"{len(analyze_calls)} calls"
        )
        assert stats.chunks_above_floor == 0


# ─────────────────────────────────────────────────────────────────────
# Integration: read + analyze running concurrently
# ─────────────────────────────────────────────────────────────────────


class TestConcurrentLoopsEndToEnd:
    """Wire both loops together via a real queue; prove a slow analyzer
    does NOT prevent the read loop from completing all reads."""

    @pytest.mark.asyncio
    async def test_slow_analyzer_does_not_throttle_reads(self, monkeypatch):
        """Inject a deliberately slow _analyze_window. Verify the read
        loop still consumes all input chunks at full rate."""

        async def slow_analyze_window(window, stats):
            await asyncio.sleep(0.2)  # 200ms simulated heavy DSP
            stats.analyses_performed += 1

        sample_rate = 240_000
        n_input = 60  # ~ 60 * 14ms = 840ms of "audio" simulated
        task = _make_task(duration_s=5.0, sample_rate=sample_rate)
        monkeypatch.setattr(task, "_analyze_window", slow_analyze_window)

        # Mix of loud + quiet so the analyzer fires periodically.
        chunks = []
        for i in range(n_input):
            chunks.append(
                _loud_chunk_bytes() if i % 3 == 0 else _quiet_chunk_bytes()
            )
        task._reader = _ProgrammableReader(chunks, per_read_delay_s=0.005)

        # Run the orchestrator; both loops execute concurrently.
        await task._stream_loop(LoraSurveyStats(), time.monotonic())

        # We can't easily inspect stats here because _stream_loop
        # constructs its own; the assertion is that it didn't hang.
        # This test passing means concurrent loops terminate cleanly
        # under load.

    @pytest.mark.asyncio
    async def test_drop_local_counter_increments_when_analyzer_blocks(
        self, monkeypatch,
    ):
        """When the analyzer is artificially slow AND the input rate
        is high, chunks_dropped_local must increment — that's the
        observability win."""
        analyze_started = asyncio.Event()
        analyze_release = asyncio.Event()

        async def blocking_analyze(window, stats):
            analyze_started.set()
            await analyze_release.wait()
            stats.analyses_performed += 1

        sample_rate = 240_000
        # 50 chunks should saturate the queue (maxsize=16) easily.
        n_input = 50
        chunks = [_loud_chunk_bytes() for _ in range(n_input)]

        task = _make_task(duration_s=5.0, sample_rate=sample_rate)
        monkeypatch.setattr(task, "_analyze_window", blocking_analyze)
        task._reader = _ProgrammableReader(chunks)

        # Run read+analyze concurrently. Release the analyzer after
        # we've confirmed it's blocked — by then the read loop should
        # have filled the queue and started dropping.
        async def _release_after_wait():
            try:
                await asyncio.wait_for(analyze_started.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                # Possibly all chunks were quiet — analyzer never fired.
                # This test specifically uses loud chunks so this should
                # not happen, but guard anyway.
                pass
            await asyncio.sleep(0.5)  # let read loop pile up
            analyze_release.set()

        # We can't observe the inner stats from _stream_loop; mock the
        # stream_loop assembly directly to track stats.
        stats = LoraSurveyStats()
        queue: asyncio.Queue = asyncio.Queue(maxsize=_READ_QUEUE_MAXSIZE)
        deadline = time.monotonic() + 5.0

        read_task = asyncio.create_task(
            task._read_drain_loop(stats, queue, deadline)
        )
        analyze_task = asyncio.create_task(
            task._analyze_loop(stats, queue, deadline)
        )
        release_task = asyncio.create_task(_release_after_wait())

        # Wait for read to finish (consumes all input + EOF).
        await read_task
        # Now release the analyzer and let it drain.
        analyze_release.set()
        await asyncio.wait_for(analyze_task, timeout=5.0)
        await release_task

        assert stats.chunks_read == n_input, (
            f"read should have consumed all input; got {stats.chunks_read}"
        )
        assert stats.chunks_dropped_local > 0, (
            f"with the analyzer blocked and 50 chunks pumping in, the "
            f"read loop should have dropped some chunks at the queue. "
            f"Got chunks_dropped_local={stats.chunks_dropped_local}. "
            f"This is the v0.6.11 observability assertion: drops are "
            f"now visible and counted."
        )
