"""v0.6.6 — fanout hot-path performance regressions.

Two specific perf bugs were found analyzing why a 3-client fanout
(rtl_433 + rtlamr + lora_survey on the same shared 915 ISM slot)
exhibited cascading queue stalls:

  1. `await self._publish_client_event(...)` inside the relay loop
     blocked distribution while subscribers ran. With 3 high-rate
     clients at 4.5 MB/s each, even a 100ms subscriber stall meant
     ~1.4 MB of backlog per client per tick.

  2. The 256-deep client queue (4 MB / ~800 ms at 4.8 MB/s) was
     barely enough headroom for one client and not enough for three
     when the survey's 64KB readexactly cycles overlapped with
     rtl_433's preamble-search CPU bursts.

These tests guard against re-introduction of either bug.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from rfcensus.hardware.rtl_tcp_fanout import (
    _CLIENT_QUEUE_DEPTH,
    _RELAY_CHUNK_SIZE,
    RtlTcpFanout,
)


# ────────────────────────────────────────────────────────────────────
# 1. Queue depth: enough for 3 high-rate clients
# ────────────────────────────────────────────────────────────────────


class TestQueueDepth:
    def test_provides_at_least_one_second_of_buffering(self):
        """At 2.4 Msps × 2 bytes = 4.8 MB/s, we need queue * chunk
        bytes >= 4_800_000 for >=1s buffering. v0.6.6 bumped 256→512
        to give ~1.7s, comfortable for multi-client streaming."""
        bytes_buffered = _CLIENT_QUEUE_DEPTH * _RELAY_CHUNK_SIZE
        seconds_at_full_rate = bytes_buffered / 4_800_000
        assert seconds_at_full_rate >= 1.0, (
            f"only {seconds_at_full_rate:.2f}s of buffering — too tight "
            f"for 3-client multi-decoder workloads"
        )

    def test_queue_depth_is_at_least_v066_value(self):
        """Document the v0.6.6 minimum so a future commit reducing it
        below that threshold raises an obvious test failure."""
        assert _CLIENT_QUEUE_DEPTH >= 512


# ────────────────────────────────────────────────────────────────────
# 2. Slow-client publish is fire-and-forget
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestSlowClientPublishNonBlocking:
    """The slow-client publish previously did `await
    self._publish_client_event(...)` inside the relay loop. Even
    though bus.publish itself doesn't await handlers, the await
    yields the event loop — a bad subscriber could stall the whole
    fanout. v0.6.6 changed it to fire-and-forget via
    asyncio.create_task with a `_bg_tasks` strong-ref set."""

    def _make_fanout(self):
        return RtlTcpFanout(
            upstream_host="127.0.0.1", upstream_port=0,
            slot_label="test",
        )

    async def test_spawn_event_does_not_await_subscriber(self):
        """When _spawn_client_event is called, control returns to the
        caller immediately even if a subscriber takes time to process."""
        from rfcensus.events import EventBus, FanoutClientEvent

        bus = EventBus()
        f = self._make_fanout()
        f._event_bus = bus

        subscriber_started = asyncio.Event()
        subscriber_release = asyncio.Event()

        async def slow_subscriber(event):
            subscriber_started.set()
            await subscriber_release.wait()

        bus.subscribe(FanoutClientEvent, slow_subscriber)

        # Call _spawn_client_event — should NOT block on subscriber
        f._spawn_client_event("127.0.0.1:5000", "slow", bytes_sent=1024)

        # Yield once so the spawned task starts
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # Subscriber should have started running (it's in flight)
        # but we got control back without waiting for it
        assert subscriber_started.is_set(), (
            "subscriber didn't start — task wasn't spawned"
        )

        # Crucially: the fanout is not waiting on subscriber_release
        # We can run more code, then release the subscriber later
        f._spawn_client_event("127.0.0.1:5001", "slow", bytes_sent=2048)
        await asyncio.sleep(0.01)

        # Cleanup: release the slow subscriber
        subscriber_release.set()
        # Drain pending tasks
        await bus.drain(timeout=1.0)
        if f._bg_tasks:
            await asyncio.gather(*f._bg_tasks, return_exceptions=True)

    async def test_spawn_event_holds_strong_ref(self):
        """asyncio.create_task tasks get GC'd if no strong ref is kept,
        triggering 'Task destroyed while pending' warnings. v0.6.6
        adds them to _bg_tasks until done."""
        from rfcensus.events import EventBus

        bus = EventBus()
        f = self._make_fanout()
        f._event_bus = bus

        f._spawn_client_event("127.0.0.1:5000", "slow", bytes_sent=1024)
        # Right after spawn, task should be in _bg_tasks
        assert len(f._bg_tasks) >= 1

        # After the task completes, it should self-remove
        await asyncio.sleep(0.05)
        # All tasks done by now (no slow subscribers)
        assert all(t.done() for t in list(f._bg_tasks))

    async def test_spawn_event_noop_without_bus(self):
        """No bus configured → no task created (no spurious warnings)."""
        f = self._make_fanout()
        assert f._event_bus is None
        f._spawn_client_event("127.0.0.1:5000", "slow", bytes_sent=1024)
        assert len(f._bg_tasks) == 0


# ────────────────────────────────────────────────────────────────────
# 3. Hot path doesn't await between clients
# ────────────────────────────────────────────────────────────────────


class TestHotPathStaticAnalysis:
    """Static check: the relay-loop distribution block must not
    contain an await inside the per-client loop. Awaits there yield
    the event loop and let one client's writer fall behind another's,
    cascading into queue overflows."""

    def test_relay_loop_per_client_block_has_no_await(self):
        """Read the source of _relay_upstream and verify the
        per-client distribution loop is fully synchronous.

        The check: between the `for client in self._clients:` line
        and the next `except asyncio.CancelledError:`, there must be
        zero `await ` tokens.
        """
        import inspect

        from rfcensus.hardware import rtl_tcp_fanout

        source = inspect.getsource(rtl_tcp_fanout.RtlTcpFanout._relay_upstream)

        # Find the distribution loop body
        in_for_loop = False
        violations: list[str] = []
        for line_no, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("for client in self._clients"):
                in_for_loop = True
                continue
            if in_for_loop:
                # Loop ends when we dedent below the for's indent.
                # The simplest robust check: stop scanning at the
                # finally block — that's well below the for loop.
                if stripped.startswith("except asyncio.CancelledError"):
                    break
                # Strip comments before checking — the docstring +
                # surrounding comment text often references "await"
                # explanatorily, which isn't a real await.
                code_only = stripped.split("#", 1)[0]
                if "await " in code_only:
                    violations.append(f"  line {line_no}: {stripped}")

        assert not violations, (
            "v0.6.6 invariant violated: await inside per-client "
            "distribution loop in _relay_upstream:\n"
            + "\n".join(violations)
        )
