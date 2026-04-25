"""v0.6.5 — plan events, SessionControl, fanout pause/resume.

Three test groups for the backend orchestration shipping in v0.6.5:

1. **Plan events** — the 5 new event types fire at the right moments
   in the wave loop and carry the right payloads. Verifies the TUI
   plan-tree widget will get a complete picture of the execution
   plan without polling the scheduler.

2. **SessionControl** — pause/resume primitives behave correctly.
   wait_not_paused returns immediately when not paused, blocks while
   paused, returns when stopped. Pause duration accounting is
   correct across multiple pause/resume cycles.

3. **Fanout pause API** — pause_writes / resume_writes work in
   isolation (without a real upstream rtl_tcp).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from rfcensus.events import (
    EventBus,
    PlanReadyEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
    WaveCompletedEvent,
    WaveStartedEvent,
)


# ────────────────────────────────────────────────────────────────────
# 1. Plan event shapes
# ────────────────────────────────────────────────────────────────────


class TestPlanEventShapes:
    """Verify the 5 new event types accept the fields the wave loop
    publishes. Pure dataclass construction tests — they catch
    accidental field-name drift between publisher and subscriber."""

    def test_plan_ready_event_defaults(self):
        e = PlanReadyEvent()
        assert e.waves == []
        assert e.total_tasks == 0
        assert e.max_parallel_per_wave == 0

    def test_plan_ready_event_full(self):
        e = PlanReadyEvent(
            session_id=42,
            waves=[
                {"index": 0, "task_count": 4, "task_summaries": [
                    "915_ism→rtlsdr-1", "433_ism→rtlsdr-2",
                    "ais→rtlsdr-3", "315_security→rtlsdr-4",
                ]},
                {"index": 1, "task_count": 2, "task_summaries": [
                    "70cm_amateur→rtlsdr-5", "p25_800→rtlsdr-1",
                ]},
            ],
            total_tasks=6,
            max_parallel_per_wave=4,
        )
        assert len(e.waves) == 2
        assert e.waves[0]["index"] == 0
        assert e.waves[0]["task_count"] == 4
        assert e.total_tasks == 6
        assert e.max_parallel_per_wave == 4

    def test_wave_started_event(self):
        e = WaveStartedEvent(
            session_id=42, wave_index=0, task_count=4, pass_n=1,
        )
        assert e.wave_index == 0
        assert e.task_count == 4
        assert e.pass_n == 1

    def test_wave_completed_event(self):
        e = WaveCompletedEvent(
            session_id=42, wave_index=0, pass_n=1,
            task_count=4, successful_count=3,
            errors=["task X failed: timeout"],
        )
        assert e.wave_index == 0
        assert e.successful_count == 3
        assert len(e.errors) == 1

    def test_task_started_event(self):
        e = TaskStartedEvent(
            session_id=42, wave_index=0, pass_n=1,
            band_id="915_ism", dongle_id="rtlsdr-00000003",
            consumer="strategy:915_ism",
        )
        assert e.band_id == "915_ism"
        assert e.dongle_id == "rtlsdr-00000003"
        assert e.consumer == "strategy:915_ism"

    @pytest.mark.parametrize(
        "status",
        ["ok", "failed", "crashed", "skipped", "timeout"],
    )
    def test_task_completed_event_all_statuses(self, status):
        e = TaskCompletedEvent(
            wave_index=0, band_id="b", dongle_id="d",
            status=status, detail="some detail",
        )
        assert e.status == status


# ────────────────────────────────────────────────────────────────────
# 2. SessionControl
# ────────────────────────────────────────────────────────────────────


class TestSessionControl:
    def test_starts_unpaused(self):
        from rfcensus.engine.session_control import SessionControl
        c = SessionControl()
        assert not c.is_paused()
        assert c.current_pause_duration_s() == 0.0
        assert c.effective_total_paused_s() == 0.0

    @pytest.mark.asyncio
    async def test_pause_then_is_paused(self):
        from rfcensus.engine.session_control import SessionControl
        c = SessionControl()
        await c.pause()
        assert c.is_paused()

    @pytest.mark.asyncio
    async def test_pause_is_idempotent(self):
        """Two pauses in a row don't restart the pause timer."""
        from rfcensus.engine.session_control import SessionControl
        c = SessionControl()
        await c.pause()
        first_start = c.pause_started_at
        await asyncio.sleep(0.05)
        await c.pause()  # idempotent
        assert c.pause_started_at == first_start

    @pytest.mark.asyncio
    async def test_resume_clears_pause(self):
        from rfcensus.engine.session_control import SessionControl
        c = SessionControl()
        await c.pause()
        await c.resume()
        assert not c.is_paused()
        assert c.pause_started_at is None

    @pytest.mark.asyncio
    async def test_resume_accumulates_paused_time(self):
        from rfcensus.engine.session_control import SessionControl
        c = SessionControl()
        await c.pause()
        await asyncio.sleep(0.1)
        await c.resume()
        # Should have ~0.1s accumulated; allow generous slack
        assert 0.08 < c.total_paused_s < 0.5
        assert c.effective_total_paused_s() == c.total_paused_s

    @pytest.mark.asyncio
    async def test_multiple_pause_resume_cycles_accumulate(self):
        from rfcensus.engine.session_control import SessionControl
        c = SessionControl()
        await c.pause()
        await asyncio.sleep(0.05)
        await c.resume()
        first_pause = c.total_paused_s
        await c.pause()
        await asyncio.sleep(0.05)
        await c.resume()
        # Second pause adds to first
        assert c.total_paused_s > first_pause
        assert c.total_paused_s > 0.08

    @pytest.mark.asyncio
    async def test_effective_paused_includes_active_pause(self):
        """While paused, effective_total_paused_s grows with wall clock."""
        from rfcensus.engine.session_control import SessionControl
        c = SessionControl()
        await c.pause()
        e1 = c.effective_total_paused_s()
        await asyncio.sleep(0.1)
        e2 = c.effective_total_paused_s()
        # Active pause keeps growing the effective total
        assert e2 > e1
        # But total_paused_s isn't updated until resume
        assert c.total_paused_s == 0.0

    @pytest.mark.asyncio
    async def test_wait_not_paused_returns_immediately_when_not_paused(self):
        from rfcensus.engine.session_control import SessionControl
        c = SessionControl()
        # Should return immediately, no timeout needed
        await asyncio.wait_for(c.wait_not_paused(), timeout=0.5)

    @pytest.mark.asyncio
    async def test_wait_not_paused_blocks_then_resumes(self):
        from rfcensus.engine.session_control import SessionControl
        c = SessionControl()
        await c.pause()
        woke_at = []

        async def waiter():
            await c.wait_not_paused()
            woke_at.append(time.monotonic())

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        # Still blocked
        assert not woke_at
        # Resume — should wake the waiter
        resume_at = time.monotonic()
        await c.resume()
        await asyncio.wait_for(task, timeout=1.0)
        assert woke_at
        # Woke within ~50ms of resume
        assert woke_at[0] - resume_at < 0.5

    @pytest.mark.asyncio
    async def test_wait_not_paused_returns_when_stopped(self):
        """Even if still paused, stop() should wake any waiters so they
        can exit cleanly."""
        from rfcensus.engine.session_control import SessionControl
        c = SessionControl()
        await c.pause()

        async def waiter():
            await c.wait_not_paused()

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        await c.stop()
        # Should wake even though pause is still set
        await asyncio.wait_for(task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_finalize_caps_active_pause(self):
        """If a session ends while paused, finalize() rolls the in-progress
        pause into total_paused_s so reporting is accurate."""
        from rfcensus.engine.session_control import SessionControl
        c = SessionControl()
        await c.pause()
        await asyncio.sleep(0.05)
        # Don't resume; finalize directly
        c.finalize()
        assert c.total_paused_s > 0.04
        assert c.pause_started_at is None


# ────────────────────────────────────────────────────────────────────
# 3. deep_pause_watcher
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestDeepPauseWatcher:
    async def test_does_not_fire_when_unpaused(self):
        """Watcher should sit quietly when the session isn't paused."""
        from rfcensus.engine.session_control import (
            SessionControl,
            deep_pause_watcher,
        )
        c = SessionControl()
        fired = []

        async def cb():
            fired.append(time.monotonic())

        task = asyncio.create_task(deep_pause_watcher(
            c, cb, threshold_s=0.1, poll_interval_s=0.01,
        ))
        await asyncio.sleep(0.3)
        await c.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fired == []

    async def test_fires_after_threshold(self):
        from rfcensus.engine.session_control import (
            SessionControl,
            deep_pause_watcher,
        )
        c = SessionControl()
        fired = []

        async def cb():
            fired.append(time.monotonic())

        task = asyncio.create_task(deep_pause_watcher(
            c, cb, threshold_s=0.1, poll_interval_s=0.02,
        ))
        await c.pause()
        await asyncio.sleep(0.25)  # well past threshold
        assert len(fired) == 1
        assert c.deep_pause_active
        await c.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_fires_only_once_per_pause(self):
        """Once deep-pause has fired, watcher shouldn't fire again
        until a new pause begins."""
        from rfcensus.engine.session_control import (
            SessionControl,
            deep_pause_watcher,
        )
        c = SessionControl()
        fired = []

        async def cb():
            fired.append(time.monotonic())

        task = asyncio.create_task(deep_pause_watcher(
            c, cb, threshold_s=0.05, poll_interval_s=0.01,
        ))
        await c.pause()
        await asyncio.sleep(0.2)
        # Should have fired once
        assert len(fired) == 1
        await c.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ────────────────────────────────────────────────────────────────────
# 4. Fanout pause API
# ────────────────────────────────────────────────────────────────────


class TestFanoutPauseAPI:
    """The pause/resume API on RtlTcpFanout. Tested in isolation —
    no real rtl_tcp upstream needed. We construct the fanout with
    no event bus and exercise just the pause-state transitions."""

    def test_starts_unpaused(self):
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout
        f = RtlTcpFanout(
            upstream_host="127.0.0.1", upstream_port=0,
            slot_label="test",
        )
        assert not f.writes_paused
        assert f.paused_drops == 0

    def test_pause_writes(self):
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout
        f = RtlTcpFanout(
            upstream_host="127.0.0.1", upstream_port=0,
            slot_label="test",
        )
        f.pause_writes()
        assert f.writes_paused

    def test_pause_writes_idempotent(self):
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout
        f = RtlTcpFanout(
            upstream_host="127.0.0.1", upstream_port=0,
            slot_label="test",
        )
        f.pause_writes()
        f.pause_writes()
        assert f.writes_paused

    def test_resume_writes_clears_paused(self):
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout
        f = RtlTcpFanout(
            upstream_host="127.0.0.1", upstream_port=0,
            slot_label="test",
        )
        f.pause_writes()
        result = f.resume_writes()
        # No clients to probe → trivially "all alive"
        assert result is True
        assert not f.writes_paused
        # Drop counter resets on resume
        assert f.paused_drops == 0

    def test_resume_writes_when_not_paused_is_noop(self):
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout
        f = RtlTcpFanout(
            upstream_host="127.0.0.1", upstream_port=0,
            slot_label="test",
        )
        # Resume from unpaused → True (trivially), no state change
        assert f.resume_writes() is True
        assert not f.writes_paused


# ────────────────────────────────────────────────────────────────────
# 5. Deep-pause teardown (disconnect_all_clients)
# ────────────────────────────────────────────────────────────────────


class TestFanoutDeepPauseTeardown:
    """v0.6.5 deep-pause primitive — disconnect downstream clients
    while keeping the fanout itself alive."""

    def test_disconnect_with_no_clients_returns_zero(self):
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout
        f = RtlTcpFanout(
            upstream_host="127.0.0.1", upstream_port=0,
            slot_label="test",
        )
        assert f.disconnect_all_clients() == 0

    def test_disconnect_marks_clients_disconnected(self):
        """Construct fake clients, verify all get marked."""
        import asyncio
        from unittest.mock import MagicMock
        from rfcensus.hardware.rtl_tcp_fanout import (
            RtlTcpFanout,
            _DownstreamClient,
        )
        f = RtlTcpFanout(
            upstream_host="127.0.0.1", upstream_port=0,
            slot_label="test",
        )
        # Inject fake clients
        for i in range(3):
            mock_writer = MagicMock()
            client = _DownstreamClient(
                writer=mock_writer, label=f"127.0.0.1:{5000 + i}",
            )
            client.queue = asyncio.Queue(maxsize=10)
            f._clients.append(client)
        n = f.disconnect_all_clients()
        assert n == 3
        for c in f._clients:
            assert c.disconnected
            c.writer.close.assert_called_once()

    def test_already_disconnected_skipped(self):
        """Calling twice doesn't double-count."""
        import asyncio
        from unittest.mock import MagicMock
        from rfcensus.hardware.rtl_tcp_fanout import (
            RtlTcpFanout,
            _DownstreamClient,
        )
        f = RtlTcpFanout(
            upstream_host="127.0.0.1", upstream_port=0,
            slot_label="test",
        )
        mock_writer = MagicMock()
        client = _DownstreamClient(writer=mock_writer, label="x")
        client.queue = asyncio.Queue(maxsize=10)
        f._clients.append(client)
        first = f.disconnect_all_clients()
        second = f.disconnect_all_clients()
        assert first == 1
        assert second == 0
