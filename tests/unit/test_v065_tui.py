"""v0.6.5 TUI tests — state reducer, filter, color, size handling.

The TUI is built around a Redux-style reducer (`reduce(state, event)`)
that's pure aside from in-place state mutation. This makes it
testable without standing up Textual: feed the reducer events, assert
on the resulting state.

Snapshot tests for widget rendering are deliberately avoided — visual
regressions are obvious in real use, and snapshot tests bind us to
cosmetic decisions that legitimately change. We test:

  • Reducer correctness across every event type
  • Filter mode behavior (categories shown, error/warning passthrough)
  • Color module fallback to plain text when disabled
  • Sizing helper returns the right verdict
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rfcensus.events import (
    ActiveChannelEvent,
    DecodeEvent,
    DetectionEvent,
    EmitterEvent,
    FanoutClientEvent,
    HardwareEvent,
    PlanReadyEvent,
    SessionEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
    WaveCompletedEvent,
    WaveStartedEvent,
)
from rfcensus.tui.state import (
    StreamEntry,
    TUIState,
    cycle_filter_mode,
    filter_stream,
    reduce,
)


# ────────────────────────────────────────────────────────────────────
# 1. Reducer — per-event coverage
# ────────────────────────────────────────────────────────────────────


class TestReducerSession:
    def test_session_started(self):
        s = TUIState()
        e = SessionEvent(session_id=42, kind="started")
        reduce(s, e)
        assert s.session_id == 42
        assert s.session_status == "running"
        assert s.session_started_at is not None
        # Pushed to stream
        assert len(s.stream) == 1
        assert "started" in s.stream[0].text

    def test_session_ended(self):
        s = TUIState()
        reduce(s, SessionEvent(session_id=42, kind="started"))
        reduce(s, SessionEvent(session_id=42, kind="ended"))
        assert s.session_status == "ended"

    def test_phase_changed_does_not_clutter_stream(self):
        """phase_changed events are noisy — don't push them."""
        s = TUIState()
        reduce(s, SessionEvent(session_id=42, kind="phase_changed",
                               phase="pass_1_wave_0"))
        assert len(s.stream) == 0


class TestReducerPlan:
    def test_plan_ready(self):
        s = TUIState()
        e = PlanReadyEvent(
            session_id=42,
            waves=[
                {"index": 0, "task_count": 3, "task_summaries":
                 ["a→d1", "b→d2", "c→d3"]},
                {"index": 1, "task_count": 1, "task_summaries": ["d→d1"]},
            ],
            total_tasks=4,
            max_parallel_per_wave=3,
        )
        reduce(s, e)
        assert s.plan_ready
        assert s.total_tasks == 4
        assert len(s.waves) == 2
        assert s.waves[0].task_count == 3
        assert s.waves[0].status == "pending"

    def test_wave_started_marks_running(self):
        s = TUIState()
        reduce(s, PlanReadyEvent(waves=[
            {"index": 0, "task_count": 2, "task_summaries": []},
            {"index": 1, "task_count": 1, "task_summaries": []},
        ], total_tasks=3))
        reduce(s, WaveStartedEvent(wave_index=0, task_count=2, pass_n=1))
        assert s.current_wave_index == 0
        assert s.current_pass_n == 1
        assert s.waves[0].status == "running"
        assert s.waves[1].status == "pending"

    def test_wave_completed_marks_completed(self):
        s = TUIState()
        reduce(s, PlanReadyEvent(waves=[
            {"index": 0, "task_count": 2, "task_summaries": []},
        ], total_tasks=2))
        reduce(s, WaveStartedEvent(wave_index=0, task_count=2, pass_n=1))
        reduce(s, WaveCompletedEvent(
            wave_index=0, pass_n=1, task_count=2, successful_count=2,
            errors=[],
        ))
        assert s.waves[0].status == "completed"
        assert s.waves[0].successful_count == 2
        assert s.waves[0].error_count == 0


class TestReducerTask:
    def test_task_started_recorded(self):
        s = TUIState()
        reduce(s, TaskStartedEvent(
            wave_index=0, pass_n=1,
            band_id="915_ism", dongle_id="rtlsdr-1",
            consumer="strategy:915_ism",
        ))
        assert (0, "915_ism") in s.active_tasks
        assert s.active_tasks[(0, "915_ism")].dongle_id == "rtlsdr-1"

    def test_task_completed_clears_active(self):
        s = TUIState()
        reduce(s, TaskStartedEvent(
            wave_index=0, band_id="915_ism", dongle_id="rtlsdr-1",
        ))
        reduce(s, TaskCompletedEvent(
            wave_index=0, band_id="915_ism", dongle_id="rtlsdr-1",
            status="ok",
        ))
        assert (0, "915_ism") not in s.active_tasks
        assert s.completed_tasks == 1

    @pytest.mark.parametrize(
        "status,expected_severity",
        [
            ("ok", "good"),
            ("skipped", "info"),
            ("failed", "warning"),
            ("crashed", "error"),
            ("timeout", "warning"),
        ],
    )
    def test_task_completed_severity(self, status, expected_severity):
        s = TUIState()
        reduce(s, TaskCompletedEvent(
            wave_index=0, band_id="b", dongle_id="d", status=status,
        ))
        assert s.stream[0].severity == expected_severity


class TestReducerHardware:
    def test_allocated_marks_active(self):
        s = TUIState()
        reduce(s, HardwareEvent(
            dongle_id="d1", kind="allocated",
            freq_hz=915_000_000, sample_rate=2_400_000,
            consumer="rtl_433:915_ism", band_id="915_ism",
        ))
        assert len(s.dongles) == 1
        d = s.dongles[0]
        assert d.status == "active"
        assert d.freq_hz == 915_000_000
        assert d.consumer == "rtl_433:915_ism"
        assert d.band_id == "915_ism"

    def test_released_clears_lease_state(self):
        s = TUIState()
        reduce(s, HardwareEvent(
            dongle_id="d1", kind="allocated",
            freq_hz=915_000_000, consumer="x", band_id="b",
        ))
        reduce(s, HardwareEvent(dongle_id="d1", kind="released"))
        d = s.dongles[0]
        assert d.status == "idle"
        assert d.consumer is None
        assert d.freq_hz is None
        assert d.band_id is None

    def test_failed_pushes_error_to_stream(self):
        s = TUIState()
        reduce(s, HardwareEvent(
            dongle_id="d1", kind="failed", detail="USB disconnect",
        ))
        assert s.dongles[0].status == "failed"
        assert s.stream[0].severity == "error"

    def test_permanent_failed_persists(self):
        s = TUIState()
        reduce(s, HardwareEvent(
            dongle_id="d1", kind="permanently_failed",
            detail="exceeded max retries",
        ))
        assert s.dongles[0].status == "permanent_failed"

    def test_dongle_order_preserved_on_failure(self):
        """Failed tile stays in its slot, doesn't shuffle to the end."""
        s = TUIState()
        reduce(s, HardwareEvent(dongle_id="d1", kind="detected"))
        reduce(s, HardwareEvent(dongle_id="d2", kind="detected"))
        reduce(s, HardwareEvent(dongle_id="d3", kind="detected"))
        reduce(s, HardwareEvent(dongle_id="d2", kind="failed"))
        # d2 is still at index 1
        assert [d.dongle_id for d in s.dongles] == ["d1", "d2", "d3"]
        assert s.dongles[1].status == "failed"

    def test_band_change_resets_in_band_decode_count(self):
        s = TUIState()
        reduce(s, HardwareEvent(
            dongle_id="d1", kind="allocated",
            freq_hz=915_000_000, consumer="x", band_id="915_ism",
        ))
        s.dongles[0].decodes_in_band = 30
        reduce(s, HardwareEvent(
            dongle_id="d1", kind="allocated",
            freq_hz=433_000_000, consumer="x", band_id="433_ism",
        ))
        assert s.dongles[0].decodes_in_band == 0


class TestReducerFanout:
    def test_connect_increments_clients(self):
        s = TUIState()
        reduce(s, FanoutClientEvent(
            slot_id="fanout[d1]", peer_addr="127.0.0.1:5000",
            event_type="connect",
        ))
        d = s.dongles[0]
        assert d.dongle_id == "d1"
        assert d.fanout_clients == 1

    def test_disconnect_decrements_clients(self):
        s = TUIState()
        reduce(s, FanoutClientEvent(
            slot_id="fanout[d1]", peer_addr="127.0.0.1:5000",
            event_type="connect",
        ))
        reduce(s, FanoutClientEvent(
            slot_id="fanout[d1]", peer_addr="127.0.0.1:5000",
            event_type="disconnect",
        ))
        assert s.dongles[0].fanout_clients == 0

    def test_disconnect_below_zero_clamped(self):
        """Defensive — never go negative even on stray disconnect."""
        s = TUIState()
        reduce(s, FanoutClientEvent(
            slot_id="fanout[d1]", event_type="disconnect",
        ))
        assert s.dongles[0].fanout_clients == 0


class TestReducerCounters:
    def test_decode_increments_total(self):
        s = TUIState()
        reduce(s, DecodeEvent(freq_hz=915_000_000))
        assert s.total_decodes == 1

    def test_emitter_confirmed_increments(self):
        s = TUIState()
        reduce(s, EmitterEvent(
            kind="confirmed", emitter_id=42,
            typical_freq_hz=433_000_000, confidence=0.9,
        ))
        assert s.total_emitters_confirmed == 1
        assert s.stream[0].severity == "highlight"

    def test_detection_increments(self):
        s = TUIState()
        reduce(s, DetectionEvent(
            technology="lora", freq_hz=915_000_000, confidence=0.8,
            evidence="chirp",
        ))
        assert s.total_detections == 1
        assert s.stream[0].severity == "highlight"


class TestReducerActiveChannel:
    def test_new_channel_pushes_to_stream(self):
        s = TUIState()
        reduce(s, ActiveChannelEvent(
            kind="new", freq_center_hz=915_000_000,
            bandwidth_hz=125_000, snr_db=10.0,
            classification="pulsed",
        ))
        assert len(s.stream) == 1

    def test_updated_channel_does_not_push(self):
        s = TUIState()
        reduce(s, ActiveChannelEvent(
            kind="updated", freq_center_hz=915_000_000,
        ))
        assert len(s.stream) == 0


# ────────────────────────────────────────────────────────────────────
# 2. Stream cap
# ────────────────────────────────────────────────────────────────────


class TestStreamCap:
    def test_stream_capped_at_capacity(self):
        s = TUIState(stream_capacity=10)
        for i in range(50):
            reduce(s, EmitterEvent(
                kind="confirmed", emitter_id=i,
                typical_freq_hz=915_000_000, confidence=0.5,
            ))
        assert len(s.stream) == 10
        # Newest first
        assert "49" in s.stream[0].text


# ────────────────────────────────────────────────────────────────────
# 3. Filter modes
# ────────────────────────────────────────────────────────────────────


class TestFilterModes:
    def _make_entries(self):
        from rfcensus.events import Event
        ts = datetime.now(timezone.utc)
        return [
            StreamEntry(
                timestamp=ts, severity="info", category="session",
                text="session msg",
            ),
            StreamEntry(
                timestamp=ts, severity="info", category="task",
                text="task msg",
            ),
            StreamEntry(
                timestamp=ts, severity="info", category="decode",
                text="decode msg",
            ),
            StreamEntry(
                timestamp=ts, severity="highlight", category="emitter",
                text="emitter msg",
            ),
            StreamEntry(
                timestamp=ts, severity="warning", category="task",
                text="warning task",
            ),
            StreamEntry(
                timestamp=ts, severity="error", category="hardware",
                text="error hw",
            ),
        ]

    def test_minimal_shows_only_session_hardware(self):
        entries = self._make_entries()
        out = filter_stream(entries, "minimal")
        cats = {e.category for e in out}
        # Errors and warnings always pass; plus session/hardware
        assert "session" in cats
        assert "hardware" in cats  # error severity always passes anyway
        # Tasks/decodes/emitters not in minimal — but warning task passes
        # via severity rule
        assert any(e.text == "warning task" for e in out)
        assert not any(e.text == "task msg" for e in out)
        assert not any(e.text == "decode msg" for e in out)
        assert not any(e.text == "emitter msg" for e in out)

    def test_filtered_includes_emitters_not_decodes(self):
        entries = self._make_entries()
        out = filter_stream(entries, "filtered")
        texts = {e.text for e in out}
        assert "emitter msg" in texts
        assert "decode msg" not in texts

    def test_verbose_includes_everything(self):
        entries = self._make_entries()
        out = filter_stream(entries, "verbose")
        texts = {e.text for e in out}
        assert "decode msg" in texts
        assert "task msg" in texts
        assert "emitter msg" in texts

    def test_warnings_always_pass(self):
        entries = self._make_entries()
        # Warning task should appear even in minimal
        out = filter_stream(entries, "minimal")
        assert any(e.text == "warning task" for e in out)

    def test_errors_always_pass(self):
        entries = self._make_entries()
        out = filter_stream(entries, "minimal")
        assert any(e.text == "error hw" for e in out)

    def test_cycle_filter_progression(self):
        assert cycle_filter_mode("filtered") == "verbose"
        assert cycle_filter_mode("verbose") == "minimal"
        assert cycle_filter_mode("minimal") == "filtered"
        assert cycle_filter_mode("garbage") == "filtered"  # safe fallback


# ────────────────────────────────────────────────────────────────────
# 4. Color module
# ────────────────────────────────────────────────────────────────────


class TestColor:
    def test_styled_with_color_wraps(self):
        from rfcensus.tui import color
        color.configure_color(True)
        out = color.styled("active", "hello")
        assert "[" in out and "]" in out and "hello" in out

    def test_styled_without_color_returns_plain(self):
        from rfcensus.tui import color
        color.configure_color(False)
        try:
            out = color.styled("active", "hello")
            assert out == "hello"
        finally:
            color.configure_color(True)  # reset for other tests

    def test_unknown_token_returns_empty_style(self):
        from rfcensus.tui import color
        color.configure_color(True)
        # Unknown style → styled returns plain text (style() returns "")
        assert color.styled("nonexistent", "x") == "x"

    def test_dongle_status_glyph(self):
        from rfcensus.tui import color
        assert color.dongle_status_glyph("active") == "●"
        assert color.dongle_status_glyph("idle") == "○"
        assert color.dongle_status_glyph("failed") == "✗"
        assert color.dongle_status_glyph("permanent_failed") == "✗"


# ────────────────────────────────────────────────────────────────────
# 5. Sizing
# ────────────────────────────────────────────────────────────────────


class TestSizing:
    def test_check_tty_returns_msg_when_no_tty(self, monkeypatch):
        """If stdout isn't a TTY, refuse."""
        import sys
        from rfcensus.tui.app import check_tty_and_size

        class FakeStdout:
            def isatty(self):
                return False

        monkeypatch.setattr(sys, "stdout", FakeStdout())
        ok, msg = check_tty_and_size()
        assert not ok
        assert "TTY" in msg

    def test_min_constants_sane(self):
        from rfcensus.tui.app import MIN_COLS, MIN_ROWS
        assert MIN_COLS == 80
        assert MIN_ROWS == 24
