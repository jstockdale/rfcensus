"""v0.6.17 — TUI usability improvements.

Items addressed in this release:

  1. Per-task status glyphs in the plan tree (was only a wave-level
     marker; couldn't see which task within a failed wave failed).
  2. Completed waves keep their task lines visible (was: collapsed
     to a single ✓/✗ line, hiding which tasks ran).
  3. Process stats (cpu/rss) moved from FooterBar to HeaderBar for
     readability.
  4. New per-dongle metrics: bands_visited count, fanout_bytes_sent,
     band_started_at (on-band time).
  5. Dongle detail no longer shows misleading "(total)" decode/
     detection rows that mixed productivity across different bands.
  6. Snapshot renamed to "report"; richer content including per-band
     activity table and per-task plan progress.
  7. Antenna ID added to dongle detail (config-static; pulled from
     DongleConfig at app mount).
  8. Log-mode toggle no longer kills the runner (was a critical bug
     in inventory.py — pending.cancel() ran unconditionally).

Tests verify each of these in isolation. Tests for tile content +
selection model are in test_v0616_selection_model.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest import mock
from unittest.mock import MagicMock

import pytest

from rfcensus.tui.state import (
    DongleState, TUIState, WaveState, seed_dongle_metadata,
)


# ─────────────────────────────────────────────────────────────────────
# 1. Per-task status tracking + reducer wiring
# ─────────────────────────────────────────────────────────────────────


class TestTaskStatusReducerWiring:
    def _make_state_with_plan(self):
        """Build a TUIState with a 2-wave plan: wave 0 has 2 tasks
        (band_a, band_b), wave 1 has 1 task (band_c)."""
        from rfcensus.events import PlanReadyEvent
        from rfcensus.tui.state import reduce
        s = TUIState()
        e = PlanReadyEvent(
            session_id=1,
            total_tasks=3,
            waves=[
                {"index": 0, "task_count": 2,
                 "task_summaries": ["band_a→rtlsdr-1", "band_b→rtlsdr-2"]},
                {"index": 1, "task_count": 1,
                 "task_summaries": ["band_c→rtlsdr-1"]},
            ],
            max_parallel_per_wave=2,
        )
        reduce(s, e)
        return s

    def test_plan_ready_initializes_task_statuses_to_pending(self):
        s = self._make_state_with_plan()
        assert s.waves[0].task_statuses == ["pending", "pending"]
        assert s.waves[1].task_statuses == ["pending"]

    def test_task_started_marks_status_running(self):
        from rfcensus.events import TaskStartedEvent
        from rfcensus.tui.state import reduce
        s = self._make_state_with_plan()
        reduce(s, TaskStartedEvent(
            wave_index=0, band_id="band_a", dongle_id="rtlsdr-1",
            consumer="rtl_433:band_a",
        ))
        assert s.waves[0].task_statuses[0] == "running"
        # Other tasks unchanged
        assert s.waves[0].task_statuses[1] == "pending"
        assert s.waves[1].task_statuses[0] == "pending"

    def test_task_completed_marks_terminal_status(self):
        from rfcensus.events import TaskStartedEvent, TaskCompletedEvent
        from rfcensus.tui.state import reduce
        s = self._make_state_with_plan()
        reduce(s, TaskStartedEvent(
            wave_index=0, band_id="band_a", dongle_id="rtlsdr-1",
            consumer="rtl_433:band_a",
        ))
        reduce(s, TaskCompletedEvent(
            wave_index=0, band_id="band_a", dongle_id="rtlsdr-1",
            consumer="rtl_433:band_a", status="ok",
        ))
        assert s.waves[0].task_statuses[0] == "ok"

    def test_task_completed_failed_status(self):
        from rfcensus.events import TaskCompletedEvent
        from rfcensus.tui.state import reduce
        s = self._make_state_with_plan()
        reduce(s, TaskCompletedEvent(
            wave_index=0, band_id="band_b", dongle_id="rtlsdr-2",
            consumer="rtl_power:band_b", status="failed",
            detail="some error",
        ))
        assert s.waves[0].task_statuses[1] == "failed"
        assert s.waves[0].task_statuses[0] == "pending"  # other unaffected

    def test_unknown_band_is_silently_ignored(self):
        # If a TaskCompletedEvent fires for a band that's not in the
        # wave's task_summaries (shouldn't happen, but defensively),
        # we no-op rather than crash.
        from rfcensus.events import TaskCompletedEvent
        from rfcensus.tui.state import reduce
        s = self._make_state_with_plan()
        reduce(s, TaskCompletedEvent(
            wave_index=0, band_id="nonexistent_band",
            dongle_id="rtlsdr-99", consumer="ghost", status="ok",
        ))
        # No statuses changed
        assert s.waves[0].task_statuses == ["pending", "pending"]


# ─────────────────────────────────────────────────────────────────────
# 2. Plan tree per-task glyphs + all-waves-expanded
# ─────────────────────────────────────────────────────────────────────


class TestPlanTreeGlyphs:
    def _make_widget(self):
        from rfcensus.tui.widgets.plan_tree import PlanTree
        return PlanTree()

    def test_task_glyph_for_each_status(self):
        from rfcensus.tui.widgets.plan_tree import _task_glyph
        # Not testing exact glyph chars (those can change); test that
        # each status maps to SOMETHING distinct from pending.
        for status in ("ok", "running", "failed", "crashed", "timeout",
                       "skipped"):
            char, style = _task_glyph(status)
            assert char != "", f"{status} should have a glyph"
        # Pending and unknown both map to no glyph
        assert _task_glyph("pending") == ("", "")
        assert _task_glyph("garbage") == ("", "")

    def test_failed_task_glyph_visible_in_render(self):
        w = self._make_widget()
        waves = [
            WaveState(
                index=0, task_count=2, status="completed",
                successful_count=1, error_count=1,
                task_summaries=["good_band→rtlsdr-1",
                                "bad_band→rtlsdr-2"],
                task_statuses=["ok", "failed"],
            ),
        ]
        w.update_plan(waves, current_index=0)
        rendered = str(w.render())
        # Both task names visible
        assert "good_band" in rendered
        assert "bad_band" in rendered
        # The failed task should have an X glyph next to it
        assert "✗" in rendered

    def test_partial_failure_uses_warning_marker(self):
        # Wave-level marker: ⚠ for partial failure (some ok, some
        # failed); ✗ only when ALL tasks failed.
        w = self._make_widget()
        waves = [
            WaveState(
                index=0, task_count=3, status="completed",
                successful_count=2, error_count=1,
                task_summaries=["a", "b", "c"],
                task_statuses=["ok", "failed", "ok"],
            ),
        ]
        w.update_plan(waves, current_index=0)
        rendered = str(w.render())
        assert "⚠" in rendered, "Partial failure should use ⚠"

    def test_total_failure_uses_x_marker(self):
        w = self._make_widget()
        waves = [
            WaveState(
                index=0, task_count=2, status="completed",
                successful_count=0, error_count=2,
                task_summaries=["a", "b"],
                task_statuses=["failed", "failed"],
            ),
        ]
        w.update_plan(waves, current_index=0)
        rendered = str(w.render())
        # Wave-level X (total failure marker)
        assert "✗" in rendered

    def test_clean_completion_uses_check(self):
        w = self._make_widget()
        waves = [
            WaveState(
                index=0, task_count=2, status="completed",
                successful_count=2, error_count=0,
                task_summaries=["a", "b"],
                task_statuses=["ok", "ok"],
            ),
        ]
        w.update_plan(waves, current_index=0)
        rendered = str(w.render())
        assert "✓" in rendered


# ─────────────────────────────────────────────────────────────────────
# 3. New per-dongle state fields
# ─────────────────────────────────────────────────────────────────────


class TestDongleStateNewFields:
    def test_seed_dongle_metadata_creates_with_antenna(self):
        s = TUIState()
        seed_dongle_metadata(s, "rtlsdr-1",
                             model="rtlsdr_v3", antenna_id="whip_915")
        assert len(s.dongles) == 1
        d = s.dongles[0]
        assert d.dongle_id == "rtlsdr-1"
        assert d.model == "rtlsdr_v3"
        assert d.antenna_id == "whip_915"

    def test_seed_dongle_metadata_idempotent(self):
        s = TUIState()
        seed_dongle_metadata(s, "rtlsdr-1", antenna_id="whip_915")
        seed_dongle_metadata(s, "rtlsdr-1", antenna_id="whip_915")
        assert len(s.dongles) == 1  # no duplicate

    def test_bands_visited_set_populated_on_allocate(self):
        from rfcensus.events import HardwareEvent
        from rfcensus.tui.state import reduce
        s = TUIState()
        reduce(s, HardwareEvent(
            kind="allocated", dongle_id="rtlsdr-1", consumer="c1",
            band_id="band_a", freq_hz=915_000_000, sample_rate=2_400_000,
        ))
        reduce(s, HardwareEvent(
            kind="released", dongle_id="rtlsdr-1",
        ))
        reduce(s, HardwareEvent(
            kind="allocated", dongle_id="rtlsdr-1", consumer="c2",
            band_id="band_b", freq_hz=433_000_000, sample_rate=2_400_000,
        ))
        d = s.dongles[0]
        assert d.bands_visited == {"band_a", "band_b"}

    def test_band_started_at_set_on_allocate_only_on_band_change(self):
        from rfcensus.events import HardwareEvent
        from rfcensus.tui.state import reduce
        s = TUIState()
        t1 = datetime.now(timezone.utc)
        t2 = t1 + timedelta(seconds=30)
        # First allocation — band_started_at should be set
        reduce(s, HardwareEvent(
            kind="allocated", dongle_id="rtlsdr-1", consumer="c1",
            band_id="band_a", freq_hz=915_000_000, sample_rate=2_400_000,
            timestamp=t1,
        ))
        d = s.dongles[0]
        assert d.band_started_at == t1
        # Same band re-allocated (e.g. consumer rotation) — band_started_at
        # should NOT change because the band is the same
        reduce(s, HardwareEvent(
            kind="allocated", dongle_id="rtlsdr-1", consumer="c2",
            band_id="band_a", freq_hz=915_000_000, sample_rate=2_400_000,
            timestamp=t2,
        ))
        assert d.band_started_at == t1, (
            "band_started_at should only update when the band changes"
        )

    def test_band_started_at_cleared_on_release(self):
        from rfcensus.events import HardwareEvent
        from rfcensus.tui.state import reduce
        s = TUIState()
        reduce(s, HardwareEvent(
            kind="allocated", dongle_id="rtlsdr-1", consumer="c1",
            band_id="band_a", freq_hz=915_000_000, sample_rate=2_400_000,
        ))
        reduce(s, HardwareEvent(
            kind="released", dongle_id="rtlsdr-1",
        ))
        d = s.dongles[0]
        assert d.band_started_at is None

    def test_fanout_bytes_accumulates_on_disconnect(self):
        from rfcensus.events import FanoutClientEvent
        from rfcensus.tui.state import reduce
        s = TUIState()
        # Two connect/disconnect cycles, each carrying a byte count
        reduce(s, FanoutClientEvent(
            slot_id="fanout[rtlsdr-1]", peer_addr="127.0.0.1:1",
            event_type="connect", bytes_sent=0,
        ))
        reduce(s, FanoutClientEvent(
            slot_id="fanout[rtlsdr-1]", peer_addr="127.0.0.1:1",
            event_type="disconnect", bytes_sent=1_000_000,
        ))
        reduce(s, FanoutClientEvent(
            slot_id="fanout[rtlsdr-1]", peer_addr="127.0.0.1:2",
            event_type="connect", bytes_sent=0,
        ))
        reduce(s, FanoutClientEvent(
            slot_id="fanout[rtlsdr-1]", peer_addr="127.0.0.1:2",
            event_type="disconnect", bytes_sent=2_000_000,
        ))
        d = s.dongles[0]
        assert d.fanout_bytes_sent == 3_000_000


# ─────────────────────────────────────────────────────────────────────
# 4. Dongle detail panel content
# ─────────────────────────────────────────────────────────────────────


class TestDongleDetailContent:
    def _render_detail(self, dongle):
        """Render the detail panel for one dongle and return text."""
        from rfcensus.tui.widgets.dongle_detail import DongleDetail
        d = DongleDetail()
        d.update_state([dongle], 0, [])
        return str(d.render())

    def test_antenna_shown(self):
        d = DongleState(
            dongle_id="rtlsdr-1", status="active",
            antenna_id="whip_915",
        )
        out = self._render_detail(d)
        assert "antenna" in out
        assert "whip_915" in out

    def test_antenna_dash_when_unset(self):
        d = DongleState(dongle_id="rtlsdr-1", status="idle")
        out = self._render_detail(d)
        assert "antenna" in out and "—" in out

    def test_no_total_decodes_or_detections_shown(self):
        # v0.6.17: explicitly removed the misleading "(total)" rows
        d = DongleState(
            dongle_id="rtlsdr-1", status="active",
            decodes_in_band=5, decodes_total=99,
            detections_in_band=3, detections_total=88,
        )
        out = self._render_detail(d)
        # In-band counters present
        assert "5" in out and "3" in out
        # Total values should NOT appear (they were misleading)
        assert "99" not in out
        assert "88" not in out
        # And the words "total" should not appear in the metrics
        # section either. We allow it elsewhere (e.g. session counters)
        # so check for the specific phrase used by v0.6.16.
        assert "decodes (total)" not in out
        assert "detections (total)" not in out

    def test_bytes_streamed_shown(self):
        d = DongleState(
            dongle_id="rtlsdr-1", status="active",
            fanout_bytes_sent=1_500_000,
        )
        out = self._render_detail(d)
        assert "bytes streamed" in out
        # Compact format: 1.4 MB
        assert "MB" in out

    def test_bands_visited_count_shown(self):
        d = DongleState(
            dongle_id="rtlsdr-1", status="active",
            bands_visited={"band_a", "band_b", "band_c"},
        )
        out = self._render_detail(d)
        assert "bands visited" in out
        assert "3" in out

    def test_on_band_time_shown_when_lease_active(self):
        d = DongleState(
            dongle_id="rtlsdr-1", status="active",
            consumer="c1", band_id="band_a",
            freq_hz=915_000_000, sample_rate=2_400_000,
            band_started_at=datetime.now(timezone.utc) - timedelta(seconds=125),
        )
        out = self._render_detail(d)
        assert "on band" in out

    def test_byte_format_helper(self):
        from rfcensus.tui.widgets.dongle_detail import _format_bytes
        assert _format_bytes(0) == "0 B"
        assert _format_bytes(500) == "500 B"
        assert _format_bytes(1500) == "1.5 KB"
        assert "MB" in _format_bytes(50_000_000)
        assert "GB" in _format_bytes(2_500_000_000)


# ─────────────────────────────────────────────────────────────────────
# 5. Snapshot/report content
# ─────────────────────────────────────────────────────────────────────


class TestSnapshotReportContent:
    def _build_app_with_state(self):
        """Build a TUIApp shell with a populated state for rendering."""
        from rfcensus.tui.app import TUIApp
        app = TUIApp.__new__(TUIApp)
        app.state = TUIState(site_name="testsite")
        app.state.session_id = 42
        app.state.session_started_at = (
            datetime.now(timezone.utc) - timedelta(seconds=600)
        )
        app.state.total_tasks = 5
        app.state.completed_tasks = 3
        app.state.total_decodes = 17
        app.state.total_emitters_confirmed = 4
        app.state.total_detections = 22
        # 2 dongles
        app.state.dongles.append(DongleState(
            dongle_id="rtlsdr-1", status="active",
            band_id="band_a", antenna_id="whip_915",
            decodes_in_band=10, detections_in_band=5,
            bands_visited={"band_a", "band_b"},
        ))
        app.state.dongles.append(DongleState(
            dongle_id="rtlsdr-2", status="idle",
            antenna_id="whip_433",
        ))
        # 2 waves
        app.state.waves = [
            WaveState(
                index=0, task_count=2, status="completed",
                successful_count=1, error_count=1,
                task_summaries=["band_a→rtlsdr-1", "band_b→rtlsdr-2"],
                task_statuses=["ok", "failed"],
            ),
            WaveState(
                index=1, task_count=1, status="running",
                task_summaries=["band_c→rtlsdr-1"],
                task_statuses=["running"],
            ),
        ]
        return app

    def test_snapshot_includes_session_id_and_site(self):
        app = self._build_app_with_state()
        out = app._render_snapshot_report()
        assert "#42" in out
        assert "testsite" in out

    def test_snapshot_includes_per_task_glyphs(self):
        app = self._build_app_with_state()
        out = app._render_snapshot_report()
        # Wave 0 had 1 ok, 1 failed — both should render with a glyph
        assert "band_a" in out and "band_b" in out
        assert "✓" in out  # ok task
        assert "✗" in out  # failed task

    def test_snapshot_includes_dongle_antenna(self):
        app = self._build_app_with_state()
        out = app._render_snapshot_report()
        assert "ant=whip_915" in out

    def test_snapshot_includes_bands_visited_when_more_than_one(self):
        app = self._build_app_with_state()
        out = app._render_snapshot_report()
        assert "bands_visited=2" in out

    def test_snapshot_includes_per_band_activity_when_emitters_present(self):
        # Push a few decoded events into the verbose buffer
        app = self._build_app_with_state()
        from rfcensus.tui.state import StreamEntry
        verbose = app.state.streams["verbose"]
        for _ in range(3):
            verbose.append(StreamEntry(
                timestamp=datetime.now(timezone.utc),
                severity="info", category="decode",
                text="decoded acurite at 433.920 MHz id=12345",
                raw=None,
            ))
        out = app._render_snapshot_report()
        assert "Activity by band" in out
        assert "acurite" in out


# ─────────────────────────────────────────────────────────────────────
# 6. Footer hint reflects "report" naming
# ─────────────────────────────────────────────────────────────────────


class TestFooterHintNaming:
    def test_footer_hint_uses_report_not_snapshot(self):
        from rfcensus.tui.widgets.footer import FooterBar
        # v0.6.17: renamed `s snapshot` → `s report`.
        # v0.7.3: rebound s → r so the visible label matches the
        # keystroke (s kept as a hidden alias). The footer hint is
        # the visible text and now reflects `r report`.
        assert "r report" in FooterBar.HINT
        assert "s snapshot" not in FooterBar.HINT
        # The bare " s report" (with the space prefix that
        # disambiguates from "decoders" etc.) must not appear —
        # would mean we forgot to update the hint.
        assert " s report" not in FooterBar.HINT


# ─────────────────────────────────────────────────────────────────────
# 7. Log-mode toggle bug fix (item 8)
# ─────────────────────────────────────────────────────────────────────


class TestLogModeBugFix:
    """v0.6.17 critical bug fix: pressing `l` (log mode toggle) used
    to silently kill the runner instead of switching to log view.

    Root cause: after asyncio.wait returned with tui_task in done,
    the inventory.py `for t in pending: t.cancel()` ran
    unconditionally — cancelling the still-running runner_task. Then
    `await runner_task` collected a CancelledError-shaped result and
    the session ended.

    Fix: detect the log-mode toggle via tui_app._log_mode_requested
    (set by action_toggle_log_mode before calling self.exit()) and
    skip the cancellation for runner_task in that case.

    These tests verify the flag is set correctly by the toggle action
    AND that the inventory.py cancellation logic checks the flag.
    """

    def test_action_toggle_log_mode_sets_flag(self):
        from rfcensus.tui.app import TUIApp
        app = TUIApp.__new__(TUIApp)
        app._log_mode_requested = False
        # Replace exit() with a noop so the test doesn't need a real
        # Textual app context.
        app.exit = lambda: None
        app.action_toggle_log_mode()
        assert app._log_mode_requested is True

    def test_inventory_cancellation_path_checks_log_mode_flag(self):
        # Static check: the inventory module's TUI loop must look at
        # _log_mode_requested before cancelling runner_task. This is
        # a guard against a future refactor reverting the fix.
        import inspect
        from rfcensus.commands import inventory
        src = inspect.getsource(inventory)
        assert "_log_mode_requested" in src, (
            "inventory.py must check tui_app._log_mode_requested "
            "before cancelling the runner task — see v0.6.17 bug fix"
        )
        # And the loop must explicitly skip the runner task in that
        # case. Cheap proxy: check that "log_mode_toggle" appears as
        # a guard variable name (introduced in the fix).
        assert "log_mode_toggle" in src, (
            "inventory.py must use the log_mode_toggle guard to skip "
            "runner_task cancellation"
        )
