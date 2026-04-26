"""v0.6.17 — TUI improvements per user feedback (8-item review).

Items addressed:
  1. Per-task status glyphs in plan tree (the "yellow X" was ambiguous;
     now ◆/✓/⚠/✗/· at wave level + ✓/✗/⏱/⌀/◆ per task)
  2. Plan tree shows task list for ALL waves (not just running +
     last-completed); user wants to see what completed across the run
  3. Proc stats moved from FooterBar to HeaderBar (was hard to see
     dimmed at end of long counter line)
  4. New per-dongle metrics: bytes streamed, bands visited count,
     time-on-current-band
  5. Replaced misleading decode/detection (band/total) with current-
     band-only counts; totals across different bands aren't comparable
  6. "snapshot" → "report" rename (was misleading); modal sized up;
     report includes per-band activity, recent emitters, plan progress
  7. Antenna ID shown on dongle detail (was missing entirely)
  8. Bug: log-mode toggle (l key) was killing the runner via
     unconditional pending.cancel() in inventory.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rfcensus.tui.state import (
    DongleState, TUIState, WaveState, _task_index_in_wave,
    seed_dongle_metadata,
)


# ─────────────────────────────────────────────────────────────────────
# Item 1+2: per-task status glyphs in plan tree
# ─────────────────────────────────────────────────────────────────────


class TestPlanTreePerTaskGlyphs:
    """Per-task status now visible next to each task line so user can
    see WHICH task within a failed wave failed (the v0.6.16 yellow X
    was ambiguous at the wave level)."""

    def _make_widget(self):
        from rfcensus.tui.widgets.plan_tree import PlanTree
        return PlanTree()

    def test_running_task_shows_diamond(self):
        w = self._make_widget()
        wave = WaveState(
            index=0, task_count=2,
            task_summaries=["band_a→rtlsdr-AAA0", "band_b→rtlsdr-BBB0"],
            task_statuses=["running", "pending"],
            status="running",
        )
        w.update_plan([wave], current_index=0)
        rendered = str(w.render())
        assert "◆" in rendered

    def test_failed_task_shows_x(self):
        w = self._make_widget()
        wave = WaveState(
            index=0, task_count=2,
            task_summaries=["band_a→rtlsdr-AAA0", "band_b→rtlsdr-BBB0"],
            task_statuses=["ok", "failed"],
            status="completed",
            successful_count=1,
            error_count=1,
        )
        w.update_plan([wave], current_index=0)
        rendered = str(w.render())
        # Both ✓ (for the ok task) and ✗ (for the failed task)
        assert "✓" in rendered
        assert "✗" in rendered

    def test_partial_failure_wave_marker_is_warning(self):
        # Wave with SOME failures gets ⚠, not ✗ (which is reserved
        # for total failure). This is the v0.6.17 disambiguation.
        w = self._make_widget()
        wave = WaveState(
            index=0, task_count=3,
            task_summaries=["a→x", "b→y", "c→z"],
            task_statuses=["ok", "failed", "ok"],
            status="completed",
            successful_count=2,
            error_count=1,
        )
        w.update_plan([wave], current_index=0)
        rendered = str(w.render())
        assert "⚠" in rendered

    def test_total_failure_wave_marker_is_x(self):
        w = self._make_widget()
        wave = WaveState(
            index=0, task_count=2,
            task_summaries=["a→x", "b→y"],
            task_statuses=["failed", "crashed"],
            status="completed",
            successful_count=0,
            error_count=2,
        )
        w.update_plan([wave], current_index=0)
        rendered = str(w.render())
        assert "✗" in rendered

    def test_pending_task_has_no_glyph(self):
        # Pending tasks should NOT show a glyph — the eye should be
        # drawn to running and completed ones, not the long list of
        # things that haven't started yet.
        w = self._make_widget()
        wave = WaveState(
            index=0, task_count=2,
            task_summaries=["pending_a→x", "pending_b→y"],
            task_statuses=["pending", "pending"],
            status="pending",
        )
        w.update_plan([wave], current_index=None)
        rendered = str(w.render())
        # No status glyphs in the rendered text. The wave-level marker
        # is the only one (· for pending wave).
        assert "✓" not in rendered
        assert "✗" not in rendered
        assert "◆" not in rendered  # only present if running

    def test_all_completed_waves_keep_task_lines(self):
        """User feedback (item 2): old behavior collapsed older
        completed waves to a single line. v0.6.17 keeps tasks visible
        for all waves."""
        w = self._make_widget()
        waves = [
            WaveState(
                index=0, task_count=1,
                task_summaries=["oldband→rtlsdr-OLD0"],
                task_statuses=["ok"],
                status="completed",
                successful_count=1, error_count=0,
            ),
            WaveState(
                index=1, task_count=1,
                task_summaries=["midband→rtlsdr-MID0"],
                task_statuses=["ok"],
                status="completed",
                successful_count=1, error_count=0,
            ),
            WaveState(
                index=2, task_count=1,
                task_summaries=["currband→rtlsdr-CUR0"],
                task_statuses=["running"],
                status="running",
            ),
        ]
        w.update_plan(waves, current_index=2)
        rendered = str(w.render())
        # ALL three task lines should be present
        assert "OLD0" in rendered
        assert "MID0" in rendered
        assert "CUR0" in rendered


# ─────────────────────────────────────────────────────────────────────
# Item 1+2 reducer: task status updates from events
# ─────────────────────────────────────────────────────────────────────


class TestTaskStatusReducer:
    def test_task_index_exact_match(self):
        w = WaveState(
            index=0, task_count=2,
            task_summaries=["band_a→rtlsdr-A", "band_b→rtlsdr-B"],
            task_statuses=["pending", "pending"],
        )
        assert _task_index_in_wave(w, "band_a", "rtlsdr-A") == 0
        assert _task_index_in_wave(w, "band_b", "rtlsdr-B") == 1

    def test_task_index_band_prefix_fallback(self):
        # When suggested_dongle_id was None at plan time, the summary
        # is "band_a→?" — the reducer falls back to band-prefix match.
        w = WaveState(
            index=0, task_count=1,
            task_summaries=["band_a→?"],
            task_statuses=["pending"],
        )
        assert _task_index_in_wave(w, "band_a", "rtlsdr-A") == 0

    def test_task_index_returns_none_on_no_match(self):
        w = WaveState(
            index=0, task_count=1,
            task_summaries=["band_a→x"],
            task_statuses=["pending"],
        )
        assert _task_index_in_wave(w, "nonexistent", "y") is None

    def test_task_index_returns_none_on_ambiguous_band(self):
        # Same band twice in a wave (unusual but possible) means
        # band-prefix can't disambiguate — return None rather than
        # guess. Caller treats None as no-op.
        w = WaveState(
            index=0, task_count=2,
            task_summaries=["band_a→?", "band_a→?"],
            task_statuses=["pending", "pending"],
        )
        # Exact match would still work — but here both are "band_a→?"
        # so even exact match against "band_a→x" wouldn't hit. Falls
        # through to prefix match which finds 2 → returns None.
        assert _task_index_in_wave(w, "band_a", "x") is None


# ─────────────────────────────────────────────────────────────────────
# Item 3: proc stats live on HeaderBar
# ─────────────────────────────────────────────────────────────────────


class TestProcStatsOnHeader:
    def _make_header(self):
        from unittest.mock import MagicMock
        from rfcensus.tui.widgets.header import HeaderBar

        class _SizedHeader(HeaderBar):
            @property
            def size(self):  # type: ignore[override]
                return MagicMock(width=200, height=1)

        return _SizedHeader(site_name="test")

    def test_set_proc_stats_appears_in_render(self):
        h = self._make_header()
        h.set_proc_stats(cpu_str="42%", rss_str="512 MB")
        rendered = str(h.render())
        assert "42%" in rendered
        assert "512 MB" in rendered

    def test_default_placeholders_present(self):
        h = self._make_header()
        rendered = str(h.render())
        # Until set_proc_stats is called, "—%" and "— MB" placeholders
        # render so the slot is reserved (no jumpy layout when stats
        # finally come in).
        assert "—%" in rendered or "cpu" in rendered.lower()


class TestFooterDoesNotShowProcStats:
    """v0.6.17: proc stats removed from FooterBar render. The
    set_proc_stats() method is kept as a no-op for backward compat
    with any external callers."""

    def test_set_proc_stats_is_noop(self):
        # v0.6.17: pinned no-op (proc stats moved to HeaderBar).
        # v0.7.4: briefly reversed.
        # v0.7.5: reverted again after the HeaderBar render bug
        # was root-caused (DongleStrip's dock:top was covering the
        # HeaderBar at y=0). With the dock fix, header is reliable
        # and the footer reverts to its original v0.6.17 no-op shim.
        from rfcensus.tui.widgets.footer import FooterBar
        f = FooterBar()
        f.set_proc_stats("99%", "999 MB")
        rendered = str(f.render())
        assert "99%" not in rendered
        assert "999 MB" not in rendered


# ─────────────────────────────────────────────────────────────────────
# Item 4+5: dongle detail metrics
# ─────────────────────────────────────────────────────────────────────


class TestDongleStateNewMetrics:
    def test_new_fields_default_correctly(self):
        d = DongleState(dongle_id="x")
        assert d.fanout_bytes_sent == 0
        assert d.bands_visited == set()
        assert d.band_started_at is None
        assert d.antenna_id is None

    def test_seed_dongle_metadata_sets_antenna(self):
        s = TUIState()
        seed_dongle_metadata(s, "rtlsdr-A",
                             model="rtlsdr_v3", antenna_id="whip_915")
        assert len(s.dongles) == 1
        d = s.dongles[0]
        assert d.antenna_id == "whip_915"
        assert d.model == "rtlsdr_v3"

    def test_seed_dongle_metadata_idempotent(self):
        s = TUIState()
        seed_dongle_metadata(s, "rtlsdr-A", antenna_id="whip_915")
        seed_dongle_metadata(s, "rtlsdr-A", antenna_id="marine_vhf")
        # Re-seed updates the value (allows live config reload)
        assert s.dongles[0].antenna_id == "marine_vhf"
        # Still only one dongle
        assert len(s.dongles) == 1


class TestDongleDetailRender:
    def _build_state(self, **dongle_kw):
        s = TUIState()
        defaults = dict(
            dongle_id="rtlsdr-A", status="active",
            consumer="rtl_433:band_a", band_id="band_a",
            freq_hz=433_920_000, sample_rate=2_400_000,
            decodes_in_band=14, detections_in_band=17,
            fanout_clients=2, fanout_bytes_sent=12_345_678,
            bands_visited={"band_a", "band_b"},
            antenna_id="whip_433",
            model="rtlsdr_v3",
        )
        defaults.update(dongle_kw)
        s.dongles.append(DongleState(**defaults))
        return s

    def _render(self, state):
        from rfcensus.tui.widgets.dongle_detail import DongleDetail
        d = DongleDetail()
        d.update_state(state.dongles, 0, [])
        return str(d.render())

    def test_antenna_visible_in_detail(self):
        s = self._build_state()
        rendered = self._render(s)
        assert "whip_433" in rendered, (
            "Antenna ID must appear on dongle detail (v0.6.17 item 7)"
        )

    def test_no_total_decodes_or_detections_shown(self):
        # Item 5: the "(total)" rows are gone — they were misleading
        # because they mixed productivity across different bands.
        s = self._build_state(decodes_total=99, detections_total=88)
        rendered = self._render(s)
        # The numbers from (total) shouldn't appear; only in-band
        assert " 99" not in rendered  # decodes_total
        assert " 88" not in rendered  # detections_total
        # In-band counts ARE shown
        assert "14" in rendered
        assert "17" in rendered

    def test_bytes_streamed_visible(self):
        s = self._build_state()
        rendered = self._render(s)
        # 12.3 MB or similar formatted byte count
        assert "MB" in rendered or "KB" in rendered

    def test_bands_visited_count_visible(self):
        s = self._build_state(bands_visited={"a", "b", "c"})
        rendered = self._render(s)
        assert "3" in rendered  # bands_visited count

    def test_on_band_time_visible_when_active(self):
        from datetime import timedelta
        started = datetime.now(timezone.utc) - timedelta(seconds=125)
        s = self._build_state(band_started_at=started)
        rendered = self._render(s)
        # Should show "on band" and some time string. Format is
        # like "2m05s" for 125 seconds.
        assert "on band" in rendered.lower()


# ─────────────────────────────────────────────────────────────────────
# Item 6: snapshot → report rename + richer content
# ─────────────────────────────────────────────────────────────────────


class TestReportNaming:
    def test_footer_hint_says_report_not_snapshot(self):
        from rfcensus.tui.widgets.footer import FooterBar
        f = FooterBar()
        assert "report" in f.HINT.lower()
        assert "snapshot" not in f.HINT.lower(), (
            "v0.6.17: 's' key is now labeled 'report' (not 'snapshot') "
            "since it shows the in-flight scan state, not a save-to-file"
        )


# ─────────────────────────────────────────────────────────────────────
# Item 8: log-mode toggle no longer kills runner
# ─────────────────────────────────────────────────────────────────────


class TestLogModeToggleBugFix:
    """Regression test for v0.6.17 bug: pressing 'l' (toggle log mode)
    while the runner was alive cancelled the runner due to an
    unconditional pending.cancel() loop after asyncio.wait. The fix
    in inventory.py checks tui_app._log_mode_requested before
    cancelling and skips runner_task in that case.

    We can't easily test the inventory entrypoint in isolation, but
    we can test the contract the fix relies on: TUIApp sets the flag
    before calling exit() when 'l' is pressed.
    """

    def test_action_toggle_log_mode_sets_flag(self):
        from rfcensus.tui.app import TUIApp

        # Build a stand-in app object without invoking textual lifecycle
        app = TUIApp.__new__(TUIApp)
        app._log_mode_requested = False

        # Stub exit() so action_toggle_log_mode doesn't try to do
        # textual things
        called = []
        app.exit = lambda *a, **k: called.append(("exit", a, k))

        TUIApp.action_toggle_log_mode(app)
        assert app._log_mode_requested is True, (
            "action_toggle_log_mode must set _log_mode_requested=True "
            "BEFORE calling exit() — the inventory.py log-mode bug fix "
            "relies on reading this flag to know whether to keep the "
            "runner running"
        )
        assert called  # exit was called
