"""v0.6.14 — TUI visual + interaction overhaul.

Five user-facing changes, five test groups:

  1. Plan tree widened from 28 to 36 chars + smart task abbreviation.
     Long "interlogix_security→rtlsdr-00000043" no longer wraps.

  2. Inline dongle-detail pane (replaces modal Screen). Pressing
     1-9 selects+swaps; Esc returns to events. No more modal stack
     when the user mashes number keys in succession.

  3. Status-driven border colors on the strip (green/grey/yellow/red
     based on dongle status + decode activity), with selection
     encoded as a heavy white border. Decoupled axes — a yellow-
     status tile can be selected without color confusion.

  4. Wave transitions visible in minimal filter mode. Highlighted
     severity + ▶/✓/✗ markers so they punch through the chatter.

  5. Per-dongle detection counters (detections_in_band /
     detections_total) wired from DetectionEvent metadata.band_id
     via active_tasks lookup. Reset on band change like decodes.

  6. Plan tree keeps showing the most recently completed wave's
     tasks, so the user keeps seeing what just ran when wave N
     starts.

  7. Process resource indicators (cpu/RSS) sampled cheaply from
     /proc/self once per tick.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock


# ─────────────────────────────────────────────────────────────────────
# 1. Plan tree widening + smart task abbreviation
# ─────────────────────────────────────────────────────────────────────


class TestPlanTreeWidening:
    def test_width_bumped(self):
        from rfcensus.tui.widgets.plan_tree import _PLAN_TREE_WIDTH
        # v0.6.13 was 28; v0.6.14 must be wider so the "interlogix_
        # security→rtlsdr-..." lines stop wrapping mid-token.
        assert _PLAN_TREE_WIDTH >= 32

    def test_abbreviate_task_summary_unchanged_when_short(self):
        from rfcensus.tui.widgets.plan_tree import _abbreviate_task_summary
        assert _abbreviate_task_summary("ais→rtlsdr-1") == "ais→rtlsdr-1"

    def test_abbreviate_task_summary_strips_rtlsdr_prefix(self):
        from rfcensus.tui.widgets.plan_tree import _abbreviate_task_summary
        long = "interlogix_security→rtlsdr-00000043"
        out = _abbreviate_task_summary(long)
        # Strips driver prefix and keeps last 4 chars of serial
        assert "rtlsdr-" not in out
        assert "0043" in out
        assert "interlogix_security" in out

    def test_abbreviate_task_summary_handles_hackrf_prefix(self):
        from rfcensus.tui.widgets.plan_tree import _abbreviate_task_summary
        out = _abbreviate_task_summary(
            "wide_uhf_scanner→hackrf-deadbeef0001"
        )
        assert "hackrf-" not in out
        assert "0001" in out


# ─────────────────────────────────────────────────────────────────────
# 2. main_pane_mode swaps inline (no modal stack)
# ─────────────────────────────────────────────────────────────────────


class TestMainPaneMode:
    def test_default_mode_is_events(self):
        from rfcensus.tui.state import TUIState
        s = TUIState()
        assert s.main_pane_mode == "events"

    def test_open_detail_swaps_to_dongle_mode(self):
        # Doesn't push a screen; just changes state.main_pane_mode.
        # Tested via the action method directly (not pilot) since
        # we just want to verify the side effect.
        from rfcensus.events import HardwareEvent
        from rfcensus.tui.app import TUIApp
        from rfcensus.tui.state import reduce
        app = TUIApp(runner=None, no_color=True, site_name="t")
        for did in ["d1", "d2", "d3"]:
            reduce(app.state, HardwareEvent(dongle_id=did, kind="detected"))
        # Without _refresh_all (no Textual mounted), just verify the
        # state change. action_open_detail will call _refresh_all
        # which catches all exceptions on missing widgets.
        app.action_open_detail()
        assert app.state.main_pane_mode == "dongle"

    def test_escape_returns_to_events(self):
        from rfcensus.events import HardwareEvent
        from rfcensus.tui.app import TUIApp
        from rfcensus.tui.state import reduce
        app = TUIApp(runner=None, no_color=True, site_name="t")
        reduce(app.state, HardwareEvent(dongle_id="d1", kind="detected"))
        app.state.main_pane_mode = "dongle"
        app.action_escape()
        assert app.state.main_pane_mode == "events"

    def test_escape_no_op_when_already_events(self):
        # Esc in events mode should not change anything (modal screens
        # like ConfirmQuit handle their own Esc before this fires).
        from rfcensus.tui.app import TUIApp
        app = TUIApp(runner=None, no_color=True, site_name="t")
        assert app.state.main_pane_mode == "events"
        app.action_escape()
        assert app.state.main_pane_mode == "events"


# ─────────────────────────────────────────────────────────────────────
# 3. Status-driven border colors on the strip
# ─────────────────────────────────────────────────────────────────────


class TestDongleBorderColor:
    """The new color palette: green (active+decodes), grey (active
    no decodes / idle), yellow (degraded), red (failed)."""

    def test_active_with_decodes_is_green(self):
        from rfcensus.tui.color import dongle_border_color
        assert dongle_border_color("active", has_decodes=True) == "green"

    def test_active_without_decodes_is_grey(self):
        from rfcensus.tui.color import dongle_border_color
        assert dongle_border_color("active", has_decodes=False) == "grey50"

    def test_idle_is_grey(self):
        from rfcensus.tui.color import dongle_border_color
        assert dongle_border_color("idle", has_decodes=False) == "grey50"

    def test_degraded_is_yellow(self):
        from rfcensus.tui.color import dongle_border_color
        assert dongle_border_color("degraded", has_decodes=False) == "yellow"
        # Even with decodes, degraded is yellow — degraded is the
        # primary signal
        assert dongle_border_color("degraded", has_decodes=True) == "yellow"

    def test_failed_is_red(self):
        from rfcensus.tui.color import dongle_border_color
        assert dongle_border_color("failed", has_decodes=False) == "red"
        assert (
            dongle_border_color("permanent_failed", has_decodes=False) == "red"
        )

    def test_unknown_status_falls_through_to_grey(self):
        from rfcensus.tui.color import dongle_border_color
        assert dongle_border_color("garbage", has_decodes=False) == "grey50"


class TestDongleStripCSSConstraints:
    """Regression guard for the Textual CSS hang we hit in v0.6.14
    development. Literal color name `grey50` and compound selectors
    like `.-selected.-status-X` cause Textual to hang at
    `WaitForScreenTimeout`. Lock in the working CSS shape."""

    def test_no_grey50_in_widget_css(self):
        from rfcensus.tui.widgets.dongle_strip import DongleTile
        # The DongleTile CSS must not use the literal color 'grey50'
        # in any selector — known Textual hang. Use bare 'grey'.
        assert "grey50" not in DongleTile.DEFAULT_CSS, (
            "DongleTile.DEFAULT_CSS must not use 'grey50' (causes "
            "Textual pilot hang). Use 'grey' instead."
        )

    def test_no_compound_class_selectors(self):
        from rfcensus.tui.widgets.dongle_strip import DongleTile
        # Compound class selectors like `.-selected.-status-green`
        # cause Textual pilot hang. Each rule must use at most one
        # class selector.
        for line in DongleTile.DEFAULT_CSS.splitlines():
            if "DongleTile" not in line or "{" not in line:
                continue
            selector = line.split("{")[0].strip()
            class_count = selector.count(".")
            assert class_count <= 1, (
                f"DongleTile CSS rule '{selector}' has multiple class "
                f"selectors — known Textual hang trigger. Decompose."
            )


# ─────────────────────────────────────────────────────────────────────
# 4. Wave transitions visible + visually punchy
# ─────────────────────────────────────────────────────────────────────


class TestWaveTransitionVisibility:
    def test_minimal_filter_includes_wave(self):
        from rfcensus.tui.state import FILTER_CATEGORIES
        assert "wave" in FILTER_CATEGORIES["minimal"], (
            "Minimal filter must include 'wave' so users see the "
            "transition heartbeat. Without this, the stream looks "
            "empty during multi-minute waves."
        )

    def test_wave_started_uses_highlight_severity(self):
        from rfcensus.events import WaveStartedEvent
        from rfcensus.tui.state import TUIState, reduce
        s = TUIState()
        reduce(s, WaveStartedEvent(wave_index=0, pass_n=1, task_count=4))
        wave_entries = [e for e in s.stream if e.category == "wave"]
        assert wave_entries, "wave-started must push to stream"
        assert wave_entries[0].severity == "highlight"
        assert "▶" in wave_entries[0].text

    def test_wave_completed_clean_uses_check_marker(self):
        from rfcensus.events import WaveCompletedEvent
        from rfcensus.tui.state import TUIState, reduce
        s = TUIState()
        reduce(s, WaveCompletedEvent(
            wave_index=0, pass_n=1, task_count=4,
            successful_count=4, errors=[],
        ))
        wave_entries = [e for e in s.stream if e.category == "wave"]
        assert wave_entries[0].severity == "highlight"
        assert "✓" in wave_entries[0].text

    def test_wave_completed_with_errors_uses_x_marker(self):
        from rfcensus.events import WaveCompletedEvent
        from rfcensus.tui.state import TUIState, reduce
        s = TUIState()
        reduce(s, WaveCompletedEvent(
            wave_index=0, pass_n=1, task_count=4,
            successful_count=2,
            errors=["x failed", "y failed"],
        ))
        wave_entries = [e for e in s.stream if e.category == "wave"]
        assert wave_entries[0].severity == "warning"
        assert "✗" in wave_entries[0].text
        assert "2 error(s)" in wave_entries[0].text


# ─────────────────────────────────────────────────────────────────────
# 5. Per-dongle detection counters
# ─────────────────────────────────────────────────────────────────────


class TestPerDongleDetectionCounters:
    def test_dongle_state_has_detection_fields(self):
        from rfcensus.tui.state import DongleState
        d = DongleState(dongle_id="d1")
        assert d.detections_in_band == 0
        assert d.detections_total == 0
        assert d.last_detection_at is None

    def test_detection_event_attributes_to_dongle_via_band_id(self):
        """When a DetectionEvent carries metadata.band_id and there's
        an active task on that band, the dongle running that task gets
        its counters incremented."""
        from datetime import datetime, timezone
        from rfcensus.events import (
            DetectionEvent, HardwareEvent, TaskStartedEvent,
        )
        from rfcensus.tui.state import TUIState, reduce

        s = TUIState()
        # Set up dongle + active task on band "915_ism_r900"
        reduce(s, HardwareEvent(dongle_id="d1", kind="detected"))
        reduce(s, TaskStartedEvent(
            wave_index=0, pass_n=1,
            band_id="915_ism_r900", dongle_id="d1",
            consumer="lora_survey",
        ))

        # Now publish a detection with band_id metadata
        reduce(s, DetectionEvent(
            detector_name="lora_survey",
            technology="meshtastic",
            freq_hz=913_125_000, bandwidth_hz=250_000,
            confidence=0.85, evidence="SF9",
            metadata={"band_id": "915_ism_r900"},
        ))
        d = next(d for d in s.dongles if d.dongle_id == "d1")
        assert d.detections_in_band == 1
        assert d.detections_total == 1
        assert d.last_detection_at is not None

    def test_detections_in_band_resets_on_band_change(self):
        from rfcensus.events import HardwareEvent, TaskStartedEvent, DetectionEvent
        from rfcensus.tui.state import TUIState, reduce

        s = TUIState()
        reduce(s, HardwareEvent(dongle_id="d1", kind="detected"))
        # Allocate to band A, generate a detection
        reduce(s, HardwareEvent(
            dongle_id="d1", kind="allocated",
            band_id="A", freq_hz=1, sample_rate=1, consumer="x",
        ))
        reduce(s, TaskStartedEvent(
            wave_index=0, pass_n=1, band_id="A", dongle_id="d1",
            consumer="x",
        ))
        reduce(s, DetectionEvent(
            detector_name="x", technology="t", freq_hz=1, bandwidth_hz=1,
            confidence=0.5, evidence="", metadata={"band_id": "A"},
        ))
        d = next(d for d in s.dongles if d.dongle_id == "d1")
        assert d.detections_in_band == 1
        assert d.detections_total == 1

        # Reallocate to band B → in-band counter resets, total stays
        reduce(s, HardwareEvent(
            dongle_id="d1", kind="allocated",
            band_id="B", freq_hz=2, sample_rate=2, consumer="y",
        ))
        d = next(d for d in s.dongles if d.dongle_id == "d1")
        assert d.detections_in_band == 0
        assert d.detections_total == 1  # not reset

    def test_detection_without_band_id_metadata_still_counted_globally(self):
        # Should not crash + should still increment global counter
        from rfcensus.events import DetectionEvent
        from rfcensus.tui.state import TUIState, reduce
        s = TUIState()
        reduce(s, DetectionEvent(
            detector_name="x", technology="t",
            freq_hz=1, bandwidth_hz=1, confidence=0.5, evidence="",
        ))
        assert s.total_detections == 1


# ─────────────────────────────────────────────────────────────────────
# 6. Plan tree shows most recent completed wave's tasks
# ─────────────────────────────────────────────────────────────────────


class TestPlanTreeShowsLastCompletedWave:
    def _make_widget(self):
        from rfcensus.tui.widgets.plan_tree import PlanTree

        class _SizedTree(PlanTree):
            @property
            def size(self):  # type: ignore[override]
                return MagicMock(width=40, height=30)

        return _SizedTree()

    def test_completed_wave_tasks_shown_when_it_is_most_recent(self):
        from rfcensus.tui.state import WaveState
        w = self._make_widget()
        waves = [
            WaveState(
                index=0, task_count=2, status="completed",
                successful_count=2, error_count=0,
                task_summaries=["bandA→rtlsdr-1234", "bandB→rtlsdr-5678"],
            ),
            WaveState(
                index=1, task_count=1, status="running",
                task_summaries=["bandC→rtlsdr-9999"],
            ),
        ]
        w.update_plan(waves, current_index=1)
        rendered = str(w.render())
        # Both waves' tasks should appear
        assert "1234" in rendered
        assert "5678" in rendered
        assert "9999" in rendered

    def test_older_completed_waves_collapsed(self):
        # v0.6.17: this test name is now misleading. Per item 2 in
        # user feedback, the plan tree now shows tasks for ALL waves
        # (including older completed ones) so the user can see at a
        # glance which task within an earlier failed wave was the
        # failure. The "collapse" behavior was actively unhelpful —
        # users were scrolling event logs to find what failed.
        # Test now asserts the OPPOSITE: older completed waves DO
        # show their tasks.
        from rfcensus.tui.state import WaveState
        w = self._make_widget()
        waves = [
            WaveState(
                index=0, task_count=1, status="completed",
                successful_count=1, error_count=0,
                task_summaries=["oldband→rtlsdr-OLD0"],
                task_statuses=["ok"],
            ),
            WaveState(
                index=1, task_count=1, status="completed",
                successful_count=1, error_count=0,
                task_summaries=["newband→rtlsdr-NEW1"],
                task_statuses=["ok"],
            ),
            WaveState(
                index=2, task_count=1, status="running",
                task_summaries=["currband→rtlsdr-CUR2"],
                task_statuses=["running"],
            ),
        ]
        w.update_plan(waves, current_index=2)
        rendered = str(w.render())
        # All three waves' tasks now visible
        assert "OLD0" in rendered, (
            "v0.6.17: older completed waves now keep their tasks "
            "visible so the user can see which task ran."
        )
        assert "NEW1" in rendered
        assert "CUR2" in rendered


# ─────────────────────────────────────────────────────────────────────
# 7. Process resource indicators (cpu/RSS)
# ─────────────────────────────────────────────────────────────────────


class TestProcSampler:
    def test_first_sample_returns_none_cpu(self):
        from rfcensus.tui.proc_stats import ProcSampler
        s = ProcSampler()
        snap = s.sample()
        # First call has no delta yet, so cpu should be None
        assert snap.cpu_percent is None

    def test_second_sample_returns_a_number(self):
        # Second call should produce a real (non-None) cpu_percent —
        # even 0.0 if the process is idle.
        import time
        from rfcensus.tui.proc_stats import ProcSampler
        s = ProcSampler()
        s.sample()
        time.sleep(0.05)  # ensure d_wall > 0
        snap = s.sample()
        # It might still be None on non-Linux; on Linux it should
        # return a float >= 0.
        if snap.cpu_percent is not None:
            assert snap.cpu_percent >= 0.0

    def test_rss_sampled_or_none(self):
        from rfcensus.tui.proc_stats import ProcSampler
        s = ProcSampler()
        snap = s.sample()
        # On Linux: positive integer. Elsewhere: None.
        if snap.rss_bytes is not None:
            assert snap.rss_bytes > 0

    def test_format_helpers(self):
        from rfcensus.tui.proc_stats import format_cpu, format_rss
        assert format_cpu(None) == "—%"
        assert format_cpu(12.4) == "12%"
        assert format_cpu(99.6) == "100%"
        assert format_rss(None) == "— MB"
        assert format_rss(150 * 1024 * 1024) == "150 MB"
        assert "GB" in format_rss(2 * 1024 * 1024 * 1024)


class TestProcStatsInHeader:
    """v0.6.17: proc stats moved from FooterBar to HeaderBar — they're
    easier to read in the always-visible single-line header than at
    the right end of the footer counter line. The footer's
    set_proc_stats() is now a no-op shim for backward compat."""

    def _make_header(self):
        from rfcensus.tui.widgets.header import HeaderBar

        class _SizedHeader(HeaderBar):
            @property
            def size(self):  # type: ignore[override]
                return MagicMock(width=200, height=1)

        return _SizedHeader(site_name="test")

    def test_default_proc_stats_show_placeholders(self):
        h = self._make_header()
        rendered = str(h.render())
        # Render returns a string with "—%" and "— MB" placeholders
        # in the proc-stats slot until set_proc_stats is called.
        assert "—%" in rendered or "cpu" in rendered

    def test_set_proc_stats_updates_render(self):
        h = self._make_header()
        h.set_proc_stats(cpu_str="12%", rss_str="145 MB")
        rendered = str(h.render())
        assert "12%" in rendered
        assert "145 MB" in rendered

    def test_set_proc_stats_idempotent_when_unchanged(self):
        h = self._make_header()
        h.set_proc_stats(cpu_str="12%", rss_str="145 MB")
        # Second identical call should not raise. No good way to
        # assert "no refresh queued" without Textual machinery.
        h.set_proc_stats(cpu_str="12%", rss_str="145 MB")  # OK

    def test_footer_set_proc_stats_is_noop_shim(self):
        # v0.6.17: this test pinned "footer is a no-op for proc stats"
        # after the original move to HeaderBar.
        # v0.7.4: briefly reversed the decision because HeaderBar
        # wasn't rendering for the user.
        # v0.7.5: HeaderBar dock-stack bug fixed (DongleStrip was
        # covering it via overlapping dock:top). Reverted to the
        # original v0.6.17 design — proc stats live in the header,
        # footer keeps a no-op shim for backward compat.
        from rfcensus.tui.widgets.footer import FooterBar

        class _SizedFooter(FooterBar):
            @property
            def size(self):  # type: ignore[override]
                return MagicMock(width=200, height=2)

        f = _SizedFooter()
        before = str(f.render())
        f.set_proc_stats("99%", "999 MB")
        after = str(f.render())
        assert "99%" not in after
        assert "999 MB" not in after
        assert before == after


# ─────────────────────────────────────────────────────────────────────
# 8. Footer hint phrasing
# ─────────────────────────────────────────────────────────────────────


class TestFooterHintPhrasing:
    def test_dongle_keys_phrasing_is_clearer(self):
        from rfcensus.tui.widgets.footer import FooterBar
        # User feedback: "1-0 dongle" was cryptic. v0.6.14 uses
        # "No. Keys 1 to 0 dongle".
        assert "No. Keys 1 to 0 dongle" in FooterBar.HINT


# ─────────────────────────────────────────────────────────────────────
# 9. Help overlay reflects v0.6.14 inline-pane Esc behavior
# ─────────────────────────────────────────────────────────────────────


class TestHelpOverlayUpdated:
    def test_esc_returns_to_events_documented(self):
        from rfcensus.tui.widgets.modals import _help_text
        text = _help_text().lower()
        # v0.6.16: rephrased as "close detail pane (back to events)"
        # which carries the same meaning. Accept either phrasing so
        # this test guards intent rather than exact wording.
        assert ("return to events" in text or
                "back to events" in text or
                "close detail pane" in text), (
            "Help text must document Esc behavior (close detail pane / "
            "return to events view)"
        )
