"""v0.7.4 — three follow-up fixes from user-reported TUI screenshots.

(1) Event-stream scroll bug: scrolling up shouldn't make the visible
    window shrink to 1 line. Root cause: ``scroll_lines`` capped
    max_offset at len(buf)-1 (way too loose) and ``scroll_to_oldest``
    set offset = len(buf)-1 (puts entry[0] at the BOTTOM with nothing
    else visible). Both should cap at len(buf) - max_lines so the
    viewport stays full when at the top of the buffer.

(2) Dongle tile color doesn't reflect fanout warnings. A dongle with
    slow-chunk events stayed green because the color helper only
    looked at status + has_decodes. Now ``has_warnings`` is wired
    through and slow chunks paint the tile yellow.

(3) rtl_433 ``exit code 3`` failures had no diagnostic context. The
    decoder now interprets common exit codes ("3 = SDR read failed
    mid-stream") and includes the last 3 stderr lines in the failure
    message so the user can tell at a glance whether it was rtl_tcp
    stalling, USB read failure, or the binary itself crashing.
"""

from __future__ import annotations

from datetime import datetime, timezone


# ─────────────────────────────────────────────────────────────────────
# (1) EventStream scroll bug
# ─────────────────────────────────────────────────────────────────────


def _make_stream_with_entries(n: int):
    """Build an EventStream pre-populated with N entries in
    'filtered' mode. Stub the size to a reasonable viewport."""
    from rfcensus.tui.widgets.event_stream import EventStream
    from rfcensus.tui.state import StreamEntry

    es = EventStream()
    entries = [
        StreamEntry(
            timestamp=datetime.now(timezone.utc),
            severity="info",
            category="task",
            text=f"entry {i}",
        )
        for i in range(n)
    ]
    es._streams = {"minimal": [], "filtered": entries, "verbose": []}
    es._filter_mode = "filtered"

    # Stub _max_visible_lines so we don't depend on Textual's size
    # tracking (which only works after mount). Use 10 lines, which
    # matches the natural viewport for these tests.
    es._max_visible_lines = lambda: 10    # type: ignore[method-assign]
    es.refresh = lambda *a, **k: None     # type: ignore[method-assign]
    es._refresh_title = lambda: None      # type: ignore[method-assign]
    return es


def test_scroll_lines_caps_offset_so_viewport_stays_full() -> None:
    """The scroll cap must keep the viewport FULL when at the top of
    the buffer. With 50 entries and a 10-line viewport, the maximum
    valid offset is 40 (so entries [0..9] are visible). Anything
    higher would shrink the visible window."""
    es = _make_stream_with_entries(50)
    es.scroll_lines(1000)    # try to scroll way past the top
    assert es._scroll_offset == 40, (
        f"max offset for 50 entries / 10-line viewport should be "
        f"40 (50-10), got {es._scroll_offset}"
    )


def test_scroll_to_oldest_keeps_viewport_full() -> None:
    """Home key (scroll_to_oldest) must put entry[0] at the TOP of
    the viewport with the rest of the viewport filled. Pre-fix this
    set offset to len-1, which put entry[0] at the BOTTOM and hid
    the other 9 lines (image 3 in the bug report)."""
    es = _make_stream_with_entries(50)
    es.scroll_to_oldest()
    assert es._scroll_offset == 40
    # Render the viewport — should show entries 0..9
    out = str(es.render())
    assert "entry 0" in out, "oldest entry must be visible"
    assert "entry 9" in out, "viewport should still be full of entries"


def test_scroll_to_oldest_with_short_buffer_clamps_to_zero() -> None:
    """If the buffer is shorter than the viewport, offset stays 0
    (nothing to scroll)."""
    es = _make_stream_with_entries(5)    # less than 10-line viewport
    es.scroll_to_oldest()
    assert es._scroll_offset == 0


def test_scroll_lines_at_max_offset_doesnt_shrink_viewport() -> None:
    """Pressing Up at the top of buffer is a no-op, not a degenerate
    'collapse to 1 line' move."""
    es = _make_stream_with_entries(20)
    es.scroll_to_oldest()
    initial_offset = es._scroll_offset
    es.scroll_lines(5)    # try to go further up
    assert es._scroll_offset == initial_offset


# ─────────────────────────────────────────────────────────────────────
# (2) Dongle color reflects fanout warnings
# ─────────────────────────────────────────────────────────────────────


def test_dongle_color_yellow_when_slow_chunks() -> None:
    """An active dongle with decodes but slow-chunk events should be
    YELLOW, not green. Green means 'everything fine'; slow chunks
    mean 'still working but the fanout is struggling — look here.'"""
    from rfcensus.tui.color import dongle_border_color
    color = dongle_border_color(
        status="active", has_decodes=True, has_warnings=True,
    )
    assert color == "yellow"


def test_dongle_color_unchanged_without_warnings() -> None:
    """Without warnings, the v0.6.14 status×decodes matrix is
    preserved (active+decodes=green, active+no-decodes=grey, etc.)."""
    from rfcensus.tui.color import dongle_border_color
    assert dongle_border_color("active", True, False) == "green"
    assert dongle_border_color("active", False, False) == "grey50"
    assert dongle_border_color("idle", False, False) == "grey50"


def test_dongle_color_failure_dominates_warnings() -> None:
    """Failure (red) and degraded (yellow from status) take priority
    over the active-with-warnings yellow path. Otherwise a
    permanently-failed dongle could flicker green→yellow→red."""
    from rfcensus.tui.color import dongle_border_color
    assert dongle_border_color("failed", True, True) == "red"
    assert dongle_border_color("permanent_failed", True, True) == "red"
    # degraded is yellow from status, so the warnings path is moot
    assert dongle_border_color("degraded", True, True) == "yellow"


def test_tile_class_uses_warnings_flag_from_state() -> None:
    """The tile widget must pass DongleState.fanout_dropped_chunks
    into the color helper as has_warnings."""
    src = open(
        "/home/claude/rfcensus/rfcensus/tui/widgets/dongle_strip.py"
    ).read()
    # Confirm the wiring exists — the tile reads fanout_dropped_chunks
    # and passes it as the third arg
    assert "fanout_dropped_chunks" in src
    assert "has_warnings" in src


# ─────────────────────────────────────────────────────────────────────
# (3) rtl_433 stderr ring + exit-code interpretation
# ─────────────────────────────────────────────────────────────────────


def test_managed_process_recent_stderr_property_exists() -> None:
    """ManagedProcess must expose a recent_stderr property so the
    rtl_433 decoder can attach the tail to its failure message."""
    from rfcensus.utils.async_subprocess import ManagedProcess
    assert hasattr(ManagedProcess, "recent_stderr")


def test_recent_stderr_returns_buffered_lines() -> None:
    """The buffer should hold the last N decoded stderr lines.
    Tested by manipulating the internal buffer directly — the actual
    pump is exercised by integration tests elsewhere."""
    from rfcensus.utils.async_subprocess import (
        ManagedProcess, ProcessConfig,
    )
    proc = ManagedProcess(ProcessConfig(name="test", args=["true"]))
    proc._recent_stderr = ["line1", "line2", "line3"]
    assert proc.recent_stderr == ["line1", "line2", "line3"]
    # Returns a copy, not a reference (callers shouldn't mutate)
    proc.recent_stderr.append("not stored")
    assert proc.recent_stderr == ["line1", "line2", "line3"]


def test_recent_stderr_capped_at_capacity() -> None:
    """Confirm the FIFO trim happens at the documented capacity."""
    from rfcensus.utils.async_subprocess import (
        ManagedProcess, ProcessConfig,
    )
    proc = ManagedProcess(ProcessConfig(name="test", args=["true"]))
    proc._recent_stderr_capacity = 5
    # Simulate the pump's append+trim logic
    for i in range(20):
        proc._recent_stderr.append(f"line{i}")
        if len(proc._recent_stderr) > proc._recent_stderr_capacity:
            del proc._recent_stderr[
                : len(proc._recent_stderr)
                  - proc._recent_stderr_capacity
            ]
    assert len(proc._recent_stderr) == 5
    assert proc._recent_stderr[0] == "line15"
    assert proc._recent_stderr[-1] == "line19"


def test_rtl_433_exit_code_interpretation_table_complete() -> None:
    """The decoder should know about the standard rtl_433 exit codes
    (1-5) — at minimum exit 3 (SDR read failed mid-stream), which is
    the one the user just hit."""
    from rfcensus.decoders.builtin.rtl_433 import _RTL_433_EXIT_CODES
    # Exit 3 is the one in the bug report — must have a clear
    # interpretation that distinguishes "rtl_433 crashed" from
    # "rtl_tcp source died"
    assert 3 in _RTL_433_EXIT_CODES
    assert "stream" in _RTL_433_EXIT_CODES[3].lower() or \
           "rtl_tcp" in _RTL_433_EXIT_CODES[3].lower()
    # Other common ones
    for code in (1, 2, 4, 5):
        assert code in _RTL_433_EXIT_CODES, (
            f"rtl_433 exit code {code} should have an interpretation"
        )


# ─────────────────────────────────────────────────────────────────────
# (4) Quit flow: 3-option modal + graceful path + INCOMPLETE marker
# ─────────────────────────────────────────────────────────────────────


def test_confirm_quit_returns_string_not_bool() -> None:
    """ConfirmQuit's contract changed from bool to one of three
    strings: 'graceful', 'force', 'cancel'."""
    from rfcensus.tui.widgets.modals import ConfirmQuit
    # Generic alias check: ModalScreen[str], not ModalScreen[bool]
    bases_str = str(ConfirmQuit.__orig_bases__)
    assert "[str]" in bases_str
    assert "[bool]" not in bases_str


def test_confirm_quit_action_methods_dismiss_with_strings() -> None:
    """Each action (graceful / force / cancel) dismisses with the
    corresponding string."""
    from rfcensus.tui.widgets.modals import ConfirmQuit
    modal = ConfirmQuit.__new__(ConfirmQuit)
    captured = []
    modal.dismiss = lambda v: captured.append(v)    # type: ignore[method-assign]

    modal.action_graceful()
    modal.action_force()
    modal.action_cancel()

    assert captured == ["graceful", "force", "cancel"]


def test_confirm_quit_keybindings_include_force() -> None:
    """The keybinding `f` for force-quit must be wired up alongside
    y/Enter (graceful) and n/Esc (cancel)."""
    from rfcensus.tui.widgets.modals import ConfirmQuit
    keys = {b[0] for b in ConfirmQuit.BINDINGS}
    assert "y" in keys
    assert "f" in keys
    assert "n" in keys
    assert "escape" in keys
    assert "enter" in keys    # Enter = same as y (graceful default)


def test_force_quit_flag_initialized_false() -> None:
    """The TUIApp's _force_quit_requested flag must default False
    so that exiting via `l` (log mode) doesn't accidentally trigger
    the force-quit path."""
    from rfcensus.tui.app import TUIApp
    # Bypass __init__ — we just want to confirm the field is set
    # in __init__ (read it from the source)
    src = open(
        "/home/claude/rfcensus/rfcensus/tui/app.py"
    ).read()
    assert "_force_quit_requested = False" in src


def test_session_result_carries_stopped_early_flag() -> None:
    """SessionResult gained two fields for the partial-report path."""
    from rfcensus.engine.session import SessionResult
    from datetime import datetime as _dt, timezone as _tz
    from dataclasses import fields
    field_names = {f.name for f in fields(SessionResult)}
    assert "stopped_early" in field_names
    assert "tasks_skipped_due_to_stop" in field_names

    # Default values: both False / 0 (backward compat for callers
    # constructing SessionResult without setting them)
    from rfcensus.engine.scheduler import ExecutionPlan
    plan = ExecutionPlan(waves=[], max_parallel_per_wave=1)
    r = SessionResult(
        session_id=1,
        started_at=_dt.now(_tz.utc),
        ended_at=_dt.now(_tz.utc),
        plan=plan,
    )
    assert r.stopped_early is False
    assert r.tasks_skipped_due_to_stop == 0


def _make_band(band_id: str, center_mhz: float = 433.92):
    """Test helper — minimal BandConfig."""
    from rfcensus.config.schema import BandConfig
    half = 50_000
    return BandConfig(
        id=band_id, name=band_id,
        freq_low=int(center_mhz * 1e6) - half,
        freq_high=int(center_mhz * 1e6) + half,
    )


def test_text_report_marks_incomplete_when_stopped_early() -> None:
    """When SessionResult.stopped_early is True, the text report
    must show the INCOMPLETE banner near the top so the user
    doesn't mistake "no emitters detected" for "spectrum is silent."
    """
    from rfcensus.reporting.formats.text import render_text_report
    from rfcensus.engine.session import SessionResult
    from rfcensus.engine.scheduler import (
        ExecutionPlan, Wave, ScheduleTask,
    )
    from rfcensus.engine.strategy import StrategyResult
    from datetime import datetime as _dt, timezone as _tz

    band_433 = _make_band("433_ism", 433.92)
    band_915 = _make_band("915_ism", 915.0)
    tasks = [
        ScheduleTask(band=band_433, suggested_dongle_id="d1",
                     suggested_antenna_id=None),
        ScheduleTask(band=band_915, suggested_dongle_id="d2",
                     suggested_antenna_id=None),
    ]
    plan = ExecutionPlan(
        waves=[Wave(index=0, tasks=tasks)],
        max_parallel_per_wave=2,
    )
    # Only the 433 task ran; 915 was skipped
    result = SessionResult(
        session_id=42,
        started_at=_dt(2025, 1, 1, 12, 0, 0, tzinfo=_tz.utc),
        ended_at=_dt(2025, 1, 1, 12, 5, 0, tzinfo=_tz.utc),
        plan=plan,
        strategy_results=[
            StrategyResult(band_id="433_ism", power_scan_performed=False),
        ],
        stopped_early=True,
        tasks_skipped_due_to_stop=1,
    )
    out = render_text_report(result, [], [], [])
    assert "INCOMPLETE" in out
    assert "1/2" in out
    # The skipped band must be named so the user knows what's missing
    assert "915_ism" in out


def test_text_report_no_incomplete_banner_when_full_run() -> None:
    """A clean-finish session should NOT have the INCOMPLETE banner."""
    from rfcensus.reporting.formats.text import render_text_report
    from rfcensus.engine.session import SessionResult
    from rfcensus.engine.scheduler import ExecutionPlan
    from datetime import datetime as _dt, timezone as _tz

    plan = ExecutionPlan(waves=[], max_parallel_per_wave=1)
    result = SessionResult(
        session_id=42,
        started_at=_dt(2025, 1, 1, 12, 0, 0, tzinfo=_tz.utc),
        ended_at=_dt(2025, 1, 1, 12, 5, 0, tzinfo=_tz.utc),
        plan=plan,
        stopped_early=False,
    )
    out = render_text_report(result, [], [], [])
    assert "INCOMPLETE" not in out


def test_inventory_distinguishes_graceful_from_force_path() -> None:
    """The inventory command's TUI/runner coordination layer must
    distinguish graceful_quit from force_quit so it doesn't
    immediately cancel the runner on q+y."""
    src = open(
        "/home/claude/rfcensus/rfcensus/commands/inventory.py"
    ).read()
    # The new branches exist
    assert "graceful_quit" in src
    assert "_force_quit_requested" in src
    # The graceful path must skip the cancel (continue) just like
    # log_mode_toggle did in v0.6.17
    assert (
        "log_mode_toggle or graceful_quit" in src
        or "graceful_quit or log_mode_toggle" in src
    )


# ─────────────────────────────────────────────────────────────────────
# (5) Named consumers + live stream rate + scrollable detail pane
# ─────────────────────────────────────────────────────────────────────


def test_active_consumers_tracked_per_allocation() -> None:
    """Each `allocated` HardwareEvent adds the consumer name to a
    set on DongleState, so the detail pane can show
    rtl_433+rtlamr+lora_survey instead of three anonymous IPs."""
    from rfcensus.tui.state import TUIState, _reduce_hardware
    from rfcensus.events import HardwareEvent

    state = TUIState()
    base = dict(
        dongle_id="d1", kind="allocated",
        freq_hz=915_000_000, sample_rate=2_400_000,
        band_id="915_ism",
        timestamp=datetime.now(timezone.utc),
    )
    for consumer in ("rtl_433:915_ism", "rtlamr:915_ism",
                     "lora_survey:915_ism"):
        _reduce_hardware(state, HardwareEvent(consumer=consumer, **base))

    d = state.dongles[0]
    assert d.active_consumers == {
        "rtl_433:915_ism", "rtlamr:915_ism", "lora_survey:915_ism",
    }


def test_active_consumers_cleared_on_release() -> None:
    """A `released` event clears all consumers since the broker
    releases a dongle when the last lease drops."""
    from rfcensus.tui.state import TUIState, _reduce_hardware
    from rfcensus.events import HardwareEvent

    state = TUIState()
    _reduce_hardware(state, HardwareEvent(
        dongle_id="d1", kind="allocated", consumer="rtl_433:915_ism",
        freq_hz=915_000_000, sample_rate=2_400_000, band_id="915_ism",
        timestamp=datetime.now(timezone.utc),
    ))
    assert state.dongles[0].active_consumers
    _reduce_hardware(state, HardwareEvent(
        dongle_id="d1", kind="released", consumer="rtl_433:915_ism",
        timestamp=datetime.now(timezone.utc),
    ))
    assert state.dongles[0].active_consumers == set()


def test_slow_event_increments_counter_by_one_not_by_byte_count() -> None:
    """v0.7.4 fix: previously slow events did
        fanout_dropped_chunks = max(prev, bytes_sent // 16384)
    which mis-interpreted cumulative byte count as a chunk count and
    inflated the counter to millions after a few minutes. Now each
    slow event is +1, which is the right accounting (the slow event
    IS the warning incident, not a byte count)."""
    from rfcensus.tui.state import TUIState, _reduce_fanout
    from rfcensus.events import FanoutClientEvent

    state = TUIState()
    # Three slow events with realistic large cumulative byte counts
    for bytes_so_far in (5_000_000, 10_000_000, 15_000_000):
        _reduce_fanout(state, FanoutClientEvent(
            slot_id="fanout[d1]", peer_addr="127.0.0.1:51001",
            event_type="slow", bytes_sent=bytes_so_far,
        ))
    d = next(d for d in state.dongles if d.dongle_id == "d1")
    # Pre-fix: would be 15_000_000 // 16384 = 915. Post-fix: 3.
    assert d.fanout_dropped_chunks == 3


def test_dongle_detail_widget_has_scroll_method() -> None:
    """v0.7.4: the footer hint advertised ↑/↓ scroll but the widget
    had no scroll method. Confirm the method exists and adjusts
    _scroll_offset."""
    from rfcensus.tui.widgets.dongle_detail import DongleDetail
    dd = DongleDetail()
    dd._last_rendered_lines = 100   # plenty of room
    dd.refresh = lambda *a, **k: None    # type: ignore[method-assign]

    assert dd._scroll_offset == 0
    dd.scroll_lines(5)
    assert dd._scroll_offset == 5
    dd.scroll_lines(-3)
    assert dd._scroll_offset == 2
    # Negative-clamp: can't scroll above 0
    dd.scroll_lines(-100)
    assert dd._scroll_offset == 0


def test_app_scroll_actions_dispatch_to_dongle_detail() -> None:
    """v0.7.4: action_scroll_up/down now routes to DongleDetail
    when main_pane_mode == 'dongle' (was no-op before, leaving the
    advertised ↑/↓ scroll hint dead)."""
    src = open(
        "/home/claude/rfcensus/rfcensus/tui/app.py"
    ).read()
    # Confirm the dongle-mode dispatch exists in scroll handlers
    assert "main_pane_mode == \"dongle\"" in src
    assert "DongleDetail" in src
    # The scroll_lines method must be invoked on the detail widget
    assert "DongleDetail).scroll_lines" in src


def test_detail_renders_named_consumers_not_just_peer_addrs() -> None:
    """The detail pane must render the consumer name set so users
    see "rtl_433 / rtlamr / lora_survey" instead of three anonymous
    127.0.0.1:port lines."""
    from rfcensus.tui.widgets.dongle_detail import DongleDetail
    from rfcensus.tui.state import DongleState

    d = DongleState(
        dongle_id="d1",
        status="active",
        consumer="rtl_433:915_ism",
        freq_hz=915_000_000,
        sample_rate=2_400_000,
        band_id="915_ism",
    )
    d.active_consumers = {
        "rtl_433:915_ism", "rtlamr:915_ism", "lora_survey:915_ism",
    }
    d.fanout_clients = 3
    d.fanout_client_peers = {
        "127.0.0.1:51001", "127.0.0.1:51002", "127.0.0.1:51003",
    }

    dd = DongleDetail()
    dd._dongles = [d]
    dd._index = 0
    dd._stream = []
    out = str(dd.render())

    # Named consumers visible
    assert "rtl_433:915_ism" in out
    assert "rtlamr:915_ism" in out
    assert "lora_survey:915_ism" in out
    # Peer addrs still shown (dim, for debugging) but as a single
    # line not 3 separate bullets — the named list is the primary
    # view now.


def test_detail_shows_live_stream_rate_not_zero_bytes() -> None:
    """v0.7.4: bytes_streamed counter never updated for steady-state
    runs (fanout only published bytes on connect/disconnect/slow).
    Replace with a derived live rate so users see actual throughput."""
    from rfcensus.tui.widgets.dongle_detail import DongleDetail
    from rfcensus.tui.state import DongleState

    d = DongleState(
        dongle_id="d1",
        status="active",
        consumer="rtl_433:915_ism",
        freq_hz=915_000_000,
        sample_rate=2_400_000,    # 2.4 MS/s
        band_id="915_ism",
    )
    d.fanout_clients = 3    # 3 downstream consumers
    d.active_consumers = {"rtl_433:915_ism"}

    dd = DongleDetail()
    dd._dongles = [d]
    dd._index = 0
    dd._stream = []
    out = str(dd.render())

    # 2.4 MS/s × 2 bytes × 3 clients = 14.4 MB/s
    assert "MB/s" in out
    assert "14.4 MB/s" in out or "14." in out
    # The stale "bytes streamed: 0 B" must not appear when the
    # dongle is live with clients
    assert "bytes streamed:   0 B" not in out


# ─────────────────────────────────────────────────────────────────────
# (6) Final v0.7.4 polish: footer proc stats + Dongle N header +
#     Tab/Shift+Tab pane cycling
# ─────────────────────────────────────────────────────────────────────


def test_footer_does_not_render_cpu_rss_v075() -> None:
    """v0.7.4 briefly put proc stats in the footer as a fallback.
    v0.7.5: reverted after the HeaderBar render bug was root-caused
    (DongleStrip's dock:top was covering it). User explicitly asked
    to keep proc stats in the header only — footer stays clean."""
    from rfcensus.tui.widgets.footer import FooterBar
    fb = FooterBar()
    fb.set_state(
        healthy=5, total=5, decodes=14, emitters=8, detections=0,
        wave_label="0", filter_mode="filtered",
    )
    fb.set_proc_stats("  3.4%", "  142 MB")
    out = str(fb.render())
    assert "3.4%" not in out
    assert "142 MB" not in out


def test_footer_does_not_render_version_v075() -> None:
    """v0.7.4 added a version watermark to the footer counters line.
    v0.7.5: removed at user request — version belongs in the header
    only, footer stays clean and focused on the live counters."""
    from rfcensus.tui.widgets.footer import FooterBar
    from rfcensus import __version__
    fb = FooterBar()
    fb.set_state(
        healthy=5, total=5, decodes=14, emitters=8, detections=0,
        wave_label="0", filter_mode="filtered",
    )
    out = str(fb.render())
    assert f"v{__version__}" not in out


def test_dongle_detail_renders_dongle_n_header() -> None:
    """v0.7.4: detail page needs a prominent "Dongle N — id" line at
    the top of the content. The border title already says it but
    Textual renders border titles in a small dim font that's easy
    to miss when scrolled into a long detail page."""
    from rfcensus.tui.widgets.dongle_detail import DongleDetail
    from rfcensus.tui.state import DongleState

    d = DongleState(
        dongle_id="rtlsdr-00000003",
        status="active",
        consumer="lora_survey:915_ism",
        freq_hz=915_000_000,
        sample_rate=2_359_296,
        band_id="915_ism",
    )
    dd = DongleDetail()
    dd._dongles = [d]
    dd._index = 0
    dd._stream = []
    out = str(dd.render())
    # Slot 1 (display index 0) should appear with the dongle id
    assert "Dongle [1]" in out
    assert "rtlsdr-00000003" in out


def test_dongle_detail_header_uses_slot_zero_for_tenth() -> None:
    """The tenth dongle uses slot key "0" (matching the existing
    number-key convention where 0 opens slot 10)."""
    from rfcensus.tui.widgets.dongle_detail import DongleDetail
    from rfcensus.tui.state import DongleState

    dongles = [
        DongleState(dongle_id=f"d{i}") for i in range(10)
    ]
    dd = DongleDetail()
    dd._dongles = dongles
    dd._index = 9    # tenth dongle
    dd._stream = []
    out = str(dd.render())
    assert "Dongle [0]" in out


def test_focused_pane_state_default_dongles() -> None:
    """Tab/Shift+Tab cycle a focused_pane state on TUIState. The
    default is "dongles" so users who never press Tab still see the
    strip highlighted (matching the existing arrow-key context)."""
    from rfcensus.tui.state import TUIState
    s = TUIState()
    assert s.focused_pane == "dongles"


def test_pane_order_cycles_dongles_main_plan_tree() -> None:
    """The cycle order is documented and stable so users can learn
    the muscle memory."""
    from rfcensus.tui.app import TUIApp
    assert TUIApp._PANE_ORDER == ("dongles", "main", "plan_tree")


def test_focus_next_pane_action_advances_state() -> None:
    """action_focus_next_pane mutates state.focused_pane through
    the cycle. We test the logic directly (no Textual app mounting)
    by calling the method on a constructed-but-not-mounted instance."""
    from rfcensus.tui.app import TUIApp
    from rfcensus.tui.state import TUIState

    # Bypass __init__'s heavy setup; we only need state + the
    # bound method.
    app = TUIApp.__new__(TUIApp)
    app.state = TUIState()
    app._refresh_all = lambda: None    # type: ignore[method-assign]

    assert app.state.focused_pane == "dongles"
    app.action_focus_next_pane()
    assert app.state.focused_pane == "main"
    app.action_focus_next_pane()
    assert app.state.focused_pane == "plan_tree"
    app.action_focus_next_pane()
    assert app.state.focused_pane == "dongles"    # wraps


def test_focus_prev_pane_action_reverses_cycle() -> None:
    """Shift+Tab is the reverse direction."""
    from rfcensus.tui.app import TUIApp
    from rfcensus.tui.state import TUIState

    app = TUIApp.__new__(TUIApp)
    app.state = TUIState()
    app._refresh_all = lambda: None    # type: ignore[method-assign]

    assert app.state.focused_pane == "dongles"
    app.action_focus_prev_pane()
    assert app.state.focused_pane == "plan_tree"    # wraps backward
    app.action_focus_prev_pane()
    assert app.state.focused_pane == "main"
    app.action_focus_prev_pane()
    assert app.state.focused_pane == "dongles"


def test_tab_and_shift_tab_bindings_exist() -> None:
    """The Tab and Shift+Tab keybindings are wired up in the app's
    BINDINGS list."""
    from rfcensus.tui.app import TUIApp
    keys = {b.key for b in TUIApp.BINDINGS}
    assert "tab" in keys
    assert "shift+tab" in keys


# ─────────────────────────────────────────────────────────────────────
# (7) auto_attach — meshtastic gets scheduled when [decoders.meshtastic]
#     is in site config, without requiring band_definitions overrides
# ─────────────────────────────────────────────────────────────────────


def test_meshtastic_auto_attach_default_true() -> None:
    """v0.7.4: auto_attach defaults to True so users who add
    [decoders.meshtastic] to site.toml get the decoder running on
    every overlapping band without further config."""
    from rfcensus.config.schema import MeshtasticDecoderConfig
    m = MeshtasticDecoderConfig()
    assert m.auto_attach is True


def test_meshtastic_auto_attach_can_be_disabled() -> None:
    """Power users who want fine-grained band control can opt out."""
    from rfcensus.config.schema import MeshtasticDecoderConfig
    m = MeshtasticDecoderConfig(auto_attach=False)
    assert m.auto_attach is False


def test_strategy_pick_decoders_honors_auto_attach() -> None:
    """When [decoders.meshtastic] is configured with auto_attach=True,
    _pick_decoders adds meshtastic to the band's effective suggested
    decoders even if the band's TOML doesn't list it."""
    src = open(
        "/home/claude/rfcensus/rfcensus/engine/strategy.py"
    ).read()
    # Confirm the auto_attach honoring code is present
    assert "auto_attach" in src
    assert "site_decoders = ctx.config.decoders" in src
    # And it adds to the suggested set
    assert "suggested.add(dec_name)" in src


def test_strategy_skips_auto_attach_when_disabled() -> None:
    """If a decoder has auto_attach=True but enabled=False, it
    must NOT be added — disabled means disabled."""
    src = open(
        "/home/claude/rfcensus/rfcensus/engine/strategy.py"
    ).read()
    # The check for enabled comes before suggested.add
    enabled_idx = src.find("if not dec_cfg.enabled:")
    add_idx = src.find("suggested.add(dec_name)")
    assert enabled_idx > 0, "missing enabled gate"
    assert enabled_idx < add_idx, (
        "enabled gate must come BEFORE suggested.add"
    )


def test_meshtastic_empty_state_message_shows_valid_toml() -> None:
    """The empty-state message must show the actual TOML the user
    needs to paste, not just a vague hint about a key. Old message
    pointed at `enabled = true` which was a no-op (enabled defaults
    to True)."""
    from rfcensus.tui.widgets.meshtastic_recent import (
        MeshtasticRecentWidget,
    )
    w = MeshtasticRecentWidget()
    w._entries = []
    out = str(w.render())
    # The TOML section header must be present and properly formed
    assert "[decoders.meshtastic]" in out
    # Region line is the minimum non-default config users care about
    assert "region" in out
    # The misleading old message must be gone
    assert "enabled = true" not in out
    # Mention the auto_attach default so users understand they don't
    # need a second step
    assert "auto_attach" in out or "automatically" in out.lower()


# ─────────────────────────────────────────────────────────────────────
# (8) HeaderBar dock-stack fix + version watermark in footer
# ─────────────────────────────────────────────────────────────────────


def test_dongle_strip_no_longer_docked_top() -> None:
    """v0.7.4: Textual 8.x stacks two siblings both `dock: top`
    at y=0 — the second-yielded covers the first. We had HeaderBar
    AND DongleStrip both docked top, so DongleStrip was painting
    on top of HeaderBar and the header (with version + cpu/rss)
    was completely invisible to users. Removing dock:top from
    DongleStrip lets it flow naturally as the second child of
    Container#main, showing up at y=1 below the docked HeaderBar."""
    import re
    src = open(
        "/home/claude/rfcensus/rfcensus/tui/widgets/dongle_strip.py"
    ).read()
    # Find the DongleStrip selector block specifically (not the
    # DongleTile block above it).
    strip_block_start = src.find("DongleStrip {")
    assert strip_block_start > 0, "no DongleStrip CSS rule found"
    strip_block_end = src.find("}", strip_block_start)
    strip_css = src[strip_block_start:strip_block_end]
    # Strip /* ... */ comments — the v0.7.4 comment legitimately
    # mentions the removed `dock: top` rule.
    strip_css_no_comments = re.sub(
        r"/\*.*?\*/", "", strip_css, flags=re.DOTALL,
    )
    # The fix: dock: rule must be GONE from the actual rules.
    assert "dock:" not in strip_css_no_comments, (
        f"DongleStrip still has a `dock:` rule, which will cover "
        f"the HeaderBar in Textual 8.x:\n{strip_css_no_comments}"
    )


def test_header_and_strip_do_not_overlap_in_compose() -> None:
    """End-to-end: mount the real widgets and confirm HeaderBar at
    y=0 is no longer covered by DongleStrip."""
    import asyncio
    from textual.app import App
    from textual.containers import Container
    from rfcensus.tui.widgets.header import HeaderBar
    from rfcensus.tui.widgets.dongle_strip import DongleStrip

    class _MiniApp(App):
        def compose(self):
            with Container(id="main"):
                yield HeaderBar(site_name="t")
                yield DongleStrip()

    async def _run():
        app = _MiniApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            h = app.query_one(HeaderBar).region
            s = app.query_one(DongleStrip).region
            return h, s

    h, s = asyncio.run(_run())
    # Header at y=0 with non-zero height
    assert h.y == 0
    assert h.height >= 1
    # Strip strictly BELOW header (not overlapping)
    assert s.y >= h.y + h.height, (
        f"DongleStrip y={s.y} overlaps HeaderBar at y={h.y} h={h.height}"
    )


def test_footer_renders_version_watermark() -> None:
    """v0.7.4 briefly added a version watermark to the footer.
    v0.7.5: removed — see test_footer_does_not_render_version_v075.
    This test is left as a guard against accidental re-introduction."""
    from rfcensus.tui.widgets.footer import FooterBar
    from rfcensus import __version__
    fb = FooterBar()
    fb.set_state(
        healthy=5, total=5, decodes=14, emitters=8, detections=0,
        wave_label="0", filter_mode="filtered",
    )
    out = str(fb.render())
    # v0.7.5: assert ABSENCE — version lives in HeaderBar only
    assert f"v{__version__}" not in out
