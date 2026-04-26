"""v0.7.3 — TUI binding adjustments.

Two changes shipped together:
  1. The on-demand report key moved from ``s`` to ``r`` so the
     visible footer label ("report") matches what users press. ``s``
     remains as a hidden alias so muscle memory still works.
  2. The report header now explicitly marks "IN-FLIGHT REPORT" when
     the session is still running or paused, so users don't confuse
     a Ctrl-C'd snapshot with the final session-end report.

These tests guard against accidentally re-binding ``r`` to something
else later (it would silently break a documented user workflow).
"""

from __future__ import annotations

from datetime import datetime, timezone


def test_r_is_primary_report_key() -> None:
    from rfcensus.tui.app import TUIApp
    bindings = [b for b in TUIApp.BINDINGS if b.action == "snapshot"]
    # Both bindings exist
    keys = {b.key for b in bindings}
    assert "r" in keys
    assert "s" in keys

    # `r` is the visible one (show=True is the default)
    r_binding = next(b for b in bindings if b.key == "r")
    assert r_binding.show is True
    assert r_binding.description == "report"

    # `s` is the hidden alias (kept for muscle memory)
    s_binding = next(b for b in bindings if b.key == "s")
    assert s_binding.show is False


def test_no_other_binding_uses_r_as_primary() -> None:
    """Sanity: no other shown binding uses ``r`` (would conflict with
    the report key)."""
    from rfcensus.tui.app import TUIApp
    r_owners = [
        b for b in TUIApp.BINDINGS if b.key == "r" and b.show is not False
    ]
    assert len(r_owners) == 1
    assert r_owners[0].action == "snapshot"


def test_in_flight_report_marks_partial_when_running() -> None:
    """The report header should explicitly say "IN-FLIGHT REPORT"
    when the session is still running, so users know counters and
    band lists are partial."""
    from rfcensus.tui.app import TUIApp
    from rfcensus.tui.state import TUIState

    app = TUIApp.__new__(TUIApp)    # bypass __init__ (avoids Textual app setup)
    app.state = TUIState()
    app.state.session_id = 42
    app.state.site_name = "test-site"
    app.state.session_status = "running"
    app.state.session_started_at = datetime.now(timezone.utc)

    text = app._render_snapshot_report()
    assert "IN-FLIGHT REPORT" in text
    assert "still running" in text
    assert "completed so far" in text


def test_in_flight_report_marks_partial_when_paused() -> None:
    from rfcensus.tui.app import TUIApp
    from rfcensus.tui.state import TUIState

    app = TUIApp.__new__(TUIApp)
    app.state = TUIState()
    app.state.session_id = 42
    app.state.site_name = "test-site"
    app.state.session_status = "paused"
    app.state.session_started_at = datetime.now(timezone.utc)

    text = app._render_snapshot_report()
    assert "IN-FLIGHT REPORT" in text
    assert "still paused" in text


def test_post_session_report_does_not_mark_in_flight() -> None:
    """When the session has ended, the report shouldn't carry the
    in-flight marker — the data is final at that point."""
    from rfcensus.tui.app import TUIApp
    from rfcensus.tui.state import TUIState

    app = TUIApp.__new__(TUIApp)
    app.state = TUIState()
    app.state.session_id = 42
    app.state.site_name = "test-site"
    app.state.session_status = "ended"
    app.state.session_started_at = datetime.now(timezone.utc)

    text = app._render_snapshot_report()
    assert "IN-FLIGHT REPORT" not in text
    # Still shows session id
    assert "42" in text


# ─────────────────────────────────────────────────────────────────────
# v0.7.3 — detail-mode focus follows arrow keys
# ─────────────────────────────────────────────────────────────────────
#
# Previously, after opening detail on dongle 2, pressing → would only
# move the cursor highlight to dongle 3; you then had to press Enter
# to actually update the detail pane. The new behavior: when detail is
# open, arrow keys immediately switch the detail pane to the new
# dongle. Enter is purely the toggle (open ↔ close).


def _make_app_with_dongles(n: int):
    """Build a TUIApp with N stub dongles, bypassing Textual __init__."""
    from rfcensus.tui.app import TUIApp
    from rfcensus.tui.state import TUIState, DongleState

    app = TUIApp.__new__(TUIApp)
    app.state = TUIState()
    app.state.dongles = [
        DongleState(dongle_id=f"d{i}", model="rtlsdr_v3")
        for i in range(n)
    ]
    # Stub the refresh hook so the actions don't try to talk to Textual
    app._refresh_all = lambda: None    # type: ignore[method-assign]
    return app


def test_detail_pane_follows_right_arrow_when_open() -> None:
    """Open detail on dongle 1, press →, detail should now show dongle 2."""
    app = _make_app_with_dongles(5)
    app.state.focused_dongle_index = 1
    app.action_open_detail()    # opens detail on dongle 1
    assert app.state.main_pane_mode == "dongle"
    assert app.state.detail_dongle_index == 1

    app.action_focus_next()
    assert app.state.focused_dongle_index == 2
    assert app.state.detail_dongle_index == 2, (
        "detail pane should follow focus arrows while open — was the "
        "v0.7.3 fix lost?"
    )
    assert app.state.main_pane_mode == "dongle"   # stays in detail mode


def test_detail_pane_follows_left_arrow_when_open() -> None:
    app = _make_app_with_dongles(5)
    app.state.focused_dongle_index = 3
    app.action_open_detail()
    assert app.state.detail_dongle_index == 3

    app.action_focus_prev()
    assert app.state.focused_dongle_index == 2
    assert app.state.detail_dongle_index == 2


def test_arrows_dont_open_detail_from_events_mode() -> None:
    """Arrow keys in events mode just move the cursor — they should
    NOT auto-open detail. The detail pane only opens via Enter."""
    app = _make_app_with_dongles(5)
    app.state.focused_dongle_index = 0
    assert app.state.main_pane_mode == "events"

    app.action_focus_next()
    assert app.state.focused_dongle_index == 1
    assert app.state.main_pane_mode == "events", (
        "arrows must not auto-open detail from events mode"
    )
    assert app.state.detail_dongle_index is None


def test_enter_toggles_detail_off_regardless_of_focus_position() -> None:
    """v0.7.3 simplification: Enter is now purely the open/close
    toggle. Used to have a "switch to focused" middle case that's
    no longer reachable since arrows sync the detail index
    automatically."""
    app = _make_app_with_dongles(5)
    app.state.focused_dongle_index = 2
    app.action_open_detail()       # open at 2
    assert app.state.main_pane_mode == "dongle"

    # Arrow to dongle 3 — detail follows
    app.action_focus_next()
    assert app.state.detail_dongle_index == 3

    # Press Enter — should CLOSE, not switch
    app.action_open_detail()
    assert app.state.main_pane_mode == "events"


def test_left_arrow_at_index_zero_does_not_underflow() -> None:
    app = _make_app_with_dongles(3)
    app.state.focused_dongle_index = 0
    app.action_open_detail()
    assert app.state.detail_dongle_index == 0

    app.action_focus_prev()    # at edge — no-op
    assert app.state.focused_dongle_index == 0
    assert app.state.detail_dongle_index == 0


def test_right_arrow_at_last_index_does_not_overflow() -> None:
    app = _make_app_with_dongles(3)
    app.state.focused_dongle_index = 2
    app.action_open_detail()
    assert app.state.detail_dongle_index == 2

    app.action_focus_next()    # at edge — no-op
    assert app.state.focused_dongle_index == 2
    assert app.state.detail_dongle_index == 2
