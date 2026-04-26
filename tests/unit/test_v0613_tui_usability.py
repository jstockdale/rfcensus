"""v0.6.13 — TUI usability pass.

Four user-facing changes, four test groups:
  1. `channel` events removed from 'filtered' mode (still in verbose).
     Active-channel chatter from rtl_power was drowning the actually-
     decoded emitters that 'filtered' is meant to surface.
  2. Startup banner emits the running version + launch mode (tui vs
     log) as the first log line. Without this, support tickets
     consistently start with "what version do you have?".
  3. TUI header bar shows the version on the left next to "rfcensus"
     (so the running build is always visible on screen).
  4. EventStream supports scroll: up/down by line, PgUp/PgDn by page,
     Home snaps to live tail. Critically, when new events arrive
     while the user is scrolled into history, the viewport stays
     anchored on the SAME logical entry — no yank-to-top mid-read.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────
# 1. Filter-category dedupe
# ─────────────────────────────────────────────────────────────────────


class TestChannelMovedToVerboseOnly:
    """`channel` should appear in verbose but NOT in filtered. The
    user's screenshot showed dozens of 'active channel at X MHz' lines
    crowding out everything else in filtered mode."""

    def test_filtered_excludes_channel(self):
        from rfcensus.tui.state import FILTER_CATEGORIES
        assert "channel" not in FILTER_CATEGORIES["filtered"], (
            "filtered mode must NOT include 'channel' — those events "
            "are too chatty (5-30/sec on a busy band) and were the "
            "reason a user couldn't see anything else in their TUI."
        )

    def test_verbose_still_includes_channel(self):
        from rfcensus.tui.state import FILTER_CATEGORIES
        assert "channel" in FILTER_CATEGORIES["verbose"], (
            "verbose mode must still surface channel events — that's "
            "where users go for the spectrum-occupancy view."
        )

    def test_minimal_unchanged(self):
        from rfcensus.tui.state import FILTER_CATEGORIES
        # v0.6.14: minimal now also includes 'wave' so the user
        # always sees wave-transition heartbeats even in the most
        # restrictive filter mode. (Before, minimal showed nothing
        # during multi-minute waves and looked broken.)
        assert FILTER_CATEGORIES["minimal"] == {
            "session", "hardware", "wave",
        }

    def test_filtered_still_includes_emitter_and_detection(self):
        """Don't accidentally drop the high-value categories while
        removing the noisy one."""
        from rfcensus.tui.state import FILTER_CATEGORIES
        assert "emitter" in FILTER_CATEGORIES["filtered"]
        assert "detection" in FILTER_CATEGORIES["filtered"]

    def test_filter_stream_drops_channel_events_in_filtered(self):
        """End-to-end: feed a stream containing channel events through
        filter_stream(mode='filtered') — they should be excluded."""
        from rfcensus.tui.state import StreamEntry, filter_stream
        ts = datetime.now(timezone.utc)
        stream = [
            StreamEntry(timestamp=ts, severity="info",
                        category="emitter", text="emitter A", raw=None),
            StreamEntry(timestamp=ts, severity="info",
                        category="channel", text="active channel at X MHz",
                        raw=None),
            StreamEntry(timestamp=ts, severity="info",
                        category="detection", text="detection B", raw=None),
        ]
        out = filter_stream(stream, "filtered")
        texts = [e.text for e in out]
        assert "emitter A" in texts
        assert "detection B" in texts
        assert "active channel at X MHz" not in texts, (
            "filter_stream must drop channel-category entries when "
            "mode='filtered'"
        )

    def test_filter_stream_keeps_channel_events_in_verbose(self):
        from rfcensus.tui.state import StreamEntry, filter_stream
        ts = datetime.now(timezone.utc)
        stream = [
            StreamEntry(timestamp=ts, severity="info",
                        category="channel", text="active channel at X MHz",
                        raw=None),
        ]
        out = filter_stream(stream, "verbose")
        assert len(out) == 1
        assert out[0].text == "active channel at X MHz"


# ─────────────────────────────────────────────────────────────────────
# 2. Startup banner emits version + launch mode
# ─────────────────────────────────────────────────────────────────────


class TestStartupBannerEmitsVersionAndMode:
    """First log line of any scan/inventory/hybrid run must include
    the running version AND whether we're launching tui vs log."""

    @pytest.mark.asyncio
    async def test_log_mode_banner(self, monkeypatch, caplog):
        """When tui=False, banner says 'launching <command> in log mode'."""
        # Make the function fail quickly past the banner — we don't
        # want to actually run a full scan.
        from rfcensus.commands import inventory

        async def fast_fail(*args, **kwargs):
            raise SystemExit(0)

        # Patch click.BadParameter is fine, but we need _run to bail
        # before hardware. The simplest hook: replace find_sdr_orphans
        # with a side-effect that raises. Banner runs before that.
        from rfcensus.utils import orphan_detect

        def boom(*a, **k):
            raise RuntimeError("expected: stop after banner")

        monkeypatch.setattr(orphan_detect, "find_sdr_orphans", boom)

        with caplog.at_level(logging.INFO, logger="rfcensus"):
            with pytest.raises(RuntimeError, match="stop after banner"):
                await inventory._run(
                    config_path=None, duration="1s", band_filter=None,
                    capture_power=False, include_ids=False, output=None,
                    as_json=False, all_bands=False, per_band=None,
                    gain="auto", until_quiet=None, guided=False,
                    kill_orphans=False, command_name="scan",
                    pin_strings=[], allow_pin_antenna_mismatch=False,
                    honor_pins=False, tui=False, no_color=True,
                )

        # Verify the banner appears with version + log mode
        from rfcensus import __version__
        msgs = [r.getMessage() for r in caplog.records]
        banner_lines = [
            m for m in msgs
            if m.startswith(f"rfcensus {__version__}")
        ]
        assert banner_lines, (
            f"no startup banner found in log records. "
            f"got: {msgs[:5]}"
        )
        assert "log mode" in banner_lines[0], (
            f"banner should say 'log mode' when tui=False; got: "
            f"{banner_lines[0]}"
        )

    @pytest.mark.asyncio
    async def test_tui_mode_banner(self, monkeypatch, caplog):
        """When tui=True, banner says 'launching <command> in tui mode'."""
        from rfcensus.commands import inventory
        from rfcensus.utils import orphan_detect

        def boom(*a, **k):
            raise RuntimeError("expected: stop after banner")

        monkeypatch.setattr(orphan_detect, "find_sdr_orphans", boom)

        with caplog.at_level(logging.INFO, logger="rfcensus"):
            with pytest.raises(RuntimeError, match="stop after banner"):
                await inventory._run(
                    config_path=None, duration="1s", band_filter=None,
                    capture_power=False, include_ids=False, output=None,
                    as_json=False, all_bands=False, per_band=None,
                    gain="auto", until_quiet=None, guided=False,
                    kill_orphans=False, command_name="scan",
                    pin_strings=[], allow_pin_antenna_mismatch=False,
                    honor_pins=False, tui=True, no_color=True,
                )

        from rfcensus import __version__
        msgs = [r.getMessage() for r in caplog.records]
        banner_lines = [
            m for m in msgs if m.startswith(f"rfcensus {__version__}")
        ]
        assert banner_lines and "tui mode" in banner_lines[0], (
            f"expected banner with 'tui mode'; got: {banner_lines}"
        )


# ─────────────────────────────────────────────────────────────────────
# 3. Header bar shows version
# ─────────────────────────────────────────────────────────────────────


class TestHeaderShowsVersion:
    """The TUI header bar must include the running version on the left
    so users can confirm at a glance which build they're running."""

    def _make_header(self, **kwargs):
        # Subclass to override the read-only `size` property that
        # Textual installs on Widget. We just need width for layout.
        from rfcensus.tui.widgets.header import HeaderBar

        class _SizedHeader(HeaderBar):
            @property
            def size(self):  # type: ignore[override]
                return MagicMock(width=120, height=1)

        return _SizedHeader(**kwargs)

    def test_header_render_includes_version(self):
        from rfcensus import __version__

        h = self._make_header(
            site_name="testsite", session_id=42, paused=False,
        )
        h.elapsed_s = 65.0
        rendered = h.render()
        assert f"v{__version__}" in rendered, (
            f"header should include 'v{__version__}'; got: {rendered}"
        )
        assert "testsite" in rendered
        assert "#42" in rendered

    def test_header_version_left_status_right(self):
        """Version goes on the left (with program identification);
        status indicator stays on the right."""
        from rfcensus import __version__

        h = self._make_header(site_name="x", session_id=1, paused=False)
        h.elapsed_s = 0
        rendered = h.render()
        v_pos = rendered.find(f"v{__version__}")
        running_pos = rendered.find("running")
        assert v_pos < running_pos, (
            "version must appear LEFT of the running indicator"
        )


class TestHelpOverlayShowsCurrentVersion:
    """Help overlay used to hardcode 'rfcensus 0.6.5'. v0.6.13 makes
    it dynamic so we never ship stale version info there again."""

    def test_help_text_uses_current_version(self):
        from rfcensus.tui.widgets.modals import _help_text
        from rfcensus import __version__
        text = _help_text()
        assert f"rfcensus {__version__}" in text, (
            f"help overlay must include current version; got: ...{text[-100:]}"
        )

    def test_help_text_documents_scroll_keys(self):
        from rfcensus.tui.widgets.modals import _help_text
        text = _help_text()
        # New keys discoverable from the help overlay
        assert "PgUp" in text or "PgDn" in text, (
            "help overlay must document the new scroll keys"
        )
        assert "Home" in text


# ─────────────────────────────────────────────────────────────────────
# 4. EventStream scroll behavior
# ─────────────────────────────────────────────────────────────────────


def _make_stream(n: int, category: str = "emitter"):
    """Build a list of n StreamEntries oldest-first... no wait,
    NEWEST first (the reducer prepends). We mimic that."""
    from rfcensus.tui.state import StreamEntry
    ts = datetime.now(timezone.utc)
    return [
        StreamEntry(
            timestamp=ts, severity="info", category=category,
            text=f"event #{i}", raw=None,
        )
        # i=0 is the NEWEST → at the front of the list.
        for i in range(n)
    ]


def _make_event_stream(height: int = 10, width: int = 80):
    """Build an EventStream with a faked size (Textual's Widget.size
    is a read-only property, so we subclass to override it for tests
    that need to drive layout math without spinning up Textual)."""
    from rfcensus.tui.widgets.event_stream import EventStream

    captured_h, captured_w = height, width

    class _SizedEventStream(EventStream):
        @property
        def size(self):  # type: ignore[override]
            return MagicMock(width=captured_w, height=captured_h)

    return _SizedEventStream()


class TestEventStreamScrollMechanics:
    """Verify the offset math without spinning up Textual."""

    def test_default_offset_is_zero(self):
        w = _make_event_stream()
        assert w._scroll_offset == 0
        assert w._unread_below == 0

    def test_scroll_lines_increases_offset_toward_older(self):
        """Up arrow ⇒ move toward older entries ⇒ offset increments."""
        w = _make_event_stream()
        w.update_stream(_make_stream(50), "filtered")
        w.scroll_lines(3)
        assert w._scroll_offset == 3

    def test_scroll_lines_clamps_at_zero(self):
        """Trying to scroll past live tail toward newer is a no-op."""
        w = _make_event_stream()
        w.update_stream(_make_stream(50), "filtered")
        w.scroll_lines(-5)
        assert w._scroll_offset == 0

    def test_scroll_lines_clamps_at_oldest(self):
        """v0.7.4: scrolling beyond oldest caps at (len - max_lines)
        so the viewport stays FULL when at the top of the buffer.
        Pre-v0.7.4 the cap was len-1, which let the viewport collapse
        to a single line as offset → len-1."""
        w = _make_event_stream()
        w.update_stream(_make_stream(20), "filtered")
        w.scroll_lines(1000)
        # _make_event_stream defaults to height=10 → max_visible 8
        # (after the 2-row border subtraction). 20 - 8 = 12.
        assert w._scroll_offset == 12

    def test_scroll_page_uses_widget_height(self):
        """PgDn = one visible page worth of lines."""
        w = _make_event_stream(height=12)
        w.update_stream(_make_stream(100), "filtered")
        w.scroll_page(+1)
        # height 12 - 2 (border padding) = 10 lines per page
        assert w._scroll_offset == 10

    def test_scroll_to_top_resets(self):
        """Home: snap to live tail, clear unread counter."""
        w = _make_event_stream()
        w.update_stream(_make_stream(30), "filtered")
        w.scroll_lines(5)
        w._unread_below = 7  # simulate
        w.scroll_to_top()
        assert w._scroll_offset == 0
        assert w._unread_below == 0


class TestEventStreamAnchoringOnNewEntries:
    """The CRITICAL behavior: when new events arrive while user is
    scrolled into history, their viewport must stay on the same
    LOGICAL entry — don't yank back to live."""

    def test_live_tail_stays_at_zero_on_new_entries(self):
        """When at offset=0 (live), new entries appear at top —
        offset stays 0, no unread counter."""
        w = _make_event_stream()
        w.update_stream(_make_stream(10), "filtered")
        # User is at live tail — no scroll
        assert w._scroll_offset == 0
        # New events arrive
        w.update_stream(_make_stream(15), "filtered")
        assert w._scroll_offset == 0
        assert w._unread_below == 0

    def test_history_view_anchors_on_same_entry(self):
        """User is reading entry at offset=10. 5 new events arrive.
        Offset must shift to 15 so the same logical entry is still
        at the top of the visible window."""
        w = _make_event_stream()
        w.update_stream(_make_stream(50), "filtered")
        w.scroll_lines(10)
        assert w._scroll_offset == 10

        # Imagine 5 new events were prepended. We simulate by passing
        # a stream that's 5 longer.
        w.update_stream(_make_stream(55), "filtered")
        assert w._scroll_offset == 15, (
            "offset must shift by the number of new entries to keep "
            "the user's viewport on the same logical entry"
        )
        assert w._unread_below == 5, (
            "unread counter should track entries added while user "
            "was scrolled away"
        )

    def test_unread_accumulates_across_updates(self):
        """Multiple updates while scrolled should sum into unread."""
        w = _make_event_stream()
        w.update_stream(_make_stream(20), "filtered")
        w.scroll_lines(5)
        w.update_stream(_make_stream(23), "filtered")  # +3
        w.update_stream(_make_stream(27), "filtered")  # +4
        assert w._unread_below == 7

    def test_returning_to_live_clears_unread(self):
        w = _make_event_stream()
        w.update_stream(_make_stream(20), "filtered")
        w.scroll_lines(5)
        w.update_stream(_make_stream(30), "filtered")
        assert w._unread_below == 10
        w.scroll_to_top()
        assert w._unread_below == 0
        assert w._scroll_offset == 0

    def test_filter_mode_change_resets_scroll(self):
        """Changing filter mode invalidates the scroll position
        (different set of entries entirely) — snap to live."""
        w = _make_event_stream()
        w.update_stream(_make_stream(30), "filtered")
        w.scroll_lines(8)
        assert w._scroll_offset == 8
        # Mode change
        w.update_stream(_make_stream(30), "verbose")
        assert w._scroll_offset == 0
        assert w._unread_below == 0

    def test_render_slices_at_offset(self):
        """The render output at offset>0 should start from the entry
        at index=offset, not index=0."""
        w = _make_event_stream(height=5)
        w.update_stream(_make_stream(20), "filtered")
        w.scroll_lines(7)
        rendered = w.render()
        # Page size = height - 2 = 3 visible lines
        # Should show events #7, #8, #9
        assert "event #7" in rendered
        assert "event #8" in rendered
        # Should NOT show event #0 (that's the live tail)
        assert "event #0" not in rendered

    def test_title_indicates_history_mode(self):
        """Border title should change to indicate the user is
        scrolled away from live."""
        w = _make_event_stream()
        w.update_stream(_make_stream(20), "filtered")
        # At live tail
        assert "scrolled" not in w.border_title
        w.scroll_lines(5)
        assert "scrolled" in w.border_title
        # Returning to live clears it
        w.scroll_to_top()
        assert "scrolled" not in w.border_title


# ─────────────────────────────────────────────────────────────────────
# Bindings registered on the App
# ─────────────────────────────────────────────────────────────────────


class TestScrollBindingsRegistered:
    """Make sure the new key bindings actually exist on the App
    class. Catches refactor regressions where someone removes a
    Binding but leaves the action method."""

    def test_scroll_bindings_present(self):
        from rfcensus.tui.app import TUIApp
        binding_keys = [b.key for b in TUIApp.BINDINGS]
        assert "up" in binding_keys
        assert "down" in binding_keys
        assert "pageup" in binding_keys
        assert "pagedown" in binding_keys
        assert "home" in binding_keys

    def test_scroll_actions_present(self):
        from rfcensus.tui.app import TUIApp
        for action in (
            "action_scroll_up", "action_scroll_down",
            "action_scroll_page_up", "action_scroll_page_down",
            "action_scroll_to_live",
        ):
            assert hasattr(TUIApp, action), f"missing action: {action}"
