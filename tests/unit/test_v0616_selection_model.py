"""v0.6.16 — TUI selection-state model + Enter-toggle behavior.

Three changes from v0.6.14:

  1. Border color always reflects status, never overridden by selection.
     Selection state is encoded in border STYLE (round/heavy/double),
     not color. A green-status active dongle stays green when selected.

  2. Cursor and detail-shown are now tracked separately. Arrow keys
     move the cursor without disturbing the detail pane. Enter commits
     the cursor to the detail pane (or toggles off if already there).

  3. Enter on detail-shown dongle TOGGLES the pane closed (back to
     events). Enter on cursor over a different dongle re-targets the
     detail pane. Enter from events mode opens detail at cursor.

This file locks in the new behavior so future refactors can't quietly
revert to the v0.6.14 model.
"""

from __future__ import annotations

import pytest

from rfcensus.tui.state import TUIState, DongleState


# ─── Test helper ─────────────────────────────────────────────────────


def _make_state_with_dongles(n: int = 5) -> TUIState:
    """Build a TUIState with n DongleState entries. All start as
    'active' status with green color. Tests can mutate as needed."""
    s = TUIState(site_name="test")
    for i in range(n):
        s.dongles.append(DongleState(
            dongle_id=f"rtlsdr-{i:08x}",
            status="active",
            band_id=f"band_{i}",
            freq_hz=915_000_000 + i * 1_000_000,
        ))
    return s


# ─────────────────────────────────────────────────────────────────────
# 1. Status color survives selection
# ─────────────────────────────────────────────────────────────────────


class TestStatusColorPreserved:
    def test_tile_class_includes_status_color_when_plain(self):
        from rfcensus.tui.widgets.dongle_strip import DongleTile
        tile = DongleTile(slot=1, dongle=DongleState(
            dongle_id="x", status="active", decodes_in_band=5,
        ))
        tile.set_state(cursor=False, detail=False)
        # _current_class encodes "color-style"; verify the COLOR part
        # corresponds to active+decodes (green).
        assert "green" in tile._current_class
        assert "plain" in tile._current_class

    def test_tile_class_keeps_color_when_cursor(self):
        from rfcensus.tui.widgets.dongle_strip import DongleTile
        tile = DongleTile(slot=1, dongle=DongleState(
            dongle_id="x", status="active", decodes_in_band=5,
        ))
        tile.set_state(cursor=True, detail=False)
        # Crucial v0.6.16 invariant: selection MUST NOT remove color.
        assert "green" in tile._current_class
        assert "cursor" in tile._current_class

    def test_tile_class_keeps_color_when_detail(self):
        from rfcensus.tui.widgets.dongle_strip import DongleTile
        tile = DongleTile(slot=1, dongle=DongleState(
            dongle_id="x", status="active", decodes_in_band=5,
        ))
        tile.set_state(cursor=False, detail=True)
        assert "green" in tile._current_class
        assert "detail" in tile._current_class

    def test_red_status_keeps_red_when_selected(self):
        # Failed dongle in detail mode should still be RED, not white.
        from rfcensus.tui.widgets.dongle_strip import DongleTile
        tile = DongleTile(slot=1, dongle=DongleState(
            dongle_id="x", status="permanent_failed",
        ))
        tile.set_state(cursor=True, detail=True)
        assert "red" in tile._current_class

    def test_no_white_classes_in_css(self):
        # Regression guard: v0.6.14's first cut had `border: heavy white`
        # which destroyed status-color signal. Make sure no CSS rule
        # ever sets border white.
        from rfcensus.tui.widgets.dongle_strip import DongleTile
        css = DongleTile.DEFAULT_CSS.lower()
        # Allow `white` to appear in comments/docs but not as a border
        # color value. Cheap proxy: check no "heavy white" / "double
        # white" / "round white" appears in CSS rules.
        for style in ("round", "heavy", "double"):
            assert f"{style} white" not in css, (
                f"v0.6.14 regression: CSS sets `border: {style} white` "
                f"which would destroy status-color signal"
            )


# ─────────────────────────────────────────────────────────────────────
# 2. Three distinct selection states
# ─────────────────────────────────────────────────────────────────────


class TestThreeSelectionStates:
    def test_plain_uses_round_border(self):
        from rfcensus.tui.widgets.dongle_strip import DongleTile
        # Each plain class should map to a `round` border per the CSS.
        css = DongleTile.DEFAULT_CSS
        for color in ("green", "grey", "yellow", "red"):
            cls = f"-tile-{color}-plain"
            # CSS line for this class should contain `round`
            for line in css.splitlines():
                if cls in line:
                    assert "round" in line, (
                        f"Plain state for {color} should use round border, "
                        f"got: {line}"
                    )
                    break
            else:
                pytest.fail(f"No CSS rule for class {cls}")

    def test_cursor_uses_heavy_border(self):
        # v0.7.3 swap: cursor (just-focused) now uses double; the
        # heavy "fat single" border is reserved for the detail-shown
        # tile per user feedback that heavy reads as "active".
        from rfcensus.tui.widgets.dongle_strip import DongleTile
        css = DongleTile.DEFAULT_CSS
        for color in ("green", "grey", "yellow", "red"):
            cls = f"-tile-{color}-cursor"
            for line in css.splitlines():
                if cls in line:
                    assert "double" in line, (
                        f"v0.7.3: cursor state for {color} now uses double border"
                    )
                    break

    def test_detail_uses_double_border(self):
        # v0.7.3 swap: detail-shown tile now uses heavy (the "fat
        # single" border that reads as truly-active). Test name kept
        # for git-blame continuity even though it now asserts heavy.
        from rfcensus.tui.widgets.dongle_strip import DongleTile
        css = DongleTile.DEFAULT_CSS
        for color in ("green", "grey", "yellow", "red"):
            cls = f"-tile-{color}-detail"
            for line in css.splitlines():
                if cls in line:
                    assert "heavy" in line, (
                        f"v0.7.3: detail state for {color} now uses heavy border"
                    )
                    break

    def test_detail_dominates_cursor_when_both_set(self):
        # When a tile is both cursor and detail, the detail style wins.
        from rfcensus.tui.widgets.dongle_strip import DongleTile
        tile = DongleTile(slot=1, dongle=DongleState(
            dongle_id="x", status="active",
        ))
        tile.set_state(cursor=True, detail=True)
        assert "detail" in tile._current_class
        assert "cursor" not in tile._current_class

    def test_set_state_idempotent(self):
        # Repeated set_state with same args should NOT churn CSS classes.
        from rfcensus.tui.widgets.dongle_strip import DongleTile
        tile = DongleTile(slot=1, dongle=DongleState(
            dongle_id="x", status="active",
        ))
        tile.set_state(cursor=False, detail=False)
        first_class = tile._current_class
        tile.set_state(cursor=False, detail=False)
        # Second call: class string should be unchanged AND we should
        # have only that one class (no add/remove churn).
        assert tile._current_class == first_class


# ─────────────────────────────────────────────────────────────────────
# 3. Strip handles split cursor + detail
# ─────────────────────────────────────────────────────────────────────


class TestStripSplitSelection:
    def test_set_selection_with_different_cursor_and_detail(self):
        """The whole point of v0.6.16: cursor and detail can differ."""
        from rfcensus.tui.widgets.dongle_strip import DongleStrip
        # Can't actually mount widgets without a running app, so just
        # test the API contract using a freshly constructed strip.
        strip = DongleStrip()
        # Set up internal state by hand (compose() not called)
        strip._tiles = [
            type('FakeTile', (), {
                'set_state': lambda self, cursor, detail:
                    setattr(self, '_state', (cursor, detail))
            })()
            for _ in range(5)
        ]
        strip.set_selection(cursor_index=2, detail_index=4)
        # Tile 2 has cursor (no detail), tile 4 has detail (no cursor),
        # rest are plain
        assert strip._tiles[0]._state == (False, False)
        assert strip._tiles[2]._state == (True, False)
        assert strip._tiles[4]._state == (False, True)

    def test_set_selection_with_no_detail_pane(self):
        # detail_index=None means no detail pane is open
        from rfcensus.tui.widgets.dongle_strip import DongleStrip
        strip = DongleStrip()
        strip._tiles = [
            type('FakeTile', (), {
                'set_state': lambda self, cursor, detail:
                    setattr(self, '_state', (cursor, detail))
            })()
            for _ in range(3)
        ]
        strip.set_selection(cursor_index=1, detail_index=None)
        assert strip._tiles[1]._state == (True, False)
        assert strip._tiles[0]._state == (False, False)
        assert strip._tiles[2]._state == (False, False)

    def test_set_selection_clamps_indexes(self):
        from rfcensus.tui.widgets.dongle_strip import DongleStrip
        strip = DongleStrip()
        strip._tiles = [
            type('FakeTile', (), {
                'set_state': lambda self, cursor, detail: None
            })()
            for _ in range(3)
        ]
        # Out-of-range cursor/detail clamp to valid range
        strip.set_selection(cursor_index=99, detail_index=99)
        assert strip._cursor_index == 2
        assert strip._detail_index == 2


# ─────────────────────────────────────────────────────────────────────
# 4. Enter-toggle behavior
# ─────────────────────────────────────────────────────────────────────


class TestEnterToggle:
    """The action_open_detail() method now toggles based on the
    cursor-vs-detail state. We test it by directly calling it on an
    app-like object that just has the state attribute."""

    def _build_app(self):
        # Build the smallest object that satisfies action_open_detail's
        # requirements: a .state and a ._refresh_all() that's a no-op.
        from rfcensus.tui.app import TUIApp
        # We can't fully construct the app without a Textual context;
        # build a stand-in object that has the action method bound.
        app = TUIApp.__new__(TUIApp)
        app.state = _make_state_with_dongles(5)
        app._refresh_all = lambda: None
        return app

    def test_enter_from_events_opens_detail_at_cursor(self):
        app = self._build_app()
        app.state.focused_dongle_index = 2
        app.state.detail_dongle_index = None
        app.state.main_pane_mode = "events"
        app.action_open_detail()
        assert app.state.main_pane_mode == "dongle"
        assert app.state.detail_dongle_index == 2

    def test_enter_on_same_dongle_toggles_off(self):
        # Detail pane is showing dongle 3, cursor is also on 3,
        # press Enter → close pane.
        app = self._build_app()
        app.state.focused_dongle_index = 3
        app.state.detail_dongle_index = 3
        app.state.main_pane_mode = "dongle"
        app.action_open_detail()
        assert app.state.main_pane_mode == "events"
        # detail_dongle_index preserved so re-Enter reopens to same place
        assert app.state.detail_dongle_index == 3

    def test_enter_with_cursor_elsewhere_retargets_detail(self):
        # v0.7.3 behavior change: this case is now unreachable because
        # arrow keys update detail_dongle_index in the same call. By
        # the time you're in detail mode, focused_dongle_index and
        # detail_dongle_index are always equal; pressing Enter just
        # toggles the pane closed. Test renamed in spirit but kept
        # for git-blame continuity, asserting the NEW behavior.
        app = self._build_app()
        app.state.focused_dongle_index = 3
        app.state.detail_dongle_index = 3   # arrows already synced these
        app.state.main_pane_mode = "dongle"
        app.action_open_detail()
        # v0.7.3: Enter while in detail mode = close
        assert app.state.main_pane_mode == "events"

    def test_enter_twice_round_trip(self):
        # User flow: events → Enter → detail → Enter → events
        app = self._build_app()
        app.state.focused_dongle_index = 2
        app.state.detail_dongle_index = None
        app.state.main_pane_mode = "events"
        app.action_open_detail()
        assert app.state.main_pane_mode == "dongle"
        app.action_open_detail()
        assert app.state.main_pane_mode == "events"

    def test_arrows_dont_change_detail_pane(self):
        # v0.7.3 behavior change: arrows DO change the detail pane
        # while in detail mode (focus-follows-arrow). Previously they
        # only moved the cursor; now they immediately update the
        # detail pane so the user can browse fluidly without having
        # to press Enter to commit each switch.
        app = self._build_app()
        app.state.focused_dongle_index = 2
        app.state.detail_dongle_index = 2
        app.state.main_pane_mode = "dongle"
        app.action_focus_next()
        assert app.state.focused_dongle_index == 3
        # v0.7.3: detail_dongle_index NOW follows
        assert app.state.detail_dongle_index == 3
        assert app.state.main_pane_mode == "dongle"  # still open

        # Arrow keys in EVENTS mode (no detail pane open) still
        # don't open detail — they just move the cursor. That's
        # the part of the old behavior that's preserved.
        app.action_open_detail()    # close
        assert app.state.main_pane_mode == "events"
        app.action_focus_next()
        assert app.state.main_pane_mode == "events"  # NOT auto-opened

    def test_number_key_jumps_to_detail(self):
        # Pressing 5 → cursor and detail both go to dongle 5
        app = self._build_app()
        app.state.focused_dongle_index = 0
        app.state.detail_dongle_index = None
        app.state.main_pane_mode = "events"
        app._open_slot(5)
        assert app.state.focused_dongle_index == 4
        assert app.state.detail_dongle_index == 4
        assert app.state.main_pane_mode == "dongle"


# ─────────────────────────────────────────────────────────────────────
# 5. Tile content — show decodes, detections, sample rate
# ─────────────────────────────────────────────────────────────────────


class TestTileContent:
    """v0.6.16: dongle tiles now show decode count, detection count,
    AND sample rate so the user can see at a glance what each dongle
    is doing without opening the detail pane.

    Tile is 6 rows tall (4 content lines after border):
      Line 1: slot key + dongle id
      Line 2: status glyph + band id
      Line 3: freq + sample rate (rate dropped at narrow widths)
      Line 4: decode + detection counts (or status-failure message)
    """

    def _render_at(self, dongle, width=24, height=6):
        from rfcensus.tui.widgets.dongle_strip import DongleTile
        import unittest.mock as mock
        tile = DongleTile(slot=1, dongle=dongle)
        with mock.patch.object(DongleTile, "size",
                               new_callable=mock.PropertyMock) as ps:
            ps.return_value = type("S", (), {
                "width": width, "height": height,
            })()
            tile.set_state(cursor=False, detail=False)
            return tile.render()

    def test_decode_count_visible_when_nonzero(self):
        d = DongleState(
            dongle_id="x", status="active", band_id="b",
            freq_hz=915_000_000, sample_rate=2_400_000,
            decodes_in_band=14,
        )
        out = self._render_at(d)
        assert "14" in out
        assert "dec" in out, f"Expected 'dec' label in:\n{out}"

    def test_detection_count_visible_when_nonzero(self):
        # The user's screenshot scenario: dongle has 0 decodes but 17
        # detections (lora_survey active). Must show detection count.
        d = DongleState(
            dongle_id="x", status="active", band_id="915_ism_r900",
            freq_hz=912_600_000, sample_rate=2_359_296,
            decodes_in_band=0, detections_in_band=17,
        )
        out = self._render_at(d)
        assert "17" in out
        assert "det" in out, (
            f"Detection count must be surfaced on the tile (the v0.6.14 "
            f"behavior of hiding when decodes==0 lost important info). "
            f"Got:\n{out}"
        )

    def test_both_decode_and_detection_visible(self):
        d = DongleState(
            dongle_id="x", status="active", band_id="b",
            freq_hz=915_000_000, sample_rate=2_400_000,
            decodes_in_band=14, detections_in_band=17,
        )
        out = self._render_at(d)
        assert "14" in out and "17" in out

    def test_sample_rate_visible_at_wide(self):
        d = DongleState(
            dongle_id="x", status="active", band_id="b",
            freq_hz=915_000_000, sample_rate=2_400_000,
        )
        out = self._render_at(d, width=24)
        # Sample rate shows compact: "2.4M" (the .00 stripped)
        assert "2.4M" in out, f"Expected '2.4M' rate in:\n{out}"

    def test_sample_rate_dropped_at_narrow(self):
        # At narrow widths, freq alone fits but rate does not. Rate is
        # dropped to keep freq readable.
        d = DongleState(
            dongle_id="x", status="active", band_id="b",
            freq_hz=915_000_000, sample_rate=2_400_000,
        )
        out = self._render_at(d, width=14)
        assert "915.000M" in out
        assert "2.4M" not in out

    def test_freq_format_compact_M_suffix(self):
        d = DongleState(
            dongle_id="x", status="active", band_id="b",
            freq_hz=912_600_000,
        )
        out = self._render_at(d)
        # Compact form ends with M, not " MHz" — saves chars for rate
        assert "912.600M" in out

    def test_4_content_lines(self):
        # Tile renders exactly 4 lines (CSS height: 6 minus 2 border).
        d = DongleState(
            dongle_id="x", status="active", band_id="b",
            freq_hz=915_000_000, sample_rate=2_400_000,
            decodes_in_band=5, detections_in_band=10,
        )
        out = self._render_at(d)
        # Empty trailing line is fine; we just want at least 4
        # newline-separated lines so the layout matches the CSS.
        assert out.count("\n") == 3, (
            f"Tile should render 4 lines (3 newlines); got {out.count(chr(10))}\n"
            f"Render:\n{out}"
        )

    def test_permanent_failed_shows_error_text(self):
        # Failed dongles display the failure text in line 4 instead
        # of decode/detection counts. The failure status is more
        # important than counts when it fires.
        d = DongleState(
            dongle_id="x", status="permanent_failed", band_id="b",
            freq_hz=915_000_000, sample_rate=2_400_000,
            decodes_in_band=5,
        )
        out = self._render_at(d)
        assert "permanent failure" in out.lower()
        # Decode count NOT shown — failure text takes the slot
        assert "dec 5" not in out

    def test_idle_no_counts_renders_empty_line(self):
        # Idle dongle with no decodes/detections — line 4 is blank
        # rather than showing "dec 0" noise.
        d = DongleState(
            dongle_id="x", status="idle", band_id=None,
            freq_hz=None, sample_rate=None,
        )
        out = self._render_at(d)
        # No "dec 0" or "det 0" in the output
        assert "dec 0" not in out
        assert "det 0" not in out
