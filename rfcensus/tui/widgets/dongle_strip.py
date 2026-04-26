"""Horizontal strip of dongle tiles. One tile per dongle.

v0.6.14 visual scheme (revised v0.6.16):
  • Border COLOR encodes status (green=decodes, grey=running-quiet,
    yellow=warn, red=fail). NEVER overridden — status info is too
    valuable to lose to selection state.
  • Border STYLE encodes selection state (v0.7.3 swap):
      round  = unselected (plain)
      double = cursor only (focused but detail pane shows something else)
      heavy  = detail-shown (this dongle's detail is currently displayed)
    A tile that's both cursor-focused AND detail-shown uses `heavy` —
    the detail signal dominates since the cursor will move with the
    next arrow press but the detail pane sticks until Enter/Esc.
    The "fat single" (heavy) reads as the truly-active state more
    intuitively than double, hence the swap from earlier versions.
  • The slot key in the header gets reinforcing styling that survives
    color blindness:
      plain    [1]
      cursor   [1] (bold)
      detail   [1] (bold + reverse video)

Why orthogonal status+style?
  In v0.6.14's first cut, selection forced the border to white. That
  worked for the cursor signal but DESTROYED the at-a-glance status
  reading — a green active dongle would turn white when selected and
  the user couldn't tell at a glance whether it was working. Splitting
  color (status) and style (selection) restores both signals at once.

Why single-class CSS instead of compound selectors?
  Textual's CSS pipeline hangs in `pilot.pause()` when class selectors
  use either literal color names like `grey50` or compound selectors
  like `.-status-X.-selected`. Both manifest as `WaitForScreenTimeout:
  widgets to process pending messages`. So we use 12 single-class
  combinations (4 colors × 3 selection states) — verbose but safe.
"""

from __future__ import annotations

from textual.app import ComposeResult, RenderResult
from textual.containers import Horizontal
from textual.widget import Widget

from rfcensus.tui.state import DongleState


# Three exclusive selection states for a tile. We collapse cursor+detail
# into "detail" because the user typically cares about which one's
# DETAIL is showing, not where the cursor happens to also be.
_SEL_PLAIN = "plain"
_SEL_CURSOR = "cursor"
_SEL_DETAIL = "detail"


class DongleTile(Widget):
    """One tile in the strip. Compact at narrow widths, more detail
    when the terminal is wider."""

    # 12 single-class combinations — color × selection-style. Single
    # class per tile at any time so no compound selectors needed.
    DEFAULT_CSS = """
    DongleTile {
        width: 1fr;
        min-width: 14;
        max-width: 24;
        height: 6;
        padding: 0 1;
    }
    /* status × style matrix.
       Style mapping (v0.7.3 swap — user feedback that "fat single"
       reads as the truly-active state more intuitively than double):
         plain   -> round  (thin, unobtrusive)
         cursor  -> double (two-line, says "cursor here but not active")
         detail  -> heavy  (THICK, says "this is the active dongle") */
    DongleTile.-tile-green-plain   { border: round  green;  }
    DongleTile.-tile-green-cursor  { border: double green;  }
    DongleTile.-tile-green-detail  { border: heavy  green;  }
    DongleTile.-tile-grey-plain    { border: round  grey;   }
    DongleTile.-tile-grey-cursor   { border: double grey;   }
    DongleTile.-tile-grey-detail   { border: heavy  grey;   }
    DongleTile.-tile-yellow-plain  { border: round  yellow; }
    DongleTile.-tile-yellow-cursor { border: double yellow; }
    DongleTile.-tile-yellow-detail { border: heavy  yellow; }
    DongleTile.-tile-red-plain     { border: round  red;    }
    DongleTile.-tile-red-cursor    { border: double red;    }
    DongleTile.-tile-red-detail    { border: heavy  red;    }
    """

    def __init__(self, *, slot: int, dongle: DongleState | None = None) -> None:
        super().__init__()
        self.slot = slot
        self._dongle = dongle
        self._sel = _SEL_PLAIN
        self._current_class = ""

    def update_dongle(self, dongle: DongleState | None) -> None:
        self._dongle = dongle
        self._sync_class()
        self.refresh()

    def set_state(self, *, cursor: bool, detail: bool) -> None:
        """Set the tile's selection state. detail dominates over cursor
        — a tile that's both gets the detail style. This matches user
        intent: detail pane visibility is the more "committed" state
        and the more important visual signal."""
        if detail:
            new_sel = _SEL_DETAIL
        elif cursor:
            new_sel = _SEL_CURSOR
        else:
            new_sel = _SEL_PLAIN
        # Always sync on first call (when _current_class is empty)
        # even if selection state hasn't changed; otherwise the tile
        # never gets ANY class assigned and renders without status
        # color.
        if new_sel == self._sel and self._current_class:
            return  # no-op; avoids redundant class churn
        self._sel = new_sel
        self._sync_class()
        self.refresh()

    def _sync_class(self) -> None:
        """Compute and apply the right `-tile-COLOR-STYLE` class.
        Idempotent — only mutates when target differs from current."""
        from rfcensus.tui.color import dongle_border_color

        if self._dongle is None:
            color = "grey"
        else:
            has_decodes = self._dongle.decodes_in_band > 0
            # v0.7.4: slow-chunk events make the fanout yellow even
            # while it's still producing decodes, so the user can
            # spot backpressure at a glance.
            has_warnings = self._dongle.fanout_dropped_chunks > 0
            raw = dongle_border_color(
                self._dongle.status, has_decodes, has_warnings,
            )
            # The color helper returns "grey50" sometimes; map to "grey"
            # for the CSS class lookup. ANSI palette only — see CSS hang
            # caveat in module docstring.
            color = "grey" if raw in ("grey", "grey50") else raw
        target = f"-tile-{color}-{self._sel}"
        if target == self._current_class:
            return
        if self._current_class:
            self.remove_class(self._current_class)
        self.add_class(target)
        self._current_class = target

    def render(self) -> RenderResult:
        from rfcensus.tui.color import (
            dongle_status_glyph,
            style_for_dongle_status,
            styled,
        )

        d = self._dongle
        # Slot key mapping — slot 10 displays as "0"
        key = "0" if self.slot == 10 else str(self.slot)

        # Slot key styling reflects selection state with text emphasis
        # in addition to border style. Colorblind-safe redundancy.
        # IMPORTANT: wrap the WHOLE slot block including brackets, not
        # just the digit, to avoid Rich-markup ambiguity from `[bold]X
        # [/bold]` appearing inside the literal `[]` brackets which
        # the parser conflates with a markup tag.
        if self._sel == _SEL_DETAIL:
            head_prefix = f"[bold reverse][{key}][/bold reverse]"
        elif self._sel == _SEL_CURSOR:
            head_prefix = f"[bold][{key}][/bold]"
        else:
            head_prefix = f"[{key}]"

        if d is None:
            head = f"{head_prefix} (empty)"
            return f"{head}\n\n\n"

        # Truncate dongle id for narrow display
        avail_width = max(10, self.size.width - 6)
        short_id = d.dongle_id
        if len(short_id) > avail_width:
            short_id = "…" + short_id[-(avail_width - 1):]

        head = f"{head_prefix} {short_id}"
        glyph = dongle_status_glyph(d.status)
        glyph_styled = styled(style_for_dongle_status(d.status), glyph)
        band = d.band_id or "—"
        # Line 2: status glyph + band
        line2 = f"{glyph_styled} {band}"

        # Line 3: freq + sample rate. Show rate compactly (e.g. "2.4M")
        # because rtl_sdr rates are always integer-Mhz-ish. Format
        # picks based on width: at narrow widths, freq alone; at wider,
        # freq @ rate.
        if d.freq_hz:
            freq_mhz = d.freq_hz / 1e6
            freq_str = f"{freq_mhz:.3f}M"  # 912.600M (7 chars)
        else:
            freq_str = "—"
        if d.sample_rate:
            rate_mhz = d.sample_rate / 1e6
            # Compact format: "2.4M" / "2.36M" / "10M". Strip trailing
            # zeros from the numeric part BEFORE appending the suffix
            # — rstrip on "2.40M" doesn't strip because it ends in "M".
            if rate_mhz >= 10:
                rate_str = f"{rate_mhz:.0f}M"
            elif rate_mhz >= 1:
                num = f"{rate_mhz:.2f}".rstrip("0").rstrip(".")
                rate_str = f"{num}M"
            else:
                rate_str = f"{d.sample_rate / 1e3:.0f}k"
        else:
            rate_str = ""
        # Combine if both fit in available width (avail_width is the
        # space we have for content after subtracting the slot prefix
        # and a small margin). At minimum tile width 14 (12 inside the
        # border with padding 0 1), we have ~10 usable chars per line.
        # "912.600M" is 8 chars; adding " @ 2.4M" makes 15. So at
        # narrowest we drop the rate.
        if rate_str and self.size.width >= 18:
            line3 = f"{freq_str} @ {rate_str}"
        else:
            line3 = freq_str

        # Line 4: decode count + detection count, compact. Always shown
        # when either is non-zero; for permanent-failed dongles, error
        # text takes the slot instead.
        line4 = ""
        if d.status == "permanent_failed":
            line4 = styled("error", "permanent failure")
        elif d.decodes_in_band > 0 or d.detections_in_band > 0:
            # v0.7.3: "dec N" reads as decimation in SDR contexts and
            # confused users. Use the unambiguous ↓ (received decode)
            # and ✓ (detection) glyphs, with explicit count labels for
            # the wider variant. Two pairs at most so even narrow
            # tiles fit.
            dec = d.decodes_in_band
            det = d.detections_in_band
            if self.size.width >= 20:
                parts = []
                if dec > 0:
                    parts.append(f"decodes:{dec}")
                if det > 0:
                    parts.append(f"detects:{det}")
                line4 = " · ".join(parts)
            else:
                # Compact form: glyph + count, two pairs separated by
                # space. ↓ for decodes (downstream output), ✓ for
                # detections (signal confirmed but not full decode).
                parts = []
                if dec > 0:
                    parts.append(f"↓{dec}")
                if det > 0:
                    parts.append(f"✓{det}")
                line4 = " ".join(parts)

        return f"{head}\n{line2}\n{line3}\n{line4}"


class DongleStrip(Widget):
    """Horizontal arrangement of DongleTile widgets."""

    DEFAULT_CSS = """
    DongleStrip {
        height: 8;
        /* v0.7.4: removed `dock: top` after diagnosing why HeaderBar
           was invisible — Textual 8.x stacks two siblings both
           docked to `top` at y=0, so DongleStrip was painting on
           top of HeaderBar. With dock removed, DongleStrip flows
           naturally as the second child of Container#main (after
           HeaderBar's docked y=0), getting placed at y=1. */
    }
    DongleStrip Horizontal {
        height: 8;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._tiles: list[DongleTile] = []
        self._dongles: list[DongleState] = []
        self._cursor_index = 0
        self._detail_index: int | None = None

    def compose(self) -> ComposeResult:
        yield Horizontal(id="strip-row")

    def update_dongles(self, dongles: list[DongleState]) -> None:
        """Rebuild tiles if the dongle count changed; otherwise just
        push fresh state into existing tiles. Avoids tile flicker on
        every state change."""
        self._dongles = dongles
        row = self.query_one("#strip-row", Horizontal)
        if len(self._tiles) != len(dongles):
            row.remove_children()
            self._tiles = []
            for i, d in enumerate(dongles, start=1):
                tile = DongleTile(slot=i, dongle=d)
                self._tiles.append(tile)
                row.mount(tile)
            self._apply_selection_to_tiles()
        else:
            for tile, d in zip(self._tiles, dongles):
                tile.update_dongle(d)

    def set_selection(self, *, cursor_index: int,
                      detail_index: int | None) -> None:
        """v0.6.16: split the old set_selected_index into separate
        cursor and detail signals. Called by app refresh. cursor_index
        is the focused tile (arrow keys move this). detail_index is
        which tile's detail pane is shown (None = no detail pane open)."""
        if not self._tiles:
            return
        cursor_index = max(0, min(cursor_index, len(self._tiles) - 1))
        if detail_index is not None:
            detail_index = max(0, min(detail_index, len(self._tiles) - 1))
        self._cursor_index = cursor_index
        self._detail_index = detail_index
        self._apply_selection_to_tiles()

    def _apply_selection_to_tiles(self) -> None:
        for i, tile in enumerate(self._tiles):
            tile.set_state(
                cursor=(i == self._cursor_index),
                detail=(i == self._detail_index) if self._detail_index is not None else False,
            )

    # Backward-compat aliases — pre-v0.6.16 the strip exposed
    # `set_selected_index` and `set_focused_index`. Some app code
    # still uses these names; route through to the new API treating
    # the index as a cursor change with detail unchanged.
    def set_selected_index(self, idx: int) -> None:
        self.set_selection(cursor_index=idx, detail_index=self._detail_index)

    def set_focused_index(self, idx: int) -> None:
        self.set_selection(cursor_index=idx, detail_index=self._detail_index)
