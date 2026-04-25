"""Horizontal strip of dongle tiles. One tile per dongle."""

from __future__ import annotations

from textual.app import ComposeResult, RenderResult
from textual.containers import Horizontal
from textual.widget import Widget

from rfcensus.tui.state import DongleState


class DongleTile(Widget):
    """One tile in the strip. Compact at narrow widths, more detail
    when the terminal is wider."""

    DEFAULT_CSS = """
    DongleTile {
        width: 1fr;
        min-width: 14;
        max-width: 24;
        height: 5;
        border: round $surface;
        padding: 0 1;
    }
    DongleTile.-focused {
        border: heavy $accent;
    }
    """

    def __init__(self, *, slot: int, dongle: DongleState | None = None) -> None:
        super().__init__()
        self.slot = slot
        self._dongle = dongle
        self._focused_indicator = False

    def update_dongle(self, dongle: DongleState | None) -> None:
        self._dongle = dongle
        self.refresh()

    def set_focused(self, focused: bool) -> None:
        self._focused_indicator = focused
        if focused:
            self.add_class("-focused")
        else:
            self.remove_class("-focused")
        self.refresh()

    def render(self) -> RenderResult:
        from rfcensus.tui.color import (
            dongle_status_glyph,
            style_for_dongle_status,
            styled,
        )

        d = self._dongle
        # Slot key mapping — slot 10 displays as "0"
        key = "0" if self.slot == 10 else str(self.slot)

        if d is None:
            head = f"[{key}] (empty)"
            return f"{head}\n\n\n"

        # Truncate dongle id for narrow display
        avail_width = max(10, self.size.width - 6)
        short_id = d.dongle_id
        if len(short_id) > avail_width:
            # Show last N chars (typically the serial suffix is most distinctive)
            short_id = "…" + short_id[-(avail_width - 1):]

        head = f"[{key}] {short_id}"
        glyph = dongle_status_glyph(d.status)
        glyph_styled = styled(style_for_dongle_status(d.status), glyph)
        band = d.band_id or "—"
        if d.freq_hz:
            freq = f"{d.freq_hz / 1e6:.3f} MHz"
        else:
            freq = "—"

        line2 = f"{glyph_styled} {band}"
        line3 = freq
        # 4th line: decodes if any room
        line4 = ""
        if self.size.height >= 4 and d.decodes_in_band > 0:
            line4 = f"↕ {d.decodes_in_band} decode(s)"
        elif self.size.height >= 4 and d.status == "permanent_failed":
            line4 = styled("error", "permanent failure")

        return f"{head}\n{line2}\n{line3}\n{line4}"


class DongleStrip(Widget):
    """Horizontal arrangement of DongleTile widgets."""

    DEFAULT_CSS = """
    DongleStrip {
        height: 7;
        dock: top;
    }
    DongleStrip Horizontal {
        height: 7;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._tiles: list[DongleTile] = []
        self._dongles: list[DongleState] = []
        self._focused_index = 0

    def compose(self) -> ComposeResult:
        yield Horizontal(id="strip-row")

    def update_dongles(self, dongles: list[DongleState]) -> None:
        """Rebuild tiles if the dongle count changed; otherwise just
        push fresh state into existing tiles. Avoids tile flicker on
        every state change."""
        self._dongles = dongles
        row = self.query_one("#strip-row", Horizontal)
        # If count changed, rebuild from scratch
        if len(self._tiles) != len(dongles):
            row.remove_children()
            self._tiles = []
            for i, d in enumerate(dongles, start=1):
                tile = DongleTile(slot=i, dongle=d)
                self._tiles.append(tile)
                row.mount(tile)
            # Re-apply focus
            for i, tile in enumerate(self._tiles):
                tile.set_focused(i == self._focused_index)
        else:
            for tile, d in zip(self._tiles, dongles):
                tile.update_dongle(d)

    def set_focused_index(self, idx: int) -> None:
        if not self._tiles:
            return
        idx = max(0, min(idx, len(self._tiles) - 1))
        self._focused_index = idx
        for i, tile in enumerate(self._tiles):
            tile.set_focused(i == idx)
