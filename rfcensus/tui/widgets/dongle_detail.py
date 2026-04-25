"""Per-dongle detail view — full-screen takeover when user opens a tile."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.screen import Screen
from textual.widgets import Static

from rfcensus.tui.state import DongleState, StreamEntry


class DongleDetail(Screen):
    """Full-screen detail for one dongle.

    Shows: status, current freq/SR/band/consumer, recent decode count,
    fanout client activity, recent stream entries filtered to this
    dongle. Read-only in v0.6.5 (e/r action hotkeys land in v0.6.6).
    """

    DEFAULT_CSS = """
    DongleDetail {
        background: $background;
    }
    DongleDetail > Container {
        padding: 1 2;
    }
    """

    BINDINGS = [
        ("escape", "back", "back to strip"),
        ("left", "prev", "previous dongle"),
        ("right", "next", "next dongle"),
    ]

    def __init__(
        self,
        dongles: list[DongleState],
        index: int,
        stream: list[StreamEntry],
    ) -> None:
        super().__init__()
        self._dongles = dongles
        self._index = max(0, min(index, len(dongles) - 1))
        self._stream = stream

    def compose(self) -> ComposeResult:
        with Container():
            with VerticalScroll():
                yield Static(self._render_body(), id="detail-body")

    def _render_body(self) -> str:
        from rfcensus.tui.color import (
            dongle_status_glyph,
            style_for_dongle_status,
            styled,
        )

        if not self._dongles:
            return styled("idle", "(no dongles)")
        d = self._dongles[self._index]
        glyph = styled(style_for_dongle_status(d.status),
                       dongle_status_glyph(d.status))
        slot = self._index + 1
        slot_key = "0" if slot == 10 else str(slot)
        head = (f"{styled('focus', f'[{slot_key}] {d.dongle_id}')}  "
                f"{glyph} [bold]{d.status}[/]")
        if d.status_message:
            head += f"  — {d.status_message}"

        # Body
        lines = [head, ""]
        if d.consumer:
            lines.append(f"  consumer:    {d.consumer}")
            lines.append(f"  band:        {d.band_id or '—'}")
            if d.freq_hz:
                lines.append(f"  freq:        {d.freq_hz / 1e6:.6f} MHz")
            else:
                lines.append("  freq:        —")
            if d.sample_rate:
                lines.append(f"  sample rate: {d.sample_rate} Hz")
            else:
                lines.append("  sample rate: —")
        else:
            lines.append("  (idle — no current lease)")

        lines.extend([
            "",
            f"  decodes (band):  {d.decodes_in_band}",
            f"  decodes (total): {d.decodes_total}",
            f"  fanout clients:  {d.fanout_clients}",
        ])
        if d.last_decode_at:
            lines.append(
                f"  last decode at:  "
                f"{d.last_decode_at.strftime('%H:%M:%S')}"
            )
        lines.extend(["", styled("info", "Recent events"), ""])

        # Stream entries — filter to ones mentioning this dongle id
        related = [
            e for e in self._stream
            if d.dongle_id in e.text
        ][:25]
        if related:
            for e in related:
                ts = e.timestamp.strftime("%H:%M:%S")
                lines.append(f"  [dim]{ts}[/]  {e.text}")
        else:
            lines.append(styled("idle", "  (none yet)"))

        # Footer hint
        lines.extend([
            "",
            styled(
                "idle",
                "  ←/→ prev/next dongle  ↑/↓ scroll  Esc back",
            ),
        ])
        return "\n".join(lines)

    def action_back(self) -> None:
        self.dismiss()

    def action_prev(self) -> None:
        if self._index > 0:
            self._index -= 1
            self.query_one("#detail-body", Static).update(self._render_body())

    def action_next(self) -> None:
        if self._index < len(self._dongles) - 1:
            self._index += 1
            self.query_one("#detail-body", Static).update(self._render_body())
