"""Per-dongle detail view — inline panel that swaps into the main
pane (replaces the EventStream when a dongle is selected).

v0.6.14: was a full-screen Screen modal; converted to a plain Widget
that lives in the main pane.

v0.6.17: metric set reworked per user feedback. Old layout split
decode + detection counters into "(band)" and "(total)" rows. The
"(total)" rows were misleading — a dongle that ran 3 different bands
in earlier waves accumulates a total that's NOT comparable to other
dongles' totals because they ran different bands. The total tells
you very little about the current band's productivity.

New metric set focuses on what's true and useful right now:

  • antenna  — config-static, helps the user remember which physical
               cable is plugged into this dongle (was missing entirely
               from the old layout)
  • status + status message
  • current band + freq + rate
  • on-band time (since the current band lease started)
  • decodes + detections in current band only (no spurious totals)
  • fanout clients (active downstream consumers)
  • fanout bytes streamed (lifetime; useful "is this dongle busy" cue)
  • bands visited count (how diverse this dongle's work has been)

Recent events stays at the bottom — filtered to mentions of this
dongle. The panel is meant for context-rich, glance-friendly inspection
of one dongle without leaving the main view.
"""

from __future__ import annotations

from datetime import datetime, timezone

from textual.app import RenderResult
from textual.widget import Widget

from rfcensus.tui.state import DongleState, StreamEntry


def _format_bytes(n: int) -> str:
    """Compact byte-count formatter. 0..999 → bytes; 1k..999k →
    k; 1M..999M → M; 1G+ → G. Two sig figs above 1k."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _format_elapsed(seconds: float) -> str:
    """Compact elapsed time. Same as HeaderBar's helper but local
    to keep widget dependencies minimal."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


class DongleDetail(Widget):
    """Inline per-dongle stats panel."""

    DEFAULT_CSS = """
    DongleDetail {
        background: $background;
        padding: 0 1;
        border: round $surface;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._dongles: list[DongleState] = []
        self._index: int = 0
        self._stream: list[StreamEntry] = []
        # v0.7.4: scroll offset into the rendered line array. Same
        # convention as EventStream: 0 means "scrolled to top of the
        # detail content" (which is where users land when first
        # opening detail). Positive values scroll DOWN through the
        # content.
        self._scroll_offset: int = 0
        # v0.7.4: cache of how many lines the last render() emitted,
        # used by scroll_lines to clamp max_offset. Set to a large
        # default so initial scrolls aren't clamped to 0 before the
        # first render runs.
        self._last_rendered_lines: int = 200
        self.border_title = "Dongle detail"

    def scroll_lines(self, n: int) -> None:
        """Scroll the rendered content by n lines. Positive n = down
        (see content further into the report); negative n = up.
        Clamped so the bottom of the content stays at or above the
        bottom of the viewport — no degenerate "everything off-screen"
        states. Bound to ↑/↓ via app actions when in dongle mode."""
        # We don't know the real content height until render builds
        # it; cap pessimistically using the rendered cache from the
        # last paint.
        max_offset = max(0, self._last_rendered_lines - self._max_visible_lines())
        new_offset = max(0, min(self._scroll_offset + n, max_offset))
        if new_offset == self._scroll_offset:
            return
        self._scroll_offset = new_offset
        self.refresh()

    def _max_visible_lines(self) -> int:
        try:
            return max(2, self.size.height - 2)    # subtract round border
        except Exception:
            return 25

    def update_state(
        self,
        dongles: list[DongleState],
        index: int,
        stream: list[StreamEntry],
    ) -> None:
        # v0.6.14: idempotent — only mutate state + refresh when
        # something visibly changed. Without this, the per-tick
        # _refresh_all called update_state every second even with
        # identical inputs, queueing a refresh message each time.
        # In Textual pilot tests this accumulated messages faster
        # than the test loop drained them, hitting WaitForScreenTimeout.
        new_index = max(0, min(index, max(0, len(dongles) - 1)))
        if (
            self._dongles is dongles
            and self._index == new_index
            and self._stream is stream
        ):
            return
        # v0.7.4: reset scroll offset when the focused dongle changes
        # so users always land at the top of the new dongle's content,
        # not at whatever line they'd scrolled to in the previous one.
        if self._index != new_index:
            self._scroll_offset = 0
        self._dongles = dongles
        self._index = new_index
        self._stream = stream
        # Reflect current dongle id in the panel title — only assign
        # when it actually changed (border_title= triggers a layout
        # message on Textual ≥ 0.40).
        if self._dongles and 0 <= self._index < len(self._dongles):
            d = self._dongles[self._index]
            slot = self._index + 1
            slot_key = "0" if slot == 10 else str(slot)
            new_title = f"Dongle [{slot_key}] {d.dongle_id}"
        else:
            new_title = "Dongle detail"
        if new_title != self.border_title:
            self.border_title = new_title
        self.refresh()

    def render(self) -> RenderResult:
        from rfcensus.tui.color import (
            dongle_status_glyph,
            style_for_dongle_status,
            styled,
        )

        if not self._dongles:
            return styled("idle", "(no dongles)")
        if not (0 <= self._index < len(self._dongles)):
            return styled("idle", "(invalid selection)")

        d = self._dongles[self._index]
        glyph = styled(style_for_dongle_status(d.status),
                       dongle_status_glyph(d.status))
        # v0.7.4: prominent "Dongle N — id" header at the top of the
        # content. The border_title already says this but Textual
        # renders border titles in a small dim font that's easy to
        # miss; users were asking "which dongle is this again?"
        # mid-scroll. Putting it inside the content as a bold first
        # line solves that — also survives terminals that don't
        # render rich border titles.
        slot = self._index + 1
        slot_key = "0" if slot == 10 else str(slot)
        title_line = f"[bold]Dongle [{slot_key}][/] — {d.dongle_id}"
        head = f"{glyph} [bold]{d.status}[/]"
        if d.status_message:
            head += f"  — {d.status_message}"

        # ── Static identity ─────────────────────────────────────
        # v0.6.17: antenna and model are config-static so they're
        # safe to show even when idle. They help the user mentally
        # locate the physical hardware.
        lines = [title_line, head, ""]
        if d.model:
            lines.append(f"  model:       {d.model}")
        lines.append(f"  antenna:     {d.antenna_id or '—'}")

        # ── Current lease ──────────────────────────────────────
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
            # v0.6.17: time-on-band — useful for spotting stuck
            # tasks (a band sitting at 30+ minutes when its lease
            # was supposed to be 12 minutes is a sign of trouble).
            if d.band_started_at:
                on_band = (
                    datetime.now(timezone.utc) - d.band_started_at
                ).total_seconds()
                lines.append(f"  on band:     {_format_elapsed(on_band)}")
        else:
            lines.append("  (idle — no current lease)")

        # ── Productivity (current band only — v0.6.17) ─────────
        # The old "(total)" counters were misleading because they
        # mixed productivity across bands the dongle visited at
        # different times. Now we only show in-band counts.
        lines.extend([
            "",
            f"  decodes:     {d.decodes_in_band}",
            f"  detections:  {d.detections_in_band}",
        ])
        if d.last_decode_at:
            lines.append(
                f"  last decode: "
                f"{d.last_decode_at.strftime('%H:%M:%S')}"
            )
        if d.last_detection_at:
            lines.append(
                f"  last det.:   "
                f"{d.last_detection_at.strftime('%H:%M:%S')}"
            )

        # ── Fanout + lifetime ──────────────────────────────────
        # v0.7.4: render the named consumer set (rtl_433, rtlamr,
        # lora_survey) instead of the raw peer addresses (which are
        # ephemeral TCP source ports the user can't act on). Peer
        # addresses still listed below in dim text for power users
        # who need to correlate with strace/lsof output.
        lines.extend([
            "",
            f"  fanout clients:   {d.fanout_clients}",
        ])
        if d.active_consumers:
            for consumer in sorted(d.active_consumers):
                lines.append(f"    • {consumer}")
        elif d.fanout_client_peers:
            # Fallback if for some reason we got peers without consumer
            # tracking (shouldn't happen in normal operation).
            for peer in sorted(d.fanout_client_peers):
                lines.append(f"    • [dim]{peer}[/]")
        # Show the raw peer addresses dimly below the named list so
        # both views are available — the consumers tell you "what's
        # connected", the peers tell you "as seen by netstat".
        if d.active_consumers and d.fanout_client_peers:
            lines.append(
                styled("idle",
                       f"    [{', '.join(sorted(d.fanout_client_peers))}]")
            )

        # v0.7.4: live data rate instead of cumulative bytes.
        # The fanout doesn't publish periodic byte counts (only on
        # connect/disconnect/slow), so the cumulative counter never
        # updated for steady-state runs and showed "0 B" forever.
        # The derived rate is upstream sample rate × 2 bytes/sample
        # × N downstream clients — which is what the fanout actually
        # does push downstream every second. Use decimal MB to match
        # SDR convention (2.4 MS/s × 2 B × 3 clients = 14.4 MB/s,
        # not 13.7 MiB/s).
        if d.sample_rate and d.fanout_clients > 0:
            # rtl_tcp wire format is 8-bit unsigned I + 8-bit unsigned Q
            # = 2 bytes per complex sample. Each downstream client
            # gets a copy.
            bytes_per_sec = d.sample_rate * 2 * d.fanout_clients
            mb_per_sec = bytes_per_sec / 1_000_000
            lines.append(
                f"  stream rate:      "
                f"{mb_per_sec:.1f} MB/s ({d.sample_rate / 1e6:.2f} MS/s "
                f"× 2 B × {d.fanout_clients} client"
                f"{'s' if d.fanout_clients != 1 else ''})"
            )
        elif d.fanout_bytes_sent > 0:
            # Disconnected clients have published their lifetime byte
            # totals; show that running sum.
            lines.append(
                f"  bytes streamed:   {_format_bytes(d.fanout_bytes_sent)} "
                "(from disconnected clients)"
            )

        lines.append(f"  bands visited:    {len(d.bands_visited)}")
        if d.fanout_dropped_chunks:
            lines.append(
                styled("warning",
                       f"  ⚠ slow incidents: {d.fanout_dropped_chunks}")
            )

        # ── Recent decodes (v0.7.3) ─────────────────────────────
        # User asked to see what's actually being decoded, not just
        # a counter. Last 8 entries shown here; full history via
        # `rfcensus list decodes`.
        if d.recent_decodes:
            lines.extend(["", styled("info", "Recent decodes"), ""])
            for entry in list(d.recent_decodes)[-8:]:
                ts = entry.timestamp.strftime("%H:%M:%S")
                freq_mhz = entry.freq_hz / 1e6
                lines.append(
                    f"  [dim]{ts}[/]  {freq_mhz:>7.3f}M  "
                    f"[blue]{entry.protocol:<12s}[/]  {entry.summary}"
                )

        # ── Recent detections (v0.7.3) ──────────────────────────
        if d.recent_detections:
            lines.extend(["", styled("info", "Recent detections"), ""])
            for entry in list(d.recent_detections)[-8:]:
                ts = entry.timestamp.strftime("%H:%M:%S")
                freq_mhz = entry.freq_hz / 1e6
                lines.append(
                    f"  [dim]{ts}[/]  {freq_mhz:>7.3f}M  "
                    f"[magenta]{entry.technology:<16s}[/]  "
                    f"conf={entry.confidence:.2f}"
                )

        # ── Recent events filtered to this dongle ──────────────
        # General task/status events as a fallback context line.
        related = [e for e in self._stream if d.dongle_id in e.text][:8]
        if related:
            lines.extend(["", styled("info", "Recent events"), ""])
            for e in related:
                ts = e.timestamp.strftime("%H:%M:%S")
                lines.append(f"  [dim]{ts}[/]  {e.text}")

        # ── Footer hint ────────────────────────────────────────
        lines.extend([
            "",
            styled(
                "idle",
                "  ←/→ prev/next dongle  ↑/↓ scroll  Esc back to events",
            ),
        ])
        # v0.7.4: cache total line count so scroll_lines can clamp
        # max_offset accurately on the next keystroke.
        self._last_rendered_lines = len(lines)
        # Apply scroll offset by slicing off `offset` lines from the
        # top. Footer hint stays at the bottom so users always see
        # the keybinding reminder.
        if self._scroll_offset > 0:
            offset = min(self._scroll_offset, max(0, len(lines) - 3))
            lines = lines[offset:]
            # Indicate scroll position at the top so users know there's
            # content above them.
            from rfcensus.tui.color import styled as _styled
            lines.insert(
                0,
                _styled(
                    "idle",
                    f"  [↑ {offset} line(s) above — Home / Esc to reset]",
                ),
            )
        return "\n".join(lines)
