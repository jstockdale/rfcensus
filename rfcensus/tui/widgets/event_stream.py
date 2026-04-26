"""Scrolling event stream — like `tail -f`. Oldest at top, newest at
bottom. Live tail = bottom of pane.

v0.6.16:
  • Reads from per-mode ring buffers (state.streams[filter_mode])
    rather than filtering a single shared buffer at render time. The
    upstream change in state.py keeps each mode's buffer at 5000
    entries so verbose 'channel' spam (rtl_power, 5-30/sec) doesn't
    push older 'wave' / 'emitter' / 'detection' lines off the
    'filtered' or 'minimal' views.
  • Renders chronologically (oldest top, newest bottom). When new
    entries arrive at live tail, the view scrolls UP visually as
    older entries disappear off the top — same behavior as `tail -f`
    or any normal log viewer. The previous newest-at-top model was
    confusing and made it impossible to read history naturally.
  • Scroll keys remain intuitive:
      ↑ / PgUp  = older (scroll viewport up the buffer)
      ↓ / PgDn  = newer (scroll viewport down toward live)
      Home      = jump to OLDEST entry (top of buffer)
      End       = snap to LIVE TAIL (newest, bottom)

Scroll-offset semantics: offset = N means the LAST visible line shows
the (N + max_lines)-th most recent entry. offset = 0 means the last
visible line is the newest entry (live tail). When new entries arrive
while offset > 0, we increment offset by the same amount so the user's
viewport stays anchored on the same logical entry.
"""

from __future__ import annotations

from textual.app import RenderResult
from textual.widget import Widget

from rfcensus.tui.state import StreamEntry


class EventStream(Widget):
    """Tail of recent events. Filter mode determines which buffer
    is rendered. Severity-colored. Live tail at the bottom."""

    DEFAULT_CSS = """
    EventStream {
        background: $background;
        padding: 0 1;
        border: round $surface;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        # We hold a per-mode dict reference. update_stream replaces it
        # with the latest from state. Empty dict on init — defensive
        # against early renders before any data arrives.
        self._streams: dict[str, list[StreamEntry]] = {
            "minimal": [], "filtered": [], "verbose": [],
        }
        self._filter_mode = "filtered"
        # Number of entries between the LAST visible line and the
        # live tail (newest entry). 0 = at live tail; >0 = scrolled
        # back into history. The bottom line of the viewport always
        # shows the entry at index (len - 1 - offset).
        self._scroll_offset = 0
        # Tracks how many new entries arrived while the user was NOT
        # at live tail. Reset to 0 on snap-to-live.
        self._unread_below = 0
        self.border_title = "Events (filtered)"

    # ── State sync from the app ──────────────────────────────────

    def update_stream(
        self,
        streams: "dict[str, list[StreamEntry]] | list[StreamEntry]",
        filter_mode: str,
    ) -> None:
        """Replace the per-mode buffer dict + filter mode. Called once
        per tick + on every event. Maintains the user's scroll position
        when they're reading history.

        v0.6.16 API takes a dict keyed by filter mode. Legacy callers
        (tests pre-dating per-mode buffers) pass a flat list of all
        entries; we treat that as a single shared buffer that we filter
        per-render.
        """
        # Coerce legacy list-input to dict shape. The legacy list is
        # the unfiltered stream in newest-first order (v0.6.5..v0.6.15
        # convention), so we reverse and apply per-mode filtering to
        # build the dict.
        if isinstance(streams, list):
            from rfcensus.tui.state import filter_stream
            chrono = list(reversed(streams))
            streams = {
                mode: filter_stream(chrono, mode)
                for mode in ("minimal", "filtered", "verbose")
            }
        # Compute new entries since last update, in the CURRENT filter
        # mode's buffer. If filter mode changed, scroll math doesn't
        # translate cleanly, so snap to live tail.
        if filter_mode != self._filter_mode:
            self._scroll_offset = 0
            self._unread_below = 0
        else:
            old_len = len(self._streams.get(filter_mode, []))
            new_len = len(streams.get(filter_mode, []))
            added = max(0, new_len - old_len)
            if self._scroll_offset > 0 and added > 0:
                # User is in history mode. Anchor on the same entry by
                # advancing offset by the number of new entries (since
                # each new entry pushes everything 1 position further
                # back from live).
                max_off = max(0, new_len - 1)
                self._scroll_offset = min(
                    self._scroll_offset + added, max_off,
                )
                self._unread_below += added
        self._streams = streams
        self._filter_mode = filter_mode
        self._refresh_title()
        self.refresh()

    # ── Scroll actions (driven by app bindings) ──────────────────

    def scroll_lines(self, n: int) -> None:
        """Scroll by n lines. Positive n = older (toward top of buffer).
        Negative n = newer (toward live tail). Symmetric with the
        common reading direction: pressing Up = see older = move up
        the buffer = increase offset.

        v0.7.4: cap max_offset such that the viewport stays FULL at
        the top of the buffer instead of collapsing to a single line
        as offset → len(buf)-1. The previous cap let the user scroll
        past the point where the top entry [0] reaches the top of the
        viewport, hiding everything below; the new cap clamps offset
        at (len(buf) - max_lines) so the oldest entry is the topmost
        visible line and no further scroll-up moves anything.

        When offset hits 0 (live tail), the unread counter resets."""
        buf = self._streams.get(self._filter_mode, [])
        max_lines = self._max_visible_lines()
        max_offset = max(0, len(buf) - max_lines)
        new_offset = max(0, min(self._scroll_offset + n, max_offset))
        if new_offset == self._scroll_offset:
            return
        self._scroll_offset = new_offset
        if self._scroll_offset == 0:
            self._unread_below = 0
        self._refresh_title()
        self.refresh()

    def scroll_page(self, direction: int) -> None:
        """Scroll by one visible-page worth of lines. direction = +1
        for older (page up), -1 for newer (page down)."""
        page = max(1, self.size.height - 2)
        self.scroll_lines(direction * page)

    def scroll_to_live(self) -> None:
        """Snap to the live tail (newest entry visible at bottom).
        Bound to End key by the app."""
        if self._scroll_offset == 0 and self._unread_below == 0:
            return
        self._scroll_offset = 0
        self._unread_below = 0
        self._refresh_title()
        self.refresh()

    def scroll_to_oldest(self) -> None:
        """Jump to the OLDEST entry in the current mode's buffer.

        v0.7.4: previously set offset=len-1 which put entry[0] at the
        BOTTOM of the viewport with everything else hidden. The Home
        key should put entry[0] at the TOP with the viewport still
        full of entries — which means offset = len - max_lines."""
        buf = self._streams.get(self._filter_mode, [])
        max_lines = self._max_visible_lines()
        new_offset = max(0, len(buf) - max_lines)
        if new_offset == self._scroll_offset:
            return
        self._scroll_offset = new_offset
        self._refresh_title()
        self.refresh()

    def _max_visible_lines(self) -> int:
        """How many entries fit in the viewport. Subtracts 2 for the
        round border. Floors at 2 so the slice math never breaks even
        on a degenerate 1-row terminal."""
        try:
            return max(2, self.size.height - 2)
        except Exception:
            return 2

    # Backward-compat alias — pre-v0.6.16 the method was scroll_to_top
    # which used to mean "snap to live tail" because newest was at top.
    # Now newest is at BOTTOM so the same intent maps to scroll_to_live.
    def scroll_to_top(self) -> None:
        self.scroll_to_live()

    # ── Internals ───────────────────────────────────────────────

    def _refresh_title(self) -> None:
        buf = self._streams.get(self._filter_mode, [])
        title = f"Events ({self._filter_mode})  [{len(buf)}]"
        if self._scroll_offset > 0:
            # Indicate we're in history + how many new entries arrived
            # while the user was scrolled away. Press End (or scroll
            # down to offset=0) to catch up.
            title += (
                f" — scrolled (+{self._unread_below} new below, End=live)"
            )
        self.border_title = title

    def render(self) -> RenderResult:
        from rfcensus.tui.color import style_for_severity, styled

        buf = self._streams.get(self._filter_mode, [])
        if not buf:
            return styled("idle", "(no events yet)")
        max_lines = self._max_visible_lines()

        # Slice the visible window. The bottom line of the viewport
        # corresponds to entry (len - 1 - offset). The top line is
        # max_lines-1 above that.
        end = len(buf) - self._scroll_offset           # exclusive
        start = max(0, end - max_lines)
        visible = buf[start:end]

        out = []
        for entry in visible:                    # already chronological
            ts = entry.timestamp.strftime("%H:%M:%S")
            sev_style = style_for_severity(entry.severity)
            head = styled("idle", ts)
            body = styled(sev_style, entry.text)
            out.append(f"{head}  {body}")
        return "\n".join(out)
