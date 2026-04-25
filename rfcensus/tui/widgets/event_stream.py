"""Scrolling event stream — most recent N events with severity coloring."""

from __future__ import annotations

from textual.app import RenderResult
from textual.widget import Widget

from rfcensus.tui.state import StreamEntry, filter_stream


class EventStream(Widget):
    """Tail of recent events. Filter mode determines which categories
    pass through. Severity-colored. Newest at top."""

    DEFAULT_CSS = """
    EventStream {
        background: $background;
        padding: 0 1;
        border: round $surface;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._stream: list[StreamEntry] = []
        self._filter_mode = "filtered"
        self.border_title = "Events (filtered)"

    def update_stream(
        self, stream: list[StreamEntry], filter_mode: str,
    ) -> None:
        self._stream = stream
        self._filter_mode = filter_mode
        self.border_title = f"Events ({filter_mode})"
        self.refresh()

    def render(self) -> RenderResult:
        from rfcensus.tui.color import style_for_severity, styled

        filtered = filter_stream(self._stream, self._filter_mode)
        if not filtered:
            return styled("idle", "(no events yet)")
        max_lines = max(2, self.size.height - 2)
        out = []
        for entry in filtered[:max_lines]:
            ts = entry.timestamp.strftime("%H:%M:%S")
            sev_style = style_for_severity(entry.severity)
            head = styled("idle", ts)
            body = styled(sev_style, entry.text)
            out.append(f"{head}  {body}")
        return "\n".join(out)
