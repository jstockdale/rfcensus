"""Top header bar — site name, session id, elapsed time."""

from __future__ import annotations

from datetime import datetime, timezone

from textual.app import RenderResult
from textual.reactive import reactive
from textual.widget import Widget


class HeaderBar(Widget):
    """One-line header at the top of the dashboard.

    Shows: site name, session id, elapsed time, paused indicator.
    Refreshed every second by the app's tick timer.
    """

    DEFAULT_CSS = """
    HeaderBar {
        height: 1;
        dock: top;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    """

    elapsed_s: reactive[float] = reactive(0.0)

    def __init__(self, *, site_name: str = "", session_id: int | None = None,
                 paused: bool = False) -> None:
        super().__init__()
        self.site_name = site_name
        self.session_id = session_id
        self._paused = paused

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self.refresh()

    def render(self) -> RenderResult:
        from rfcensus.tui.color import styled

        elapsed = _format_duration(self.elapsed_s)
        sid = f"#{self.session_id}" if self.session_id is not None else "—"
        site = self.site_name or "default"

        left = f"rfcensus {sid}  ·  {site}  ·  elapsed {elapsed}"
        if self._paused:
            right = styled("warning", "[PAUSED]")
        else:
            right = styled("active", "● running")

        # Right-pad left so right hugs the right edge
        width = max(40, self.size.width - 1)
        pad = max(0, width - _visible_len(left) - _visible_len_markup(right))
        return f"{left}{' ' * pad}{right}"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def _visible_len(text: str) -> int:
    return len(text)


def _visible_len_markup(text: str) -> int:
    """Approximate visible width when the string contains rich markup."""
    # Strip [...] tags
    out = []
    in_tag = False
    for c in text:
        if c == "[":
            in_tag = True
        elif c == "]":
            in_tag = False
        elif not in_tag:
            out.append(c)
    return len("".join(out))
