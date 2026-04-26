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
        # v0.6.17: process resource indicators promoted from the
        # FooterBar to here. The footer was visually crowded — proc
        # stats sat at the right end of a long counters line and
        # blended into the dim hint row below them. Up here in the
        # header (always visible, single line, no competition for the
        # eye) they're glanceable.
        self._cpu_str = "—%"
        self._rss_str = "— MB"
        # v0.7.6: when the user picks graceful-quit, the TUI stays
        # alive while the runner finishes the current wave. We
        # animate a small spinner in the header so the user knows
        # their action was acknowledged and the system is winding
        # down (not stuck). Spinner advances on every tick (1 Hz).
        self._shutting_down = False
        self._spinner_idx = 0

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self.refresh()

    def set_shutting_down(self, shutting_down: bool) -> None:
        """v0.7.6: toggle the graceful-shutdown spinner. Called by
        the app when ``state.shutting_down`` flips True after the
        user picks graceful-quit. Stays on until the TUI itself
        exits (which happens when the runner emits SessionEvent
        kind=ended)."""
        if self._shutting_down == shutting_down:
            return
        self._shutting_down = shutting_down
        self._spinner_idx = 0
        self.refresh()

    def set_proc_stats(self, cpu_str: str, rss_str: str) -> None:
        """v0.6.17: update CPU/RSS indicators. Pre-formatted by the
        ProcSampler so the header widget stays free of formatting
        policy. No-op when values haven't changed (avoids per-tick
        refresh churn)."""
        if cpu_str == self._cpu_str and rss_str == self._rss_str:
            return
        self._cpu_str = cpu_str
        self._rss_str = rss_str
        self.refresh()

    def render(self) -> RenderResult:
        from rfcensus import __version__
        from rfcensus.tui.color import styled

        elapsed = _format_duration(self.elapsed_s)
        sid = f"#{self.session_id}" if self.session_id is not None else "—"
        site = self.site_name or "default"

        # v0.6.13: include version in the program-identification chunk
        # on the left. Top-right is reserved for the running/paused
        # status indicator. Putting version next to "rfcensus" is the
        # natural reading order — it's part of what program/build is
        # active, alongside the session id and elapsed time.
        left = (
            f"rfcensus v{__version__} {sid}  ·  {site}  "
            f"·  elapsed {elapsed}"
        )
        # v0.6.17: proc stats are styled (dim) and sit between the
        # left identification chunk and the right status indicator.
        # Use a separator dot for visual grouping.
        proc = styled("idle",
                      f"cpu {self._cpu_str} · rss {self._rss_str}")
        # v0.7.6: right-side status indicator. Three states:
        #   • shutting_down — animated spinner + "shutting down…"
        #     This wins over paused/running because it's the most
        #     time-critical user-actionable state (Ctrl+Q escapes).
        #   • paused        — [PAUSED]
        #   • normal        — ● running
        if self._shutting_down:
            # Braille spinner — 8 frames, advances on each render
            # (which gets called by the 1Hz tick). Looks like a
            # tiny rotating dot pattern, very easy on the eye.
            spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            frame = spinner_frames[
                self._spinner_idx % len(spinner_frames)
            ]
            self._spinner_idx += 1
            right = styled(
                "warning",
                f"{frame} shutting down… (Ctrl+Q to force)",
            )
        elif self._paused:
            right = styled("warning", "[PAUSED]")
        else:
            right = styled("active", "● running")

        # Right-pad: proc sits in the middle, right-aligned status on
        # the far edge. Compute spacing so right hugs the edge and
        # proc has at least 4 spaces of gap on both sides.
        width = max(40, self.size.width - 1)
        right_w = _visible_len_markup(right)
        proc_w = _visible_len_markup(proc)
        left_w = _visible_len(left)
        # gap1 (left→proc), gap2 (proc→right). Distribute remaining
        # space; if it doesn't fit, push proc out (it's lower priority
        # than session id and status).
        remaining = width - left_w - proc_w - right_w
        if remaining < 8:
            # No room for proc — just left + right
            pad = max(0, width - left_w - right_w)
            return f"{left}{' ' * pad}{right}"
        gap1 = remaining // 2
        gap2 = remaining - gap1
        return f"{left}{' ' * gap1}{proc}{' ' * gap2}{right}"


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
