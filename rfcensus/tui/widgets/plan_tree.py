"""Plan tree drawer — wave-by-wave execution status."""

from __future__ import annotations

from textual.app import RenderResult
from textual.widget import Widget

from rfcensus.tui.state import WaveState


class PlanTree(Widget):
    """Wave-by-wave plan with status markers.

    Marker conventions:
      ◆ — running
      ✓ — completed (all tasks ok)
      ✗ — completed (some tasks failed)
      · — pending (not yet started)
    """

    DEFAULT_CSS = """
    PlanTree {
        width: 28;
        dock: right;
        background: $panel;
        padding: 1;
        border-left: solid $surface;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._waves: list[WaveState] = []
        self._current_index: int | None = None

    def update_plan(
        self, waves: list[WaveState], current_index: int | None,
    ) -> None:
        self._waves = waves
        self._current_index = current_index
        self.refresh()

    def render(self) -> RenderResult:
        from rfcensus.tui.color import styled

        if not self._waves:
            return styled("idle", "(plan not ready)")

        lines = [styled("info", "Plan")]
        for w in self._waves:
            if w.status == "running":
                marker = styled("active", "◆")
                head = styled("active",
                              f" Wave {w.index}  ({w.task_count} task(s))")
            elif w.status == "completed":
                if w.error_count == 0:
                    marker = styled("good", "✓")
                else:
                    marker = styled("warning", "✗")
                head = styled("idle",
                              f" Wave {w.index}  "
                              f"{w.successful_count}/{w.task_count}")
            else:
                marker = styled("idle", "·")
                head = styled("idle",
                              f" Wave {w.index}  ({w.task_count} task(s))")
            lines.append(f"{marker}{head}")
            # Show task summaries indented under running/pending waves
            if w.status in ("running", "pending"):
                for s in w.task_summaries[:5]:
                    lines.append(styled("idle", f"     {s}"))
                if len(w.task_summaries) > 5:
                    lines.append(styled("idle",
                                        f"     … {len(w.task_summaries) - 5} more"))
        return "\n".join(lines)
