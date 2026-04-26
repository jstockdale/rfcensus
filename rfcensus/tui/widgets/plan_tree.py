"""Plan tree drawer — wave-by-wave execution status.

v0.6.17 redesign of the marker vocabulary:

  Wave-level (left margin):
    ◆ — running (yellow active dot)
    ✓ — completed, all tasks ok (green checkmark)
    ⚠ — completed, some tasks failed (yellow warning sign)
    ✗ — completed, all tasks failed (red X)
    · — pending (grey)

  Task-level (right margin, on each task line):
    ✓ — task succeeded
    ✗ — task failed/crashed/timed out
    ⌀ — task skipped
    ◆ — task running
      (none) — task pending

The "yellow X" in v0.6.16 was ambiguous — it meant "wave completed
with some errors" but didn't tell you WHICH task failed; the user
had to scroll the event log to find out. v0.6.17 keeps the wave
marker but ALSO renders per-task glyphs so the failed task pops out
visually. Wave-level glyph degrades gracefully to ✗ only when ALL
tasks failed, and to ⚠ when SOME failed — distinguishing partial
vs total failure.

We also show task summaries for ALL waves (not just the running and
most-recent-completed). This is what the user wanted — to see at a
glance which tasks completed across the whole plan. The drawer can
get tall but the tradeoff is worth it; collapsing back to "current
wave only" is just a feature flag away if it becomes a problem.
"""

from __future__ import annotations

from textual.app import RenderResult
from textual.widget import Widget

from rfcensus.tui.state import WaveState


# v0.6.14: bumped from 28 to 36. At 28 chars, lines like
# "interlogix_security→rtlsdr-00000043" wrap at "rtlsdr-" leaving
# the meaningful suffix on its own line, which makes the column look
# broken. 36 fits "<band>→…<4-digit-suffix>" cleanly without horizontal
# scroll.
_PLAN_TREE_WIDTH = 36

# Max visible chars per task line BEFORE we abbreviate the dongle id.
# 4 chars indent + arrow + dongle takes about 30 in the typical case;
# we strip the "rtlsdr-" prefix and keep the last 4 hex chars of the
# serial since that's what's distinctive across the user's 5 dongles.
_MAX_TASK_LINE_LEN = _PLAN_TREE_WIDTH - 5  # 5 = padding + bullet


def _abbreviate_task_summary(s: str) -> str:
    """Shorten a "band→dongle-id" task summary so it fits the column.

    Example: "interlogix_security→rtlsdr-00000043"
          →  "interlogix_security→…0043"

    If the band name itself is longer than the budget, leave it alone
    and accept the wrap — the band is the user's primary signal; the
    dongle id is secondary identification.
    """
    if "→" not in s or len(s) <= _MAX_TASK_LINE_LEN:
        return s
    band, dongle = s.split("→", 1)
    # Strip common driver prefixes; keep the last 4 chars of the
    # serial which are the meaningful disambiguator.
    for prefix in ("rtlsdr-", "hackrf-", "airspy-"):
        if dongle.startswith(prefix):
            dongle = dongle[len(prefix):]
            break
    if len(dongle) > 4:
        dongle = "…" + dongle[-4:]
    abbreviated = f"{band}→{dongle}"
    return abbreviated


class PlanTree(Widget):
    """Wave-by-wave plan with status markers.

    Marker conventions:
      ◆ — running
      ✓ — completed (all tasks ok)
      ✗ — completed (some tasks failed)
      · — pending (not yet started)
    """

    DEFAULT_CSS = f"""
    PlanTree {{
        width: {_PLAN_TREE_WIDTH};
        dock: right;
        background: $panel;
        padding: 1;
        border-left: solid $surface;
    }}
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
        for i, w in enumerate(self._waves):
            # Wave-level glyph distinguishes partial vs total failure
            # (v0.6.17). Previously yellow ✗ was used for any non-clean
            # completion, which couldn't tell the user whether 1 task
            # failed or all of them did.
            if w.status == "running":
                marker = styled("active", "◆")
                head = styled("active",
                              f" Wave {w.index}  ({w.task_count} task(s))")
            elif w.status == "completed":
                if w.error_count == 0:
                    marker = styled("good", "✓")
                elif w.error_count >= w.task_count:
                    marker = styled("error", "✗")
                else:
                    marker = styled("warning", "⚠")
                head = styled("idle",
                              f" Wave {w.index}  "
                              f"{w.successful_count}/{w.task_count}")
            else:
                marker = styled("idle", "·")
                head = styled("idle",
                              f" Wave {w.index}  ({w.task_count} task(s))")
            lines.append(f"{marker}{head}")

            # v0.6.17: show task summaries for EVERY wave with
            # per-task status glyphs. The previous "only show running
            # + last-completed" heuristic hid which task within a
            # failed earlier wave was the failure, forcing the user
            # to scroll the event log to find out. Trade-off: the
            # drawer gets taller; users can press `t` to hide it
            # entirely if it's in the way.
            for j, s in enumerate(w.task_summaries[:5]):
                abbreviated = _abbreviate_task_summary(s)
                # Per-task status glyph — colored to match the marker
                # vocabulary above. Pending tasks get no glyph (just
                # whitespace) so the running/completed ones stand out.
                status = (
                    w.task_statuses[j]
                    if j < len(w.task_statuses) else "pending"
                )
                glyph_char, glyph_style = _task_glyph(status)
                glyph = styled(glyph_style, glyph_char) if glyph_char else " "
                lines.append(
                    styled("idle", f"   {glyph} ") + abbreviated
                )
            if len(w.task_summaries) > 5:
                lines.append(styled("idle",
                                    f"     … {len(w.task_summaries) - 5} more"))
        return "\n".join(lines)


def _task_glyph(status: str) -> tuple[str, str]:
    """Return (char, color-style-name) for a per-task status glyph.

    Returns ("", "") for pending — pending tasks render with no glyph
    so the eye lands on the active/completed ones."""
    return {
        "ok":      ("✓", "good"),
        "running": ("◆", "active"),
        "failed":  ("✗", "warning"),
        "crashed": ("✗", "error"),
        "timeout": ("⏱", "warning"),
        "skipped": ("⌀", "idle"),
    }.get(status, ("", ""))
