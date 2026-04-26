"""Bottom footer — counters and key-binding hint.

v0.6.17: proc stats (cpu/rss) moved to HeaderBar — they're easier
to read up there in the always-visible single-line header than
crammed at the right end of the footer counter line. Footer is now
just counters (top row) + key-binding hint (bottom row).
"""

from __future__ import annotations

from textual.app import RenderResult
from textual.widget import Widget


class FooterBar(Widget):
    """Two-row footer: top row is live counters, bottom row is the
    key-binding hint."""

    DEFAULT_CSS = """
    FooterBar {
        height: 2;
        dock: bottom;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    """

    # v0.7.3: bound the report key to `r` instead of `s` because the
    # visible "report" label has to match the keystroke users see in
    # the footer. `s` remains as a hidden alias for muscle memory.
    # Also added `m mesh` for the meshtastic-recent pane toggle.
    HINT = (
        "No. Keys 1 to 0 dongle  ←→ focus  Enter detail  "
        "f filter  t plan-tree  l log-mode  m mesh  r report  "
        "p pause  q quit  ? help"
    )

    def __init__(self) -> None:
        super().__init__()
        self._healthy = 0
        self._total = 0
        self._decodes = 0
        self._emitters = 0
        self._detections = 0
        self._wave_label = "—"
        self._filter_mode = "filtered"

    def set_state(
        self, *,
        healthy: int, total: int,
        decodes: int, emitters: int, detections: int,
        wave_label: str, filter_mode: str,
    ) -> None:
        self._healthy = healthy
        self._total = total
        self._decodes = decodes
        self._emitters = emitters
        self._detections = detections
        self._wave_label = wave_label
        self._filter_mode = filter_mode
        self.refresh()

    # v0.7.5: kept as a no-op shim for backward compat with the
    # v0.7.4 brief experiment of putting proc stats in the footer.
    # The HeaderBar's dock-stack bug is now fixed (DongleStrip's
    # dock:top removed) so proc stats live where they were originally
    # intended — in the header. Keeping this shim avoids a crash if
    # any external caller still invokes it.
    def set_proc_stats(self, cpu_str: str, rss_str: str) -> None:
        return  # no-op; HeaderBar owns proc stats again

    def render(self) -> RenderResult:
        from rfcensus.tui.color import styled

        dongles = f"{self._healthy}/{self._total} dongles"
        decodes = f"{self._decodes} decodes"
        emitters = styled("highlight" if self._emitters else "info",
                          f"{self._emitters} emitters")
        detections = styled("highlight" if self._detections else "info",
                            f"{self._detections} detections")
        wave = f"wave {self._wave_label}"
        filt = f"filter:{self._filter_mode}"

        line1 = (
            f"{dongles}  ·  {wave}  ·  {decodes}  ·  "
            f"{emitters}  ·  {detections}  ·  {filt}"
        )
        return f"{line1}\n{self.HINT}"
