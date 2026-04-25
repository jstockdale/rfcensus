"""Bottom footer — counters and key-binding hint."""

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

    HINT = (
        "1-0 dongle  ←→ focus  Enter detail  "
        "f filter  t plan-tree  l log-mode  s snapshot  p pause  q quit  ? help"
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

        line1 = f"{dongles}  ·  {wave}  ·  {decodes}  ·  {emitters}  ·  {detections}  ·  {filt}"
        return f"{line1}\n{self.HINT}"
