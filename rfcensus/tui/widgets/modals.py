"""Modal screens — help overlay, quit-confirm, snapshot report."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static


HELP_TEXT_TEMPLATE = """\
[bold]Key bindings[/]

  [bold]1-0[/]       jump to dongle 1-10 detail (0 = slot 10)
  [bold]←/→[/]       move cursor along the dongle strip (preview)
  [bold]Enter[/]     toggle detail pane on/off for cursor-focused dongle
  [bold]Esc[/]       close detail pane (back to events)
  [bold]f[/]         cycle event-stream filter (filtered/verbose/minimal)
  [bold]t[/]         toggle plan-tree drawer
  [bold]l[/]         toggle TUI / log mode
  [bold]p[/]         pause / resume scan
  [bold]s[/]         show in-flight report (current scan state)
  [bold]q[/]         end the session (with confirmation)
  [bold]?[/]         this help

[bold]Dongle strip visual cues[/]

  Border [bold]color[/] always shows status (green/grey/yellow/red)
  Border [bold]style[/] shows selection state:
    round  = unselected
    heavy  = cursor here (preview)
    double = detail pane is showing this one

[bold]Event stream scroll[/]

  [bold]↑/↓[/]       scroll one line (toward older / newer)
  [bold]PgUp/PgDn[/] scroll one page
  [bold]Home[/]      snap to live tail (newest)

[dim]rfcensus {version} — press ? or Esc to close[/]
"""


def _help_text() -> str:
    """v0.6.13: dynamically inject the running version into the help
    overlay. Previously hardcoded to 0.6.5 and silently went stale."""
    from rfcensus import __version__
    return HELP_TEXT_TEMPLATE.format(version=__version__)


# Backward-compat alias for any external import (none expected, but
# cheap insurance).
HELP_TEXT = HELP_TEXT_TEMPLATE


class HelpOverlay(ModalScreen):
    """Centered help modal. Esc or ? to close."""

    DEFAULT_CSS = """
    HelpOverlay {
        align: center middle;
    }
    HelpOverlay > Container {
        width: 60;
        height: auto;
        max-height: 30;
        padding: 1 2;
        background: $panel;
        border: round $accent;
    }
    """

    BINDINGS = [
        ("escape,question_mark", "dismiss", "close"),
    ]

    def compose(self) -> ComposeResult:
        with Container():
            yield Static(_help_text(), id="help-text")

    def action_dismiss(self) -> None:
        self.dismiss()


class ConfirmQuit(ModalScreen[str]):
    """End-session confirmation. Returns one of:

    - ``"graceful"`` — user pressed y/Enter. Lets the current wave
      finish, releases dongles cleanly, then shows the (partial)
      report. The "right" answer for almost all real cases.
    - ``"force"``    — user pressed f. Cancels the running wave
      immediately, in-flight subprocess decoders get SIGTERM, the
      report still renders but with whatever data was captured
      before the abort.
    - ``"cancel"``   — user pressed n / Esc. Continue the session.

    v0.7.4: added the third option ("force") and split the previous
    boolean return into a string. Previously q+y unconditionally
    HARD-cancelled the runner task mid-wave (see inventory.py
    ``log_mode_toggle`` cancellation block) which orphaned subprocess
    decoders and sometimes caused the report to never render because
    ``runner_task.result()`` raised CancelledError.
    """

    DEFAULT_CSS = """
    ConfirmQuit {
        align: center middle;
    }
    ConfirmQuit > Container {
        width: 64;
        height: auto;
        padding: 1 2;
        background: $panel;
        border: round $warning;
    }
    ConfirmQuit Button {
        margin: 1 1 0 0;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "cancel"),
        ("y", "graceful", "graceful"),
        ("enter", "graceful", "graceful"),
        ("f", "force", "force"),
        ("n", "cancel", "no"),
    ]

    def compose(self) -> ComposeResult:
        with Container():
            yield Static(
                "End the running session?\n\n"
                "[bold]y[/] / Enter — finish the current wave, then "
                "show the report (recommended)\n"
                "    The TUI stays open while the wave drains so you "
                "can watch counters tick down.\n"
                "    Look for the [bold]shutting down…[/] spinner in "
                "the top-right corner.\n\n"
                "[bold]f[/] — force quit NOW (cancels in-flight tasks; "
                "report will be incomplete)\n"
                "    Use when you don't need a clean shutdown — the "
                "report shows what made it to the DB.\n\n"
                "[bold]n[/] / Esc — keep running\n\n"
                "[dim]Pinned dongles continue running until you stop "
                "them separately.[/]\n"
                "[dim]Ctrl+Q anywhere = panic exit (no report at all).[/]",
            )
            yield Button("Finish wave & show report (y)", id="graceful",
                         variant="warning")
            yield Button("Force quit (f)", id="force", variant="error")
            yield Button("Cancel (n)", id="cancel", variant="primary")

    def action_graceful(self) -> None:
        self.dismiss("graceful")

    def action_force(self) -> None:
        self.dismiss("force")

    def action_cancel(self) -> None:
        self.dismiss("cancel")

    def on_button_pressed(self, event) -> None:
        # Map button id → return value
        self.dismiss(event.button.id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class ReportModal(ModalScreen):
    """Scrollable snapshot of the current scan state.

    v0.6.17: bumped from 90x30 to 80% of the viewport (capped at
    140x50). The richer in-flight report needs the room — per-band
    activity tables and recent-emitter lists were getting truncated
    horizontally and clipped vertically in the old size."""

    DEFAULT_CSS = """
    ReportModal {
        align: center middle;
    }
    ReportModal > Container {
        width: 80%;
        height: 80%;
        max-width: 140;
        max-height: 50;
        background: $panel;
        border: round $accent;
        padding: 1 2;
    }
    """

    BINDINGS = [
        ("escape,s", "dismiss", "close"),
    ]

    def __init__(self, report_text: str) -> None:
        super().__init__()
        self._text = report_text

    def compose(self) -> ComposeResult:
        with Container():
            with VerticalScroll():
                yield Static(self._text)

    def action_dismiss(self) -> None:
        self.dismiss()


class ScanComplete(ModalScreen[str]):
    """v0.7.5: shown when the runner finishes a planned scan
    (all waves complete, no more passes pending). Replaces the
    abrupt v0.7.4 behavior where TUI just exited and dumped the
    final report to stdout — jarring; users wondered if it crashed.

    Returns one of:
      • ``"report"`` — exit TUI immediately and show the final report
      • ``"stay"``   — stay in the TUI to inspect dongle detail,
                       event stream, etc. Final report shows when
                       user later quits with `q`.
    """

    DEFAULT_CSS = """
    ScanComplete {
        align: center middle;
    }
    ScanComplete > Container {
        width: 64;
        height: auto;
        padding: 1 2;
        background: $panel;
        border: round $success;
    }
    ScanComplete Button {
        margin: 1 1 0 0;
    }
    """

    BINDINGS = [
        # Default action is "see the report" — that's what most
        # users want next. Esc and `s` mean "stay in TUI".
        ("enter", "go_report", "report"),
        ("r", "go_report", "report"),
        ("escape", "stay", "stay"),
        ("s", "stay", "stay"),
    ]

    def __init__(self, summary_line: str) -> None:
        super().__init__()
        self._summary_line = summary_line

    def compose(self) -> ComposeResult:
        from textual.widgets import Static, Button
        with Container():
            yield Static(
                "[bold green]🎉 Scan Complete![/]\n\n"
                f"{self._summary_line}\n\n"
                "[dim]What next?[/]\n\n"
                "[bold]r[/] / Enter — close the TUI and show the "
                "full report\n"
                "[bold]s[/] / Esc   — stay in the TUI to inspect "
                "(report shows when you quit)\n",
            )
            yield Button("Show report (r)", id="report",
                         variant="success")
            yield Button("Stay in TUI (s)", id="stay",
                         variant="primary")

    def action_go_report(self) -> None:
        self.dismiss("report")

    def action_stay(self) -> None:
        self.dismiss("stay")

    def on_button_pressed(self, event) -> None:
        self.dismiss(event.button.id)
