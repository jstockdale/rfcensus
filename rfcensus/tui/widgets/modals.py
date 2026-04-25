"""Modal screens — help overlay, quit-confirm, snapshot report."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static


HELP_TEXT = """\
[bold]Key bindings[/]

  [bold]1-0[/]       open dongle 1-10 detail (0 = slot 10)
  [bold]←/→[/]       move focus along the dongle strip
  [bold]Enter[/]     open focused dongle's detail
  [bold]f[/]         cycle event-stream filter (filtered/verbose/minimal)
  [bold]t[/]         toggle plan-tree drawer
  [bold]l[/]         toggle TUI / log mode
  [bold]p[/]         pause / resume scan
  [bold]s[/]         snapshot report (current state)
  [bold]q[/]         end the session (with confirmation)
  [bold]?[/]         this help

[bold]Detail screen[/]

  [bold]←/→[/]       previous / next dongle
  [bold]↑/↓[/]       scroll
  [bold]Esc[/]       back to strip

[dim]rfcensus 0.6.5 — press ? or Esc to close[/]
"""


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
            yield Static(HELP_TEXT, id="help-text")

    def action_dismiss(self) -> None:
        self.dismiss()


class ConfirmQuit(ModalScreen[bool]):
    """End-session confirmation. Returns True if user confirmed."""

    DEFAULT_CSS = """
    ConfirmQuit {
        align: center middle;
    }
    ConfirmQuit > Container {
        width: 50;
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
        ("y", "confirm", "yes"),
        ("n", "cancel", "no"),
    ]

    def compose(self) -> ComposeResult:
        with Container():
            yield Static(
                "End the running session?\n\n"
                "[dim]Pinned dongles will continue until you stop them "
                "separately. y / n / Esc[/]",
            )
            yield Button("Yes (y)", id="yes", variant="warning")
            yield Button("No (n)", id="no", variant="primary")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class ReportModal(ModalScreen):
    """Scrollable snapshot of the current scan state."""

    DEFAULT_CSS = """
    ReportModal {
        align: center middle;
    }
    ReportModal > Container {
        width: 90;
        height: 30;
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
