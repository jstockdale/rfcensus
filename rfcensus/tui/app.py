"""Textual application — top-level dashboard.

Subscribes to the EventBus once, pumps every event through the
state reducer, and triggers widget refreshes after each batch.

Lifecycle:

  app = TUIApp(runner=session_runner, no_color=False)
  await app.run_async()  # blocks while UI is up

Key bindings live on the App; modal screens have their own bindings
that take precedence while open (Textual's ScreenStack handles this).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal

from rfcensus.events import (
    ActiveChannelEvent,
    DecodeEvent,
    DecoderFailureEvent,
    DetectionEvent,
    EmitterEvent,
    Event,
    FanoutClientEvent,
    HardwareEvent,
    PlanReadyEvent,
    SessionEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
    WaveCompletedEvent,
    WaveStartedEvent,
)
from rfcensus.tui.color import configure_color
from rfcensus.tui.state import (
    TUIState,
    cycle_filter_mode,
    reduce,
)
from rfcensus.tui.widgets.dongle_detail import DongleDetail
from rfcensus.tui.widgets.dongle_strip import DongleStrip
from rfcensus.tui.widgets.event_stream import EventStream
from rfcensus.tui.widgets.footer import FooterBar
from rfcensus.tui.widgets.header import HeaderBar
from rfcensus.tui.widgets.modals import (
    ConfirmQuit,
    HelpOverlay,
    ReportModal,
)
from rfcensus.tui.widgets.plan_tree import PlanTree
from rfcensus.utils.logging import get_logger

if TYPE_CHECKING:
    from rfcensus.engine.session import SessionRunner

log = get_logger(__name__)


# Minimum terminal size to launch the TUI. Below this we refuse and
# fall back to log mode.
MIN_COLS = 80
MIN_ROWS = 24


# Threshold at which the plan-tree drawer is shown by default.
# Below this, it stays hidden until user toggles with `t`.
PLAN_TREE_AUTO_SHOW_COLS = 100


# Event types we subscribe to. Order doesn't matter; reducer dispatches
# by type.
SUBSCRIBED_EVENT_TYPES = (
    SessionEvent,
    PlanReadyEvent,
    WaveStartedEvent,
    WaveCompletedEvent,
    TaskStartedEvent,
    TaskCompletedEvent,
    HardwareEvent,
    FanoutClientEvent,
    DecodeEvent,
    EmitterEvent,
    DetectionEvent,
    DecoderFailureEvent,
    ActiveChannelEvent,
)


class TUIApp(App):
    """The dashboard."""

    CSS = """
    #main {
        height: 1fr;
    }
    #strip-row {
        height: 7;
    }
    #content-row {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("question_mark,slash", "help", "help"),
        Binding("escape", "escape", "back / cancel"),
        Binding("f", "cycle_filter", "filter"),
        Binding("t", "toggle_plan_tree", "plan tree"),
        Binding("p", "pause_resume", "pause"),
        Binding("q", "quit_session", "quit"),
        Binding("s", "snapshot", "snapshot report"),
        Binding("l", "toggle_log_mode", "log mode"),
        Binding("left", "focus_prev", "focus prev"),
        Binding("right", "focus_next", "focus next"),
        Binding("enter", "open_detail", "detail"),
        # Number keys: 1-9 + 0 for slot 10
        *[
            Binding(str(i), f"open_slot_{i}", f"slot {i}", show=False)
            for i in range(1, 10)
        ],
        Binding("0", "open_slot_0", "slot 10", show=False),
    ]

    def __init__(
        self,
        runner: "SessionRunner | None" = None,
        *,
        no_color: bool = False,
        site_name: str = "default",
    ) -> None:
        super().__init__()
        self.runner = runner
        self.state = TUIState(site_name=site_name)
        self._no_color = no_color
        self._event_subs = []
        self._detail_active = False
        self._tick_task: asyncio.Task | None = None
        self._log_mode_requested = False

        # Color setup (must happen before any widget renders)
        configure_color(not no_color)

    # ── Lifecycle ──────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Container(id="main"):
            yield HeaderBar(site_name=self.state.site_name)
            yield DongleStrip()
            with Horizontal(id="content-row"):
                yield EventStream()
                yield PlanTree()
            yield FooterBar()

    async def on_mount(self) -> None:
        # Subscribe to the bus
        if self.runner is not None:
            for evt_type in SUBSCRIBED_EVENT_TYPES:
                sub = self.runner.event_bus.subscribe(
                    evt_type, self._on_event,
                )
                self._event_subs.append(sub)

        # Hide plan tree if terminal is narrow
        if self.size.width < PLAN_TREE_AUTO_SHOW_COLS:
            self.state.plan_tree_visible = False
            try:
                self.query_one(PlanTree).display = False
            except Exception:
                pass

        # Start tick timer for elapsed-time updates
        self._tick_task = asyncio.create_task(self._tick_loop())

        # First render
        self._refresh_all()

    async def on_unmount(self) -> None:
        for sub in self._event_subs:
            try:
                sub.unsubscribe()
            except Exception:
                pass
        if self._tick_task is not None:
            self._tick_task.cancel()

    # ── Event ingestion ────────────────────────────────────────────

    async def _on_event(self, event: Event) -> None:
        """Bus subscriber. Updates state then triggers refresh."""
        try:
            reduce(self.state, event)
            self._refresh_all()
        except Exception:
            log.exception("TUI reducer/refresh failed for event %s", event)

    # ── Tick loop ──────────────────────────────────────────────────

    async def _tick_loop(self) -> None:
        """Once per second, refresh the header/footer for elapsed
        time. Other widgets refresh on event arrival."""
        try:
            while True:
                await asyncio.sleep(1.0)
                self._refresh_dynamic()
        except asyncio.CancelledError:
            pass

    # ── Refresh helpers ────────────────────────────────────────────

    def _refresh_all(self) -> None:
        try:
            strip = self.query_one(DongleStrip)
            strip.update_dongles(self.state.dongles)
            strip.set_focused_index(self.state.focused_dongle_index)
        except Exception:
            pass
        try:
            self.query_one(EventStream).update_stream(
                self.state.stream, self.state.filter_mode,
            )
        except Exception:
            pass
        try:
            tree = self.query_one(PlanTree)
            tree.update_plan(self.state.waves, self.state.current_wave_index)
            tree.display = self.state.plan_tree_visible
        except Exception:
            pass
        self._refresh_dynamic()

    def _refresh_dynamic(self) -> None:
        """Header (elapsed) + footer (counters). Cheap, called once
        per second by the tick loop and after every event."""
        try:
            header = self.query_one(HeaderBar)
            if self.state.session_started_at is not None:
                elapsed = (
                    datetime.now(timezone.utc) - self.state.session_started_at
                ).total_seconds()
                # Subtract paused time
                elapsed -= self.state.paused_total_s
                if self.runner is not None:
                    elapsed -= self.runner.control.effective_total_paused_s()
                header.elapsed_s = max(0.0, elapsed)
            header.set_paused(self._is_paused())
        except Exception:
            pass
        try:
            footer = self.query_one(FooterBar)
            healthy = sum(
                1 for d in self.state.dongles
                if d.status in ("active", "idle")
            )
            wave_label = (
                f"{self.state.current_wave_index}"
                if self.state.current_wave_index is not None else "—"
            )
            footer.set_state(
                healthy=healthy, total=len(self.state.dongles),
                decodes=self.state.total_decodes,
                emitters=self.state.total_emitters_confirmed,
                detections=self.state.total_detections,
                wave_label=wave_label,
                filter_mode=self.state.filter_mode,
            )
        except Exception:
            pass

    def _is_paused(self) -> bool:
        if self.runner is not None:
            return self.runner.control.is_paused()
        return False

    # ── Action handlers ────────────────────────────────────────────

    def action_help(self) -> None:
        self.push_screen(HelpOverlay())

    def action_escape(self) -> None:
        # Top-level Esc is a no-op; modal screens handle their own Esc
        # via their own bindings.
        pass

    def action_cycle_filter(self) -> None:
        self.state.filter_mode = cycle_filter_mode(self.state.filter_mode)
        self._refresh_all()

    def action_toggle_plan_tree(self) -> None:
        self.state.plan_tree_visible = not self.state.plan_tree_visible
        try:
            self.query_one(PlanTree).display = self.state.plan_tree_visible
        except Exception:
            pass

    def action_pause_resume(self) -> None:
        if self.runner is None:
            return
        async def _toggle():
            if self.runner.control.is_paused():
                await self.runner.resume_session()
            else:
                await self.runner.pause_session()
            self._refresh_dynamic()
        asyncio.create_task(_toggle())

    def action_quit_session(self) -> None:
        async def _confirm_then_quit():
            confirmed = await self.push_screen_wait(ConfirmQuit())
            if confirmed:
                if self.runner is not None:
                    self.runner.request_stop()
                # Print 73 satellite as the final TUI message; the
                # session-end report prints from the CLI side after exit.
                self.exit()
        asyncio.create_task(_confirm_then_quit())

    def action_snapshot(self) -> None:
        text = self._render_snapshot_report()
        self.push_screen(ReportModal(text))

    def action_toggle_log_mode(self) -> None:
        # In v0.6.5 we just exit — the CLI sees us exit and resumes
        # log streaming. Re-entering TUI would require relaunching the
        # command, which is fine for v0.6.5.
        self._log_mode_requested = True
        self.exit()

    def action_focus_prev(self) -> None:
        if self.state.focused_dongle_index > 0:
            self.state.focused_dongle_index -= 1
            self._refresh_all()

    def action_focus_next(self) -> None:
        if self.state.focused_dongle_index < len(self.state.dongles) - 1:
            self.state.focused_dongle_index += 1
            self._refresh_all()

    def action_open_detail(self) -> None:
        if not self.state.dongles:
            return
        self.push_screen(DongleDetail(
            self.state.dongles,
            self.state.focused_dongle_index,
            self.state.stream,
        ))

    # Generated number-key actions
    def _open_slot(self, slot: int) -> None:
        idx = slot - 1
        if 0 <= idx < len(self.state.dongles):
            self.state.focused_dongle_index = idx
            self.action_open_detail()

    def action_open_slot_1(self): self._open_slot(1)
    def action_open_slot_2(self): self._open_slot(2)
    def action_open_slot_3(self): self._open_slot(3)
    def action_open_slot_4(self): self._open_slot(4)
    def action_open_slot_5(self): self._open_slot(5)
    def action_open_slot_6(self): self._open_slot(6)
    def action_open_slot_7(self): self._open_slot(7)
    def action_open_slot_8(self): self._open_slot(8)
    def action_open_slot_9(self): self._open_slot(9)
    def action_open_slot_0(self): self._open_slot(10)  # 0 = slot 10

    # ── Snapshot report ────────────────────────────────────────────

    def _render_snapshot_report(self) -> str:
        """Render a brief snapshot of current state. Different from
        the full end-of-session report (which lives in the report
        renderer); this is meant for in-flight peeking."""
        lines = []
        sid = self.state.session_id or "—"
        lines.append(f"Session #{sid} — {self.state.site_name}")
        if self.state.session_started_at:
            elapsed = (
                datetime.now(timezone.utc) - self.state.session_started_at
            ).total_seconds()
            lines.append(f"Elapsed: {int(elapsed)}s "
                         f"(paused: {int(self.state.paused_total_s)}s)")
        lines.append("")
        lines.append(f"Plan: {len(self.state.waves)} wave(s), "
                     f"{self.state.completed_tasks}/{self.state.total_tasks} tasks done")
        lines.append("")
        lines.append("Dongles:")
        for i, d in enumerate(self.state.dongles, start=1):
            slot = "0" if i == 10 else str(i)
            consumer = f" — {d.consumer}" if d.consumer else ""
            band = f" [{d.band_id}]" if d.band_id else ""
            lines.append(
                f"  [{slot}] {d.dongle_id}  "
                f"{d.status}{consumer}{band}"
            )
        lines.append("")
        lines.append(f"Counters: decodes={self.state.total_decodes}, "
                     f"emitters={self.state.total_emitters_confirmed}, "
                     f"detections={self.state.total_detections}")
        lines.append("")
        lines.append("Recent events:")
        for e in self.state.stream[:15]:
            ts = e.timestamp.strftime("%H:%M:%S")
            lines.append(f"  {ts}  [{e.severity}] {e.text}")
        return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# Sizing check — used by CLI before launching the app
# ────────────────────────────────────────────────────────────────────


def check_tty_and_size() -> tuple[bool, str]:
    """Return (ok, message). If not ok, the CLI should fall back to
    log mode and print the message."""
    import os
    import sys

    if not sys.stdout.isatty():
        return False, "stdout is not a TTY (running under a pipe or CI?)"
    try:
        size = os.get_terminal_size()
    except OSError:
        return False, "could not determine terminal size"
    if size.columns < MIN_COLS or size.lines < MIN_ROWS:
        return False, (
            f"terminal too small ({size.columns}x{size.lines}); "
            f"need at least {MIN_COLS}x{MIN_ROWS}"
        )
    return True, ""
