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
from rfcensus.tui.widgets.meshtastic_recent import MeshtasticRecentWidget
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
        # v0.7.6: Ctrl+C now opens the same ConfirmQuit modal as `q`.
        # Previously SIGINT was handled by the runner's signal handler
        # which silently flipped _stop_requested — the user got no
        # visible feedback for several seconds until the wave finished
        # and the TUI exited. Now Ctrl+C is a first-class TUI action
        # that surfaces the choice dialog (graceful / force / cancel)
        # so the user always sees their keypress acknowledged.
        Binding("ctrl+c", "quit_session", "quit", show=False),
        # v0.7.6: Ctrl+Q is the panic button — immediate cancel, no
        # report, no confirmation. For users who know they want out
        # NOW and don't care about the partial report. The DB still
        # has data; user can recover via `rfcensus list decodes
        # --session N`.
        Binding("ctrl+q", "fast_quit", "fast quit", show=False),
        # v0.7.3: report binding moved s → r so the key matches the
        # label users see in the footer ("r"eport vs the old "s"napshot
        # which was confusing). Keep `s` as a hidden alias so muscle
        # memory from previous releases still works without surfacing
        # in the footer.
        Binding("r", "snapshot", "report"),
        Binding("s", "snapshot", "report (alias)", show=False),
        Binding("l", "toggle_log_mode", "log mode"),
        Binding("m", "toggle_meshtastic", "meshtastic"),
        Binding("left", "focus_prev", "focus prev"),
        Binding("right", "focus_next", "focus next"),
        Binding("enter", "open_detail", "detail"),
        # v0.7.4: Tab/Shift+Tab cycle the FOCUSED PANE (dongles ↔
        # main pane ↔ plan tree). Distinct from arrow keys which
        # navigate WITHIN the focused pane. The focused pane's border
        # title brightens so the user knows which context their keys
        # land in. Plan-tree intra-pane navigation is a v0.7.5 feature;
        # for now Tabbing to it is a visual-only cue.
        Binding("tab", "focus_next_pane", "next pane"),
        Binding("shift+tab", "focus_prev_pane", "prev pane"),
        # v0.6.13: event-stream scroll. up/down by line, PgUp/PgDn
        # by page, Home jumps to OLDEST, End snaps to live tail
        # (newest, bottom). v0.6.16 reordered the rendering to
        # chronological (oldest top, newest bottom) so Home/End now
        # match standard log-viewer convention. The stream tracks its
        # own offset and won't yank the user back to live when new
        # events arrive while they're reading history.
        Binding("up", "scroll_up", "scroll up", show=False),
        Binding("down", "scroll_down", "scroll down", show=False),
        Binding("pageup", "scroll_page_up", "page up", show=False),
        Binding("pagedown", "scroll_page_down", "page down", show=False),
        Binding("home", "scroll_to_oldest", "oldest", show=False),
        Binding("end", "scroll_to_live", "live tail", show=False),
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
        # v0.7.4: distinguishes graceful (wait for current wave) from
        # force (cancel immediately) when the user picks an exit path
        # in the ConfirmQuit modal. Read by inventory.py's TUI/runner
        # coordination layer to decide whether to await the runner or
        # cancel it on TUI exit.
        self._force_quit_requested = False
        # v0.7.6: panic-button flag set by Ctrl+Q. Coordinated with
        # _force_quit_requested (both flip together) so the CLI's
        # coordination layer treats it as force-quit-but-skip-report.
        self._fast_quit_requested = False
        # v0.7.5: ensures the scan-complete modal only fires once
        # per session. Without this, every late-arriving SessionEvent
        # (which can happen if subscribers are slow) would re-trigger
        # the modal.
        self._scan_complete_shown = False
        # v0.6.14: process-stats sampler. Reads /proc/self/stat +
        # statm once per tick (1 Hz) → ~10 µs per call. Negligible.
        from rfcensus.tui.proc_stats import ProcSampler
        self._proc_sampler = ProcSampler()

        # Color setup (must happen before any widget renders)
        configure_color(not no_color)

    # ── Lifecycle ──────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Container(id="main"):
            yield HeaderBar(site_name=self.state.site_name)
            yield DongleStrip()
            with Horizontal(id="content-row"):
                # v0.6.14: both EventStream and DongleDetail live in
                # the main pane simultaneously; visibility is toggled
                # by `state.main_pane_mode`. Replaces the old
                # push_screen modal which stacked when the user
                # pressed multiple number keys.
                #
                # v0.7.2: MeshtasticRecentWidget joins as a third
                # mode ("meshtastic") accessible via `m`, surfacing
                # the rolling Meshtastic-decode tail next to the
                # generic event stream + per-dongle detail panes.
                #
                # v0.7.3: explicit initial display state for non-
                # default panes prevents a first-frame race where
                # all four panes briefly render before _refresh_all
                # hides three of them. Default mode is "events", so
                # only EventStream starts visible.
                yield EventStream()
                dd = DongleDetail()
                dd.display = False
                yield dd
                yield PlanTree()    # has its own visibility toggle (`t`)
                mw = MeshtasticRecentWidget()
                mw.display = False
                yield mw
            yield FooterBar()

    async def on_mount(self) -> None:
        # Subscribe to the bus
        if self.runner is not None:
            for evt_type in SUBSCRIBED_EVENT_TYPES:
                sub = self.runner.event_bus.subscribe(
                    evt_type, self._on_event,
                )
                self._event_subs.append(sub)

        # v0.6.17: seed display-only metadata (antenna id, model) from
        # the static config so the dongle detail panel can show
        # antenna info even before the first hardware event arrives
        # for that dongle. The data is config-defined and never
        # changes during a session.
        if self.runner is not None:
            try:
                from rfcensus.tui.state import seed_dongle_metadata
                for dcfg in self.runner.config.dongles:
                    seed_dongle_metadata(
                        self.state, dcfg.id,
                        model=dcfg.model,
                        antenna_id=dcfg.antenna,
                    )
            except Exception:
                # Not critical — detail pane will just show "—" for
                # antenna if we can't read the config.
                pass

        # Hide plan tree if terminal is narrow
        if self.size.width < PLAN_TREE_AUTO_SHOW_COLS:
            self.state.plan_tree_visible = False
            try:
                self.query_one(PlanTree).display = False
            except Exception:
                pass

        # v0.6.14: dongle detail starts hidden; events pane is the
        # default. Number keys + Enter switch into dongle mode; Esc
        # switches back. Visibility is reconciled by _refresh_all
        # but the initial mount needs to set it once before the first
        # frame to avoid a flash.
        try:
            self.query_one(DongleDetail).display = False
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
            # v0.7.5: catch the natural session end and offer the
            # user a celebration modal with a choice to view the
            # report immediately or stay in the TUI to inspect.
            # Was: TUI exited abruptly when the runner returned,
            # which felt like a crash to users.
            #
            # Only trigger on natural end (SessionEvent.kind=ended),
            # not on user-initiated stop (q+y/q+f, Ctrl-C). Those
            # paths already have their own confirmation flow.
            if (
                isinstance(event, SessionEvent)
                and event.kind == "ended"
                and not self._scan_complete_shown
                and not getattr(
                    self.runner, "_stop_requested", False
                )
            ):
                self._scan_complete_shown = True
                asyncio.create_task(self._show_scan_complete_modal())
        except Exception:
            log.exception("TUI reducer/refresh failed for event %s", event)

    async def _show_scan_complete_modal(self) -> None:
        """v0.7.5: present the ScanComplete modal with summary,
        then act on the user's choice (exit-to-report vs stay-in-TUI).
        Wrapped in a task so the bus subscriber callback returns
        immediately — modals can take arbitrary time."""
        from rfcensus.tui.widgets.modals import ScanComplete
        # Summary line: condense plan progress + counters into one
        # line that gives the user "what just happened?" without
        # making them wait for the full report render.
        n_decodes = self.state.total_decodes
        n_emitters = self.state.total_emitters_confirmed
        n_detections = self.state.total_detections
        n_tasks = self.state.completed_tasks
        n_total = self.state.total_tasks or n_tasks
        elapsed = "—"
        if self.state.session_started_at:
            from datetime import datetime as _dt, timezone as _tz
            secs = (
                _dt.now(_tz.utc) - self.state.session_started_at
            ).total_seconds()
            if secs < 60:
                elapsed = f"{int(secs)}s"
            elif secs < 3600:
                elapsed = f"{int(secs // 60)}m{int(secs % 60):02d}s"
            else:
                h = int(secs // 3600)
                m = int((secs % 3600) // 60)
                elapsed = f"{h}h{m:02d}m"
        summary = (
            f"[bold]{n_tasks}/{n_total}[/] task(s) executed in "
            f"[bold]{elapsed}[/]\n"
            f"  • [bold]{n_decodes}[/] decodes\n"
            f"  • [bold]{n_emitters}[/] confirmed emitters\n"
            f"  • [bold]{n_detections}[/] passive detections"
        )
        choice = await self.push_screen_wait(ScanComplete(summary))
        if choice == "report":
            # User wants the full report now — exit TUI; the CLI
            # coordination layer will await runner_task (already
            # done) and render the final report.
            self.exit()
        # else "stay" — TUI stays open. The user can browse dongles,
        # check the meshtastic pane, etc. When they later press `q`,
        # the existing quit flow runs and the report renders on exit.

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
        # v0.7.4: figure out which pane has Tab-focus so each pane's
        # border title can render a "▸ " marker. Cheap to compute and
        # the only thing that needs to know is the title prefix.
        focused = self.state.focused_pane

        def _focus_mark(pane_name: str) -> str:
            return "▸ " if focused == pane_name else "  "

        try:
            strip = self.query_one(DongleStrip)
            strip.update_dongles(self.state.dongles)
            # v0.6.16: pass BOTH cursor and detail index. detail_index
            # is None when the detail pane isn't open (events mode);
            # the strip uses this to render only-cursor vs cursor-and-
            # detail vs detail-only-not-cursor states distinctly.
            detail_idx = (
                self.state.detail_dongle_index
                if self.state.main_pane_mode == "dongle" else None
            )
            strip.set_selection(
                cursor_index=self.state.focused_dongle_index,
                detail_index=detail_idx,
            )
            strip.border_title = f"{_focus_mark('dongles')}Dongles"
        except Exception:
            pass
        # v0.6.14: keep BOTH the events pane and the dongle-detail
        # pane in sync, but only one is visible at a time. Setting
        # display=False removes the widget from layout entirely so the
        # other expands to fill the row — exactly like a tab swap.
        try:
            es = self.query_one(EventStream)
            # v0.6.16: pass per-mode buffers dict instead of single list.
            es.update_stream(self.state.streams, self.state.filter_mode)
            es.display = (self.state.main_pane_mode == "events")
            # v0.7.4: prepend focus marker; preserve the dynamic title
            # (which encodes filter mode + scroll position) by
            # stripping any prior marker before re-applying.
            t = (es.border_title or "Events").lstrip("▸ ").lstrip()
            es.border_title = f"{_focus_mark('main')}{t}"
        except Exception:
            pass
        try:
            dd = self.query_one(DongleDetail)
            # v0.6.16: detail pane shows detail_dongle_index, not
            # focused_dongle_index. This is the change that lets the
            # cursor and the detail-shown dongle be different — arrow
            # keys move the cursor without disturbing what's being
            # shown. detail_dongle_index is set by Enter or number key.
            detail_idx_for_pane = (
                self.state.detail_dongle_index
                if self.state.detail_dongle_index is not None
                else self.state.focused_dongle_index
            )
            dd.update_state(
                self.state.dongles,
                detail_idx_for_pane,
                self.state.stream,
            )
            dd.display = (self.state.main_pane_mode == "dongle")
            t = (dd.border_title or "Dongle detail").lstrip("▸ ").lstrip()
            dd.border_title = f"{_focus_mark('main')}{t}"
        except Exception:
            pass
        try:
            tree = self.query_one(PlanTree)
            tree.update_plan(self.state.waves, self.state.current_wave_index)
            tree.display = self.state.plan_tree_visible
            tree.border_title = f"{_focus_mark('plan_tree')}Plan"
        except Exception:
            pass
        try:
            mw = self.query_one(MeshtasticRecentWidget)
            mw.update_entries(self.state.meshtastic_recent)
            mw.display = (self.state.main_pane_mode == "meshtastic")
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
            # v0.7.6: surface graceful-shutdown state in the header
            # so the user sees a spinner + "shutting down" message
            # instead of the silent wait while the wave drains.
            header.set_shutting_down(self.state.shutting_down)
            # v0.6.17: push proc stats to header. v0.7.5 reverted
            # the v0.7.4 footer copy after the dock-stack render
            # bug was fixed (DongleStrip's dock:top covered the
            # header). Header is now the canonical home for
            # version + cpu/rss + shutdown state.
            from rfcensus.tui.proc_stats import format_cpu, format_rss
            stats = self._proc_sampler.sample()
            cpu_s = format_cpu(stats.cpu_percent)
            rss_s = format_rss(stats.rss_bytes)
            header.set_proc_stats(cpu_s, rss_s)
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
            # v0.7.5: footer no longer renders proc stats — header
            # is the canonical home now that the dock-stack bug
            # that hid the header is fixed (DongleStrip's dock:top
            # was covering it). User explicitly asked to revert
            # the v0.7.4 belt-and-suspenders footer copy.
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
        # v0.6.14: top-level Esc returns the main pane to events
        # mode if a dongle detail panel is currently shown. Modal
        # screens (ConfirmQuit, HelpOverlay, etc.) handle their own
        # Esc via their own bindings before this fires.
        if self.state.main_pane_mode != "events":
            self.state.main_pane_mode = "events"
            self._refresh_all()

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
            choice = await self.push_screen_wait(ConfirmQuit())
            if choice == "cancel" or choice is None:
                return
            if self.runner is not None:
                self.runner.request_stop()
            if choice == "force":
                # v0.7.4 force semantics — exit TUI immediately,
                # CLI cancels runner and renders partial report.
                self._force_quit_requested = True
                self.exit()
            else:
                # v0.7.6 graceful semantics — DON'T exit the TUI.
                # Instead, set the header spinner and let the user
                # watch the last wave drain. The TUI will exit on
                # its own when SessionEvent.kind=ended arrives, OR
                # the user can hit Ctrl+Q to escalate to fast quit.
                self.state.shutting_down = True
                self._refresh_dynamic()    # surface spinner now
        asyncio.create_task(_confirm_then_quit())

    def action_fast_quit(self) -> None:
        """v0.7.6: Ctrl+Q — panic exit. Cancel the runner, skip the
        report render, exit immediately. The DB still has whatever
        was captured; user can recover via `rfcensus list decodes
        --session N`. Single keypress, no confirmation."""
        self._fast_quit_requested = True
        self._force_quit_requested = True    # short-circuit graceful in CLI
        if self.runner is not None:
            self.runner.request_stop()
        self.exit()

    def action_snapshot(self) -> None:
        # v0.7.5: Textual actions can be async. We await the
        # ReportBuilder render in a background task so the action
        # handler returns immediately, keeping the UI responsive
        # while the report builds (DB queries can take a moment on
        # long sessions).
        asyncio.create_task(self._snapshot_async())

    async def _snapshot_async(self) -> None:
        try:
            text = await self._render_snapshot_report_async()
        except Exception as exc:
            log.exception("snapshot render failed")
            text = f"⚠ snapshot render failed: {exc}"
        self.push_screen(ReportModal(text))

    def action_toggle_log_mode(self) -> None:
        # In v0.6.5 we just exit — the CLI sees us exit and resumes
        # log streaming. Re-entering TUI would require relaunching the
        # command, which is fine for v0.6.5.
        self._log_mode_requested = True
        self.exit()

    def action_toggle_meshtastic(self) -> None:
        """Toggle the meshtastic-recent pane.

        Same shape as the dongle-detail toggle: pressing `m` swaps the
        main pane between events mode and the meshtastic-decode tail.
        Pressing `m` again returns to events mode.
        """
        if self.state.main_pane_mode == "meshtastic":
            self.state.main_pane_mode = "events"
        else:
            self.state.main_pane_mode = "meshtastic"
        self._refresh_all()

    def action_focus_prev(self) -> None:
        if self.state.focused_dongle_index > 0:
            self.state.focused_dongle_index -= 1
            # v0.7.3: when the dongle detail pane is open, focus
            # navigation should immediately update the pane to the
            # newly-focused dongle. Previously the user had to arrow
            # over to the next dongle and then press Enter to commit
            # the switch — clunky for browsing 3-5 dongles in detail.
            # Detail mode is now the toggle (Enter to enter, Esc/Enter
            # to leave); arrows fluidly walk between dongles within it.
            if self.state.main_pane_mode == "dongle":
                self.state.detail_dongle_index = (
                    self.state.focused_dongle_index
                )
            self._refresh_all()

    # v0.7.4: pane-level focus cycling. Tab and Shift+Tab move
    # focus between the three top-level panes:
    #   "dongles"   → top tile strip
    #   "main"      → middle pane (events / dongle detail / mesh)
    #   "plan_tree" → right-side plan tree
    # Each gets a brighter border title when focused. Arrow keys
    # remain context-aware (they currently dispatch based on
    # main_pane_mode); the focused_pane indicator is informational
    # for users navigating with the keyboard. Plan-tree intra-pane
    # navigation lands in v0.7.5.

    _PANE_ORDER = ("dongles", "main", "plan_tree")

    def action_focus_next_pane(self) -> None:
        order = self._PANE_ORDER
        i = order.index(self.state.focused_pane) if (
            self.state.focused_pane in order
        ) else 0
        self.state.focused_pane = order[(i + 1) % len(order)]
        self._refresh_all()

    def action_focus_prev_pane(self) -> None:
        order = self._PANE_ORDER
        i = order.index(self.state.focused_pane) if (
            self.state.focused_pane in order
        ) else 0
        self.state.focused_pane = order[(i - 1) % len(order)]
        self._refresh_all()

    def action_focus_next(self) -> None:
        if self.state.focused_dongle_index < len(self.state.dongles) - 1:
            self.state.focused_dongle_index += 1
            # v0.7.3: detail pane follows focus — see action_focus_prev.
            if self.state.main_pane_mode == "dongle":
                self.state.detail_dongle_index = (
                    self.state.focused_dongle_index
                )
            self._refresh_all()

    def action_open_detail(self) -> None:
        # v0.7.3: simplified to two cases (was three before arrows
        # started syncing the detail pane to focus):
        #   (a) Detail pane closed → open at the focused dongle
        #   (b) Detail pane open → close (back to events mode)
        # Switching between dongles WHILE in detail mode is now
        # arrow-key-only (action_focus_prev/next sync the detail
        # index). Enter is purely the toggle.
        if not self.state.dongles:
            return
        if self.state.main_pane_mode == "dongle":
            self.state.main_pane_mode = "events"
        else:
            self.state.detail_dongle_index = self.state.focused_dongle_index
            self.state.main_pane_mode = "dongle"
        self._refresh_all()

    # ── v0.6.13 (revised v0.6.16): event stream scroll actions ───
    # The EventStream widget owns its scroll offset. Direction
    # convention with v0.6.16's chronological rendering (oldest top,
    # newest bottom):
    #   scroll_lines(+n): see OLDER (move viewport up the buffer)
    #   scroll_lines(-n): see NEWER (move viewport down toward live)
    # Up arrow visually moves the viewport up = shows older entries
    # (which are higher in the buffer) = positive offset.

    def action_scroll_up(self) -> None:
        # v0.7.4: now also handles dongle-detail mode. The detail
        # pane's footer hint advertised ↑/↓ scroll but it was a
        # no-op until this was wired up. In dongle mode, ↑ moves
        # one line UP in the detail content (same convention as
        # less/man); in events mode, ↑ shows one OLDER entry.
        if self.state.main_pane_mode == "events":
            self.query_one(EventStream).scroll_lines(1)
        elif self.state.main_pane_mode == "dongle":
            self.query_one(DongleDetail).scroll_lines(-1)

    def action_scroll_down(self) -> None:
        if self.state.main_pane_mode == "events":
            self.query_one(EventStream).scroll_lines(-1)
        elif self.state.main_pane_mode == "dongle":
            self.query_one(DongleDetail).scroll_lines(+1)

    def action_scroll_page_up(self) -> None:
        if self.state.main_pane_mode == "events":
            self.query_one(EventStream).scroll_page(+1)
        elif self.state.main_pane_mode == "dongle":
            dd = self.query_one(DongleDetail)
            dd.scroll_lines(-max(1, dd.size.height - 2))

    def action_scroll_page_down(self) -> None:
        if self.state.main_pane_mode == "events":
            self.query_one(EventStream).scroll_page(-1)
        elif self.state.main_pane_mode == "dongle":
            dd = self.query_one(DongleDetail)
            dd.scroll_lines(+max(1, dd.size.height - 2))

    def action_scroll_to_live(self) -> None:
        # End key: snap to live tail (newest visible at bottom).
        if self.state.main_pane_mode == "events":
            self.query_one(EventStream).scroll_to_live()
        elif self.state.main_pane_mode == "dongle":
            # In detail mode, "live tail" doesn't really apply —
            # treat End as "scroll to bottom of content".
            dd = self.query_one(DongleDetail)
            dd.scroll_lines(10_000)    # clamp will cap to max

    def action_scroll_to_oldest(self) -> None:
        # Home key: jump to oldest entry in current mode's buffer
        # (events) or top of detail content (dongle).
        if self.state.main_pane_mode == "events":
            self.query_one(EventStream).scroll_to_oldest()
        elif self.state.main_pane_mode == "dongle":
            dd = self.query_one(DongleDetail)
            dd._scroll_offset = 0
            dd.refresh()

    # Generated number-key actions
    def _open_slot(self, slot: int) -> None:
        # v0.6.16: number keys are direct-jump-to-detail. Sets both
        # cursor and detail to the chosen slot, and forces detail mode
        # on. Distinct from arrow keys (which only move cursor) and
        # Enter (which toggles based on cursor vs detail).
        idx = slot - 1
        if 0 <= idx < len(self.state.dongles):
            self.state.focused_dongle_index = idx
            self.state.detail_dongle_index = idx
            self.state.main_pane_mode = "dongle"
            self._refresh_all()

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

    async def _render_snapshot_report_async(self) -> str:
        """Render an in-flight report of session state.

        v0.6.17: built from in-memory event-stream string parsing —
        much lower fidelity than the final report.
        v0.7.5: rewritten to use the same ``ReportBuilder`` pipeline
        as the final session-end report. Synthesizes a partial
        ``SessionResult`` from the runner's stashed plan + current
        strategy results, then runs render_text_report against the
        live DB. The user gets THE SAME quality report at any time
        during the scan as they get at the end — just with a clear
        "IN-FLIGHT REPORT — session still running" header so they
        know counters reflect work-so-far.
        """
        # If we don't have a live runner, fall back to the legacy
        # in-memory renderer (rare — TUI without runner is mainly
        # used for headless tests).
        if self.runner is None:
            return self._render_snapshot_legacy()
        sid = getattr(self.runner, "_current_session_id", None)
        plan = getattr(self.runner, "_current_plan", None)
        if sid is None or plan is None:
            # Session hasn't fully started yet — show what we know.
            return self._render_snapshot_legacy()

        from rfcensus.engine.session import SessionResult
        from rfcensus.reporting.report import ReportBuilder

        # Synthesize a partial result. stopped_early=False because
        # the user is just asking for a peek; the session is still
        # running. The IN-FLIGHT banner above the report makes this
        # unambiguous.
        started = (
            self.state.session_started_at or datetime.now(timezone.utc)
        )
        partial = SessionResult(
            session_id=sid,
            started_at=started,
            ended_at=datetime.now(timezone.utc),
            plan=plan,
            strategy_results=list(self.runner._strategy_results),
            total_decodes=sum(
                r.decodes_emitted
                for r in self.runner._strategy_results
            ),
            warnings=list(plan.warnings),
            stopped_early=False,
        )

        builder = ReportBuilder(self.runner.db)
        try:
            report_body = await builder.render(
                partial,
                fmt="text",
                site_name=self.state.site_name or "default",
                command_name="snapshot",
            )
        except Exception as exc:
            return (
                f"⚠ snapshot via ReportBuilder failed: {exc}\n"
                f"Falling back to in-memory snapshot:\n\n"
                + self._render_snapshot_legacy()
            )

        # Prepend the in-flight banner so users know this is a peek
        # mid-session and counters reflect work-so-far.
        banner = (
            "╭─ IN-FLIGHT REPORT — session still running "
            f"(status: {self.state.session_status}) ─╮\n"
            "  Bands not yet visited won't appear; counters reflect "
            "completed waves only.\n"
            "  Press `r` again any time for an updated snapshot.\n"
            "╰─────────────────────────────────────────────────────────╯"
            "\n\n"
        )
        return banner + report_body

    def _render_snapshot_report(self) -> str:
        """Sync wrapper kept for backward compat with any caller that
        invoked the v0.6.17 sync API. Always returns the legacy
        in-memory render — the proper async path is now
        ``_render_snapshot_report_async``."""
        return self._render_snapshot_legacy()

    def _render_snapshot_legacy(self) -> str:
        """Legacy in-memory snapshot renderer. Kept as a fallback
        when the runner isn't fully initialized (no session id yet,
        plan not built) or when the ReportBuilder path fails."""
        from collections import Counter
        import re

        lines = []
        sid = self.state.session_id or "—"
        if self.state.session_status in {"running", "paused"}:
            lines.append(
                f"╭─ IN-FLIGHT REPORT — session #{sid} still "
                f"{self.state.session_status} ─╮"
            )
            lines.append(
                "  Counters and per-band breakdowns reflect work "
                "completed so far only."
            )
        else:
            lines.append(f"Session #{sid} — {self.state.site_name}")
        if self.state.site_name and self.state.session_status in {"running", "paused"}:
            lines.append(f"  Site: {self.state.site_name}")
        if self.state.session_started_at:
            elapsed = (
                datetime.now(timezone.utc) - self.state.session_started_at
            ).total_seconds()
            lines.append(f"Elapsed: {int(elapsed)}s "
                         f"(paused: {int(self.state.paused_total_s)}s)")
        lines.append("")

        # ── Plan progress with per-task status ──────────────────
        lines.append(
            f"Plan: {len(self.state.waves)} wave(s), "
            f"{self.state.completed_tasks}/{self.state.total_tasks} tasks done"
        )
        for w in self.state.waves:
            if w.status == "completed":
                if w.error_count == 0:
                    wmark = "✓"
                elif w.error_count >= w.task_count:
                    wmark = "✗"
                else:
                    wmark = "⚠"
                summary = f"{w.successful_count}/{w.task_count}"
            elif w.status == "running":
                wmark = "◆"
                summary = f"{w.task_count} task(s)"
            else:
                wmark = "·"
                summary = f"{w.task_count} task(s)"
            lines.append(f"  {wmark} Wave {w.index}  {summary}")
            # Per-task glyphs — only show non-pending so the failures
            # pop. For all-pending waves we elide entirely.
            for j, s in enumerate(w.task_summaries):
                status = (
                    w.task_statuses[j]
                    if j < len(w.task_statuses) else "pending"
                )
                if status == "pending":
                    continue
                tmark = {
                    "ok": "✓", "running": "◆", "failed": "✗",
                    "crashed": "✗", "timeout": "⏱", "skipped": "⌀",
                }.get(status, "·")
                lines.append(f"      {tmark} {s}  [{status}]")
        lines.append("")

        # ── Per-dongle compact stats ────────────────────────────
        lines.append("Dongles:")
        for i, d in enumerate(self.state.dongles, start=1):
            slot = "0" if i == 10 else str(i)
            ant = f" ant={d.antenna_id}" if d.antenna_id else ""
            band = f" band={d.band_id}" if d.band_id else ""
            decs = (
                f" dec={d.decodes_in_band} det={d.detections_in_band}"
                if (d.decodes_in_band or d.detections_in_band) else ""
            )
            visited = (
                f" bands_visited={len(d.bands_visited)}"
                if len(d.bands_visited) > 1 else ""
            )
            lines.append(
                f"  [{slot}] {d.dongle_id}  {d.status}"
                f"{ant}{band}{decs}{visited}"
            )
        lines.append("")

        # ── Per-band activity (derived from event stream) ───────
        # Walk the verbose buffer and tally decodes/detections per band.
        # The verbose buffer caps at ~5000 entries which covers tens of
        # minutes at typical event rates — for very long sessions some
        # early activity may have rolled off, which we note.
        verbose = self.state.streams.get("verbose", [])
        band_decodes: Counter = Counter()
        band_detections: Counter = Counter()
        # Match patterns: "decoded ... at <freq> MHz" → freq buckets
        # don't always carry a band id, so for robustness we tally
        # by band when we can pull it from the event raw payload, and
        # fall back to grouping by frequency rounded to ±100 kHz.
        decode_re = re.compile(r"decoded\s+(\S+)\s+at\s+(\d+\.\d+)\s*MHz",
                               re.IGNORECASE)
        detect_re = re.compile(r"detected\s+(\S+)\s+at\s+(\d+\.\d+)\s*MHz",
                               re.IGNORECASE)
        emitter_lines: list[str] = []
        for e in verbose:
            if e.category == "decode":
                m = decode_re.search(e.text)
                if m:
                    decoder, freq = m.group(1), m.group(2)
                    band_decodes[f"{decoder}@{freq}MHz"] += 1
            elif e.category == "detection":
                m = detect_re.search(e.text)
                if m:
                    decoder, freq = m.group(1), m.group(2)
                    band_detections[f"{decoder}@{freq}MHz"] += 1
            elif e.category == "emitter":
                emitter_lines.append(e.text)

        if band_decodes or band_detections:
            lines.append("Activity by band:")
            seen = set()
            combined: list[tuple[int, int, str]] = []
            for k in set(band_decodes) | set(band_detections):
                combined.append(
                    (band_decodes[k], band_detections[k], k)
                )
            # Sort by total activity descending (sum of decodes+detections)
            combined.sort(key=lambda t: -(t[0] + t[1]))
            for dec, det, k in combined[:30]:
                lines.append(f"  {k:36s}  decodes={dec:3d}  detections={det:3d}")
                seen.add(k)
            if len(combined) > 30:
                lines.append(f"  ... and {len(combined) - 30} more")
            lines.append("")

        # ── Recent confirmed emitters ──────────────────────────
        if emitter_lines:
            lines.append(f"Recent emitters ({len(emitter_lines)} in buffer):")
            # Show last 15 emitter events; they're already chronological
            for t in emitter_lines[-15:]:
                lines.append(f"  • {t}")
            lines.append("")

        # ── Counters ────────────────────────────────────────────
        lines.append(
            f"Counters: decodes={self.state.total_decodes}  "
            f"emitters={self.state.total_emitters_confirmed}  "
            f"detections={self.state.total_detections}"
        )
        lines.append("")

        # ── Recent events tail ────────────────────────────────
        # Use verbose buffer and show the last 30 lines (chronological).
        lines.append("Recent events:")
        recent = verbose[-30:]
        if recent:
            for e in recent:
                ts = e.timestamp.strftime("%H:%M:%S")
                lines.append(f"  {ts}  [{e.severity}] {e.text}")
        else:
            lines.append("  (none yet)")
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
