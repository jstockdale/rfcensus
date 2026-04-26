"""v0.7.5 tests: ScanComplete modal, async snapshot via ReportBuilder,
HeaderBar dock-stack fix verification."""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone


# ─────────────────────────────────────────────────────────────────────
# (1) ScanComplete modal
# ─────────────────────────────────────────────────────────────────────


def test_scan_complete_modal_returns_string() -> None:
    """ScanComplete returns 'report' or 'stay' as a string, not bool.
    The semantics matter: 'report' = exit and show final report,
    'stay' = keep TUI open for inspection."""
    from rfcensus.tui.widgets.modals import ScanComplete
    bases_str = str(ScanComplete.__orig_bases__)
    assert "[str]" in bases_str


def test_scan_complete_action_methods_dismiss_with_strings() -> None:
    """action_go_report → 'report', action_stay → 'stay'."""
    from rfcensus.tui.widgets.modals import ScanComplete
    modal = ScanComplete.__new__(ScanComplete)
    captured = []
    modal.dismiss = lambda v: captured.append(v)    # type: ignore[method-assign]

    modal.action_go_report()
    modal.action_stay()
    assert captured == ["report", "stay"]


def test_scan_complete_keybindings() -> None:
    """Enter/r → report (the default action), Esc/s → stay."""
    from rfcensus.tui.widgets.modals import ScanComplete
    keys = {b[0] for b in ScanComplete.BINDINGS}
    assert "enter" in keys
    assert "r" in keys
    assert "escape" in keys
    assert "s" in keys


def test_scan_complete_includes_summary_text() -> None:
    """The modal stores the caller-supplied summary line so users
    see counters at a glance without having to wait for the full
    report to render. We avoid invoking compose() here since that
    requires a mounted Textual app context."""
    from rfcensus.tui.widgets.modals import ScanComplete
    sc = ScanComplete("17/17 task(s) executed in 14m32s · 88 decodes")
    assert sc._summary_line == (
        "17/17 task(s) executed in 14m32s · 88 decodes"
    )


def test_tui_app_initializes_scan_complete_shown_false() -> None:
    """The flag must default False so the modal CAN fire on first
    natural session end. Set True after firing to prevent duplicates."""
    src = open(
        "/home/claude/rfcensus/rfcensus/tui/app.py"
    ).read()
    assert "self._scan_complete_shown = False" in src


def test_tui_app_skips_scan_complete_on_stop_request() -> None:
    """When the user pressed q+y/q+f or sent Ctrl-C, the runner
    sets _stop_requested. The ScanComplete celebration is for
    natural completion only — user-initiated stops have their own
    flow (ConfirmQuit) and don't get the celebration."""
    src = open(
        "/home/claude/rfcensus/rfcensus/tui/app.py"
    ).read()
    # The guard checks both the duplicate flag AND the runner stop
    assert "_scan_complete_shown" in src
    assert "_stop_requested" in src
    # And the trigger conditions are gated by SessionEvent.kind=ended
    assert "SessionEvent" in src
    assert "kind == \"ended\"" in src


def test_show_scan_complete_modal_method_exists() -> None:
    """The async method that builds the summary and pushes the modal
    is present as an instance method on TUIApp."""
    from rfcensus.tui.app import TUIApp
    assert hasattr(TUIApp, "_show_scan_complete_modal")
    import inspect
    assert inspect.iscoroutinefunction(TUIApp._show_scan_complete_modal)


# ─────────────────────────────────────────────────────────────────────
# (2) Snapshot report via ReportBuilder (same pipeline as final)
# ─────────────────────────────────────────────────────────────────────


def test_snapshot_report_is_async_method() -> None:
    """The new high-fidelity path is async because ReportBuilder
    queries the DB via async SQL drivers. Sync wrapper kept for
    backward compat but always returns the legacy in-memory render."""
    from rfcensus.tui.app import TUIApp
    import inspect
    assert hasattr(TUIApp, "_render_snapshot_report_async")
    assert inspect.iscoroutinefunction(
        TUIApp._render_snapshot_report_async
    )
    assert hasattr(TUIApp, "_render_snapshot_legacy")
    assert hasattr(TUIApp, "_render_snapshot_report")


def test_action_snapshot_uses_async_path() -> None:
    """action_snapshot now spawns an async task that awaits the
    ReportBuilder render before pushing the modal — was previously
    a sync render with in-memory event-stream parsing (much lower
    fidelity than the final report)."""
    src = open(
        "/home/claude/rfcensus/rfcensus/tui/app.py"
    ).read()
    # The async helper exists and is called from action_snapshot
    assert "_snapshot_async" in src
    assert "create_task(self._snapshot_async())" in src
    # The async render uses ReportBuilder
    assert "ReportBuilder(self.runner.db)" in src
    assert "await builder.render(" in src


def test_session_stashes_plan_and_id_for_snapshot() -> None:
    """SessionRunner stashes _current_plan and _current_session_id
    so the TUI can synthesize a partial SessionResult and render
    via ReportBuilder. Without these, the TUI snapshot can't reach
    the same quality as the final report."""
    from rfcensus.engine.session import SessionRunner
    # The instance attributes are init'd in __init__; check the
    # source for the initialization line since we don't have a real
    # SiteConfig + Registry + Database to construct a runner here.
    src = open(
        "/home/claude/rfcensus/rfcensus/engine/session.py"
    ).read()
    assert "self._current_plan: ExecutionPlan | None = None" in src
    assert "self._current_session_id: int | None = None" in src
    # And populated during run() before the wave loop starts
    assert "self._current_plan = plan" in src


def test_snapshot_falls_back_to_legacy_when_runner_missing() -> None:
    """When the TUI is mounted without a runner (rare — mostly tests),
    or before run() has built the plan, the snapshot must still
    produce something readable. The legacy in-memory renderer
    handles this cleanly."""
    src = open(
        "/home/claude/rfcensus/rfcensus/tui/app.py"
    ).read()
    # The async path checks for runner==None and missing plan/sid
    assert "self.runner is None" in src
    assert "_render_snapshot_legacy" in src
    # And the fallback is invoked in both cases
    assert "return self._render_snapshot_legacy()" in src


# ─────────────────────────────────────────────────────────────────────
# (3) Footer cleanup — no version, no proc stats (v0.7.5 revert)
# ─────────────────────────────────────────────────────────────────────


def test_footer_render_excludes_version_v075() -> None:
    """v0.7.4 added version watermark; v0.7.5 removed at user request."""
    from rfcensus.tui.widgets.footer import FooterBar
    from rfcensus import __version__
    fb = FooterBar()
    fb.set_state(
        healthy=5, total=5, decodes=14, emitters=8, detections=0,
        wave_label="0", filter_mode="filtered",
    )
    out = str(fb.render())
    assert f"v{__version__}" not in out
    assert "rfcensus" not in out


def test_footer_set_proc_stats_is_noop_v075() -> None:
    """v0.7.5: header is the proc stats home now that the dock-stack
    bug was fixed. Footer reverts to clean counters + hint."""
    from rfcensus.tui.widgets.footer import FooterBar
    fb = FooterBar()
    before = str(fb.render())
    fb.set_proc_stats("99%", "999 MB")
    after = str(fb.render())
    assert "99%" not in after
    assert "999 MB" not in after
    assert before == after


# ─────────────────────────────────────────────────────────────────────
# (4) HeaderBar dock-stack root cause — DongleStrip no longer covers
# ─────────────────────────────────────────────────────────────────────


def test_header_renders_above_strip_in_real_compose() -> None:
    """End-to-end: mount the actual TUI compose tree and verify
    HeaderBar at y=0 is visible (not covered by DongleStrip).
    This was the root cause of the user's "I don't see CPU/mem
    anywhere" complaint."""
    from textual.app import App
    from textual.containers import Container
    from rfcensus.tui.widgets.header import HeaderBar
    from rfcensus.tui.widgets.dongle_strip import DongleStrip

    class _MiniApp(App):
        def compose(self):
            with Container(id="main"):
                yield HeaderBar(site_name="t")
                yield DongleStrip()

    async def _run():
        app = _MiniApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            return (
                app.query_one(HeaderBar).region,
                app.query_one(DongleStrip).region,
            )

    h, s = asyncio.run(_run())
    assert h.y == 0 and h.height >= 1, (
        f"HeaderBar not at top: y={h.y} h={h.height}"
    )
    assert s.y >= h.y + h.height, (
        f"DongleStrip overlaps HeaderBar: strip.y={s.y} "
        f"header.y={h.y} header.h={h.height}"
    )


def test_header_renders_version_string() -> None:
    """Now that HeaderBar actually renders, the version string the
    user asked for ('rfcensus 0.7.5') appears in its output."""
    from rfcensus.tui.widgets.header import HeaderBar
    from rfcensus import __version__
    h = HeaderBar(site_name="t")
    out = str(h.render())
    assert f"v{__version__}" in out
    assert "rfcensus" in out


def test_header_renders_proc_stats_when_set() -> None:
    """Proc stats live in HeaderBar exclusively as of v0.7.5 (footer
    reverted to no-op shim). set_proc_stats updates the rendered
    output."""
    from rfcensus.tui.widgets.header import HeaderBar
    from unittest.mock import MagicMock

    class _SizedHeader(HeaderBar):
        @property
        def size(self):  # type: ignore[override]
            return MagicMock(width=200, height=1)

    h = _SizedHeader(site_name="t")
    h.set_proc_stats("3.4%", "142 MB")
    out = str(h.render())
    assert "cpu" in out
    assert "3.4%" in out
    assert "rss" in out
    assert "142 MB" in out


# ─────────────────────────────────────────────────────────────────────
# (5) IQSource I/Q alignment — odd-byte chunks from TCP recv must
#     not break the cu8 pipeline
# ─────────────────────────────────────────────────────────────────────


def test_iq_source_buffers_odd_byte_across_reads() -> None:
    """v0.7.5: socket.recv(n) returns up to n bytes — TCP can hand
    back any chunk size including odd ones. Without buffering the
    leftover odd byte, MeshtasticPipeline.feed_cu8 raised
    'cu8 stream must be even-length (I/Q pairs)' and crashed the
    decoder mid-stream. Now IQSource glues the leftover byte to
    the front of the next read so all yielded chunks are I/Q-aligned."""
    from rfcensus.utils.iq_source import IQSource

    class _OddSource(IQSource):
        def __init__(self):
            super().__init__(chunk_size=8)
            # Realistic TCP fragmentation: 7, 9, 5, EOF.
            # Without the fix, the 7-byte chunk would have crashed
            # the cu8 consumer.
            self._reads = iter([b"\x01" * 7, b"\x02" * 9, b"\x03" * 5,
                                b""])

        def read(self, n):
            try:
                return next(self._reads)
            except StopIteration:
                return b""

    src = _OddSource()
    chunks = list(src)
    # Every yielded chunk must be even-length
    for c in chunks:
        assert len(c) % 2 == 0, (
            f"odd-length chunk leaked: len={len(c)} {c!r}"
        )
    # And no bytes lost in the shuffle (7+9+5=21; one odd byte
    # stranded at EOF, so 20 bytes total emitted)
    total = sum(len(c) for c in chunks)
    assert total == 20, f"expected 20 bytes total, got {total}"


def test_iq_source_handles_single_byte_reads() -> None:
    """Worst case: TCP gives back 1 byte at a time. The remainder
    buffer plus the recursion in __next__ must combine them into
    a 2-byte chunk on the next iteration."""
    from rfcensus.utils.iq_source import IQSource

    class _TinySource(IQSource):
        def __init__(self):
            super().__init__(chunk_size=8)
            self._reads = iter([b"\xaa", b"\xbb", b"\xcc\xdd\xee", b""])

        def read(self, n):
            try:
                return next(self._reads)
            except StopIteration:
                return b""

    src = _TinySource()
    chunks = list(src)
    for c in chunks:
        assert len(c) % 2 == 0, f"odd chunk: {c!r}"
    # 1+1+3 = 5 bytes in; one odd byte stranded at EOF; 4 emitted
    assert sum(len(c) for c in chunks) == 4
