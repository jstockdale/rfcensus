"""TUI state — single source of truth, updated by event reducer.

The TUI is built as a Redux-style architecture: every visible widget
reads from a single `TUIState` dataclass, and the only way to change
state is via `reduce(state, event)`. This makes the rendering
deterministic and the state-update logic testable in isolation
(no Textual dependency, no event-loop mocking — just call the
reducer with synthetic events and assert on the output state).

Why this matters
----------------

The alternative (each widget subscribes to bus events directly and
mutates its own state) gets messy fast: widgets that need data from
multiple event types need to coordinate, race conditions appear when
events arrive out of order, and testing requires standing up the
whole UI. The reducer pattern keeps state changes explicit and
serialized.

How it's used
-------------

The Textual app subscribes to the EventBus once, stuffs every event
into a queue, and a single coroutine pumps the queue → reducer → state.
After each event, the app calls `widget.refresh()` on widgets that
might display different data. Widgets pull from `app.state` when they
render — they never store derived data themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

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


# ────────────────────────────────────────────────────────────────────
# Per-dongle state
# ────────────────────────────────────────────────────────────────────


@dataclass
class DongleState:
    """All the per-dongle info the dashboard needs to render one tile.

    Updated by HardwareEvent and FanoutClientEvent reducers. The
    DongleStrip widget reads this; the DongleDetail widget reads this
    plus its own derived data (recent decodes, fanout activity log).
    """

    dongle_id: str
    model: str = ""
    # Status: "idle" / "active" / "degraded" / "failed" / "permanent_failed"
    status: str = "idle"
    # Currently leased — None means free.
    consumer: str | None = None
    freq_hz: int | None = None
    sample_rate: int | None = None
    band_id: str | None = None
    # Decoder counters (for the detail screen)
    decodes_total: int = 0
    decodes_in_band: int = 0  # resets when band changes
    last_decode_at: datetime | None = None
    # Fanout activity (for detail screen)
    fanout_clients: int = 0
    fanout_dropped_chunks: int = 0
    # Status message (for failure detail)
    status_message: str = ""


# ────────────────────────────────────────────────────────────────────
# Per-wave / per-task plan state
# ────────────────────────────────────────────────────────────────────


@dataclass
class WaveState:
    """State of one wave in the plan."""

    index: int
    task_count: int
    task_summaries: list[str] = field(default_factory=list)
    # "pending" / "running" / "completed"
    status: str = "pending"
    successful_count: int = 0
    error_count: int = 0


@dataclass
class TaskState:
    """In-flight task (cleared from the dict when TaskCompletedEvent
    fires)."""

    band_id: str
    dongle_id: str
    consumer: str
    started_at: datetime


# ────────────────────────────────────────────────────────────────────
# Event-stream entry
# ────────────────────────────────────────────────────────────────────


@dataclass
class StreamEntry:
    """One row in the EventStream widget."""

    timestamp: datetime
    severity: str        # "info" | "good" | "warning" | "error" | "highlight"
    category: str        # "session" | "wave" | "task" | "decode" | "emitter" |
                         # "detection" | "hardware" | "fanout" | "channel"
    text: str
    # Original event for debugging / filtering
    raw: Event | None = None


# ────────────────────────────────────────────────────────────────────
# Top-level state
# ────────────────────────────────────────────────────────────────────


@dataclass
class TUIState:
    """Single source of truth for the TUI."""

    # Session metadata
    session_id: int | None = None
    session_started_at: datetime | None = None
    site_name: str = ""
    duration_s: float | None = None  # None = indefinite
    # "running" / "paused" / "ending" / "ended"
    session_status: str = "running"
    paused_total_s: float = 0.0  # excluded from elapsed math when computing remaining

    # Per-dongle state, ordered by tile slot (1..N). Always preserves
    # original detection order — failures don't reorder.
    dongles: list[DongleState] = field(default_factory=list)

    # Plan state
    plan_ready: bool = False
    waves: list[WaveState] = field(default_factory=list)
    current_wave_index: int | None = None
    current_pass_n: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0  # cumulative across all waves
    # Active in-flight tasks, keyed by (wave_index, band_id)
    active_tasks: dict[tuple[int, str], TaskState] = field(default_factory=dict)

    # Decode + detection counters (footer)
    total_decodes: int = 0
    total_emitters_confirmed: int = 0
    total_detections: int = 0

    # Event stream (most recent N entries kept). Newest at index 0.
    stream: list[StreamEntry] = field(default_factory=list)
    stream_capacity: int = 500  # cap to avoid unbounded growth

    # UI-controlled state — not driven by events
    focused_dongle_index: int = 0  # which tile in the strip has focus
    detail_dongle_index: int | None = None  # None = strip view; int = detail view
    plan_tree_visible: bool = True
    filter_mode: str = "filtered"  # "minimal" | "filtered" | "verbose"
    help_visible: bool = False
    confirm_quit_visible: bool = False
    report_modal_visible: bool = False


# ────────────────────────────────────────────────────────────────────
# Reducer entry point
# ────────────────────────────────────────────────────────────────────


def reduce(state: TUIState, event: Event) -> TUIState:
    """Apply one event to the state. Mutates and returns `state`.

    Mutation rather than copy because (a) the state can grow large
    (500 stream entries × N dongles), (b) we always want every reader
    to see the new value, no chance of stale references. The TUI's
    re-render after reduce is what actually triggers visual update;
    the reducer just keeps the data correct.

    Unknown event types are ignored. Returning state unchanged for
    irrelevant events lets the app pump every bus event through here
    without filtering.
    """
    # Dispatch by isinstance — fast for our small event taxonomy.
    if isinstance(event, SessionEvent):
        _reduce_session(state, event)
    elif isinstance(event, PlanReadyEvent):
        _reduce_plan_ready(state, event)
    elif isinstance(event, WaveStartedEvent):
        _reduce_wave_started(state, event)
    elif isinstance(event, WaveCompletedEvent):
        _reduce_wave_completed(state, event)
    elif isinstance(event, TaskStartedEvent):
        _reduce_task_started(state, event)
    elif isinstance(event, TaskCompletedEvent):
        _reduce_task_completed(state, event)
    elif isinstance(event, HardwareEvent):
        _reduce_hardware(state, event)
    elif isinstance(event, FanoutClientEvent):
        _reduce_fanout(state, event)
    elif isinstance(event, DecodeEvent):
        _reduce_decode(state, event)
    elif isinstance(event, EmitterEvent):
        _reduce_emitter(state, event)
    elif isinstance(event, DetectionEvent):
        _reduce_detection(state, event)
    elif isinstance(event, DecoderFailureEvent):
        _reduce_decoder_failure(state, event)
    elif isinstance(event, ActiveChannelEvent):
        _reduce_active_channel(state, event)
    return state


# ────────────────────────────────────────────────────────────────────
# Per-event reducers
# ────────────────────────────────────────────────────────────────────


def _reduce_session(state: TUIState, e: SessionEvent) -> None:
    if e.kind == "started":
        state.session_id = e.session_id
        state.session_started_at = e.timestamp
        state.session_status = "running"
        _push_stream(state, e, "info", "session",
                     f"session {e.session_id} started")
    elif e.kind == "ended":
        state.session_status = "ended"
        _push_stream(state, e, "info", "session", "session ended")
    elif e.kind == "phase_changed":
        # Phase events are noisy — don't push to stream by default.
        # The wave-started/completed events convey the same info more
        # structurally.
        pass


def _reduce_plan_ready(state: TUIState, e: PlanReadyEvent) -> None:
    state.plan_ready = True
    state.total_tasks = e.total_tasks
    state.waves = [
        WaveState(
            index=w["index"],
            task_count=w["task_count"],
            task_summaries=list(w.get("task_summaries", [])),
            status="pending",
        )
        for w in e.waves
    ]
    _push_stream(
        state, e, "info", "session",
        f"plan: {len(e.waves)} wave(s), {e.total_tasks} task(s)",
    )


def _reduce_wave_started(state: TUIState, e: WaveStartedEvent) -> None:
    state.current_wave_index = e.wave_index
    state.current_pass_n = e.pass_n
    if 0 <= e.wave_index < len(state.waves):
        state.waves[e.wave_index].status = "running"
    _push_stream(
        state, e, "info", "wave",
        f"wave {e.wave_index} started ({e.task_count} task(s))",
    )


def _reduce_wave_completed(state: TUIState, e: WaveCompletedEvent) -> None:
    if 0 <= e.wave_index < len(state.waves):
        w = state.waves[e.wave_index]
        w.status = "completed"
        w.successful_count = e.successful_count
        w.error_count = len(e.errors)
    severity = "good" if not e.errors else "warning"
    _push_stream(
        state, e, severity, "wave",
        f"wave {e.wave_index} done: {e.successful_count}/{e.task_count} "
        f"ok" + (f" ({len(e.errors)} error(s))" if e.errors else ""),
    )


def _reduce_task_started(state: TUIState, e: TaskStartedEvent) -> None:
    state.active_tasks[(e.wave_index, e.band_id)] = TaskState(
        band_id=e.band_id,
        dongle_id=e.dongle_id,
        consumer=e.consumer,
        started_at=e.timestamp,
    )
    _push_stream(
        state, e, "info", "task",
        f"task {e.band_id}→{e.dongle_id} started",
    )


def _reduce_task_completed(state: TUIState, e: TaskCompletedEvent) -> None:
    state.active_tasks.pop((e.wave_index, e.band_id), None)
    state.completed_tasks += 1
    severity = {
        "ok": "good",
        "skipped": "info",
        "failed": "warning",
        "crashed": "error",
        "timeout": "warning",
    }.get(e.status, "info")
    detail = f" — {e.detail}" if e.detail else ""
    _push_stream(
        state, e, severity, "task",
        f"task {e.band_id}→{e.dongle_id} {e.status}{detail}",
    )


def _reduce_hardware(state: TUIState, e: HardwareEvent) -> None:
    dongle = _ensure_dongle(state, e.dongle_id)
    if e.kind == "detected":
        dongle.status = "idle"
    elif e.kind == "healthy":
        # Don't downgrade an "active" tile to idle just because a
        # health check came through — only set idle when explicitly
        # released.
        if dongle.status not in ("active", "degraded", "failed"):
            dongle.status = "idle"
    elif e.kind == "allocated":
        dongle.status = "active"
        dongle.consumer = e.consumer
        dongle.freq_hz = e.freq_hz
        dongle.sample_rate = e.sample_rate
        # Reset in-band counter when band changes
        if e.band_id != dongle.band_id:
            dongle.decodes_in_band = 0
        dongle.band_id = e.band_id
    elif e.kind == "released":
        dongle.status = "idle"
        dongle.consumer = None
        dongle.freq_hz = None
        dongle.sample_rate = None
        dongle.band_id = None
        dongle.fanout_clients = 0  # all decoders disconnected
    elif e.kind == "degraded":
        dongle.status = "degraded"
        dongle.status_message = e.detail or "degraded"
    elif e.kind == "failed":
        dongle.status = "failed"
        dongle.status_message = e.detail or "failed"
    elif e.kind == "permanently_failed":
        dongle.status = "permanent_failed"
        dongle.status_message = e.detail or "permanently failed"
    elif e.kind == "reconnected":
        dongle.status = "idle"
        dongle.status_message = "reconnected"

    # Surface failures and reconnects in the stream; allocate/release
    # are visible enough in the strip itself.
    if e.kind in ("failed", "permanently_failed", "degraded", "reconnected"):
        sev = {
            "failed": "error",
            "permanently_failed": "error",
            "degraded": "warning",
            "reconnected": "good",
        }[e.kind]
        _push_stream(
            state, e, sev, "hardware",
            f"{e.dongle_id}: {e.kind} — {e.detail}",
        )


def _reduce_fanout(state: TUIState, e: FanoutClientEvent) -> None:
    # slot_id is "fanout[<dongle_id>]" — extract the inner id
    dongle_id = e.slot_id
    if dongle_id.startswith("fanout[") and dongle_id.endswith("]"):
        dongle_id = dongle_id[len("fanout["):-1]
    dongle = _ensure_dongle(state, dongle_id)
    if e.event_type == "connect":
        dongle.fanout_clients += 1
    elif e.event_type == "disconnect":
        dongle.fanout_clients = max(0, dongle.fanout_clients - 1)
    elif e.event_type == "slow":
        dongle.fanout_dropped_chunks = max(
            dongle.fanout_dropped_chunks, e.bytes_sent // 16384,
        )
        # Slow events are informative but rate-limited at source
        _push_stream(
            state, e, "warning", "fanout",
            f"{dongle_id}: client {e.peer_addr} slow",
        )
    elif e.event_type == "dropped":
        _push_stream(
            state, e, "error", "fanout",
            f"{dongle_id}: client {e.peer_addr} dropped",
        )


def _reduce_decode(state: TUIState, e: DecodeEvent) -> None:
    state.total_decodes += 1
    # Find the dongle currently leased at e.freq_hz (best effort) and
    # bump its counter. If no match, just bump the global counter.
    for d in state.dongles:
        if d.consumer and d.freq_hz and abs(d.freq_hz - e.freq_hz) < 100_000:
            d.decodes_total += 1
            d.decodes_in_band += 1
            d.last_decode_at = e.timestamp
            break
    # Don't push to stream — too noisy. The footer counter is enough.


def _reduce_emitter(state: TUIState, e: EmitterEvent) -> None:
    if e.kind == "confirmed":
        state.total_emitters_confirmed += 1
        _push_stream(
            state, e, "highlight", "emitter",
            f"✓ {e.emitter_id} confirmed at "
            f"{e.typical_freq_hz / 1e6:.3f} MHz "
            f"(conf={e.confidence:.2f})",
        )
    elif e.kind == "new":
        _push_stream(
            state, e, "good", "emitter",
            f"new {e.emitter_id} at {e.typical_freq_hz / 1e6:.3f} MHz",
        )


def _reduce_detection(state: TUIState, e: DetectionEvent) -> None:
    state.total_detections += 1
    _push_stream(
        state, e, "highlight", "detection",
        f"detected {e.technology} at {e.freq_hz / 1e6:.3f} MHz "
        f"(conf={e.confidence:.2f}) — {e.evidence[:60]}",
    )


def _reduce_decoder_failure(state: TUIState, e: DecoderFailureEvent) -> None:
    _push_stream(
        state, e, "warning", "task",
        f"decoder {e.decoder_name} on {e.band_id} failed after "
        f"{e.elapsed_s:.0f}s ({e.remaining_s:.0f}s remaining)",
    )


def _reduce_active_channel(state: TUIState, e: ActiveChannelEvent) -> None:
    # Only "new" channels are worth surfacing; updates and gone events
    # are too chatty.
    if e.kind == "new":
        _push_stream(
            state, e, "info", "channel",
            f"active channel at {e.freq_center_hz / 1e6:.3f} MHz "
            f"({e.classification}, BW={e.bandwidth_hz // 1000} kHz, "
            f"SNR={e.snr_db:.1f} dB)",
        )


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _ensure_dongle(state: TUIState, dongle_id: str) -> DongleState:
    """Find or create the DongleState for this id. Preserves slot
    order — never reorders the list, which would disorient the user."""
    for d in state.dongles:
        if d.dongle_id == dongle_id:
            return d
    new = DongleState(dongle_id=dongle_id)
    state.dongles.append(new)
    return new


def _push_stream(
    state: TUIState, event: Event, severity: str, category: str, text: str,
) -> None:
    """Insert at index 0 (newest first) and trim to capacity."""
    entry = StreamEntry(
        timestamp=event.timestamp or datetime.now(timezone.utc),
        severity=severity,
        category=category,
        text=text,
        raw=event,
    )
    state.stream.insert(0, entry)
    if len(state.stream) > state.stream_capacity:
        state.stream = state.stream[: state.stream_capacity]


# ────────────────────────────────────────────────────────────────────
# Filter selection (used by EventStream widget)
# ────────────────────────────────────────────────────────────────────


# Categories shown in each filter mode. "minimal" is the bare minimum
# to know if the scan is working; "filtered" (default) adds emitters
# and channel detections; "verbose" adds task and decode chatter.
FILTER_CATEGORIES: dict[str, set[str]] = {
    "minimal": {"session", "hardware"},
    "filtered": {
        "session", "hardware", "wave", "emitter", "detection", "channel",
    },
    "verbose": {
        "session", "hardware", "wave", "task", "decode",
        "emitter", "detection", "channel", "fanout",
    },
}


def filter_stream(
    stream: list[StreamEntry], mode: str,
) -> list[StreamEntry]:
    """Return entries that should display under this filter mode.
    Errors and warnings always pass through regardless of mode."""
    cats = FILTER_CATEGORIES.get(mode, FILTER_CATEGORIES["filtered"])
    out = []
    for e in stream:
        if e.severity in ("error", "warning"):
            out.append(e)
            continue
        if e.category in cats:
            out.append(e)
    return out


def cycle_filter_mode(current: str) -> str:
    """`f` hotkey progression: filtered → verbose → minimal → filtered."""
    return {
        "filtered": "verbose",
        "verbose": "minimal",
        "minimal": "filtered",
    }.get(current, "filtered")
