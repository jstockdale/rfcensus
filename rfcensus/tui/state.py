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
    # v0.6.14: per-dongle detection counters. Decoder produces
    # decodes; signal-classifier-style detectors (lora_survey,
    # ais_detector, etc.) produce detections. Both are valuable to
    # see per-dongle in the detail panel — a dongle can be running
    # for 10 minutes with 0 decodes but 30 detections, which is
    # important context. Resets like decodes_in_band on band change.
    detections_total: int = 0
    detections_in_band: int = 0
    last_detection_at: datetime | None = None
    # Fanout activity (for detail screen)
    fanout_clients: int = 0
    fanout_dropped_chunks: int = 0
    # v0.6.17: cumulative bytes the fanout has streamed downstream
    # to all clients of this dongle. From FanoutClientEvent.bytes_sent
    # which fires at connect/disconnect/slow/dropped boundaries — so
    # the value is approximate (lags real-time by up to one event)
    # but precise enough for at-a-glance "is this dongle busy" sense.
    fanout_bytes_sent: int = 0
    # v0.6.17: how many distinct band IDs this dongle has been leased
    # to since session start. Useful as a "this dongle is being used
    # productively across the plan" indicator.
    bands_visited: set[str] = field(default_factory=set)
    # v0.6.17: when the current band lease started. Lets the detail
    # pane show "on this band for Xs" without having to derive from
    # event timestamps. Set by the consumer reducer; cleared on
    # release.
    band_started_at: datetime | None = None
    # v0.6.17: antenna config id (e.g. "whip_915", "marine_vhf").
    # Pulled from DongleConfig at session start by the runner; the
    # TUI never updates this. Display-only.
    antenna_id: str | None = None
    # v0.7.3: per-dongle ring buffers and a fanout-peer set so the
    # detail pane can show actual recent activity rather than just
    # counters. The user explicitly asked for these — counters alone
    # weren't enough to tell at a glance which dongle was producing
    # what. Capacity 25 is plenty for an at-a-glance view; older
    # entries belong in `rfcensus list decodes` for forensic browsing.
    recent_decodes: list["_DongleDecodeEntry"] = field(default_factory=list)
    recent_detections: list["_DongleDetectionEntry"] = field(
        default_factory=list,
    )
    # Per-peer client list. The fanout reports peer_addr (IP:port) on
    # connect/disconnect; we keep the live set here so the detail
    # pane can list "rtl_433 + rtlamr + lora_survey" instead of just
    # "fanout clients: 3". Peer→consumer mapping is best-effort: the
    # fanout doesn't know which decoder owns each TCP socket, but
    # consumers usually identify themselves via their TCP source port
    # being mapped through the broker. For now we just show peer_addr.
    fanout_client_peers: set[str] = field(default_factory=set)
    # v0.7.4: live set of consumer names currently leasing this dongle.
    # Each "allocated" HardwareEvent adds; "released" clears. Most
    # dongles have one consumer; fanout-shared dongles like 915 ISM
    # typically have 2-3 (rtl_433 + rtlamr + lora_survey). Used by
    # the detail pane to render named consumers next to the raw
    # peer_addr list, since the fanout itself doesn't know which
    # decoder owns each downstream TCP socket.
    active_consumers: set[str] = field(default_factory=set)
    # Status message (for failure detail)
    status_message: str = ""


@dataclass
class _DongleDecodeEntry:
    """Compact per-dongle decode snapshot for the detail pane.

    A subset of the full DecodeEvent — just what the detail-list
    needs. Kept in a 25-entry ring on each DongleState."""

    timestamp: datetime
    freq_hz: int
    protocol: str
    summary: str    # short payload preview (formatted)


@dataclass
class _DongleDetectionEntry:
    """Per-dongle detection snapshot — analogous to _DongleDecodeEntry
    but for detector events (lora_survey etc.)."""

    timestamp: datetime
    freq_hz: int
    technology: str
    confidence: float


# ────────────────────────────────────────────────────────────────────
# Per-wave / per-task plan state
# ────────────────────────────────────────────────────────────────────


@dataclass
class WaveState:
    """State of one wave in the plan.

    v0.6.17: per-task status tracking. Each task in `task_summaries`
    has a corresponding status in `task_statuses` (same length, same
    order). Status values: "pending" / "running" / "ok" / "failed" /
    "crashed" / "skipped" / "timeout". The plan-tree widget renders
    a status glyph next to each task line so the user can see at a
    glance which task within a failed wave actually failed.
    """

    index: int
    task_count: int
    task_summaries: list[str] = field(default_factory=list)
    # "pending" / "running" / "completed"
    status: str = "pending"
    successful_count: int = 0
    error_count: int = 0
    # v0.6.17: per-task status, parallel to task_summaries.
    task_statuses: list[str] = field(default_factory=list)


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


@dataclass
class MeshtasticDecodeEntry:
    """One row in the TUI's recent-Meshtastic-decodes ring buffer.

    Cherry-picks the fields the widget renders so we don't keep the
    whole DecodeEvent (with its full payload dict) in memory for
    hours of a long monitor run."""

    timestamp: datetime
    freq_hz: int
    preset: str
    crc_ok: bool
    decrypted: bool
    channel_hash: int | None     # the wire-side hash byte
    from_node: int | None        # source node ID (None if not decrypted)
    to_node: int | None          # destination, 0xFFFFFFFF = broadcast
    summary: str                 # pre-formatted via payload_format
    rssi_dbm: float | None
    snr_db: float | None


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

    # Per-protocol recent-decode ring buffers. Currently only used for
    # Meshtastic (the only decoder whose payload we render with a
    # protocol-aware formatter); other protocols keep showing up only
    # as the global ``total_decodes`` counter to avoid screen-real-
    # estate competition between many noisy rtl_433 sensors.
    #
    # Capacity small (50 entries) because a TUI list view above a
    # few dozen rows isn't readable anyway — older entries belong
    # in `rfcensus list decodes` for forensic browsing.
    meshtastic_recent: list["MeshtasticDecodeEntry"] = field(
        default_factory=list,
    )
    meshtastic_recent_capacity: int = 50

    # Event stream — v0.6.16 per-filter-mode ring buffers.
    #
    # WHY PER-MODE: a single 500-entry buffer was getting flooded by
    # the verbose 'channel' category (rtl_power produces 5-30 lines/sec
    # on a busy band), pushing older 'wave', 'emitter', and 'detection'
    # entries off the buffer within tens of seconds. With per-mode
    # buffers, the 'minimal' and 'filtered' views retain their
    # respective categories far longer because the noisy verbose stuff
    # doesn't land in them.
    #
    # ORDER: chronological (oldest at index 0, newest at -1). v0.6.5
    # used newest-at-index-0 for cheap O(1) inserts; we now use append
    # so the rendering can do oldest-top-newest-bottom (like tail -f)
    # without a reverse-on-render. Append is also O(1) amortized; the
    # only difference is trim-from-front when capacity hits, which is
    # O(n) but happens only when the buffer is full and is a single
    # slice operation.
    #
    # CAPACITY: 5000 per mode. At 30 events/sec (busy verbose) that's
    # ~3 minutes of full retention. With minimal+filtered modes seeing
    # only a few events/sec, retention is hours+.
    streams: dict[str, list["StreamEntry"]] = field(
        default_factory=lambda: {
            "minimal": [], "filtered": [], "verbose": [],
        }
    )
    stream_capacity: int = 5000

    @property
    def stream(self) -> list["StreamEntry"]:
        """Backward-compat: returns the verbose buffer in newest-first
        order. Used by dongle_detail's recent-events list and the
        snapshot report. Allocates a new list per access; that's fine
        because both consumers slice the first N entries and don't
        loop tightly. New code should index `streams[mode]` directly
        (chronological, append-friendly)."""
        return list(reversed(self.streams.get("verbose", [])))

    # UI-controlled state — not driven by events
    focused_dongle_index: int = 0  # which tile in the strip has focus
    detail_dongle_index: int | None = None  # None = strip view; int = detail view
    # v0.6.14: which widget owns the central pane.
    #   "events" → EventStream visible (default startup state)
    #   "dongle" → DongleDetail visible (focused on focused_dongle_index)
    # Replaces the old push_screen-based modal which caused stacking
    # when the user pressed multiple number keys in succession.
    main_pane_mode: str = "events"
    plan_tree_visible: bool = True
    filter_mode: str = "filtered"  # "minimal" | "filtered" | "verbose"
    help_visible: bool = False
    confirm_quit_visible: bool = False
    report_modal_visible: bool = False
    # v0.7.4: which top-level pane currently has keyboard focus.
    # Tab / Shift+Tab cycle through:
    #   "dongles"   → arrow keys move between dongle tiles
    #   "main"      → arrow keys scroll the main pane (events / detail)
    #   "plan_tree" → arrow keys navigate plan items (v0.7.5 feature;
    #                 for now this just shows a visual focus indicator)
    # The pane with focus gets a brighter border title in the render
    # layer so the user knows which arrow-key context they're in.
    focused_pane: str = "dongles"
    # v0.7.6: True after the user picked graceful-quit; HeaderBar
    # renders a "shutting down" spinner so the user knows their
    # action was acknowledged and the wave is winding down.
    # Cleared automatically when the session ends and the TUI exits.
    shutting_down: bool = False


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
            # v0.6.17: every task starts pending; transitions to
            # "running" when its TaskStartedEvent fires and to a
            # terminal status when TaskCompletedEvent fires.
            task_statuses=["pending"] * len(w.get("task_summaries", [])),
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
    # v0.6.14: wave transitions get 'highlight' severity (cyan bold)
    # and an explicit ▶ marker so they punch visually through the
    # surrounding chatter. Wave boundaries are the single most useful
    # "where am I in the run" cue and they were getting lost.
    _push_stream(
        state, e, "highlight", "wave",
        f"▶ wave {e.wave_index} started ({e.task_count} task(s))",
    )


def _reduce_wave_completed(state: TUIState, e: WaveCompletedEvent) -> None:
    if 0 <= e.wave_index < len(state.waves):
        w = state.waves[e.wave_index]
        w.status = "completed"
        w.successful_count = e.successful_count
        w.error_count = len(e.errors)
    # v0.6.14: explicit ✓ / ✗ marker + highlight severity. The marker
    # is the same vocabulary as the plan-tree drawer so the two views
    # use a consistent language.
    if not e.errors:
        marker = "✓"
        severity = "highlight"
    else:
        marker = "✗"
        severity = "warning"
    _push_stream(
        state, e, severity, "wave",
        f"{marker} wave {e.wave_index} done: "
        f"{e.successful_count}/{e.task_count} ok"
        + (f" ({len(e.errors)} error(s))" if e.errors else ""),
    )


def _reduce_task_started(state: TUIState, e: TaskStartedEvent) -> None:
    state.active_tasks[(e.wave_index, e.band_id)] = TaskState(
        band_id=e.band_id,
        dongle_id=e.dongle_id,
        consumer=e.consumer,
        started_at=e.timestamp,
    )
    # v0.6.17: mark this task's status in its wave so the plan tree
    # can render a glyph next to the line.
    if 0 <= e.wave_index < len(state.waves):
        w = state.waves[e.wave_index]
        idx = _task_index_in_wave(w, e.band_id, e.dongle_id)
        if idx is not None and idx < len(w.task_statuses):
            w.task_statuses[idx] = "running"
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
    # v0.6.17: record terminal status on the per-task slot
    if 0 <= e.wave_index < len(state.waves):
        w = state.waves[e.wave_index]
        idx = _task_index_in_wave(w, e.band_id, e.dongle_id)
        if idx is not None and idx < len(w.task_statuses):
            w.task_statuses[idx] = e.status
    _push_stream(
        state, e, severity, "task",
        f"task {e.band_id}→{e.dongle_id} {e.status}{detail}",
    )


def _task_index_in_wave(
    w: WaveState, band_id: str, dongle_id: str,
) -> int | None:
    """Find the index of a task within a wave's task_summaries list.

    Task summaries are formatted "<band_id>→<dongle_id>" (with U+2192
    arrow). We try an exact match first; if that fails (because the
    PlanReady event used a different summary format), we fall back to
    matching just the band_id prefix. Returns None if no match — the
    caller should treat this as a no-op rather than crash.
    """
    target_full = f"{band_id}→{dongle_id}"
    for i, s in enumerate(w.task_summaries):
        if s == target_full:
            return i
    # Fallback: band-prefix match. Only safe if the band appears once
    # in this wave (which is the normal case — the scheduler doesn't
    # double-allocate a band within a wave).
    matches = [
        i for i, s in enumerate(w.task_summaries)
        if s == band_id or s.startswith(f"{band_id}→")
    ]
    if len(matches) == 1:
        return matches[0]
    return None


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
        # v0.7.4: collect consumers when fanout-sharing. With single-
        # consumer dongles the set has 1 entry that mirrors `consumer`;
        # with fanout the set grows to 2-3 as each decoder + sidecar
        # gets its own allocate event. The detail pane reads this set
        # to render named consumers in the fanout-clients list.
        if e.consumer:
            dongle.active_consumers.add(e.consumer)
        dongle.freq_hz = e.freq_hz
        dongle.sample_rate = e.sample_rate
        # Reset in-band counter when band changes
        if e.band_id != dongle.band_id:
            dongle.decodes_in_band = 0
            # v0.6.14: detections_in_band tracks the same scope
            dongle.detections_in_band = 0
            # v0.6.17: track when this band's lease started + add to
            # the cumulative set of bands this dongle has worked.
            dongle.band_started_at = e.timestamp
        dongle.band_id = e.band_id
        if e.band_id:
            dongle.bands_visited.add(e.band_id)
    elif e.kind == "released":
        dongle.status = "idle"
        # v0.7.4: a single "released" event clears ALL consumers
        # because the broker releases the dongle when the last lease
        # is dropped. (Per-consumer release would require a different
        # event shape.)
        dongle.consumer = None
        dongle.active_consumers.clear()
        dongle.freq_hz = None
        dongle.sample_rate = None
        dongle.band_id = None
        dongle.band_started_at = None  # v0.6.17
        dongle.fanout_clients = 0  # all decoders disconnected
        # v0.7.3: clear the per-peer set and the per-band recent
        # rings so when this dongle is reallocated to the next band
        # the detail pane shows fresh state instead of stale peers
        # and decodes from the previous band.
        dongle.fanout_client_peers.clear()
        dongle.recent_decodes.clear()
        dongle.recent_detections.clear()
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
        # v0.7.3: track per-peer set so the detail pane can list
        # actual clients ("rtl_433+rtlamr+lora_survey") instead of
        # just a count. The set is keyed on peer_addr (IP:port).
        if e.peer_addr:
            dongle.fanout_client_peers.add(e.peer_addr)
    elif e.event_type == "disconnect":
        dongle.fanout_clients = max(0, dongle.fanout_clients - 1)
        if e.peer_addr:
            dongle.fanout_client_peers.discard(e.peer_addr)
        # v0.6.17: accumulate bytes from this client's full session.
        # disconnect carries the lifetime byte count for the client.
        if e.bytes_sent > 0:
            dongle.fanout_bytes_sent += e.bytes_sent
    elif e.event_type == "slow":
        # v0.7.4: previous code computed
        #   fanout_dropped_chunks = max(prev, e.bytes_sent // 16384)
        # which mis-interpreted bytes_sent (the client's CUMULATIVE
        # bytes received, monotonically increasing) as a dropped-
        # chunk count. After a few minutes of streaming the counter
        # was in the millions even with no actual drops. The slow
        # event itself is the warning signal — increment by 1 per
        # incident instead.
        dongle.fanout_dropped_chunks += 1
        # Slow events are informative but rate-limited at source
        _push_stream(
            state, e, "warning", "fanout",
            f"{dongle_id}: client {e.peer_addr} slow",
        )
    elif e.event_type == "dropped":
        # v0.6.17: drop also carries the byte count up to the drop;
        # credit it to the dongle's transferred total.
        if e.bytes_sent > 0:
            dongle.fanout_bytes_sent += e.bytes_sent
        _push_stream(
            state, e, "error", "fanout",
            f"{dongle_id}: client {e.peer_addr} dropped",
        )


def _compact_decode_summary(e: DecodeEvent) -> str:
    """Tiny one-line summary for the per-dongle recent-decode list.

    Delegates to the per-protocol formatter for known protocols
    (meshtastic gets the from→to arrow, rtl_433 family gets msg_type
    + commodity, etc.) and falls back to a generic key=value dump
    for unknown protocols. Truncated to ~50 chars to fit the narrow
    detail-pane column."""
    try:
        from rfcensus.reporting.payload_format import format_payload
        s = format_payload(e.protocol, e.payload or {})
    except Exception:
        s = repr(e.payload)
    if len(s) > 50:
        s = s[:47] + "..."
    return s


def _reduce_decode(state: TUIState, e: DecodeEvent) -> None:
    state.total_decodes += 1
    # Find the dongle currently leased that's listening to e.freq_hz.
    # v0.7.3 fix: previously used a 100 kHz tolerance against the
    # dongle's tuned center, which is way too tight — rtl_433 emits
    # decodes with the SIGNAL frequency (e.g. 433.470 MHz when the
    # dongle is centered at 433.920 MHz with 2.4 MS/s sample rate),
    # so almost no decodes matched and per-dongle counters stayed at
    # 0-2 while the global counter showed 10+. The correct check is
    # whether the decode frequency falls inside the dongle's
    # instantaneous bandwidth (sample_rate ± a small edge guard for
    # filter rolloff). For 2.4 MS/s that's ±1.2 MHz, comfortably
    # covering the full 433 ISM activity.
    for d in state.dongles:
        if not d.consumer or not d.freq_hz:
            continue
        # Half-bandwidth with 5% rolloff guard. Default to ±1 MHz when
        # sample_rate isn't set yet (event arrives before first
        # HardwareEvent for the dongle); 1 MHz is a safer default than
        # 100 kHz and won't false-positive on the typical multi-band
        # dongle layout where centers are tens of MHz apart.
        half_bw = (d.sample_rate or 2_000_000) * 0.475
        if abs(d.freq_hz - e.freq_hz) <= half_bw:
            d.decodes_total += 1
            d.decodes_in_band += 1
            d.last_decode_at = e.timestamp
            # Per-dongle recent-decode ring (capacity 25). Used by
            # the dongle detail pane to render a "Recent decodes"
            # list alongside the bare counter.
            d.recent_decodes.append(_DongleDecodeEntry(
                timestamp=e.timestamp,
                freq_hz=e.freq_hz,
                protocol=e.protocol,
                summary=_compact_decode_summary(e),
            ))
            if len(d.recent_decodes) > 25:
                del d.recent_decodes[: len(d.recent_decodes) - 25]
            break
    # Don't push to the event stream — too noisy. The footer counter
    # is enough for protocols we don't have a dedicated view for.

    # Per-protocol ring buffers. Currently only Meshtastic gets one;
    # adding more is just a matter of writing the formatter and the
    # widget. The widget reads from this buffer once per render tick.
    if e.protocol == "meshtastic":
        from rfcensus.reporting.payload_format import format_payload
        payload = e.payload or {}
        entry = MeshtasticDecodeEntry(
            timestamp=e.timestamp,
            freq_hz=e.freq_hz,
            preset=str(payload.get("preset", "?")),
            crc_ok=bool(payload.get("crc_ok", False)),
            decrypted=bool(payload.get("decrypted", False)),
            channel_hash=payload.get("channel_hash"),
            from_node=payload.get("from_node"),
            to_node=payload.get("to_node"),
            summary=format_payload(e.protocol, payload),
            rssi_dbm=e.rssi_dbm,
            snr_db=e.snr_db,
        )
        state.meshtastic_recent.append(entry)
        # Trim FIFO when over capacity
        if len(state.meshtastic_recent) > state.meshtastic_recent_capacity:
            del state.meshtastic_recent[
                : len(state.meshtastic_recent)
                  - state.meshtastic_recent_capacity
            ]


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
    # v0.6.14: attribute per-dongle. DetectionEvent doesn't carry
    # dongle_id directly (intentional — detectors reason at the band
    # level), but lora_survey + similar publishers stuff `band_id`
    # into metadata. We map band_id → dongle via the in-flight
    # active_tasks dict (keyed (wave, band_id), but we don't know
    # which wave is current in the lookup, so we scan).
    band_id = e.metadata.get("band_id") if e.metadata else None
    if band_id:
        for (_w, b_id), task in state.active_tasks.items():
            if b_id == band_id:
                d = _ensure_dongle(state, task.dongle_id)
                d.detections_total += 1
                d.detections_in_band += 1
                d.last_detection_at = e.timestamp
                # v0.7.3: per-dongle recent-detection ring for the
                # detail pane's "Recent detections" list.
                d.recent_detections.append(_DongleDetectionEntry(
                    timestamp=e.timestamp,
                    freq_hz=e.freq_hz,
                    technology=e.technology,
                    confidence=e.confidence,
                ))
                if len(d.recent_detections) > 25:
                    del d.recent_detections[
                        : len(d.recent_detections) - 25
                    ]
                break
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


def seed_dongle_metadata(
    state: TUIState, dongle_id: str, *,
    model: str = "", antenna_id: str | None = None,
) -> None:
    """v0.6.17: pre-populate display-only metadata that comes from the
    static config (DongleConfig in site.toml) rather than from runtime
    events. Called once at TUIApp mount per declared dongle. Idempotent
    — safe to re-call if the config reloads.
    """
    d = _ensure_dongle(state, dongle_id)
    if model:
        d.model = model
    if antenna_id is not None:
        d.antenna_id = antenna_id


def _push_stream(
    state: TUIState, event: Event, severity: str, category: str, text: str,
) -> None:
    """Append an entry to every per-mode buffer where it's visible.

    v0.6.16: writes go to all 3 buffers simultaneously (chronological
    append). Memory cost: an entry visible in all 3 modes is duplicated
    3× — at ~150 bytes per entry × 5000 cap × 3 modes = ~2.3 MB worst-
    case. The duplication keeps the buffer logic dead simple and lets
    the renderer just index `streams[mode]` with no merge step.
    """
    entry = StreamEntry(
        timestamp=event.timestamp or datetime.now(timezone.utc),
        severity=severity,
        category=category,
        text=text,
        raw=event,
    )
    for mode, cats in FILTER_CATEGORIES.items():
        # Errors and warnings always pass through, regardless of mode.
        # Otherwise the entry only goes into modes whose category set
        # includes its category.
        if entry.severity in ("error", "warning") or entry.category in cats:
            buf = state.streams.setdefault(mode, [])
            buf.append(entry)
            if len(buf) > state.stream_capacity:
                # Drop oldest entries (front of list). Trim in chunks
                # of 10% to amortize the slice cost — without this we'd
                # do an O(n) memmove on every insert once the buffer
                # is full. With 10% trimming, amortized cost per insert
                # is O(1).
                drop_n = max(1, state.stream_capacity // 10)
                state.streams[mode] = buf[drop_n:]


# ────────────────────────────────────────────────────────────────────
# Filter selection (used by EventStream widget)
# ────────────────────────────────────────────────────────────────────


# Categories shown in each filter mode. "minimal" is the bare minimum
# to know if the scan is working; "filtered" (default) adds emitters
# and detection events; "verbose" adds task chatter, decoder output,
# and channel-occupancy events. v0.6.13: 'channel' moved out of
# 'filtered' — the rtl_power-derived "active channel at X MHz" lines
# fire 5-30/sec on a busy band and were drowning out the actually
# decoded emitters and detections that 'filtered' is meant to surface.
# Users who want the spectrum-occupancy view can press `f` to cycle to
# verbose. v0.6.14: 'wave' added to minimal — without it the user
# sees an empty stream during entire 12-min waves and reasonably
# concludes the TUI is broken; wave transition messages are the
# single most-important "the system is alive and progressing"
# heartbeat so they belong in every filter mode.
FILTER_CATEGORIES: dict[str, set[str]] = {
    "minimal": {"session", "hardware", "wave"},
    "filtered": {
        "session", "hardware", "wave", "emitter", "detection",
    },
    "verbose": {
        "session", "hardware", "wave", "task", "decode",
        "emitter", "detection", "channel", "fanout",
    },
}


def filter_stream(
    stream: list[StreamEntry], mode: str,
) -> list[StreamEntry]:
    """Filter entries by category for a given filter mode.

    v0.6.16 compatibility shim: with per-mode ring buffers in TUIState,
    callers should use `state.streams[mode]` directly to get an already-
    filtered chronological list. This function still exists for legacy
    callers (and tests) that pass any list of StreamEntry — it filters
    by category just as before. Errors and warnings always pass through.
    """
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
