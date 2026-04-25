"""Event types flowing through the rfcensus event bus.

All internal components communicate by publishing events to a shared bus.
Decoders emit `DecodeEvent`, spectrum backends emit `PowerSampleEvent`, the
emitter tracker emits `EmitterEvent` when it creates or updates emitters, etc.

UI consumers (TUI, web UI, report generators) subscribe to the events they
care about. The event bus decouples producers from consumers and lets us
swap UIs without touching pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class Event:
    """Base event. All events carry a timestamp and optional session id."""

    timestamp: datetime = field(default_factory=_utc_now)
    session_id: int | None = None


# ------------------------------------------------------------
# Spectrum / power scanning events
# ------------------------------------------------------------


@dataclass(slots=True)
class PowerSampleEvent(Event):
    """One FFT bin sample from a spectrum backend."""

    dongle_id: str = ""
    freq_hz: int = 0
    bin_width_hz: int = 0
    power_dbm: float = 0.0


@dataclass(slots=True)
class ActiveChannelEvent(Event):
    """An active channel has been identified (or its state changed).

    The occupancy analyzer emits these based on accumulated PowerSamples.
    """

    kind: Literal["new", "updated", "gone"] = "new"
    dongle_id: str = ""
    freq_center_hz: int = 0
    bandwidth_hz: int = 0
    peak_power_dbm: float = 0.0
    avg_power_dbm: float = 0.0
    noise_floor_dbm: float = 0.0
    snr_db: float = 0.0
    classification: str = "unknown"
    # v0.6.3: persistence_ratio is now the real occupancy ratio:
    # total_active_samples / total_samples observed for this bin.
    # Replaced the pre-v0.6.3 formula `min(1.0, sample_count / 60.0)`
    # which was just a sample-count cap and meant "persist=100%" was
    # reported for any bin seen for ≥60 sweeps regardless of how
    # often it was actually active.
    persistence_ratio: float = 0.0
    # Total number of power-scan samples observed at this bin while
    # the channel was tracked. Consumers should treat persistence_ratio
    # with caution below ~10 samples — at low n the ratio is coarse
    # (1/3, 2/3, 3/3 with no values in between).
    sample_count: int = 0
    confidence: float = 0.0


@dataclass(slots=True)
class WideChannelEvent(Event):
    """A composite wide-bandwidth channel inferred from coherent
    activity across multiple adjacent narrow bins.

    Emitted by `WideChannelAggregator`. Distinct from `ActiveChannelEvent`
    (which is always per-bin) because wide-bandwidth signals like LoRa
    (125/250/500 kHz) never fit in a single power-scan bin and never
    present as continuously-active single bins — a LoRa chirp sweeps
    across the whole channel in under a symbol period, so at any instant
    only a narrow slice is lit. Over a time window, all bins in the
    channel show transient activity.

    Consumers that care about wide-bandwidth signals (LoRa, Meshtastic,
    FM broadcast, DMR voice traffic, etc.) subscribe to this event;
    narrow-band detectors continue using `ActiveChannelEvent` as before.
    """

    dongle_id: str = ""
    # Center frequency of the composite channel (midpoint of constituent
    # bins' frequency span)
    freq_center_hz: int = 0
    # Composite bandwidth — matches the template that triggered emission
    # (e.g., 125_000 for LoRa SF7/125kHz channels)
    bandwidth_hz: int = 0
    # Which target template this composite matched. Useful for
    # downstream detectors that want to treat different widths
    # differently (e.g., Meshtastic commonly uses 250 kHz).
    matched_template_hz: int = 0
    # How many of the constituent narrow bins saw activity within the
    # aggregation window. Higher = more confident the signal really
    # spanned the full template width vs. a handful of narrow carriers
    # coincidentally near each other.
    constituent_bin_count: int = 0
    # What fraction of the template's frequency span was covered by
    # active bins (0.0 to 1.0). We emit when this exceeds a threshold
    # (typically 0.5) — partial coverage is expected because LoRa
    # chirps don't linger on every bin simultaneously.
    coverage_ratio: float = 0.0
    # Power statistics across all constituent bins during the window
    peak_power_dbm: float = 0.0
    avg_power_dbm: float = 0.0
    noise_floor_dbm: float = 0.0
    # Time span during which this activity was observed
    first_seen: datetime = field(default_factory=_utc_now)
    last_seen: datetime = field(default_factory=_utc_now)


# ------------------------------------------------------------
# Decode / emitter events
# ------------------------------------------------------------


@dataclass(slots=True)
class DecodeEvent(Event):
    """A decoder produced a frame."""

    decoder_name: str = ""
    protocol: str = ""
    dongle_id: str = ""
    freq_hz: int = 0
    rssi_dbm: float | None = None
    snr_db: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    raw_hex: str | None = None
    decoder_confidence: float = 1.0
    # Filled in by the validator
    validated: bool | None = None
    validation_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EmitterEvent(Event):
    """Emitter tracker emits these when emitters are created or updated."""

    kind: Literal["new", "confirmed", "updated", "decayed"] = "new"
    emitter_id: int = 0
    protocol: str = ""
    device_id_hash: str = ""
    classification: str = ""
    confidence: float = 0.0
    observation_count: int = 0
    typical_freq_hz: int = 0
    typical_rssi_dbm: float = 0.0


# ------------------------------------------------------------
# Anomaly / discovery events
# ------------------------------------------------------------


@dataclass(slots=True)
class AnomalyEvent(Event):
    """Something worth the user's attention."""

    kind: str = ""
    freq_hz: int | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DetectionEvent(Event):
    """A detector has recognized a known technology on a channel.

    Distinct from `AnomalyEvent`: an anomaly is "something unexplained,"
    a detection is "I recognize this signal as protocol X — consider
    handing off to specialized tool Y for full analysis."
    """

    detector_name: str = ""
    technology: str = ""  # "lora", "p25_control", "wifi_bt_ism", etc.
    freq_hz: int = 0
    bandwidth_hz: int = 0
    confidence: float = 0.0
    evidence: str = ""
    hand_off_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------
# Hardware / session lifecycle events
# ------------------------------------------------------------


@dataclass(slots=True)
class HardwareEvent(Event):
    """Hardware state transition.

    For `kind="allocated"` and `kind="released"` events, the optional
    `freq_hz`, `sample_rate`, and `consumer` fields carry the per-lease
    info the TUI needs to render a dongle's current state without
    parsing the freeform `detail` string. They default to safe values
    so existing publishers that only set `dongle_id` + `kind` keep
    working unchanged.

    For other `kind` values these fields are typically unset and the
    `detail` string carries the human-readable message.
    """

    dongle_id: str = ""
    kind: Literal[
        "detected", "healthy", "degraded", "failed", "allocated", "released",
        "reconnected", "permanently_failed",
    ] = "detected"
    detail: str = ""
    # v0.6.4: structured fields for TUI consumption. The dashboard's
    # DongleStrip uses these to render per-tile state (current freq,
    # sample rate, what consumer holds the lease) without needing to
    # parse `detail`. Optional everywhere — None means "not applicable
    # to this event kind" rather than "unknown".
    freq_hz: int | None = None
    sample_rate: int | None = None
    consumer: str | None = None
    # Optional band identifier (e.g. "915_ism") so the TUI can show
    # which band is currently being scanned on this dongle. Producers
    # that don't have a band context (e.g. raw probe / health checks)
    # leave this None.
    band_id: str | None = None


@dataclass(slots=True)
class FanoutClientEvent(Event):
    """A multi-client rtl_tcp fanout slot transitioned state.

    The fanout serves one upstream rtl_tcp source to N downstream
    decoders. The TUI's per-dongle detail view shows fanout client
    activity to indicate which decoders are currently consuming from
    a shared dongle.

    `event_type` values:
      • `"connect"`   — a new downstream client attached
      • `"disconnect"` — a downstream client cleanly detached
      • `"slow"`      — a downstream client is consuming slower than
                        upstream produces; backpressure / drop risk
      • `"dropped"`   — a downstream client was dropped due to slow
                        consumption (the fanout's last-resort defense
                        against unbounded buffering)
    """

    slot_id: str = ""
    peer_addr: str = ""
    event_type: Literal["connect", "disconnect", "slow", "dropped"] = "connect"
    bytes_sent: int = 0


@dataclass(slots=True)
class DecoderFailureEvent(Event):
    """A decoder run ended unexpectedly early in a way that suggests
    hardware loss (e.g. USB unplug, dongle reset). The session uses
    these to schedule retries when the dongle reconnects."""

    band_id: str = ""
    dongle_id: str = ""
    decoder_name: str = ""
    elapsed_s: float = 0.0
    remaining_s: float = 0.0


@dataclass(slots=True)
class SessionEvent(Event):
    """Session lifecycle transition."""

    kind: Literal["started", "ended", "phase_changed"] = "started"
    phase: str = ""
    detail: str = ""


# ────────────────────────────────────────────────────────────────────
# v0.6.5: Plan events for the TUI dashboard's plan-tree widget.
#
# These events expose the scheduler's wave-by-wave plan and execution
# progress to subscribers without requiring them to poll the scheduler.
# The TUI's PlanTree subscribes to these and renders an updating tree
# view; future consumers (web UI, log analyzer, monitoring sidecar)
# can subscribe without adding polling code.
#
# Publish order:
#   1. PlanReadyEvent — once at start, after the scheduler finalizes
#      the wave plan
#   2. WaveStartedEvent  — at the top of each wave loop iteration
#   3. TaskStartedEvent  — for each task as it begins
#   4. TaskCompletedEvent — for each task as it ends
#   5. WaveCompletedEvent — at the end of each wave loop iteration
# ────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PlanReadyEvent(Event):
    """The scheduler has computed an execution plan.

    Published once at the start of a session, before any waves run.
    Consumers (TUI, logs, etc.) use this to render the plan up front.

    `waves` is a list of dicts with at least:
      • index: int — wave number (0-based)
      • task_count: int
      • task_summaries: list[str] — short "band_id→dongle_id" strings,
        one per task in the wave, in scheduled order
    The dict shape (rather than a dedicated WaveSpec dataclass) keeps
    the event lightweight and avoids importing scheduler internals into
    the events module.
    """

    waves: list[dict] = field(default_factory=list)
    total_tasks: int = 0
    max_parallel_per_wave: int = 0


@dataclass(slots=True)
class WaveStartedEvent(Event):
    """A wave is about to begin executing its tasks."""

    wave_index: int = 0
    task_count: int = 0
    pass_n: int = 0  # Pass number (for hybrid mode's repeat passes)


@dataclass(slots=True)
class WaveCompletedEvent(Event):
    """A wave has finished — all its tasks have ended (success or error).

    `errors` lists per-task error strings; an empty list means all tasks
    succeeded. `task_count` and `successful_count` together let the TUI
    show a "5/7 succeeded" summary without re-counting from task events.
    """

    wave_index: int = 0
    pass_n: int = 0
    task_count: int = 0
    successful_count: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaskStartedEvent(Event):
    """A single task within a wave has started.

    `consumer` is the same string passed to broker.allocate(consumer=...)
    so consumers can correlate with HardwareEvent allocate/release.
    """

    wave_index: int = 0
    pass_n: int = 0
    band_id: str = ""
    dongle_id: str = ""
    consumer: str = ""


@dataclass(slots=True)
class TaskCompletedEvent(Event):
    """A single task has finished.

    `status` is one of:
      • "ok"       — task ran and returned normally
      • "failed"   — strategy reported errors but didn't crash
      • "crashed"  — strategy raised an exception
      • "skipped"  — task was skipped (e.g. unassigned dongle, user skip)
      • "timeout"  — task hit its per-wave deadline
    `detail` carries a short human-readable explanation when relevant.
    """

    wave_index: int = 0
    pass_n: int = 0
    band_id: str = ""
    dongle_id: str = ""
    consumer: str = ""
    status: Literal["ok", "failed", "crashed", "skipped", "timeout"] = "ok"
    detail: str = ""
