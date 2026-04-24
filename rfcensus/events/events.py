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
    persistence_ratio: float = 0.0
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
    """Hardware state transition."""

    dongle_id: str = ""
    kind: Literal[
        "detected", "healthy", "degraded", "failed", "allocated", "released",
        "reconnected", "permanently_failed",
    ] = "detected"
    detail: str = ""


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
