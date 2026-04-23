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
