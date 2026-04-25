"""Plain-dataclass records for database rows.

These are dumb data containers. Mapping between records and the storage
layer lives in `repositories.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class SessionRecord:
    id: int | None
    command: str
    started_at: datetime
    ended_at: datetime | None = None
    site_name: str | None = None
    config_snap: dict[str, Any] | None = None
    notes: str | None = None


@dataclass(slots=True)
class DongleRecord:
    id: str
    serial: str | None
    model: str
    driver: str
    capabilities: dict[str, Any]
    first_seen: datetime
    last_seen: datetime
    notes: str | None = None


@dataclass(slots=True)
class PowerSampleRecord:
    id: int | None
    session_id: int
    dongle_id: str
    timestamp: datetime
    freq_hz: int
    bin_width_hz: int
    power_dbm: float


@dataclass(slots=True)
class ActiveChannelRecord:
    id: int | None
    session_id: int
    freq_center_hz: int
    bandwidth_hz: int
    first_seen: datetime
    last_seen: datetime
    peak_power_dbm: float | None = None
    avg_power_dbm: float | None = None
    noise_floor_dbm: float | None = None
    classification: str | None = None
    persistence_ratio: float | None = None
    # v0.6.3: total samples observed at this bin during tracking.
    # Needed so consumers can gate on whether persistence_ratio has
    # accumulated enough evidence to be trustworthy (a 100%-persistent
    # channel with sample_count=2 is not the same as 100% with n=600).
    # Nullable for rows written by pre-v0.6.3 versions of rfcensus.
    sample_count: int | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DecodeRecord:
    id: int | None
    session_id: int
    dongle_id: str
    timestamp: datetime
    decoder: str
    protocol: str
    freq_hz: int
    rssi_dbm: float | None
    snr_db: float | None
    payload: dict[str, Any]
    raw_hex: str | None = None
    validated: bool = False
    validation_reasons: list[str] = field(default_factory=list)
    decoder_confidence: float = 1.0


@dataclass(slots=True)
class EmitterRecord:
    id: int | None
    protocol: str
    device_id: str
    device_id_hash: str
    classification: str | None
    first_seen: datetime
    last_seen: datetime
    observation_count: int = 0
    typical_freq_hz: int | None = None
    freq_variance: float | None = None
    typical_rssi_dbm: float | None = None
    rssi_variance: float | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnomalyRecord:
    id: int | None
    session_id: int | None
    detected_at: datetime
    kind: str
    freq_hz: int | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass(slots=True)
class DetectionRecord:
    id: int | None
    session_id: int | None
    detector: str
    technology: str
    freq_hz: int
    detected_at: datetime
    bandwidth_hz: int | None = None
    confidence: float = 0.0
    evidence: str = ""
    hand_off_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------
# Helpers for JSON round-trips
# ------------------------------------------------------------


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, separators=(",", ":"))


def json_loads(text: str | None) -> Any:
    if not text:
        return None
    return json.loads(text)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"can't serialize {type(obj)!r}")
