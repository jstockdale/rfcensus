"""JSON export format for machine consumption."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

from rfcensus.engine.session import SessionResult
from rfcensus.reporting.privacy import scrub_emitter
from rfcensus.storage.models import (
    ActiveChannelRecord,
    AnomalyRecord,
    DetectionRecord,
    EmitterRecord,
)


def render_json_report(
    result: SessionResult,
    emitters: list[EmitterRecord],
    anomalies: list[AnomalyRecord],
    detections: list[DetectionRecord] | None = None,
    active_channels: list[ActiveChannelRecord] | None = None,
    *,
    include_ids: bool = False,
) -> str:
    detections = detections or []
    active_channels = active_channels or []
    payload: dict[str, Any] = {
        "session": {
            "id": result.session_id,
            "started_at": result.started_at.isoformat(),
            "ended_at": result.ended_at.isoformat(),
            "total_decodes": result.total_decodes,
        },
        "plan": {
            "waves": [
                {
                    "index": w.index,
                    "tasks": [
                        {
                            "band_id": t.band.id,
                            "band_name": t.band.name,
                            "dongle_id": t.suggested_dongle_id,
                            "antenna_id": t.suggested_antenna_id,
                            "notes": t.notes,
                        }
                        for t in w.tasks
                    ],
                }
                for w in result.plan.waves
            ],
            "warnings": list(result.plan.warnings),
            "unassigned": list(result.plan.unassigned),
        },
        "emitters": [
            _emitter_to_dict(e, include_ids) for e in emitters
        ],
        "anomalies": [
            {
                "id": a.id,
                "kind": a.kind,
                "freq_hz": a.freq_hz,
                "description": a.description,
                "detected_at": a.detected_at.isoformat(),
                "metadata": a.metadata,
            }
            for a in anomalies
        ],
        # v0.5.36: detections (previously present in the text report
        # but missing from JSON) and active_channels (newly surfaced).
        "detections": [
            {
                "id": d.id,
                "detector": d.detector,
                "technology": d.technology,
                "freq_hz": d.freq_hz,
                "bandwidth_hz": d.bandwidth_hz,
                "confidence": d.confidence,
                "evidence": d.evidence,
                "hand_off_tools": list(d.hand_off_tools),
                "detected_at": d.detected_at.isoformat(),
                "metadata": d.metadata,
            }
            for d in detections
        ],
        "active_channels": [
            {
                "id": ch.id,
                "freq_center_hz": ch.freq_center_hz,
                "bandwidth_hz": ch.bandwidth_hz,
                "peak_power_dbm": ch.peak_power_dbm,
                "avg_power_dbm": ch.avg_power_dbm,
                "noise_floor_dbm": ch.noise_floor_dbm,
                "classification": ch.classification,
                "persistence_ratio": ch.persistence_ratio,
                "confidence": ch.confidence,
                "first_seen": ch.first_seen.isoformat(),
                "last_seen": ch.last_seen.isoformat(),
                "metadata": ch.metadata,
            }
            for ch in active_channels
        ],
        "strategy_results": [
            {
                "band_id": sr.band_id,
                "decoders_run": sr.decoders_run,
                "power_scan_performed": sr.power_scan_performed,
                "decodes_emitted": sr.decodes_emitted,
                "errors": sr.errors,
            }
            for sr in result.strategy_results
        ],
    }
    return json.dumps(payload, indent=2, default=_json_default)


def _emitter_to_dict(e: EmitterRecord, include_ids: bool) -> dict[str, Any]:
    display = scrub_emitter(e, include_raw_ids=include_ids)
    return {
        "id": display.id,
        "protocol": display.protocol,
        "device_id": display.device_id,
        "device_id_hash": display.device_id_hash,
        "classification": display.classification,
        "first_seen": display.first_seen.isoformat(),
        "last_seen": display.last_seen.isoformat(),
        "observation_count": display.observation_count,
        "typical_freq_hz": display.typical_freq_hz,
        "typical_rssi_dbm": display.typical_rssi_dbm,
        "confidence": display.confidence,
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj.__dict__
    raise TypeError(f"not JSON-serializable: {type(obj)}")
