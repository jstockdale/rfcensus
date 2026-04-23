"""Privacy helpers for report generation."""

from __future__ import annotations

from rfcensus.storage.models import EmitterRecord


def scrub_emitter(record: EmitterRecord, include_raw_ids: bool) -> EmitterRecord:
    """Return a copy of the emitter with device_id replaced by hash unless opted in."""
    if include_raw_ids:
        return record
    # We mutate a shallow copy to avoid touching the cached instance.
    scrubbed = EmitterRecord(
        id=record.id,
        protocol=record.protocol,
        device_id=f"hash:{record.device_id_hash}",
        device_id_hash=record.device_id_hash,
        classification=record.classification,
        first_seen=record.first_seen,
        last_seen=record.last_seen,
        observation_count=record.observation_count,
        typical_freq_hz=record.typical_freq_hz,
        freq_variance=record.freq_variance,
        typical_rssi_dbm=record.typical_rssi_dbm,
        rssi_variance=record.rssi_variance,
        confidence=record.confidence,
        metadata=dict(record.metadata),
    )
    return scrubbed
