"""Emitter tracker.

Validated DecodeEvents are fed into the tracker. It:

• Looks up or creates an EmitterRecord keyed by (protocol, device_id)
• Updates rolling stats (observation count, avg freq, avg RSSI)
• Adjusts confidence based on validation outcomes
• Emits EmitterEvent when an emitter is created, confirmed, or updated

Storage is SQLite via `EmitterRepo`; we cache in memory for the session
to avoid a round-trip per decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rfcensus.analysis.validator import DecodeValidator, _device_id_from_payload
from rfcensus.events import DecodeEvent, EmitterEvent, EventBus
from rfcensus.storage.models import DecodeRecord, EmitterRecord
from rfcensus.storage.repositories import DecodeRepo, EmitterRepo
from rfcensus.utils.hashing import hash_id
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class _RunningStats:
    """Welford-style running mean/variance over one numerical stream."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def observe(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)


@dataclass
class _EmitterState:
    id: int | None
    protocol: str
    device_id: str
    classification: str
    first_seen: datetime
    last_seen: datetime
    observation_count: int
    confidence: float
    freq_stats: _RunningStats
    rssi_stats: _RunningStats
    metadata: dict[str, Any]


class EmitterTracker:
    """Turns validated decodes into persistent emitters."""

    def __init__(
        self,
        event_bus: EventBus,
        decode_repo: DecodeRepo,
        emitter_repo: EmitterRepo,
        validator: DecodeValidator,
        salt: str,
        min_confirmations: int = 3,
    ):
        self.event_bus = event_bus
        self.decode_repo = decode_repo
        self.emitter_repo = emitter_repo
        self.validator = validator
        self.salt = salt
        self.min_confirmations = min_confirmations
        self._cache: dict[tuple[str, str], _EmitterState] = {}
        self._confirmed: set[tuple[str, str]] = set()

    async def handle_decode(self, event: DecodeEvent, session_id: int) -> None:
        """Called for every DecodeEvent. Validates, persists, updates emitter."""
        # Validate
        result = self.validator.validate(event)
        event.validated = result.accept
        event.validation_reasons = result.reasons

        # Persist the raw decode regardless of validation outcome
        record = DecodeRecord(
            id=None,
            session_id=session_id,
            dongle_id=event.dongle_id,
            timestamp=event.timestamp,
            decoder=event.decoder_name,
            protocol=event.protocol,
            freq_hz=event.freq_hz,
            rssi_dbm=event.rssi_dbm,
            snr_db=event.snr_db,
            payload=event.payload,
            raw_hex=event.raw_hex,
            validated=result.accept,
            validation_reasons=result.reasons,
            decoder_confidence=event.decoder_confidence,
        )
        decode_id = await self.decode_repo.insert(record)

        if not result.accept:
            log.debug(
                "rejected decode: %s/%s reasons=%s",
                event.decoder_name,
                event.protocol,
                result.reasons,
            )
            return

        device_id = _device_id_from_payload(event.payload)
        if not device_id:
            # No identifier; can't track as an emitter
            return

        key = (event.protocol, device_id)
        state = self._cache.get(key)
        now = event.timestamp

        if state is None:
            existing = await self.emitter_repo.find(event.protocol, device_id)
            if existing is None:
                # Create new emitter
                device_hash = hash_id(device_id, self.salt)
                new_record = EmitterRecord(
                    id=None,
                    protocol=event.protocol,
                    device_id=device_id,
                    device_id_hash=device_hash,
                    classification=_classify_emitter(event.protocol, event.payload),
                    first_seen=now,
                    last_seen=now,
                    observation_count=1,
                    typical_freq_hz=event.freq_hz,
                    typical_rssi_dbm=event.rssi_dbm,
                    confidence=0.15 + result.confidence_delta,
                    metadata={},
                )
                emitter_id = await self.emitter_repo.insert(new_record)
                state = _EmitterState(
                    id=emitter_id,
                    protocol=event.protocol,
                    device_id=device_id,
                    classification=new_record.classification or "unknown",
                    first_seen=now,
                    last_seen=now,
                    observation_count=1,
                    confidence=new_record.confidence,
                    freq_stats=_RunningStats(),
                    rssi_stats=_RunningStats(),
                    metadata={},
                )
                state.freq_stats.observe(event.freq_hz)
                if event.rssi_dbm is not None:
                    state.rssi_stats.observe(event.rssi_dbm)
                self._cache[key] = state
                await self.emitter_repo.link_observation(decode_id, emitter_id)
                await self.event_bus.publish(
                    EmitterEvent(
                        session_id=session_id,
                        kind="new",
                        emitter_id=emitter_id,
                        protocol=event.protocol,
                        device_id_hash=device_hash,
                        classification=state.classification,
                        confidence=state.confidence,
                        observation_count=1,
                        typical_freq_hz=event.freq_hz,
                        typical_rssi_dbm=event.rssi_dbm or 0.0,
                    )
                )
                return
            # Rehydrate existing emitter
            state = _EmitterState(
                id=existing.id,
                protocol=existing.protocol,
                device_id=existing.device_id,
                classification=existing.classification or "unknown",
                first_seen=existing.first_seen,
                last_seen=existing.last_seen,
                observation_count=existing.observation_count,
                confidence=existing.confidence,
                freq_stats=_RunningStats(),
                rssi_stats=_RunningStats(),
                metadata=existing.metadata,
            )
            self._cache[key] = state

        # Update existing emitter
        state.observation_count += 1
        state.last_seen = now
        state.freq_stats.observe(event.freq_hz)
        if event.rssi_dbm is not None:
            state.rssi_stats.observe(event.rssi_dbm)
        state.confidence = min(1.0, state.confidence + result.confidence_delta)

        record_out = EmitterRecord(
            id=state.id,
            protocol=state.protocol,
            device_id=state.device_id,
            device_id_hash=hash_id(state.device_id, self.salt),
            classification=state.classification,
            first_seen=state.first_seen,
            last_seen=state.last_seen,
            observation_count=state.observation_count,
            typical_freq_hz=int(state.freq_stats.mean),
            freq_variance=state.freq_stats.variance,
            typical_rssi_dbm=state.rssi_stats.mean if state.rssi_stats.n else None,
            rssi_variance=state.rssi_stats.variance,
            confidence=state.confidence,
            metadata=state.metadata,
        )
        await self.emitter_repo.update(record_out)
        if state.id is not None:
            await self.emitter_repo.link_observation(decode_id, state.id)

        # Confirm if threshold reached
        if (
            state.observation_count >= self.min_confirmations
            and key not in self._confirmed
        ):
            self._confirmed.add(key)
            await self.event_bus.publish(
                EmitterEvent(
                    session_id=session_id,
                    kind="confirmed",
                    emitter_id=state.id or 0,
                    protocol=state.protocol,
                    device_id_hash=record_out.device_id_hash,
                    classification=state.classification,
                    confidence=state.confidence,
                    observation_count=state.observation_count,
                    typical_freq_hz=record_out.typical_freq_hz or 0,
                    typical_rssi_dbm=record_out.typical_rssi_dbm or 0.0,
                )
            )
        else:
            await self.event_bus.publish(
                EmitterEvent(
                    session_id=session_id,
                    kind="updated",
                    emitter_id=state.id or 0,
                    protocol=state.protocol,
                    device_id_hash=record_out.device_id_hash,
                    classification=state.classification,
                    confidence=state.confidence,
                    observation_count=state.observation_count,
                    typical_freq_hz=record_out.typical_freq_hz or 0,
                    typical_rssi_dbm=record_out.typical_rssi_dbm or 0.0,
                )
            )


def _classify_emitter(protocol: str, payload: dict[str, Any]) -> str:
    """Map a protocol + payload to a human-friendly emitter classification."""
    if protocol == "tpms":
        return "tpms_sensor"
    if protocol in ("ert_scm", "ert_scm_plus", "ert_idm"):
        commodity = payload.get("commodity")
        # ERT commodity codes: 4,5,7,8 electric; 2,12 gas; 3,11,13 water
        if commodity in (4, 5, 7, 8):
            return "electric_meter"
        if commodity in (2, 12):
            return "gas_meter"
        if commodity in (3, 11, 13):
            return "water_meter"
        return "utility_meter"
    if protocol == "r900" or protocol == "r900_bcd":
        return "water_meter"
    if protocol in ("ais_class_a", "ais_class_b"):
        return "vessel"
    if protocol == "aprs":
        return "amateur_station"
    if protocol == "pocsag":
        return "pager_recipient"
    if protocol == "flex":
        return "pager_recipient"
    if protocol == "weather_station":
        return "weather_station"
    if protocol == "keyfob":
        return "keyfob"
    if protocol == "doorbell":
        return "doorbell"
    if "security" in protocol:
        return "security_sensor"
    return "unknown_emitter"
