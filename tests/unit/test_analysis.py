"""Tests for analysis layer: validator and emitter tracker."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rfcensus.analysis.tracker import EmitterTracker
from rfcensus.analysis.validator import DecodeValidator, _device_id_from_payload
from rfcensus.config.schema import ValidationConfig
from rfcensus.events import DecodeEvent, EmitterEvent, EventBus
from rfcensus.storage.models import SessionRecord
from rfcensus.storage.repositories import DecodeRepo, EmitterRepo, SessionRepo


class TestValidator:
    def test_accepts_good_decode(self):
        v = DecodeValidator(ValidationConfig())
        e = DecodeEvent(
            decoder_name="rtl_433", protocol="tpms",
            freq_hz=433920000, rssi_dbm=-45.0, snr_db=12.0,
            payload={"_device_id": "abc123"},
        )
        result = v.validate(e)
        assert result.accept
        assert result.confidence_delta > 0

    def test_rejects_saturated_rssi(self):
        v = DecodeValidator(ValidationConfig())
        e = DecodeEvent(
            decoder_name="rtl_433", protocol="tpms",
            rssi_dbm=-0.1, snr_db=5.0,
            payload={"_device_id": "abc123"},
        )
        result = v.validate(e)
        assert not result.accept
        assert "compression" in " ".join(result.reasons)

    def test_rejects_suspicious_id_pattern(self):
        v = DecodeValidator(ValidationConfig())
        e = DecodeEvent(
            decoder_name="rtl_433", protocol="tpms",
            rssi_dbm=-50.0, snr_db=10.0,
            payload={"_device_id": "ffffff"},
        )
        result = v.validate(e)
        assert not result.accept

    def test_rejects_constant_digit_id(self):
        v = DecodeValidator(ValidationConfig())
        e = DecodeEvent(
            decoder_name="rtl_433", protocol="tpms",
            rssi_dbm=-50.0, snr_db=10.0,
            payload={"_device_id": "9999"},
        )
        result = v.validate(e)
        assert not result.accept
        assert "constant-digit" in " ".join(result.reasons)

    def test_low_snr_soft_accept(self):
        v = DecodeValidator(ValidationConfig(min_snr_db=6.0))
        e = DecodeEvent(
            decoder_name="rtl_433", protocol="tpms",
            rssi_dbm=-50.0, snr_db=2.0,
            payload={"_device_id": "abc123"},
        )
        result = v.validate(e)
        assert result.accept  # soft accept
        assert result.confidence_delta < 0.1  # but reduced confidence

    def test_strong_snr_boosts_confidence(self):
        v = DecodeValidator(ValidationConfig())
        e = DecodeEvent(
            decoder_name="rtl_433", protocol="tpms",
            rssi_dbm=-30.0, snr_db=22.0,
            payload={"_device_id": "abc123"},
        )
        result = v.validate(e)
        assert result.accept
        assert result.confidence_delta > 0.1

    def test_rate_limit_triggers_on_flood(self):
        # Very low rate limit to make testing easy
        v = DecodeValidator(ValidationConfig(max_decodes_per_minute_per_decoder=3))
        accepted = 0
        for _ in range(20):
            e = DecodeEvent(
                decoder_name="rtl_433", protocol="tpms",
                rssi_dbm=-50.0, snr_db=10.0,
                payload={"_device_id": "abc123"},
            )
            r = v.validate(e)
            if r.accept:
                accepted += 1
        # Should be at most 3 (burst) that pass
        assert accepted <= 3


class TestDeviceIdExtraction:
    def test_extracts_underscore_device_id(self):
        assert _device_id_from_payload({"_device_id": "X"}) == "X"

    def test_extracts_mmsi(self):
        assert _device_id_from_payload({"mmsi": 123456}) == "123456"

    def test_extracts_callsign(self):
        assert _device_id_from_payload({"callsign": "K6ABC"}) == "K6ABC"

    def test_returns_none_when_no_id(self):
        assert _device_id_from_payload({"model": "x"}) is None

    def test_prefers_underscore_key(self):
        # _device_id wins over ID field
        payload = {"_device_id": "canonical", "id": "other"}
        assert _device_id_from_payload(payload) == "canonical"


@pytest.mark.asyncio
class TestEmitterTracker:
    async def test_creates_new_emitter_on_first_observation(self, db, salt):
        session_repo = SessionRepo(db)
        session_id = await session_repo.create(SessionRecord(
            id=None, command="test", started_at=datetime.now(timezone.utc),
        ))

        bus = EventBus()
        validator = DecodeValidator(ValidationConfig())
        tracker = EmitterTracker(
            bus, DecodeRepo(db), EmitterRepo(db), validator, salt=salt,
        )

        evt = DecodeEvent(
            decoder_name="rtl_433", protocol="tpms",
            freq_hz=433920000, rssi_dbm=-45.0, snr_db=12.0,
            payload={"_device_id": "abc123"},
        )
        await tracker.handle_decode(evt, session_id)

        emitters = await EmitterRepo(db).for_session(session_id)
        assert len(emitters) == 1
        assert emitters[0].protocol == "tpms"
        assert emitters[0].observation_count == 1

    async def test_confirmed_after_threshold(self, db, salt):
        session_repo = SessionRepo(db)
        session_id = await session_repo.create(SessionRecord(
            id=None, command="test", started_at=datetime.now(timezone.utc),
        ))

        bus = EventBus()
        events_captured: list[tuple[str, int]] = []

        async def capture(e: EmitterEvent):
            events_captured.append((e.kind, e.observation_count))

        bus.subscribe(EmitterEvent, capture)

        validator = DecodeValidator(ValidationConfig())
        tracker = EmitterTracker(
            bus, DecodeRepo(db), EmitterRepo(db), validator, salt=salt,
            min_confirmations=3,
        )

        for _ in range(5):
            evt = DecodeEvent(
                decoder_name="rtl_433", protocol="tpms",
                freq_hz=433920000, rssi_dbm=-45.0, snr_db=12.0,
                payload={"_device_id": "abc123"},
            )
            await tracker.handle_decode(evt, session_id)

        await bus.drain(timeout=2.0)
        kinds = [k for k, _ in events_captured]
        assert "new" in kinds
        assert "confirmed" in kinds
        # Confirmed only fires once
        assert kinds.count("confirmed") == 1

    async def test_hashes_device_id(self, db, salt):
        session_repo = SessionRepo(db)
        session_id = await session_repo.create(SessionRecord(
            id=None, command="test", started_at=datetime.now(timezone.utc),
        ))

        bus = EventBus()
        tracker = EmitterTracker(
            bus, DecodeRepo(db), EmitterRepo(db),
            DecodeValidator(ValidationConfig()),
            salt=salt,
        )

        evt = DecodeEvent(
            decoder_name="rtl_433", protocol="tpms",
            freq_hz=433920000, rssi_dbm=-45.0, snr_db=12.0,
            payload={"_device_id": "raw-secret-id"},
        )
        await tracker.handle_decode(evt, session_id)

        emitters = await EmitterRepo(db).for_session(session_id)
        assert emitters[0].device_id == "raw-secret-id"  # raw in DB
        assert emitters[0].device_id_hash != "raw-secret-id"  # but hash is different
        assert len(emitters[0].device_id_hash) == 12
