"""Tests for the storage writer consumers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rfcensus.events import ActiveChannelEvent, AnomalyEvent, EventBus, PowerSampleEvent
from rfcensus.storage import attach_writers
from rfcensus.storage.models import SessionRecord
from rfcensus.storage.repositories import (
    ActiveChannelRepo,
    AnomalyRepo,
    PowerSampleRepo,
    SessionRepo,
)


@pytest.mark.asyncio
class TestStorageWriters:
    async def test_active_channel_event_persists(self, db):
        session_id = await SessionRepo(db).create(
            SessionRecord(
                id=None, command="test", started_at=datetime.now(timezone.utc)
            )
        )

        bus = EventBus()
        attach_writers(
            bus=bus,
            session_id=session_id,
            active_channel_repo=ActiveChannelRepo(db),
            power_sample_repo=PowerSampleRepo(db),
            anomaly_repo=AnomalyRepo(db),
            capture_power=False,
        )

        await bus.publish(
            ActiveChannelEvent(
                session_id=session_id,
                kind="new",
                dongle_id="d1",
                freq_center_hz=915_000_000,
                bandwidth_hz=10_000,
                peak_power_dbm=-40.0,
                avg_power_dbm=-45.0,
                noise_floor_dbm=-75.0,
                snr_db=35.0,
                classification="pulsed",
                confidence=0.5,
            )
        )
        await bus.drain(timeout=2.0)

        rows = await ActiveChannelRepo(db).for_session(session_id)
        assert len(rows) == 1
        assert rows[0].freq_center_hz == 915_000_000
        assert rows[0].classification == "pulsed"

    async def test_anomaly_event_persists(self, db):
        session_id = await SessionRepo(db).create(
            SessionRecord(
                id=None, command="test", started_at=datetime.now(timezone.utc)
            )
        )

        bus = EventBus()
        attach_writers(
            bus=bus,
            session_id=session_id,
            active_channel_repo=ActiveChannelRepo(db),
            power_sample_repo=PowerSampleRepo(db),
            anomaly_repo=AnomalyRepo(db),
        )

        await bus.publish(
            AnomalyEvent(
                session_id=session_id,
                kind="unknown_carrier",
                freq_hz=320_020_000,
                description="persistent narrowband",
            )
        )
        await bus.drain(timeout=2.0)

        rows = await AnomalyRepo(db).for_session(session_id)
        assert len(rows) == 1
        assert rows[0].kind == "unknown_carrier"

    async def test_power_samples_opt_in(self, db):
        """Power samples should NOT be persisted unless capture_power=True."""
        session_id = await SessionRepo(db).create(
            SessionRecord(
                id=None, command="test", started_at=datetime.now(timezone.utc)
            )
        )

        bus = EventBus()
        batcher = attach_writers(
            bus=bus,
            session_id=session_id,
            active_channel_repo=ActiveChannelRepo(db),
            power_sample_repo=PowerSampleRepo(db),
            anomaly_repo=AnomalyRepo(db),
            capture_power=False,  # Not enabled
        )
        assert batcher is None

        await bus.publish(
            PowerSampleEvent(
                session_id=session_id,
                dongle_id="d1",
                freq_hz=100_000_000,
                bin_width_hz=10_000,
                power_dbm=-55.0,
            )
        )
        await bus.drain(timeout=1.0)

        rows = await db.fetchall(
            "SELECT * FROM power_samples WHERE session_id = ?", (session_id,)
        )
        assert len(rows) == 0

    async def test_power_samples_opt_in_enabled(self, db):
        session_id = await SessionRepo(db).create(
            SessionRecord(
                id=None, command="test", started_at=datetime.now(timezone.utc)
            )
        )

        bus = EventBus()
        batcher = attach_writers(
            bus=bus,
            session_id=session_id,
            active_channel_repo=ActiveChannelRepo(db),
            power_sample_repo=PowerSampleRepo(db),
            anomaly_repo=AnomalyRepo(db),
            capture_power=True,
        )
        assert batcher is not None

        for i in range(10):
            await bus.publish(
                PowerSampleEvent(
                    session_id=session_id,
                    dongle_id="d1",
                    freq_hz=100_000_000 + i * 10_000,
                    bin_width_hz=10_000,
                    power_dbm=-55.0 + i,
                )
            )
        await bus.drain(timeout=2.0)
        await batcher.stop()  # Flushes pending batch

        rows = await db.fetchall(
            "SELECT * FROM power_samples WHERE session_id = ?", (session_id,)
        )
        assert len(rows) == 10
