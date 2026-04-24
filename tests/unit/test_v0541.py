"""v0.5.41 integration test: the full pipeline from LoRa detection to
queue submission works end to end.

  WideChannelEvent
   → LoraDetector.on_wide_channel
   → DetectionEvent (with needs_iq_confirmation=True)
   → DetectionWriter.handle
      → DetectionRepo.insert → detection_id
      → queue.submit(ConfirmationTask(detection_id, ...))

Also validates DetectionRepo.update_metadata() against a real in-memory
DB, which is the write-back path the confirmation task uses.
"""

from __future__ import annotations

import pytest

from rfcensus.detectors.builtin.lora import LoraDetector
from rfcensus.engine.confirmation_queue import ConfirmationQueue
from rfcensus.events import DetectionEvent, EventBus, WideChannelEvent
from rfcensus.storage.db import Database
from rfcensus.storage.repositories import DetectionRepo
from rfcensus.storage.writer import DetectionWriter


class TestDetectionAutoSubmits:
    @pytest.mark.asyncio
    async def test_lora_detection_reaches_queue(self):
        """End-to-end: WideChannelEvent → DetectionEvent → DB insert →
        queue submission with the DB-assigned detection_id."""
        # In-memory DB (migrations run automatically on first use)
        db = Database(":memory:")
        repo = DetectionRepo(db)
        queue = ConfirmationQueue()

        bus = EventBus()
        # Use the real DetectionWriter with the real queue
        writer = DetectionWriter(repo, session_id=1, confirmation_queue=queue)
        bus.subscribe(DetectionEvent, writer.handle)

        # Also need a session row so foreign keys are satisfied
        from rfcensus.storage.models import SessionRecord
        from rfcensus.storage.repositories import SessionRepo
        from datetime import datetime, timezone
        session_repo = SessionRepo(db)
        await session_repo.create(
            SessionRecord(
                id=None, command="test",
                started_at=datetime.now(timezone.utc),
            )
        )

        # Attach LoRa detector
        detector = LoraDetector()
        detector.attach(bus=bus, session_id=1, iq_service=None)

        # Fire a wide-channel event
        await bus.publish(
            WideChannelEvent(
                session_id=1,
                freq_center_hz=906_875_000,
                bandwidth_hz=250_000,
                matched_template_hz=250_000,
                constituent_bin_count=10,
                coverage_ratio=0.90,
            )
        )
        # Drain cascades: WideChannel -> DetectionEvent -> writer (async
        # DB insert + queue submit). Call drain multiple times to let
        # the downstream handler complete.
        await bus.drain()
        import asyncio as _asyncio
        await _asyncio.sleep(0.05)
        await bus.drain()

        # Queue should have received a ConfirmationTask
        assert queue.pending_count() == 1

        # And the DB has the detection row with needs_iq_confirmation=True
        detections = await repo.for_session(1)
        assert len(detections) == 1
        det = detections[0]
        assert det.metadata.get("needs_iq_confirmation") is True
        assert det.metadata.get("estimated_sf") is None  # deferred

    @pytest.mark.asyncio
    async def test_update_metadata_backfills_sf_and_bumps_confidence(self):
        """The confirmation task runner calls repo.update_metadata
        with SF/variant/iq_confirmed. Verify the row is correctly
        updated and confidence bumps."""
        db = Database(":memory:")
        repo = DetectionRepo(db)
        from rfcensus.storage.models import DetectionRecord
        from datetime import datetime, timezone

        # Session row for FK
        from rfcensus.storage.models import SessionRecord
        from rfcensus.storage.repositories import SessionRepo
        session_repo = SessionRepo(db)
        await session_repo.create(
            SessionRecord(
                id=None, command="test",
                started_at=datetime.now(timezone.utc),
            )
        )

        # Insert a LoRa detection with no SF
        record = DetectionRecord(
            id=None,
            session_id=1,
            detector="lora",
            technology="lora",
            freq_hz=906_875_000,
            bandwidth_hz=250_000,
            confidence=0.70,
            evidence="test",
            hand_off_tools=[],
            detected_at=datetime.now(timezone.utc),
            metadata={"needs_iq_confirmation": True, "estimated_sf": None},
        )
        det_id = await repo.insert(record)

        # Backfill with a successful confirmation
        await repo.update_metadata(
            detection_id=det_id,
            estimated_sf=11,
            variant="meshtastic_long_fast",
            iq_confirmed=True,
            chirp_confidence=0.95,
        )

        # Read back
        results = await repo.for_session(1)
        assert len(results) == 1
        updated = results[0]
        assert updated.metadata["estimated_sf"] == 11
        assert updated.metadata["variant"] == "meshtastic_long_fast"
        assert updated.metadata["iq_confirmed"] is True
        # Confidence bumped by 0.15
        assert updated.confidence == pytest.approx(0.85, abs=0.01)

    @pytest.mark.asyncio
    async def test_update_metadata_is_idempotent_on_missing(self):
        """Updating a nonexistent detection_id should not raise."""
        db = Database(":memory:")
        repo = DetectionRepo(db)
        # Don't raise, just log a warning
        await repo.update_metadata(
            detection_id=999_999,
            estimated_sf=11,
        )  # no exception
