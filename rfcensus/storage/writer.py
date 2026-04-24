"""Storage writer.

Subscribes to the event bus and persists selected events to SQLite.
Writing is intentionally decoupled from producers — decoders and
spectrum backends just publish; this module persists.

Three consumers are bundled here:

• `ActiveChannelWriter` — `ActiveChannelEvent` → `active_channels` rows
• `PowerSampleBatcher` — `PowerSampleEvent` → `power_samples` rows (batched;
  enabled only if `capture_power` is true)
• `AnomalyWriter` — `AnomalyEvent` → `anomalies` rows

The emitter/decode writer lives in `analysis.tracker` because it's tightly
coupled with emitter logic. That split is deliberate: infrastructure-style
persistence here, business-logic persistence there.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from rfcensus.events import (
    ActiveChannelEvent,
    AnomalyEvent,
    DetectionEvent,
    EventBus,
    PowerSampleEvent,
)
from rfcensus.storage.models import (
    ActiveChannelRecord,
    AnomalyRecord,
    DetectionRecord,
    PowerSampleRecord,
)
from rfcensus.storage.repositories import (
    ActiveChannelRepo,
    AnomalyRepo,
    DetectionRepo,
    PowerSampleRepo,
)
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


class ActiveChannelWriter:
    """Persists ActiveChannelEvents as active_channels rows.

    An ActiveChannelEvent with kind='new' creates a row; kind='updated'
    updates the existing row's last_seen/peak_power; kind='gone' is a
    final update fixing the persistence stats.
    """

    def __init__(self, repo: ActiveChannelRepo, session_id: int):
        self.repo = repo
        self.session_id = session_id
        # Map of (session_id, freq_center_hz) → row id to avoid a SELECT per event
        self._id_cache: dict[tuple[int, int], int] = {}

    async def handle(self, event: ActiveChannelEvent) -> None:
        # Only persist events for this session
        if event.session_id and event.session_id != self.session_id:
            return
        key = (self.session_id, event.freq_center_hz)

        if event.kind == "new" or key not in self._id_cache:
            # Double-check the DB in case another consumer raced us
            existing = await self.repo.find_by_center(
                self.session_id, event.freq_center_hz
            )
            if existing is not None:
                self._id_cache[key] = existing.id or 0
                record = ActiveChannelRecord(
                    id=existing.id,
                    session_id=self.session_id,
                    freq_center_hz=event.freq_center_hz,
                    bandwidth_hz=event.bandwidth_hz,
                    first_seen=existing.first_seen,
                    last_seen=event.timestamp,
                    peak_power_dbm=max(
                        existing.peak_power_dbm or event.peak_power_dbm,
                        event.peak_power_dbm,
                    ),
                    avg_power_dbm=event.avg_power_dbm,
                    noise_floor_dbm=event.noise_floor_dbm,
                    classification=event.classification,
                    persistence_ratio=event.persistence_ratio,
                    confidence=event.confidence,
                    metadata={"dongle_id": event.dongle_id},
                )
                await self.repo.upsert(record)
                return

            record = ActiveChannelRecord(
                id=None,
                session_id=self.session_id,
                freq_center_hz=event.freq_center_hz,
                bandwidth_hz=event.bandwidth_hz,
                first_seen=event.timestamp,
                last_seen=event.timestamp,
                peak_power_dbm=event.peak_power_dbm,
                avg_power_dbm=event.avg_power_dbm,
                noise_floor_dbm=event.noise_floor_dbm,
                classification=event.classification,
                persistence_ratio=event.persistence_ratio,
                confidence=event.confidence,
                metadata={"dongle_id": event.dongle_id},
            )
            row_id = await self.repo.upsert(record)
            self._id_cache[key] = row_id
            return

        # updated / gone
        row_id = self._id_cache.get(key)
        if row_id is None:
            return
        record = ActiveChannelRecord(
            id=row_id,
            session_id=self.session_id,
            freq_center_hz=event.freq_center_hz,
            bandwidth_hz=event.bandwidth_hz,
            first_seen=event.timestamp,  # Ignored by update path; repo uses last_seen only
            last_seen=event.timestamp,
            peak_power_dbm=event.peak_power_dbm,
            avg_power_dbm=event.avg_power_dbm,
            noise_floor_dbm=event.noise_floor_dbm,
            classification=event.classification,
            persistence_ratio=event.persistence_ratio,
            confidence=event.confidence,
            metadata={"dongle_id": event.dongle_id},
        )
        await self.repo.upsert(record)


class PowerSampleBatcher:
    """Batches PowerSampleEvents and flushes them to power_samples.

    Power samples arrive at potentially thousands per second. Writing
    one row per event would hammer the database lock. We buffer up to
    `batch_size` rows or `flush_interval_s` seconds and write in one go.
    """

    def __init__(
        self,
        repo: PowerSampleRepo,
        session_id: int,
        *,
        batch_size: int = 500,
        flush_interval_s: float = 2.0,
    ):
        self.repo = repo
        self.session_id = session_id
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self._buffer: list[PowerSampleRecord] = []
        self._flush_task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    def start(self) -> None:
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(
                self._flush_loop(), name="power-sample-flush"
            )

    async def handle(self, event: PowerSampleEvent) -> None:
        if event.session_id and event.session_id != self.session_id:
            return
        self._buffer.append(
            PowerSampleRecord(
                id=None,
                session_id=self.session_id,
                dongle_id=event.dongle_id,
                timestamp=event.timestamp,
                freq_hz=event.freq_hz,
                bin_width_hz=event.bin_width_hz,
                power_dbm=event.power_dbm,
            )
        )
        if len(self._buffer) >= self.batch_size:
            await self._flush()

    async def _flush(self) -> None:
        if not self._buffer:
            return
        batch, self._buffer = self._buffer, []
        try:
            await self.repo.insert_many(batch)
        except Exception:
            log.exception("power sample flush failed (dropping %d rows)", len(batch))

    async def _flush_loop(self) -> None:
        try:
            while not self._stopping.is_set():
                try:
                    await asyncio.wait_for(
                        self._stopping.wait(), timeout=self.flush_interval_s
                    )
                except TimeoutError:
                    pass
                await self._flush()
        except asyncio.CancelledError:
            await self._flush()
            raise

    async def stop(self) -> None:
        self._stopping.set()
        if self._flush_task is not None:
            try:
                await asyncio.wait_for(self._flush_task, timeout=5.0)
            except TimeoutError:
                self._flush_task.cancel()
        await self._flush()


class AnomalyWriter:
    """Persists AnomalyEvents to the anomalies table."""

    def __init__(self, repo: AnomalyRepo, session_id: int):
        self.repo = repo
        self.session_id = session_id

    async def handle(self, event: AnomalyEvent) -> None:
        record = AnomalyRecord(
            id=None,
            session_id=self.session_id,
            detected_at=event.timestamp,
            kind=event.kind,
            freq_hz=event.freq_hz,
            description=event.description,
            metadata=dict(event.metadata),
        )
        await self.repo.insert(record)


class DetectionWriter:
    """Persists DetectionEvents to the detections table.

    v0.5.41: if a confirmation queue is wired in AND the detection's
    metadata contains `needs_iq_confirmation=True`, submits a
    ConfirmationTask to the queue using the freshly-assigned
    detection_id. This is how LoRa-family detections get deferred
    SF/variant classification without the detector itself needing
    direct access to the queue.
    """

    def __init__(
        self,
        repo: DetectionRepo,
        session_id: int,
        confirmation_queue: "ConfirmationQueue | None" = None,
    ):
        self.repo = repo
        self.session_id = session_id
        self.confirmation_queue = confirmation_queue

    async def handle(self, event: DetectionEvent) -> None:
        record = DetectionRecord(
            id=None,
            session_id=self.session_id,
            detector=event.detector_name,
            technology=event.technology,
            freq_hz=event.freq_hz,
            bandwidth_hz=event.bandwidth_hz,
            confidence=event.confidence,
            evidence=event.evidence,
            hand_off_tools=list(event.hand_off_tools),
            detected_at=event.timestamp,
            metadata=dict(event.metadata),
        )
        detection_id = await self.repo.insert(record)

        # v0.5.41: auto-submit to confirmation queue if requested
        needs_confirmation = bool(event.metadata.get("needs_iq_confirmation"))
        if needs_confirmation and self.confirmation_queue is not None:
            from rfcensus.engine.confirmation_queue import ConfirmationTask
            task = ConfirmationTask(
                detection_id=detection_id,
                freq_hz=event.freq_hz,
                bandwidth_hz=event.bandwidth_hz or 0,
                technology=event.technology,
                detector_name=event.detector_name,
            )
            await self.confirmation_queue.submit(task)


def attach_writers(
    bus: EventBus,
    session_id: int,
    active_channel_repo: ActiveChannelRepo,
    power_sample_repo: PowerSampleRepo | None,
    anomaly_repo: AnomalyRepo,
    detection_repo: DetectionRepo | None = None,
    *,
    capture_power: bool = False,
    confirmation_queue: "ConfirmationQueue | None" = None,
) -> PowerSampleBatcher | None:
    """Subscribe persistence consumers to the bus.

    Returns the PowerSampleBatcher if `capture_power=True` so the caller
    can stop it at shutdown.

    v0.5.41: if `confirmation_queue` is provided AND `detection_repo`
    is available, DetectionEvents whose metadata contains
    `needs_iq_confirmation=True` are automatically submitted to the
    queue after persistence (so the ConfirmationTask carries the real
    detection_id, not a placeholder).
    """
    channel_writer = ActiveChannelWriter(active_channel_repo, session_id)
    bus.subscribe(ActiveChannelEvent, channel_writer.handle)

    anomaly_writer = AnomalyWriter(anomaly_repo, session_id)
    bus.subscribe(AnomalyEvent, anomaly_writer.handle)

    if detection_repo is not None:
        detection_writer = DetectionWriter(
            detection_repo, session_id,
            confirmation_queue=confirmation_queue,
        )
        bus.subscribe(DetectionEvent, detection_writer.handle)

    batcher: PowerSampleBatcher | None = None
    if capture_power and power_sample_repo is not None:
        batcher = PowerSampleBatcher(power_sample_repo, session_id)
        batcher.start()
        bus.subscribe(PowerSampleEvent, batcher.handle)

    return batcher
