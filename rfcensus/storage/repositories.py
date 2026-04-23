"""Repository classes: thin query helpers for each table.

Each repo handles the mapping between dataclass records and rows. They
don't do any business logic. Higher layers (emitter tracker, inventory
engine) compose these.
"""

from __future__ import annotations

from datetime import datetime

from rfcensus.storage.db import Database
from rfcensus.storage.models import (
    ActiveChannelRecord,
    AnomalyRecord,
    DecodeRecord,
    DetectionRecord,
    DongleRecord,
    EmitterRecord,
    PowerSampleRecord,
    SessionRecord,
    json_dumps,
    json_loads,
)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _dt(text: str) -> datetime:
    return datetime.fromisoformat(text)


# ------------------------------------------------------------
# Sessions
# ------------------------------------------------------------


class SessionRepo:
    def __init__(self, db: Database):
        self.db = db

    async def create(self, record: SessionRecord) -> int:
        cur = await self.db.execute(
            """
            INSERT INTO sessions (command, started_at, site_name, config_snap, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.command,
                _iso(record.started_at),
                record.site_name,
                json_dumps(record.config_snap) if record.config_snap else None,
                record.notes,
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]

    async def end(self, session_id: int, ended_at: datetime) -> None:
        await self.db.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (_iso(ended_at), session_id),
        )

    async def attach_hardware(
        self,
        session_id: int,
        dongle_id: str,
        antenna_id: str | None,
        role: str,
    ) -> None:
        await self.db.execute(
            """
            INSERT OR REPLACE INTO session_hardware
                (session_id, dongle_id, antenna_id, role)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, dongle_id, antenna_id, role),
        )

    async def recent(self, limit: int = 20) -> list[SessionRecord]:
        rows = await self.db.fetchall(
            """
            SELECT id, command, started_at, ended_at, site_name, config_snap, notes
            FROM sessions ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        )
        return [
            SessionRecord(
                id=row["id"],
                command=row["command"],
                started_at=_dt(row["started_at"]),
                ended_at=_dt(row["ended_at"]) if row["ended_at"] else None,
                site_name=row["site_name"],
                config_snap=json_loads(row["config_snap"]),
                notes=row["notes"],
            )
            for row in rows
        ]

    async def by_id(self, session_id: int) -> SessionRecord | None:
        row = await self.db.fetchone(
            """
            SELECT id, command, started_at, ended_at, site_name, config_snap, notes
            FROM sessions WHERE id = ?
            """,
            (session_id,),
        )
        if row is None:
            return None
        return SessionRecord(
            id=row["id"],
            command=row["command"],
            started_at=_dt(row["started_at"]),
            ended_at=_dt(row["ended_at"]) if row["ended_at"] else None,
            site_name=row["site_name"],
            config_snap=json_loads(row["config_snap"]),
            notes=row["notes"],
        )

    async def mark_baseline(self, session_id: int, is_baseline: bool = True) -> bool:
        """Mark or unmark a session as the reference baseline for this site."""
        cur = await self.db.execute(
            "UPDATE sessions SET is_baseline = ? WHERE id = ?",
            (int(is_baseline), session_id),
        )
        return cur.rowcount > 0

    async def current_baseline(self, site_name: str | None = None) -> SessionRecord | None:
        """Return the most recent session marked as baseline (optionally for a site)."""
        if site_name:
            row = await self.db.fetchone(
                """
                SELECT id, command, started_at, ended_at, site_name, config_snap, notes
                FROM sessions WHERE is_baseline = 1 AND site_name = ?
                ORDER BY started_at DESC LIMIT 1
                """,
                (site_name,),
            )
        else:
            row = await self.db.fetchone(
                """
                SELECT id, command, started_at, ended_at, site_name, config_snap, notes
                FROM sessions WHERE is_baseline = 1
                ORDER BY started_at DESC LIMIT 1
                """
            )
        if row is None:
            return None
        return SessionRecord(
            id=row["id"],
            command=row["command"],
            started_at=_dt(row["started_at"]),
            ended_at=_dt(row["ended_at"]) if row["ended_at"] else None,
            site_name=row["site_name"],
            config_snap=json_loads(row["config_snap"]),
            notes=row["notes"],
        )


# ------------------------------------------------------------
# Dongles
# ------------------------------------------------------------


class DongleRepo:
    def __init__(self, db: Database):
        self.db = db

    async def upsert(self, record: DongleRecord) -> None:
        await self.db.execute(
            """
            INSERT INTO dongles (id, serial, model, driver, capabilities, first_seen, last_seen, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                serial = excluded.serial,
                model = excluded.model,
                driver = excluded.driver,
                capabilities = excluded.capabilities,
                last_seen = excluded.last_seen,
                notes = excluded.notes
            """,
            (
                record.id,
                record.serial,
                record.model,
                record.driver,
                json_dumps(record.capabilities),
                _iso(record.first_seen),
                _iso(record.last_seen),
                record.notes,
            ),
        )

    async def all(self) -> list[DongleRecord]:
        rows = await self.db.fetchall(
            """
            SELECT id, serial, model, driver, capabilities, first_seen, last_seen, notes
            FROM dongles ORDER BY first_seen
            """
        )
        return [
            DongleRecord(
                id=row["id"],
                serial=row["serial"],
                model=row["model"],
                driver=row["driver"],
                capabilities=json_loads(row["capabilities"]) or {},
                first_seen=_dt(row["first_seen"]),
                last_seen=_dt(row["last_seen"]),
                notes=row["notes"],
            )
            for row in rows
        ]


# ------------------------------------------------------------
# Power samples (very high volume – batched writes)
# ------------------------------------------------------------


class PowerSampleRepo:
    def __init__(self, db: Database):
        self.db = db

    async def insert_many(self, records: list[PowerSampleRecord]) -> None:
        if not records:
            return
        params = [
            (
                r.session_id,
                r.dongle_id,
                _iso(r.timestamp),
                r.freq_hz,
                r.bin_width_hz,
                r.power_dbm,
            )
            for r in records
        ]
        await self.db.executemany(
            """
            INSERT INTO power_samples
                (session_id, dongle_id, timestamp, freq_hz, bin_width_hz, power_dbm)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            params,
        )

    async def prune_older_than(self, cutoff: datetime) -> int:
        cur = await self.db.execute(
            "DELETE FROM power_samples WHERE timestamp < ?", (_iso(cutoff),)
        )
        return cur.rowcount


# ------------------------------------------------------------
# Active channels
# ------------------------------------------------------------


class ActiveChannelRepo:
    def __init__(self, db: Database):
        self.db = db

    async def upsert(self, record: ActiveChannelRecord) -> int:
        if record.id is not None:
            await self.db.execute(
                """
                UPDATE active_channels SET
                    last_seen = ?,
                    peak_power_dbm = ?,
                    avg_power_dbm = ?,
                    noise_floor_dbm = ?,
                    classification = ?,
                    persistence_ratio = ?,
                    confidence = ?,
                    metadata = ?
                WHERE id = ?
                """,
                (
                    _iso(record.last_seen),
                    record.peak_power_dbm,
                    record.avg_power_dbm,
                    record.noise_floor_dbm,
                    record.classification,
                    record.persistence_ratio,
                    record.confidence,
                    json_dumps(record.metadata),
                    record.id,
                ),
            )
            return record.id

        cur = await self.db.execute(
            """
            INSERT INTO active_channels
                (session_id, freq_center_hz, bandwidth_hz, first_seen, last_seen,
                 peak_power_dbm, avg_power_dbm, noise_floor_dbm, classification,
                 persistence_ratio, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.session_id,
                record.freq_center_hz,
                record.bandwidth_hz,
                _iso(record.first_seen),
                _iso(record.last_seen),
                record.peak_power_dbm,
                record.avg_power_dbm,
                record.noise_floor_dbm,
                record.classification,
                record.persistence_ratio,
                record.confidence,
                json_dumps(record.metadata),
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]

    async def find_by_center(
        self, session_id: int, freq_center_hz: int, tolerance_hz: int = 0
    ) -> ActiveChannelRecord | None:
        """Look up an active_channel row for this session centered at freq_center_hz."""
        if tolerance_hz == 0:
            row = await self.db.fetchone(
                """
                SELECT * FROM active_channels
                WHERE session_id = ? AND freq_center_hz = ?
                LIMIT 1
                """,
                (session_id, freq_center_hz),
            )
        else:
            row = await self.db.fetchone(
                """
                SELECT * FROM active_channels
                WHERE session_id = ?
                  AND freq_center_hz BETWEEN ? AND ?
                ORDER BY ABS(freq_center_hz - ?) LIMIT 1
                """,
                (
                    session_id,
                    freq_center_hz - tolerance_hz,
                    freq_center_hz + tolerance_hz,
                    freq_center_hz,
                ),
            )
        if row is None:
            return None
        return ActiveChannelRecord(
            id=row["id"],
            session_id=row["session_id"],
            freq_center_hz=row["freq_center_hz"],
            bandwidth_hz=row["bandwidth_hz"],
            first_seen=_dt(row["first_seen"]),
            last_seen=_dt(row["last_seen"]),
            peak_power_dbm=row["peak_power_dbm"],
            avg_power_dbm=row["avg_power_dbm"],
            noise_floor_dbm=row["noise_floor_dbm"],
            classification=row["classification"],
            persistence_ratio=row["persistence_ratio"],
            confidence=row["confidence"],
            metadata=json_loads(row["metadata"]) or {},
        )

    async def for_session(self, session_id: int) -> list[ActiveChannelRecord]:
        rows = await self.db.fetchall(
            "SELECT * FROM active_channels WHERE session_id = ? ORDER BY freq_center_hz",
            (session_id,),
        )
        return [
            ActiveChannelRecord(
                id=row["id"],
                session_id=row["session_id"],
                freq_center_hz=row["freq_center_hz"],
                bandwidth_hz=row["bandwidth_hz"],
                first_seen=_dt(row["first_seen"]),
                last_seen=_dt(row["last_seen"]),
                peak_power_dbm=row["peak_power_dbm"],
                avg_power_dbm=row["avg_power_dbm"],
                noise_floor_dbm=row["noise_floor_dbm"],
                classification=row["classification"],
                persistence_ratio=row["persistence_ratio"],
                confidence=row["confidence"],
                metadata=json_loads(row["metadata"]) or {},
            )
            for row in rows
        ]


# ------------------------------------------------------------
# Decodes
# ------------------------------------------------------------


class DecodeRepo:
    def __init__(self, db: Database):
        self.db = db

    async def insert(self, record: DecodeRecord) -> int:
        cur = await self.db.execute(
            """
            INSERT INTO decodes
                (session_id, dongle_id, timestamp, decoder, protocol, freq_hz,
                 rssi_dbm, snr_db, payload, raw_hex, validated, validation_reasons,
                 decoder_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.session_id,
                record.dongle_id,
                _iso(record.timestamp),
                record.decoder,
                record.protocol,
                record.freq_hz,
                record.rssi_dbm,
                record.snr_db,
                json_dumps(record.payload),
                record.raw_hex,
                int(record.validated),
                json_dumps(record.validation_reasons),
                record.decoder_confidence,
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]

    async def for_session(
        self, session_id: int, validated_only: bool = False
    ) -> list[DecodeRecord]:
        where = "WHERE session_id = ?"
        params: tuple = (session_id,)
        if validated_only:
            where += " AND validated = 1"
        rows = await self.db.fetchall(
            f"SELECT * FROM decodes {where} ORDER BY timestamp", params
        )
        return [_decode_from_row(row) for row in rows]


def _decode_from_row(row) -> DecodeRecord:
    return DecodeRecord(
        id=row["id"],
        session_id=row["session_id"],
        dongle_id=row["dongle_id"],
        timestamp=_dt(row["timestamp"]),
        decoder=row["decoder"],
        protocol=row["protocol"],
        freq_hz=row["freq_hz"],
        rssi_dbm=row["rssi_dbm"],
        snr_db=row["snr_db"],
        payload=json_loads(row["payload"]) or {},
        raw_hex=row["raw_hex"],
        validated=bool(row["validated"]),
        validation_reasons=json_loads(row["validation_reasons"]) or [],
        decoder_confidence=row["decoder_confidence"],
    )


# ------------------------------------------------------------
# Emitters
# ------------------------------------------------------------


class EmitterRepo:
    def __init__(self, db: Database):
        self.db = db

    async def find(self, protocol: str, device_id: str) -> EmitterRecord | None:
        row = await self.db.fetchone(
            "SELECT * FROM emitters WHERE protocol = ? AND device_id = ?",
            (protocol, device_id),
        )
        if row is None:
            return None
        return _emitter_from_row(row)

    async def insert(self, record: EmitterRecord) -> int:
        cur = await self.db.execute(
            """
            INSERT INTO emitters
                (protocol, device_id, device_id_hash, classification,
                 first_seen, last_seen, observation_count,
                 typical_freq_hz, freq_variance, typical_rssi_dbm, rssi_variance,
                 confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.protocol,
                record.device_id,
                record.device_id_hash,
                record.classification,
                _iso(record.first_seen),
                _iso(record.last_seen),
                record.observation_count,
                record.typical_freq_hz,
                record.freq_variance,
                record.typical_rssi_dbm,
                record.rssi_variance,
                record.confidence,
                json_dumps(record.metadata),
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]

    async def update(self, record: EmitterRecord) -> None:
        if record.id is None:
            raise ValueError("cannot update emitter without id")
        await self.db.execute(
            """
            UPDATE emitters SET
                last_seen = ?,
                observation_count = ?,
                typical_freq_hz = ?,
                freq_variance = ?,
                typical_rssi_dbm = ?,
                rssi_variance = ?,
                confidence = ?,
                classification = ?,
                metadata = ?
            WHERE id = ?
            """,
            (
                _iso(record.last_seen),
                record.observation_count,
                record.typical_freq_hz,
                record.freq_variance,
                record.typical_rssi_dbm,
                record.rssi_variance,
                record.confidence,
                record.classification,
                json_dumps(record.metadata),
                record.id,
            ),
        )

    async def link_observation(self, decode_id: int, emitter_id: int) -> None:
        await self.db.execute(
            """
            INSERT OR IGNORE INTO observations (decode_id, emitter_id)
            VALUES (?, ?)
            """,
            (decode_id, emitter_id),
        )

    async def for_session(self, session_id: int) -> list[EmitterRecord]:
        rows = await self.db.fetchall(
            """
            SELECT e.* FROM emitters e
            WHERE EXISTS (
                SELECT 1 FROM observations o
                JOIN decodes d ON o.decode_id = d.id
                WHERE o.emitter_id = e.id AND d.session_id = ?
            )
            ORDER BY e.protocol, e.confidence DESC
            """,
            (session_id,),
        )
        return [_emitter_from_row(row) for row in rows]

    async def all(
        self, *, min_confidence: float = 0.0, protocol: str | None = None
    ) -> list[EmitterRecord]:
        query = "SELECT * FROM emitters WHERE confidence >= ?"
        params: list = [min_confidence]
        if protocol:
            query += " AND protocol = ?"
            params.append(protocol)
        query += " ORDER BY protocol, confidence DESC"
        rows = await self.db.fetchall(query, tuple(params))
        return [_emitter_from_row(row) for row in rows]


def _emitter_from_row(row) -> EmitterRecord:
    return EmitterRecord(
        id=row["id"],
        protocol=row["protocol"],
        device_id=row["device_id"],
        device_id_hash=row["device_id_hash"],
        classification=row["classification"],
        first_seen=_dt(row["first_seen"]),
        last_seen=_dt(row["last_seen"]),
        observation_count=row["observation_count"],
        typical_freq_hz=row["typical_freq_hz"],
        freq_variance=row["freq_variance"],
        typical_rssi_dbm=row["typical_rssi_dbm"],
        rssi_variance=row["rssi_variance"],
        confidence=row["confidence"],
        metadata=json_loads(row["metadata"]) or {},
    )


# ------------------------------------------------------------
# Anomalies
# ------------------------------------------------------------


class AnomalyRepo:
    def __init__(self, db: Database):
        self.db = db

    async def insert(self, record: AnomalyRecord) -> int:
        cur = await self.db.execute(
            """
            INSERT INTO anomalies
                (session_id, detected_at, kind, freq_hz, description, metadata, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.session_id,
                _iso(record.detected_at),
                record.kind,
                record.freq_hz,
                record.description,
                json_dumps(record.metadata),
                int(record.resolved),
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]

    async def for_session(self, session_id: int) -> list[AnomalyRecord]:
        rows = await self.db.fetchall(
            "SELECT * FROM anomalies WHERE session_id = ? ORDER BY detected_at",
            (session_id,),
        )
        return [
            AnomalyRecord(
                id=row["id"],
                session_id=row["session_id"],
                detected_at=_dt(row["detected_at"]),
                kind=row["kind"],
                freq_hz=row["freq_hz"],
                description=row["description"],
                metadata=json_loads(row["metadata"]) or {},
                resolved=bool(row["resolved"]),
            )
            for row in rows
        ]


# ------------------------------------------------------------
# Detections
# ------------------------------------------------------------


class DetectionRepo:
    def __init__(self, db: Database):
        self.db = db

    async def insert(self, record: DetectionRecord) -> int:
        cur = await self.db.execute(
            """
            INSERT INTO detections (session_id, detector, technology, freq_hz,
                                    bandwidth_hz, confidence, evidence,
                                    hand_off_tools, detected_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.session_id,
                record.detector,
                record.technology,
                record.freq_hz,
                record.bandwidth_hz,
                record.confidence,
                record.evidence,
                json_dumps(record.hand_off_tools),
                _iso(record.detected_at),
                json_dumps(record.metadata),
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]

    async def for_session(self, session_id: int) -> list[DetectionRecord]:
        rows = await self.db.fetchall(
            "SELECT * FROM detections WHERE session_id = ? ORDER BY detected_at",
            (session_id,),
        )
        return [_row_to_detection(r) for r in rows]

    async def all(self, technology: str | None = None) -> list[DetectionRecord]:
        if technology:
            rows = await self.db.fetchall(
                "SELECT * FROM detections WHERE technology = ? ORDER BY detected_at DESC",
                (technology,),
            )
        else:
            rows = await self.db.fetchall(
                "SELECT * FROM detections ORDER BY detected_at DESC"
            )
        return [_row_to_detection(r) for r in rows]


def _row_to_detection(row) -> DetectionRecord:
    return DetectionRecord(
        id=row["id"],
        session_id=row["session_id"],
        detector=row["detector"],
        technology=row["technology"],
        freq_hz=row["freq_hz"],
        bandwidth_hz=row["bandwidth_hz"],
        confidence=row["confidence"],
        evidence=row["evidence"] or "",
        hand_off_tools=json_loads(row["hand_off_tools"]) or [],
        detected_at=_dt(row["detected_at"]),
        metadata=json_loads(row["metadata"]) or {},
    )
