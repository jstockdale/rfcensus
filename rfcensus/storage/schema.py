"""SQLite schema for rfcensus.

We use sqlite3 directly instead of an ORM. The schema is small, the
queries are simple, and the dependency footprint stays minimal. Migrations
are applied by numeric version number; each `_V{N}` function runs if the
database's PRAGMA user_version is less than N.
"""

from __future__ import annotations

import sqlite3
from typing import Final

SCHEMA_VERSION: Final[int] = 2


def apply_migrations(conn: sqlite3.Connection) -> None:
    """Bring the database up to SCHEMA_VERSION."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current < 1:
        _v1(conn)
        conn.execute("PRAGMA user_version = 1")
    if current < 2:
        _v2(conn)
        conn.execute("PRAGMA user_version = 2")
    conn.commit()


def _v1(conn: sqlite3.Connection) -> None:
    """Initial schema."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            command      TEXT NOT NULL,
            started_at   TEXT NOT NULL,
            ended_at     TEXT,
            site_name    TEXT,
            config_snap  TEXT,
            notes        TEXT
        );

        CREATE TABLE IF NOT EXISTS dongles (
            id            TEXT PRIMARY KEY,
            serial        TEXT UNIQUE,
            model         TEXT NOT NULL,
            driver        TEXT NOT NULL,
            capabilities  TEXT NOT NULL,
            first_seen    TEXT NOT NULL,
            last_seen     TEXT NOT NULL,
            notes         TEXT
        );

        CREATE TABLE IF NOT EXISTS session_hardware (
            session_id  INTEGER NOT NULL REFERENCES sessions(id),
            dongle_id   TEXT NOT NULL REFERENCES dongles(id),
            antenna_id  TEXT,
            role        TEXT,
            PRIMARY KEY (session_id, dongle_id)
        );

        CREATE TABLE IF NOT EXISTS power_samples (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    INTEGER NOT NULL REFERENCES sessions(id),
            dongle_id     TEXT NOT NULL,
            timestamp     TEXT NOT NULL,
            freq_hz       INTEGER NOT NULL,
            bin_width_hz  INTEGER NOT NULL,
            power_dbm     REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_power_session_freq
            ON power_samples(session_id, freq_hz);
        CREATE INDEX IF NOT EXISTS idx_power_session_time
            ON power_samples(session_id, timestamp);

        CREATE TABLE IF NOT EXISTS active_channels (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id          INTEGER NOT NULL REFERENCES sessions(id),
            freq_center_hz      INTEGER NOT NULL,
            bandwidth_hz        INTEGER NOT NULL,
            first_seen          TEXT NOT NULL,
            last_seen           TEXT NOT NULL,
            peak_power_dbm      REAL,
            avg_power_dbm       REAL,
            noise_floor_dbm     REAL,
            classification      TEXT,
            persistence_ratio   REAL,
            confidence          REAL,
            metadata            TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_channels_session
            ON active_channels(session_id);
        CREATE INDEX IF NOT EXISTS idx_channels_session_freq
            ON active_channels(session_id, freq_center_hz);

        CREATE TABLE IF NOT EXISTS decodes (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id          INTEGER NOT NULL REFERENCES sessions(id),
            dongle_id           TEXT NOT NULL,
            timestamp           TEXT NOT NULL,
            decoder             TEXT NOT NULL,
            protocol            TEXT NOT NULL,
            freq_hz             INTEGER NOT NULL,
            rssi_dbm            REAL,
            snr_db              REAL,
            payload             TEXT NOT NULL,
            raw_hex             TEXT,
            validated           INTEGER DEFAULT 0,
            validation_reasons  TEXT,
            decoder_confidence  REAL DEFAULT 1.0
        );
        CREATE INDEX IF NOT EXISTS idx_decodes_session_decoder
            ON decodes(session_id, decoder);
        CREATE INDEX IF NOT EXISTS idx_decodes_session_time
            ON decodes(session_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_decodes_protocol_time
            ON decodes(protocol, timestamp);

        CREATE TABLE IF NOT EXISTS emitters (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            protocol           TEXT NOT NULL,
            device_id          TEXT NOT NULL,
            device_id_hash     TEXT NOT NULL,
            classification     TEXT,
            first_seen         TEXT NOT NULL,
            last_seen          TEXT NOT NULL,
            observation_count  INTEGER DEFAULT 0,
            typical_freq_hz    INTEGER,
            freq_variance      REAL,
            typical_rssi_dbm   REAL,
            rssi_variance      REAL,
            confidence         REAL DEFAULT 0.0,
            metadata           TEXT,
            UNIQUE(protocol, device_id)
        );
        CREATE INDEX IF NOT EXISTS idx_emitters_protocol
            ON emitters(protocol);
        CREATE INDEX IF NOT EXISTS idx_emitters_confidence
            ON emitters(confidence);

        CREATE TABLE IF NOT EXISTS observations (
            decode_id   INTEGER NOT NULL REFERENCES decodes(id),
            emitter_id  INTEGER NOT NULL REFERENCES emitters(id),
            PRIMARY KEY (decode_id, emitter_id)
        );
        CREATE INDEX IF NOT EXISTS idx_observations_emitter
            ON observations(emitter_id);

        CREATE TABLE IF NOT EXISTS anomalies (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   INTEGER REFERENCES sessions(id),
            detected_at  TEXT NOT NULL,
            kind         TEXT NOT NULL,
            freq_hz      INTEGER,
            description  TEXT,
            metadata     TEXT,
            resolved     INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_anomalies_session
            ON anomalies(session_id);
        """
    )


def _v2(conn: sqlite3.Connection) -> None:
    """Adds the detections table (technology-level pattern matches).

    Also adds `baseline_ref` on sessions so users can mark a session
    as the baseline for `diff` comparisons.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    INTEGER REFERENCES sessions(id),
            detector      TEXT NOT NULL,
            technology    TEXT NOT NULL,
            freq_hz       INTEGER NOT NULL,
            bandwidth_hz  INTEGER,
            confidence    REAL,
            evidence      TEXT,
            hand_off_tools TEXT,
            detected_at   TEXT NOT NULL,
            metadata      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_detections_session
            ON detections(session_id);
        CREATE INDEX IF NOT EXISTS idx_detections_technology
            ON detections(technology);

        ALTER TABLE sessions ADD COLUMN is_baseline INTEGER DEFAULT 0;
        """
    )
