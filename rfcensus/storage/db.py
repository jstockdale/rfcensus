"""SQLite database connection and lifecycle.

All database access goes through a single `Database` object that owns one
`sqlite3.Connection`. SQLite itself supports only one writer at a time, so
we serialize writes within the async runtime via a lock. Reads can be
concurrent but we keep it simple by serializing all access for now; this
is plenty fast for the workloads rfcensus generates.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

from rfcensus.storage.schema import apply_migrations
from rfcensus.utils.logging import get_logger
from rfcensus.utils.paths import database_path

log = get_logger(__name__)

_DB_SINGLETON: Database | None = None


class Database:
    """Thin async-friendly wrapper around a sqlite3 connection.

    Supports both file-backed and in-memory databases. Pass `":memory:"`
    (as a Path or str) to create an ephemeral in-memory database —
    useful for monitor-mode sessions that don't want to persist data,
    and for tests. In-memory databases are automatically cleaned up
    when the Database is closed / GC'd.
    """

    def __init__(self, path: Path | str):
        # Accept either Path or str so `:memory:` magic works (Path
        # would resolve it as a literal filename in CWD, which is
        # wrong — the string passed to sqlite3.connect must be the
        # exact sentinel `:memory:` for it to create an in-memory DB).
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def is_in_memory(self) -> bool:
        return str(self.path) == ":memory:"

    def _ensure_open(self) -> sqlite3.Connection:
        if self._conn is None:
            log.debug("opening sqlite database at %s", self.path)
            self._conn = sqlite3.connect(
                str(self.path),
                detect_types=sqlite3.PARSE_DECLTYPES,
                isolation_level=None,  # autocommit; we manage transactions
                check_same_thread=False,  # safe: we serialize via asyncio lock
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            # WAL mode doesn't apply to in-memory databases; sqlite
            # silently ignores the pragma but it's cleaner to skip it.
            if not self.is_in_memory:
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            apply_migrations(self._conn)
        return self._conn

    async def execute(
        self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()
    ) -> sqlite3.Cursor:
        async with self._lock:
            conn = self._ensure_open()
            return await asyncio.to_thread(conn.execute, sql, params)

    async def executemany(
        self, sql: str, params_seq: list[tuple[Any, ...]]
    ) -> sqlite3.Cursor:
        async with self._lock:
            conn = self._ensure_open()
            return await asyncio.to_thread(conn.executemany, sql, params_seq)

    async def fetchone(
        self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()
    ) -> sqlite3.Row | None:
        async with self._lock:
            conn = self._ensure_open()
            cur = await asyncio.to_thread(conn.execute, sql, params)
            return cur.fetchone()

    async def fetchall(
        self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()
    ) -> list[sqlite3.Row]:
        async with self._lock:
            conn = self._ensure_open()
            cur = await asyncio.to_thread(conn.execute, sql, params)
            return cur.fetchall()

    async def transaction(self) -> TransactionContext:
        """Use in an `async with` block for a batched write."""
        await self._lock.acquire()
        try:
            conn = self._ensure_open()
            conn.execute("BEGIN")
            return TransactionContext(self, conn)
        except Exception:
            self._lock.release()
            raise

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class TransactionContext:
    def __init__(self, db: Database, conn: sqlite3.Connection):
        self.db = db
        self.conn = conn

    async def __aenter__(self) -> TransactionContext:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.conn.execute("COMMIT")
            else:
                self.conn.execute("ROLLBACK")
        finally:
            self.db._lock.release()

    def execute(
        self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()
    ) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(
        self, sql: str, params_seq: list[tuple[Any, ...]]
    ) -> sqlite3.Cursor:
        return self.conn.executemany(sql, params_seq)


def get_database(path: Path | None = None) -> Database:
    """Return the singleton database instance, creating if needed."""
    global _DB_SINGLETON
    resolved = path or database_path()
    if _DB_SINGLETON is None or _DB_SINGLETON.path != resolved:
        _DB_SINGLETON = Database(resolved)
    return _DB_SINGLETON


def reset_database_singleton() -> None:
    """Close and discard the singleton. Used in tests."""
    global _DB_SINGLETON
    if _DB_SINGLETON is not None:
        _DB_SINGLETON.close()
    _DB_SINGLETON = None
