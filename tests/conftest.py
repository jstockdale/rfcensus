"""Shared pytest fixtures for rfcensus tests."""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from rfcensus.storage.db import Database


@pytest_asyncio.fixture
async def db() -> AsyncIterator[Database]:
    """Fresh temporary SQLite database per test."""
    tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmpfile.close()
    database = Database(Path(tmpfile.name))
    try:
        yield database
    finally:
        database.close()
        Path(tmpfile.name).unlink(missing_ok=True)


@pytest.fixture
def salt() -> str:
    return "test-salt-for-unit-tests"
