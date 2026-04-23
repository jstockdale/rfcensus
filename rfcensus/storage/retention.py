"""Retention pruner.

Deletes rows older than the retention horizons configured in
`ResourceConfig`. Called at session start and periodically.

Retention rules (defaults):

• power_samples: keep 7 days
• decodes: keep 90 days
• sessions and emitters: never auto-deleted (they're the long-term memory)
• anomalies: never auto-deleted

If the user wants to purge emitters, they can do so explicitly via a
future `rfcensus prune --emitters` command.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from rfcensus.config.schema import ResourceConfig
from rfcensus.storage.db import Database
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


async def prune(db: Database, config: ResourceConfig) -> dict[str, int]:
    """Delete rows older than their retention windows. Returns counts."""
    now = datetime.now(timezone.utc)
    counts: dict[str, int] = {}

    if config.power_sample_retention_days > 0:
        cutoff = now - timedelta(days=config.power_sample_retention_days)
        cur = await db.execute(
            "DELETE FROM power_samples WHERE timestamp < ?", (cutoff.isoformat(),)
        )
        counts["power_samples"] = cur.rowcount
        log.debug("pruned %d power_sample rows older than %s", cur.rowcount, cutoff)

    if config.decode_retention_days > 0:
        cutoff = now - timedelta(days=config.decode_retention_days)
        # Delete decodes + dangling observations. Order matters for FK.
        cur = await db.execute(
            """
            DELETE FROM observations
            WHERE decode_id IN (SELECT id FROM decodes WHERE timestamp < ?)
            """,
            (cutoff.isoformat(),),
        )
        counts["observations"] = cur.rowcount
        cur = await db.execute(
            "DELETE FROM decodes WHERE timestamp < ?", (cutoff.isoformat(),)
        )
        counts["decodes"] = cur.rowcount
        log.debug("pruned %d decode rows older than %s", cur.rowcount, cutoff)

    return counts
