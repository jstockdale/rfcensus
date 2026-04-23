"""Report builder.

Ties together session results, database queries for emitters / anomalies
/ detections, privacy scrubbing, and format rendering.
"""

from __future__ import annotations

from rfcensus.engine.session import SessionResult
from rfcensus.reporting.formats.json import render_json_report
from rfcensus.reporting.formats.text import render_text_report
from rfcensus.storage.db import Database
from rfcensus.storage.repositories import AnomalyRepo, DetectionRepo, EmitterRepo


class ReportBuilder:
    """Generates a report for a given SessionResult."""

    def __init__(self, db: Database):
        self.db = db
        self.emitter_repo = EmitterRepo(db)
        self.anomaly_repo = AnomalyRepo(db)
        self.detection_repo = DetectionRepo(db)

    async def render(
        self,
        result: SessionResult,
        *,
        fmt: str = "text",
        include_ids: bool = False,
        site_name: str = "default",
    ) -> str:
        emitters = await self.emitter_repo.for_session(result.session_id)
        anomalies = await self.anomaly_repo.for_session(result.session_id)
        detections = await self.detection_repo.for_session(result.session_id)

        previously_known_ids: set[int] = {
            e.id for e in emitters
            if e.id is not None and e.first_seen < result.started_at
        }

        if fmt == "json":
            return render_json_report(
                result, emitters, anomalies, detections,
                include_ids=include_ids,
            )
        return render_text_report(
            result, emitters, anomalies, detections,
            include_ids=include_ids,
            site_name=site_name,
            previously_known_ids=previously_known_ids,
        )
