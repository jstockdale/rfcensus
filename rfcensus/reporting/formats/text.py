"""Plain text inventory report generator.

Produces a human-readable summary of a completed session: emitters found,
their classification + confidence, any anomalies, detections of known
technologies, and warnings.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from rfcensus.engine.session import SessionResult
from rfcensus.reporting.privacy import scrub_emitter
from rfcensus.storage.models import AnomalyRecord, DetectionRecord, EmitterRecord


def render_text_report(
    result: SessionResult,
    emitters: list[EmitterRecord],
    anomalies: list[AnomalyRecord],
    detections: list[DetectionRecord] | None = None,
    *,
    include_ids: bool = False,
    site_name: str = "default",
    previously_known_ids: set[int] | None = None,
) -> str:
    detections = detections or []
    lines: list[str] = []
    lines.append("═" * 72)
    lines.append(f" rfcensus inventory report — session {result.session_id}")
    lines.append(f" site: {site_name}")
    lines.append(f" started: {_fmt(result.started_at)}")
    lines.append(f" ended:   {_fmt(result.ended_at)}")
    duration = (result.ended_at - result.started_at).total_seconds()
    lines.append(f" duration: {_humanize_duration(duration)}")
    lines.append("═" * 72)

    if not include_ids:
        lines.append("")
        lines.append(
            "Device IDs below are hashed. Re-run with --include-ids to see raw values."
        )

    lines.append("")
    lines.append(f"Plan: {len(result.plan.waves)} wave(s), "
                 f"{len(result.plan.tasks)} band task(s)")
    if result.plan.unassigned:
        lines.append(
            f"Unassigned bands (no dongle coverage): "
            f"{', '.join(result.plan.unassigned)}"
        )
    if result.plan.warnings:
        lines.append("Warnings:")
        for w in result.plan.warnings:
            lines.append(f"  • {w}")

    # Emitters grouped by protocol
    lines.append("")
    lines.append("─" * 72)
    lines.append(" Emitters detected")
    lines.append("─" * 72)

    if not emitters:
        lines.append("  (none)")
    else:
        by_protocol: dict[str, list[EmitterRecord]] = defaultdict(list)
        for e in emitters:
            by_protocol[e.protocol].append(e)

        confirmed_threshold = 3
        new_ids = previously_known_ids or set()

        for protocol in sorted(by_protocol.keys()):
            emitter_list = sorted(
                by_protocol[protocol], key=lambda r: r.confidence, reverse=True
            )
            confirmed = [
                e for e in emitter_list if e.observation_count >= confirmed_threshold
            ]
            tentative = [
                e for e in emitter_list if e.observation_count < confirmed_threshold
            ]
            lines.append("")
            lines.append(
                f"  {protocol}: {len(confirmed)} confirmed, {len(tentative)} tentative"
            )
            for e in emitter_list:
                badge = "✓" if e.observation_count >= confirmed_threshold else "?"
                new_mark = " [new]" if (e.id and e.id not in new_ids) else ""
                display = scrub_emitter(e, include_raw_ids=include_ids)
                rssi = (
                    f"{display.typical_rssi_dbm:+.1f} dBm"
                    if display.typical_rssi_dbm is not None
                    else "no RSSI"
                )
                freq_mhz = (
                    display.typical_freq_hz / 1_000_000 if display.typical_freq_hz else 0
                )
                lines.append(
                    f"    {badge} {display.device_id}"
                    f"  conf={display.confidence:.2f}"
                    f"  obs={display.observation_count}"
                    f"  {freq_mhz:.3f} MHz  {rssi}"
                    f"  [{display.classification or 'unclassified'}]"
                    f"{new_mark}"
                )

    # Detections (technologies identified for hand-off)
    if detections:
        lines.append("")
        lines.append("─" * 72)
        lines.append(" Technologies detected (hand off for deeper analysis)")
        lines.append("─" * 72)
        for d in detections:
            tools = ", ".join(d.hand_off_tools) if d.hand_off_tools else ""
            freq = f"{d.freq_hz / 1_000_000:.3f} MHz" if d.freq_hz else ""
            lines.append(
                f"  • {d.technology:20s} {freq}  conf={d.confidence:.2f}"
            )
            if d.evidence:
                lines.append(f"      evidence: {d.evidence}")
            if tools:
                lines.append(f"      suggested tools: {tools}")

    # Anomalies
    if anomalies:
        lines.append("")
        lines.append("─" * 72)
        lines.append(" Anomalies (worth investigating)")
        lines.append("─" * 72)
        for a in anomalies:
            freq = f"{a.freq_hz / 1_000_000:.3f} MHz" if a.freq_hz else ""
            lines.append(
                f"  • {a.kind:20s} {freq}  {a.description or ''}"
            )

    # Strategy summary
    lines.append("")
    lines.append("─" * 72)
    lines.append(" Execution summary")
    lines.append("─" * 72)
    for sr in result.strategy_results:
        lines.append(
            f"  {sr.band_id}: decoders={','.join(sr.decoders_run) or 'none'}"
            f"  power_scan={'yes' if sr.power_scan_performed else 'no'}"
            f"  decodes={sr.decodes_emitted}"
        )
        for err in sr.errors:
            lines.append(f"    ! {err}")

    lines.append("")
    lines.append(f"Total validated decodes: {result.total_decodes}")
    lines.append("═" * 72)

    return "\n".join(lines) + "\n"


def _fmt(dt: datetime) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _humanize_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"
