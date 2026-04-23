"""`rfcensus export` — export session data."""

from __future__ import annotations

import csv
import io
import json as _json
from pathlib import Path

import click

from rfcensus.commands.base import bootstrap, run_async
from rfcensus.reporting.privacy import scrub_emitter
from rfcensus.storage.repositories import (
    AnomalyRepo,
    DecodeRepo,
    EmitterRepo,
)


@click.group(name="export")
def cli() -> None:
    """Export data to JSON, CSV, or text."""


@cli.command(name="session")
@click.argument("session_id", type=int)
@click.option("--config", "config_path", type=click.Path(path_type=Path))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="json",
    show_default=True,
)
@click.option("--output", type=click.Path(path_type=Path))
@click.option("--include-ids", is_flag=True)
def export_session(
    session_id: int,
    config_path: Path | None,
    fmt: str,
    output: Path | None,
    include_ids: bool,
) -> None:
    """Export a complete session (emitters, anomalies, summary)."""
    run_async(_session(session_id, config_path, fmt, output, include_ids))


async def _session(
    session_id: int,
    config_path: Path | None,
    fmt: str,
    output: Path | None,
    include_ids: bool,
) -> None:
    rt = await bootstrap(config_path=config_path, detect=False)
    emitter_repo = EmitterRepo(rt.db)
    anomaly_repo = AnomalyRepo(rt.db)

    emitters = await emitter_repo.for_session(session_id)
    anomalies = await anomaly_repo.for_session(session_id)

    if fmt == "json":
        payload = {
            "session_id": session_id,
            "emitters": [
                {
                    "id": e.id,
                    "protocol": e.protocol,
                    "device_id": (
                        e.device_id if include_ids else f"hash:{e.device_id_hash}"
                    ),
                    "classification": e.classification,
                    "observation_count": e.observation_count,
                    "typical_freq_hz": e.typical_freq_hz,
                    "typical_rssi_dbm": e.typical_rssi_dbm,
                    "confidence": e.confidence,
                    "first_seen": e.first_seen.isoformat(),
                    "last_seen": e.last_seen.isoformat(),
                }
                for e in emitters
            ],
            "anomalies": [
                {
                    "id": a.id,
                    "kind": a.kind,
                    "freq_hz": a.freq_hz,
                    "description": a.description,
                    "detected_at": a.detected_at.isoformat(),
                }
                for a in anomalies
            ],
        }
        text = _json.dumps(payload, indent=2)
    else:
        lines = [f"Session {session_id}", "=" * 60, "", "Emitters:"]
        for e in emitters:
            display = scrub_emitter(e, include_raw_ids=include_ids)
            lines.append(
                f"  {display.protocol}  {display.device_id}  "
                f"obs={display.observation_count}  conf={display.confidence:.2f}"
            )
        if anomalies:
            lines.append("")
            lines.append("Anomalies:")
            for a in anomalies:
                lines.append(
                    f"  {a.kind}  {a.freq_hz}  {a.description or ''}"
                )
        text = "\n".join(lines) + "\n"

    if output:
        output.write_text(text, encoding="utf-8")
        click.echo(f"Wrote {output}")
    else:
        click.echo(text)


@cli.command(name="emitters")
@click.option("--config", "config_path", type=click.Path(path_type=Path))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "csv"]),
    default="json",
    show_default=True,
)
@click.option("--output", type=click.Path(path_type=Path))
@click.option("--min-confidence", type=float, default=0.0)
@click.option("--protocol")
@click.option("--include-ids", is_flag=True)
def export_emitters(
    config_path: Path | None,
    fmt: str,
    output: Path | None,
    min_confidence: float,
    protocol: str | None,
    include_ids: bool,
) -> None:
    """Export all emitters ever recorded."""
    run_async(_emitters(config_path, fmt, output, min_confidence, protocol, include_ids))


async def _emitters(
    config_path: Path | None,
    fmt: str,
    output: Path | None,
    min_confidence: float,
    protocol: str | None,
    include_ids: bool,
) -> None:
    rt = await bootstrap(config_path=config_path, detect=False)
    repo = EmitterRepo(rt.db)
    emitters = await repo.all(min_confidence=min_confidence, protocol=protocol)

    if fmt == "json":
        payload = [
            {
                "id": e.id,
                "protocol": e.protocol,
                "device_id": (
                    e.device_id if include_ids else f"hash:{e.device_id_hash}"
                ),
                "classification": e.classification,
                "observation_count": e.observation_count,
                "typical_freq_hz": e.typical_freq_hz,
                "typical_rssi_dbm": e.typical_rssi_dbm,
                "confidence": e.confidence,
                "first_seen": e.first_seen.isoformat(),
                "last_seen": e.last_seen.isoformat(),
            }
            for e in emitters
        ]
        text = _json.dumps(payload, indent=2)
    else:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "id",
                "protocol",
                "device_id",
                "classification",
                "obs",
                "confidence",
                "freq_hz",
                "rssi_dbm",
                "first_seen",
                "last_seen",
            ]
        )
        for e in emitters:
            writer.writerow(
                [
                    e.id,
                    e.protocol,
                    e.device_id if include_ids else f"hash:{e.device_id_hash}",
                    e.classification or "",
                    e.observation_count,
                    f"{e.confidence:.3f}",
                    e.typical_freq_hz or "",
                    (f"{e.typical_rssi_dbm:.1f}" if e.typical_rssi_dbm is not None else ""),
                    e.first_seen.isoformat(),
                    e.last_seen.isoformat(),
                ]
            )
        text = buf.getvalue()

    if output:
        output.write_text(text, encoding="utf-8")
        click.echo(f"Wrote {output}")
    else:
        click.echo(text)


@cli.command(name="decodes")
@click.argument("session_id", type=int)
@click.option("--config", "config_path", type=click.Path(path_type=Path))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "csv"]),
    default="json",
    show_default=True,
)
@click.option("--output", type=click.Path(path_type=Path))
@click.option("--validated-only", is_flag=True)
def export_decodes(
    session_id: int,
    config_path: Path | None,
    fmt: str,
    output: Path | None,
    validated_only: bool,
) -> None:
    """Export every decode from a session (potentially large)."""
    run_async(_decodes(session_id, config_path, fmt, output, validated_only))


async def _decodes(
    session_id: int,
    config_path: Path | None,
    fmt: str,
    output: Path | None,
    validated_only: bool,
) -> None:
    rt = await bootstrap(config_path=config_path, detect=False)
    repo = DecodeRepo(rt.db)
    decodes = await repo.for_session(session_id, validated_only=validated_only)

    if fmt == "json":
        payload = [
            {
                "id": d.id,
                "timestamp": d.timestamp.isoformat(),
                "decoder": d.decoder,
                "protocol": d.protocol,
                "freq_hz": d.freq_hz,
                "rssi_dbm": d.rssi_dbm,
                "snr_db": d.snr_db,
                "validated": d.validated,
                "payload": d.payload,
            }
            for d in decodes
        ]
        text = _json.dumps(payload, indent=2)
    else:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["id", "timestamp", "decoder", "protocol", "freq_hz", "rssi_dbm", "snr_db", "validated"]
        )
        for d in decodes:
            writer.writerow(
                [
                    d.id,
                    d.timestamp.isoformat(),
                    d.decoder,
                    d.protocol,
                    d.freq_hz,
                    d.rssi_dbm,
                    d.snr_db,
                    int(d.validated),
                ]
            )
        text = buf.getvalue()

    if output:
        output.write_text(text, encoding="utf-8")
        click.echo(f"Wrote {output}")
    else:
        click.echo(text)
