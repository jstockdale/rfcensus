"""`rfcensus list` — list dongles, bands, decoders, sessions, emitters."""

from __future__ import annotations

import json
from pathlib import Path

import click

from rfcensus.commands.base import bootstrap, run_async
from rfcensus.decoders.registry import get_registry
from rfcensus.reporting.privacy import scrub_emitter
from rfcensus.storage.repositories import EmitterRepo, SessionRepo


@click.group(name="list")
def cli() -> None:
    """List hardware, bands, decoders, sessions, or emitters."""


@cli.command(name="dongles")
@click.option("--config", "config_path", type=click.Path(path_type=Path))
@click.option("--json", "as_json", is_flag=True)
def list_dongles(config_path: Path | None, as_json: bool) -> None:
    """Show attached SDR dongles."""
    run_async(_dongles(config_path, as_json))


async def _dongles(config_path: Path | None, as_json: bool) -> None:
    rt = await bootstrap(config_path=config_path, detect=True)
    if as_json:
        payload = [
            {
                "id": d.id,
                "serial": d.serial,
                "model": d.model,
                "driver": d.driver,
                "status": d.status.value,
                "freq_range_hz": list(d.capabilities.freq_range_hz),
                "wide_scan_capable": d.capabilities.wide_scan_capable,
                "antenna": d.antenna.id if d.antenna else None,
            }
            for d in rt.registry.dongles
        ]
        click.echo(json.dumps(payload, indent=2))
        return

    if not rt.registry.dongles:
        click.echo("No dongles detected.")
        return
    click.echo(f"{'id':<20s} {'model':<20s} {'serial':<20s} {'antenna':<20s}")
    click.echo("─" * 80)
    for d in rt.registry.dongles:
        click.echo(
            f"{d.id:<20s} {d.model:<20s} {(d.serial or '?'):<20s} "
            f"{(d.antenna.id if d.antenna else '-'):<20s}"
        )


@cli.command(name="bands")
@click.option("--config", "config_path", type=click.Path(path_type=Path))
@click.option("--json", "as_json", is_flag=True)
def list_bands(config_path: Path | None, as_json: bool) -> None:
    """Show configured bands."""
    run_async(_bands(config_path, as_json))


async def _bands(config_path: Path | None, as_json: bool) -> None:
    rt = await bootstrap(config_path=config_path, detect=False)
    enabled = {b.id for b in rt.config.enabled_bands()}
    bands = rt.config.band_definitions

    if as_json:
        click.echo(
            json.dumps(
                [
                    {
                        "id": b.id,
                        "name": b.name,
                        "freq_low": b.freq_low,
                        "freq_high": b.freq_high,
                        "strategy": b.strategy.value,
                        "enabled": b.id in enabled,
                        "opt_in": b.opt_in,
                    }
                    for b in bands
                ],
                indent=2,
            )
        )
        return
    click.echo(f"{'id':<24s} {'name':<40s} {'span':<20s} {'status':<10s}")
    click.echo("─" * 100)
    for b in bands:
        span = f"{b.freq_low/1e6:.2f}-{b.freq_high/1e6:.2f} MHz"
        status = "enabled" if b.id in enabled else ("opt-in" if b.opt_in else "disabled")
        click.echo(f"{b.id:<24s} {b.name:<40s} {span:<20s} {status:<10s}")


@cli.command(name="decoders")
@click.option("--json", "as_json", is_flag=True)
def list_decoders(as_json: bool) -> None:
    """Show available decoder plugins."""
    run_async(_decoders(as_json))


async def _decoders(as_json: bool) -> None:
    registry = get_registry()
    rows = []
    for name in registry.names():
        cls = registry.get(name)
        if cls is None:
            continue
        decoder = cls()
        avail = await decoder.check_available()
        rows.append(
            {
                "name": name,
                "protocols": cls.capabilities.protocols,
                "cpu_cost": cls.capabilities.cpu_cost,
                "available": avail.available,
                "reason": avail.reason,
                "opt_in": cls.capabilities.opt_in,
            }
        )
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    click.echo(f"{'name':<16s} {'cpu':<10s} {'available':<10s} notes")
    click.echo("─" * 80)
    for r in rows:
        avail = "✓" if r["available"] else "✗"
        opt = " (opt-in)" if r["opt_in"] else ""
        note = "" if r["available"] else r["reason"]
        click.echo(f"{r['name']:<16s} {r['cpu_cost']:<10s} {avail:<10s} {note}{opt}")


@cli.command(name="sessions")
@click.option("--config", "config_path", type=click.Path(path_type=Path))
@click.option("--limit", default=20, type=int)
@click.option("--json", "as_json", is_flag=True)
def list_sessions(config_path: Path | None, limit: int, as_json: bool) -> None:
    """Show recent sessions."""
    run_async(_sessions(config_path, limit, as_json))


async def _sessions(config_path: Path | None, limit: int, as_json: bool) -> None:
    rt = await bootstrap(config_path=config_path, detect=False)
    repo = SessionRepo(rt.db)
    sessions = await repo.recent(limit=limit)
    if as_json:
        click.echo(
            json.dumps(
                [
                    {
                        "id": s.id,
                        "command": s.command,
                        "started_at": s.started_at.isoformat(),
                        "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                        "site": s.site_name,
                    }
                    for s in sessions
                ],
                indent=2,
            )
        )
        return

    if not sessions:
        click.echo("No sessions recorded yet.")
        return
    click.echo(f"{'id':<6s} {'command':<12s} {'started':<24s} {'duration':<12s}")
    click.echo("─" * 70)
    for s in sessions:
        dur = (
            f"{(s.ended_at - s.started_at).total_seconds():.0f}s"
            if s.ended_at
            else "(incomplete)"
        )
        click.echo(
            f"{str(s.id):<6s} {s.command:<12s} "
            f"{s.started_at.astimezone().strftime('%Y-%m-%d %H:%M:%S'):<24s} "
            f"{dur:<12s}"
        )


@cli.command(name="emitters")
@click.option("--config", "config_path", type=click.Path(path_type=Path))
@click.option("--protocol", help="Filter by protocol (e.g. tpms, ert_scm).")
@click.option("--min-confidence", type=float, default=0.0)
@click.option("--include-ids", is_flag=True, help="Show raw device IDs.")
@click.option("--json", "as_json", is_flag=True)
def list_emitters(
    config_path: Path | None,
    protocol: str | None,
    min_confidence: float,
    include_ids: bool,
    as_json: bool,
) -> None:
    """Show emitters tracked across all sessions."""
    run_async(_emitters(config_path, protocol, min_confidence, include_ids, as_json))


async def _emitters(
    config_path: Path | None,
    protocol: str | None,
    min_confidence: float,
    include_ids: bool,
    as_json: bool,
) -> None:
    rt = await bootstrap(config_path=config_path, detect=False)
    repo = EmitterRepo(rt.db)
    emitters = await repo.all(min_confidence=min_confidence, protocol=protocol)

    if as_json:
        click.echo(
            json.dumps(
                [
                    {
                        "id": e.id,
                        "protocol": e.protocol,
                        "device_id": (e.device_id if include_ids else f"hash:{e.device_id_hash}"),
                        "classification": e.classification,
                        "observation_count": e.observation_count,
                        "confidence": e.confidence,
                        "typical_freq_hz": e.typical_freq_hz,
                    }
                    for e in emitters
                ],
                indent=2,
            )
        )
        return

    if not emitters:
        click.echo("No emitters recorded.")
        return

    if not include_ids:
        click.echo("(Device IDs hashed. Use --include-ids to show raw values.)")
        click.echo()

    click.echo(f"{'protocol':<20s} {'id':<24s} {'conf':<6s} {'obs':<6s} {'freq (MHz)':<14s}")
    click.echo("─" * 80)
    for e in emitters:
        display = scrub_emitter(e, include_raw_ids=include_ids)
        freq = f"{display.typical_freq_hz/1e6:.3f}" if display.typical_freq_hz else "-"
        click.echo(
            f"{display.protocol:<20s} {display.device_id:<24s} "
            f"{display.confidence:<6.2f} {display.observation_count:<6d} {freq:<14s}"
        )


@cli.command(name="antennas")
@click.option("--config", "config_path", type=click.Path(path_type=Path))
def list_antennas(config_path: Path | None) -> None:
    """Show antennas declared in the antenna library."""
    run_async(_antennas(config_path))


async def _antennas(config_path: Path | None) -> None:
    rt = await bootstrap(config_path=config_path, detect=False)
    click.echo(f"{'id':<24s} {'name':<40s} {'range':<20s}")
    click.echo("─" * 90)
    for a in rt.config.antennas:
        low_mhz = a.usable_range[0] / 1e6
        high_mhz = a.usable_range[1] / 1e6
        click.echo(
            f"{a.id:<24s} {a.name:<40s} {low_mhz:.0f}-{high_mhz:.0f} MHz"
        )


@cli.command(name="detectors")
@click.option("--json", "as_json", is_flag=True)
def list_detectors(as_json: bool) -> None:
    """Show available detectors (pattern-based recognizers for known technologies)."""
    from rfcensus.detectors.registry import get_registry as get_det_registry

    reg = get_det_registry()
    rows = []
    for name in reg.names():
        cls = reg.get(name)
        if cls is None:
            continue
        caps = cls.capabilities
        ranges_mhz = [
            f"{low/1e6:.0f}-{high/1e6:.0f}"
            for low, high in caps.relevant_freq_ranges
        ]
        rows.append({
            "name": name,
            "technologies": caps.detected_technologies,
            "bands_mhz": ranges_mhz,
            "uses_iq": caps.consumes_iq,
            "hand_off_tools": list(caps.hand_off_tools),
            "description": caps.description,
        })
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    click.echo(f"{'name':<14s} {'technologies':<30s} {'uses IQ':<8s} bands (MHz)")
    click.echo("─" * 95)
    for r in rows:
        techs = ",".join(r["technologies"])[:28]
        iq = "yes" if r["uses_iq"] else "no"
        bands = "; ".join(r["bands_mhz"])
        click.echo(f"{r['name']:<14s} {techs:<30s} {iq:<8s} {bands}")
        if r["hand_off_tools"]:
            click.echo(f"    → hand off: {', '.join(r['hand_off_tools'])}")


@cli.command(name="detections")
@click.option("--config", "config_path", type=click.Path(path_type=Path))
@click.option("--technology", help="Filter by technology (e.g. lora, p25_trunked_system).")
@click.option("--session", "session_id", type=int, help="Limit to one session.")
@click.option("--json", "as_json", is_flag=True)
def list_detections(
    config_path: Path | None,
    technology: str | None,
    session_id: int | None,
    as_json: bool,
) -> None:
    """Show detections across sessions."""
    run_async(_detections_cmd(config_path, technology, session_id, as_json))


async def _detections_cmd(
    config_path: Path | None,
    technology: str | None,
    session_id: int | None,
    as_json: bool,
) -> None:
    from rfcensus.storage.repositories import DetectionRepo

    rt = await bootstrap(config_path=config_path, detect=False)
    repo = DetectionRepo(rt.db)
    if session_id is not None:
        records = await repo.for_session(session_id)
    else:
        records = await repo.all(technology=technology)

    if as_json:
        click.echo(json.dumps([
            {
                "id": r.id,
                "session_id": r.session_id,
                "detector": r.detector,
                "technology": r.technology,
                "freq_hz": r.freq_hz,
                "bandwidth_hz": r.bandwidth_hz,
                "confidence": r.confidence,
                "evidence": r.evidence,
                "hand_off_tools": r.hand_off_tools,
                "detected_at": r.detected_at.isoformat(),
            } for r in records
        ], indent=2))
        return
    if not records:
        click.echo("No detections recorded.")
        return
    click.echo(f"{'session':<8s} {'technology':<24s} {'freq (MHz)':<14s} {'conf':<6s} tools")
    click.echo("─" * 90)
    for r in records:
        freq = f"{r.freq_hz/1e6:.3f}"
        tools = ", ".join(r.hand_off_tools[:3])
        click.echo(
            f"{str(r.session_id):<8s} {r.technology:<24s} "
            f"{freq:<14s} {r.confidence:<6.2f} {tools}"
        )
