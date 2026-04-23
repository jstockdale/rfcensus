"""`rfcensus doctor` — diagnose hardware and decoder availability."""

from __future__ import annotations

from pathlib import Path

import click

from rfcensus.commands.base import bootstrap, run_async
from rfcensus.decoders.registry import get_registry
from rfcensus.hardware.health import check_all
from rfcensus.utils.paths import database_path


@click.command(name="doctor")
@click.option("--config", "config_path", type=click.Path(path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def cli(config_path: Path | None, as_json: bool) -> None:
    """Check that hardware, binaries, config, and DB are all healthy."""
    exit_code = run_async(_doctor(config_path, as_json))
    raise SystemExit(exit_code or 0)


async def _doctor(config_path: Path | None, as_json: bool) -> int:
    rt = await bootstrap(config_path=config_path, detect=True)

    sections: list[dict] = []
    any_fatal = False

    # 1. Hardware detection
    hardware_section: dict = {"name": "hardware", "entries": []}
    if not rt.registry.dongles:
        hardware_section["entries"].append(
            {"status": "warning", "text": "no SDR dongles detected"}
        )
    else:
        reports = await check_all(rt.registry.dongles)
        for dongle, report in zip(rt.registry.dongles, reports, strict=True):
            hardware_section["entries"].append(
                {
                    "status": report.status.value,
                    "text": (
                        f"{dongle.model} id={dongle.id} serial={dongle.serial or '?'} "
                        f"driver={dongle.driver} status={report.status.value}"
                        + (f" ppm={report.ppm_estimate:+.0f}" if report.ppm_estimate else "")
                    ),
                    "notes": report.notes,
                }
            )
            if report.status.value == "failed":
                any_fatal = True

    # Registry-level diagnostics (e.g. duplicate-serial warnings) live
    # outside per-dongle status. Surface them as their own entries so
    # users see them prominently.
    for diag in rt.registry.diagnostics:
        if diag and diag.startswith("⚠"):
            hardware_section["entries"].append(
                {"status": "warning", "text": diag, "notes": []}
            )
    sections.append(hardware_section)

    # 2. Decoder binaries
    decoder_section: dict = {"name": "decoders", "entries": []}
    reg = get_registry()
    for name in reg.names():
        cls = reg.get(name)
        if cls is None:
            continue
        decoder = cls()
        avail = await decoder.check_available()
        status = "ok" if avail.available else ("missing" if cls.capabilities.opt_in else "missing-required")
        decoder_section["entries"].append(
            {
                "status": status,
                "text": f"{name}: {'available' if avail.available else avail.reason}",
            }
        )
        if not avail.available and not cls.capabilities.opt_in:
            any_fatal = True
    sections.append(decoder_section)

    # 3. Config
    config_section: dict = {
        "name": "config",
        "entries": [
            {"status": "ok", "text": f"loaded from {rt.config_path}"},
            {
                "status": "ok",
                "text": (
                    f"site={rt.config.site.name} region={rt.config.site.region} "
                    f"dongles_declared={len(rt.config.dongles)} "
                    f"bands_available={len(rt.config.band_definitions)} "
                    f"bands_enabled={len(rt.config.enabled_bands())}"
                ),
            },
        ],
    }
    declared_serials = {d.serial for d in rt.config.dongles if d.serial}
    detected_serials = {d.serial for d in rt.registry.dongles if d.serial}
    missing = declared_serials - detected_serials
    for missing_serial in missing:
        config_section["entries"].append(
            {
                "status": "warning",
                "text": f"declared dongle serial={missing_serial} not currently attached",
            }
        )
    sections.append(config_section)

    # 3.5. Antenna coverage — show which enabled bands actually have a
    # usable antenna match. Surfaces silent gaps that would otherwise
    # only show up when the user runs scan/inventory.
    if rt.registry.dongles and rt.config.enabled_bands():
        from rfcensus.engine.coverage import compute_coverage

        coverage = compute_coverage(rt.config.enabled_bands(), rt.registry.dongles)
        coverage_section: dict = {"name": "antenna coverage", "entries": []}
        if not coverage.has_gaps:
            coverage_section["entries"].append({
                "status": "ok",
                "text": (
                    f"all {coverage.total} enabled bands have a usable antenna match"
                ),
            })
        else:
            coverage_section["entries"].append({
                "status": "warning",
                "text": (
                    f"{len(coverage.matched)} of {coverage.total} enabled bands "
                    f"have a usable antenna match; {len(coverage.missing)} cannot "
                    f"be scanned with current antennas"
                ),
                "notes": [
                    f"{cov.band.id} ({cov.band.center_hz/1e6:.1f} MHz)"
                    for cov in coverage.missing
                ],
            })
            for sug in coverage.suggestions:
                coverage_section["entries"].append({
                    "status": "warning",
                    "text": f"suggestion: {sug}",
                })
            coverage_section["entries"].append({
                "status": "warning",
                "text": (
                    "or run scan/inventory with --all-bands to attempt the "
                    "missing bands anyway (poor reception expected)"
                ),
            })
        sections.append(coverage_section)

    # 4. Database
    db_path = database_path()
    db_size = db_path.stat().st_size if db_path.exists() else 0
    db_section = {
        "name": "database",
        "entries": [
            {
                "status": "ok",
                "text": f"{db_path} ({db_size / 1024:.1f} KB)",
            }
        ],
    }
    sections.append(db_section)

    # Render
    if as_json:
        import json

        click.echo(json.dumps({"sections": sections, "fatal": any_fatal}, indent=2))
    else:
        _render_text(sections)

    return 1 if any_fatal else 0


def _render_text(sections: list[dict]) -> None:
    for section in sections:
        click.echo(f"\n── {section['name']} ─────────────────────────────")
        for entry in section["entries"]:
            icon = {
                "ok": "✓",
                "healthy": "✓",
                "detected": "✓",
                "warning": "!",
                "degraded": "!",
                "missing": "?",
                "unavailable": "?",
                "missing-required": "✗",
                "failed": "✗",
            }.get(entry["status"], "·")
            click.echo(f"  {icon} {entry['text']}")
            for note in entry.get("notes", []):
                click.echo(f"      {note}")
