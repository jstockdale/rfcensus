"""`rfcensus suggest` — recommendation commands.

Currently:
  • suggest antennas — fleet antenna optimizer

Future subcommands could include suggest gain, suggest bands, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from rfcensus.commands.base import run_async, bootstrap

log = logging.getLogger(__name__)


@click.group(name="suggest", help="Recommendation tools (antennas, etc.).")
def cli() -> None:
    pass


@cli.command(
    name="antennas",
    help=(
        "Optimize antenna assignment across all attached dongles. "
        "Shows the recommended assignment, identifies coverage gaps, "
        "and suggests antennas to buy if the catalog can't cover "
        "everything."
    ),
)
@click.option("--config", "config_path", type=click.Path(path_type=Path))
@click.option(
    "--apply",
    is_flag=True,
    help="Write the recommended assignments to site.toml without prompting.",
)
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Synonym for --apply (matches the convention used by serialize).",
)
@click.option(
    "--from-scratch",
    is_flag=True,
    help=(
        "Show the optimal assignment ignoring current state (useful when "
        "you're starting fresh or want to see what's possible regardless "
        "of how things are wired today)."
    ),
)
def antennas_cmd(
    config_path: Path | None,
    apply: bool,
    yes: bool,
    from_scratch: bool,
) -> None:
    run_async(_run_antennas(
        config_path=config_path,
        apply=(apply or yes),
        from_scratch=from_scratch,
    ))


async def _run_antennas(
    *, config_path: Path | None, apply: bool, from_scratch: bool,
) -> None:
    from rfcensus.hardware.antenna import Antenna
    from rfcensus.hardware.fleet_optimizer import (
        optimize_fleet, diff_against_current,
    )

    rt = await bootstrap(config_path=config_path, detect=True)

    if not rt.registry.dongles:
        click.echo(
            "No dongles detected. Run `rfcensus doctor` for diagnostics.",
            err=True,
        )
        raise SystemExit(1)

    enabled = rt.config.enabled_bands()
    if not enabled:
        click.echo(
            "No bands enabled in your config. Add some bands and rerun.",
            err=True,
        )
        raise SystemExit(1)

    catalog = [Antenna.from_config(a) for a in rt.config.antennas]
    if not catalog:
        click.echo(
            "No antennas in catalog. The default antenna library should "
            "have been loaded — check your config.",
            err=True,
        )
        raise SystemExit(1)

    click.echo("═" * 72)
    click.echo(" Fleet antenna optimizer")
    click.echo("═" * 72)
    click.echo()
    click.echo(
        f"  Optimizing {len([d for d in rt.registry.dongles if d.is_usable()])} "
        f"dongle(s) across {len(enabled)} enabled band(s) using "
        f"{len(catalog)} antenna(s) in catalog..."
    )
    click.echo()

    plan = optimize_fleet(
        dongles=rt.registry.dongles,
        enabled_bands=enabled,
        available_antennas=catalog,
    )

    # Build the human-readable plan output
    click.echo("─ Recommended assignment ─")
    click.echo()
    antennas_by_id = {a.id: a for a in rt.config.antennas}
    for dongle in rt.registry.dongles:
        if not dongle.is_usable():
            continue
        proposed_id = plan.assignments.get(dongle.id)
        if proposed_id:
            ant_name = antennas_by_id.get(proposed_id).name if proposed_id in antennas_by_id else proposed_id
            click.echo(f"  {dongle.id:<32s} → {ant_name} (id={proposed_id})")
        else:
            click.echo(
                f"  {dongle.id:<32s} → no useful antenna in catalog "
                f"(consider buying one — see suggestions below)"
            )

    click.echo()
    click.echo(
        f"  This plan covers {plan.well_covered_count} of {len(enabled)} "
        f"enabled bands well (score ≥ 0.7)."
    )

    if plan.uncovered_bands:
        click.echo()
        click.echo("─ Bands not covered by this plan ─")
        for band_id in plan.uncovered_bands:
            click.echo(f"  • {band_id}")

    if plan.shopping_suggestions:
        click.echo()
        click.echo("─ Shopping suggestions ─")
        for s in plan.shopping_suggestions:
            click.echo(f"  • {s.rationale}")
            click.echo(
                f"    Bands unlocked: "
                f"{', '.join(s.bands_unlocked[:5])}"
                + (
                    f" and {len(s.bands_unlocked) - 5} more"
                    if len(s.bands_unlocked) > 5 else ""
                )
            )

    # Diff against current state (unless --from-scratch)
    if not from_scratch:
        current = {
            d.id: d.antenna.id if d.antenna else None
            for d in rt.registry.dongles if d.is_usable()
        }
        diff = diff_against_current(plan, current)
        click.echo()
        click.echo("─ Changes from current ─")
        if not diff.changes:
            click.echo("  ✓ Your current assignments are already optimal. Nothing to change.")
            return
        for dongle_id, current_ant, proposed_ant in diff.changes:
            cur_label = current_ant or "(none)"
            prop_label = proposed_ant or "(none)"
            click.echo(f"  {dongle_id:<32s} {cur_label} → {prop_label}")

    # Apply or prompt
    if not _has_changes_to_apply(plan, rt):
        return

    if apply:
        n_written = _write_assignments_to_config(plan, rt.config, config_path or rt.config.source_path)
        click.echo()
        click.echo(f"  ✓ Updated {n_written} dongle assignment(s) in site.toml")
        return

    click.echo()
    if click.confirm("Apply this plan to your site.toml?", default=False):
        n_written = _write_assignments_to_config(plan, rt.config, config_path or rt.config.source_path)
        click.echo(f"  ✓ Updated {n_written} dongle assignment(s) in site.toml")
    else:
        click.echo("  No changes written.")


def _has_changes_to_apply(plan, rt) -> bool:
    """True if the plan would change at least one current assignment."""
    for dongle in rt.registry.dongles:
        if not dongle.is_usable():
            continue
        proposed = plan.assignments.get(dongle.id)
        current = dongle.antenna.id if dongle.antenna else None
        if proposed != current:
            return True
    return False


def _write_assignments_to_config(plan, config, config_path) -> int:
    """Update the [[dongles]] stanzas in site.toml with new antenna refs.

    Reads the existing TOML, updates only the antenna fields for dongles
    in the plan, writes it back. Returns the number of stanzas changed.
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    try:
        import tomli_w
    except ImportError:
        click.echo(
            "  ✗ tomli_w not available — can't write TOML. "
            "Install it with `pip install tomli-w`.",
            err=True,
        )
        return 0

    if not config_path or not Path(config_path).exists():
        click.echo("  ✗ No config file path available; can't write.", err=True)
        return 0

    path = Path(config_path)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    n_changed = 0
    for stanza in data.get("dongles", []):
        sid = stanza.get("id")
        if sid in plan.assignments:
            new_ant = plan.assignments[sid]
            if stanza.get("antenna") != new_ant:
                if new_ant:
                    stanza["antenna"] = new_ant
                else:
                    stanza.pop("antenna", None)
                n_changed += 1

    if n_changed:
        path.write_text(tomli_w.dumps(data), encoding="utf-8")
    return n_changed
