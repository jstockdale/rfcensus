"""`rfcensus init` — create a starter config and detect hardware."""

from __future__ import annotations

from pathlib import Path

import click

from rfcensus.commands.base import run_async
from rfcensus.config import ConfigError
from rfcensus.config.loader import write_default_site_config
from rfcensus.hardware.registry import detect_hardware
from rfcensus.utils.hashing import generate_salt
from rfcensus.utils.paths import config_dir, site_config_path


@click.command(name="init")
@click.option(
    "--overwrite", is_flag=True, help="Overwrite an existing config file."
)
@click.option(
    "--wizard",
    is_flag=True,
    help="Interactive prompts for dongle and antenna setup (stub — writes defaults for now).",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    help="Path to site config file (default: XDG config dir).",
)
def cli(overwrite: bool, wizard: bool, config_path: Path | None) -> None:
    """Create a starter site.toml and generate a privacy salt."""
    target = config_path or site_config_path()
    try:
        written = write_default_site_config(target, overwrite=overwrite)
    except ConfigError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1) from None

    salt_path = config_dir() / "salt"
    if not salt_path.exists() or overwrite:
        salt_path.write_text(generate_salt() + "\n", encoding="utf-8")
        salt_path.chmod(0o600)

    click.echo(f"Wrote site config: {written}")
    click.echo(f"Wrote privacy salt: {salt_path}")

    click.echo("")
    click.echo("Detecting attached hardware...")
    run_async(_detect_and_suggest())

    if wizard:
        click.echo(
            "\n(wizard flag is a stub for now — interactive prompts coming in a later pass)"
        )

    click.echo("\nEdit the site config to declare your dongles, then run:")
    click.echo("  rfcensus doctor       # verify setup")
    click.echo("  rfcensus inventory    # run a full site survey")


async def _detect_and_suggest() -> None:
    registry = await detect_hardware(force=True)
    if not registry.dongles:
        click.echo("  No SDR dongles detected. Plug one in and rerun `rfcensus doctor`.")
        return
    click.echo(f"  Found {len(registry.dongles)} dongle(s):")
    for d in registry.dongles:
        serial = d.serial or "(no serial)"
        click.echo(f"    • {d.model:<20s} serial={serial}  driver={d.driver}")
    click.echo("\nTo use these, add stanzas like:")
    for d in registry.dongles[:2]:
        click.echo("")
        click.echo(f"  [[dongles]]")
        click.echo(f"  id = \"{d.id}\"")
        click.echo(f"  serial = \"{d.serial or ''}\"")
        click.echo(f"  model = \"{d.model}\"")
        click.echo(f"  driver = \"{d.driver}\"")
        click.echo(f"  antenna = \"whip_generic_small\"  # choose from `rfcensus list antennas`")
