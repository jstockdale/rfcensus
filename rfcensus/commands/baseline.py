"""`rfcensus baseline` — mark a session as the reference baseline.

A baseline session is the "this is what normal looks like" reference
other sessions get compared against. Useful for: "I did a careful 1-hour
inventory last week — anything new since then is worth flagging."
"""

from __future__ import annotations

from pathlib import Path

import click

from rfcensus.commands.base import bootstrap, run_async
from rfcensus.storage.repositories import SessionRepo


@click.group(name="baseline")
def cli() -> None:
    """Manage baseline reference sessions."""


@cli.command(name="set")
@click.argument("session_id", type=int)
@click.option("--config", "config_path", type=click.Path(path_type=Path))
def baseline_set(session_id: int, config_path: Path | None) -> None:
    """Mark SESSION_ID as the baseline."""
    run_async(_set(session_id, config_path))


async def _set(session_id: int, config_path: Path | None) -> None:
    rt = await bootstrap(config_path=config_path, detect=False)
    repo = SessionRepo(rt.db)
    session = await repo.by_id(session_id)
    if session is None:
        click.echo(f"Session {session_id} not found.", err=True)
        raise SystemExit(1)
    ok = await repo.mark_baseline(session_id, True)
    if ok:
        click.echo(f"Marked session {session_id} ({session.command}) as baseline.")
    else:
        click.echo(f"Failed to mark session {session_id}.", err=True)
        raise SystemExit(1)


@cli.command(name="clear")
@click.argument("session_id", type=int)
@click.option("--config", "config_path", type=click.Path(path_type=Path))
def baseline_clear(session_id: int, config_path: Path | None) -> None:
    """Unmark SESSION_ID as a baseline."""
    run_async(_clear(session_id, config_path))


async def _clear(session_id: int, config_path: Path | None) -> None:
    rt = await bootstrap(config_path=config_path, detect=False)
    repo = SessionRepo(rt.db)
    ok = await repo.mark_baseline(session_id, False)
    if ok:
        click.echo(f"Cleared baseline flag on session {session_id}.")


@cli.command(name="show")
@click.option("--config", "config_path", type=click.Path(path_type=Path))
@click.option("--site", help="Restrict to a site name.")
def baseline_show(config_path: Path | None, site: str | None) -> None:
    """Show the current baseline session."""
    run_async(_show(config_path, site))


async def _show(config_path: Path | None, site: str | None) -> None:
    rt = await bootstrap(config_path=config_path, detect=False)
    repo = SessionRepo(rt.db)
    baseline = await repo.current_baseline(site_name=site)
    if baseline is None:
        scope = f" for site '{site}'" if site else ""
        click.echo(f"No baseline session set{scope}.")
        return
    click.echo(
        f"Baseline: session {baseline.id} ({baseline.command}), "
        f"started {baseline.started_at.astimezone().strftime('%Y-%m-%d %H:%M')}, "
        f"site={baseline.site_name or '(none)'}"
    )
