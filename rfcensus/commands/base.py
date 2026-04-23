"""Shared helpers for CLI commands.

Every command needs to load config, optionally detect hardware, and open
the database. This module centralizes that startup.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import click

from rfcensus.config import ConfigError, SiteConfig, load_config
from rfcensus.config.loader import write_default_site_config
from rfcensus.hardware.registry import HardwareRegistry, detect_hardware
from rfcensus.storage.db import Database, get_database
from rfcensus.utils.logging import configure_logging, get_logger
from rfcensus.utils.paths import (
    database_path,
    log_path,
    site_config_path,
)

log = get_logger(__name__)


@dataclass
class Runtime:
    """Populated by `bootstrap()` and passed to commands."""

    config: SiteConfig
    config_path: Path
    registry: HardwareRegistry
    db: Database


async def bootstrap(
    *,
    config_path: Path | None = None,
    detect: bool = True,
    create_missing_config: bool = True,
) -> Runtime:
    """Common startup: logging, config, hardware, database."""
    cfg_path = config_path or site_config_path()

    if not cfg_path.exists() and create_missing_config:
        try:
            write_default_site_config(cfg_path)
            click.echo(
                f"Created default config at {cfg_path}. "
                "Edit it to declare your dongles and antennas.",
                err=True,
            )
        except ConfigError as exc:
            click.echo(f"Could not create default config: {exc}", err=True)

    try:
        config = load_config(cfg_path)
    except ConfigError as exc:
        click.echo(f"Config error: {exc}", err=True)
        raise SystemExit(2) from None

    if detect:
        registry = await detect_hardware(force=True)
        warnings = registry.apply_config(config)
        for w in warnings:
            log.warning(w)
    else:
        registry = HardwareRegistry()

    db = get_database(database_path())

    return Runtime(config=config, config_path=cfg_path, registry=registry, db=db)


def run_async(coro):
    """Run an async coroutine on a new loop and return its result.

    Returns the coroutine's return value, or raises SystemExit(130)
    on KeyboardInterrupt.
    """
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        click.echo("\nInterrupted.", err=True)
        raise SystemExit(130) from None


def setup_logging(log_level: str, quiet: bool, logfile: Path | None) -> None:
    effective_logfile = logfile or log_path()
    configure_logging(
        level=log_level,
        logfile=effective_logfile,
        quiet=quiet,
    )
