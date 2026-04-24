"""rfcensus CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import click

from rfcensus import __version__
from rfcensus.commands import baseline as baseline_cmd
from rfcensus.commands import diff as diff_cmd
from rfcensus.commands import doctor as doctor_cmd
from rfcensus.commands import export as export_cmd
from rfcensus.commands import init as init_cmd
from rfcensus.commands import inventory as inventory_cmd
from rfcensus.commands import list as list_cmd
from rfcensus.commands import monitor as monitor_cmd
from rfcensus.commands import serialize as serialize_cmd
from rfcensus.commands import setup as setup_cmd
from rfcensus.commands import suggest as suggest_cmd
from rfcensus.commands.base import setup_logging


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="rfcensus — site survey and inventory for the RF environment around you.",
)
@click.version_option(
    __version__,
    prog_name="rfcensus",
    message="%(prog)s, version %(version)s\n73 🛰️",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
)
@click.option(
    "-v", "--verbose",
    count=True,
    help=(
        "Increase verbosity. -v shows INFO (the default); -vv shows DEBUG "
        "including every subprocess's stderr output. Overrides --log-level."
    ),
)
@click.option(
    "--logfile",
    type=click.Path(path_type=Path),
    help="Log file path (default: XDG state dir).",
)
@click.option("--no-logfile", is_flag=True, help="Disable file logging.")
@click.option("--quiet", is_flag=True, help="Suppress console logging.")
def main(
    log_level: str,
    verbose: int,
    logfile: Path | None,
    no_logfile: bool,
    quiet: bool,
) -> None:
    # -v / -vv overrides --log-level so the shorthand is predictable.
    if verbose >= 2:
        log_level = "DEBUG"
    elif verbose == 1:
        log_level = "INFO"
    setup_logging(
        log_level=log_level.upper(),
        quiet=quiet,
        logfile=None if no_logfile else logfile,
    )


main.add_command(init_cmd.cli)
main.add_command(setup_cmd.cli)
main.add_command(serialize_cmd.cli)
main.add_command(doctor_cmd.cli)
main.add_command(list_cmd.cli)
main.add_command(inventory_cmd.cli_inventory)
main.add_command(inventory_cmd.cli_scan)
main.add_command(monitor_cmd.monitor_cli)
main.add_command(export_cmd.cli)
main.add_command(diff_cmd.cli)
main.add_command(baseline_cmd.cli)
main.add_command(suggest_cmd.cli)
