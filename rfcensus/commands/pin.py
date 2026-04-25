"""`rfcensus pin` — manage decoder→dongle pins.

Pinning dedicates a dongle to one decoder + frequency for the entire
session. See `rfcensus.engine.pinning` for the runtime mechanics.

Subcommands:

  rfcensus pin              — interactive wizard (no args)
  rfcensus pin list         — show currently configured pins
  rfcensus pin add          — add a pin without the wizard
  rfcensus pin remove ID    — remove the pin on dongle ID
  rfcensus pin clear        — remove all pins (with confirmation)

The wizard reuses the menu helpers from `setup.py` and the frequency
profile catalogue from `_frequency_guide.py` so the look-and-feel
matches the rest of the friendly CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

try:
    import tomli_w
except ImportError:  # pragma: no cover
    tomli_w = None

try:
    import tomllib  # py311+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from rfcensus.commands._frequency_guide import COMMON_FREQUENCIES
from rfcensus.commands.base import bootstrap, run_async
from rfcensus.commands.setup import _menu, _menu_with_back
from rfcensus.utils.paths import site_config_path


# ────────────────────────────────────────────────────────────────────
# CLI group
# ────────────────────────────────────────────────────────────────────


@click.group(
    name="pin",
    invoke_without_command=True,
    help=(
        "Manage decoder→dongle pins. With no subcommand, runs the "
        "interactive wizard. Pinning dedicates a dongle to one decoder "
        "at one frequency for the entire session — useful when you "
        "want gap-free coverage of a specific target."
    ),
)
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path),
    help="Site config path (default: XDG config dir)",
)
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None) -> None:
    if ctx.invoked_subcommand is None:
        run_async(_wizard(config_path))
    else:
        ctx.ensure_object(dict)
        ctx.obj["config_path"] = config_path


@cli.command(name="list", help="Show currently configured pins.")
@click.pass_context
def cli_list(ctx: click.Context) -> None:
    config_path = ctx.obj.get("config_path") if ctx.obj else None
    run_async(_list(config_path))


@cli.command(
    name="add",
    help=(
        "Add a pin without the wizard. Format: "
        "DONGLE:DECODER@FREQ[:SR] (e.g. 00000043:rtl_433@433.92M). "
        "Repeat to add multiple."
    ),
)
@click.argument("specs", nargs=-1, required=True)
@click.pass_context
def cli_add(ctx: click.Context, specs: tuple[str, ...]) -> None:
    config_path = ctx.obj.get("config_path") if ctx.obj else None
    run_async(_add(config_path, list(specs)))


@cli.command(name="remove", help="Remove the pin on the given dongle id.")
@click.argument("dongle_id")
@click.pass_context
def cli_remove(ctx: click.Context, dongle_id: str) -> None:
    config_path = ctx.obj.get("config_path") if ctx.obj else None
    run_async(_remove(config_path, dongle_id))


@cli.command(
    name="clear",
    help="Remove all pins from the config (asks confirmation).",
)
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt.")
@click.pass_context
def cli_clear(ctx: click.Context, yes: bool) -> None:
    config_path = ctx.obj.get("config_path") if ctx.obj else None
    run_async(_clear(config_path, force=yes))


# ────────────────────────────────────────────────────────────────────
# Subcommand implementations
# ────────────────────────────────────────────────────────────────────


async def _list(config_path: Path | None) -> None:
    """Print currently configured pins."""
    rt = await bootstrap(config_path=config_path, detect=False)
    pins = [
        (d.id, d.pin) for d in rt.config.dongles if d.pin is not None
    ]
    if not pins:
        click.echo("No pins configured.")
        click.echo()
        click.echo("Add one with `rfcensus pin add` or run `rfcensus pin` "
                   "for the wizard.")
        return
    click.echo(f"Configured pins ({len(pins)}):")
    click.echo()
    for dongle_id, pin in pins:
        click.echo(
            f"  • {dongle_id} → {pin.decoder} @ "
            f"{pin.freq_hz / 1e6:.3f} MHz "
            f"({pin.access_mode})"
            + (
                f", sr={pin.sample_rate / 1e6:.2f}M"
                if pin.sample_rate else ""
            )
        )


async def _add(config_path: Path | None, specs: list[str]) -> None:
    """Add pins from CLI args without prompting."""
    from rfcensus.engine.pinning import parse_cli_pin

    target = config_path or site_config_path()
    if not target.exists():
        raise click.BadParameter(
            f"No config at {target}. Run `rfcensus setup` first."
        )

    parsed = []
    for s in specs:
        try:
            parsed.append(parse_cli_pin(s))
        except ValueError as exc:
            raise click.BadParameter(str(exc)) from None

    data = _read_toml(target)
    n_changed = _apply_pins_to_toml(data, parsed)
    if n_changed == 0:
        click.echo("No changes (pins already match).")
        return
    _write_toml(target, data)
    click.echo(f"✓ Wrote {n_changed} pin(s) to {target}")
    for spec in parsed:
        click.echo(
            f"  • {spec.dongle_id} → {spec.decoder} @ "
            f"{spec.freq_hz / 1e6:.3f} MHz"
        )


async def _remove(config_path: Path | None, dongle_id: str) -> None:
    target = config_path or site_config_path()
    if not target.exists():
        raise click.BadParameter(f"No config at {target}.")
    data = _read_toml(target)
    found = False
    for stanza in data.get("dongles", []):
        if stanza.get("id") == dongle_id and "pin" in stanza:
            del stanza["pin"]
            found = True
            break
    if not found:
        click.echo(f"No pin found on dongle {dongle_id}.")
        return
    _write_toml(target, data)
    click.echo(f"✓ Removed pin on dongle {dongle_id} from {target}")


async def _clear(config_path: Path | None, force: bool = False) -> None:
    target = config_path or site_config_path()
    if not target.exists():
        raise click.BadParameter(f"No config at {target}.")
    data = _read_toml(target)
    pinned = [
        s for s in data.get("dongles", []) if "pin" in s
    ]
    if not pinned:
        click.echo("No pins to clear.")
        return
    if not force:
        click.echo(f"This will remove {len(pinned)} pin(s):")
        for s in pinned:
            click.echo(f"  • {s.get('id')}")
        if not click.confirm("Continue?", default=False):
            click.echo("Cancelled.")
            return
    for s in pinned:
        del s["pin"]
    _write_toml(target, data)
    click.echo(f"✓ Cleared {len(pinned)} pin(s).")


# ────────────────────────────────────────────────────────────────────
# Wizard
# ────────────────────────────────────────────────────────────────────


async def _wizard(config_path: Path | None) -> None:
    """The interactive flow."""
    target = config_path or site_config_path()
    if not target.exists():
        click.echo(f"No config at {target}.")
        click.echo("Run `rfcensus setup` first to configure dongles + antennas.")
        raise SystemExit(1)

    rt = await bootstrap(config_path=config_path, detect=True)
    if not rt.registry.dongles:
        click.echo("No dongles detected. Plug some in and re-run.", err=True)
        raise SystemExit(1)

    _say_header()

    # Show fleet summary up front so the user can see what's available
    _show_fleet(rt.config, rt.registry)

    # Per-dongle walk
    new_pins: dict[str, dict[str, Any]] = {}  # dongle_id → pin dict
    removed: list[str] = []  # dongle_ids whose pin was explicitly removed

    for dongle in rt.registry.usable():
        existing_dongle_cfg = next(
            (d for d in rt.config.dongles if d.id == dongle.id), None,
        )
        existing_pin = (
            existing_dongle_cfg.pin if existing_dongle_cfg else None
        )

        decision = _walk_dongle(dongle, existing_pin)
        if decision == "skip":
            continue
        if decision == "remove":
            removed.append(dongle.id)
            continue
        # decision is a pin dict
        new_pins[dongle.id] = decision

    # Anything to write?
    if not new_pins and not removed:
        click.echo()
        click.echo("No changes. Re-run any time.")
        return

    # Show summary + confirm
    click.echo()
    click.echo("─ Summary ─")
    if new_pins:
        click.echo(f"  Adding/updating {len(new_pins)} pin(s):")
        for did, p in new_pins.items():
            click.echo(
                f"    • {did} → {p['decoder']} @ "
                f"{p['freq_hz'] / 1e6:.3f} MHz"
            )
    if removed:
        click.echo(f"  Removing {len(removed)} pin(s):")
        for did in removed:
            click.echo(f"    • {did}")
    click.echo()
    if not click.confirm("Write these changes to your site.toml?", default=True):
        click.echo("Cancelled. No changes written.")
        return

    data = _read_toml(target)
    _apply_pin_dicts_to_toml(data, new_pins, removed)
    _write_toml(target, data)
    click.echo()
    click.echo(f"✓ Updated {target}")
    click.echo()
    click.echo(
        "These pins will take effect on the next `rfcensus hybrid` "
        "run. (`scan` and `inventory` ignore pins on purpose so an "
        "old pin can't silently break your scans.)"
    )


def _walk_dongle(dongle, existing_pin) -> Any:
    """Walk one dongle through the pinning prompts.

    Returns one of:
      • "skip"            — no change to this dongle's pin status
      • "remove"          — explicitly remove an existing pin
      • dict              — the new pin settings (decoder, freq_hz,
                            optional sample_rate, access_mode)

    Selection logic lives in `engine.pin_session` so the TUI can
    drive the same flow with Textual widgets instead of click prompts.
    """
    click.echo()
    click.echo("─" * 72)
    click.echo(f" {dongle.id}  ({dongle.model})")
    if dongle.antenna:
        click.echo(f"  antenna: {dongle.antenna.name}")
    else:
        click.echo("  antenna: (none)")
    if existing_pin:
        click.echo(
            f"  current pin: {existing_pin.decoder} @ "
            f"{existing_pin.freq_hz / 1e6:.3f} MHz"
        )
    click.echo()

    if existing_pin:
        click.echo("  This dongle already has a pin. What do you want to do?")
        options = [
            "Keep the current pin as-is",
            "Replace it with a different pin",
            "Remove the pin (return dongle to scheduler pool)",
        ]
        choice = _menu(options, prompt=">")
        if choice == 0:
            return "skip"
        if choice == 2:
            return "remove"
        # else fall through to the pin-builder
    else:
        click.echo(
            "  Should this dongle be dedicated to a specific decoder + "
            "frequency for the whole session?"
        )
        options = [
            "No — leave it for the scheduler (current behavior)",
            "Yes — pick a decoder + frequency to pin to",
        ]
        choice = _menu(options, prompt=">")
        if choice == 0:
            return "skip"

    # Build the pin: pick frequency, then decoder, then validate
    freq_hz = _pick_frequency(dongle)
    if freq_hz is None:
        click.echo("  Cancelled — leaving this dongle unpinned.")
        return "skip"

    decoder_name = _pick_decoder(freq_hz)
    if decoder_name is None:
        click.echo("  Cancelled.")
        return "skip"

    # Validate via the headless API. Should always pass here because
    # _pick_frequency already filtered to dongle-coverable freqs and
    # _pick_decoder picks from the registry — but if a custom freq
    # slipped through outside the dongle's antenna range, we want to
    # tell the user before writing the TOML.
    from rfcensus.engine.pin_session import validate_pin
    result = validate_pin(
        dongle,
        freq_hz=freq_hz,
        decoder=decoder_name,
        sample_rate=None,
        access_mode="exclusive",
    )
    if not result.ok:
        click.echo("  ✗ Pin failed validation:")
        for err in result.errors:
            click.echo(f"      • {err.message}")
        return "skip"
    return result.spec


def _pick_frequency(dongle) -> int | None:
    """Prompt for the pin frequency. Returns Hz or None if cancelled.

    Selection logic comes from `engine.pin_session.available_frequencies`
    so the TUI's edit-pin modal can present the same picks.
    """
    from rfcensus.engine.pin_session import available_frequencies

    covering = available_frequencies(dongle)

    if not covering:
        click.echo(
            "  No common frequencies match this dongle's antenna + "
            "hardware range. You can still type a custom frequency."
        )
        return _custom_freq()

    click.echo()
    click.echo("  Which frequency should this dongle be pinned to?")
    options = [opt.label for opt in covering] + ["Custom — type my own"]
    while True:
        choice = _menu_with_back(options, prompt=">")
        if choice == -1:
            return None
        if choice < len(covering):
            return covering[choice].freq_hz
        # Custom
        custom = _custom_freq()
        if custom is not None:
            return custom


def _custom_freq() -> int | None:
    """Prompt for a custom frequency. Parsing in `pin_session.parse_custom_freq`."""
    from rfcensus.engine.pin_session import parse_custom_freq

    while True:
        raw = click.prompt(
            "  Frequency (e.g. 433.92M, 162M, 850k, 2400000)",
            default="", show_default=False,
        )
        if not raw.strip():
            return None
        try:
            return parse_custom_freq(raw)
        except ValueError as exc:
            click.echo(f"    × {exc}")


def _pick_decoder(freq_hz: int) -> str | None:
    """Prompt for a decoder. Suggested-for-freq decoders sorted first.

    Selection logic comes from `engine.pin_session.available_decoders`.
    """
    from rfcensus.engine.pin_session import available_decoders

    options = available_decoders(freq_hz)
    if not options:
        click.echo(
            "  No decoders are registered. Check your install — "
            "`rfcensus list decoders` should show some."
        )
        return None

    click.echo()
    suggested_names = [o.name for o in options if o.suggested]
    if suggested_names:
        click.echo(
            f"  Suggested decoders for "
            f"{freq_hz / 1e6:.3f} MHz: {', '.join(suggested_names)}"
        )

    labels = [opt.label for opt in options]
    choice = _menu_with_back(labels, prompt=">")
    if choice == -1:
        return None
    return options[choice].name


# ────────────────────────────────────────────────────────────────────
# Display helpers
# ────────────────────────────────────────────────────────────────────


def _say_header() -> None:
    click.echo("═" * 72)
    click.echo(" rfcensus pin — dedicate dongles to specific decoders")
    click.echo("═" * 72)
    click.echo()
    click.echo(
        " A pinned dongle runs one decoder at one frequency for the entire"
    )
    click.echo(
        " session. Pinned dongles are removed from the scheduler's pool —"
    )
    click.echo(
        " the rest keep doing the normal exploration scan."
    )


def _show_fleet(config, registry) -> None:
    click.echo()
    click.echo("─ Detected fleet ─")
    for dongle in registry.usable():
        existing = next(
            (d for d in config.dongles if d.id == dongle.id), None,
        )
        ant = dongle.antenna.name if dongle.antenna else "(no antenna)"
        pin_str = ""
        if existing and existing.pin:
            pin_str = (
                f"  📌 pinned: {existing.pin.decoder} @ "
                f"{existing.pin.freq_hz / 1e6:.3f} MHz"
            )
        click.echo(f"  {dongle.id}  ({dongle.model}, {ant}){pin_str}")


# ────────────────────────────────────────────────────────────────────
# TOML I/O
# ────────────────────────────────────────────────────────────────────


def _read_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _write_toml(path: Path, data: dict[str, Any]) -> None:
    if tomli_w is None:
        raise click.ClickException(
            "tomli_w not available — can't write TOML. "
            "Install it: pip install tomli-w"
        )
    path.write_text(tomli_w.dumps(data), encoding="utf-8")


def _apply_pins_to_toml(
    data: dict[str, Any], specs: list,
) -> int:
    """Apply parsed PinSpec list to the TOML dict in-place.

    Returns the number of stanzas changed. Each spec finds (or creates)
    the matching dongle stanza by id and sets its `pin` sub-table.
    """
    n_changed = 0
    by_id = {s.get("id"): s for s in data.get("dongles", [])}
    for spec in specs:
        stanza = by_id.get(spec.dongle_id)
        if stanza is None:
            # The dongle isn't in the config yet — most likely the user
            # is trying to pin a dongle by serial that hasn't been
            # added via `rfcensus setup`. Create a minimal stanza so
            # the pin survives, but warn.
            click.echo(
                f"  ! Note: dongle {spec.dongle_id} isn't in your "
                f"config yet. Adding a minimal stanza. Run "
                f"`rfcensus setup new` to assign it an antenna.",
                err=True,
            )
            stanza = {"id": spec.dongle_id, "model": "rtlsdr_v3"}
            data.setdefault("dongles", []).append(stanza)
            by_id[spec.dongle_id] = stanza
        new_pin: dict[str, Any] = {
            "decoder": spec.decoder,
            "freq_hz": spec.freq_hz,
        }
        if spec.sample_rate is not None:
            new_pin["sample_rate"] = spec.sample_rate
        if spec.access_mode.value != "exclusive":
            new_pin["access_mode"] = spec.access_mode.value
        if stanza.get("pin") != new_pin:
            stanza["pin"] = new_pin
            n_changed += 1
    return n_changed


def _apply_pin_dicts_to_toml(
    data: dict[str, Any],
    new_pins: dict[str, dict[str, Any]],
    removed: list[str],
) -> None:
    """Apply wizard-built pin dicts and removals to the TOML dict
    in-place."""
    by_id = {s.get("id"): s for s in data.get("dongles", [])}
    for did, pin_dict in new_pins.items():
        stanza = by_id.get(did)
        if stanza is None:
            click.echo(
                f"  ! Note: dongle {did} isn't in config; adding a "
                f"minimal stanza.",
                err=True,
            )
            stanza = {"id": did, "model": "rtlsdr_v3"}
            data.setdefault("dongles", []).append(stanza)
            by_id[did] = stanza
        # Strip None-valued optional fields before writing
        clean = {k: v for k, v in pin_dict.items() if v is not None}
        if clean.get("access_mode") == "exclusive":
            del clean["access_mode"]  # default; omit for tidiness
        stanza["pin"] = clean
    for did in removed:
        stanza = by_id.get(did)
        if stanza and "pin" in stanza:
            del stanza["pin"]
