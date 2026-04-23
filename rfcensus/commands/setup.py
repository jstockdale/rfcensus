"""`rfcensus setup` — interactive wizard for dongle + antenna configuration.

Detects connected dongles and walks the user through assigning an
antenna to each. Designed to be re-runnable: existing config is
preserved, only the `[[dongles]]` stanzas are updated.

The wizard is opinionated about being friendly:

• "I don't know" is always a valid answer
• Telescopic-whip users get explicit cm measurements
• Frequencies are explained ("315 MHz: TPMS, older security") not just numbered
• Custom antennas can be defined inline
• Existing dongle config is shown and can be reused

For non-interactive bulk use, see `rfcensus init` (writes a default
config without prompting) or edit ~/.config/rfcensus/site.toml directly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
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

from rfcensus.commands._frequency_guide import (
    COMMON_FREQUENCIES,
    FrequencyProfile,
    beginner_recommendations,
    find_profile,
    quarter_wave_cm,
)
from rfcensus.commands.base import run_async
from rfcensus.config.loader import load_config
from rfcensus.hardware.dongle import Dongle
from rfcensus.hardware.registry import detect_hardware
from rfcensus.utils.paths import config_dir, site_config_path


# ──────────────────────────────────────────────────────────────────
# State the wizard accumulates
# ──────────────────────────────────────────────────────────────────


@dataclass
class _DongleAssignment:
    """The wizard's decision about one dongle."""

    dongle: Dongle
    antenna_id: str | None = None
    skip: bool = False
    notes: str = ""


@dataclass
class _WizardState:
    detected: list[Dongle]
    existing_dongles_by_serial: dict[str, dict[str, Any]] = field(default_factory=dict)
    existing_antennas: list[dict[str, Any]] = field(default_factory=list)
    custom_antennas: list[dict[str, Any]] = field(default_factory=list)
    assignments: list[_DongleAssignment] = field(default_factory=list)
    library_antennas: list[dict[str, Any]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────


@click.group(
    name="setup", invoke_without_command=True,
    help=(
        "Interactively configure connected dongles and antennas.\n\n"
        "Without a subcommand, walks every detected dongle. Use "
        "`setup new` to walk only dongles that aren't already in your config."
    ),
)
@click.option("--config", "config_path", type=click.Path(path_type=Path),
              help="Site config path (default: XDG config dir)")
@click.option("--dry-run", is_flag=True,
              help="Show what would be written, don't modify config")
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None, dry_run: bool) -> None:
    """Interactively configure connected dongles and antennas.

    Detects what's plugged in, walks you through each one. Re-run any
    time you swap hardware — existing config is preserved, only dongle
    stanzas are updated.
    """
    # Default behavior when no subcommand is given: walk every dongle
    # (current setup behavior).
    if ctx.invoked_subcommand is None:
        run_async(_setup(config_path, dry_run, only_new=False))
    else:
        # Stash the parent flags for subcommands to read
        ctx.ensure_object(dict)
        ctx.obj["config_path"] = config_path
        ctx.obj["dry_run"] = dry_run


@cli.command(name="new")
@click.pass_context
def cli_new(ctx: click.Context) -> None:
    """Walk only dongles that aren't already in your config.

    Use this after plugging in additional hardware. Existing dongle
    assignments are preserved untouched; you only answer questions for
    the new ones.
    """
    config_path = ctx.obj.get("config_path") if ctx.obj else None
    dry_run = ctx.obj.get("dry_run", False) if ctx.obj else False
    run_async(_setup(config_path, dry_run, only_new=True))


async def _setup(config_path: Path | None, dry_run: bool, *, only_new: bool = False) -> None:
    target = config_path or site_config_path()

    _say_header(only_new=only_new)

    # 1. Detect hardware
    click.echo("Detecting connected hardware...\n")
    registry = await detect_hardware(force=True)
    if not registry.dongles:
        click.echo(
            "No SDR dongles detected. Plug one in (RTL-SDR or HackRF), "
            "make sure the kernel sees it (`lsusb`), and rerun.",
            err=True,
        )
        click.echo(
            "If this is unexpected, check that you've installed the udev "
            "rules — usually `sudo apt install rtl-sdr` or `sudo apt install hackrf`.",
            err=True,
        )
        raise SystemExit(1)

    # 2. Load existing config (if any) — we preserve everything except [[dongles]]
    state = _WizardState(detected=registry.dongles)
    state.library_antennas = _load_library_antennas()

    if target.exists():
        try:
            existing = _read_toml(target)
            for d in existing.get("dongles", []):
                serial = d.get("serial")
                if serial:
                    state.existing_dongles_by_serial[serial] = d
            state.existing_antennas = existing.get("antennas", [])
            click.echo(
                f"Found existing config at {target}. "
                f"I'll preserve everything except dongle assignments.\n"
            )
        except Exception as exc:
            click.echo(
                f"Could not parse existing config ({exc}). "
                f"I'll write a new one.\n",
                err=True,
            )
            existing = {}
    else:
        existing = {}
        click.echo(f"No config at {target} yet — starting fresh.\n")

    click.echo(f"Found {len(state.detected)} dongle(s):\n")
    for i, d in enumerate(state.detected, 1):
        prior = ""
        if d.serial and d.serial in state.existing_dongles_by_serial:
            prior_ant = state.existing_dongles_by_serial[d.serial].get("antenna", "?")
            prior = f"  (previously: antenna={prior_ant})"
        # Show the disambiguated id (e.g. "rtlsdr-00000001-idx0") so the
        # user can tell duplicates apart even when they share a serial.
        click.echo(
            f"  [{i}] {d.model:<20s} serial={d.serial or '(none)':<12s} "
            f"id={d.id}{prior}"
        )
    click.echo()

    # If any serials are duplicated, the setup wizard MUST NOT proceed
    # to write a config — index-based ids would break on USB re-enumeration.
    # Offer to fix immediately by reserializing.
    duplicate_warnings = [
        d for d in registry.diagnostics if d and d.startswith("⚠")
    ]
    if duplicate_warnings:
        for w in duplicate_warnings:
            click.echo(w)
        click.echo()
        click.echo("rfcensus can write distinct serials to fix this permanently.")
        click.echo("This is a one-time operation per dongle.")
        click.echo()
        click.echo("If you decline, setup cannot continue: writing index-based")
        click.echo("ids to your config would break on the next USB re-enumeration.")
        click.echo()
        if not click.confirm(
            "Reserialize colliding dongles now?", default=True
        ):
            click.echo()
            click.echo("OK, no changes made.")
            click.echo("You can run `rfcensus serialize` later, then re-run `rfcensus setup`.")
            return

        # Hand off to the serialize command's logic
        from rfcensus.commands.serialize import _run as run_serialize
        await run_serialize(target, dry_run=False, yes=True)

        # Re-detect to pick up the new serials
        click.echo()
        click.echo("Re-detecting hardware after serialization...")
        registry = await detect_hardware(force=True)
        state.detected = registry.dongles
        # Refresh the existing-dongle map since serials may have moved
        if target.exists():
            try:
                existing = _read_toml(target)
                state.existing_dongles_by_serial = {}
                for d in existing.get("dongles", []):
                    serial = d.get("serial")
                    if serial:
                        state.existing_dongles_by_serial[serial] = d
            except Exception:
                pass
        click.echo()
        click.echo(f"Now found {len(state.detected)} dongle(s):")
        for i, d in enumerate(state.detected, 1):
            click.echo(f"  [{i}] {d.model:<20s} serial={d.serial or '(none)'}")
        click.echo()

    # If `setup new` was invoked, filter the walk list to dongles that
    # aren't already in the existing config. Existing dongles are
    # preserved untouched — we copy their previous assignments into
    # state.assignments so the writer's "merge" sees them and doesn't
    # accidentally drop them.
    if only_new:
        already_known = []
        new_dongles = []
        for d in state.detected:
            is_known = (
                d.serial and d.serial in state.existing_dongles_by_serial
            ) or any(
                exist.get("id") == d.id
                for exist in (existing.get("dongles", []) if existing else [])
            )
            if is_known:
                already_known.append(d)
            else:
                new_dongles.append(d)

        # Preserve existing assignments by appending an Assignment record
        # whose values come from the existing config stanza
        for d in already_known:
            existing_stanza = (
                state.existing_dongles_by_serial.get(d.serial)
                if d.serial else None
            )
            if existing_stanza is None:
                continue
            state.assignments.append(_make_preserved_assignment(d, existing_stanza))

        if not new_dongles:
            click.echo(
                "All connected dongles are already in your config. "
                "Nothing new to set up."
            )
            click.echo()
            click.echo(
                "  • Run `rfcensus setup` to walk through every dongle "
                "(change antennas, etc.)"
            )
            click.echo("  • Run `rfcensus list dongles` to see current configuration")
            return

        # Limit the walk to only new dongles
        state.detected = new_dongles
        click.echo(
            f"Of {len(already_known) + len(new_dongles)} attached dongle(s), "
            f"{len(already_known)} already configured (preserved), "
            f"{len(new_dongles)} new:"
        )
        for i, d in enumerate(new_dongles, 1):
            click.echo(
                f"  [{i}] {d.model:<20s} serial={d.serial or '(none)':<12s} "
                f"id={d.id}"
            )
        click.echo()

    if not click.confirm("Walk through each dongle now?", default=True):
        click.echo("OK, no changes made.")
        return

    # 3. Walk through each dongle
    for i, dongle in enumerate(state.detected, 1):
        click.echo()
        click.echo("─" * 72)
        click.echo(f" Dongle {i} of {len(state.detected)}: {dongle.model} (serial {dongle.serial or '(none)'})")
        click.echo("─" * 72)
        assignment = _walk_dongle(dongle, state)
        state.assignments.append(assignment)

    # 4. Show summary, confirm, write
    click.echo()
    click.echo("═" * 72)
    click.echo(" Summary")
    click.echo("═" * 72)
    for a in state.assignments:
        if a.skip:
            click.echo(f"  · {a.dongle.id}: skipped")
        else:
            click.echo(f"  ✓ {a.dongle.id} → antenna={a.antenna_id or '(none)'}")
    if state.custom_antennas:
        click.echo()
        click.echo(f"  + {len(state.custom_antennas)} new antenna(s) defined")

    if dry_run:
        click.echo("\n[dry-run] Would write to:", target)
        new_toml = _build_new_config(target, existing, state)
        click.echo("\n--- new config ---")
        click.echo(new_toml)
        return

    if not click.confirm("\nWrite this configuration?", default=True):
        click.echo("OK, no changes made.")
        return

    _write_config(target, existing, state)
    click.echo(f"\nWrote {target}")
    click.echo("\nNext steps:")
    click.echo("  rfcensus doctor       # verify the setup")
    click.echo("  rfcensus scan         # quick 5-min survey")


# ──────────────────────────────────────────────────────────────────
# Per-dongle interactive flow
# ──────────────────────────────────────────────────────────────────


def _walk_dongle(dongle: Dongle, state: _WizardState) -> _DongleAssignment:
    assignment = _DongleAssignment(dongle=dongle)

    # If we've seen this serial before, offer to reuse
    if dongle.serial and dongle.serial in state.existing_dongles_by_serial:
        prior = state.existing_dongles_by_serial[dongle.serial]
        prior_ant = prior.get("antenna", "(none)")
        click.echo(f"\n  This dongle was previously configured with antenna='{prior_ant}'.")
        choice = _menu([
            f"Keep using '{prior_ant}'",
            "Pick a different antenna",
            "Skip this dongle (don't include in config)",
        ], prompt="What would you like to do?")
        if choice == 0:
            assignment.antenna_id = prior_ant
            return assignment
        if choice == 2:
            assignment.skip = True
            return assignment
        # else: fall through to picking new antenna

    # What kind of antenna is connected?
    click.echo("\n  What antenna is connected to this dongle?")
    options = [
        "Telescopic whip (I'll help you tune it)",
        "Tuned whip — already cut for a specific frequency",
        "Wideband (discone, log-periodic, etc.)",
        "Specific antenna from the library",
        "Custom antenna I'll define now",
        "I don't know / I'll figure it out later",
        "Skip this dongle",
    ]
    choice = _menu(options, prompt="  >")

    if choice == 0:
        assignment.antenna_id = _flow_telescopic(dongle, state)
    elif choice == 1:
        assignment.antenna_id = _flow_tuned_whip(dongle, state)
    elif choice == 2:
        assignment.antenna_id = _flow_wideband(dongle, state)
    elif choice == 3:
        assignment.antenna_id = _flow_pick_from_library(dongle, state)
    elif choice == 4:
        assignment.antenna_id = _flow_define_custom(dongle, state)
    elif choice == 5:
        assignment.antenna_id = _flow_dont_know(dongle, state)
    elif choice == 6:
        assignment.skip = True

    return assignment


def _flow_telescopic(dongle: Dongle, state: _WizardState) -> str | None:
    """Help the user tune a telescopic whip to a specific frequency."""
    click.echo("\n  Great. Telescopic whips work best when set to a quarter")
    click.echo("  wavelength of the signal you're trying to receive.\n")
    profile = _pick_frequency(dongle, allow_dont_know=True)
    if profile is None:
        click.echo("  No problem. I'll assign a generic small whip for now.")
        click.echo("  You can re-run `rfcensus setup` to refine this later.")
        return "whip_generic_small"

    qw = profile.quarter_wave_cm
    hw = profile.half_wave_cm
    click.echo(f"\n  For {profile.label.split(' — ')[0]}, set your telescopic whip to:")
    click.echo(f"     • {qw:.1f} cm (quarter wavelength) — recommended")
    click.echo(f"     • {hw:.1f} cm (half wavelength) — needs a ground plane")
    click.echo()
    click.echo("  Tip: extend each section, measure with a ruler from the base of")
    click.echo("  the threaded connector to the very tip. Aim for the quarter-wave.")
    if profile.notes:
        click.echo(f"\n  Note: {profile.notes}")

    # Accept either Enter (use suggested length, original target frequency)
    # OR an actual length in cm (recompute resonant frequency from physics).
    # Cheap whips have section boundaries and tolerances — if the user
    # ends up at 11 cm instead of 8.2 cm, the antenna is actually tuned to
    # ~681 MHz, not 915 MHz. Capturing that honestly gives the matcher
    # accurate suitability scores later.
    click.echo()
    actual_length_str = click.prompt(
        "  Press Enter when set to the recommended length, or type the\n"
        "  ACTUAL length in cm (e.g. '11' or '11.5') if it's different",
        default="", show_default=False,
    ).strip()

    actual_qw_cm = qw
    actual_resonant_hz = profile.freq_hz
    if actual_length_str:
        try:
            actual_qw_cm = float(actual_length_str)
            if actual_qw_cm <= 0 or actual_qw_cm > 500:
                raise ValueError("length must be 0-500 cm")
            # Speed of light c ≈ 29979.2458 cm/MHz; quarter wavelength
            # in cm = c / (4 × f_MHz) → invert to get f_MHz from cm.
            actual_resonant_mhz = 29979.2458 / (4 * actual_qw_cm)
            actual_resonant_hz = int(actual_resonant_mhz * 1_000_000)
            click.echo(
                f"\n  Got {actual_qw_cm:.1f} cm. That's a quarter-wave at "
                f"{actual_resonant_mhz:.0f} MHz."
            )
            if abs(actual_resonant_mhz - profile.freq_hz / 1e6) > profile.freq_hz / 1e6 * 0.10:
                click.echo(
                    f"  (Different from the {profile.freq_hz/1e6:.0f} MHz target "
                    f"you picked — I'll record the actual tuning so the band "
                    f"matcher gets accurate scores.)"
                )
        except ValueError as exc:
            click.echo(
                f"  Couldn't parse '{actual_length_str}' as a length: {exc}. "
                f"Using the suggested {qw:.1f} cm."
            )
            actual_qw_cm = qw
            actual_resonant_hz = profile.freq_hz

    # Telescopic whips ALWAYS create a custom antenna stanza, even
    # when the user picks a frequency that has a library entry. Two
    # reasons:
    #   • Library antennas like whip_915 represent purchased products
    #     with specific impedance matching networks. A telescopic
    #     extended to 8.2 cm has a different impedance profile and a
    #     usefully wider bandwidth (±15% vs the library's ±10%).
    #   • Conflating the two caused a bug where, e.g., a telescopic
    #     "set for 162 MHz marine" would fail to match a 144 MHz
    #     amateur band even though physics says the antenna covers it
    #     fine.
    # The "Specific antenna from the library" path stays available for
    # users who actually have a Diamond X510 or Comet GP-3 etc.
    res_mhz = actual_resonant_hz / 1_000_000
    name = (
        f"Telescopic whip @ {actual_qw_cm:.1f} cm "
        f"(tuned for {res_mhz:.0f} MHz)"
    )
    custom = {
        "id": f"whip_telescopic_{int(res_mhz)}mhz",
        "name": name,
        "antenna_type": "whip",
        "resonant_freq_hz": actual_resonant_hz,
        "usable_range": [int(actual_resonant_hz * 0.5), int(actual_resonant_hz * 1.5)],
        "gain_dbi": 2.15,
        "polarization": "vertical",
    }
    state.custom_antennas.append(custom)
    return custom["id"]


def _flow_tuned_whip(dongle: Dongle, state: _WizardState) -> str | None:
    """User has a permanently tuned whip. Find a library match or define custom."""
    click.echo("\n  What frequency is the whip tuned for?")
    profile = _pick_frequency(dongle, allow_dont_know=False)
    if profile is None:
        return None

    suggested = profile.suggested_antenna_id
    if suggested and any(a["id"] == suggested for a in state.library_antennas):
        click.echo(f"\n  I'll use the library antenna '{suggested}'.")
        return suggested

    click.echo(f"\n  No library match for {profile.freq_hz/1e6:.1f} MHz; defining a custom one.")
    return _define_custom_for_freq(profile.freq_hz, state)


def _flow_wideband(dongle: Dongle, state: _WizardState) -> str | None:
    click.echo("\n  Wideband antennas trade off optimal performance for any single")
    click.echo("  frequency in exchange for covering a wide range. Common types:")
    click.echo()
    options = [
        "Discone (typically 25 MHz - 1.3 GHz)",
        "1090 MHz ADS-B dipole (filter + tuned dipole)",
        "Marine VHF antenna (150-175 MHz)",
        "800-900 MHz magmount (cellular/public safety)",
        "Generic small whip (24 MHz - 1.7 GHz, mediocre everywhere)",
        "Other / custom — I'll define it",
    ]
    library_ids = [
        "discone",
        "dipole_1090",
        "marine_vhf",
        "magmount_800_900",
        "whip_generic_small",
        None,
    ]
    choice = _menu(options, prompt="  Which type?")
    if choice < 5 and library_ids[choice] is not None:
        return library_ids[choice]
    return _flow_define_custom(dongle, state)


def _flow_pick_from_library(dongle: Dongle, state: _WizardState) -> str | None:
    if not state.library_antennas:
        click.echo("  Antenna library is empty (this is unusual).")
        return None
    click.echo("\n  Antennas in your library:")
    options = []
    ids = []
    for a in state.library_antennas:
        rng_low = a["usable_range"][0] / 1e6
        rng_high = a["usable_range"][1] / 1e6
        options.append(f"{a['id']:<24s} {a['name']}  ({rng_low:.0f}-{rng_high:.0f} MHz)")
        ids.append(a["id"])
    options.append("None of these — define a custom one")
    ids.append(None)
    choice = _menu(options, prompt="  Which one?")
    if ids[choice] is None:
        return _flow_define_custom(dongle, state)
    return ids[choice]


def _flow_define_custom(dongle: Dongle, state: _WizardState) -> str | None:
    click.echo("\n  Defining a custom antenna.")
    name = click.prompt("  Friendly name (e.g. 'Roof discone, 25-1300 MHz')")
    antenna_id = click.prompt(
        "  Short id (lowercase, no spaces)",
        default=name.lower().replace(" ", "_").replace(",", "")[:24],
    )

    has_resonance = click.confirm(
        "  Is this a tuned antenna with a specific resonant frequency?",
        default=False,
    )
    resonant: int | None = None
    if has_resonance:
        resonant_mhz = click.prompt("  Resonant frequency (MHz)", type=float)
        resonant = int(resonant_mhz * 1_000_000)

    low_mhz = click.prompt("  Lowest usable frequency (MHz)", type=float)
    high_mhz = click.prompt("  Highest usable frequency (MHz)", type=float)
    if high_mhz <= low_mhz:
        click.echo("  Range high must exceed low. Swapping.")
        low_mhz, high_mhz = high_mhz, low_mhz

    custom: dict[str, Any] = {
        "id": antenna_id,
        "name": name,
        "antenna_type": "custom",
        "usable_range": [int(low_mhz * 1_000_000), int(high_mhz * 1_000_000)],
        "gain_dbi": 2.15,
        "polarization": "vertical",
    }
    if resonant:
        custom["resonant_freq_hz"] = resonant
    state.custom_antennas.append(custom)
    return antenna_id


def _flow_dont_know(dongle: Dongle, state: _WizardState) -> str | None:
    """The "I don't know" branch — guide rather than fail."""
    click.echo("\n  No problem. Let me help you figure this out.\n")
    click.echo("  A few questions:\n")

    looks = _menu(
        [
            "It looks like a small antenna with extending sections (telescopic)",
            "It looks like a stiff straight rod (probably tuned whip)",
            "It's bigger / has many spokes / looks unusual (probably wideband)",
            "I really don't know what's connected",
        ],
        prompt="  Physical description?",
    )

    if looks == 0:
        click.echo("\n  That's a telescopic whip. The most useful thing you can do")
        click.echo("  is decide what frequency you want to listen on, then tune the")
        click.echo("  whip to that frequency.")
        click.echo()
        click.echo("  If you have no preference, here are good places to start:")
        for p in beginner_recommendations():
            click.echo(f"    • {p.freq_hz/1e6:6.1f} MHz: {p.typical_traffic.split('.')[0]}")
        click.echo()
        if click.confirm("  Pick a starting frequency now?", default=True):
            return _flow_telescopic(dongle, state)
        click.echo("  OK. I'll assign a generic whip and you can re-run `rfcensus setup` later.")
        return "whip_generic_small"

    if looks == 1:
        click.echo("\n  That's a tuned whip. Quarter-wave whips have a specific length")
        click.echo("  that tells you their resonant frequency:\n")
        click.echo("     ~5-10 cm  →  ~900-1800 MHz (cellular, ISM)")
        click.echo("     ~15-20 cm →  ~400-500 MHz (UHF)")
        click.echo("     ~20-25 cm →  ~300-350 MHz (TPMS, security)")
        click.echo("     ~50-75 cm →  ~100-150 MHz (VHF, marine, AIS)")
        click.echo()
        if click.confirm("  Want to estimate from physical length?", default=True):
            length_cm = click.prompt("  Length in cm (tip to base of connector)", type=float)
            est_mhz = (29979.2458 / length_cm) / 4
            click.echo(f"\n  Quarter-wave at {length_cm:.1f} cm → ~{est_mhz:.0f} MHz")
            close = find_profile(int(est_mhz * 1_000_000), tolerance_pct=0.15)
            if close:
                click.echo(f"  That's near {close.label}")
                if click.confirm("  Use that as the antenna's tuned frequency?", default=True):
                    return _flow_tuned_whip_with_freq(close, state)
            return _define_custom_for_freq(int(est_mhz * 1_000_000), state)
        click.echo("  OK. Defaulting to generic small whip.")
        return "whip_generic_small"

    if looks == 2:
        click.echo("\n  Sounds like a wideband antenna. Let's see if we can identify it.")
        return _flow_wideband(dongle, state)

    # Really doesn't know — try a fleet-aware suggestion before falling
    # back to the generic whip. We look at what the user's other dongles
    # already cover and recommend an antenna that fills the biggest gap.
    suggestion_id = _try_fleet_aware_suggestion(dongle, state)
    if suggestion_id:
        return suggestion_id

    click.echo("\n  That's fine. I'll assign a generic whip — it'll work mediocre-ly")
    click.echo("  everywhere from 24 MHz to 1.7 GHz, which is enough to start.")
    click.echo("  When you've got more info, run `rfcensus setup` again to refine.")
    return "whip_generic_small"


def _try_fleet_aware_suggestion(
    dongle: Dongle, state: _WizardState,
) -> str | None:
    """Compute a fleet-aware antenna recommendation and prompt the user.

    Looks at the user's other dongles (with already-assigned antennas)
    and the enabled bands. Identifies the biggest coverage gap and
    suggests an antenna from the catalog. Falls back to a quarter-wave
    whip recommendation with physical length if the catalog has no match.

    Returns the chosen antenna id, or None if the user declines (caller
    falls back to generic small whip).
    """
    try:
        from rfcensus.hardware.antenna_suggestion import suggest_for_new_dongle
        from rfcensus.config.loader import load_config
        from rfcensus.hardware.antenna import Antenna

        config = load_config()
        enabled = config.enabled_bands()
        if not enabled:
            return None

        # Build "other dongles" — those that already have an antenna
        # assigned (either preserved from prior config or just chosen
        # earlier in this wizard run). Skip the dongle we're configuring.
        other_dongles: list[Dongle] = []
        assigned_by_id = {a.dongle.id: a.antenna_id for a in state.assignments}
        antennas_by_id = {a.id: a for a in config.antennas}
        for d in state.detected:
            if d.id == dongle.id:
                continue
            ant_id = assigned_by_id.get(d.id)
            if not ant_id:
                continue
            ant_cfg = antennas_by_id.get(ant_id)
            if ant_cfg:
                # Make a shallow copy with antenna attached for matcher
                d_copy = type(d)(**{k: getattr(d, k) for k in d.__dataclass_fields__})
                d_copy.antenna = Antenna.from_config(ant_cfg)
                other_dongles.append(d_copy)

        if not other_dongles:
            return None  # No other dongles configured; fleet-awareness has nothing to add

        # Build available antenna catalog
        available = [Antenna.from_config(a) for a in config.antennas]

        suggestion = suggest_for_new_dongle(
            new_dongle=dongle,
            other_dongles=other_dongles,
            enabled_bands=enabled,
            available_antennas=available,
        )

        click.echo()
        click.echo("  ─ Looking at your other configured dongles ─")
        click.echo(f"  {suggestion.rationale}")
        click.echo()

        if suggestion.antenna_id:
            ant_name = next(
                (a.name for a in config.antennas if a.id == suggestion.antenna_id),
                suggestion.antenna_id,
            )
            click.echo(f"  Recommended: {ant_name} (id={suggestion.antenna_id})")
            if suggestion.bands_covered:
                click.echo(
                    f"  Would unlock: "
                    f"{', '.join(suggestion.bands_covered[:5])}"
                    + (
                        f" and {len(suggestion.bands_covered) - 5} more"
                        if len(suggestion.bands_covered) > 5 else ""
                    )
                )
            click.echo()
            if click.confirm("  Use this antenna?", default=True):
                return suggestion.antenna_id

        elif suggestion.is_quarter_wave_fallback:
            # No catalog match — recommend a telescopic whip extended to
            # the calculated quarter-wave length
            click.echo(
                f"  Quarter-wave length for "
                f"{suggestion.fallback_freq_mhz:.0f} MHz: "
                f"{suggestion.fallback_length_cm:.1f} cm"
            )
            click.echo(
                f"  A telescopic whip extended to that length will work "
                f"as a starting point."
            )
            if suggestion.buy_suggestion:
                click.echo()
                click.echo(f"  Tip: {suggestion.buy_suggestion}")
            click.echo()
            click.echo(
                f"  I'll assign the generic small whip for now. You can "
                f"swap to a tuned antenna later by re-running setup."
            )
            return "whip_generic_small"

        return None
    except Exception as exc:
        log.debug("fleet-aware suggestion failed: %s", exc)
        return None


def _flow_tuned_whip_with_freq(profile: FrequencyProfile, state: _WizardState) -> str | None:
    suggested = profile.suggested_antenna_id
    if suggested and any(a["id"] == suggested for a in state.library_antennas):
        return suggested
    return _define_custom_for_freq(profile.freq_hz, state)


def _define_custom_for_freq(freq_hz: int, state: _WizardState) -> str:
    qw = quarter_wave_cm(freq_hz)
    custom = {
        "id": f"whip_{int(freq_hz/1_000_000)}mhz",
        "name": f"Quarter-wave whip @ {qw:.1f} cm ({freq_hz/1e6:.1f} MHz)",
        "antenna_type": "whip",
        "resonant_freq_hz": freq_hz,
        "usable_range": [int(freq_hz * 0.5), int(freq_hz * 1.5)],
        "gain_dbi": 2.15,
        "polarization": "vertical",
    }
    state.custom_antennas.append(custom)
    return custom["id"]


# ──────────────────────────────────────────────────────────────────
# Frequency picker (used in multiple flows)
# ──────────────────────────────────────────────────────────────────


def _pick_frequency(
    dongle: Dongle, *, allow_dont_know: bool
) -> FrequencyProfile | None:
    """Ask the user to pick a target frequency. May return None for 'I don't know'."""
    options: list[str] = []
    profiles: list[FrequencyProfile | None] = []

    # Filter to frequencies this dongle can actually tune to
    low, high = dongle.capabilities.freq_range_hz
    for p in COMMON_FREQUENCIES:
        if low <= p.freq_hz <= high:
            options.append(p.label)
            profiles.append(p)

    if not options:
        click.echo("  This dongle's tuning range doesn't cover any common bands.")
        click.echo("  Falling back to manual entry.")
        return _manual_frequency_entry(dongle)

    options.append("Other (enter manually)")
    profiles.append(None)
    if allow_dont_know:
        options.append("I don't know — help me decide")
        profiles.append(None)

    while True:
        choice = _menu(options, prompt="  Frequency?")
        if choice == len(options) - 1 and allow_dont_know:
            # User chose "I don't know — help me decide". The beginner
            # flow has its own back support; if it returns None we
            # treat that as "back to this frequency menu" rather than
            # cascading None up.
            result = _suggest_frequency_for_beginner(dongle)
            if result is None:
                continue  # back to the frequency menu
            return result
        if choice == len(options) - (2 if allow_dont_know else 1):
            return _manual_frequency_entry(dongle)
        return profiles[choice]


def _manual_frequency_entry(dongle: Dongle) -> FrequencyProfile | None:
    while True:
        raw = click.prompt(
            "  Frequency in MHz (or 'cancel')", default="cancel", show_default=False
        )
        if raw.lower() in ("cancel", "c", "q", ""):
            return None
        try:
            mhz = float(raw)
        except ValueError:
            click.echo("  Not a valid number. Try again.")
            continue
        freq_hz = int(mhz * 1_000_000)
        low, high = dongle.capabilities.freq_range_hz
        if not (low <= freq_hz <= high):
            click.echo(
                f"  {mhz} MHz is outside this dongle's range "
                f"({low/1e6:.0f}-{high/1e6:.0f} MHz). Try again."
            )
            continue
        # Synthesize a minimal profile
        return FrequencyProfile(
            label=f"{mhz:.3f} MHz (custom)",
            freq_hz=freq_hz,
            region="unknown",
            typical_traffic="(user-specified frequency)",
            decoders=(),
        )


def _suggest_frequency_for_beginner(dongle: Dongle) -> FrequencyProfile | None:
    """Walk a confused user through picking a starting frequency.

    Both this menu and the follow-up "which one?" support a back option
    so the user can recover from a wrong choice without Ctrl-C. Returns
    None if the user backs out twice (signal to the caller to drop back
    to the previous-level frequency menu).
    """
    while True:
        click.echo("\n  Let's narrow it down. What kind of stuff are you curious about?\n")
        options = [
            "Cars, sensors, weather stations, smart home (315/433/915 MHz)",
            "Aircraft (1090 MHz ADS-B — needs HackRF or NESDR with proper antenna)",
            "Boats, marine (162 MHz AIS)",
            "Amateur radio (144 MHz APRS, 2m)",
            "Public safety (P25 trunked, 700/800 MHz)",
            "I just want to listen to whatever is around — pick something",
        ]
        choice = _menu_with_back(options, prompt="  >")
        if choice == -1:
            return None  # Back out to previous menu

        # Expanded the "pick something" set to include 315 MHz (TPMS,
        # very common for users with cars) — previously it only had
        # 4 options which felt arbitrary compared to the parent menu.
        candidates_by_choice: dict[int, list[int]] = {
            0: [915_000_000, 433_920_000, 315_000_000],
            1: [1_090_000_000],
            2: [162_000_000],
            3: [144_390_000],
            4: [851_000_000],
            5: [915_000_000, 433_920_000, 315_000_000, 162_000_000, 144_390_000],
        }
        low, high = dongle.capabilities.freq_range_hz
        candidates = [
            f for f in candidates_by_choice.get(choice, [])
            if low <= f <= high
        ]
        if not candidates:
            click.echo("  Unfortunately your dongle doesn't tune to those bands.")
            click.echo("  Falling back to manual entry.")
            return _manual_frequency_entry(dongle)

        if len(candidates) == 1:
            profile = find_profile(candidates[0])
            if profile:
                click.echo(f"\n  Recommended: {profile.label}")
                click.echo(f"  Why: {profile.typical_traffic}")
                return profile

        # Multiple — let them pick (with back option to revisit category)
        profiles = [find_profile(f) for f in candidates if find_profile(f)]
        profile_options = [
            f"{p.freq_hz/1e6:6.1f} MHz — {p.typical_traffic[:60]}"
            for p in profiles if p
        ]
        pick = _menu_with_back(profile_options, prompt="  Which one?")
        if pick == -1:
            continue  # Back to category menu — loop top
        return profiles[pick]


# ──────────────────────────────────────────────────────────────────
# Library + config IO
# ──────────────────────────────────────────────────────────────────


def _load_library_antennas() -> list[dict[str, Any]]:
    """Load the built-in antenna library so we can offer them as choices."""
    try:
        config = load_config()
        out = []
        for a in config.antennas:
            out.append({
                "id": a.id,
                "name": a.name,
                "usable_range": list(a.usable_range),
                "resonant_freq_hz": a.resonant_freq_hz,
            })
        return out
    except Exception:
        return []


def _build_new_config(target: Path, existing: dict, state: _WizardState) -> str:
    """Construct the new TOML config text."""
    if tomli_w is None:
        raise RuntimeError("tomli_w is required to write TOML; pip install tomli-w")

    out = dict(existing)  # shallow copy

    # Replace [[dongles]] entirely with the wizard's assignments
    new_dongles = []
    for a in state.assignments:
        if a.skip:
            continue
        d = a.dongle
        new_dongles.append({
            "id": d.id,
            "serial": d.serial or "",
            "model": d.model,
            "driver": d.driver,
            "antenna": a.antenna_id or "whip_generic_small",
        })
    out["dongles"] = new_dongles

    # Append any custom antennas defined during the wizard.
    # Preserve user's existing custom antennas; new ones with new ids get added.
    existing_antenna_ids = {a.get("id") for a in out.get("antennas", [])}
    merged_antennas = list(out.get("antennas", []))
    for a in state.custom_antennas:
        if a["id"] not in existing_antenna_ids:
            merged_antennas.append(a)
    if merged_antennas:
        out["antennas"] = merged_antennas

    # Ensure site stanza exists
    if "site" not in out:
        out["site"] = {"name": "default", "region": "US"}

    return tomli_w.dumps(out)


def _write_config(target: Path, existing: dict, state: _WizardState) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    text = _build_new_config(target, existing, state)
    header = (
        "# rfcensus site configuration\n"
        "# Generated/updated by `rfcensus setup`. You can edit by hand;\n"
        "# re-running setup will preserve everything except [[dongles]].\n\n"
    )
    target.write_text(header + text, encoding="utf-8")


def _read_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────
# Pretty UI helpers
# ──────────────────────────────────────────────────────────────────


def _say_header(*, only_new: bool = False) -> None:
    click.echo("═" * 72)
    if only_new:
        click.echo(" rfcensus setup new — configure only unconfigured dongles")
    else:
        click.echo(" rfcensus setup")
    click.echo("═" * 72)
    click.echo()


def _make_preserved_assignment(
    dongle: Dongle, existing_stanza: dict[str, Any]
) -> _DongleAssignment:
    """Build an assignment that preserves the prior config for an
    already-known dongle. Used by `setup new` so existing dongles flow
    through the wizard's writer step unchanged."""
    return _DongleAssignment(
        dongle=dongle,
        antenna_id=existing_stanza.get("antenna"),
        skip=False,
        notes=existing_stanza.get("notes", ""),
    )


def _menu(options: list[str], *, prompt: str = "?") -> int:
    """Render a numbered menu and return the user's 0-indexed choice."""
    for i, opt in enumerate(options, 1):
        click.echo(f"    {i}) {opt}")
    while True:
        raw = click.prompt(f"  {prompt}", default="1", show_default=False)
        try:
            n = int(raw)
        except ValueError:
            click.echo(f"  Enter a number from 1 to {len(options)}.")
            continue
        if 1 <= n <= len(options):
            return n - 1
        click.echo(f"  Choose 1 to {len(options)}.")


def _menu_with_back(options: list[str], *, prompt: str = "?") -> int:
    """Like _menu but appends a '← back' option as the last entry.
    Returns -1 if the user chose back, otherwise the 0-indexed choice
    among the original options.

    Used in sub-menus where the user might realize they took a wrong
    turn and want to return to the parent menu without having to Ctrl-C
    and restart. Critical for the "I don't know — help me decide" flow
    which can land users in a small option set they can't escape from.
    """
    augmented = options + ["← back to previous menu"]
    choice = _menu(augmented, prompt=prompt)
    if choice == len(options):
        return -1
    return choice
