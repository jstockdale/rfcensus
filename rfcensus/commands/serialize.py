"""`rfcensus serialize` — write distinct serials to colliding RTL-SDR dongles.

Standalone command for the case where a user wants to fix duplicate
serials without re-running the full setup wizard. The setup wizard also
calls into the same orchestration as a precondition gate.

Safety pattern (per the design discussion):

• Plan first, show what we'd do, get explicit y/N confirmation
• Pre-flight check (all target dongles openable)
• For each assignment requiring a change:
    - Backup EEPROM to a timestamped file
    - Write new serial via rtl_eeprom
    - Try software USB reset (best-effort)
    - Verify by re-probing rtl_test
    - If verify fails → ask user to physically replug, retry once
    - If still fails → stop, surface the backup-restore command
• Sequential, not parallel
• Update site.toml atomically (.tmp + rename, .bak.<timestamp> first)
"""

from __future__ import annotations

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

try:
    import tomli_w
except ImportError:  # pragma: no cover
    tomli_w = None

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from rfcensus.commands.base import run_async
from rfcensus.hardware.dongle import Dongle
from rfcensus.hardware.registry import detect_hardware
from rfcensus.hardware.serialization import (
    ReserializationPlan,
    SerialAssignment,
    WriteOutcome,
    _current_rtl_serials,
    backup_eeprom,
    format_replug_prompt,
    plan_reserialization,
    preflight_check,
    try_software_reset,
    verify_serial_via_rtl_test,
    wait_for_serials,
    write_serial,
)
from rfcensus.utils.logging import get_logger
from rfcensus.utils.paths import site_config_path

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────


@click.command(name="serialize")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path),
    help="Site config path (default: XDG config dir)",
)
@click.option(
    "--dry-run", is_flag=True,
    help="Show the plan, do not write to any EEPROM or config file",
)
@click.option(
    "--yes", "-y", is_flag=True,
    help="Skip confirmation prompts (DANGEROUS — only for scripts)",
)
def cli(config_path: Path | None, dry_run: bool, yes: bool) -> None:
    """Write distinct serials to RTL-SDR dongles that share a serial.

    Many cheap RTL-SDR boards ship from the factory with serial 00000001.
    If you have more than one, they collide and can't be reliably
    distinguished across reboots. This command writes new serials to
    the EEPROM so each dongle is permanently identifiable.

    Safe to run anytime — if no duplicates are found, nothing happens.
    """
    run_async(_run(config_path, dry_run, yes))


async def _run(config_path: Path | None, dry_run: bool, yes: bool) -> None:
    target = config_path or site_config_path()

    click.echo("═" * 72)
    click.echo(" rfcensus serialize")
    click.echo("═" * 72)
    click.echo()

    # 1. Detect attached hardware
    click.echo("Detecting connected hardware...")
    registry = await detect_hardware(force=True)
    rtl_dongles = [d for d in registry.dongles if d.driver == "rtlsdr"]

    if not rtl_dongles:
        click.echo("No RTL-SDR dongles detected. Nothing to do.")
        return

    # 2. Load existing config (if any) — both serial set and full stanzas
    existing_config_serials = _load_existing_config_serials(target)
    existing_config_dongles = _load_existing_config_dongles(target)

    # 3. Detect collisions and prompt for keeper choice if interactive.
    # When multiple dongles share a serial AND have different models, ask
    # the user which one keeps the original serial. This is meaningful
    # because the user often has a strong preference (e.g. "my V4 should
    # stay 00000001 because that's what my config calls it").
    keeper_overrides = _prompt_for_keepers(rtl_dongles, existing_config_dongles, yes)

    # 4. Plan
    plan = plan_reserialization(
        detected=rtl_dongles,
        existing_config_serials=frozenset(existing_config_serials),
        keeper_overrides=keeper_overrides,
        existing_config=existing_config_dongles,
    )

    if plan.is_empty:
        click.echo(
            f"Found {len(rtl_dongles)} RTL-SDR dongle(s), all with unique serials. "
            f"Nothing to do."
        )
        return

    # 4. Show the plan
    _show_plan(plan, rtl_dongles, existing_config_serials, target)

    # 5. Confirm
    if dry_run:
        click.echo("\n[dry-run] Would execute the plan above. No changes made.")
        return

    if not yes:
        click.echo()
        click.echo("⚠  This writes to the EEPROM of physical hardware.")
        click.echo("   Interrupted writes can leave the EEPROM in a bad state.")
        click.echo("   We back up first, but please don't unplug mid-write.")
        if not click.confirm("Proceed?", default=False):
            click.echo("Aborted. No changes made.")
            return

    # 6. Pre-flight: all target dongles openable?
    target_indices = [a.driver_index for a in plan.changes]
    click.echo("\nPre-flight check (all target dongles openable)...")
    ok, errs = await preflight_check(target_indices)
    if not ok:
        click.echo("\nPre-flight failed:", err=True)
        for e in errs:
            click.echo(f"  ✗ {e}", err=True)
        click.echo("\nResolve these issues and rerun.", err=True)
        raise SystemExit(1)
    click.echo("  ✓ all target dongles are openable")

    # 7. Batch flow: backup all → write all → single replug prompt with
    # hotplug detection → verify all. This is much friendlier than the
    # old per-dongle "write, prompt, replug, verify" loop because the
    # user doesn't have to figure out which physical dongle just got
    # rewritten between writes.
    outcomes = await _execute_batch(plan, rtl_dongles, yes=yes)

    # Aggregate outcomes — abort early-exit messaging only if NOTHING
    # succeeded; otherwise show what worked and what didn't.
    n_failed = sum(1 for o in outcomes if not o.fully_succeeded)
    if n_failed:
        click.echo()
        click.echo(
            f"⚠ {n_failed} of {len(outcomes)} dongle(s) had issues.",
            err=True,
        )
        for o in outcomes:
            if not o.fully_succeeded and o.error:
                click.echo(
                    f"   • idx={o.assignment.driver_index} "
                    f"({o.assignment.original_serial} → "
                    f"{o.assignment.new_serial}): {o.error}",
                    err=True,
                )
        # Hard-fail only if nothing got written
        n_writes = sum(1 for o in outcomes if o.write_success)
        if n_writes == 0:
            click.echo(
                "  No EEPROMs were modified. Resolve the issues and re-run.",
                err=True,
            )
            raise SystemExit(1)
        click.echo(
            f"  {n_writes} of {len(outcomes)} write(s) succeeded; "
            f"continuing with config update.",
            err=True,
        )

    # 8. Update site.toml
    click.echo()
    click.echo("─" * 72)
    click.echo(" Config update")
    click.echo("─" * 72)
    if target.exists():
        backup_target_path = _backup_site_toml(target)
        click.echo(f"  Backed up existing config to {backup_target_path}")
        n_changed = _update_site_toml(target, plan)
        if n_changed:
            click.echo(f"  Updated {n_changed} dongle stanza(s) in {target}")
        else:
            click.echo(f"  No dongle stanzas in {target} needed updating")
    else:
        click.echo(f"  No config at {target} — skipping config update")
        click.echo(f"  Run `rfcensus setup` next to write dongle assignments")

    # 9. Done
    click.echo()
    click.echo("═" * 72)
    click.echo(" Done. New serials are live.")
    click.echo("═" * 72)
    click.echo()
    click.echo("Next steps:")
    if target.exists():
        click.echo("  rfcensus setup new      # configure the newly-serialized dongles")
    else:
        click.echo("  rfcensus setup          # walk through every dongle and assign antennas")
    click.echo("  rfcensus doctor          # verify the setup")


# ──────────────────────────────────────────────────────────────────
# One-dongle execution flow
# ──────────────────────────────────────────────────────────────────


async def _execute_batch(
    plan: ReserializationPlan, rtl_dongles: list[Dongle], *, yes: bool
) -> list[WriteOutcome]:
    """Batched flow: backup all → write all → bulk replug → verify all.

    Replaces the old per-dongle "write then prompt user to identify and
    replug THAT specific dongle" loop, which was confusing because the
    user had no clear visual signal which physical device just got
    rewritten. With batching, the user just unplugs every reserialized
    dongle in one go, plugs them back in (any order, any port), and we
    detect each one as it shows up.
    """
    outcomes: list[WriteOutcome] = [
        WriteOutcome(assignment=a) for a in plan.changes
    ]

    # ── Phase 1: backup all EEPROMs ──
    click.echo()
    click.echo("─" * 72)
    click.echo(f" Phase 1/4: backing up EEPROMs ({len(plan.changes)} dongle(s))")
    click.echo("─" * 72)
    for outcome in outcomes:
        a = outcome.assignment
        try:
            outcome.backup_path = await backup_eeprom(
                a.driver_index, a.original_serial,
            )
            click.echo(
                f"  ✓ idx={a.driver_index} ({a.model:<18s}) → "
                f"{outcome.backup_path.name}"
            )
        except Exception as exc:
            outcome.error = f"backup failed: {exc}"
            click.echo(
                f"  ✗ idx={a.driver_index}: backup failed — {exc}", err=True,
            )

    # If even one backup failed, abort writes — we don't want to write
    # without a recovery path.
    if any(o.error for o in outcomes):
        click.echo()
        click.echo(
            "✗ One or more backups failed. Aborting before any writes.",
            err=True,
        )
        click.echo(
            "  No EEPROMs have been modified. Resolve the backup issue "
            "(check `lsof | grep rtl` for processes holding the device) "
            "and rerun.",
            err=True,
        )
        return outcomes

    # ── Phase 2: write all new serials, back to back ──
    click.echo()
    click.echo("─" * 72)
    click.echo(f" Phase 2/4: writing new serials ({len(plan.changes)} dongle(s))")
    click.echo("─" * 72)
    for outcome in outcomes:
        a = outcome.assignment
        success, message = await write_serial(a.driver_index, a.new_serial)
        outcome.write_success = success
        if success:
            click.echo(
                f"  ✓ idx={a.driver_index} ({a.model:<18s}): "
                f"{a.original_serial} → {a.new_serial}"
            )
        else:
            outcome.error = f"write failed: {message}"
            click.echo(
                f"  ✗ idx={a.driver_index}: write failed — {message}",
                err=True,
            )

    n_writes_ok = sum(1 for o in outcomes if o.write_success)
    if n_writes_ok == 0:
        click.echo()
        click.echo("✗ All writes failed. EEPROMs are unchanged.", err=True)
        return outcomes

    # ── Phase 3: bulk unplug + replug, with hotplug detection ──
    expected_new_serials = {
        o.assignment.new_serial for o in outcomes if o.write_success
    }
    click.echo()
    click.echo("─" * 72)
    click.echo(" Phase 3/4: replug to apply changes")
    click.echo("─" * 72)
    click.echo()
    click.echo(
        f"  EEPROMs are written but the kernel still has the OLD serials "
        f"cached.\n"
        f"  To activate the new serials, please:\n"
    )
    click.echo(f"    1. Unplug all {n_writes_ok} reserialized dongle(s) from USB")
    click.echo("    2. Plug them back in (any order, any port)")
    click.echo()

    # Try a software reset first — sometimes that's enough to trigger
    # re-enumeration without physical interaction.
    click.echo("  → trying software USB reset first...")
    indices = [o.assignment.driver_index for o in outcomes if o.write_success]
    reset_worked = await try_software_reset(indices[0]) if indices else False
    if reset_worked:
        await asyncio.sleep(2.0)
        # Quick verify check — maybe we don't need physical replug at all
        current = await _current_rtl_serials()
        if expected_new_serials.issubset({s for s in current if s}):
            click.echo("  ✓ software reset triggered re-enumeration; serials are live")
            for outcome in outcomes:
                if outcome.write_success:
                    outcome.verify_success = True
            return outcomes
        click.echo(
            "      software reset didn't expose new serials — physical "
            "replug needed"
        )
    else:
        click.echo("      software reset unavailable (this is normal on most setups)")

    # Decision: prompt for replug detection if stdin is a TTY (we have
    # a human to wait for). The --yes flag does NOT bypass this — same
    # lesson as the picker bug (v0.5.4): --yes is for skipping yes/no
    # confirmations, not for skipping content/state operations like
    # waiting for the user to physically replug a dongle. Replugging is
    # not optional — the kernel has the OLD serials cached and the new
    # serials won't be visible until re-enumeration happens.
    import sys
    if not sys.stdin.isatty():
        click.echo()
        click.echo(
            "  No TTY detected (running in a script/pipe). Skipping "
            "interactive replug detection. Replug the dongles manually, "
            "then run `rfcensus doctor` to verify."
        )
        return outcomes

    # Single-phase model: just watch for the new serials to appear.
    # We don't gate on "all unplugged first" — that would force the user
    # into a batch-only flow. Sequential unplug+replug works just as
    # well: each new serial counts as it arrives, regardless of order
    # or whether other dongles have been unplugged yet.
    expected_list = sorted(expected_new_serials)
    click.echo()
    click.echo(
        f"  Waiting for {n_writes_ok} new serial(s) to appear:"
    )
    for s in expected_list:
        click.echo(f"      • {s}")
    click.echo()
    click.echo(
        "  You can unplug and replug all at once, or one at a time —"
    )
    click.echo("  whatever's easier. We'll detect each new serial as it shows up.")
    click.echo("  (Press Ctrl-C if you've already finished or want to skip.)")
    click.echo()

    def _on_arrived(serial: str, n_seen: int, n_total: int) -> None:
        click.echo(f"      ✓ detected new serial {serial} ({n_seen}/{n_total})")

    try:
        replug_ok, seen_serials, missing_serials = await wait_for_serials(
            expected_new_serials,
            timeout_s=None,  # indefinite — user may walk away
            on_arrived=_on_arrived,
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        click.echo("\n  (skipped wait; running final verify)")
        seen_serials = set()
        missing_serials = expected_new_serials

    # ── Phase 4: final verification ──
    click.echo()
    click.echo("─" * 72)
    click.echo(" Phase 4/4: final verification")
    click.echo("─" * 72)
    final_serials = set(s for s in await _current_rtl_serials() if s)
    for outcome in outcomes:
        if not outcome.write_success:
            continue
        new_s = outcome.assignment.new_serial
        if new_s in final_serials:
            outcome.verify_success = True
            click.echo(f"  ✓ {new_s} present and accounted for")
        else:
            outcome.error = (
                outcome.error
                or f"new serial {new_s} not detected after replug"
            )
            click.echo(
                f"  ✗ {new_s} not detected. Try unplugging that "
                f"specific dongle once more, then run `rfcensus doctor`.",
                err=True,
            )
    return outcomes


# ──────────────────────────────────────────────────────────────────
# Plan display
# ──────────────────────────────────────────────────────────────────


def _show_plan(
    plan: ReserializationPlan,
    detected: list[Dongle],
    existing_config_serials: set[str],
    target: Path,
) -> None:
    by_serial: dict[str, list[SerialAssignment]] = {}
    for a in plan.assignments:
        by_serial.setdefault(a.original_serial, []).append(a)

    click.echo()
    click.echo(f"Found {len([s for s, g in by_serial.items() if len(g) > 1])} colliding "
               f"serial(s) across {len(plan.assignments)} dongle(s).")
    click.echo()

    for original_serial, group in by_serial.items():
        click.echo(f"  Serial {original_serial} (currently shared by {len(group)} dongle(s)):")
        for a in group:
            arrow = "keeps" if a.keeps_original else "→"
            tag = f"  ({a.model}, idx {a.driver_index})"
            if a.keeps_original:
                click.echo(f"    • {a.original_serial} {arrow} {a.new_serial}{tag}")
            else:
                click.echo(f"    • {a.original_serial} {arrow} {a.new_serial}{tag}")

    if existing_config_serials:
        click.echo()
        click.echo(
            f"  (Existing config at {target} references serials "
            f"{sorted(existing_config_serials)} — those are protected from collision.)"
        )


# ──────────────────────────────────────────────────────────────────
# Config IO
# ──────────────────────────────────────────────────────────────────


def _load_existing_config_serials(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("could not parse %s: %s", path, exc)
        return set()
    out: set[str] = set()
    for d in data.get("dongles", []):
        s = d.get("serial")
        if s:
            out.add(s)
    return out


def _load_existing_config_dongles(path: Path) -> dict[str, dict]:
    """Load full dongle stanzas keyed by serial. Used by the interactive
    keeper picker to show "this dongle was previously configured as ..."
    info to help the user pick which dongle keeps the original serial.
    """
    if not path.exists():
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("could not parse %s: %s", path, exc)
        return {}
    out: dict[str, dict] = {}
    for d in data.get("dongles", []):
        s = d.get("serial")
        if s:
            out[s] = d
    return out


def _has_meaningful_choice(group: list, existing_config: dict[str, dict]) -> bool:
    """Decide whether picking a keeper for this group involves a real
    decision the user should make.

    A choice is meaningful when:
      • The group contains more than one model (e.g. V4 + 2 generics) —
        the user almost certainly cares which keeps the original serial
      • OR existing config maps this serial to a specific model AND
        the group contains a dongle of that model (the user previously
        named this dongle and the choice should be confirmed)

    Otherwise — if all dongles in the group are the same model with no
    config preference — they're functionally interchangeable and we can
    silently pick the lowest driver_index without bothering the user.
    """
    models = {d.model for d in group}
    if len(models) > 1:
        return True
    if not group:
        return False
    serial = group[0].serial
    if serial in existing_config:
        prior_model = existing_config[serial].get("model")
        if prior_model and any(d.model == prior_model for d in group):
            return True
    return False


def _prompt_for_keepers(
    rtl_dongles: list,
    existing_config: dict[str, dict],
    yes: bool,
) -> dict[str, int]:
    """Interactive picker: when a serial is shared by multiple dongles
    AND there's a meaningful choice between them (different models, or
    existing config has a preference), ask the user which one keeps
    that serial. Skip silently when the choice doesn't matter (all
    dongles in the group are the same model).

    The `yes` flag does NOT bypass this — it only skips yes/no
    confirmations like "Proceed with EEPROM write?". Picking a keeper
    is a content choice, not a confirmation. The picker is bypassed
    only when stdin is not a TTY (e.g. running in a script).

    Returns a map of serial → driver_index of the chosen keeper.
    """
    from rfcensus.hardware.serialization import describe_dongle_for_picker
    import sys

    overrides: dict[str, int] = {}
    is_tty = sys.stdin.isatty()

    # Group dongles by serial to find collisions
    by_serial: dict[str, list] = {}
    for d in rtl_dongles:
        if d.serial:
            by_serial.setdefault(d.serial, []).append(d)

    for serial, group in sorted(by_serial.items()):
        if len(group) <= 1:
            continue

        # Sort by driver_index for stable display
        group_sorted = sorted(
            group,
            key=lambda d: (d.driver_index if d.driver_index is not None else 999_999),
        )

        # Compute the auto-default that plan_reserialization would pick
        # so we can show it as the default in the prompt
        prior_model = existing_config.get(serial, {}).get("model")
        default_dongle = None
        default_reason = ""
        if prior_model:
            matches = [d for d in group_sorted if d.model == prior_model]
            if matches:
                default_dongle = matches[0]
                default_reason = f"(matches existing config: model={prior_model})"
        if default_dongle is None:
            default_dongle = group_sorted[0]
            default_reason = "(lowest driver index)"

        # Skip the prompt when there's no meaningful choice (all same
        # model, no config preference) — silently use the auto-default.
        if not _has_meaningful_choice(group_sorted, existing_config):
            overrides[serial] = default_dongle.driver_index
            continue

        # Skip the prompt when stdin is not a TTY (script/pipe context).
        # Auto-default with a log entry so the user can see the choice.
        if not is_tty:
            log.info(
                "no TTY for keeper picker; auto-selecting idx=%d (%s) "
                "to keep serial %s %s",
                default_dongle.driver_index, default_dongle.model,
                serial, default_reason,
            )
            overrides[serial] = default_dongle.driver_index
            continue

        click.echo()
        click.echo(f"  Multiple dongles share serial {serial}.")
        click.echo(f"  Which should keep {serial}? Others will be renumbered.")
        click.echo()
        for i, d in enumerate(group_sorted, 1):
            marker = " [default]" if d is default_dongle else ""
            click.echo(
                f"    {i}) {describe_dongle_for_picker(d, existing_config)}{marker}"
            )
        if default_reason:
            click.echo(
                f"     Default: option "
                f"{group_sorted.index(default_dongle) + 1} {default_reason}"
            )
        click.echo()
        choice = click.prompt(
            f"  Keep serial {serial} on which dongle?",
            type=click.IntRange(1, len(group_sorted)),
            default=group_sorted.index(default_dongle) + 1,
            show_default=True,
        )
        chosen = group_sorted[choice - 1]
        overrides[serial] = chosen.driver_index
        click.echo(
            f"  ✓ Dongle at idx={chosen.driver_index} ({chosen.model}) "
            f"will keep serial {serial}."
        )

    return overrides


def _backup_site_toml(target: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = target.with_suffix(f".toml.bak.{timestamp}")
    shutil.copy2(target, backup_path)
    return backup_path


def _update_site_toml(target: Path, plan: ReserializationPlan) -> int:
    """Rewrite [[dongles]] stanzas to reflect new serials.

    For each dongle that changed serial, we look for stanzas matching by
    `serial` field. If multiple stanzas share the same old serial (which
    is itself a sign of a misconfigured config), we update them all — but
    in practice this case shouldn't arise after one clean serialize run.

    Returns the number of stanzas updated.
    """
    if tomli_w is None:
        raise RuntimeError("tomli_w is required to write TOML; pip install tomli-w")

    data = tomllib.loads(target.read_text(encoding="utf-8"))
    dongles_list: list[dict[str, Any]] = data.get("dongles", [])

    # Map old serial → list of new assignments. If multiple dongles changed
    # FROM the same old serial, we need to be careful — but again, this
    # should be a one-off transient.
    changes_by_old: dict[str, list[SerialAssignment]] = {}
    for a in plan.changes:
        changes_by_old.setdefault(a.original_serial, []).append(a)

    n_changed = 0
    for stanza in dongles_list:
        old_serial = stanza.get("serial")
        if old_serial in changes_by_old:
            # Pop the next assignment for this old serial. If multiple, FIFO.
            new_a = changes_by_old[old_serial].pop(0)
            stanza["serial"] = new_a.new_serial
            # Update id if it was the auto-generated form
            old_id = stanza.get("id", "")
            if old_id == f"rtlsdr-{old_serial}" or old_id.startswith(f"rtlsdr-{old_serial}-idx"):
                stanza["id"] = f"rtlsdr-{new_a.new_serial}"
            n_changed += 1
            if not changes_by_old[old_serial]:
                del changes_by_old[old_serial]

    data["dongles"] = dongles_list

    # Atomic write: tmp + rename
    text = tomli_w.dumps(data)
    header = (
        "# rfcensus site configuration\n"
        "# Updated by `rfcensus serialize` — old serials renamed to new.\n\n"
    )
    tmp = target.with_suffix(".toml.tmp")
    tmp.write_text(header + text, encoding="utf-8")
    tmp.replace(target)
    return n_changed
