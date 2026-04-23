"""`rfcensus inventory` and `rfcensus scan` — run surveys."""

from __future__ import annotations

import re
from pathlib import Path

import click

from rfcensus.commands.base import bootstrap, run_async
from rfcensus.engine.session import SessionRunner
from rfcensus.events import DecodeEvent, EmitterEvent
from rfcensus.reporting.report import ReportBuilder
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


_DURATION_RE = re.compile(r"^\s*(\d+)\s*([smh]?)\s*$")
_INDEFINITE_TOKENS = frozenset({"0", "forever", "indefinite", "inf", "infinity"})


def _parse_duration(value: str, default_s: float) -> float:
    """Parse '30m', '1h', '45s', or '600' into seconds.

    Special values: '0' or 'forever' return 0.0, which the runner
    interprets as indefinite (loop until SIGINT).
    """
    if not value:
        return default_s
    if value.strip().lower() in _INDEFINITE_TOKENS:
        return 0.0
    m = _DURATION_RE.match(value.lower())
    if not m:
        raise click.BadParameter(
            f"invalid duration '{value}'. Use like '30m', '1h', '45s', '600', "
            f"or '0'/'forever' for indefinite."
        )
    n = int(m.group(1))
    suffix = m.group(2) or "s"
    factor = {"s": 1, "m": 60, "h": 3600}[suffix]
    return float(n * factor)


def _parse_gain(value: str | None) -> str:
    """Validate a gain value. Returns 'auto' or a numeric dB string.

    Accepts: 'auto', or a number 0-50 (dB). Rejects anything else with
    a clear error message.
    """
    if value is None or value == "":
        return "auto"
    v = value.strip().lower()
    if v == "auto":
        return "auto"
    try:
        n = float(v)
    except ValueError:
        raise click.BadParameter(
            f"invalid gain '{value}'. Use 'auto' or a dB value like '40'."
        )
    if not (0.0 <= n <= 50.0):
        raise click.BadParameter(
            f"gain {n} out of range. RTL-SDR tuner gain is typically 0-50 dB."
        )
    # Format without trailing zeros: 40.0 → "40", 37.2 → "37.2"
    if n == int(n):
        return str(int(n))
    return str(n)


def _build_inventory_cli(
    default_duration: str,
    command_name: str,
    help_text: str,
    *,
    default_per_band: str | None = None,
):
    @click.command(name=command_name, help=help_text)
    @click.option("--config", "config_path", type=click.Path(path_type=Path))
    @click.option(
        "--duration",
        default=default_duration,
        show_default=True,
        help="Run duration, e.g. 30m, 1h, 600. Use 0 or 'forever' to run until Ctrl-C.",
    )
    @click.option(
        "--bands",
        "band_filter",
        help="Comma-separated band IDs to restrict to.",
    )
    @click.option(
        "--capture-power",
        is_flag=True,
        help="Persist every power sample to SQLite (large).",
    )
    @click.option(
        "--include-ids",
        is_flag=True,
        help="Show raw device IDs in the final report.",
    )
    @click.option(
        "--output",
        type=click.Path(path_type=Path),
        help="Write report to file instead of stdout.",
    )
    @click.option("--json", "as_json", is_flag=True, help="JSON report output.")
    @click.option(
        "--all-bands",
        is_flag=True,
        help=(
            "Attempt every enabled band even if no antenna is well-matched. "
            "Severely detuned bands will produce poor or unreliable decodes; "
            "use this when you want to confirm absence rather than presence."
        ),
    )
    @click.option(
        "--per-band",
        default=default_per_band,
        show_default=True,
        help=(
            "Time per band per pass (e.g. 30s, 5m). When set, the run loops "
            "through the wave list as many times as fit in --duration. When "
            "omitted, --duration is divided evenly across active waves and "
            "each band runs once."
        ),
    )
    @click.option(
        "--gain",
        default="auto",
        show_default=True,
        help=(
            "RTL-SDR tuner gain in dB (e.g. 40), or 'auto' for AGC. "
            "Manual gain often catches more distant signals than auto. "
            "Try 30-45 dB for most setups."
        ),
    )
    @click.option(
        "--until-quiet",
        default=None,
        help=(
            "Exit early when no new emitter has been seen for this long "
            "(e.g. 10m, 1h). Useful for unattended runs — e.g. `inventory "
            "--duration forever --until-quiet 30m` runs until the site "
            "goes quiet for 30 min, then exits with results persisted."
        ),
    )
    @click.option(
        "--guided",
        is_flag=True,
        help=(
            "Interactive mode for users with one telescopic-whip dongle. "
            "Pauses between bands and prompts you to retune the whip to "
            "the right length. Single-pass only — incompatible with "
            "--per-band and --duration forever."
        ),
    )
    @click.option(
        "--kill-orphans",
        is_flag=True,
        help=(
            "If orphan SDR processes from a previous rfcensus session "
            "are detected (rtl_tcp, rtl_fm, multimon-ng, etc. from an "
            "uncleanly-exited scan), SIGTERM them before starting. "
            "Default is to warn and proceed — which typically means "
            "the dongles are busy and the scan fails mysteriously."
        ),
    )
    def cli(
        config_path: Path | None,
        duration: str,
        band_filter: str | None,
        capture_power: bool,
        include_ids: bool,
        output: Path | None,
        as_json: bool,
        all_bands: bool,
        per_band: str | None,
        gain: str,
        until_quiet: str | None,
        guided: bool,
        kill_orphans: bool,
    ) -> None:
        run_async(
            _run(
                config_path=config_path,
                duration=duration,
                band_filter=band_filter,
                capture_power=capture_power,
                include_ids=include_ids,
                output=output,
                as_json=as_json,
                all_bands=all_bands,
                per_band=per_band,
                gain=gain,
                until_quiet=until_quiet,
                guided=guided,
                kill_orphans=kill_orphans,
                command_name=command_name,
            )
        )

    return cli


async def _run(
    *,
    config_path: Path | None,
    duration: str,
    band_filter: str | None,
    capture_power: bool,
    include_ids: bool,
    output: Path | None,
    as_json: bool,
    all_bands: bool,
    per_band: str | None,
    gain: str,
    until_quiet: str | None,
    guided: bool,
    kill_orphans: bool,
    command_name: str,
) -> None:
    # Guided mode is single-pass-only by design. Round-robin or
    # indefinite would prompt the user dozens of times — hostile.
    if guided and (per_band or duration in ("0", "forever", "indefinite", "inf")):
        raise click.BadParameter(
            "--guided is for single-pass scans. Remove --per-band and use a "
            "finite --duration."
        )

    # Orphan-process check — two phases:
    #
    # Phase A (before bootstrap): scan /proc for leftover SDR processes
    # from an uncleanly-exited session. These commonly hold USB dongles
    # hostage, causing the "busy" probe errors that make a dongle look
    # dead. If --kill-orphans is set, clean them up here; otherwise just
    # note them and let Phase B decide whether to prompt.
    #
    # Phase B (after bootstrap + health-check, below): if any dongle
    # came back BUSY and we have orphans that look like they reference
    # the same device index, prompt the user (unless --kill-orphans was
    # already passed or stdout is non-interactive). This is the high-
    # evidence case — we know the orphans are causing observable harm.
    from rfcensus.utils.orphan_detect import (
        find_sdr_orphans, log_orphans,
        kill_orphans as kill_orphan_processes,
        guess_orphan_device_indices,
    )
    # Orphans younger than 2 seconds are almost certainly a legit
    # user process started just before rfcensus (e.g. `rtl_test` in
    # a nearby terminal). Excluding them avoids false positives.
    orphans_at_startup = find_sdr_orphans(min_age_s=2.0)
    if orphans_at_startup and kill_orphans:
        log.info(
            "--kill-orphans set; cleaning up %d orphan process(es) "
            "before probing hardware",
            len(orphans_at_startup),
        )
        log_orphans(orphans_at_startup)
        clean, forced = kill_orphan_processes(orphans_at_startup)
        log.info(
            "orphan cleanup: %d exited on SIGTERM, %d required SIGKILL",
            clean, forced,
        )
        # Brief settling delay so USB descriptors get reclaimed by
        # the kernel before we probe. Empirically 500ms is enough.
        import asyncio as _asyncio
        await _asyncio.sleep(0.5)
        # After killing, rescan — some might have spawned children
        # that are still running, or we might have new ones
        orphans_at_startup = find_sdr_orphans(min_age_s=0.0)
        if orphans_at_startup:
            log.warning(
                "still %d orphan(s) after cleanup; will re-check after probe",
                len(orphans_at_startup),
            )
    elif orphans_at_startup:
        # Don't prompt yet — wait until we know whether they're
        # actually causing problems (Phase B below).
        log.info(
            "detected %d orphan SDR process(es); will check after probe "
            "whether they're blocking any dongles",
            len(orphans_at_startup),
        )

    rt = await bootstrap(config_path=config_path, detect=True)
    duration_s = _parse_duration(duration, default_s=1800.0)
    indefinite = duration_s == 0.0
    per_band_s: float | None = None
    if per_band:
        per_band_s = _parse_duration(per_band, default_s=300.0)
        if per_band_s == 0.0:
            raise click.BadParameter(
                "--per-band cannot be 0/forever; use a positive duration like 30s or 5m"
            )

    until_quiet_s: float | None = None
    if until_quiet:
        until_quiet_s = _parse_duration(until_quiet, default_s=600.0)
        if until_quiet_s == 0.0:
            raise click.BadParameter(
                "--until-quiet cannot be 0/forever; use a positive duration like 10m or 1h"
            )

    # Indefinite runs require a per-band budget — without one, the first
    # band would run forever and the wave loop would never advance.
    if indefinite and per_band_s is None:
        per_band_s = 300.0  # 5 min default per band per pass
        click.echo(
            "Indefinite mode (--duration 0/forever) requires a per-band "
            "budget; defaulting to --per-band 5m. Override with --per-band.",
            err=True,
        )

    gain_normalized = _parse_gain(gain)

    if band_filter:
        wanted = {b.strip() for b in band_filter.split(",") if b.strip()}
        rt.config.bands.enabled = list(wanted)

    if not rt.registry.dongles:
        click.echo("No dongles detected. Run `rfcensus doctor` for diagnostics.", err=True)
        raise SystemExit(1)

    # ------------------------------------------------------------
    # Phase B of orphan handling: now that hardware is detected,
    # health-check early so we can surface BUSY dongles before the
    # scan commits. If BUSY dongles + orphans co-occur, the orphans
    # are almost certainly the cause. Prompt the user (unless
    # already auto-killed or stdout non-interactive).
    # ------------------------------------------------------------
    from rfcensus.hardware.health import check_all as _check_all_dongles
    from rfcensus.hardware.dongle import DongleStatus
    import sys as _sys

    log.info("health-checking %d dongle(s)", len(rt.registry.dongles))
    await _check_all_dongles(rt.registry.dongles)
    # Re-scan for orphans in case Phase A kill left children or a
    # new rtl_* got spawned between then and now. Also recomputes
    # the cmdline → device-index mapping for correlation.
    orphans_now = find_sdr_orphans(min_age_s=0.0)
    busy_dongles = [
        d for d in rt.registry.dongles
        if d.status == DongleStatus.BUSY
    ]

    if busy_dongles and orphans_now:
        # High-confidence evidence: probe failures + orphan processes.
        # Correlate by device index when possible.
        orphans_by_idx = guess_orphan_device_indices(orphans_now)
        log.warning(
            "⚠ %d dongle(s) returned BUSY from probe, and %d orphan "
            "SDR process(es) are running. The orphans are almost "
            "certainly holding these dongles:",
            len(busy_dongles), len(orphans_now),
        )
        for d in busy_dongles:
            matching = orphans_by_idx.get(d.driver_index, []) if d.driver_index is not None else []
            if matching:
                pid_list = ", ".join(
                    f"pid={o.pid} ({o.comm})" for o in matching
                )
                log.warning(
                    "  • %s (driver_index=%d, serial=%s): held by %s",
                    d.id, d.driver_index, d.serial or "?", pid_list,
                )
            else:
                log.warning(
                    "  • %s (driver_index=%s, serial=%s): BUSY but no "
                    "orphan explicitly references this index",
                    d.id,
                    d.driver_index if d.driver_index is not None else "?",
                    d.serial or "?",
                )

        # Decide how to proceed. Priority:
        #   1. --kill-orphans was passed → auto-kill, re-probe
        #   2. stdout is a TTY → prompt user
        #   3. non-TTY → just warn and proceed (scripts/CI)
        should_kill = kill_orphans
        if not should_kill and _sys.stdin.isatty() and _sys.stdout.isatty():
            click.echo("", err=True)  # spacing
            click.echo(
                "  [k] kill the orphans and retry the probe",
                err=True,
            )
            click.echo(
                "  [p] proceed anyway — BUSY dongles will be skipped (default)",
                err=True,
            )
            click.echo("  [q] quit", err=True)
            click.echo("", err=True)
            # Default to "proceed" — killing is a destructive action
            # (SIGTERM/SIGKILL to another process) and should not
            # happen from a stray Enter keypress. Users who want to
            # kill can press 'k' or pass --kill-orphans upfront.
            choice = click.prompt(
                "Choose", type=click.Choice(["k", "p", "q"]),
                default="p", show_default=True,
            )
            if choice == "q":
                click.echo("aborted by user", err=True)
                raise SystemExit(1)
            elif choice == "k":
                should_kill = True
            # else "p": fall through, proceed with busy dongles

        if should_kill:
            log.info(
                "killing %d orphan process(es) and re-probing hardware",
                len(orphans_now),
            )
            clean, forced = kill_orphan_processes(orphans_now)
            log.info(
                "orphan cleanup: %d exited on SIGTERM, %d required SIGKILL",
                clean, forced,
            )
            # USB kernel driver needs a beat to re-enumerate
            import asyncio as _asyncio
            await _asyncio.sleep(0.5)
            # Re-run the health check — any dongle that was BUSY
            # because of an orphan should now probe clean.
            log.info("re-health-checking %d dongle(s) after orphan cleanup",
                     len(rt.registry.dongles))
            # Reset BUSY dongles to DETECTED so check_all can try
            # them again (check_all only probes non-FAILED dongles).
            for d in rt.registry.dongles:
                if d.status == DongleStatus.BUSY:
                    d.status = DongleStatus.DETECTED
                    d.health_notes.clear()
            await _check_all_dongles(rt.registry.dongles)
            still_busy = [
                d for d in rt.registry.dongles
                if d.status == DongleStatus.BUSY
            ]
            if still_busy:
                log.warning(
                    "⚠ %d dongle(s) still BUSY after orphan cleanup — "
                    "something else is holding them (kernel driver, "
                    "another user's process, or hardware stuck state). "
                    "Proceeding; these dongles will be skipped.",
                    len(still_busy),
                )
            else:
                log.info(
                    "✓ all previously-BUSY dongles are now healthy"
                )

    elif busy_dongles and not orphans_now:
        # BUSY dongles but no orphans — something else is wrong.
        # Kernel driver, permissions, another user's process, stuck
        # hardware. Can't help with the orphan hammer.
        log.warning(
            "⚠ %d dongle(s) returned BUSY but no orphan SDR processes "
            "were found. Likely causes: kernel driver claimed the "
            "device (`rmmod dvb_usb_rtl28xxu` may help), another user "
            "or container is using it, or the device is in a stuck "
            "state (try unplugging and replugging).",
            len(busy_dongles),
        )
    elif orphans_now and not busy_dongles:
        # Orphans exist but nothing is obviously blocked. Probably
        # they're on dongles/ports rfcensus isn't using, or they
        # shared-exit cleanly. Warn at INFO so it's visible but
        # doesn't alarm.
        log.info(
            "note: %d orphan SDR process(es) are running but no "
            "dongles are BUSY — they may be inert. Use "
            "--kill-orphans if you want them cleaned up anyway.",
            len(orphans_now),
        )

    # ------------------------------------------------------------

    # Surface the coverage report BEFORE starting the session, so the
    # user can see what's being skipped (and re-run with --all-bands if
    # they want it anyway) before committing to the run.
    from rfcensus.engine.coverage import compute_coverage, render_coverage_report

    enabled = rt.config.enabled_bands()
    coverage = compute_coverage(enabled, rt.registry.dongles)
    for line in render_coverage_report(coverage, all_bands_flag_used=all_bands):
        click.echo(line)

    # Guided mode: build hooks BEFORE constructing SessionRunner so we
    # can pass them in. If no telescopic dongle exists, warn but
    # proceed — prompts will still tell the user what to do.
    before_task_hook = None
    after_session_hook = None
    if guided:
        from rfcensus.commands.guided import (
            build_guided_config, find_telescopic_dongle,
            make_after_session_callback, make_before_band_callback,
            show_plan,
        )
        # Build the antennas-by-id map from config (so we have the
        # original quarter-wave length for end-of-scan reset)
        antennas_by_id = {a.id: a.model_dump() for a in rt.config.antennas}
        chosen_dongle, chosen_ant, msg = find_telescopic_dongle(
            rt.registry.dongles, antennas_by_id,
        )
        if msg:
            click.echo(msg, err=True)
        if chosen_dongle is None:
            # No telescopic — fall back to first usable dongle and
            # synthesize a minimal antenna_dict. Prompts will still
            # tell the user what cm to set.
            usable = rt.registry.usable()
            if not usable:
                raise click.BadParameter(
                    "--guided needs at least one usable dongle"
                )
            chosen_dongle = usable[0]
            chosen_ant = (
                antennas_by_id.get(chosen_dongle.antenna.id)
                if chosen_dongle.antenna else None
            )
        guided_cfg = build_guided_config(
            chosen_dongle, chosen_ant, rt.config.enabled_bands(),
        )
        show_plan(guided_cfg)
        if not click.confirm(
            "  Proceed with guided scan?", default=True,
        ):
            click.echo("  Cancelled.")
            return
        before_task_hook = make_before_band_callback(guided_cfg)
        after_session_hook = make_after_session_callback(
            guided_cfg,
            config_path or rt.config.source_path or Path("~/.config/rfcensus/site.toml").expanduser(),
        )

    runner = SessionRunner(
        command=command_name,
        config=rt.config,
        registry=rt.registry,
        db=rt.db,
        duration_s=duration_s,
        capture_power=capture_power,
        all_bands=all_bands,
        per_band_s=per_band_s,
        indefinite=indefinite,
        gain=gain_normalized,
        until_quiet_s=until_quiet_s,
        before_task_hook=before_task_hook,
        after_session_hook=after_session_hook,
        # Inventory already ran check_all() above (so we could
        # prompt about busy dongles before committing to the scan),
        # so skip the duplicate check inside the runner.
        skip_health_check=True,
    )

    # Simple progress line: emit a one-liner per new emitter and per confirmed one.
    def _on_emitter(event: EmitterEvent) -> None:
        if event.kind == "new":
            click.echo(
                f"[new]       {event.protocol:<16s} "
                f"{event.device_id_hash:<12s}  {event.typical_freq_hz/1e6:.3f} MHz"
            )
        elif event.kind == "confirmed":
            click.echo(
                f"[confirmed] {event.protocol:<16s} "
                f"{event.device_id_hash:<12s}  conf={event.confidence:.2f}"
            )

    runner.event_bus.subscribe(EmitterEvent, _on_emitter)

    decode_counter = {"n": 0}

    def _on_decode(event: DecodeEvent) -> None:
        decode_counter["n"] += 1
        if decode_counter["n"] % 50 == 0:
            click.echo(f"  ...{decode_counter['n']} decodes heard so far", err=True)

    runner.event_bus.subscribe(DecodeEvent, _on_decode)

    if indefinite:
        click.echo(
            f"Running {command_name} indefinitely with {len(rt.registry.usable())} "
            f"dongle(s) (Ctrl-C to exit cleanly, Ctrl-C twice to force)..."
        )
    else:
        click.echo(
            f"Running {command_name} for {duration_s:.0f}s with "
            f"{len(rt.registry.usable())} dongle(s)..."
        )
    result = await runner.run()

    builder = ReportBuilder(rt.db)
    report = await builder.render(
        result,
        fmt="json" if as_json else "text",
        include_ids=include_ids,
        site_name=rt.config.site.name,
    )

    if output:
        output.write_text(report, encoding="utf-8")
        click.echo(f"\nReport written to {output}")
    else:
        click.echo("\n" + report)


cli_inventory = _build_inventory_cli(
    default_duration="1h",
    command_name="inventory",
    help_text=(
        "Run a thorough multi-pass site survey. Loops through every band "
        "many times to catch transient signals (key fobs, garage doors, "
        "occasional sensor reports). For a quick read instead, use `scan`."
    ),
    default_per_band="1m",  # Round-robin: 1 minute per band per pass
)
cli_scan = _build_inventory_cli(
    default_duration="5m",
    command_name="scan",
    help_text=(
        "Quick single-pass scan of enabled bands. Each band gets one "
        "longer dwell window. For a thorough multi-pass survey, use "
        "`inventory` instead."
    ),
    default_per_band=None,  # Single pass; duration divided evenly
)
