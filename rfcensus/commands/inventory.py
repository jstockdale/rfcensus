"""`rfcensus inventory` and `rfcensus scan` — run surveys."""

from __future__ import annotations

import asyncio
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


def _synthesize_partial_result(runner) -> "Optional[SessionResult]":
    """v0.7.6: build a partial SessionResult from a force-cancelled
    runner so the report builder still has something to render.

    The runner stashes _current_session_id and _current_plan during
    run() (introduced in v0.7.5 for the TUI snapshot report). When
    force-quit cancels the runner mid-wave, those stashes survive on
    the runner object even though run() never returned a SessionResult.
    We assemble one here from the stashed pieces + the strategy
    results that did complete.

    Returns None when the cancellation happened before plan build
    (i.e. nothing to report). The caller handles None gracefully.
    """
    from datetime import datetime, timezone
    from rfcensus.engine.session import SessionResult

    sid = getattr(runner, "_current_session_id", None)
    plan = getattr(runner, "_current_plan", None)
    if sid is None or plan is None:
        return None
    strategy_results = list(getattr(runner, "_strategy_results", []))
    started = datetime.now(timezone.utc)    # placeholder; ended is now
    return SessionResult(
        session_id=sid,
        started_at=started,
        ended_at=datetime.now(timezone.utc),
        plan=plan,
        strategy_results=strategy_results,
        total_decodes=sum(
            r.decodes_emitted for r in strategy_results
        ),
        warnings=list(plan.warnings),
        # v0.7.4 INCOMPLETE banner triggers on this flag
        stopped_early=True,
    )


def _build_inventory_cli(
    default_duration: str,
    command_name: str,
    help_text: str,
    *,
    default_per_band: str | None = None,
    honor_pins: bool = False,
):
    """Factory for the inventory/scan/hybrid command trio.

    They share most flags + flow; the differences are:
      • default_duration — "5m" for scan, "forever" for inventory & hybrid
      • default_per_band — "1m" for inventory/hybrid, None for scan
      • honor_pins — True ONLY for hybrid. When False:
          - --pin and --allow-pin-antenna-mismatch errors-out as a
            "use rfcensus hybrid" redirect (the flags still appear in
            --help so users discover the feature)
          - Pins declared in site.toml print a one-line warning at
            startup ("ignored — use 'rfcensus hybrid'") then are dropped
        When True:
          - Both flags work normally
          - Config-declared pins are honored
    """
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
    @click.option(
        "--pin",
        "pin_strings",
        multiple=True,
        metavar="DONGLE:DECODER@FREQ[:SR]",
        help=(
            "Pin a decoder to a dongle for the entire session. "
            f"{'Honored by `hybrid` only.' if not honor_pins else ''}"
            " The dongle is leased exclusively at startup and runs "
            "the named decoder at the named frequency until the "
            "session ends. Repeat --pin for multiple. Format examples: "
            "'00000043:rtl_433@433.92M' or "
            "'00000043:rtl_433@433920000:2400000'. "
            f"{'Use `rfcensus hybrid` to pin dongles; `inventory` and `scan` ignore pins on purpose.' if not honor_pins else 'CLI pins override config-declared pins on the same dongle.'}"
        ),
    )
    @click.option(
        "--allow-pin-antenna-mismatch",
        is_flag=True,
        help=(
            "Allow pinning a decoder to a dongle whose antenna "
            "doesn't cover the pin's frequency. "
            f"{'Only meaningful with `rfcensus hybrid`.' if not honor_pins else 'Default is to refuse such pins as almost-certainly-typos.'}"
        ),
    )
    @click.option(
        "--tui",
        is_flag=True,
        help=(
            "Launch the interactive dashboard while the scan runs. "
            "Requires a TTY at least 80×24. Falls back to log mode "
            "with a one-line notice if the terminal is too small "
            "or stdout is piped."
        ),
    )
    @click.option(
        "--no-color",
        is_flag=True,
        help=(
            "Disable color output in the dashboard and log. Honors "
            "the NO_COLOR env var as well."
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
        pin_strings: tuple[str, ...],
        allow_pin_antenna_mismatch: bool,
        tui: bool,
        no_color: bool,
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
                pin_strings=list(pin_strings),
                allow_pin_antenna_mismatch=allow_pin_antenna_mismatch,
                honor_pins=honor_pins,
                tui=tui,
                no_color=no_color,
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
    pin_strings: list[str] | None = None,
    allow_pin_antenna_mismatch: bool = False,
    honor_pins: bool = False,
    tui: bool = False,
    no_color: bool = False,
) -> None:
    # v0.6.13: emit a startup banner with the version + mode (tui vs
    # log) as the very first thing _run does — before any orphan
    # cleanup or hardware detection. Two reasons:
    #   1. When users send back a log it's immediately clear which
    #      build produced it (we've shipped 8 versions in 2 weeks; the
    #      first question on every bug report is "what version?").
    #   2. The TUI takes 2-3s to come up after launch. Logging "tui
    #      mode" up front confirms the user picked the right
    #      invocation while they wait for the screen to render.
    from rfcensus import __version__ as _rfcensus_version
    log.info(
        "rfcensus %s — launching %s in %s mode",
        _rfcensus_version, command_name, "tui" if tui else "log",
    )

    # Hard error if user passed --pin to a command that doesn't honor
    # it. Footgun: silently ignoring --pin would mean the user thinks
    # they pinned a dongle and finds out later they didn't. Better to
    # fail loudly with a redirect.
    if pin_strings and not honor_pins:
        raise click.BadParameter(
            f"--pin requires the `hybrid` subcommand. "
            f"`{command_name}` ignores pins on purpose so accidental "
            f"persistent config doesn't break your scan. Try: "
            f"`rfcensus hybrid --pin {pin_strings[0]}`"
        )
    if allow_pin_antenna_mismatch and not honor_pins:
        raise click.BadParameter(
            f"--allow-pin-antenna-mismatch only applies when pins are "
            f"in use. Use it with `rfcensus hybrid`."
        )
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

    # v0.6.0: build the pin spec list (only if this command honors pins).
    # When not honored: warn at startup so users notice they have pins
    # in config that aren't doing anything in this command, then drop them.
    pin_specs = []
    config_pins = [d for d in rt.config.dongles if d.pin is not None]
    if honor_pins:
        if pin_strings or config_pins:
            from rfcensus.engine.pinning import gather_pins
            try:
                pin_specs = gather_pins(rt.config, pin_strings)
            except ValueError as exc:
                raise click.BadParameter(f"--pin: {exc}") from None
            if pin_specs:
                click.echo(
                    f"Pinning {len(pin_specs)} dongle(s) for this session:",
                    err=True,
                )
                for spec in pin_specs:
                    src = (
                        "(config)"
                        if spec.source == "config"
                        else "(--pin)"
                    )
                    click.echo(
                        f"  • {spec.dongle_id} → {spec.decoder} @ "
                        f"{spec.freq_hz / 1e6:.3f} MHz {src}",
                        err=True,
                    )
    else:
        if config_pins:
            click.echo(
                f"Note: {len(config_pins)} pin(s) in config will be "
                f"ignored — `{command_name}` doesn't honor pins. "
                f"Use `rfcensus hybrid` to honor them.",
                err=True,
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
        pin_specs=pin_specs,
        allow_pin_antenna_mismatch=allow_pin_antenna_mismatch,
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
                f"{event.device_id_hash:<12s}  "
                f"{event.typical_freq_hz/1e6:.3f} MHz  "
                f"conf={event.confidence:.2f}"
            )

    runner.event_bus.subscribe(EmitterEvent, _on_emitter)

    decode_counter = {"n": 0}

    def _on_decode(event: DecodeEvent) -> None:
        decode_counter["n"] += 1
        if decode_counter["n"] % 50 == 0:
            click.echo(f"  ...{decode_counter['n']} decodes heard so far", err=True)

    runner.event_bus.subscribe(DecodeEvent, _on_decode)

    if indefinite:
        # Educational startup banner. The point of indefinite is to
        # leave it running and check back later. The banner tells the
        # user (in the first second on screen) how to opt back into a
        # finite run if that's what they actually wanted, and points
        # at --until-quiet which is almost always what unattended
        # users actually want ("run until nothing new for X, then exit").
        click.echo(
            f"Running {command_name} indefinitely with "
            f"{len(rt.registry.usable())} dongle(s). "
            f"Ctrl-C for clean shutdown + report (twice to force quit).",
            err=True,
        )
        click.echo(
            "  Tip: --duration 1h for a finite run, or "
            "--until-quiet 30m to exit automatically when the site "
            "goes quiet.",
            err=True,
        )
    else:
        click.echo(
            f"Running {command_name} for {duration_s:.0f}s with "
            f"{len(rt.registry.usable())} dongle(s)..."
        )

    # v0.6.5: --tui launches the dashboard alongside runner.run().
    # Falls back to log mode if the terminal is too small or stdout
    # is piped. Color is disabled if --no-color or NO_COLOR is set.
    if tui:
        from rfcensus.tui.app import TUIApp, check_tty_and_size
        ok, msg = check_tty_and_size()
        if not ok:
            click.echo(
                f"--tui unavailable ({msg}); falling back to log mode.",
                err=True,
            )
        else:
            from rfcensus.tui.color import detect_color_support
            color_enabled = detect_color_support() and not no_color
            tui_app = TUIApp(
                runner=runner,
                no_color=not color_enabled,
                site_name=rt.config.site.name,
            )

            # Run runner + TUI concurrently. Whichever finishes first,
            # cancel the other. Runner finishing means scan is done;
            # TUI finishing means user hit q (which calls
            # runner.request_stop) or l (log-mode toggle).
            #
            # v0.7.6: tell the runner the TUI owns this terminal so
            # it skips installing its own SIGINT handler. Textual
            # binds Ctrl+C as a key event and routes it through the
            # TUI's quit modal — having two handlers fire on the
            # same keypress was the cause of the "I cancelled the
            # quit but the wave still wrapped up" bug.
            runner.tui_active = True
            runner_task = asyncio.create_task(
                runner.run(), name="rfcensus-runner",
            )
            tui_task = asyncio.create_task(
                tui_app.run_async(), name="rfcensus-tui",
            )

            done, pending = await asyncio.wait(
                {runner_task, tui_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            # v0.6.17 / v0.7.4 bug fix: only cancel the OTHER side
            # when the user actually requested termination AND wants
            # an immediate force-cancel. Three cases now:
            #
            # 1. log_mode_toggle (user pressed `l`): TUI exited, but
            #    runner should KEEP RUNNING in foreground. Don't cancel.
            #
            # 2. graceful_quit (user pressed q+y): TUI exited with
            #    _stop_requested set. Don't cancel — let the runner
            #    finish its current wave naturally so leases release
            #    cleanly and subprocess decoders flush. Without this,
            #    we orphaned rtl_433 processes mid-wave AND
            #    runner_task.result() raised CancelledError, swallowing
            #    the report.
            #
            # 3. force_quit (user pressed q+f): cancel the runner now.
            #    User accepted that the report will be incomplete.
            #
            # 4. fast_quit (user pressed Ctrl+Q): cancel runner, skip
            #    the report render entirely. Panic button — user
            #    knows they want OUT and doesn't care about the
            #    partial report. The DB still has the data; user can
            #    recover via `rfcensus list decodes --session N`.
            fast_quit = (
                tui_task in done
                and getattr(tui_app, "_fast_quit_requested", False)
            )
            graceful_quit = (
                tui_task in done
                and not fast_quit
                and getattr(tui_app, "_force_quit_requested", False) is False
                and (
                    runner.control.stopped.is_set()
                    or getattr(runner, "_stop_requested", False)
                )
            )
            log_mode_toggle = (
                tui_task in done
                and getattr(tui_app, "_log_mode_requested", False)
                and not runner.control.stopped.is_set()
                and not getattr(runner, "_stop_requested", False)
            )
            for t in pending:
                if t is runner_task and (log_mode_toggle or graceful_quit):
                    continue  # leave the runner running / shutting down
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

            # v0.7.6: fast-quit short-circuit — print one line and
            # bail. No report render, no DB query, no JSON. The
            # user already knows what they did.
            if fast_quit:
                sid = getattr(runner, "_current_session_id", None)
                sid_hint = (
                    f"\n  Recover partial data with: rfcensus list "
                    f"decodes --session {sid}"
                    if sid is not None else ""
                )
                click.echo(
                    f"\n[fast-quit (Ctrl+Q) — runner cancelled, "
                    f"report skipped]{sid_hint}\n",
                    err=True,
                )
                click.echo("73 🛰️")
                return

            # Get the runner result. If runner crashed, propagate.
            if runner_task in done:
                result = runner_task.result()
            else:
                # TUI exited first (q or l). If it was l OR graceful q,
                # we still need to wait for runner to finish naturally.
                if log_mode_toggle:
                    # v0.6.14: log-mode toggle — re-attach the console
                    # log handler before awaiting the runner.
                    #
                    # Why this is needed: Textual takes over the
                    # terminal during TUI mode and any RichHandler
                    # writing to the original sys.stderr ends up
                    # writing into the live screen buffer (or a torn-
                    # down handle, depending on Textual version).
                    # Either way, the handler that was attached at
                    # CLI entry no longer produces visible output
                    # after the TUI exits. Reconfiguring logging here
                    # rebuilds the handler against the now-restored
                    # stderr, so the user sees real log output for
                    # the rest of the run.
                    #
                    # Going BACK to TUI from log mode would require
                    # tearing down the handler again and re-running
                    # tui_app.run_async(); that's a v0.7 feature.
                    # For now we tell the user how to get back.
                    from rfcensus.utils.logging import configure_logging
                    from rfcensus.commands.base import log_path
                    configure_logging(
                        level="INFO",
                        logfile=log_path(),
                        quiet=False,
                    )
                    click.echo(
                        "\n[log mode — TUI closed; scan continues "
                        "in foreground. Ctrl-C to stop, or quit and "
                        "relaunch with --tui to return to the dashboard]\n",
                        err=True,
                    )
                    result = await runner_task
                elif graceful_quit:
                    # v0.7.4: TUI closed with a graceful-quit request.
                    # The runner is shutting down on its own — we just
                    # need to wait for it. Re-attach the console log
                    # handler so the user sees what's happening (could
                    # be 30-60s of "finishing band X" messages).
                    from rfcensus.utils.logging import configure_logging
                    from rfcensus.commands.base import log_path
                    configure_logging(
                        level="INFO",
                        logfile=log_path(),
                        quiet=False,
                    )
                    click.echo(
                        "\n[graceful shutdown requested — waiting for "
                        "current wave to finish so leases release "
                        "cleanly. Ctrl-C to force immediate exit.]\n",
                        err=True,
                    )
                    try:
                        result = await runner_task
                    except asyncio.CancelledError:
                        # User Ctrl-C'd while waiting — fall through
                        # with whatever the runner managed to capture
                        # before being cancelled.
                        click.echo(
                            "\n[force-quit during graceful shutdown; "
                            "report below reflects partial data]\n",
                            err=True,
                        )
                        result = runner_task.result() if not runner_task.cancelled() else None
                else:
                    # Force quit confirmed (q+f): runner was cancelled
                    # in the loop above; await to surface result/exc.
                    try:
                        result = await runner_task
                    except asyncio.CancelledError:
                        # v0.7.6: synthesize a partial SessionResult
                        # from the runner's stashed plan + session id
                        # + strategy results so the report still
                        # renders. Previously result=None hit
                        # ReportBuilder.render(None, ...) →
                        # AttributeError, which the user experienced
                        # as "force quit does nothing" because the
                        # exception fired during Textual teardown
                        # and was swallowed. Now we render whatever
                        # made it to the DB with the INCOMPLETE
                        # banner the v0.7.4 SessionResult gained.
                        result = _synthesize_partial_result(runner)

            # v0.7.6: result may legitimately be None if the partial
            # synthesis itself failed (no session_id ever assigned —
            # runner cancelled before run() got past plan build).
            # In that case there's nothing to report; tell the user
            # honestly rather than crashing the report builder.
            if result is None:
                click.echo(
                    "\n[no report — session was cancelled before any "
                    "tasks executed]\n",
                    err=True,
                )
                click.echo("73 🛰️")
                return

            builder = ReportBuilder(rt.db)
            try:
                report = await builder.render(
                    result,
                    fmt="json" if as_json else "text",
                    include_ids=include_ids,
                    site_name=rt.config.site.name,
                    command_name=command_name,
                )
            except Exception as exc:
                # v0.7.6: defensive — partial-result rendering can
                # hit weird states (DB closed mid-write, session row
                # not flushed yet). Surface it instead of dying.
                click.echo(
                    f"\n[report render failed: {exc}]\n"
                    f"[partial data is in the DB; try "
                    f"`rfcensus list decodes --session "
                    f"{getattr(result, 'session_id', '?')}`]\n",
                    err=True,
                )
                click.echo("73 🛰️")
                return

            if output:
                output.write_text(report, encoding="utf-8")
                click.echo(f"\nReport written to {output}")
            else:
                click.echo("\n" + report)
            click.echo("\n73 🛰️")
            return

    # Non-TUI path (original behavior)
    result = await runner.run()

    builder = ReportBuilder(rt.db)
    report = await builder.render(
        result,
        fmt="json" if as_json else "text",
        include_ids=include_ids,
        site_name=rt.config.site.name,
        command_name=command_name,
    )

    if output:
        output.write_text(report, encoding="utf-8")
        click.echo(f"\nReport written to {output}")
    else:
        click.echo("\n" + report)


cli_inventory = _build_inventory_cli(
    default_duration="forever",
    command_name="inventory",
    help_text=(
        "Run an exhaustive multi-pass site survey. Defaults to running "
        "until you Ctrl-C — intermittent emitters (paging once an hour, "
        "security sensors that only chirp on events) need long runs to "
        "catch. Use `scan` for a quick read, or `hybrid` if you want to "
        "dedicate some dongles to specific decoders while the rest "
        "explore."
    ),
    default_per_band="1m",  # Round-robin: 1 minute per band per pass
    honor_pins=False,
)
cli_scan = _build_inventory_cli(
    default_duration="5m",
    command_name="scan",
    help_text=(
        "Quick single-pass scan of enabled bands. Each band gets one "
        "longer dwell window. Use this to figure out what's at a site; "
        "follow up with `inventory` for thorough enumeration or "
        "`hybrid` for pinned-decoder + scheduler runs."
    ),
    default_per_band=None,  # Single pass; duration divided evenly
    honor_pins=False,
)
cli_hybrid = _build_inventory_cli(
    default_duration="forever",
    command_name="hybrid",
    help_text=(
        "Inventory + dedicated decoder pinning. Honors `[dongles.pin]` "
        "from your site.toml plus any --pin CLI flags. Pinned dongles "
        "run their decoder for the full session; the rest do the "
        "normal exploration scan in parallel. Defaults to running "
        "until you Ctrl-C — pinning's main use case is gap-free "
        "long-running coverage of specific targets."
    ),
    default_per_band="1m",  # Round-robin for the unpinned dongles
    honor_pins=True,
)
