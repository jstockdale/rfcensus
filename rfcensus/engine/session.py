"""Session runner: end-to-end orchestration of an inventory or scan run."""

from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone

from rfcensus.analysis import DecodeValidator, EmitterTracker
from rfcensus.config.schema import SiteConfig
from rfcensus.decoders.registry import get_registry
from rfcensus.detectors.registry import get_registry as get_detector_registry
from rfcensus.engine.dispatcher import Dispatcher
from rfcensus.engine.scheduler import ExecutionPlan, Scheduler, ScheduleTask
from rfcensus.engine.strategy import StrategyContext, StrategyResult
from rfcensus.events import DecodeEvent, EventBus, SessionEvent
from rfcensus.hardware.broker import DongleBroker, NoDongleAvailable
from rfcensus.hardware.health import check_all
from rfcensus.hardware.registry import HardwareRegistry
from rfcensus.storage import (
    ActiveChannelRepo,
    AnomalyRepo,
    DecodeRepo,
    DetectionRepo,
    DongleRepo,
    EmitterRepo,
    PowerSampleRepo,
    SessionRepo,
    attach_writers,
)
from rfcensus.storage.db import Database
from rfcensus.storage.models import DongleRecord, SessionRecord
from rfcensus.storage.retention import prune
from rfcensus.utils.hashing import generate_salt
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


def _compute_per_wave_duration(total_s: float, n_active_waves: int) -> float:
    """Divide a total session duration evenly across active waves.

    Unassigned waves consume ~0 time so we don't count them. If there are
    no active waves, return the original total (caller will skip everything
    anyway, but we don't want a divide-by-zero).

    The minimum per-wave duration is 1 second — below that, decoders won't
    even spawn cleanly. If the user asks for something pathological like
    --duration 2s with 6 waves, we honor the floor and the total wall
    time may slightly exceed --duration.
    """
    if n_active_waves <= 0:
        return total_s
    per_wave = total_s / n_active_waves
    return max(1.0, per_wave)


@dataclass
class SessionResult:
    session_id: int
    started_at: datetime
    ended_at: datetime
    plan: ExecutionPlan
    strategy_results: list[StrategyResult] = field(default_factory=list)
    total_decodes: int = 0
    warnings: list[str] = field(default_factory=list)
    # v0.7.4: True when the session exited because of a user stop
    # request (Ctrl-C, TUI q+y, TUI q+f) rather than running to its
    # planned completion. The report renderer marks the output as
    # INCOMPLETE when this is set so the user knows that absent
    # detections / undecided bands are due to early termination
    # rather than confirmed silence.
    stopped_early: bool = False
    # v0.7.4: number of planned tasks that didn't get to execute
    # (i.e. were in waves the loop skipped after stop_requested).
    # Computed by the runner before returning the result.
    tasks_skipped_due_to_stop: int = 0


@dataclass
class _PendingRetry:
    """A band that didn't get its full window because its dongle died.

    Re-spawned as a sidecar task when any dongle (preferably the same
    one, but any compatible dongle works) reconnects.
    """

    band: object  # BandConfig — using object to avoid circular import
    band_id: str
    dongle_id_when_failed: str
    remaining_s: float


class SessionRunner:
    def __init__(
        self,
        command: str,
        config: SiteConfig,
        registry: HardwareRegistry,
        db: Database,
        duration_s: float = 1800.0,
        capture_power: bool = False,
        all_bands: bool = False,
        per_band_s: float | None = None,
        indefinite: bool = False,
        gain: str = "auto",
        until_quiet_s: float | None = None,
        max_dongle_failures: int = 3,
        before_task_hook: callable | None = None,
        after_session_hook: callable | None = None,
        skip_health_check: bool = False,
        # v0.6.0: decoder pinning. If pin_specs is non-empty, those
        # dongles are leased exclusively at session bootstrap and a
        # supervised decoder loop runs on each for the session lifetime.
        # The scheduler never sees them. See engine/pinning.py.
        pin_specs: list | None = None,
        allow_pin_antenna_mismatch: bool = False,
    ):
        # Hooks for interactive modes like --guided. before_task_hook
        # is called before each task starts; if it returns "skip" the
        # task is skipped. after_session_hook runs after the wave loop
        # finishes (before final cleanup) for end-of-scan UX like
        # restoring antenna state.
        self.before_task_hook = before_task_hook
        self.after_session_hook = after_session_hook
        # When the caller has already run check_all() (e.g. the inventory
        # command does this early so it can prompt the user about busy
        # dongles before committing to a scan), set this to skip the
        # duplicate check. The existing dongle statuses are trusted.
        self.skip_health_check = skip_health_check
        self.command = command
        self.config = config
        self.registry = registry
        self.db = db
        self.duration_s = duration_s
        self.capture_power = capture_power
        self.all_bands = all_bands
        self.per_band_s = per_band_s  # None = compute from duration / wave_count
        self.indefinite = indefinite
        self.gain = gain
        self.until_quiet_s = until_quiet_s  # None = disabled
        self.max_dongle_failures = max_dongle_failures
        self.event_bus = EventBus()
        self.broker = DongleBroker(registry, self.event_bus)
        # v0.6.5: pause/resume coordination. Created here so the TUI
        # (or any other controller) can grab a reference via
        # `runner.control` and toggle pause without going through the
        # session loop. The wave loop and fanouts both consult this
        # object — see engine/session_control.py for full lifecycle.
        from rfcensus.engine.session_control import SessionControl
        self.control = SessionControl()
        # Set by the SIGINT handler installed during run(). Checked between
        # waves and between passes for graceful shutdown.
        self._stop_requested = False
        self._sigint_count = 0
        # Pending retries: bands that didn't get their full window because
        # their dongle died mid-run. Re-spawned as sidecar tasks when the
        # dongle reconnects.
        self._pending_retries: list[_PendingRetry] = []
        self._sidecar_tasks: set[asyncio.Task] = set()
        # v0.6.5: deep-pause watcher task. Tracked separately from
        # _sidecar_tasks because it's an infinite loop — the sidecar
        # cleanup gather would block forever waiting for it. Cancelled
        # explicitly during session teardown.
        self._deep_pause_task: asyncio.Task | None = None
        # Per-dongle failure counter — if >= max_dongle_failures, the
        # dongle is added to _permanently_failed and re-probe will not
        # restore it for the rest of the session.
        self._dongle_failure_counts: dict[str, int] = {}
        self._permanently_failed: set[str] = set()
        # For --until-quiet: track when we last saw a NEW emitter
        self._last_new_emitter_at: float = 0.0
        # Strategy results need to be accumulated across waves AND sidecar
        # retries. Promoted from local var to instance attr so retries
        # can append.
        self._strategy_results: list[StrategyResult] = []
        # v0.7.5: stashed during run() so the TUI snapshot report can
        # render against the real DB via ReportBuilder, mirroring the
        # final-report quality. None before run() starts.
        self._current_plan: ExecutionPlan | None = None
        self._current_session_id: int | None = None
        # v0.7.6: set by inventory.py before runner.run() when the
        # TUI is the active controller. The runner uses it to skip
        # installing its own SIGINT handler — Textual captures
        # Ctrl+C as a key event and routes it to the TUI's
        # action_quit_session for the ConfirmQuit modal. Two
        # handlers competing for Ctrl+C produced confusing behavior
        # (silent stop + visible modal both firing).
        self.tui_active: bool = False
        # Set during run() so sidecar retries can spawn strategies
        self._dispatcher = None
        self._ctx = None
        # Map band id → BandConfig for quick lookup from event handlers
        self._band_by_id: dict[str, object] = {}

        # v0.6.0: pinning state. List of PinSpec objects (or empty),
        # parsed from config + CLI by the calling command. The actual
        # PinningOutcome (with live supervisor tasks) is built during
        # run() and stashed on self._pinning_outcome for teardown.
        self._pin_specs = list(pin_specs or [])
        self._allow_pin_antenna_mismatch = allow_pin_antenna_mismatch
        self._pinning_outcome = None  # set in run() after start_pinned_tasks

    def _handle_sigint(self) -> None:
        """SIGINT handler installed during run().

        First Ctrl-C: log a friendly message and set stop_requested. The
        wave loop checks this flag between bands and exits cleanly,
        persisting all data collected so far.

        Second Ctrl-C: restore the default handler (so a third one would
        kill us hard) and raise KeyboardInterrupt via the loop, which
        cancels in-flight tasks and accelerates exit.
        """
        self._sigint_count += 1
        if self._sigint_count == 1:
            log.warning(
                "received SIGINT — finishing current band, then exiting cleanly. "
                "Press Ctrl-C again to force immediate exit."
            )
            self._stop_requested = True
        else:
            log.warning("received second SIGINT — forcing immediate exit")
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            # Cancel everything in the current task chain
            for task in asyncio.all_tasks():
                task.cancel()

    def _record_failure(self, evt) -> None:
        """Subscriber for DecoderFailureEvent. Tracks the failure count
        per dongle and queues a retry if the dongle hasn't permanently
        failed yet."""
        self._dongle_failure_counts[evt.dongle_id] = (
            self._dongle_failure_counts.get(evt.dongle_id, 0) + 1
        )
        n = self._dongle_failure_counts[evt.dongle_id]

        if n >= self.max_dongle_failures:
            if evt.dongle_id not in self._permanently_failed:
                self._permanently_failed.add(evt.dongle_id)
                log.warning(
                    "⚠ dongle %s has failed %d times; marking as permanently "
                    "failed for the rest of this session. Replace the dongle "
                    "or restart rfcensus to retry.",
                    evt.dongle_id, n,
                )
            return  # Don't queue a retry

        # Queue retry. Look up the band by id from our cached map.
        band = self._band_by_id.get(evt.band_id)
        if band is None:
            log.debug("no band mapping for %s; cannot queue retry", evt.band_id)
            return
        # Don't double-queue: if a retry for this band is already pending
        # (perhaps from a previous wave), skip
        if any(p.band_id == evt.band_id for p in self._pending_retries):
            return
        self._pending_retries.append(_PendingRetry(
            band=band,
            band_id=evt.band_id,
            dongle_id_when_failed=evt.dongle_id,
            remaining_s=evt.remaining_s,
        ))
        log.info(
            "queued retry for %s (will run when a dongle reconnects, "
            "%.0fs remaining time budget)",
            evt.band_id, evt.remaining_s,
        )

    def _record_success(self, dongle_id: str) -> None:
        """Reset a dongle's consecutive-failure counter after a successful
        decoder run on it. Called after each wave's results are tallied."""
        if dongle_id in self._dongle_failure_counts:
            del self._dongle_failure_counts[dongle_id]

    async def _spawn_pending_retries(self) -> None:
        """Spawn sidecar tasks for any pending retries that can now run.

        Called after re-probe restores any dongles. Walks the pending
        list; for each retry whose dongle (or any compatible dongle) is
        now usable, spawn a strategy execution as a background task.
        """
        if not self._pending_retries or self._dispatcher is None:
            return
        usable_ids = {d.id for d in self.registry.usable()}
        if not usable_ids:
            return  # Nothing to spawn onto

        # Take retries we can run now; leave others queued
        runnable = []
        still_pending = []
        for retry in self._pending_retries:
            # The original dongle preferred, but any usable dongle that
            # covers the band is acceptable — broker.allocate will pick
            # the best match.
            if any(
                d.covers(retry.band.center_hz) for d in self.registry.usable()
            ):
                runnable.append(retry)
            else:
                still_pending.append(retry)
        self._pending_retries = still_pending

        for retry in runnable:
            log.info(
                "spawning retry of %s (%.0fs remaining)",
                retry.band_id, retry.remaining_s,
            )
            task = asyncio.create_task(
                self._run_retry(retry),
                name=f"retry-{retry.band_id}",
            )
            self._sidecar_tasks.add(task)
            task.add_done_callback(self._sidecar_tasks.discard)

    async def _run_retry(self, retry: _PendingRetry) -> None:
        """Execute a queued retry. Uses a per-call StrategyContext with
        a reduced duration so the retry doesn't run longer than the time
        the band would have gotten in the original wave."""
        if self._dispatcher is None or self._ctx is None:
            return
        try:
            strategy = self._dispatcher.strategy_for(retry.band)
            # Use a shallow-copied context with the retry's duration
            from dataclasses import replace
            retry_ctx = replace(self._ctx, duration_s=retry.remaining_s)
            result = await strategy.execute(retry.band, retry_ctx)
            self._strategy_results.append(result)
            log.info(
                "retry of %s completed: %d decodes",
                retry.band_id, result.decodes_emitted,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("retry of %s failed: %s", retry.band_id, exc)

    def _handle_emitter(self, evt) -> None:
        """Track when we last saw a NEW emitter (for --until-quiet)."""
        if evt.kind == "new":
            self._last_new_emitter_at = asyncio.get_event_loop().time()

    def _is_quiet(self) -> bool:
        """True if --until-quiet is enabled and the quiet window has
        elapsed since the last new emitter."""
        if self.until_quiet_s is None or self.until_quiet_s <= 0:
            return False
        if self._last_new_emitter_at == 0.0:
            return False  # Haven't seen any emitter yet; don't exit
        elapsed = asyncio.get_event_loop().time() - self._last_new_emitter_at
        return elapsed >= self.until_quiet_s

    # ------------------------------------------------------------
    # v0.6.6: confirmation queue removed. The legacy LoRa detector
    # was the only producer; LoraSurveyTask does its own IQ analysis
    # inline via survey_iq_window, so deferred confirmation is no
    # longer needed. If a future detector wants the same pattern,
    # reintroduce a fresh queue scoped to that detector's needs.
    # ------------------------------------------------------------


    def _log_hardware_summary(self) -> None:
        """Print a one-line-per-dongle health summary after the health
        check completes. Makes it obvious when a dongle is BUSY or
        FAILED — previously the user would only notice via unassigned
        bands later in the plan log, which was easy to miss.
        """
        from rfcensus.hardware.dongle import DongleStatus

        status_markers = {
            DongleStatus.HEALTHY: "✓",
            DongleStatus.DETECTED: "✓",
            DongleStatus.DEGRADED: "~",
            DongleStatus.BUSY: "✗",
            DongleStatus.FAILED: "✗",
            DongleStatus.UNAVAILABLE: "✗",
        }
        healthy = sum(
            1 for d in self.registry.dongles
            if d.status in (DongleStatus.HEALTHY, DongleStatus.DETECTED)
        )
        total = len(self.registry.dongles)

        # Header line with the aggregate
        if healthy == total:
            log.info("hardware: %d/%d dongles healthy", healthy, total)
        else:
            unhealthy = [
                d for d in self.registry.dongles
                if d.status not in (DongleStatus.HEALTHY, DongleStatus.DETECTED)
            ]
            kinds = ", ".join(
                f"{d.status.value}" for d in unhealthy
            )
            log.warning(
                "hardware: %d/%d dongles healthy (%d unusable: %s)",
                healthy, total, total - healthy, kinds,
            )

        # Per-dongle line with antenna and status
        for d in self.registry.dongles:
            marker = status_markers.get(d.status, "?")
            antenna_label = d.antenna.id if d.antenna is not None else "no antenna"
            note = ""
            if d.health_notes:
                note = f" — {d.health_notes[-1]}"
            log.info(
                "  %s %s (%s) %s%s",
                marker, d.id, antenna_label, d.status.value, note,
            )

    def _log_unassigned_warning(self, plan) -> None:
        """If any bands couldn't be assigned to a dongle, log a clear
        warning listing them. Without this, unassigned bands only show
        up as `→unassigned` in the middle of the plan log and users
        miss it.
        """
        unassigned = [
            t.band.id for t in plan.tasks
            if t.suggested_dongle_id is None
        ]
        if not unassigned:
            return
        log.warning(
            "%d band(s) could not be scanned (no dongle with suitable "
            "antenna available): %s",
            len(unassigned), ", ".join(unassigned),
        )
        log.warning(
            "  check `rfcensus doctor` for per-dongle status, or run "
            "with --all-bands to force-scan anyway"
        )

    def _log_decoder_binary_preflight(self) -> None:
        """Check all decoder binaries upfront and log which are missing.

        Without this, a missing binary only surfaces when rfcensus
        tries to run that decoder mid-scan, typically producing
        a very confusing log ("rtlamr emitted 0 decodes" with no
        further explanation). Running the check at plan time means
        the operator can install missing binaries before committing
        to a long scan.

        Resolves each decoder's actual `external_binary` name (from
        its DecoderCapabilities), not the decoder's registration
        name. For most decoders they match (rtl_433, rtlamr, etc.),
        but multimon registers as "multimon" while its binary is
        "multimon-ng" — a naming mismatch that was producing false
        positives in the preflight report before v0.5.27.
        """
        from rfcensus.utils.async_subprocess import _binary_on_path
        from rfcensus.decoders.registry import get_registry

        registry = get_registry()
        missing: list[tuple[str, str]] = []  # (decoder_name, binary_name)
        present: list[tuple[str, str]] = []
        for name in registry.names():
            decoder_cls = registry.get(name)
            # Instantiate with no settings (defaults) to read caps.
            # The capabilities object carries external_binary which
            # is authoritative; decoder.name is only for registration.
            try:
                inst = decoder_cls()
                binary_name = (
                    inst.capabilities.external_binary or name
                )
            except Exception:
                # If construction fails for any reason, fall back to
                # decoder name and let the decoder's own run-time
                # error handling report the real issue.
                binary_name = name

            if _binary_on_path(binary_name):
                present.append((name, binary_name))
            else:
                missing.append((name, binary_name))

        if not missing:
            log.info(
                "decoder binaries: all %d decoder(s) found on PATH",
                len(present),
            )
            return

        log.warning(
            "decoder binaries: %d missing on PATH, %d found",
            len(missing), len(present),
        )
        for decoder_name, binary_name in missing:
            if decoder_name == binary_name:
                log.warning(
                    "  ✗ %s — not found (bands assigned to this "
                    "decoder will produce 0 decodes)",
                    decoder_name,
                )
            else:
                log.warning(
                    "  ✗ %s (binary: %s) — not found (bands assigned "
                    "to this decoder will produce 0 decodes)",
                    decoder_name, binary_name,
                )
        log.warning(
            "  to fix: install each binary and ensure its directory "
            "is on PATH. Go-based decoders (rtlamr) install to "
            "~/go/bin by default — add `export PATH=$PATH:$HOME/go/bin` "
            "to your shell rc."
        )

    # ────────────────────────────────────────────────────────────────
    # v0.6.5: external control surface (TUI calls these)
    # ────────────────────────────────────────────────────────────────

    async def pause_session(self) -> None:
        """Pause the active session. Idempotent.

        Effect:
          • Sets self.control.paused so the wave loop blocks before
            the next wave.
          • Pauses every active shared fanout's writes — decoders
            block on socket reads. No process kills.

        On long pauses, the deep_pause_watcher (started in run())
        eventually triggers decoder teardown to release CPU.
        """
        await self.control.pause()
        try:
            self.broker.pause_all_fanouts()
        except Exception:
            log.exception("broker.pause_all_fanouts failed (non-fatal)")

    async def resume_session(self) -> dict[str, bool]:
        """Resume the paused session.

        Returns a per-dongle dict {dongle_id: all_clients_alive} so the
        caller can log per-fanout health post-pause. False entries mean
        one or more decoder clients on that fanout died during pause
        (their TCP socket broke when resume_writes probed it).

        v0.6.5 behavior: dead clients are NOT auto-restarted within
        the current wave. The decoder's strategy task will surface the
        socket-closed error and end its run. The wave still completes;
        subsequent waves re-allocate everything fresh. The
        "always-restart-crashed-decoders" pattern is broader v0.6.6+
        work that handles this and equivalent in-wave failures
        (decoder process OOM, USB unplug, etc.) uniformly.
        """
        results = {}
        try:
            results = self.broker.resume_all_fanouts()
        except Exception:
            log.exception("broker.resume_all_fanouts failed (non-fatal)")
        await self.control.resume()

        # Surface dead-client warnings so the user sees what happened
        # instead of silently losing decoders.
        dead = [d for d, alive in results.items() if not alive]
        if dead:
            log.warning(
                "after resume: %d fanout(s) had dead clients: %s. "
                "Affected decoders will end with errors and be "
                "re-allocated on subsequent waves.",
                len(dead), ", ".join(dead),
            )
        return results

    def request_stop(self) -> None:
        """Mark the session for graceful shutdown.

        Same effect as Ctrl-C: wave loop exits between waves, in-flight
        tasks finish naturally, leases release. Used by the TUI's `q`
        confirm path.
        """
        self._stop_requested = True
        # Also wake any pause-waiters so they exit cleanly instead of
        # blocking forever.
        try:
            asyncio.get_event_loop().create_task(self.control.stop())
        except Exception:
            pass

    async def run(self) -> SessionResult:
        started = datetime.now(timezone.utc)

        salt = self.config.privacy.hash_salt
        if salt == "auto":
            salt = generate_salt()

        try:
            counts = await prune(self.db, self.config.resources)
            if counts:
                log.info("retention: pruned %s", counts)
        except Exception:
            log.exception("retention prune failed (continuing)")

        if self.registry.dongles:
            if self.skip_health_check:
                log.debug(
                    "skipping health check (already done by caller); "
                    "using current dongle statuses"
                )
                self._log_hardware_summary()
            else:
                log.info("health-checking %d dongle(s)", len(self.registry.dongles))
                await check_all(self.registry.dongles)
                self._log_hardware_summary()

        session_repo = SessionRepo(self.db)
        dongle_repo = DongleRepo(self.db)
        decode_repo = DecodeRepo(self.db)
        emitter_repo = EmitterRepo(self.db)
        active_channel_repo = ActiveChannelRepo(self.db)
        anomaly_repo = AnomalyRepo(self.db)
        detection_repo = DetectionRepo(self.db)
        power_sample_repo = PowerSampleRepo(self.db)

        session_id = await session_repo.create(
            SessionRecord(
                id=None,
                command=self.command,
                started_at=started,
                site_name=self.config.site.name,
                config_snap=self.config.model_dump(),
            )
        )
        # Stash session_id on self so any background task that needs
        # to build a DetectionRecord can find the right FK.
        self._current_session_id = session_id

        for dongle in self.registry.usable():
            await dongle_repo.upsert(
                DongleRecord(
                    id=dongle.id,
                    serial=dongle.serial,
                    model=dongle.model,
                    driver=dongle.driver,
                    capabilities=dongle.capabilities.as_dict(),
                    first_seen=started,
                    last_seen=started,
                    notes=None,
                )
            )
            await session_repo.attach_hardware(
                session_id=session_id,
                dongle_id=dongle.id,
                antenna_id=dongle.antenna.id if dongle.antenna else None,
                role="auto",
            )

        power_batcher = attach_writers(
            bus=self.event_bus,
            session_id=session_id,
            active_channel_repo=active_channel_repo,
            power_sample_repo=power_sample_repo,
            anomaly_repo=anomaly_repo,
            detection_repo=detection_repo,
            capture_power=self.capture_power,
        )

        # Attach detectors — they subscribe to ActiveChannelEvent and
        # emit DetectionEvent; the DetectionWriter persists them.
        # Detectors that declare consumes_iq=True get an
        # IQCaptureService for opportunistic IQ inspection.
        from rfcensus.spectrum import IQCaptureService

        iq_service = IQCaptureService(self.broker)
        detector_registry = get_detector_registry()
        detectors = detector_registry.all_instances()
        for detector in detectors:
            detector.attach(
                self.event_bus,
                session_id,
                iq_service=iq_service if detector.capabilities.consumes_iq else None,
            )
        log.info(
            "attached %d detector(s): %s",
            len(detectors),
            ", ".join(d.name for d in detectors),
        )

        registry_handle = get_registry()
        validator = DecodeValidator(
            self.config.validation, decoder_names=registry_handle.names()
        )
        tracker = EmitterTracker(
            event_bus=self.event_bus,
            decode_repo=decode_repo,
            emitter_repo=emitter_repo,
            validator=validator,
            salt=salt,
            min_confirmations=self.config.validation.min_confirmations_for_confirmed,
        )

        async def _tracker_handler(event: DecodeEvent) -> None:
            await tracker.handle_decode(event, session_id)

        self.event_bus.subscribe(DecodeEvent, _tracker_handler)

        await self.event_bus.publish(
            SessionEvent(session_id=session_id, kind="started", phase="init")
        )

        # v0.6.0: start pinned-decoder supervisors BEFORE the scheduler
        # plans, so the broker's exclusive-lease tracking removes pinned
        # dongles from the candidate pool. The scheduler then naturally
        # operates on the remaining (unpinned) dongles.
        if self._pin_specs:
            from rfcensus.engine.pinning import (
                start_pinned_tasks, summarize_pinning_outcome,
                warn_if_all_dongles_pinned,
            )
            try:
                self._pinning_outcome = await start_pinned_tasks(
                    self._pin_specs,
                    registry=self.registry,
                    broker=self.broker,
                    decoder_registry=registry_handle,
                    event_bus=self.event_bus,
                    session_id=session_id,
                    gain=self.gain,
                    allow_antenna_mismatch=self._allow_pin_antenna_mismatch,
                )
            except NoDongleAvailable as exc:
                # Fatal pin validation. Surface to the caller — they
                # should abort the session cleanly. Re-raise.
                log.error("pin validation failed: %s", exc)
                raise
            for line in summarize_pinning_outcome(self._pinning_outcome):
                log.info(line)
            all_pinned = warn_if_all_dongles_pinned(
                self._pinning_outcome, self.registry,
            )
            if all_pinned:
                log.warning(all_pinned)

        scheduler = Scheduler(
            self.config, self.broker,
            all_bands=self.all_bands,
            decoder_registry=registry_handle,
        )
        bands = self.config.enabled_bands()
        plan = scheduler.plan(bands)
        # v0.7.5: stash on self so the TUI's `r` snapshot report can
        # synthesize a partial SessionResult and run the same
        # ReportBuilder pipeline as the final report. Without this,
        # the TUI report had to parse event-stream strings to
        # reconstruct band activity — much lower fidelity than a
        # DB-backed render.
        self._current_plan = plan
        log.info(
            "plan: %d wave(s), %d task(s) total",
            len(plan.waves), len(plan.tasks),
        )
        # v0.6.5: publish PlanReadyEvent so the TUI can render the
        # wave plan up front. Wrapped in try/except so a broken
        # subscriber doesn't break planning.
        try:
            from rfcensus.events import PlanReadyEvent
            wave_payloads: list[dict] = []
            for w in plan.waves:
                summaries = []
                for t in w.tasks:
                    summaries.append(
                        f"{t.band.id}→{t.suggested_dongle_id or '?'}"
                    )
                wave_payloads.append({
                    "index": w.index,
                    "task_count": len(w.tasks),
                    "task_summaries": summaries,
                })
            await self.event_bus.publish(PlanReadyEvent(
                session_id=session_id,
                waves=wave_payloads,
                total_tasks=len(plan.tasks),
                max_parallel_per_wave=plan.max_parallel_per_wave,
            ))
        except Exception:
            log.exception("PlanReadyEvent publish failed (non-fatal)")
        self._log_unassigned_warning(plan)
        self._log_decoder_binary_preflight()

        # Compute per-wave duration. If --per-band was set explicitly, use
        # it directly; otherwise divide the total budget across active waves
        # (current behavior). For indefinite runs (--duration 0/forever), we
        # require a per-band value (defaulted at the CLI layer).
        n_active_waves = sum(
            1 for w in plan.waves
            if any(t.suggested_dongle_id is not None for t in w.tasks)
        )
        if self.per_band_s is not None:
            per_wave_duration_s = float(self.per_band_s)
        else:
            per_wave_duration_s = _compute_per_wave_duration(
                total_s=self.duration_s, n_active_waves=n_active_waves
            )

        # Plan the number of passes. With a per-band budget set, we can fit
        # multiple passes through the wave list within --duration; without it
        # we run a single pass (preserves prior behavior).
        if self.indefinite:
            max_passes: int | None = None  # unbounded; loop until SIGINT
            log.info(
                "indefinite mode: looping waves with %.0fs per band "
                "(Ctrl-C to exit cleanly)",
                per_wave_duration_s,
            )
        elif self.per_band_s is not None and n_active_waves > 0:
            total_per_pass = per_wave_duration_s * n_active_waves
            max_passes = max(1, int(self.duration_s // total_per_pass))
            actual_total = max_passes * total_per_pass
            log.info(
                "duration budget: %.0fs total / %d active wave(s) × %.0fs/band "
                "= %d pass(es), %.0fs actual total",
                self.duration_s, n_active_waves, per_wave_duration_s,
                max_passes, actual_total,
            )
        else:
            max_passes = 1
            if n_active_waves > 1:
                log.info(
                    "duration budget: %.0fs total / %d active wave(s) = %.0fs per wave",
                    self.duration_s, n_active_waves, per_wave_duration_s,
                )

        self._dispatcher = Dispatcher(self.config)
        self._ctx = StrategyContext(
            config=self.config,
            event_bus=self.event_bus,
            broker=self.broker,
            decoder_registry=registry_handle,
            session_id=session_id,
            duration_s=per_wave_duration_s,
            gain=self.gain,
            all_bands=self.all_bands,
        )
        # Build band-id lookup map for retry handling
        self._band_by_id = {b.id: b for b in bands}

        # Subscribe to failure + emitter events for retry/until-quiet logic
        from rfcensus.events import DecoderFailureEvent, EmitterEvent

        async def _on_failure(evt):
            self._record_failure(evt)
        self.event_bus.subscribe(DecoderFailureEvent, _on_failure)

        async def _on_emitter(evt):
            self._handle_emitter(evt)
        self.event_bus.subscribe(EmitterEvent, _on_emitter)

        sem = asyncio.Semaphore(plan.max_parallel_per_wave)

        # Multi-pass mode = anything where we'll loop more than once.
        # In single-pass scan mode, re-probe is unnecessary.
        is_multi_pass = (max_passes is None) or (max_passes > 1)

        # Install SIGINT handler for graceful shutdown. First Ctrl-C sets
        # stop_requested; we finish the current band then exit. Second
        # Ctrl-C restores the default handler so KeyboardInterrupt fires
        # immediately.
        #
        # v0.7.6: skip the install when the TUI owns the terminal.
        # Textual captures Ctrl+C as a key event and routes it to the
        # TUI's bound action_quit_session, which presents the
        # ConfirmQuit modal. If we ALSO install our own SIGINT handler
        # here, both fire on Ctrl+C — the TUI shows the modal AND the
        # runner silently flips _stop_requested behind it. The user
        # then picks "cancel" in the modal but the wave still wraps
        # up unexpectedly. Disabling our handler when self.tui_active
        # is True makes the TUI the sole authority for Ctrl+C.
        loop = asyncio.get_event_loop()
        sigint_installed = False
        if not getattr(self, "tui_active", False):
            try:
                loop.add_signal_handler(signal.SIGINT, self._handle_sigint)
                sigint_installed = True
            except (NotImplementedError, RuntimeError):
                # Some platforms (Windows under certain conditions) don't allow
                # signal handlers in the asyncio loop. Fall back silently — the
                # default SIGINT behavior (KeyboardInterrupt) still works.
                pass

        try:
            pass_n = 0
            last_reprobe = asyncio.get_event_loop().time()
            REPROBE_NORMAL_S = 60.0   # Healthy fleet: re-probe once per minute
            REPROBE_FAILED_S = 15.0   # Something's failed: faster recovery

            # v0.6.5: deep-pause watcher. When the user keeps the
            # session paused longer than DEEP_PAUSE_THRESHOLD_S (20s),
            # tear down all shared fanouts to release CPU. On resume
            # the wave loop's normal restart path re-allocates and
            # re-launches everything fresh. Lightweight task — sleeps
            # most of the time.
            from rfcensus.engine.session_control import (
                deep_pause_watcher,
            )

            async def _on_deep_pause():
                # Deep-pause threshold crossed (>20s paused).
                # Disconnect every active downstream fanout client so
                # decoder processes exit cleanly on socket EOF. The
                # upstream rtl_tcp + lease stay alive, so the dongle
                # isn't returned to the broker pool — when the user
                # resumes, the wave loop's natural restart path
                # re-allocates and re-launches without contention.
                try:
                    n = self.broker.deep_pause_teardown_fanouts()
                    if n > 0:
                        log.info(
                            "deep-pause: %d downstream client(s) "
                            "disconnected; decoders will exit and the "
                            "next wave will rebuild",
                            n,
                        )
                except Exception:
                    log.exception("deep-pause callback failed")

            deep_pause_task = asyncio.create_task(
                deep_pause_watcher(self.control, _on_deep_pause),
                name="deep_pause_watcher",
            )
            # Don't add to _sidecar_tasks — that set's cleanup loop
            # awaits everything to completion, and this task is an
            # infinite loop. Track it on `self` so the teardown path
            # can cancel it explicitly.
            self._deep_pause_task = deep_pause_task

            while True:
                pass_n += 1
                if max_passes is not None and pass_n > max_passes:
                    break
                if self._stop_requested:
                    break
                if max_passes is None or max_passes > 1:
                    log.info("─── pass %d %s ───", pass_n,
                             f"of {max_passes}" if max_passes else "(indefinite)")

                # Periodic re-probe in any multi-pass mode (not just
                # indefinite). Adaptive interval: faster when failures
                # exist so users get quick recovery feedback.
                if is_multi_pass:
                    now = asyncio.get_event_loop().time()
                    has_failed = any(
                        d.id not in self._permanently_failed
                        and not d.is_usable()
                        for d in self.registry.dongles
                    )
                    interval = REPROBE_FAILED_S if has_failed else REPROBE_NORMAL_S
                    if now - last_reprobe >= interval:
                        last_reprobe = now
                        try:
                            from rfcensus.hardware.registry import reprobe_for_recovery
                            n_back, n_gone = await asyncio.wait_for(
                                reprobe_for_recovery(
                                    self.registry,
                                    exclude=self._permanently_failed,
                                ),
                                timeout=15.0,
                            )
                            if n_back:
                                # Surface prominently — this is good news
                                # the user wants to see, not buried in DEBUG
                                log.warning(
                                    "✓ %d dongle(s) reconnected after re-probe",
                                    n_back,
                                )
                                # Spawn any retries that can now run
                                await self._spawn_pending_retries()
                        except (asyncio.TimeoutError, TimeoutError):
                            log.warning("re-probe timed out; will retry next interval")
                        except Exception as exc:
                            log.warning("re-probe failed: %s", exc)

                for wave in plan.waves:
                    if not wave.tasks:
                        continue
                    if self._stop_requested:
                        log.info("stop requested; exiting wave loop")
                        break

                    # v0.6.5: honor pause between waves. If user hit
                    # `p`, block here until they hit `p` again. The
                    # paused wall-clock time is tracked in
                    # self.control.total_paused_s and excluded from
                    # the duration ceiling later. While paused the
                    # currently-active fanouts have already been
                    # quick-paused via control.on_pause() callback;
                    # this gate is the wave-level coordination so we
                    # don't launch new tasks during a pause.
                    if self.control.is_paused():
                        log.info(
                            "wave loop paused — waiting for resume...",
                        )
                        await self.control.wait_not_paused()
                        if self.control.stopped.is_set():
                            log.info(
                                "stop requested during pause; exiting wave loop",
                            )
                            break
                        log.info("wave loop resumed")

                    log.info("executing wave %d (%d task(s))", wave.index, len(wave.tasks))
                    await self.event_bus.publish(
                        SessionEvent(
                            session_id=session_id,
                            kind="phase_changed",
                            phase=f"pass_{pass_n}_wave_{wave.index}",
                            detail=f"{len(wave.tasks)} tasks",
                        )
                    )
                    # v0.6.5: WaveStartedEvent for the TUI plan-tree.
                    try:
                        from rfcensus.events import WaveStartedEvent
                        await self.event_bus.publish(WaveStartedEvent(
                            session_id=session_id,
                            wave_index=wave.index,
                            task_count=len(wave.tasks),
                            pass_n=pass_n,
                        ))
                    except Exception:
                        log.exception(
                            "WaveStartedEvent publish failed (non-fatal)",
                        )

                    async def _run_task(task):
                        async with sem:
                            consumer_label = f"strategy:{task.band.id}"
                            # v0.6.5: TaskStartedEvent fires unconditionally
                            # so the dashboard sees the task even if it
                            # gets skipped/crashed downstream.
                            try:
                                from rfcensus.events import TaskStartedEvent
                                await self.event_bus.publish(TaskStartedEvent(
                                    session_id=session_id,
                                    wave_index=wave.index,
                                    pass_n=pass_n,
                                    band_id=task.band.id,
                                    dongle_id=task.suggested_dongle_id or "",
                                    consumer=consumer_label,
                                ))
                            except Exception:
                                log.exception(
                                    "TaskStartedEvent publish failed",
                                )

                            task_status = "ok"
                            task_detail = ""
                            task_result = None
                            try:
                                if task.suggested_dongle_id is None:
                                    log.warning(
                                        "skipping %s: no usable dongle", task.band.id
                                    )
                                    task_status = "skipped"
                                    task_detail = "no usable dongle"
                                    task_result = StrategyResult(
                                        band_id=task.band.id, errors=["unassigned"]
                                    )
                                    return task_result
                                # Interactive hook for guided modes — runs
                                # before strategy starts, can skip the band
                                if self.before_task_hook is not None:
                                    action = await self.before_task_hook(task)
                                    if action == "skip":
                                        log.info(
                                            "user skipped %s via before_task_hook",
                                            task.band.id,
                                        )
                                        task_status = "skipped"
                                        task_detail = "user skipped"
                                        task_result = StrategyResult(
                                            band_id=task.band.id,
                                            errors=["user_skipped"],
                                        )
                                        return task_result
                                strategy = self._dispatcher.strategy_for(task.band)
                                log.info(
                                    "starting %s on %s (dongle %s)%s",
                                    strategy.__class__.__name__,
                                    task.band.id,
                                    task.suggested_dongle_id,
                                    (
                                        f" [decoders: {sorted(task.allowed_decoders)}]"
                                        if task.allowed_decoders is not None
                                        else ""
                                    ),
                                )
                                task_result = await strategy.execute(
                                    task.band,
                                    self._ctx,
                                    allowed_decoders=task.allowed_decoders,
                                )
                                # Reset consecutive-failure counter on success:
                                # a successful run wipes prior bad-luck history
                                if task_result.ended_reason != "hardware_lost":
                                    self._record_success(task.suggested_dongle_id)
                                if task_result.errors:
                                    task_status = "failed"
                                    task_detail = task_result.errors[0]
                                return task_result
                            except Exception as exc:
                                task_status = "crashed"
                                task_detail = str(exc)
                                raise
                            finally:
                                # v0.6.5: always publish TaskCompletedEvent,
                                # whether the task succeeded, was skipped, or
                                # raised. This is the dashboard's signal that
                                # a task is no longer in flight.
                                try:
                                    from rfcensus.events import TaskCompletedEvent
                                    await self.event_bus.publish(TaskCompletedEvent(
                                        session_id=session_id,
                                        wave_index=wave.index,
                                        pass_n=pass_n,
                                        band_id=task.band.id,
                                        dongle_id=task.suggested_dongle_id or "",
                                        consumer=consumer_label,
                                        status=task_status,
                                        detail=task_detail,
                                    ))
                                except Exception:
                                    log.exception(
                                        "TaskCompletedEvent publish failed",
                                    )

                    raw = await asyncio.gather(
                        *(_run_task(t) for t in wave.tasks), return_exceptions=True
                    )
                    successful_count = 0
                    wave_errors: list[str] = []
                    for r in raw:
                        if isinstance(r, Exception):
                            log.error("strategy raised: %s", r)
                            wave_errors.append(str(r))
                            continue
                        self._strategy_results.append(r)
                        if r is not None and not r.errors:
                            successful_count += 1
                        elif r is not None and r.errors:
                            wave_errors.extend(r.errors)
                    # v0.6.5: WaveCompletedEvent.
                    try:
                        from rfcensus.events import WaveCompletedEvent
                        await self.event_bus.publish(WaveCompletedEvent(
                            session_id=session_id,
                            wave_index=wave.index,
                            pass_n=pass_n,
                            task_count=len(wave.tasks),
                            successful_count=successful_count,
                            errors=wave_errors,
                        ))
                    except Exception:
                        log.exception(
                            "WaveCompletedEvent publish failed (non-fatal)",
                        )

                # End-of-pass: check --until-quiet. Only after at least
                # one full pass — we need the EmitterTracker to have had
                # a chance to fire at least once.
                if self._is_quiet():
                    log.info(
                        "--until-quiet triggered: no new emitters in %.0fs; "
                        "exiting cleanly",
                        self.until_quiet_s,
                    )
                    break

        finally:
            if sigint_installed:
                try:
                    loop.remove_signal_handler(signal.SIGINT)
                except (NotImplementedError, RuntimeError):
                    pass

        # v0.6.5: cancel the deep-pause watcher first. It's an
        # infinite loop, so the _sidecar_tasks gather below would
        # block forever waiting for it.
        if self._deep_pause_task is not None:
            self._deep_pause_task.cancel()
            try:
                await asyncio.wait_for(
                    self._deep_pause_task, timeout=2.0,
                )
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
            self._deep_pause_task = None

        # Wait briefly for any in-flight sidecar retries to complete
        if self._sidecar_tasks:
            log.info(
                "waiting for %d sidecar retry task(s) to finish",
                len(self._sidecar_tasks),
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._sidecar_tasks, return_exceptions=True),
                    timeout=30.0,
                )
            except (asyncio.TimeoutError, TimeoutError):
                log.warning("sidecar retries did not finish in 30s; cancelling")
                for t in self._sidecar_tasks:
                    t.cancel()

        # End-of-session interactive hook for guided modes — runs
        # before cleanup so the user can be prompted while the session
        # state is still meaningful (e.g. for antenna restoration).
        if self.after_session_hook is not None:
            try:
                await self.after_session_hook(self._strategy_results)
            except Exception as exc:
                log.warning("after_session_hook raised: %s", exc)

        # Cleanup: each step gets a timeout, with a clear log if it hangs.
        # Before this fix, a hung release/publish here would silently wedge
        # the entire process indefinitely.
        if power_batcher is not None:
            try:
                await asyncio.wait_for(power_batcher.stop(), timeout=10.0)
            except (asyncio.TimeoutError, TimeoutError):
                log.warning("power_batcher.stop() timed out after 10s; continuing")

        try:
            await asyncio.wait_for(self.event_bus.drain(timeout=15.0), timeout=20.0)
        except (asyncio.TimeoutError, TimeoutError):
            log.warning("event_bus.drain() timed out after 20s; continuing")

        # v0.6.0: stop pinned supervisors BEFORE broker.shutdown so the
        # broker can cleanly release the leases (and tear down any
        # rtl_tcp slots they were holding).
        if self._pinning_outcome is not None:
            from rfcensus.engine.pinning import stop_pinned_tasks
            try:
                await asyncio.wait_for(
                    stop_pinned_tasks(self._pinning_outcome, self.broker),
                    timeout=15.0,
                )
            except (asyncio.TimeoutError, TimeoutError):
                log.warning(
                    "stop_pinned_tasks() timed out after 15s; continuing"
                )

        try:
            await asyncio.wait_for(self.broker.shutdown(), timeout=15.0)
        except (asyncio.TimeoutError, TimeoutError):
            log.warning("broker.shutdown() timed out after 15s; continuing")

        ended = datetime.now(timezone.utc)
        try:
            await asyncio.wait_for(session_repo.end(session_id, ended), timeout=10.0)
        except (asyncio.TimeoutError, TimeoutError):
            log.warning("session_repo.end() timed out after 10s; continuing")
        try:
            await asyncio.wait_for(
                self.event_bus.publish(
                    SessionEvent(session_id=session_id, kind="ended", phase="finalize")
                ),
                timeout=5.0,
            )
        except (asyncio.TimeoutError, TimeoutError):
            log.warning("final SessionEvent publish timed out; continuing")

        total_decodes = sum(r.decodes_emitted for r in self._strategy_results)
        # v0.7.4: report skipped tasks so the report can mark the
        # session INCOMPLETE and tell the user exactly what didn't run.
        # A "skipped" task is one that was in the plan but never made
        # it into a StrategyResult (the wave loop exited before it
        # ran, typically because of a stop request mid-pass).
        executed_band_ids = {r.band_id for r in self._strategy_results}
        skipped = sum(
            1 for task in plan.tasks
            if task.band.id not in executed_band_ids
        )
        return SessionResult(
            session_id=session_id,
            started_at=started,
            ended_at=ended,
            plan=plan,
            strategy_results=self._strategy_results,
            total_decodes=total_decodes,
            warnings=list(plan.warnings),
            stopped_early=self._stop_requested,
            tasks_skipped_due_to_stop=(skipped if self._stop_requested else 0),
        )
