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
from rfcensus.hardware.broker import DongleBroker
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
        # Set by the SIGINT handler installed during run(). Checked between
        # waves and between passes for graceful shutdown.
        self._stop_requested = False
        self._sigint_count = 0
        # Pending retries: bands that didn't get their full window because
        # their dongle died mid-run. Re-spawned as sidecar tasks when the
        # dongle reconnects.
        self._pending_retries: list[_PendingRetry] = []
        self._sidecar_tasks: set[asyncio.Task] = set()
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
        # Set during run() so sidecar retries can spawn strategies
        self._dispatcher = None
        self._ctx = None
        # Map band id → BandConfig for quick lookup from event handlers
        self._band_by_id: dict[str, object] = {}
        # v0.5.41: confirmation queue. LoRa (and potentially other)
        # detectors emit DetectionEvents flagged needs_iq_confirmation=True
        # during discovery; the DetectionWriter auto-submits those to
        # this queue. Between waves (in the execution loop below) we
        # inspect the queue and fill any wave's idle dongle slots with
        # confirmation clusters. Populated by DetectionWriter in real
        # time as discovery runs.
        from rfcensus.engine.confirmation_queue import ConfirmationQueue
        self._confirmation_queue = ConfirmationQueue()

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
    # v0.5.41: LoRa confirmation integration
    # ------------------------------------------------------------

    def _augment_wave_with_confirmations(self, wave, detection_repo) -> None:
        """Add confirmation-cluster tasks to this wave for any idle
        dongle slots where the confirmation queue has matching work.

        Called once per wave, just before execution. Does NOT grow
        the wave past its natural size — we only fill slots that
        would otherwise be idle (not reserved for primary tasks).

        Matching criteria:
          • Dongle covers the cluster's freq range (dongle.covers(freq))
          • Dongle's antenna matches the cluster's band reasonably
            (delegated to the wave planner's antenna matcher)
          • Dongle is not already reserved by this wave's primary tasks

        Each confirmation cluster gets scheduled against exactly one
        dongle and becomes a ScheduleTask with confirmation_cluster
        set. The normal _run_task dispatch picks this up and routes
        to _run_confirmation_task.
        """
        if not self._confirmation_queue.has_work():
            return

        # Which dongles are already reserved by this wave's primary
        # tasks? Walk the wave's existing tasks and collect their
        # suggested dongles; these are unavailable for confirmation.
        reserved: set[str] = set()
        for task in wave.tasks:
            if task.suggested_dongle_id is not None:
                reserved.add(task.suggested_dongle_id)

        # Usable dongles minus reserved = idle pool for this wave
        usable = self.registry.usable()
        idle_dongles = [d for d in usable if d.id not in reserved]
        if not idle_dongles:
            return

        # For each idle dongle, try to match a cluster from the queue.
        # Greedy: first compatible match wins. Clusters already used
        # in this wave are tracked so we don't schedule the same
        # cluster on two dongles.
        added = 0
        scheduled_clusters: list[object] = []

        for dongle in idle_dongles:
            # Cluster compatibility: the cluster's center must be
            # within the dongle's tuning range.
            # Use registry's covers() helper — same check as
            # Scheduler.plan() uses for primary bands.
            if not hasattr(dongle, "covers"):
                continue
            candidates = self._confirmation_queue.clusters_in_range(
                freq_low=int(getattr(dongle, "freq_low", 0) or 0),
                freq_high=int(
                    getattr(dongle, "freq_high", 0) or 10_000_000_000
                ),
            )
            if not candidates:
                continue

            # Skip clusters already scheduled for earlier dongles in
            # this wave (avoid double-booking)
            for cluster in candidates:
                if id(cluster) in {id(c) for c in scheduled_clusters}:
                    continue
                # Synthesize a BandConfig-like marker so the task's
                # antenna / wave-loop code has something to reference.
                # We don't need the full BandConfig API — just an id
                # and center_hz for logging.
                from rfcensus.config.schema import BandConfig
                synthetic_band = BandConfig(
                    id=f"confirm_{cluster.center_freq_hz // 1000}khz",
                    name=f"LoRa confirmation @ {cluster.center_freq_hz/1e6:.3f} MHz",
                    freq_low=cluster.min_freq_hz,
                    freq_high=cluster.max_freq_hz,
                    strategy="decoder_only",  # any valid value; unused for this path
                    effective_power_scan_bin_hz=25_000,
                )
                task = ScheduleTask(
                    band=synthetic_band,
                    suggested_dongle_id=dongle.id,
                    suggested_antenna_id=(
                        dongle.antenna.id if dongle.antenna else None
                    ),
                    dongles_needed=1,
                    confirmation_cluster=cluster,
                )
                wave.tasks.append(task)

                # Mark all tasks in this cluster as scheduled (remove
                # from pending) to prevent another wave from picking
                # them up.
                # Run this synchronously on the asyncio loop
                import asyncio as _asyncio
                try:
                    _asyncio.get_event_loop().create_task(
                        self._confirmation_queue.mark_scheduled(cluster)
                    )
                except Exception:
                    pass

                scheduled_clusters.append(cluster)
                added += 1
                break  # one cluster per idle dongle

        if added:
            log.info(
                "v0.5.41: filled %d idle slot(s) in wave %d with "
                "LoRa confirmation cluster(s); queue status: %s",
                added, wave.index,
                self._confirmation_queue.status_line(),
            )

    async def _run_confirmation_task(self, task, detection_repo):
        """Execute a confirmation cluster using an allocated lease.

        Follows the same lease-allocation pattern as primary strategies,
        but uses capture_with_lease() since we need multiple sub-captures
        on the same dongle over the task's listening window.
        """
        from rfcensus.engine.confirmation_task import run_batched_confirmation
        from rfcensus.engine.dongle_broker import AccessMode, DongleRequirements
        from rfcensus.spectrum import IQCaptureService

        cluster = task.confirmation_cluster
        if cluster is None:
            return StrategyResult(
                band_id=task.band.id,
                errors=["no confirmation cluster attached"],
            )

        iq_service = IQCaptureService(self.broker)

        requirements = DongleRequirements(
            freq_hz=cluster.center_freq_hz,
            sample_rate=cluster.sample_rate,
            access_mode=AccessMode.EXCLUSIVE,
            prefer_driver="rtlsdr",
        )

        lease = None
        confirmed_ids: set[int] = set()
        try:
            try:
                lease = await self.broker.allocate(
                    requirements,
                    consumer=f"lora_confirmation:{cluster.center_freq_hz//1000}khz",
                    timeout=5.0,
                )
            except Exception as exc:
                log.warning(
                    "confirmation task: failed to allocate dongle for "
                    "%s: %s. Queue entries restored to pending for next wave.",
                    cluster.describe(), exc,
                )
                # Return cluster to pending so it might get retried
                # in a later wave
                async with self._confirmation_queue._lock:
                    for t in cluster.tasks:
                        self._confirmation_queue._pending[t.dedup_key] = t
                    self._confirmation_queue._in_flight.discard(id(cluster))
                return StrategyResult(
                    band_id=task.band.id,
                    errors=[f"lease allocation failed: {exc}"],
                )

            # Progress callback — emit to stderr so it's visible
            def _progress(msg: str) -> None:
                import click
                try:
                    click.echo(msg, err=True)
                except Exception:
                    pass

            confirmed_ids = await run_batched_confirmation(
                cluster=cluster,
                lease=lease,
                iq_service=iq_service,
                detection_repo=detection_repo,
                session_id=self._current_session_id if hasattr(
                    self, "_current_session_id"
                ) else 0,
                progress_cb=_progress,
            )
        finally:
            if lease is not None:
                try:
                    await self.broker.release(lease)
                except Exception:
                    log.exception("error releasing confirmation lease")
            await self._confirmation_queue.mark_completed(
                cluster, confirmed_ids
            )

        return StrategyResult(
            band_id=task.band.id,
            decodes_emitted=len(confirmed_ids),
        )

    async def _offer_confirmation_wave(self, detection_repo) -> None:
        """Prompt the operator to run an additional confirmation-only
        wave if the queue still has pending work after all main waves
        complete. 120-second timeout with default N.

        Estimates time cost so the operator can make an informed choice:
        N clusters × 120 s worst case, but typically much less because
        of early exit when transmitters burst promptly.
        """
        import click

        pending_count = self._confirmation_queue.pending_count()
        if pending_count == 0:
            return

        clusters = self._confirmation_queue.cluster_for_capture()
        if not clusters:
            return

        # Worst-case time: one cluster per free dongle, serialized
        # (which isn't actually what'd happen — they'd run in
        # parallel, one per dongle), capped at max_duration_s.
        worst_s = sum(c.max_duration_s for c in clusters)
        usable_count = max(1, len(self.registry.usable()))
        estimated_s = worst_s / usable_count
        prompt = (
            f"\n{pending_count} LoRa detection(s) still need IQ confirmation "
            f"for SF classification.\n"
            f"Estimated ≤ {estimated_s/60:.0f} min to confirm "
            f"({len(clusters)} cluster(s), {usable_count} dongle(s) free, "
            f"{worst_s/60:.0f} min total budget).\n"
            f"Run an additional confirmation-only wave? [y/N] "
            f"(120s timeout → N): "
        )
        click.echo(prompt, err=True, nl=False)

        answer = await self._prompt_with_timeout(timeout_s=120.0)
        if answer is None or answer.strip().lower() not in ("y", "yes"):
            click.echo(
                f"\nSkipping confirmation wave. {pending_count} "
                f"LoRa detection(s) kept with estimated_sf=None. "
                f"Run `rfcensus scan` again later to re-attempt.",
                err=True,
            )
            return

        click.echo(
            f"\nRunning confirmation-only wave: {len(clusters)} cluster(s)...",
            err=True,
        )

        # Execute each cluster against any available dongle.
        # Parallel across dongles.
        from rfcensus.engine.scheduler import Wave, ScheduleTask
        from rfcensus.config.schema import BandConfig

        tasks_list: list[ScheduleTask] = []
        usable = list(self.registry.usable())
        for cluster, dongle in zip(clusters, usable):
            synthetic_band = BandConfig(
                id=f"confirm_{cluster.center_freq_hz // 1000}khz",
                name=f"LoRa confirmation @ {cluster.center_freq_hz/1e6:.3f} MHz",
                freq_low=cluster.min_freq_hz,
                freq_high=cluster.max_freq_hz,
                strategy="decoder_only",
                effective_power_scan_bin_hz=25_000,
            )
            tasks_list.append(
                ScheduleTask(
                    band=synthetic_band,
                    suggested_dongle_id=dongle.id,
                    suggested_antenna_id=(
                        dongle.antenna.id if dongle.antenna else None
                    ),
                    dongles_needed=1,
                    confirmation_cluster=cluster,
                )
            )
            await self._confirmation_queue.mark_scheduled(cluster)

        # Run in parallel
        results = await asyncio.gather(
            *(self._run_confirmation_task(t, detection_repo) for t in tasks_list),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                log.error("confirmation wave task raised: %s", r)
                continue
            self._strategy_results.append(r)

        click.echo(
            f"Confirmation wave complete: "
            f"{self._confirmation_queue.completed_count()} confirmed, "
            f"{self._confirmation_queue.abandoned_count()} timed out.",
            err=True,
        )

    async def _prompt_with_timeout(self, *, timeout_s: float) -> str | None:
        """Read a single line from stdin with a timeout. Returns None
        on timeout (caller treats as default answer). Intended for
        the Y/N confirmation-wave prompt.

        Non-tty stdin (e.g. CI, piped input) returns None immediately
        so automated runs don't hang for 120s waiting for a human.
        """
        import sys
        if not sys.stdin.isatty():
            return None

        try:
            loop = asyncio.get_event_loop()
            answer = await asyncio.wait_for(
                loop.run_in_executor(None, sys.stdin.readline),
                timeout=timeout_s,
            )
            return answer
        except (asyncio.TimeoutError, TimeoutError):
            return None
        except Exception:
            return None


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
            confirmation_queue=self._confirmation_queue,
        )

        # Attach detectors — they subscribe to ActiveChannelEvent
        # and emit DetectionEvent; the DetectionWriter persists them.
        # Attach detectors — they subscribe to ActiveChannelEvent and emit
        # DetectionEvent; the DetectionWriter persists them. Detectors that
        # declare consumes_iq=True get an IQCaptureService for escalated
        # confirmation.
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

        scheduler = Scheduler(
            self.config, self.broker,
            all_bands=self.all_bands,
            decoder_registry=registry_handle,
        )
        bands = self.config.enabled_bands()
        plan = scheduler.plan(bands)
        log.info(
            "plan: %d wave(s), %d task(s) total",
            len(plan.waves), len(plan.tasks),
        )
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
        loop = asyncio.get_event_loop()
        sigint_installed = False
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

                    # v0.5.41: fill idle slots in this wave with
                    # confirmation clusters. Queue may be empty on
                    # wave 0, but waves 1+ pick up work submitted by
                    # detections from earlier waves.
                    self._augment_wave_with_confirmations(wave, detection_repo)

                    log.info("executing wave %d (%d task(s))", wave.index, len(wave.tasks))
                    await self.event_bus.publish(
                        SessionEvent(
                            session_id=session_id,
                            kind="phase_changed",
                            phase=f"pass_{pass_n}_wave_{wave.index}",
                            detail=f"{len(wave.tasks)} tasks",
                        )
                    )

                    async def _run_task(task):
                        async with sem:
                            if task.suggested_dongle_id is None:
                                log.warning(
                                    "skipping %s: no usable dongle", task.band.id
                                )
                                return StrategyResult(
                                    band_id=task.band.id, errors=["unassigned"]
                                )
                            # v0.5.41: confirmation tasks route to a
                            # different executor (IQ capture + DDC +
                            # chirp analysis), not the normal strategy
                            # pipeline. The band is synthesized just
                            # for antenna matching.
                            if task.confirmation_cluster is not None:
                                return await self._run_confirmation_task(
                                    task, detection_repo
                                )
                            # Interactive hook for guided modes — runs
                            # before strategy starts, can skip the band
                            if self.before_task_hook is not None:
                                action = await self.before_task_hook(task)
                                if action == "skip":
                                    log.info(
                                        "user skipped %s via before_task_hook",
                                        task.band.id,
                                    )
                                    return StrategyResult(
                                        band_id=task.band.id,
                                        errors=["user_skipped"],
                                    )
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
                            result = await strategy.execute(
                                task.band,
                                self._ctx,
                                allowed_decoders=task.allowed_decoders,
                            )
                            # Reset consecutive-failure counter on success:
                            # a successful run wipes prior bad-luck history
                            if result.ended_reason != "hardware_lost":
                                self._record_success(task.suggested_dongle_id)
                            return result

                    raw = await asyncio.gather(
                        *(_run_task(t) for t in wave.tasks), return_exceptions=True
                    )
                    for r in raw:
                        if isinstance(r, Exception):
                            log.error("strategy raised: %s", r)
                            continue
                        self._strategy_results.append(r)

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

            # v0.5.41: after all main waves complete, offer to run an
            # additional confirmation-only wave if the queue still has
            # pending tasks that didn't fit into any main wave's idle
            # slots. The whole dongle pool is free at this point.
            if (
                not self._stop_requested
                and self._confirmation_queue.has_work()
            ):
                await self._offer_confirmation_wave(detection_repo)
        finally:
            if sigint_installed:
                try:
                    loop.remove_signal_handler(signal.SIGINT)
                except (NotImplementedError, RuntimeError):
                    pass

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
        return SessionResult(
            session_id=session_id,
            started_at=started,
            ended_at=ended,
            plan=plan,
            strategy_results=self._strategy_results,
            total_decodes=total_decodes,
            warnings=list(plan.warnings),
        )
