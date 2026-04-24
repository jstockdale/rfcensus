"""v0.5.41: confirmation queue for post-detection IQ capture.

Architecture
============

The LoRa detector (and potentially others) emits DetectionEvents during
inventory scans that lack IQ-derived metadata like spreading factor and
variant label. Rather than fight the scheduler for a dongle in real time
(which failed in v0.5.40 scans — all dongles were exclusively leased),
we submit deferred tasks to this queue.

The wave planner consumes the queue during wave packing: any time a
dongle+antenna pair is idle in an upcoming wave AND the queue has a
cluster whose center frequency falls in that dongle's tuning range,
the planner schedules the cluster as a confirmation task in that slot.
This way:

  • Confirmation tasks compete for dongles through the same mechanism
    as primary tasks — no parallel allocation path, no races.
  • Waves with idle capacity (wave 3-4 in your typical scan) naturally
    fill those slots with confirmation work.
  • Waves that are fully packed don't grow; their confirmation backlog
    either waits for the next wave with room, or (at session end) the
    user is prompted to run one additional confirmation-only wave.

Batched clustering
==================

A single IQ capture at 2.4 Msps covers a ~2 MHz window around the
tuner center. If multiple detections fall within one such window, one
capture produces IQ for all of them — extractable via digital down-
conversion (see `rfcensus.tools.dsp.digital_downconvert`). This is
dramatically more efficient than per-detection captures for clusters
of LoRaWAN channels.

`cluster_for_capture()` greedily groups pending tasks by proximity
such that each cluster fits in one capture window. The 2 MHz default
leaves 200 kHz of guard on each side of a 2.4 Msps capture where
aliasing and DC-spike artifacts degrade signal quality.

Dedup
=====

A given transmitter may be detected many times across a scan (e.g., a
LoRaWAN gateway sends every 1-2 min, multiple rtl_power sweeps catch
it). `submit()` deduplicates by `(freq_rounded_to_10kHz, bandwidth)` —
one queue entry per detected transmitter regardless of how many
DetectionEvents named it.

Patience
========

LoRa transmitters are bursty: Meshtastic beacons every 30s-5min,
LoRaWAN class A uplinks every 1-60min. A single 500ms capture is
unlikely to catch a burst. The task runner (confirmation_task.py)
captures in 2-second chunks and keeps listening until either all
tasks in the cluster report a chirp OR the task's max_duration_s
(default 120s) elapses. Outstanding tasks at timeout get `estimated_sf
= None` with a WARNING — operator can try again by running another
scan later.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable

from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


# Default IQ capture window budget — a 2.4 Msps capture cleanly
# covers ±1.2 MHz around the tuner center. Leaving 200 kHz guard on
# each side avoids DC-spike and aliasing artifacts at the capture
# band edges. Cluster spans up to 2 MHz.
DEFAULT_CLUSTER_COVERAGE_HZ = 2_000_000

# Default bucket size for frequency-dedup. A LoRaWAN gateway's center
# frequency as reported by the aggregator can drift ±10 kHz between
# detections due to rtl_power bin alignment. 50 kHz dedup resolution
# is coarse enough to merge these but fine enough that two adjacent
# LoRaWAN channels (200 kHz apart typical) stay distinct.
DEFAULT_DEDUP_BUCKET_HZ = 50_000

# Default sample rate for confirmation captures. Matches rtl_sdr's
# cleanest supported rate and provides ~2 MHz of usable signal
# bandwidth around the tuner center.
DEFAULT_CAPTURE_SAMPLE_RATE = 2_400_000


@dataclass
class ConfirmationTask:
    """A single detection awaiting IQ confirmation.

    Carries the information needed to:
      (a) Schedule: freq_hz + bandwidth_hz determine dongle+antenna
          compatibility; the dedup_key prevents duplicate submissions.
      (b) Execute: detection_id links back to the DB row to update with
          the estimated_sf / variant / iq_confirmed fields.
      (c) Report: the technology + freq are logged during the "listening"
          phase so the operator sees progress.

    Immutable once submitted — the queue tracks outstanding/completed
    state separately.
    """

    detection_id: int
    freq_hz: int
    bandwidth_hz: int
    technology: str  # "lora" / "lorawan" / "meshtastic" — used in log lines
    detector_name: str  # who submitted this (for future multi-detector use)

    @property
    def dedup_key(self) -> tuple[int, int]:
        """Bucket by (rounded_freq, bandwidth). Same bucket = same
        transmitter, multiple DetectionEvents collapse to one queue
        entry. See DEFAULT_DEDUP_BUCKET_HZ for rationale."""
        return (
            round(self.freq_hz / DEFAULT_DEDUP_BUCKET_HZ)
            * DEFAULT_DEDUP_BUCKET_HZ,
            self.bandwidth_hz,
        )


@dataclass
class BatchedConfirmationTask:
    """A cluster of ConfirmationTasks sharing one IQ capture.

    All `tasks` have center frequencies within `capture_coverage_hz`
    of each other — one IQ capture at `center_freq_hz` covers all of
    them. Per-task DDC extracts each individual channel from the
    wideband capture.

    Scheduler sees this as a single task with:
      • center_freq_hz — the tuner frequency
      • max_freq_hz / min_freq_hz — the covered span (for antenna match)
      • sample_rate — DEFAULT_CAPTURE_SAMPLE_RATE
      • max_duration_s — listening budget (default 120s)

    The cluster completes when either all tasks have been confirmed
    or max_duration_s elapses.
    """

    tasks: list[ConfirmationTask]
    capture_coverage_hz: int = DEFAULT_CLUSTER_COVERAGE_HZ
    sample_rate: int = DEFAULT_CAPTURE_SAMPLE_RATE
    max_duration_s: float = 120.0

    @property
    def center_freq_hz(self) -> int:
        """Midpoint of the cluster's freq range. Captures are tuned
        here; each task's baseband shift = task.freq_hz - center_freq_hz."""
        if not self.tasks:
            raise ValueError("empty cluster has no center")
        lo = min(t.freq_hz for t in self.tasks)
        hi = max(t.freq_hz for t in self.tasks)
        return (lo + hi) // 2

    @property
    def min_freq_hz(self) -> int:
        return min(t.freq_hz for t in self.tasks) - max(t.bandwidth_hz for t in self.tasks) // 2

    @property
    def max_freq_hz(self) -> int:
        return max(t.freq_hz for t in self.tasks) + max(t.bandwidth_hz for t in self.tasks) // 2

    @property
    def span_hz(self) -> int:
        """Tip-to-tip frequency span including each task's half-bandwidth."""
        return self.max_freq_hz - self.min_freq_hz

    @property
    def size(self) -> int:
        return len(self.tasks)

    def describe(self) -> str:
        freqs = ", ".join(f"{t.freq_hz/1e6:.3f}" for t in self.tasks)
        return (
            f"cluster@{self.center_freq_hz/1e6:.3f}MHz "
            f"spanning {self.span_hz/1e3:.0f}kHz, "
            f"{self.size} task(s): {freqs}"
        )


class ConfirmationQueue:
    """Accepts ConfirmationTasks, clusters them, hands clusters to the
    scheduler.

    Thread-safety: designed for single-threaded asyncio. All state
    mutations happen in coroutine context.

    Observability: `pending_count()`, `outstanding_count()`, and
    `completed_count()` give the scheduler and the UI visibility into
    queue depth.
    """

    def __init__(
        self,
        *,
        cluster_coverage_hz: int = DEFAULT_CLUSTER_COVERAGE_HZ,
        dedup_bucket_hz: int = DEFAULT_DEDUP_BUCKET_HZ,
    ):
        self.cluster_coverage_hz = cluster_coverage_hz
        self.dedup_bucket_hz = dedup_bucket_hz
        # Submitted, not yet scheduled
        self._pending: dict[tuple[int, int], ConfirmationTask] = {}
        # Scheduled but not yet completed (cluster-ids that are in-flight)
        self._in_flight: set[int] = set()
        # Completed — kept for session-end reporting
        self._completed: list[ConfirmationTask] = []
        # Abandoned — timed out or errored
        self._abandoned: list[ConfirmationTask] = []
        self._lock = asyncio.Lock()
        # Callback: set by consumers (UI, tests) to hear about newly
        # pending clusters. Optional.
        self.on_submit: Callable[[ConfirmationTask], None] | None = None

    # ------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------

    async def submit(self, task: ConfirmationTask) -> bool:
        """Submit a task. Returns True if newly queued, False if this
        transmitter is already pending or in-flight (deduped).

        Dedup-by-key prevents the queue from filling with 20 entries
        for the same LoRaWAN gateway that got detected 20 times during
        a long scan.
        """
        async with self._lock:
            key = task.dedup_key
            if key in self._pending:
                log.debug(
                    "confirmation submit deduped: %.3f MHz bw=%d kHz "
                    "already pending",
                    task.freq_hz / 1e6, task.bandwidth_hz // 1000,
                )
                return False
            # Check if already completed this session
            for done in self._completed:
                if done.dedup_key == key:
                    log.debug(
                        "confirmation submit skipped: %.3f MHz already "
                        "confirmed this session",
                        task.freq_hz / 1e6,
                    )
                    return False
            self._pending[key] = task
            log.info(
                "confirmation queued: %s at %.3f MHz (%d kHz bw) — "
                "SF classification pending IQ capture",
                task.technology, task.freq_hz / 1e6,
                task.bandwidth_hz // 1000,
            )
            if self.on_submit is not None:
                try:
                    self.on_submit(task)
                except Exception:
                    log.exception("on_submit callback raised")
            return True

    # ------------------------------------------------------------
    # Clustering (pure, testable independently)
    # ------------------------------------------------------------

    def cluster_for_capture(
        self, tasks: list[ConfirmationTask] | None = None
    ) -> list[BatchedConfirmationTask]:
        """Greedy-pack pending tasks into clusters, each fitting within
        cluster_coverage_hz.

        The algorithm:
          1. Sort pending tasks by center frequency.
          2. For each task: add to current cluster if doing so keeps
             the cluster span ≤ cluster_coverage_hz. Otherwise close
             the current cluster and start a new one.
          3. Return the list of clusters.

        This is a classic interval-packing greedy: optimal for minimum
        number of clusters given the constraint.

        Pass `tasks` explicitly to cluster a subset (e.g., only those
        compatible with a given dongle/antenna). Default = all pending.
        """
        if tasks is None:
            tasks = list(self._pending.values())
        if not tasks:
            return []
        sorted_tasks = sorted(tasks, key=lambda t: t.freq_hz)
        clusters: list[list[ConfirmationTask]] = []
        current: list[ConfirmationTask] = [sorted_tasks[0]]
        for task in sorted_tasks[1:]:
            proposed_min = min(current[0].freq_hz, task.freq_hz)
            proposed_max = max(
                current[-1].freq_hz, task.freq_hz
            )
            # Include bandwidth on each end so the cluster's usable
            # span accommodates the channel widths.
            max_bw = max(
                max(t.bandwidth_hz for t in current),
                task.bandwidth_hz,
            )
            proposed_span = proposed_max - proposed_min + max_bw
            if proposed_span <= self.cluster_coverage_hz:
                current.append(task)
            else:
                clusters.append(current)
                current = [task]
        if current:
            clusters.append(current)
        return [
            BatchedConfirmationTask(tasks=c, capture_coverage_hz=self.cluster_coverage_hz)
            for c in clusters
        ]

    def clusters_in_range(
        self, freq_low: int, freq_high: int
    ) -> list[BatchedConfirmationTask]:
        """Return clusters whose tasks all fit within the given
        frequency range. Used by the wave planner to match clusters
        against a dongle+antenna's tuning range.

        A cluster is "in range" if ALL its tasks are within
        [freq_low, freq_high]. Partial overlap is not enough — a
        cluster is an atomic scheduling unit.
        """
        matching_tasks = [
            t for t in self._pending.values()
            if freq_low <= t.freq_hz <= freq_high
        ]
        return self.cluster_for_capture(matching_tasks)

    # ------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------

    async def mark_scheduled(self, cluster: BatchedConfirmationTask) -> None:
        """Remove cluster's tasks from pending, mark as in-flight."""
        async with self._lock:
            for task in cluster.tasks:
                self._pending.pop(task.dedup_key, None)
            self._in_flight.add(id(cluster))

    async def mark_completed(
        self, cluster: BatchedConfirmationTask, confirmed_ids: set[int]
    ) -> None:
        """Cluster finished. confirmed_ids are detection IDs that
        successfully got a chirp classification; others are abandoned.
        """
        async with self._lock:
            self._in_flight.discard(id(cluster))
            for task in cluster.tasks:
                if task.detection_id in confirmed_ids:
                    self._completed.append(task)
                else:
                    self._abandoned.append(task)

    # ------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------

    def pending_count(self) -> int:
        return len(self._pending)

    def outstanding_count(self) -> int:
        return len(self._pending) + len(self._in_flight)

    def completed_count(self) -> int:
        return len(self._completed)

    def abandoned_count(self) -> int:
        return len(self._abandoned)

    def has_work(self) -> bool:
        return bool(self._pending)

    def status_line(self) -> str:
        return (
            f"confirmation queue: {self.pending_count()} pending, "
            f"{len(self._in_flight)} in-flight, "
            f"{self.completed_count()} confirmed, "
            f"{self.abandoned_count()} abandoned"
        )
