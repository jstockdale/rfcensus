"""Wide-channel aggregator.

Purpose
=======

Narrow power-scan bins (~10-25 kHz each, the typical rtl_power output)
can't see LoRa/Meshtastic signals as single channels:

  • A LoRa channel is 125/250/500 kHz wide — 5-50x a scan bin.
  • LoRa chirps SWEEP the channel: at any instant, only a narrow
    slice is lit. The full channel width never shows up in a single
    power sample.
  • LoRa bursts are short (~75-300 ms) — shorter than the 1-second
    continuous-activity threshold the OccupancyAnalyzer uses to emit
    its per-bin ActiveChannelEvents. So even if we lucky-catch a
    chirp in a bin, we still won't get an ActiveChannelEvent.

The aggregator solves both problems by:

  1. Observing ABOVE-FLOOR power samples directly, before the
     hold-time debouncing. Single-sample transients are tracked.
  2. Maintaining a rolling window of recent activity per bin.
  3. Scanning for groups of adjacent bins that collectively cover a
     target bandwidth template (125/250/500 kHz by default), and
     emitting a `WideChannelEvent` when coverage crosses a threshold.

The "coverage ratio" matters because LoRa chirps don't light every
bin simultaneously — we need some fraction of the template's bins to
have shown activity WITHIN THE WINDOW, not right now.

Scope
=====

This module does NOT attempt to detect LoRa specifically, nor decode
symbols, nor classify spreading factor. It only surfaces "there's
coherent activity that occupies a LoRa-sized slice of spectrum."
Downstream detectors (LoraDetector, future MeshtasticDetector) consume
the WideChannelEvent and do the actual classification. This keeps the
aggregator narrow-purpose and testable.

Other wide-bandwidth signals (DMR voice at 12.5 kHz, P25 at 12.5 kHz,
FM broadcast at 200 kHz, 5G n71 at many MHz) can reuse the aggregator
by configuring different templates.

Dedup
=====

Once a (center_freq, template_bw) pair has emitted, we suppress
further emissions for it until:

  • All constituent bins go silent (activity ages out of the window), OR
  • The refractory period elapses (to detect renewed activity).

This prevents a persistent LoRaWAN gateway from flooding the event
stream — we want "detected LoRa here" once, not every 500ms.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from rfcensus.events import EventBus, WideChannelEvent
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


# -------------------------------------------------------------------
# Defaults — tuned for LoRa/Meshtastic detection
# -------------------------------------------------------------------


# LoRa standard channel widths. 125 kHz is LoRaWAN SF7-12 default,
# 250 kHz is Meshtastic default, 500 kHz is high-throughput LoRa.
DEFAULT_TEMPLATES_HZ = (125_000, 250_000, 500_000)

# The activity rolling window. LoRa chirps take 1-10 ms to traverse
# the full bandwidth; a LoRa packet (preamble + payload) takes
# ~50-500 ms. Meshtastic retry gaps are a few seconds. A 5-second
# window catches typical bursty LoRa traffic while remaining short
# enough that the "coverage" reading reflects current activity.
DEFAULT_WINDOW_S = 5.0

# Fraction of the template bandwidth that must show active bins
# within the window for an emission. Too low → false positives from
# a few narrow carriers. Too high → miss sparse LoRa chirps.
# 0.5 is empirically a good starting point.
DEFAULT_COVERAGE_THRESHOLD = 0.50

# After emission, don't re-emit for the same (center_freq,
# template_bw) until this long has passed. Keeps a steady LoRaWAN
# gateway from flooding the event stream.
DEFAULT_REFRACTORY_S = 60.0

# How close adjacent bins can be (in multiples of bin width) and
# still count as "adjacent" for aggregation. 1.5x allows for small
# gaps from missing samples or off-grid bins. Above 3x we'd start
# merging genuinely unrelated carriers.
DEFAULT_ADJACENCY_TOLERANCE = 1.5

# Bandwidth tolerance when matching aggregated span to a template.
# A 250 kHz template tolerates 200-300 kHz spans. Matches the LoRa
# detector's own tolerance for consistency.
DEFAULT_BANDWIDTH_TOLERANCE = 0.20

# v0.5.40: Temporal simultaneity requirement.
#
# Why: rtl_power sweeps across a band, visiting each bin for ~1 ms per
# sweep at ~1 Hz. A 5-second rolling window means a bin is "in the
# window" if it was above-floor ANY time in the last 5 seconds — not
# if it's currently active. In an active ISM band, every 125 kHz
# sub-channel sees some traffic over 5 seconds (garage remotes, TPMS,
# smart meters, neighbor's WiFi spur, etc.) Aggregating all of that
# as a "composite" is wrong: those bursts are unrelated and
# non-simultaneous.
#
# Fix: when checking whether N adjacent bins form a composite, require
# all of their most-recent activity to be within `simultaneity_window_s`
# of each other. LoRa packets are 75-500 ms depending on SF; 200 ms
# is a safe threshold that catches even SF7 bursts while rejecting the
# multi-second accumulation of sweep-visited noise.
#
# This is the single biggest fix in v0.5.40 for realistic rtl_power
# scanning — see CHANGELOG / v0.5.40 scan evidence.
DEFAULT_SIMULTANEITY_WINDOW_S = 0.20

# v0.5.40: Dedup — overlap threshold for same-template matches.
# 0.30 is tighter than the 0.50 used in v0.5.38 and rejects adjacent
# sliding-window matches that only overlap by a few bins. For
# different-template matches (upgrade path), 0.50 still applies.
SAME_TEMPLATE_OVERLAP_THRESHOLD = 0.30
DIFFERENT_TEMPLATE_OVERLAP_THRESHOLD = 0.50

# v0.5.40: Dedup — center-distance check for same-template matches.
# If two same-template candidates have centers closer than
# `template_hz / center_distance_divisor`, they're the same signal
# seen at shifted window positions. Blocks the "sliding window walks
# across a continuous ridge" artifact that dumped hundreds of
# composite events in v0.5.38 scans.
DEFAULT_CENTER_DISTANCE_DIVISOR = 2.0


@dataclass
class _BinActivity:
    """Rolling stats for one bin. We keep one of these per bin that
    has EVER been seen active, pruned when activity ages out of the
    window."""

    freq_hz: int
    bin_width_hz: int
    first_seen: datetime
    last_seen: datetime
    peak_power_dbm: float
    sum_power_dbm: float = 0.0
    sample_count: int = 0
    noise_floor_dbm: float = -100.0

    def update(self, now: datetime, power_dbm: float, floor: float) -> None:
        self.last_seen = now
        self.peak_power_dbm = max(self.peak_power_dbm, power_dbm)
        self.sum_power_dbm += power_dbm
        self.sample_count += 1
        self.noise_floor_dbm = floor

    @property
    def avg_power_dbm(self) -> float:
        return (
            self.sum_power_dbm / self.sample_count
            if self.sample_count
            else self.peak_power_dbm
        )


@dataclass
class _Candidate:
    """One possible composite match found during a scan. The scan
    collects these across all templates before deciding which to
    emit — this lets us rank by (template_hz, coverage) and skip
    overlapping candidates rather than emitting three overlapping
    events for a single burst."""

    in_span: list[int]
    template_hz: int
    bin_width: int
    coverage: float
    composite_center: int
    actual_span: int
    # The frequency range this composite actually covers (bin-edge to
    # bin-edge). Used for overlap dedup across both candidates within
    # a scan and recent emissions across scans.
    freq_low: int
    freq_high: int


@dataclass
class WideChannelAggregator:
    """Aggregates above-floor bin activity into wide-channel detections.

    Integration pattern:

        aggregator = WideChannelAggregator(event_bus, session_id=sid)
        analyzer = OccupancyAnalyzer(
            event_bus=event_bus, session_id=sid,
            wide_aggregator=aggregator,
        )

    The OccupancyAnalyzer calls aggregator.observe(...) on every above-
    floor sample. The aggregator itself scans and emits on a cooldown
    to avoid per-sample scan cost.

    v0.5.43: `scan_interval_s` enforces the cooldown. Bin activity is
    updated on every observe() (cheap dict update), but the full
    template-matching scan runs at most every scan_interval_s seconds.
    Prior to v0.5.43 the cooldown was claimed in docs but not actually
    implemented — which caused a field regression where the sync scan
    on every sample (1040 samples per rtl_power sweep × 3 templates)
    blocked the asyncio event loop for hundreds of milliseconds per
    sweep, long enough that decoder fanout writer tasks fell behind
    their socket buffers and downstream rtl_433 clients hit read
    timeouts and exited — appearing as `ended_by=both_simultaneously`
    disconnects across multiple fanouts at the same wall-clock
    moment.

    Thread/async safety: all methods assume single-threaded asyncio
    execution. State is not protected by locks.
    """

    event_bus: EventBus
    session_id: int = 0
    # Which template widths to look for (in Hz)
    templates_hz: tuple[int, ...] = DEFAULT_TEMPLATES_HZ
    window_s: float = DEFAULT_WINDOW_S
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD
    refractory_s: float = DEFAULT_REFRACTORY_S
    adjacency_tolerance: float = DEFAULT_ADJACENCY_TOLERANCE
    bandwidth_tolerance: float = DEFAULT_BANDWIDTH_TOLERANCE
    # v0.5.40: temporal simultaneity — bins in a composite must have
    # all been seen active within this many seconds of each other.
    # See DEFAULT_SIMULTANEITY_WINDOW_S comment.
    simultaneity_window_s: float = DEFAULT_SIMULTANEITY_WINDOW_S
    # v0.5.40: center-distance dedup divisor. Two same-template
    # candidates with centers closer than template_hz /
    # center_distance_divisor are considered the same signal.
    center_distance_divisor: float = DEFAULT_CENTER_DISTANCE_DIVISOR
    # v0.5.43: scan cooldown. When > 0, the full template scan runs at
    # most this often. Bin state still updates on every observe() so
    # no activity is lost; the scan just batches. 200 ms is well below
    # typical LoRa burst duration (100-500 ms+) so real bursts still
    # get caught. Each sweep from a 26 MHz rtl_power at 25 kHz bins
    # has ~1040 samples arriving in a burst once per second; with no
    # cooldown that's 1040 × 3 templates = 3120 scans per second,
    # synchronous on the event loop. With 200 ms cooldown it's 5
    # scans per sweep.
    #
    # Default 0 keeps the v0.5.38-v0.5.42 behavior (scan on every
    # sample) for tests and third-party code that may relied on it.
    # Production callers should set this to 0.2 explicitly — see
    # engine/strategy.py where the aggregator is wired for scans.
    scan_interval_s: float = 0.0

    def __post_init__(self) -> None:
        # Per-bin activity state, keyed by freq_hz
        self._bins: dict[int, _BinActivity] = {}
        # Dedup: (freq_low, freq_high, template_hz) → last emission time
        # Keying by frequency range (not just snapped center) lets us
        # detect true overlaps between candidates of any width;
        # keying also by template_hz lets upgrade logic in
        # `_overlaps_recent_emission` distinguish a 125 kHz emission
        # (which may be superseded) from a 250 kHz one (which blocks).
        self._last_emitted: dict[tuple[int, int, int], datetime] = {}
        # Dongle that's feeding us samples (for attribution)
        self._current_dongle_id: str = ""
        # v0.5.43: cooldown tracker for backward-compatible inline scan
        # (when no background scanner is running). Uses the monotonic
        # event-loop clock. None means "never scanned" → first observe
        # forces a scan.
        self._last_scan_monotonic: float | None = None
        # v0.5.44: background scanner task. When not None, observe()
        # does NOT scan — the task runs _scan_and_emit on a regular
        # cadence. This decouples CPU-heavy template matching from
        # the observe() hot path (which runs at rtl_power sample rate,
        # ~1000/s in the 915 MHz ISM band) and lets the event loop
        # keep servicing decoder fanout writer tasks even when a
        # single scan takes tens of ms.
        self._scanner_task: asyncio.Task | None = None
        # v0.5.44: flag set when stop() is called so the loop can
        # distinguish cancellation-for-shutdown from other cancels.
        self._scanner_stopping: bool = False
        # v0.5.44: instrumentation — warn if the bin dict grows
        # unexpectedly large (suggests above-floor threshold is too
        # permissive) or if a single scan takes too long (suggests
        # we need finer-grained yields). Rate-limited to avoid log
        # flooding.
        self._last_bin_count_warn: float = 0.0
        self._last_scan_duration_warn: float = 0.0
        # v0.6.5: LoRa-detection diagnostics. When a user reports "no
        # LoRa detections even though there's definitely traffic", we
        # want to know where the pipeline dropped the signal. These
        # counters are reset at start() and emitted at stop() as a
        # diagnostic summary, so users can see:
        #   • scans_run: did the scanner even run?
        #   • scans_with_bins: was there any activity to look at?
        #   • template_matches_considered: did any span match a
        #     template bandwidth (125/250/500 kHz)?
        #   • coverage_failed / simultaneity_failed: which filter
        #     rejected the matches? This is the key signal for
        #     diagnosing rtl_power-sweep-vs-LoRa-burst mismatch.
        #   • emissions: how many WideChannelEvents actually fired.
        self._diag_scans_run: int = 0
        self._diag_scans_with_bins: int = 0
        self._diag_template_matches_considered: int = 0
        self._diag_coverage_failed: int = 0
        self._diag_simultaneity_failed: int = 0
        self._diag_bandwidth_failed: int = 0
        self._diag_emissions: int = 0

    async def observe(
        self,
        freq_hz: int,
        bin_width_hz: int,
        power_dbm: float,
        noise_floor_dbm: float,
        now: datetime,
        dongle_id: str,
    ) -> None:
        """Record one above-floor sample.

        v0.5.44: O(1) path only. Updates bin state and returns. The
        expensive template scan runs on a background task started
        via start() — NOT on the observe() hot path. This decouples
        sample-rate work (can be 1000+ Hz) from CPU-heavy scanning
        (which may take 10-100ms per run in busy bands) so the
        event loop keeps servicing other coroutines.

        For backward compatibility, if start() has NOT been called
        (no background scanner active), observe() falls back to
        inline scanning with v0.5.43's cooldown semantics. This
        preserves behavior for tests and callers that never added
        the start/stop lifecycle.
        """
        self._current_dongle_id = dongle_id

        bin_state = self._bins.get(freq_hz)
        if bin_state is None:
            bin_state = _BinActivity(
                freq_hz=freq_hz,
                bin_width_hz=bin_width_hz,
                first_seen=now,
                last_seen=now,
                peak_power_dbm=power_dbm,
                noise_floor_dbm=noise_floor_dbm,
            )
            self._bins[freq_hz] = bin_state
        bin_state.update(now, power_dbm, noise_floor_dbm)

        # v0.5.44: if background scanner is running, observe() is done.
        # The task will scan on its own cadence.
        if self._scanner_task is not None:
            return

        # Backward compat path: no scanner task, so run inline scan
        # honoring the v0.5.43 cooldown. Tests and callers that don't
        # use start/stop still get scans; they just get them on the
        # observe hot path (which is fine for test workloads).
        try:
            mono_now = asyncio.get_event_loop().time()
        except RuntimeError:
            mono_now = None

        if (
            self._last_scan_monotonic is not None
            and mono_now is not None
            and (mono_now - self._last_scan_monotonic) < self.scan_interval_s
        ):
            return
        if mono_now is not None:
            self._last_scan_monotonic = mono_now

        await self._scan_and_emit(now)

    # ----------------------------------------------------------------
    # v0.5.44: background scanner lifecycle
    # ----------------------------------------------------------------

    async def start(self) -> None:
        """Launch the background scanner task.

        After this call, observe() stops doing inline scans — the
        task drives all scanning on a fixed cadence (scan_interval_s,
        defaulting to 0.2 s if configured that way, or 0.5 s if still
        at the backward-compat default of 0).

        Idempotent: calling start() a second time is a no-op.
        """
        if self._scanner_task is not None:
            return
        self._scanner_stopping = False
        # If scan_interval_s was left at 0 (backward-compat default)
        # but start() is being explicitly called, use a sensible
        # production cadence. 0.5 s = 2 Hz, plenty fast to catch
        # LoRa bursts (100-500 ms) from the 5-second rolling window.
        effective_interval = (
            self.scan_interval_s if self.scan_interval_s > 0 else 0.5
        )
        self._scanner_effective_interval_s = effective_interval
        self._scanner_task = asyncio.create_task(
            self._scanner_loop(effective_interval),
            name="wide-channel-scanner",
        )
        log.info(
            "wide-channel scanner task started (cadence %.2fs)",
            effective_interval,
        )

    async def stop(self) -> None:
        """Cancel the background scanner task and wait for cleanup.

        Safe to call multiple times; safe to call if never started.
        """
        if self._scanner_task is None:
            return
        self._scanner_stopping = True
        self._scanner_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._scanner_task
        self._scanner_task = None
        # v0.6.5: emit diagnostic summary so users can tell whether
        # the aggregator found LoRa composites or rejected them at
        # each filter stage. Critical for diagnosing "no LoRa
        # detections despite known traffic" — the ratios reveal
        # whether the signal never matched a template (bandwidth
        # or coverage filter), or did match but failed simultaneity
        # (rtl_power sweep vs. LoRa burst timing mismatch).
        log.info(
            "wide-channel aggregator diagnostics: "
            "scans=%d scans_with_bins=%d template_matches=%d "
            "bandwidth_fail=%d coverage_fail=%d simultaneity_fail=%d "
            "emitted=%d",
            self._diag_scans_run,
            self._diag_scans_with_bins,
            self._diag_template_matches_considered,
            self._diag_bandwidth_failed,
            self._diag_coverage_failed,
            self._diag_simultaneity_failed,
            self._diag_emissions,
        )
        log.info("wide-channel scanner task stopped")

    async def _scanner_loop(self, interval_s: float) -> None:
        """Background scan loop. Runs until cancelled.

        Each iteration: sleep for interval_s, then run _scan_and_emit
        using the current wall-clock time (no sample timestamp
        available at this level — we're decoupled from the sample
        stream). Exceptions in a single iteration are logged but
        don't kill the loop; a 500ms backoff prevents tight error
        loops.
        """
        try:
            while True:
                await asyncio.sleep(interval_s)
                try:
                    scan_start = asyncio.get_event_loop().time()
                    await self._scan_and_emit(datetime.now(timezone.utc))
                    scan_elapsed = asyncio.get_event_loop().time() - scan_start
                    # Instrument long scans. Rate-limit warnings to
                    # once per 10 s so a persistently-busy band
                    # doesn't spam the log.
                    if scan_elapsed > 0.1:
                        now_mono = asyncio.get_event_loop().time()
                        if now_mono - self._last_scan_duration_warn > 10.0:
                            log.warning(
                                "wide-channel scan took %.0f ms with "
                                "%d active bins — event loop held for "
                                "that duration. If decoder fanout "
                                "clients disconnect, further tuning "
                                "may be needed.",
                                scan_elapsed * 1000.0, len(self._bins),
                            )
                            self._last_scan_duration_warn = now_mono
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception(
                        "wide-channel scanner iteration failed — "
                        "backing off 500 ms and continuing"
                    )
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            if not self._scanner_stopping:
                log.warning(
                    "wide-channel scanner cancelled unexpectedly (not "
                    "via stop())"
                )
            raise

    async def _scan_and_emit(self, now: datetime) -> None:
        """Prune stale bins, then look for composites matching templates.

        Strategy: collect ALL candidate composites across all templates,
        rank by quality (widest template + highest coverage), emit the
        best one, then emit any others whose frequency range does NOT
        overlap with a already-emitted composite. This prevents firing
        3-5 overlapping events for a single 250 kHz LoRa burst (which
        happens naturally because the sliding-window search finds
        partial matches at many start positions).

        v0.5.44: cooperative yields between templates so even in a
        pathologically-busy band (thousands of active bins) the event
        loop gets control back between chunks of work. Without yields,
        the full scan was an atomic blocking operation from the loop's
        perspective, which was the root cause of the v0.5.43-era
        decoder-fanout cascade.
        """
        self._diag_scans_run += 1
        self._prune_stale(now)
        if not self._bins:
            return
        self._diag_scans_with_bins += 1

        # v0.5.44: instrument unusual bin counts. A 26 MHz rtl_power
        # scan at 25 kHz bins has ~1040 total bins; in a quiet band
        # we'd expect maybe 20-50 above-floor. Over 500 suggests the
        # OccupancyAnalyzer's above-floor threshold is too permissive
        # (we're tracking noise as activity) — separate tuning but
        # worth surfacing. Rate-limited so it doesn't spam.
        bin_count = len(self._bins)
        if bin_count > 500:
            try:
                now_mono = asyncio.get_event_loop().time()
            except RuntimeError:
                now_mono = 0.0
            if now_mono - self._last_bin_count_warn > 30.0:
                log.warning(
                    "wide-channel aggregator tracking %d active bins — "
                    "expected < 200 in normal ISM traffic. Above-floor "
                    "threshold may be too low (noise being tracked as "
                    "activity), or the band is genuinely saturated.",
                    bin_count,
                )
                self._last_bin_count_warn = now_mono

        sorted_freqs = sorted(self._bins.keys())
        bin_widths = sorted(b.bin_width_hz for b in self._bins.values())
        bin_width = bin_widths[len(bin_widths) // 2]
        if bin_width <= 0:
            return

        # Collect all candidate matches, yielding between templates so
        # the event loop gets a turn even on large bin counts.
        candidates: list[_Candidate] = []
        for i, template_hz in enumerate(self.templates_hz):
            self._collect_template_candidates(
                template_hz=template_hz,
                sorted_freqs=sorted_freqs,
                bin_width=bin_width,
                now=now,
                out=candidates,
            )
            # v0.5.44: yield between templates. Each template's work
            # is bounded (O(n×k) where k is ~10 bins per span), but
            # back-to-back on 3 templates with 1000+ bins can block
            # the loop for 100+ms. One await between each gives
            # other coroutines (fanout writers) a chance to drain.
            if i + 1 < len(self.templates_hz):
                await asyncio.sleep(0)

        if not candidates:
            return

        # Rank: prefer wider templates, then higher coverage. A 250 kHz
        # match at 60% coverage is generally more informative than
        # a 125 kHz match at 100% coverage if they overlap — the
        # wider signal is more likely the real thing.
        candidates.sort(
            key=lambda c: (c.template_hz, c.coverage), reverse=True
        )

        # Emit non-overlapping winners
        emitted_ranges: list[tuple[int, int]] = []
        for cand in candidates:
            # Skip if overlaps a candidate we already emitted this scan
            if self._overlaps_any(cand.freq_low, cand.freq_high, emitted_ranges):
                continue
            # Skip if overlaps a recent emission (refractory, with
            # upgrade semantics — see `_overlaps_recent_emission`)
            if self._overlaps_recent_emission(
                cand.freq_low, cand.freq_high, cand.template_hz, now
            ):
                continue
            await self._emit_candidate(cand, now)
            emitted_ranges.append((cand.freq_low, cand.freq_high))
            self._last_emitted[
                (cand.freq_low, cand.freq_high, cand.template_hz)
            ] = now

    def _overlaps_any(
        self,
        low: int,
        high: int,
        ranges: list[tuple[int, int]],
        min_overlap_frac: float = 0.5,
    ) -> bool:
        """True if (low, high) overlaps any of the given ranges by at
        least min_overlap_frac of the smaller span."""
        for r_low, r_high in ranges:
            overlap_low = max(r_low, low)
            overlap_high = min(r_high, high)
            if overlap_high <= overlap_low:
                continue
            overlap = overlap_high - overlap_low
            smaller = min(high - low, r_high - r_low)
            if smaller <= 0:
                continue
            if overlap / smaller >= min_overlap_frac:
                return True
        return False

    def _overlaps_recent_emission(
        self, low: int, high: int, template_hz: int, now: datetime
    ) -> bool:
        """True if this candidate is blocked by a recent emission.

        v0.5.40: two-tier overlap logic to fix "sliding window walks
        across a continuous ridge" noise while preserving upgrade
        semantics.

        A recent emission blocks a candidate if EITHER:

          • Same-template case (r_template == template_hz):
              - Frequency ranges overlap by ≥ 30% of the smaller span
                (tighter than pre-v0.5.40's 50%), OR
              - The two centers are closer than template_hz /
                center_distance_divisor (default template/2). This
                center-distance check catches the sliding-window
                artifact where adjacent window positions on a single
                continuous ridge each fire their own composite.

          • Different-template case (r_template > template_hz):
              - Frequency ranges overlap by ≥ 50% of smaller span.
              - Wider template wins; narrower candidate is blocked.

          • Upgrade case (r_template < template_hz):
              - Frequency ranges overlap by ≥ 50% of smaller span.
              - Wider new candidate allowed through; old (narrower)
                entry is removed from refractory. See progressive-
                burst explanation in v0.5.38 comments.

        Side effect: removes superseded narrower entries from
        `self._last_emitted` so they don't accumulate.
        """
        refractory = timedelta(seconds=self.refractory_s)
        superseded: list[tuple[int, int, int]] = []
        blocked = False
        cand_center = (low + high) // 2
        center_threshold = int(template_hz / self.center_distance_divisor)
        for (r_low, r_high, r_template), t in self._last_emitted.items():
            if (now - t) > refractory:
                continue

            # Center-distance check (same-template only; a 500 kHz
            # match and a 125 kHz match can legitimately share a
            # center — that's the upgrade case, not a dup).
            if r_template == template_hz:
                r_center = (r_low + r_high) // 2
                if abs(cand_center - r_center) < center_threshold:
                    blocked = True
                    break

            # Range-overlap check with tier-specific threshold
            overlap_low = max(r_low, low)
            overlap_high = min(r_high, high)
            if overlap_high <= overlap_low:
                continue
            overlap = overlap_high - overlap_low
            smaller = min(high - low, r_high - r_low)
            if smaller <= 0:
                continue
            overlap_frac = overlap / smaller

            if r_template == template_hz:
                threshold = SAME_TEMPLATE_OVERLAP_THRESHOLD
            else:
                threshold = DIFFERENT_TEMPLATE_OVERLAP_THRESHOLD

            if overlap_frac < threshold:
                continue

            # Overlapping. Is the existing emission same-or-wider?
            if r_template >= template_hz:
                blocked = True
                break
            # Existing is narrower than this candidate — supersede it.
            superseded.append((r_low, r_high, r_template))
        for key in superseded:
            self._last_emitted.pop(key, None)
        return blocked

    def _collect_template_candidates(
        self,
        *,
        template_hz: int,
        sorted_freqs: list[int],
        bin_width: int,
        now: datetime,
        out: list,
    ) -> None:
        """Find all valid composite matches for this template and
        append them to `out` as _Candidate objects. Does NOT emit."""
        bins_per_template = max(
            1, int(round(template_hz / bin_width))
        )
        min_needed = max(2, int(bins_per_template * self.coverage_threshold))
        if len(sorted_freqs) < min_needed:
            return

        max_gap_hz = int(bin_width * self.adjacency_tolerance)
        n = len(sorted_freqs)

        for start_idx in range(n - min_needed + 1):
            start_freq = sorted_freqs[start_idx]
            span_low = start_freq - bin_width // 2
            span_high = start_freq + template_hz + bin_width // 2
            in_span: list[int] = []
            prev_freq = None
            for idx in range(start_idx, n):
                f = sorted_freqs[idx]
                if f > span_high:
                    break
                if prev_freq is not None and (f - prev_freq) > max_gap_hz:
                    break
                in_span.append(f)
                prev_freq = f

            if not in_span:
                continue

            actual_span = (in_span[-1] + bin_width // 2) - (
                in_span[0] - bin_width // 2
            )
            span_err = abs(actual_span - template_hz) / template_hz
            if span_err > self.bandwidth_tolerance:
                self._diag_bandwidth_failed += 1
                continue

            self._diag_template_matches_considered += 1
            coverage = len(in_span) / bins_per_template
            if coverage < self.coverage_threshold:
                self._diag_coverage_failed += 1
                continue

            # v0.5.40: temporal simultaneity. The bins must all have
            # been seen active within a short window of each other —
            # not just "within the 5-second rolling window". This
            # rejects sweep-induced false composites in active ISM
            # bands while still catching real ~100-500ms LoRa bursts.
            # See DEFAULT_SIMULTANEITY_WINDOW_S comment for the full
            # reasoning (tl;dr: rtl_power scans bins at ~1 Hz, so
            # "active in 5s" is NOT "simultaneously active").
            if not self._is_simultaneous(in_span):
                self._diag_simultaneity_failed += 1
                continue

            composite_center = (in_span[0] + in_span[-1]) // 2
            cand_low = in_span[0] - bin_width // 2
            cand_high = in_span[-1] + bin_width // 2

            out.append(
                _Candidate(
                    in_span=in_span,
                    template_hz=template_hz,
                    bin_width=bin_width,
                    coverage=coverage,
                    composite_center=composite_center,
                    actual_span=actual_span,
                    freq_low=cand_low,
                    freq_high=cand_high,
                )
            )

    def _is_simultaneous(self, freqs: list[int]) -> bool:
        """v0.5.40: verify all the bins in this candidate composite
        were active within `simultaneity_window_s` of each other.

        Uses the `last_seen` timestamps stored in each bin's activity
        record. The spread (max - min) must be smaller than the
        simultaneity window.

        Returns True if:
          • All freqs exist in self._bins (defensive; should always
            be true since candidates come from sorted_freqs which is
            from self._bins).
          • (max last_seen - min last_seen) ≤ simultaneity_window_s.

        Why `last_seen` and not `first_seen`: a rapidly-sweeping
        rtl_power will bump last_seen every time it re-visits a bin
        that still has a signal. last_seen represents "most recent
        activity", which is what simultaneity should compare.
        """
        if len(freqs) == 0:
            return False
        last_times = []
        for f in freqs:
            b = self._bins.get(f)
            if b is None:
                return False  # defensive; shouldn't happen
            last_times.append(b.last_seen)
        spread = (max(last_times) - min(last_times)).total_seconds()
        return spread <= self.simultaneity_window_s

    async def _emit_candidate(self, cand, now: datetime) -> None:
        self._diag_emissions += 1
        await self._emit_composite(
            in_span=cand.in_span,
            template_hz=cand.template_hz,
            bin_width=cand.bin_width,
            coverage=cand.coverage,
            composite_center=cand.composite_center,
            actual_span=cand.actual_span,
            now=now,
        )

    async def _emit_composite(
        self,
        *,
        in_span: list[int],
        template_hz: int,
        bin_width: int,
        coverage: float,
        composite_center: int,
        actual_span: int,
        now: datetime,
    ) -> None:
        """Build and publish a WideChannelEvent for the given composite."""
        states = [self._bins[f] for f in in_span if f in self._bins]
        if not states:
            return

        peak_power = max(s.peak_power_dbm for s in states)
        avg_power = sum(s.avg_power_dbm for s in states) / len(states)
        noise_floor = sum(s.noise_floor_dbm for s in states) / len(states)
        first_seen = min(s.first_seen for s in states)
        last_seen = max(s.last_seen for s in states)

        await self.event_bus.publish(
            WideChannelEvent(
                session_id=self.session_id,
                dongle_id=self._current_dongle_id,
                freq_center_hz=composite_center,
                bandwidth_hz=actual_span,
                matched_template_hz=template_hz,
                constituent_bin_count=len(in_span),
                coverage_ratio=coverage,
                peak_power_dbm=peak_power,
                avg_power_dbm=avg_power,
                noise_floor_dbm=noise_floor,
                first_seen=first_seen,
                last_seen=last_seen,
                timestamp=now,
            )
        )
        log.debug(
            "wide channel composite: %.3f MHz, %d kHz span, %d bins, "
            "coverage=%.0f%% (template %d kHz)",
            composite_center / 1e6,
            actual_span // 1000,
            len(in_span),
            coverage * 100,
            template_hz // 1000,
        )

    def _prune_stale(self, now: datetime) -> None:
        """Drop bins whose last activity is older than the window."""
        cutoff = now - timedelta(seconds=self.window_s)
        stale = [f for f, b in self._bins.items() if b.last_seen < cutoff]
        for f in stale:
            self._bins.pop(f, None)

        # Also age out dedup entries well past refractory
        dedup_cutoff = now - timedelta(seconds=self.refractory_s * 2)
        stale_keys = [
            k for k, t in self._last_emitted.items() if t < dedup_cutoff
        ]
        for k in stale_keys:
            self._last_emitted.pop(k, None)

    def _snap_center(self, freq_hz: int, template_hz: int) -> int:
        """Snap a center frequency to the nearest multiple of the
        template width so that a slightly-drifting source reports as
        the same channel across observations.

        Not the same as channel-number discretization — LoRaWAN uses
        specific channel plans that don't always align with arbitrary
        multiples. This is just a dedup key; good enough that a gateway
        whose detected center wobbles by ±10 kHz doesn't re-emit every
        time.
        """
        return round(freq_hz / template_hz) * template_hz
