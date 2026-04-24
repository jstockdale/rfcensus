"""Scheduler: builds a wave-based execution plan.

A `Wave` is a set of `ScheduleTask`s that can run in parallel because
they don't share dongles. Waves run sequentially. This guarantees the
broker never sees impossible concurrent requests.

First-fit packing by best-match score. Good enough for the typical
4-dongle + 5-10 band setup; can refine later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from rfcensus.config.schema import BandConfig, SiteConfig
from rfcensus.hardware.antenna import AntennaMatcher
from rfcensus.hardware.broker import AccessMode, DongleBroker
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ScheduleTask:
    band: BandConfig
    suggested_dongle_id: str | None
    suggested_antenna_id: str | None
    notes: list[str] = field(default_factory=list)
    # Total number of dongles this band's strategy execution will want
    # — primary decoders + shared decoders + optional power-scan sidecar.
    # Used by wave packing to avoid placing a band in a wave that can't
    # host all its sidecars. At least 1 (the primary). Zero-valued tasks
    # are ones we couldn't assign at all.
    dongles_needed: int = 1
    # Optional restriction on which decoders run for this task. None
    # (default) means "run every decoder matched by band.suggested_decoders
    # and the registry." A set restricts _pick_decoders to exactly those
    # names. Used by the plan-time splitter to defer surplus exclusive
    # decoders to later waves when the current wave's suitable-dongle
    # pool can't host them all in parallel (v0.5.35). Set elements are
    # decoder names (e.g. "multimon", "direwolf").
    allowed_decoders: set[str] | None = None
    # v0.5.41: if set, this task is a LoRa confirmation batch (not a
    # normal strategy run). The session dispatch checks this field
    # first and routes to run_batched_confirmation() when present,
    # which allocates the suggested dongle exclusively, captures IQ
    # at the cluster's center frequency, and classifies each task in
    # the cluster via DDC + chirp analysis. The `band` field is
    # synthesized to cover the cluster's span for antenna matching.
    confirmation_cluster: object | None = None  # BatchedConfirmationTask


def _estimate_dongle_needs(
    band: BandConfig, registry=None
) -> int:
    """How many distinct dongles will this band's strategy actually want?

    Mirrors what the strategy classes actually spawn at runtime:

      • power_primary / exploration: run rtl_power only (1 dongle).
        Items in suggested_decoders are advisory for future on-demand
        activation; they don't allocate anything upfront.

      • decoder_only: count matched decoders. Each exclusive decoder
        gets its own dongle, all shared decoders share one dongle.

      • decoder_primary: count decoders + add rtl_power sidecar if the
        band is wide or has power_scan_parallel set. If suggested_decoders
        lists no actual decoder (common with detector-only bands like
        P25 where "p25" is a detector not a decoder), fall through to
        the power_primary behavior (rtl_power only, 1 dongle).

    Over-estimation is conservative — causes more waves than necessary.
    Under-estimation is aggressive — packs waves tightly and lets
    runtime allocation fail gracefully (v0.5.14 defer). Goal: match
    runtime behavior as closely as possible so planning doesn't guess.
    """
    from rfcensus.config.schema import StrategyKind

    runs_power_scan = (
        band.power_scan_parallel or band.bandwidth_hz >= 5_000_000
    )

    # power_primary and exploration strategies don't spawn decoders.
    # Their "primary" IS rtl_power.
    if band.strategy in (StrategyKind.POWER_PRIMARY, StrategyKind.EXPLORATION):
        return 1

    # decoder_only / decoder_primary — count matched decoders
    n_exclusive = 0
    n_shared = 0
    suggested = set(band.suggested_decoders)

    if registry is not None:
        for name in registry.names():
            # STRICT filter — empty suggested_decoders means "no decoder"
            # for planning purposes, not "any decoder." Runtime's
            # _pick_decoders has the same behavior effectively (its
            # `suggested and name not in suggested` check allows all
            # when suggested is empty, but in practice empty means
            # decoder_only/decoder_primary bands shouldn't have zero
            # suggestions — if they do, treat as no decoder).
            if not suggested or name not in suggested:
                continue
            cls = registry.get(name)
            if cls is None:
                continue
            caps = cls.capabilities
            if not any(
                low <= band.freq_high and high >= band.freq_low
                for low, high in caps.freq_ranges
            ):
                continue
            if caps.access_mode == AccessMode.SHARED:
                n_shared += 1
            else:
                n_exclusive += 1
    else:
        # No registry — approximate using the shared-decoder hint set
        shared_hint = {"rtlamr"}
        for name in suggested:
            if name in shared_hint:
                n_shared += 1
            else:
                n_exclusive += 1

    # Shared decoders share one dongle, exclusives each get their own
    decoder_slots = n_exclusive + (1 if n_shared > 0 else 0)

    if decoder_slots == 0:
        # No decoder actually matches — the strategy runs only its
        # power scan (if wide) or nothing meaningful. Either way,
        # just 1 dongle at runtime (rtl_power or idle).
        return 1

    if band.strategy == StrategyKind.DECODER_PRIMARY and runs_power_scan:
        return decoder_slots + 1

    return decoder_slots


def _suitable_dongle_ids(band: BandConfig, usable_dongles) -> set[str]:
    """IDs of dongles whose antenna can cover this band's center."""
    return {
        d.id for d in usable_dongles
        if d.antenna is not None and d.antenna.covers(band.center_hz)
    }


@dataclass
class Wave:
    index: int
    tasks: list[ScheduleTask]


@dataclass
class ExecutionPlan:
    waves: list[Wave]
    max_parallel_per_wave: int
    warnings: list[str] = field(default_factory=list)
    unassigned: list[str] = field(default_factory=list)

    @property
    def tasks(self) -> list[ScheduleTask]:
        return [t for w in self.waves for t in w.tasks]


class Scheduler:
    def __init__(
        self,
        config: SiteConfig,
        broker: DongleBroker,
        *,
        all_bands: bool = False,
        decoder_registry=None,
    ):
        self.config = config
        self.broker = broker
        self.matcher = AntennaMatcher()
        self.all_bands = all_bands
        # Used to estimate how many dongles each band's strategy will
        # want (primary decoders + shared + power-scan sidecar). When
        # None, we fall back to suggested_decoders as a rough guide.
        self.decoder_registry = decoder_registry

    def plan(self, bands: list[BandConfig]) -> ExecutionPlan:
        usable = self.broker.registry.usable()
        warnings: list[str] = []
        unassigned: list[str] = []

        # Track per-dongle assignment count as we walk bands. Passed to
        # the matcher so equivalent dongles get spread across bands —
        # without this, a band's assignment is independent of others'
        # and we end up double-loading one dongle while equally-good
        # dongles sit idle (e.g. two whip_915 dongles, both 915 bands
        # going to the same one).
        dongle_load: dict[str, int] = {}
        scored: list[tuple[float, ScheduleTask, str | None]] = []
        for band in bands:
            candidates: list[tuple[str, object]] = []
            for d in usable:
                if not d.covers(band.center_hz):
                    continue
                candidates.append((d.id, d.antenna))
            match = self.matcher.best_pairing(
                band, candidates,
                ignore_threshold=self.all_bands,
                dongle_load=dongle_load,
            )
            task = ScheduleTask(
                band=band,
                suggested_dongle_id=match.dongle_id if match else None,
                suggested_antenna_id=match.antenna_id if match else None,
                dongles_needed=_estimate_dongle_needs(
                    band, registry=self.decoder_registry,
                ),
            )
            if match:
                task.notes.extend(match.warnings)
                dongle_load[match.dongle_id] = dongle_load.get(match.dongle_id, 0) + 1
                scored.append((match.score, task, match.dongle_id))
            else:
                note = f"no dongle covers {band.name} with suitable antenna"
                task.notes.append(note)
                warnings.append(note)
                unassigned.append(band.id)
                scored.append((0.0, task, None))

        # Sort primarily by score desc (best-fit bands first) but break
        # ties by dongles_needed desc — a band that will reserve multiple
        # slots should be placed before a same-score band that only
        # needs 1, otherwise the 1-dongle band can claim a slot that
        # the bigger band would have needed for its sidecar. Example:
        # 915_ism (score 1.0, needs 3) vs pocsag_929 (score 1.0, needs
        # 1) — if pocsag_929 is placed first on 00000003, 915_ism is
        # forced to a later wave because 00000003 is gone. Placing
        # 915_ism first reserves both whip_915 dongles in one wave.
        scored.sort(key=lambda t: (t[0], t[1].dongles_needed), reverse=True)

        waves: list[Wave] = []
        # Per-wave reserved dongle set. Each band reserves:
        #   • its primary dongle (specific id)
        #   • (dongles_needed - 1) sidecar slots from its antenna-suitable
        #     pool (any IDs not yet reserved in this wave)
        # This prevents two bands from trying to share the same antenna-
        # suitable dongle — one as a primary, another as a sidecar pool
        # slot. Without this check, e.g. pocsag_929 (primary on 00000003)
        # would be placed in the same wave as 915_ism whose rtlamr sidecar
        # also needs 00000003, and the sidecar would then fail at runtime.
        wave_reservations: list[set[str]] = []

        for _score, task, dongle_id in scored:
            if dongle_id is None:
                waves.append(Wave(index=len(waves), tasks=[task]))
                wave_reservations.append(set())
                continue

            suitable_ids = _suitable_dongle_ids(task.band, usable)
            sidecars_needed = max(0, task.dongles_needed - 1)
            # Cap effective sidecar reservations by fleet availability.
            # If the band's antenna-suitable pool is only 1 dongle, we
            # can't reserve any sidecar slots — the primary is the
            # whole pool. Without this cap, such a band would never
            # fit in any wave (it wants more suitable dongles than
            # exist) and the placement loop would bump it to its own
            # wave forever, producing the v0.5.16 "13 waves, mostly 1
            # task each" regression. Fleet shortage degrades gracefully
            # at runtime via v0.5.14 defer + v0.5.13 hard antenna check.
            fleet_sidecar_budget = max(0, len(suitable_ids) - 1)
            effective_sidecars = min(sidecars_needed, fleet_sidecar_budget)

            placed = False
            for wave, reserved in zip(waves, wave_reservations):
                if dongle_id in reserved:
                    continue  # primary dongle already claimed this wave
                free_suitable = suitable_ids - reserved - {dongle_id}
                if len(free_suitable) >= effective_sidecars:
                    wave.tasks.append(task)
                    reserved.add(dongle_id)
                    # Reserve sidecar slots from the suitable pool.
                    # Pick deterministically (sorted) so plans are
                    # repeatable and don't depend on set ordering.
                    for sidecar_id in sorted(free_suitable)[:effective_sidecars]:
                        reserved.add(sidecar_id)
                    placed = True
                    break
            if not placed:
                # New wave. Reserve primary + as many sidecar slots as
                # the fleet supports.
                new_reserved = {dongle_id}
                remaining = suitable_ids - {dongle_id}
                for sidecar_id in sorted(remaining)[:effective_sidecars]:
                    new_reserved.add(sidecar_id)
                waves.append(Wave(index=len(waves), tasks=[task]))
                wave_reservations.append(new_reserved)

        # v0.5.35: split multi-exclusive-decoder conflicts across waves.
        # Some bands (like aprs_2m with direwolf + multimon both
        # requesting exclusive dongle access) can't run all decoders
        # in a single wave when the band's antenna-suitable dongle
        # pool is smaller than the exclusive-decoder count. Rather
        # than silently failing at runtime (the v0.5.34 behavior), we
        # defer surplus decoders to later waves with spare capacity.
        #
        # This mutates `waves` and `wave_reservations` in place —
        # may mark existing tasks with `allowed_decoders` and may
        # append new waves for tasks that don't fit anywhere existing.
        self._defer_surplus_exclusive_decoders(
            waves, wave_reservations, usable
        )

        cpus = os.cpu_count() or 4
        # Decoder processes are I/O-bound (USB reads + binary parsing),
        # not CPU-pegged. Capping at cpu_budget_fraction alone leaves
        # dongles sitting idle when the user has more hardware than
        # half-CPU. The right limit is "as many as we have usable
        # dongles, capped to a sane CPU number to avoid pathological
        # contention." Users who need a tighter cap can still set
        # max_concurrent_decoders explicitly.
        n_usable = sum(1 for d in self.broker.registry.dongles if d.is_usable())
        fraction = self.config.resources.cpu_budget_fraction
        cpu_cap = max(1, int(cpus * fraction))
        if n_usable > 0:
            # Saturate dongles, but never exceed 4× cpu_cap to prevent
            # runaway when the user plugs in 12 SDRs on a Pi
            max_parallel = max(1, min(n_usable, cpu_cap * 4))
        else:
            max_parallel = cpu_cap
        if self.config.resources.max_concurrent_decoders is not None:
            max_parallel = self.config.resources.max_concurrent_decoders

        for wave in waves:
            summary = ", ".join(
                f"{t.band.id}→{t.suggested_dongle_id or 'unassigned'}"
                for t in wave.tasks
            )
            log.info(
                "wave %d (%d task%s): %s",
                wave.index,
                len(wave.tasks),
                "" if len(wave.tasks) == 1 else "s",
                summary,
            )

        # Report on dongles not used as primary assignment. Silent idle
        # dongles are confusing — the user wants to know why their
        # hardware isn't being utilized. Note: a dongle may still be
        # used as a *sidecar* (rtl_power or rtlamr via rtl_tcp) at
        # runtime; we can't predict that here, only that it didn't get
        # a primary assignment. We phrase the message accordingly.
        assigned_ids = {
            t.suggested_dongle_id for t in scored_tasks_iter(waves)
            if t.suggested_dongle_id
        }
        unused = [d for d in usable if d.id not in assigned_ids]
        for d in unused:
            reason = _explain_unused(d, bands)
            log.info(
                "dongle %s not assigned to any band as primary decoder%s "
                "(it may still be used as a sidecar for power scans or "
                "rtl_tcp-shared decoders at runtime)",
                d.id,
                f" — {reason}" if reason else "",
            )
            warnings.append(
                f"dongle {d.id} is not assigned to any band as primary"
                + (f" ({reason})" if reason else "")
            )

        return ExecutionPlan(
            waves=waves,
            max_parallel_per_wave=max_parallel,
            warnings=warnings,
            unassigned=unassigned,
        )

    def _defer_surplus_exclusive_decoders(
        self,
        waves: list[Wave],
        wave_reservations: list[set[str]],
        usable,
    ) -> None:
        """Split tasks whose band has more exclusive-access decoders
        than the current wave's antenna-suitable dongle pool can host
        concurrently. Surplus decoders are deferred to later waves with
        spare capacity (most-spare first); a new wave is appended if
        nothing existing has room.

        Mutates `waves` and `wave_reservations` in place.

        Without a decoder registry, we can't classify decoders as
        exclusive vs shared, so we return without modifying the plan.
        This matches the defensive posture of `_estimate_dongle_needs`.

        The aprs_2m example that motivated this (v0.5.34 regression
        notes): band suggests `["direwolf", "multimon"]`, both
        exclusive-access. Only one 2m-capable dongle in the fleet. At
        runtime the first to call broker.allocate() wins the dongle;
        the loser logs "no dongle available" and silently does nothing.
        With this splitter, direwolf stays in the band's primary wave
        and multimon gets a retry task in a later wave where the same
        dongle is free — both decoders run, sequentially rather than
        concurrently.
        """
        if self.decoder_registry is None:
            return

        # Snapshot indices — we iterate a copy since we may append new
        # waves mid-loop. Appended waves aren't re-examined: a new wave
        # created for a deferred decoder runs exactly the one decoder
        # we placed there, so there's nothing further to split.
        n_initial_waves = len(waves)
        for wave_idx in range(n_initial_waves):
            wave = waves[wave_idx]
            for task in list(wave.tasks):
                self._maybe_split_task(
                    task, wave_idx, waves, wave_reservations, usable
                )

    def _maybe_split_task(
        self,
        task: ScheduleTask,
        wave_idx: int,
        waves: list[Wave],
        wave_reservations: list[set[str]],
        usable,
    ) -> None:
        """Examine one task. If its band has a multi-exclusive-decoder
        conflict with wave capacity, split it: constrain the original
        task to a subset of decoders and create retry tasks for the
        rest in later waves.
        """
        if task.suggested_dongle_id is None:
            return
        if task.allowed_decoders is not None:
            # Already constrained — likely a retry task we created on
            # an earlier pass. Don't split it again.
            return

        band = task.band
        suggested = list(band.suggested_decoders)  # preserve user order

        # Classify matched decoders. Preserve suggested_decoders
        # ordering so "first listed = gets the primary dongle" is
        # deterministic and reflects user intent.
        exclusive_decoders: list[str] = []
        shared_decoders: list[str] = []
        registry_names = set(self.decoder_registry.names())
        for name in suggested:
            if name not in registry_names:
                continue
            cls = self.decoder_registry.get(name)
            if cls is None:
                continue
            caps = cls.capabilities
            if not any(
                low <= band.freq_high and high >= band.freq_low
                for low, high in caps.freq_ranges
            ):
                continue
            if caps.access_mode == AccessMode.EXCLUSIVE:
                exclusive_decoders.append(name)
            else:
                shared_decoders.append(name)

        if len(exclusive_decoders) <= 1:
            return  # no multi-exclusive conflict to worry about

        # Capacity math for this wave.
        #   suitable = all fleet dongles whose antenna covers the band
        #   reserved_by_others = reservations in this wave EXCLUDING this task's primary
        #   available_here = dongles this task could use (primary + free suitables in the wave)
        #   capacity_here = count of exclusive decoders that fit in this wave
        suitable = _suitable_dongle_ids(band, usable)
        reserved_by_others = (
            wave_reservations[wave_idx] - {task.suggested_dongle_id}
        )
        available_here = (
            suitable - reserved_by_others
        )  # includes task.suggested_dongle_id
        capacity_here = len(available_here)

        if capacity_here >= len(exclusive_decoders):
            # All fit in this wave — reserve the additional dongles so
            # other waves' planning doesn't later steal them. (The
            # initial packing only reserved the primary + sidecar
            # budget; it didn't know to reserve one slot per
            # exclusive decoder.)
            extras_to_reserve = len(exclusive_decoders) - 1
            extra_pool = sorted(
                available_here - {task.suggested_dongle_id}
            )
            for extra_id in extra_pool[:extras_to_reserve]:
                wave_reservations[wave_idx].add(extra_id)
            return

        # Doesn't fit. Keep `capacity_here` exclusives here (by
        # suggested_decoders order); defer the rest.
        keep = exclusive_decoders[:capacity_here]
        defer = exclusive_decoders[capacity_here:]

        # Shared decoders always ride with the primary task — they
        # don't block on dongle exclusivity and the band's shared
        # protocols only need decoding once per scan.
        task.allowed_decoders = set(keep) | set(shared_decoders)

        # Reserve the extra dongles for the kept exclusives.
        if len(keep) > 1:
            extras_to_reserve = len(keep) - 1
            extra_pool = sorted(
                available_here - {task.suggested_dongle_id}
            )
            for extra_id in extra_pool[:extras_to_reserve]:
                wave_reservations[wave_idx].add(extra_id)

        task.notes.append(
            f"plan split: {len(defer)} exclusive decoder(s) deferred "
            f"to later wave(s) — {','.join(defer)} "
            f"(wave {wave_idx} capacity={capacity_here}, "
            f"needed={len(exclusive_decoders)})"
        )

        for decoder_name in defer:
            self._place_deferred_decoder(
                band=band,
                decoder_name=decoder_name,
                source_wave_idx=wave_idx,
                waves=waves,
                wave_reservations=wave_reservations,
                suitable=suitable,
                suggested_antenna_id=task.suggested_antenna_id,
            )

    def _place_deferred_decoder(
        self,
        *,
        band: BandConfig,
        decoder_name: str,
        source_wave_idx: int,
        waves: list[Wave],
        wave_reservations: list[set[str]],
        suitable: set[str],
        suggested_antenna_id: str | None,
    ) -> None:
        """Find the best later wave for `decoder_name` to run on
        `band`, or append a new wave. "Best" = most spare suitable
        dongles (ties broken by earliest wave). If no later wave has
        any suitable dongle free, append a new final wave.
        """
        best_idx: int | None = None
        best_spare: int = -1
        best_dongle: str | None = None

        for idx in range(source_wave_idx + 1, len(waves)):
            free = suitable - wave_reservations[idx]
            if not free:
                continue
            spare = len(free)
            if spare > best_spare:
                best_spare = spare
                best_idx = idx
                # Deterministic pick from the free set
                best_dongle = sorted(free)[0]

        new_task = ScheduleTask(
            band=band,
            suggested_dongle_id=best_dongle,
            suggested_antenna_id=suggested_antenna_id,
            dongles_needed=1,
            allowed_decoders={decoder_name},
            notes=[
                f"plan retry: {decoder_name}@{band.id} deferred from "
                f"wave {source_wave_idx}"
            ],
        )

        if best_idx is None:
            # No later wave has any suitable dongle free. Append a new
            # wave. Pick any suitable dongle to assign; if the fleet
            # has none at all (pathological), bail without placing.
            if not suitable:
                log.warning(
                    "cannot place deferred decoder %s@%s: no suitable "
                    "dongle in fleet",
                    decoder_name,
                    band.id,
                )
                return
            default_dongle = sorted(suitable)[0]
            new_task.suggested_dongle_id = default_dongle
            new_wave = Wave(index=len(waves), tasks=[new_task])
            waves.append(new_wave)
            wave_reservations.append({default_dongle})
            log.info(
                "plan: appended wave %d for deferred %s@%s (no earlier "
                "wave had spare capacity)",
                new_wave.index,
                decoder_name,
                band.id,
            )
        else:
            waves[best_idx].tasks.append(new_task)
            wave_reservations[best_idx].add(best_dongle)  # type: ignore[arg-type]
            log.info(
                "plan: placed deferred %s@%s in wave %d (spare=%d)",
                decoder_name,
                band.id,
                best_idx,
                best_spare,
            )


def scored_tasks_iter(waves):
    """Walk all tasks across all waves (helper for assignment reporting)."""
    for w in waves:
        yield from w.tasks


def _explain_unused(dongle, bands):
    """Describe why a dongle wasn't assigned to any band as primary.

    The most common reason is antenna-frequency mismatch: e.g. a
    telescopic extended to 11 cm is tuned for ~680 MHz but if no
    enabled band falls in ~580–780 MHz, the dongle can't be used.
    This function computes which enabled bands the dongle's hardware
    COULD tune to vs which its antenna can actually receive, and
    phrases the gap in user-friendly terms.
    """
    hw_covers = [b for b in bands if dongle.covers(b.center_hz)]
    if not hw_covers:
        low_mhz = dongle.capabilities.freq_range_hz[0] / 1e6
        high_mhz = dongle.capabilities.freq_range_hz[1] / 1e6
        return (
            f"no enabled band falls in the dongle's tuning range "
            f"({low_mhz:.0f}-{high_mhz:.0f} MHz)"
        )
    if dongle.antenna is None:
        return "no antenna declared in config"
    ant_covers = [
        b for b in hw_covers if dongle.antenna.covers(b.center_hz)
    ]
    if not ant_covers:
        ant_low_mhz = dongle.antenna.usable_range[0] / 1e6
        ant_high_mhz = dongle.antenna.usable_range[1] / 1e6
        return (
            f"antenna {dongle.antenna.id} covers {ant_low_mhz:.0f}-"
            f"{ant_high_mhz:.0f} MHz but no enabled band falls in that "
            f"range — re-run setup to assign a different antenna, or "
            f"use --all-bands to force-assign despite the mismatch"
        )
    # Has candidates but lost every band to load-balancing or
    # score-based tie-breaking (an equivalent or better dongle covers
    # the same bands). Not a config problem; just "nothing left to do."
    return (
        f"all enabled bands it can cover were assigned to other "
        f"equally-capable dongles"
    )
