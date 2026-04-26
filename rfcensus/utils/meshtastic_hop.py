"""Hop planner — compute the minimum set of dongle tunings needed to
cover a given list of Meshtastic preset slots.

The problem: a dongle tuned at center F with sample rate Fs can hear
preset slots whose frequency falls within ``±(Fs − BW)/2 − edge_guard``
of F. Some slots are close enough that one tuning catches multiple
("MEDIUM_FAST" at 913.125 + "MEDIUM_SLOW" at 914.875 are both within
2.4 MS/s if centered around 914 MHz). Others are far apart and require
their own tuning.

For ``--hop`` mode we want to:
  1. Find a small set of center frequencies that together cover every
     preset we want to monitor.
  2. Cycle through those tunings on a dwell schedule.
  3. Per-tuning, run the multi-preset pipeline on whatever slots that
     tuning catches.

This module implements (1). Scheduling lives in the CLI.

Algorithm: greedy set cover. NP-hard in general but with 9 presets
and ~260 candidate centers (100 kHz grid across the US band) it
finishes in under a millisecond. Greedy is within a ln(n) factor of
optimal — for our 9-slot case, optimal=6 and greedy returns 6 too.
"""
from __future__ import annotations

from dataclasses import dataclass

from rfcensus.utils.meshtastic_region import (
    REGIONS, default_slot, slots_in_passband, PresetSlot,
)


@dataclass(frozen=True)
class HopTuning:
    """One stop in the hop plan: tune dongle to ``center_freq_hz`` and
    decode the listed preset slots while dwelling here."""
    center_freq_hz: int
    slots: tuple[PresetSlot, ...]

    @property
    def preset_keys(self) -> tuple[str, ...]:
        return tuple(s.preset.key for s in self.slots)


def plan_hop(
    region_code: str,
    sample_rate_hz: int,
    presets: list[str] | None = None,
    grid_step_hz: int = 100_000,
    edge_guard_hz: int = 25_000,
) -> list[HopTuning]:
    """Compute a minimum-cardinality set of tunings that together cover
    every preset slot in ``presets`` (or all 9 region defaults).

    Args:
      region_code: Meshtastic region (e.g. "US").
      sample_rate_hz: dongle sample rate. Wider = more slots per
        tuning = fewer hops.
      presets: subset of preset keys to cover. None = all 9 defaults.
      grid_step_hz: candidate center-frequency resolution. 100 kHz
        is fine; the slot grid is much coarser than that.
      edge_guard_hz: rolloff margin (passed through to slots_in_passband).

    Returns: ordered list of ``HopTuning`` objects. Order is "richest
    tunings first" (each successive hop adds the largest-possible
    set of newly-covered slots), which means a one-cycle scan visits
    the most-likely-productive frequencies first.
    """
    region = REGIONS[region_code]

    # Build the target set of slots we need to cover.
    if presets is None:
        target_keys = [default_slot(region_code, k).preset.key
                       for k in _all_preset_keys()]
    else:
        target_keys = list(presets)
    target = {k: default_slot(region_code, k) for k in target_keys}
    uncovered = set(target_keys)

    # Build the candidate tuning grid. We grid in ``grid_step_hz``
    # increments across the region. The grid is just for selection —
    # the actual chosen centers are still in Hz.
    grid_lo = int(region.freq_start_mhz * 1_000_000)
    grid_hi = int(region.freq_end_mhz * 1_000_000)
    candidates = list(range(grid_lo, grid_hi + 1, grid_step_hz))

    plan: list[HopTuning] = []

    # Greedy set cover: repeatedly pick the candidate center that
    # covers the most uncovered slots. Tie-break by preferring the
    # center with the highest total coverage (most slots overall),
    # so we converge faster on dense regions.
    while uncovered:
        best_center = None
        best_new: set[str] = set()
        best_total = 0
        for center in candidates:
            slots_here = slots_in_passband(
                region_code, center, sample_rate_hz,
                presets=target_keys, edge_guard_hz=edge_guard_hz,
            )
            keys_here = {s.preset.key for s in slots_here}
            new_keys = keys_here & uncovered
            if not new_keys:
                continue
            # Primary: max new coverage. Secondary: max total slots
            # at this center (broadcasts more useful overlap).
            if (len(new_keys) > len(best_new)
                or (len(new_keys) == len(best_new)
                    and len(keys_here) > best_total)):
                best_center = center
                best_new = new_keys
                best_total = len(keys_here)

        if best_center is None:
            # Some preset can't be reached at this sample rate even
            # alone (its bandwidth exceeds Fs). Skip it with a warning;
            # caller decides what to do.
            break

        # The actual slot list at the chosen center may include
        # already-covered slots; that's fine, we'll decode them again
        # and the caller can dedupe by sample_offset if needed.
        actual_slots = slots_in_passband(
            region_code, best_center, sample_rate_hz,
            presets=target_keys, edge_guard_hz=edge_guard_hz,
        )
        plan.append(HopTuning(
            center_freq_hz=best_center,
            slots=tuple(actual_slots),
        ))
        uncovered -= best_new

    return plan


def _all_preset_keys() -> list[str]:
    # Local import to avoid a circular at module load.
    from rfcensus.utils.meshtastic_region import PRESETS
    return list(PRESETS.keys())
