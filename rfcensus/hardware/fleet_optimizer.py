"""Fleet antenna optimizer.

Given a set of dongles, an antenna catalog, and a list of enabled
bands, find an assignment of antenna→dongle that maximizes coverage.

The optimization is small (typically 3-8 dongles, ~10 antennas) so we
brute-force the assignment search. We also identify shopping
suggestions: antennas that aren't in the catalog but would meaningfully
improve coverage if added.

Used by:
  • `rfcensus suggest antennas` — print recommended assignment, optionally
    apply to site.toml
  • Programmatically by setup wizard for batch suggestions

Decision criterion (per the design discussion):
  Maximize the number of bands with score >= 0.7 (well-covered).
  Tiebreak on the sum of match scores across all bands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from rfcensus.config.schema import BandConfig
from rfcensus.hardware.antenna import Antenna
from rfcensus.hardware.dongle import Dongle


# Threshold for "well covered" — matches the threshold used by
# AntennaMatcher and compute_coverage.
WELL_COVERED_THRESHOLD = 0.7


@dataclass
class FleetPlan:
    """Output of the fleet optimizer."""

    # dongle_id → chosen antenna id (or None if no antenna in catalog
    # would help this dongle for any enabled band)
    assignments: dict[str, str | None] = field(default_factory=dict)
    # Bands that nothing in the proposed plan covers well — surfaced as
    # gaps even after optimization
    uncovered_bands: list[str] = field(default_factory=list)
    # Total number of bands the plan covers well
    well_covered_count: int = 0
    # Sum of match scores across all (dongle, antenna, covered band)
    total_score: float = 0.0
    # "Buy this antenna to fill X bands" suggestions — bands not coverable
    # by any antenna currently in catalog
    shopping_suggestions: list[ShoppingSuggestion] = field(default_factory=list)


@dataclass
class ShoppingSuggestion:
    """A buy-this hint for bands no current antenna can cover."""

    target_freq_mhz: float
    quarter_wave_length_cm: float
    bands_unlocked: list[str]
    rationale: str


@dataclass
class PlanDiff:
    """Diff between current assignments and the optimized plan."""

    # dongle_id → (current_ant, proposed_ant)
    changes: list[tuple[str, str | None, str | None]] = field(default_factory=list)
    # dongle_id assignments that don't change
    unchanged: list[str] = field(default_factory=list)


def optimize_fleet(
    dongles: list[Dongle],
    enabled_bands: list[BandConfig],
    available_antennas: list[Antenna],
    pinned_freqs: dict[str, int] | None = None,
) -> FleetPlan:
    """Find the best antenna assignment for the given dongles.

    Brute-force search over the product of (dongle × antenna). For
    typical SDR fleets (3-8 dongles, ~10 antennas) this is well under
    100M evaluations. We add a "no antenna" slot to allow leaving a
    dongle unassigned when no antenna helps.

    v0.6.0 — `pinned_freqs` (dongle_id → freq_hz) constrains the
    optimizer for pinned dongles: their antenna MUST cover the pin
    frequency at >= WELL_COVERED_THRESHOLD. Without this, the
    optimizer might suggest swapping the antenna on a pinned dongle
    to one that's better for some unrelated band, breaking the pin.
    """
    if not dongles:
        return FleetPlan()

    usable_dongles = [d for d in dongles if d.is_usable()]
    if not usable_dongles:
        return FleetPlan()

    pinned_freqs = pinned_freqs or {}

    # Per-dongle, only consider antennas that could plausibly cover at
    # least one enabled band (filters huge fleets quickly). For pinned
    # dongles, additionally require the antenna to cover the pin freq
    # well — pinning is the user's explicit intent and the optimizer
    # must not break it.
    per_dongle_options: list[list[Antenna | None]] = []
    for d in usable_dongles:
        pin_freq = pinned_freqs.get(d.id)
        if pin_freq is not None:
            # Pinned: antenna is MANDATORY (no None option). The pin's
            # antenna must cover the pin freq at well-covered level.
            # If the catalog has nothing suitable, the pinned dongle
            # ends up with no acceptable assignment — surface as a
            # coverage gap rather than silently swapping to None.
            plausible: list[Antenna | None] = []
            from rfcensus.config.schema import BandConfig as _BC
            synth = _BC(
                id=f"_pin_{d.id}",
                name="pin",
                freq_low=pin_freq - 1000,
                freq_high=pin_freq + 1000,
            )
            for ant in available_antennas:
                if not ant.covers(pin_freq):
                    continue
                if ant.suitability_for_band(synth) >= WELL_COVERED_THRESHOLD:
                    plausible.append(ant)
            if not plausible:
                # No catalog antenna can satisfy the pin — record a
                # None placeholder so the optimizer doesn't crash; the
                # downstream uncovered_bands logic will surface this.
                plausible.append(None)
        else:
            # Unpinned: original logic
            plausible = [None]
            for ant in available_antennas:
                for band in enabled_bands:
                    if not d.covers(band.center_hz):
                        continue
                    if ant.suitability_for_band(band) >= 0.3:
                        plausible.append(ant)
                        break  # at least one band works; include this antenna
        per_dongle_options.append(plausible)

    # Brute force the assignment. With per-dongle filtering most fleets
    # have <100 options per dongle; 8 dongles × 10 options = 10^8 worst
    # case. We could add memoization but it's not worth it at this scale.
    best_score_count = -1
    best_total_score = -1.0
    best_assignment: list[Antenna | None] = []

    for combo in product(*per_dongle_options):
        score_count, total_score = _evaluate(combo, usable_dongles, enabled_bands)
        if score_count > best_score_count or (
            score_count == best_score_count and total_score > best_total_score
        ):
            best_score_count = score_count
            best_total_score = total_score
            best_assignment = list(combo)

    # Identify which bands the chosen plan still doesn't cover
    uncovered = []
    for band in enabled_bands:
        best_for_band = 0.0
        for d, ant in zip(usable_dongles, best_assignment):
            if ant is None or not d.covers(band.center_hz):
                continue
            best_for_band = max(best_for_band, ant.suitability_for_band(band))
        if best_for_band < WELL_COVERED_THRESHOLD:
            uncovered.append(band.id)

    # Generate shopping suggestions for the uncovered bands
    shopping = _generate_shopping_suggestions(
        uncovered_bands=[b for b in enabled_bands if b.id in uncovered],
        max_suggestions=3,
    )

    return FleetPlan(
        assignments={
            d.id: (ant.id if ant else None)
            for d, ant in zip(usable_dongles, best_assignment)
        },
        uncovered_bands=uncovered,
        well_covered_count=best_score_count,
        total_score=best_total_score,
        shopping_suggestions=shopping,
    )


def _evaluate(
    assignment: tuple,
    dongles: list[Dongle],
    bands: list[BandConfig],
) -> tuple[int, float]:
    """Return (well_covered_count, total_score) for a candidate assignment."""
    score_count = 0
    total_score = 0.0
    for band in bands:
        best_for_band = 0.0
        for d, ant in zip(dongles, assignment):
            if ant is None or not d.covers(band.center_hz):
                continue
            score = ant.suitability_for_band(band)
            best_for_band = max(best_for_band, score)
        if best_for_band >= WELL_COVERED_THRESHOLD:
            score_count += 1
        total_score += best_for_band
    return score_count, total_score


def _generate_shopping_suggestions(
    uncovered_bands: list[BandConfig], max_suggestions: int = 3,
) -> list[ShoppingSuggestion]:
    """Group uncovered bands by frequency cluster, recommend a tuned
    antenna per cluster (with quarter-wave length).

    Returns at most max_suggestions, sorted by number of bands unlocked.
    """
    if not uncovered_bands:
        return []

    # Cluster bands within 20% frequency band of each other
    clusters: list[list[BandConfig]] = []
    by_freq = sorted(uncovered_bands, key=lambda b: b.center_hz)
    for b in by_freq:
        added = False
        for cl in clusters:
            ratio = b.center_hz / cl[0].center_hz
            if 0.80 <= ratio <= 1.20:
                cl.append(b)
                added = True
                break
        if not added:
            clusters.append([b])

    # Sort clusters by size descending
    clusters.sort(key=len, reverse=True)

    suggestions: list[ShoppingSuggestion] = []
    for cl in clusters[:max_suggestions]:
        target_mhz = sum(b.center_hz / 1e6 for b in cl) / len(cl)
        quarter_wave_cm = (29979.2458 / target_mhz) / 4
        suggestions.append(ShoppingSuggestion(
            target_freq_mhz=target_mhz,
            quarter_wave_length_cm=quarter_wave_cm,
            bands_unlocked=[b.id for b in cl],
            rationale=(
                f"An antenna tuned to ~{target_mhz:.0f} MHz "
                f"(quarter-wave length: {quarter_wave_cm:.1f} cm) would "
                f"cover {len(cl)} currently-uncovered band(s)."
            ),
        ))
    return suggestions


def diff_against_current(
    plan: FleetPlan, current_assignments: dict[str, str | None],
) -> PlanDiff:
    """Compare the optimizer's plan to the user's current assignments."""
    diff = PlanDiff()
    for dongle_id, proposed in plan.assignments.items():
        current = current_assignments.get(dongle_id)
        if current == proposed:
            diff.unchanged.append(dongle_id)
        else:
            diff.changes.append((dongle_id, current, proposed))
    return diff
