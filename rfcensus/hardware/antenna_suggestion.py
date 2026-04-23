"""Fleet-aware antenna suggestion.

Used by:
  • setup wizard, when a user adds a new dongle and isn't sure what
    antenna to assign — we look at what their other dongles cover and
    recommend something that fills the biggest coverage gap
  • antenna-plan command, which optimizes antenna assignment across
    all dongles at once (this module provides the per-dongle reasoning)

The core insight: an antenna recommendation in isolation is just a band
match. An antenna recommendation in the context of a fleet should
prioritize **filling gaps** — there's no point recommending another
915 MHz whip if three other dongles already cover 915.
"""

from __future__ import annotations

from dataclasses import dataclass

from rfcensus.config.schema import BandConfig
from rfcensus.hardware.antenna import Antenna
from rfcensus.hardware.dongle import Dongle


@dataclass
class GapSuggestion:
    """An antenna recommendation for a given dongle, fleet-aware."""

    antenna_id: str | None     # id from catalog, or None if no match
    rationale: str             # human-readable explanation
    bands_covered: list[str]   # band ids this antenna would unlock
    is_quarter_wave_fallback: bool = False
    fallback_freq_mhz: float | None = None  # for telescopic fallback
    fallback_length_cm: float | None = None
    buy_suggestion: str | None = None  # "consider buying X for ..."


def suggest_for_new_dongle(
    new_dongle: Dongle,
    other_dongles: list[Dongle],
    enabled_bands: list[BandConfig],
    available_antennas: list[Antenna],
) -> GapSuggestion:
    """Recommend an antenna for `new_dongle` that fills the biggest
    coverage gap left by `other_dongles`.

    Decision tree:
      1. Compute which enabled bands have NO good coverage from
         other_dongles (no antenna match score >= 0.7)
      2. Filter to bands that new_dongle can cover (frequency-wise)
      3. Group remaining missing bands by frequency cluster
      4. Pick the largest cluster
      5. Find an antenna in the catalog that covers it well
      6. If no catalog match: compute a quarter-wave whip recommendation
         with the physical length, since telescopic whips are common
      7. If everything is already covered: recommend a generic small
         whip and note that the fleet is already in good shape
    """
    # Find bands not well covered by other dongles AND that new_dongle
    # could help with frequency-wise
    uncovered_bands: list[BandConfig] = []
    for band in enabled_bands:
        if not new_dongle.covers(band.center_hz):
            continue  # this dongle can't help with this band anyway
        best_score = 0.0
        for d in other_dongles:
            if not d.is_usable() or not d.covers(band.center_hz) or not d.antenna:
                continue
            score = d.antenna.suitability_for_band(band)
            best_score = max(best_score, score)
        if best_score < 0.7:
            uncovered_bands.append(band)

    if not uncovered_bands:
        return GapSuggestion(
            antenna_id="whip_generic_small",
            rationale=(
                "Your other dongles already cover all enabled bands well. "
                "A generic small whip on this dongle gives you flexibility "
                "to retune later without buying anything new."
            ),
            bands_covered=[],
        )

    # Find the antenna in the catalog that covers the most uncovered bands.
    # Tiebreak: total suitability score across the bands it covers.
    best_antenna_id: str | None = None
    best_antenna_score = 0.0
    best_antenna_bands: list[str] = []

    for ant in available_antennas:
        covered_here = []
        score_sum = 0.0
        for band in uncovered_bands:
            score = ant.suitability_for_band(band)
            if score >= 0.7:
                covered_here.append(band.id)
                score_sum += score
        if len(covered_here) > len(best_antenna_bands) or (
            len(covered_here) == len(best_antenna_bands)
            and score_sum > best_antenna_score
        ):
            best_antenna_id = ant.id
            best_antenna_score = score_sum
            best_antenna_bands = covered_here

    if best_antenna_id and best_antenna_bands:
        gap_freqs_mhz = sorted({b.center_hz / 1e6 for b in uncovered_bands})
        return GapSuggestion(
            antenna_id=best_antenna_id,
            rationale=(
                f"Your fleet is missing coverage for "
                f"{len(uncovered_bands)} band(s) "
                f"({_summarize_freqs(gap_freqs_mhz)}). "
                f"This antenna covers {len(best_antenna_bands)} of them well."
            ),
            bands_covered=best_antenna_bands,
        )

    # No catalog match — fall back to a quarter-wave whip recommendation
    # for the largest gap cluster
    cluster = _largest_freq_cluster(uncovered_bands)
    target_mhz = sum(b.center_hz / 1e6 for b in cluster) / len(cluster)
    quarter_wave_cm = (29979.2458 / target_mhz) / 4

    return GapSuggestion(
        antenna_id=None,
        rationale=(
            f"No antenna in the catalog matches your gap at "
            f"~{target_mhz:.0f} MHz."
        ),
        bands_covered=[b.id for b in cluster],
        is_quarter_wave_fallback=True,
        fallback_freq_mhz=target_mhz,
        fallback_length_cm=quarter_wave_cm,
        buy_suggestion=(
            f"For best results, get an antenna tuned to {target_mhz:.0f} MHz "
            f"(quarter-wave length: {quarter_wave_cm:.1f} cm). A telescopic "
            f"whip extended to that length will work as a starting point."
        ),
    )


def _summarize_freqs(freqs_mhz: list[float], max_show: int = 3) -> str:
    """Human-readable frequency list summary."""
    if not freqs_mhz:
        return "(none)"
    if len(freqs_mhz) <= max_show:
        return ", ".join(f"{f:.0f} MHz" for f in freqs_mhz)
    shown = ", ".join(f"{f:.0f} MHz" for f in freqs_mhz[:max_show])
    return f"{shown} and {len(freqs_mhz) - max_show} more"


def _largest_freq_cluster(
    bands: list[BandConfig], cluster_width_pct: float = 0.20,
) -> list[BandConfig]:
    """Group bands into frequency clusters and return the largest one.

    Two bands are in the same cluster if their center frequencies are
    within cluster_width_pct of each other. This is a rough proxy for
    "could be covered by the same antenna" — a 20% cluster width matches
    typical resonant antenna bandwidth.
    """
    if not bands:
        return []
    by_freq = sorted(bands, key=lambda b: b.center_hz)
    clusters: list[list[BandConfig]] = [[by_freq[0]]]
    for b in by_freq[1:]:
        last = clusters[-1][-1]
        ratio = b.center_hz / last.center_hz
        if 1 - cluster_width_pct <= ratio <= 1 + cluster_width_pct:
            clusters[-1].append(b)
        else:
            clusters.append([b])
    return max(clusters, key=len)
