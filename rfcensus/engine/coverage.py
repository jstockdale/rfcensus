"""Antenna coverage diagnostics for the user.

When the scheduler can't find a usable antenna for a band, it silently
drops the band today. That's correct behavior (we don't want to produce
false decodes from severely detuned antennas) but it's a bad user
experience — users see "0 decodes on AIS" without knowing whether AIS
was even attempted.

This module computes a coverage report from a set of bands and the
attached hardware, and formats it for human consumption. Used by:

  • scan / inventory: shown at the top of the run, before any RF work
  • doctor: shown as a section, preventive

The report:
  • Counts how many bands have a usable antenna match
  • Lists missing bands with frequency and human-readable name
  • Groups missing bands by frequency range and suggests antenna types
"""

from __future__ import annotations

from dataclasses import dataclass

from rfcensus.config.schema import BandConfig
from rfcensus.hardware.antenna import AntennaMatcher
from rfcensus.hardware.dongle import Dongle


@dataclass
class BandCoverage:
    """Per-band coverage assessment."""

    band: BandConfig
    has_match: bool
    score: float = 0.0
    matched_dongle_id: str | None = None
    matched_antenna_id: str | None = None


@dataclass
class CoverageReport:
    """Aggregated coverage report for an enabled band list."""

    matched: list[BandCoverage]
    missing: list[BandCoverage]
    suggestions: list[str]

    @property
    def total(self) -> int:
        return len(self.matched) + len(self.missing)

    @property
    def has_gaps(self) -> bool:
        return len(self.missing) > 0


def compute_coverage(
    bands: list[BandConfig],
    dongles: list[Dongle],
    matcher: AntennaMatcher | None = None,
) -> CoverageReport:
    """Run the matcher against each band and classify by coverage.

    Always uses the strict (default-threshold) matcher — the report is
    about what would happen WITHOUT --all-bands, so the user can decide
    whether to enable it.
    """
    matcher = matcher or AntennaMatcher()
    matched: list[BandCoverage] = []
    missing: list[BandCoverage] = []

    for band in bands:
        candidates: list[tuple[str, object]] = []
        for d in dongles:
            if not d.is_usable():
                continue
            if not d.covers(band.center_hz):
                continue
            candidates.append((d.id, d.antenna))
        match = matcher.best_pairing(band, candidates)
        if match is not None:
            matched.append(BandCoverage(
                band=band,
                has_match=True,
                score=match.score,
                matched_dongle_id=match.dongle_id,
                matched_antenna_id=match.antenna_id,
            ))
        else:
            missing.append(BandCoverage(band=band, has_match=False))

    suggestions = _suggest_antennas(missing)
    return CoverageReport(matched=matched, missing=missing, suggestions=suggestions)


# ──────────────────────────────────────────────────────────────────
# Antenna suggestions: group missing bands by frequency range
# ──────────────────────────────────────────────────────────────────


# Frequency range → human-readable antenna recommendation. Order matters:
# the first matching range wins, so put more-specific ranges first.
_RANGE_RECOMMENDATIONS: list[tuple[int, int, str]] = [
    (
        25_000_000, 88_000_000,
        "an HF/lower-VHF antenna (long-wire, fan dipole, or wideband discone). "
        "RTL-SDR direct-sampling mode also helps below 30 MHz",
    ),
    (
        88_000_000, 174_000_000,
        "a VHF antenna covering 144-174 MHz (a 2m vertical, discone, or "
        "telescopic whip extended to ~50 cm)",
    ),
    (
        300_000_000, 400_000_000,
        "a 315 MHz quarter-wave whip (~24 cm) or a wideband UHF antenna",
    ),
    (
        400_000_000, 520_000_000,
        "a UHF antenna covering 400-520 MHz (a 70cm whip ~17 cm long, "
        "or a 70cm-tuned vertical)",
    ),
    (
        700_000_000, 1_000_000_000,
        "an 800/900 MHz antenna (a magmount UHF cellular antenna works "
        "well, or a quarter-wave whip ~8 cm long)",
    ),
    (
        1_050_000_000, 1_150_000_000,
        "a 1090 MHz dipole or a filtered LNA combo — generic whips lose "
        "significant range above 1 GHz",
    ),
    (
        1_700_000_000, 6_000_000_000,
        "a HackRF or AirSpy with a 2.4 GHz patch or wideband antenna — "
        "RTL-SDRs cannot tune above ~1.7 GHz",
    ),
]


def _suggest_antennas(missing: list[BandCoverage]) -> list[str]:
    """Group missing bands by frequency range, suggest one antenna per group."""
    if not missing:
        return []

    # Group missing bands by which recommendation range they fall into
    grouped: dict[tuple[int, int, str], list[BandCoverage]] = {}
    other: list[BandCoverage] = []
    for cov in missing:
        center = cov.band.center_hz
        for low, high, rec in _RANGE_RECOMMENDATIONS:
            if low <= center <= high:
                key = (low, high, rec)
                grouped.setdefault(key, []).append(cov)
                break
        else:
            other.append(cov)

    suggestions: list[str] = []
    for (low, high, rec), bands_in_range in sorted(grouped.items()):
        n = len(bands_in_range)
        plural = "band" if n == 1 else "bands"
        suggestions.append(
            f"{n} {plural} in {_human_freq(low)}-{_human_freq(high)}: add {rec}"
        )
    if other:
        suggestions.append(
            f"{len(other)} band(s) at unusual frequencies — see list above"
        )
    return suggestions


def _human_freq(hz: int) -> str:
    if hz >= 1_000_000_000:
        return f"{hz / 1_000_000_000:.1f} GHz".replace(".0 ", " ")
    if hz >= 1_000_000:
        return f"{hz / 1_000_000:.0f} MHz"
    return f"{hz / 1000:.0f} kHz"


# ──────────────────────────────────────────────────────────────────
# Console rendering
# ──────────────────────────────────────────────────────────────────


def render_coverage_report(
    report: CoverageReport, *, all_bands_flag_used: bool = False
) -> list[str]:
    """Build a list of console lines describing the coverage report.

    Returns empty list if there's nothing worth reporting (everything
    covered AND --all-bands not in play).
    """
    if not report.has_gaps and not all_bands_flag_used:
        return []

    lines: list[str] = []
    lines.append("─── Antenna coverage ───")
    lines.append("")

    matched_n = len(report.matched)
    total_n = report.total
    if matched_n == total_n:
        lines.append(f"  ✓ All {total_n} enabled bands have a usable antenna match")
        return lines

    lines.append(
        f"  ✓ {matched_n} of {total_n} enabled bands have a usable antenna match"
    )

    if all_bands_flag_used:
        # Distinguish forced (sub-threshold) matches when --all-bands is on.
        # Today we don't have that distinction in the report; this is a
        # placeholder for when we thread the all_bands flag through.
        forced = [c for c in report.matched if c.score < 0.3]
        if forced:
            lines.append(
                f"  ⚠ {len(forced)} of those {matched_n} are severely detuned "
                f"(--all-bands is on; expect unreliable decodes from these)"
            )

    if report.missing:
        n = len(report.missing)
        lines.append(f"  ⚠ {n} bands cannot be scanned with current antennas:")
        lines.append("")
        # Two-column-ish table: id, freq, name
        max_id = max(len(c.band.id) for c in report.missing)
        for cov in report.missing:
            freq = _human_freq(cov.band.center_hz)
            lines.append(
                f"    {cov.band.id:<{max_id}}  {freq:>8}   {cov.band.name}"
            )
        lines.append("")

        if report.suggestions:
            lines.append("  To enable these bands:")
            for sug in report.suggestions:
                lines.append(f"    • {sug}")
            lines.append(
                "    • Or re-run with --all-bands to attempt anyway "
                "(severely detuned, expect unreliable decodes)"
            )
            lines.append("")

        lines.append(
            f"  Continuing with {matched_n} of {total_n} bands. "
            f"Press Ctrl-C to abort."
        )
        lines.append("")

    return lines
