"""Tests for the antenna-coverage report and --all-bands flag.

Covers:
  • compute_coverage classifies bands correctly (matched vs missing)
  • _suggest_antennas groups missing bands by frequency range
  • render_coverage_report produces output (and stays silent when
    everything is fine)
  • AntennaMatcher.best_pairing(ignore_threshold=True) accepts
    sub-threshold matches with a warning
  • The 0.0-score floor is always respected (an antenna that doesn't
    physically cover the frequency cannot receive it, even with
    --all-bands)
"""

from __future__ import annotations

import pytest

from rfcensus.config.schema import AntennaConfig, BandConfig
from rfcensus.engine.coverage import (
    BandCoverage,
    CoverageReport,
    _suggest_antennas,
    compute_coverage,
    render_coverage_report,
)
from rfcensus.hardware.antenna import Antenna, AntennaMatcher
from rfcensus.hardware.dongle import Dongle, DongleCapabilities, DongleStatus


def _caps() -> DongleCapabilities:
    return DongleCapabilities(
        freq_range_hz=(24_000_000, 1_700_000_000),
        max_sample_rate=2_400_000,
        bits_per_sample=8,
        bias_tee_capable=False,
        tcxo_ppm=10.0,
    )


def _dongle_with_antenna(idx: int, antenna: Antenna | None) -> Dongle:
    d = Dongle(
        id=f"rtl-{idx}", serial=f"0000000{idx}", model="rtlsdr_generic",
        driver="rtlsdr", capabilities=_caps(), status=DongleStatus.HEALTHY,
        driver_index=idx,
    )
    d.antenna = antenna
    return d


def _whip(id: str, resonant_mhz: float) -> Antenna:
    f = int(resonant_mhz * 1_000_000)
    return Antenna.from_config(AntennaConfig(
        id=id, name=id,
        antenna_type="whip", resonant_freq_hz=f,
        usable_range=(int(f * 0.85), int(f * 1.15)),
    ))


def _band(id: str, center_mhz: float, name: str = None) -> BandConfig:
    f = int(center_mhz * 1_000_000)
    return BandConfig(
        id=id, name=name or id,
        freq_low=int(f * 0.99),
        freq_high=int(f * 1.01),
    )


# ──────────────────────────────────────────────────────────────────
# AntennaMatcher.ignore_threshold
# ──────────────────────────────────────────────────────────────────


class TestIgnoreThreshold:
    def test_default_drops_below_threshold(self):
        matcher = AntennaMatcher(threshold=0.3)
        whip_915 = _whip("whip_915", 915)
        band = _band("ais", 162)  # way out of range for whip_915
        match = matcher.best_pairing(band, [("rtl-0", whip_915)])
        assert match is None

    def test_ignore_threshold_still_returns_none_for_zero_coverage(self):
        """Hard floor: an antenna whose usable_range doesn't include the
        frequency cannot receive it, even with --all-bands. Score 0.0 means
        physical impossibility, not just bad reception."""
        matcher = AntennaMatcher(threshold=0.3)
        whip_915 = _whip("whip_915", 915)  # usable ~778-1052 MHz
        band = _band("ais", 162)  # not in usable range
        match = matcher.best_pairing(band, [("rtl-0", whip_915)], ignore_threshold=True)
        assert match is None

    def test_ignore_threshold_accepts_in_range_low_score(self):
        """If antenna physically covers the frequency but at low score,
        ignore_threshold should accept it with a warning."""
        matcher = AntennaMatcher(threshold=0.3)
        # Construct a wide-range antenna with low resonance match
        ant = Antenna.from_config(AntennaConfig(
            id="wide", name="wide",
            antenna_type="whip",
            resonant_freq_hz=915_000_000,
            usable_range=(100_000_000, 1_000_000_000),  # very wide
        ))
        band = _band("ais", 162)
        match = matcher.best_pairing(band, [("rtl-0", ant)], ignore_threshold=True)
        assert match is not None
        assert match.score < 0.3
        # Should have a "severely detuned" warning
        assert any("severely detuned" in w for w in match.warnings)

    def test_default_threshold_unchanged_with_ignore_false(self):
        matcher = AntennaMatcher(threshold=0.3)
        whip_915 = _whip("whip_915", 915)
        band = _band("915_ism", 915)
        match = matcher.best_pairing(band, [("rtl-0", whip_915)], ignore_threshold=False)
        assert match is not None
        assert match.score >= 0.9


# ──────────────────────────────────────────────────────────────────
# compute_coverage
# ──────────────────────────────────────────────────────────────────


class TestComputeCoverage:
    def test_all_bands_matched(self):
        d = _dongle_with_antenna(0, _whip("whip_915", 915))
        bands = [_band("915_ism", 915)]
        report = compute_coverage(bands, [d])
        assert len(report.matched) == 1
        assert len(report.missing) == 0
        assert not report.has_gaps

    def test_all_bands_missing(self):
        # John's situation: 915/433/315 whips, looking at VHF
        dongles = [
            _dongle_with_antenna(0, _whip("whip_915", 915)),
            _dongle_with_antenna(1, _whip("whip_433", 433)),
            _dongle_with_antenna(2, _whip("whip_315", 315)),
        ]
        bands = [
            _band("aprs_2m", 144, "2m amateur"),
            _band("ais", 162, "AIS"),
        ]
        report = compute_coverage(bands, dongles)
        assert len(report.matched) == 0
        assert len(report.missing) == 2
        assert report.has_gaps
        assert report.suggestions  # should suggest a VHF antenna

    def test_mixed_match_and_miss(self):
        dongles = [_dongle_with_antenna(0, _whip("whip_915", 915))]
        bands = [
            _band("915_ism", 915),
            _band("aprs_2m", 144),
        ]
        report = compute_coverage(bands, dongles)
        assert len(report.matched) == 1
        assert len(report.missing) == 1
        assert report.matched[0].band.id == "915_ism"
        assert report.missing[0].band.id == "aprs_2m"

    def test_unusable_dongles_ignored(self):
        d = _dongle_with_antenna(0, _whip("whip_915", 915))
        d.status = DongleStatus.FAILED  # not usable
        report = compute_coverage([_band("915_ism", 915)], [d])
        assert len(report.missing) == 1


# ──────────────────────────────────────────────────────────────────
# Antenna suggestions
# ──────────────────────────────────────────────────────────────────


class TestSuggestAntennas:
    def test_no_missing_no_suggestions(self):
        assert _suggest_antennas([]) == []

    def test_groups_vhf_bands_into_one_suggestion(self):
        """John's case: 5 VHF bands missing should produce 1 grouped suggestion."""
        missing = [
            BandCoverage(band=_band(name, freq, name), has_match=False)
            for name, freq in [
                ("aprs_2m", 144),
                ("marine_vhf", 157),
                ("ais", 162),
                ("nws_weather", 162.4),
                ("business_vhf", 155),
            ]
        ]
        suggestions = _suggest_antennas(missing)
        assert len(suggestions) == 1
        assert "5 bands" in suggestions[0]
        assert "VHF" in suggestions[0] or "144" in suggestions[0]

    def test_separate_groups_for_different_ranges(self):
        missing = [
            BandCoverage(band=_band("aprs", 144), has_match=False),
            BandCoverage(band=_band("adsb", 1090), has_match=False),
        ]
        suggestions = _suggest_antennas(missing)
        assert len(suggestions) == 2  # one per range


# ──────────────────────────────────────────────────────────────────
# render_coverage_report
# ──────────────────────────────────────────────────────────────────


class TestRenderCoverageReport:
    def test_silent_when_no_gaps_and_no_flag(self):
        """Don't add noise when everything is fine."""
        d = _dongle_with_antenna(0, _whip("whip_915", 915))
        report = compute_coverage([_band("915_ism", 915)], [d])
        lines = render_coverage_report(report, all_bands_flag_used=False)
        assert lines == []

    def test_shows_summary_when_all_bands_used_even_without_gaps(self):
        d = _dongle_with_antenna(0, _whip("whip_915", 915))
        report = compute_coverage([_band("915_ism", 915)], [d])
        lines = render_coverage_report(report, all_bands_flag_used=True)
        assert any("Antenna coverage" in line for line in lines)

    def test_shows_gaps_with_suggestions(self):
        d = _dongle_with_antenna(0, _whip("whip_915", 915))
        bands = [_band("915_ism", 915), _band("aprs_2m", 144)]
        report = compute_coverage(bands, [d])
        lines = render_coverage_report(report)
        text = "\n".join(lines)
        assert "1 of 2" in text
        assert "aprs_2m" in text
        assert "--all-bands" in text
        assert "Press Ctrl-C to abort" in text
