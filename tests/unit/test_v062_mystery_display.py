"""v0.6.2 — mystery carrier display: clustering + saturation summary.

The v0.5.36 'Mystery carriers' section produced unreadable output in
busy RF environments because:

  1. A single carrier whose energy spans 3-5 adjacent FFT bins
     produced 3-5 near-duplicate ActiveChannelRecord rows. The
     'top 10 by persistence' list often showed THE SAME carrier
     three times under slightly different frequencies.

  2. Bands like business_uhf and frs_gmrs have hundreds of
     legitimate (just-undecoded) carriers. Listing 'top 10
     mysteries' from a 900-carrier band is meaningless – they're
     all 100% persistence with similar SNR, and they're not
     mysterious anyway, just unsupported.

v0.6.2 fixes:

  • _cluster_adjacent_channels merges records within
    _ADJACENT_CLUSTER_WINDOW_HZ of each other.
  • _mystery_score combines persistence × SNR (capped at 30 dB)
    so ranking isn't dominated by 100%-persistence ties.
  • Bands with > _SATURATED_BAND_THRESHOLD post-clustering carriers
    switch to a one-paragraph summary instead of enumerating rows.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from rfcensus.reporting.formats.text import (
    _ADJACENT_CLUSTER_WINDOW_HZ,
    _MAX_UNTRACKED_CHANNELS_PER_BAND,
    _SATURATED_BAND_THRESHOLD,
    _ChannelCluster,
    _cluster_adjacent_channels,
    _format_cluster,
    _format_saturated_band_summary,
    _mystery_score,
    _render_band_mystery_section,
)
from rfcensus.storage.models import ActiveChannelRecord


# ────────────────────────────────────────────────────────────────────
# Test fixtures
# ────────────────────────────────────────────────────────────────────


def _t(minute: int, second: int = 0) -> datetime:
    return datetime(2026, 1, 1, 12, minute, second, tzinfo=timezone.utc)


def _ch(
    freq_hz: int,
    *,
    bandwidth_hz: int = 10_000,
    peak_dbm: float | None = -40.0,
    floor_dbm: float | None = -85.0,
    persistence: float | None = 0.5,
    classification: str | None = None,
    duration_s: int = 30,
) -> ActiveChannelRecord:
    return ActiveChannelRecord(
        id=None,
        session_id=42,
        freq_center_hz=freq_hz,
        bandwidth_hz=bandwidth_hz,
        first_seen=_t(0),
        last_seen=_t(0, duration_s),
        peak_power_dbm=peak_dbm,
        avg_power_dbm=(peak_dbm - 3.0) if peak_dbm is not None else None,
        noise_floor_dbm=floor_dbm,
        classification=classification,
        persistence_ratio=persistence,
        confidence=persistence,
    )


# ────────────────────────────────────────────────────────────────────
# Clustering
# ────────────────────────────────────────────────────────────────────


class TestCluster:
    def test_empty_input_returns_empty(self):
        assert _cluster_adjacent_channels([]) == []

    def test_single_channel_one_cluster(self):
        c = _ch(freq_hz=433_920_000)
        clusters = _cluster_adjacent_channels([c])
        assert len(clusters) == 1
        assert clusters[0].count == 1

    def test_adjacent_bins_merged(self):
        """Three bins from one carrier (5 kHz apart) → one cluster."""
        chs = [
            _ch(freq_hz=433_920_000, peak_dbm=-30.0),
            _ch(freq_hz=433_925_000, peak_dbm=-35.0),
            _ch(freq_hz=433_930_000, peak_dbm=-40.0),
        ]
        clusters = _cluster_adjacent_channels(chs)
        assert len(clusters) == 1
        assert clusters[0].count == 3

    def test_distant_carriers_stay_separate(self):
        """Two carriers 100 kHz apart (well outside cluster window)
        remain distinct."""
        chs = [
            _ch(freq_hz=433_920_000),
            _ch(freq_hz=434_020_000),
        ]
        clusters = _cluster_adjacent_channels(chs)
        assert len(clusters) == 2

    def test_window_boundary_inclusive(self):
        """Two channels exactly _ADJACENT_CLUSTER_WINDOW_HZ apart
        should cluster (boundary is inclusive)."""
        chs = [
            _ch(freq_hz=433_920_000),
            _ch(freq_hz=433_920_000 + _ADJACENT_CLUSTER_WINDOW_HZ),
        ]
        clusters = _cluster_adjacent_channels(chs)
        assert len(clusters) == 1

    def test_window_just_past_boundary_separate(self):
        """One Hz past the cluster window → separate clusters."""
        chs = [
            _ch(freq_hz=433_920_000),
            _ch(freq_hz=433_920_000 + _ADJACENT_CLUSTER_WINDOW_HZ + 1),
        ]
        clusters = _cluster_adjacent_channels(chs)
        assert len(clusters) == 2

    def test_chained_clustering(self):
        """A-B within window, B-C within window, A-C beyond window —
        all three should still cluster because we merge greedily as
        we sweep left-to-right."""
        chs = [
            _ch(freq_hz=915_000_000),
            _ch(freq_hz=915_020_000),  # within 25k of #1
            _ch(freq_hz=915_040_000),  # within 25k of #2 but 40k from #1
        ]
        clusters = _cluster_adjacent_channels(chs)
        assert len(clusters) == 1
        assert clusters[0].count == 3

    def test_unsorted_input_handled(self):
        """Input is sorted internally before clustering."""
        chs = [
            _ch(freq_hz=433_930_000),
            _ch(freq_hz=433_920_000),
            _ch(freq_hz=433_925_000),
        ]
        clusters = _cluster_adjacent_channels(chs)
        assert len(clusters) == 1
        assert clusters[0].count == 3


class TestClusterAggregation:
    def test_representative_freq_is_loudest_member(self):
        """The cluster's reported frequency should be the loudest
        member's frequency — the spillover bins are skirts of the
        actual carrier."""
        chs = [
            _ch(freq_hz=433_920_000, peak_dbm=-50.0),
            _ch(freq_hz=433_925_000, peak_dbm=-30.0),  # loudest
            _ch(freq_hz=433_930_000, peak_dbm=-45.0),
        ]
        cluster = _cluster_adjacent_channels(chs)[0]
        assert cluster.representative_freq_hz == 433_925_000

    def test_peak_is_max_across_members(self):
        chs = [
            _ch(freq_hz=433_920_000, peak_dbm=-50.0),
            _ch(freq_hz=433_925_000, peak_dbm=-30.0),
        ]
        cluster = _cluster_adjacent_channels(chs)[0]
        assert cluster.peak_power_dbm == -30.0

    def test_floor_is_min_across_members(self):
        """Quietest neighbouring floor = best estimate of true floor
        under this carrier."""
        chs = [
            _ch(freq_hz=433_920_000, floor_dbm=-80.0),
            _ch(freq_hz=433_925_000, floor_dbm=-90.0),  # quietest
        ]
        cluster = _cluster_adjacent_channels(chs)[0]
        assert cluster.noise_floor_dbm == -90.0

    def test_persistence_is_max_across_members(self):
        chs = [
            _ch(freq_hz=433_920_000, persistence=0.3),
            _ch(freq_hz=433_925_000, persistence=0.9),  # most persistent
        ]
        cluster = _cluster_adjacent_channels(chs)[0]
        assert cluster.persistence_ratio == 0.9

    def test_classification_is_modal(self):
        chs = [
            _ch(freq_hz=433_920_000, classification="pulsed"),
            _ch(freq_hz=433_925_000, classification="pulsed"),
            _ch(freq_hz=433_930_000, classification="intermittent"),
        ]
        cluster = _cluster_adjacent_channels(chs)[0]
        assert cluster.classification == "pulsed"

    def test_handles_missing_data(self):
        """A cluster where some members lack peak/floor/persistence
        still produces sane aggregates."""
        chs = [
            _ch(freq_hz=433_920_000, peak_dbm=None, floor_dbm=None,
                persistence=None),
            _ch(freq_hz=433_925_000, peak_dbm=-40.0, floor_dbm=-80.0,
                persistence=0.5),
        ]
        cluster = _cluster_adjacent_channels(chs)[0]
        # Aggregates use only the members that have data
        assert cluster.peak_power_dbm == -40.0
        assert cluster.noise_floor_dbm == -80.0
        assert cluster.persistence_ratio == 0.5


# ────────────────────────────────────────────────────────────────────
# Scoring
# ────────────────────────────────────────────────────────────────────


class TestMysteryScore:
    def _cluster(self, **kw):
        return _ChannelCluster(members=[_ch(freq_hz=433_920_000, **kw)])

    def test_zero_persistence_zero_score(self):
        c = self._cluster(persistence=0.0, peak_dbm=-30.0, floor_dbm=-90.0)
        assert _mystery_score(c) == 0.0

    def test_high_snr_high_persistence_high_score(self):
        c = self._cluster(persistence=1.0, peak_dbm=-30.0, floor_dbm=-90.0)
        # SNR=60 dB capped at 30, persistence=1.0 → score = 1.0
        assert _mystery_score(c) == pytest.approx(1.0)

    def test_low_snr_high_persistence_low_score(self):
        """A constant carrier just above the floor isn't mysterious —
        it's barely there. Low SNR drags the score down even at 100%
        persistence."""
        c = self._cluster(persistence=1.0, peak_dbm=-83.0, floor_dbm=-85.0)
        score = _mystery_score(c)
        assert score < 0.3  # 2 dB SNR / 30 = 0.067

    def test_unknown_snr_uses_middling_factor(self):
        """A cluster with no SNR data shouldn't score 0; we don't know
        if it's mysterious or not. Use 0.5 as the SNR multiplier."""
        c = self._cluster(persistence=1.0, peak_dbm=None, floor_dbm=None)
        assert _mystery_score(c) == pytest.approx(0.5)

    def test_ranking_distinguishes_otherwise_tied_carriers(self):
        """Two carriers at 100% persistence — the one with higher SNR
        should rank higher. (Pre-v0.6.2 they tied, which is why
        saturated bands produced useless 'top 10' lists.)"""
        a = self._cluster(persistence=1.0, peak_dbm=-30.0, floor_dbm=-90.0)
        b = self._cluster(persistence=1.0, peak_dbm=-50.0, floor_dbm=-60.0)
        assert _mystery_score(a) > _mystery_score(b)


# ────────────────────────────────────────────────────────────────────
# Per-band rendering — saturated vs detailed
# ────────────────────────────────────────────────────────────────────


class TestRenderBandMysterySection:
    def test_small_band_renders_detailed_list(self):
        """Below saturation threshold → enumerate carriers."""
        chs = [
            _ch(freq_hz=433_920_000 + i * 100_000, persistence=0.9)
            for i in range(5)
        ]
        lines = _render_band_mystery_section(
            "433_ism", "433 MHz ISM", chs,
        )
        # Header + 5 rows (no truncation, no saturation)
        assert any("(5 active)" in l for l in lines)
        assert not any("saturated" in l for l in lines)
        # All five frequencies appear
        text = "\n".join(lines)
        for i in range(5):
            mhz = (433_920_000 + i * 100_000) / 1_000_000
            assert f"{mhz:.3f}" in text

    def test_saturated_band_renders_summary(self):
        """Above saturation threshold → summary, not enumeration.

        The synthetic case mirrors the user's reported business_uhf
        output: hundreds of distinct carriers all at high persistence.
        """
        # Spaced 50 kHz apart → all separate clusters → > threshold
        n = _SATURATED_BAND_THRESHOLD + 50
        chs = [
            _ch(
                freq_hz=451_000_000 + i * 50_000,
                peak_dbm=-30.0 + (i % 5),
                floor_dbm=-80.0,
                persistence=1.0,
            )
            for i in range(n)
        ]
        lines = _render_band_mystery_section(
            "business_uhf", "Business/land mobile UHF", chs,
        )
        text = "\n".join(lines)
        assert "saturated" in text
        assert "individual mystery enumeration suppressed" in text
        # Should mention how many carriers
        assert str(n) in text
        # Should suggest the monitor command
        assert "rfcensus monitor business_uhf" in text
        # Should NOT enumerate — i.e. shouldn't have N rows of
        # individual frequencies
        freq_lines = [l for l in lines if "MHz  peak=" in l]
        assert len(freq_lines) == 0, (
            f"saturated mode should not enumerate per-row data; "
            f"got {len(freq_lines)} rows"
        )

    def test_clustering_reduces_count_in_header(self):
        """When clustering merges raw bins, the header should note
        both the cluster count and the raw bin count."""
        # 10 bins from 2 carriers (5 each, all within window)
        chs = []
        for base in (915_000_000, 915_500_000):
            for i in range(5):
                chs.append(_ch(
                    freq_hz=base + i * 5_000,
                    peak_dbm=-30.0,
                    floor_dbm=-80.0,
                    persistence=1.0,
                ))
        lines = _render_band_mystery_section(
            "915_ism", "915 ISM", chs,
        )
        text = "\n".join(lines)
        # 2 carriers from 10 raw bins
        assert "2 carriers from 10 raw bins" in text

    def test_saturated_band_summary_lists_strongest(self):
        """Saturated summary should still mention the strongest carriers
        so the user has somewhere to start investigating."""
        n = _SATURATED_BAND_THRESHOLD + 10
        chs = [
            _ch(
                freq_hz=451_000_000 + i * 50_000,
                peak_dbm=-30.0,
                floor_dbm=-80.0,
                persistence=1.0,
            )
            for i in range(n)
        ]
        # Make one obvious standout
        chs.append(_ch(
            freq_hz=455_000_000, peak_dbm=+22.1, floor_dbm=-9.0,
            persistence=1.0,
        ))
        lines = _render_band_mystery_section(
            "business_uhf", "Business UHF", chs,
        )
        text = "\n".join(lines)
        assert "strongest" in text
        # The +22 dBm standout should be in the strongest list
        assert "455.000" in text

    def test_band_character_line_in_summary(self):
        """Saturated summary should include a band-character line
        with max peak, median, and a strong-count."""
        n = _SATURATED_BAND_THRESHOLD + 5
        chs = [
            _ch(
                freq_hz=770_000_000 + i * 100_000,
                peak_dbm=-15.0 + i * 0.5,
                floor_dbm=-85.0,
                persistence=1.0,
            )
            for i in range(n)
        ]
        lines = _render_band_mystery_section(
            "p25_700_public_safety", "P25 700", chs,
        )
        text = "\n".join(lines)
        assert "band character" in text
        assert "max peak" in text
        assert "median" in text


# ────────────────────────────────────────────────────────────────────
# Cluster line formatting
# ────────────────────────────────────────────────────────────────────


class TestFormatCluster:
    def test_single_bin_format_unchanged(self):
        """For single-bin clusters the line shape matches v0.5.36 so
        existing parsers / users keep working."""
        c = _cluster_adjacent_channels([_ch(
            freq_hz=433_920_000, peak_dbm=-30.0, floor_dbm=-80.0,
            persistence=0.9,
        )])[0]
        line = _format_cluster(c)
        assert "433.920 MHz" in line
        assert "peak=-30.0 dBm" in line
        assert "floor=-80.0 dBm" in line
        assert "persist=90%" in line
        # No span annotation for single bins
        assert "bins" not in line

    def test_multi_bin_format_includes_span(self):
        """Clustered multi-bin entries get a span+count annotation
        so the user knows it's been merged."""
        chs = [
            _ch(freq_hz=433_920_000, peak_dbm=-50.0),
            _ch(freq_hz=433_925_000, peak_dbm=-30.0),
            _ch(freq_hz=433_930_000, peak_dbm=-45.0),
        ]
        c = _cluster_adjacent_channels(chs)[0]
        line = _format_cluster(c)
        assert "433.925 MHz" in line  # representative is the loudest
        assert "3bins" in line  # cluster size annotated


# ────────────────────────────────────────────────────────────────────
# End-to-end: verify the user's reported output shape is fixed
# ────────────────────────────────────────────────────────────────────


class TestUserReportedRegression:
    """The user pasted a 'Mystery carriers' section in which:
      • 70cm_amateur showed 1245 active, top 3 entries all within
        20 kHz of each other (bin spillover from one carrier)
      • business_uhf showed 929 entries, 919 not shown
      • frs_gmrs showed 924, etc.
    These tests verify the v0.6.2 output handles those cases sanely.
    """

    def test_bin_spillover_collapses_in_70cm_case(self):
        """Synthesize the user's exact 70cm top-3: 426.682/693/698
        within 16 kHz of each other → must merge to one cluster."""
        chs = [
            _ch(freq_hz=426_682_000, peak_dbm=14.2, floor_dbm=3.0,
                persistence=1.0),
            _ch(freq_hz=426_693_000, peak_dbm=12.6, floor_dbm=0.6,
                persistence=1.0),
            _ch(freq_hz=426_698_000, peak_dbm=6.2, floor_dbm=-4.5,
                persistence=1.0),
        ]
        clusters = _cluster_adjacent_channels(chs)
        assert len(clusters) == 1
        # Representative frequency = loudest
        assert clusters[0].representative_freq_hz == 426_682_000
        # Combined peak = max
        assert clusters[0].peak_power_dbm == 14.2

    def test_business_uhf_size_triggers_saturated_mode(self):
        """929 carriers > _SATURATED_BAND_THRESHOLD → summary mode."""
        n = 929
        chs = [
            _ch(
                freq_hz=451_000_000 + i * 30_000,  # > _ADJACENT_CLUSTER_WINDOW_HZ so each is its own cluster
                peak_dbm=-10.0,
                floor_dbm=-50.0,
                persistence=1.0,
            )
            for i in range(n)
        ]
        lines = _render_band_mystery_section(
            "business_uhf", "Business UHF", chs,
        )
        text = "\n".join(lines)
        assert "saturated" in text
        # No per-row enumeration
        freq_lines = [l for l in lines if "MHz  peak=" in l]
        assert len(freq_lines) == 0

    def test_433_ism_stays_in_detailed_mode(self):
        """The user's 433_ism (21 active) is below threshold and was
        actually a useful section in the original output. v0.6.2 must
        keep that case rendering normally."""
        n = 21
        chs = [
            _ch(
                freq_hz=433_300_000 + i * 50_000,
                peak_dbm=-3.0 - i * 0.3,
                floor_dbm=-15.5,
                persistence=1.0 - 0.05 * i,
            )
            for i in range(n)
        ]
        lines = _render_band_mystery_section("433_ism", "433 MHz ISM", chs)
        text = "\n".join(lines)
        # Detailed mode → still has the (N active) header
        assert "(21 active)" in text
        # Should NOT trigger saturation suppression
        assert "saturated" not in text
        # Should enumerate (capped at _MAX_UNTRACKED_CHANNELS_PER_BAND)
        freq_lines = [l for l in lines if "MHz  peak=" in l]
        assert len(freq_lines) == _MAX_UNTRACKED_CHANNELS_PER_BAND

    def test_p25_voice_floods_handled(self):
        """P25 voice channels flood the list because we only detect
        the control channel. 753 carriers → saturated summary."""
        n = 753
        chs = [
            _ch(
                freq_hz=770_000_000 + i * 30_000,  # > _ADJACENT_CLUSTER_WINDOW_HZ so each is its own cluster
                peak_dbm=-10.0 - (i % 20),
                floor_dbm=-25.0,
                persistence=1.0,
            )
            for i in range(n)
        ]
        lines = _render_band_mystery_section(
            "p25_700_public_safety", "P25 700 MHz", chs,
        )
        text = "\n".join(lines)
        assert "saturated" in text
        # And the user gets pointed at the right tool
        assert "rfcensus monitor p25_700_public_safety" in text
