"""v0.5.36 tests: active channels surfaced in the report.

Background
==========

Prior to v0.5.36, the `rfcensus inventory` report showed:
  • Emitters (from decoder output)
  • Detections (from detector classification)
  • Anomalies
  • Per-band execution summary with power_scan=yes/no

But it did NOT show the `active_channels` table contents — frequencies
that the power scan observed as active (SNR above threshold for long
enough to matter) but that no decoder output and no detector
classified. Users saw `power_scan=yes` lines without any visible
result, creating the impression that power scanning did nothing.

In reality the data was being collected — it was flowing through
OccupancyAnalyzer → ActiveChannelEvent → ActiveChannelWriter → 
active_channels table — but stopped there unless a detector picked
it up.

v0.5.36 surfaces these "mystery carriers" in a dedicated report
section, filtering out channels that are already represented as
emitters or detections (to avoid duplication) and grouping by the
band they fall in.

The tests below assert:
  • Filter correctly excludes channels matching a known emitter or detection
  • Filter includes channels with no match (within bandwidth tolerance)
  • Band grouping uses the session's plan to identify each channel's band
  • Text report includes a "Mystery carriers" section when data exists
  • Text report omits the section entirely when nothing is untracked
  • JSON report includes active_channels AND detections fields
  • Truncation kicks in at MAX_UNTRACKED_CHANNELS_PER_BAND
  • Channel formatting handles missing-data gracefully
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from rfcensus.config.schema import BandConfig
from rfcensus.engine.scheduler import ExecutionPlan, ScheduleTask, Wave
from rfcensus.engine.session import SessionResult
from rfcensus.engine.strategy import StrategyResult
from rfcensus.reporting.formats.json import render_json_report
from rfcensus.reporting.formats.text import (
    _FREQ_MATCH_TOLERANCE_HZ,
    _MAX_UNTRACKED_CHANNELS_PER_BAND,
    _band_id_for_freq,
    _format_active_channel,
    _select_untracked_channels,
    render_text_report,
)
from rfcensus.storage.models import (
    ActiveChannelRecord,
    AnomalyRecord,
    DetectionRecord,
    EmitterRecord,
)


# ------------------------------------------------------------------
# Fixture helpers
# ------------------------------------------------------------------


def _t(hours: int = 0, minutes: int = 0, seconds: int = 0) -> datetime:
    """Deterministic timestamps for fixtures."""
    base = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(hours=hours, minutes=minutes, seconds=seconds)


def _band(id: str, center_mhz: float, bw_khz: float = 200) -> BandConfig:
    half = int(bw_khz * 1000 // 2)
    return BandConfig(
        id=id,
        name=id,
        freq_low=int(center_mhz * 1e6) - half,
        freq_high=int(center_mhz * 1e6) + half,
    )


def _make_session_result(bands: list[BandConfig]) -> SessionResult:
    tasks = [
        ScheduleTask(
            band=b,
            suggested_dongle_id=f"dongle-{i}",
            suggested_antenna_id=None,
        )
        for i, b in enumerate(bands)
    ]
    plan = ExecutionPlan(
        waves=[Wave(index=0, tasks=tasks)],
        max_parallel_per_wave=4,
    )
    return SessionResult(
        session_id=42,
        started_at=_t(0),
        ended_at=_t(1),
        plan=plan,
        strategy_results=[
            StrategyResult(band_id=b.id, power_scan_performed=True)
            for b in bands
        ],
        total_decodes=0,
    )


def _emitter(
    freq_hz: int,
    protocol: str = "interlogix_security",
    obs: int = 5,
    confidence: float = 0.5,
) -> EmitterRecord:
    return EmitterRecord(
        id=hash((freq_hz, protocol)) & 0xFFFFFF,
        protocol=protocol,
        device_id=str(freq_hz),
        device_id_hash="dead" + "beef" * 4,
        classification="security_sensor",
        first_seen=_t(0, 5),
        last_seen=_t(0, 55),
        observation_count=obs,
        typical_freq_hz=freq_hz,
        typical_rssi_dbm=-30.0,
        confidence=confidence,
    )


def _detection(freq_hz: int, technology: str = "lora") -> DetectionRecord:
    return DetectionRecord(
        id=1,
        session_id=42,
        detector="lora",
        technology=technology,
        freq_hz=freq_hz,
        detected_at=_t(0, 10),
        bandwidth_hz=125_000,
        confidence=0.9,
        evidence="heuristic match",
        hand_off_tools=["gr-lora"],
    )


def _active_channel(
    freq_hz: int,
    bandwidth_hz: int = 25_000,
    peak_dbm: float = -40.0,
    floor_dbm: float = -85.0,
    persistence: float = 0.1,
    classification: str | None = None,
) -> ActiveChannelRecord:
    return ActiveChannelRecord(
        id=None,
        session_id=42,
        freq_center_hz=freq_hz,
        bandwidth_hz=bandwidth_hz,
        first_seen=_t(0, 1),
        last_seen=_t(0, 30),
        peak_power_dbm=peak_dbm,
        avg_power_dbm=peak_dbm - 3.0,
        noise_floor_dbm=floor_dbm,
        classification=classification,
        persistence_ratio=persistence,
        confidence=persistence,
    )


# ------------------------------------------------------------------
# _select_untracked_channels
# ------------------------------------------------------------------


class TestSelectUntrackedChannels:
    def test_channel_with_no_emitters_is_returned(self):
        ch = _active_channel(freq_hz=915_000_000)
        result = _select_untracked_channels([ch], [], [])
        assert result == [ch]

    def test_channel_matching_emitter_is_excluded(self):
        """A channel at 433.525 MHz (10 kHz bin) should match an
        emitter at 433.534 MHz — they're the same carrier, just
        binning drift."""
        emitter = _emitter(freq_hz=433_534_000)
        ch = _active_channel(
            freq_hz=433_525_000, bandwidth_hz=10_000
        )
        result = _select_untracked_channels([ch], [emitter], [])
        assert result == []

    def test_channel_matching_detection_is_excluded(self):
        detection = _detection(freq_hz=915_250_000, technology="lora")
        ch = _active_channel(
            freq_hz=915_260_000, bandwidth_hz=125_000
        )
        result = _select_untracked_channels([ch], [], [detection])
        assert result == []

    def test_far_away_emitter_does_not_match(self):
        """A channel at 915 MHz should NOT match an emitter at 433 MHz.
        (Sanity check against a too-generous tolerance.)"""
        emitter = _emitter(freq_hz=433_534_000)
        ch = _active_channel(freq_hz=915_000_000)
        result = _select_untracked_channels([ch], [emitter], [])
        assert result == [ch]

    def test_tolerance_boundary_within_envelope(self):
        """An emitter just inside the bandwidth+tolerance envelope
        matches the channel."""
        ch = _active_channel(
            freq_hz=433_500_000, bandwidth_hz=10_000
        )
        # envelope = [433.500 - 5k - 20k, 433.500 + 5k + 20k]
        #          = [433.475, 433.525]
        emitter_just_inside = _emitter(freq_hz=433_524_000)
        assert _select_untracked_channels(
            [ch], [emitter_just_inside], []
        ) == []

    def test_tolerance_boundary_outside_envelope(self):
        """An emitter just outside the envelope does NOT match."""
        ch = _active_channel(
            freq_hz=433_500_000, bandwidth_hz=10_000
        )
        # envelope = [433.475, 433.525]
        emitter_just_outside = _emitter(freq_hz=433_530_000)
        # Beyond 433.525, so should remain untracked
        assert _select_untracked_channels(
            [ch], [emitter_just_outside], []
        ) == [ch]

    def test_emitter_with_none_freq_ignored(self):
        """Emitters without a typical_freq_hz shouldn't accidentally
        match every channel."""
        emitter = _emitter(freq_hz=433_534_000)
        emitter.typical_freq_hz = None
        ch = _active_channel(freq_hz=433_525_000)
        assert _select_untracked_channels([ch], [emitter], []) == [ch]

    def test_multiple_channels_partial_match(self):
        """Mix: one channel matches an emitter, one doesn't — only
        the unmatched one stays."""
        emitter = _emitter(freq_hz=433_534_000)
        matched = _active_channel(
            freq_hz=433_530_000, bandwidth_hz=10_000
        )
        unmatched = _active_channel(freq_hz=915_250_000)
        result = _select_untracked_channels(
            [matched, unmatched], [emitter], []
        )
        assert result == [unmatched]


# ------------------------------------------------------------------
# _band_id_for_freq
# ------------------------------------------------------------------


class TestBandIdForFreq:
    def test_freq_in_band_returns_id(self):
        bands = [
            _band("433_ism", 433.92, bw_khz=1_000),
            _band("915_ism", 915.0, bw_khz=26_000),
        ]
        assert _band_id_for_freq(bands, 433_000_000 + 500_000) == "433_ism"

    def test_freq_in_second_band(self):
        bands = [
            _band("433_ism", 433.92, bw_khz=1_000),
            _band("915_ism", 915.0, bw_khz=26_000),
        ]
        assert _band_id_for_freq(bands, 915_500_000) == "915_ism"

    def test_freq_outside_all_bands_returns_none(self):
        bands = [_band("433_ism", 433.92, bw_khz=1_000)]
        assert _band_id_for_freq(bands, 915_000_000) is None

    def test_empty_bands_returns_none(self):
        assert _band_id_for_freq([], 915_000_000) is None

    def test_freq_exactly_on_low_edge(self):
        bands = [_band("test", 100.0, bw_khz=1_000)]
        # center=100M, bw=1M → freq_low=99.5M, freq_high=100.5M
        assert _band_id_for_freq(bands, 99_500_000) == "test"

    def test_freq_exactly_on_high_edge(self):
        bands = [_band("test", 100.0, bw_khz=1_000)]
        assert _band_id_for_freq(bands, 100_500_000) == "test"


# ------------------------------------------------------------------
# _format_active_channel
# ------------------------------------------------------------------


class TestFormatActiveChannel:
    def test_full_data(self):
        ch = _active_channel(
            freq_hz=915_250_000,
            peak_dbm=-40.0,
            floor_dbm=-85.0,
            persistence=0.37,
        )
        line = _format_active_channel(ch)
        assert "915.250 MHz" in line
        assert "-40.0 dBm" in line
        assert "-85.0 dBm" in line
        assert "37%" in line
        # duration was 29 minutes in the fixture (t(0,1) to t(0,30))
        assert "m" in line  # minutes format

    def test_missing_peak_power(self):
        ch = _active_channel(freq_hz=100_000_000)
        ch.peak_power_dbm = None
        line = _format_active_channel(ch)
        assert "? dBm" in line

    def test_missing_persistence(self):
        ch = _active_channel(freq_hz=100_000_000)
        ch.persistence_ratio = None
        line = _format_active_channel(ch)
        # Persistence shown as "?" when missing
        assert "persist=?" in line

    def test_classification_shown(self):
        ch = _active_channel(
            freq_hz=100_000_000, classification="probable_lora"
        )
        line = _format_active_channel(ch)
        assert "[probable_lora]" in line

    def test_no_classification_shows_unclassified(self):
        ch = _active_channel(freq_hz=100_000_000)
        ch.classification = None
        line = _format_active_channel(ch)
        assert "[unclassified]" in line


# ------------------------------------------------------------------
# End-to-end text report
# ------------------------------------------------------------------


class TestTextReportActiveChannelsSection:
    def _render(self, *, active_channels, emitters=None, detections=None):
        bands = [
            _band("433_ism", 433.92, bw_khz=1_000),
            _band("915_ism", 915.0, bw_khz=26_000),
        ]
        result = _make_session_result(bands)
        return render_text_report(
            result,
            emitters or [],
            anomalies=[],
            detections=detections or [],
            active_channels=active_channels,
        )

    def test_mystery_carriers_section_appears_with_data(self):
        """Untracked active channels produce a dedicated section."""
        ch = _active_channel(freq_hz=915_250_000)
        text = self._render(active_channels=[ch])
        assert "Mystery carriers" in text
        assert "915.250 MHz" in text

    def test_no_section_when_no_active_channels(self):
        """No `Mystery carriers` section if there's nothing to show."""
        text = self._render(active_channels=[])
        assert "Mystery carriers" not in text

    def test_no_section_when_all_channels_already_decoded(self):
        """If every active channel matches an existing emitter, the
        section should be suppressed — we only surface mystery
        carriers, not duplicates."""
        emitter = _emitter(freq_hz=433_920_000)
        ch = _active_channel(
            freq_hz=433_920_000, bandwidth_hz=10_000
        )
        text = self._render(
            active_channels=[ch], emitters=[emitter]
        )
        assert "Mystery carriers" not in text

    def test_channel_outside_any_band_bucketed_under_unknown(self):
        """If an active channel happens to report a frequency outside
        every planned band, it should land under the fallback
        '(outside scanned bands)' group rather than silently vanish."""
        ch = _active_channel(freq_hz=2_400_000_000)  # 2.4 GHz
        text = self._render(active_channels=[ch])
        assert "(outside scanned bands)" in text

    def test_truncation_per_band(self):
        """When a band has more channels than the per-band cap, the
        extras are summarized rather than dumped."""
        channels = [
            _active_channel(
                freq_hz=915_000_000 + 10_000 * i,
                persistence=0.5 - 0.01 * i,
            )
            for i in range(_MAX_UNTRACKED_CHANNELS_PER_BAND + 5)
        ]
        text = self._render(active_channels=channels)
        assert "5 more in 915_ism not shown" in text

    def test_ordering_by_persistence(self):
        """Within a band, channels are ordered most-persistent first
        so the operator sees the most suspicious signals at the top."""
        low = _active_channel(freq_hz=915_100_000, persistence=0.1)
        high = _active_channel(freq_hz=915_200_000, persistence=0.9)
        text = self._render(active_channels=[low, high])
        # The high-persistence channel should appear before the low one
        idx_high = text.find("915.200 MHz")
        idx_low = text.find("915.100 MHz")
        assert idx_high >= 0 and idx_low >= 0
        assert idx_high < idx_low, (
            f"high-persistence channel (0.9) should appear before "
            f"low-persistence (0.1); got indices "
            f"{idx_high} (high) vs {idx_low} (low)"
        )


# ------------------------------------------------------------------
# JSON report
# ------------------------------------------------------------------


class TestJsonReportActiveChannels:
    def _render(self, active_channels, detections=None):
        import json
        bands = [_band("915_ism", 915.0, bw_khz=26_000)]
        result = _make_session_result(bands)
        out = render_json_report(
            result,
            [],
            [],
            detections or [],
            active_channels,
        )
        return json.loads(out)

    def test_active_channels_key_exists(self):
        payload = self._render(active_channels=[])
        assert "active_channels" in payload
        assert payload["active_channels"] == []

    def test_active_channels_entries_serialized(self):
        ch = _active_channel(
            freq_hz=915_250_000, peak_dbm=-40.0, persistence=0.42
        )
        payload = self._render(active_channels=[ch])
        assert len(payload["active_channels"]) == 1
        entry = payload["active_channels"][0]
        assert entry["freq_center_hz"] == 915_250_000
        assert entry["peak_power_dbm"] == -40.0
        assert entry["persistence_ratio"] == 0.42
        # Timestamps should be ISO strings (round-trippable)
        assert "first_seen" in entry
        assert "last_seen" in entry

    def test_detections_key_exists(self):
        """v0.5.36 also adds detections to the JSON report (previously
        missing — the text report had them, JSON did not)."""
        payload = self._render(active_channels=[])
        assert "detections" in payload
        assert payload["detections"] == []

    def test_detections_serialized(self):
        d = _detection(freq_hz=915_000_000, technology="lora")
        payload = self._render(active_channels=[], detections=[d])
        assert len(payload["detections"]) == 1
        entry = payload["detections"][0]
        assert entry["technology"] == "lora"
        assert entry["freq_hz"] == 915_000_000
        assert entry["hand_off_tools"] == ["gr-lora"]


# ------------------------------------------------------------------
# ReportBuilder integration
# ------------------------------------------------------------------


class TestReportBuilderFetchesActiveChannels:
    """ReportBuilder must actually call the ActiveChannelRepo —
    otherwise the feature is inert regardless of what the text/json
    formats can do."""

    def test_report_builder_has_active_channel_repo(self):
        import inspect

        from rfcensus.reporting import report as report_mod
        src = inspect.getsource(report_mod)
        assert "ActiveChannelRepo" in src, (
            "ReportBuilder must import ActiveChannelRepo to populate "
            "the new report section."
        )
        assert "active_channel_repo" in src, (
            "ReportBuilder must hold an active_channel_repo instance."
        )

    def test_report_builder_fetches_channels_in_render(self):
        import inspect

        from rfcensus.reporting.report import ReportBuilder
        src = inspect.getsource(ReportBuilder.render)
        assert "for_session" in src and "active_channel_repo" in src, (
            "ReportBuilder.render must call active_channel_repo.for_session"
            " to retrieve this session's active channels for reporting."
        )


# ------------------------------------------------------------------
# Constants sanity
# ------------------------------------------------------------------


class TestModuleConstants:
    def test_tolerance_is_reasonable(self):
        """The frequency-match tolerance needs to be wide enough to
        cover normal binning drift but narrow enough not to cause
        false merges between genuinely distinct channels."""
        # > 1 kHz but < 100 kHz feels like a sane engineering band
        assert 1_000 < _FREQ_MATCH_TOLERANCE_HZ < 100_000

    def test_per_band_cap_is_reasonable(self):
        """The per-band display cap controls report readability."""
        assert 5 <= _MAX_UNTRACKED_CHANNELS_PER_BAND <= 50
