"""v0.5.38 integration tests: wide-channel aggregation drives LoRa
detection on simulated Meshtastic traffic.

These tests validate the full pipeline:

  PowerSample-level above-floor observations
    → WideChannelAggregator
    → WideChannelEvent
    → LoraDetector.on_wide_channel
    → DetectionEvent with technology=lora/lorawan/meshtastic + variant

Why this matters
================

Before v0.5.38, the LoRa detector required `ActiveChannelEvent` with
bandwidth_hz ≈ 125/250/500 kHz. Since power-scan bins are 10-25 kHz
wide, no single bin matched — AND since LoRa packets are short
(<1s), no bin stayed continuously active long enough to pass the
OccupancyAnalyzer's 1-second hold time. Result: LoRa detector never
fired on real Meshtastic or LoRaWAN traffic.

v0.5.38 fixes this by introducing WideChannelAggregator, which
observes above-floor samples directly (bypassing hold-time debouncing)
and emits WideChannelEvent when adjacent bins collectively span a
LoRa-standard template. The LoRa detector now subscribes to these
events and fires immediately, including SF estimation from chirp
slope when IQ confirmation is available.

Validated cases below:
  • Meshtastic LongFast (SF11 / 250 kHz) at a US channel
  • Meshtastic MediumFast (SF9 / 250 kHz) at a US channel
  • Generic LoRa 125 kHz traffic (e.g., LoRaWAN SF7)
  • Traffic outside LoRa bands does NOT trigger detection
  • Narrow single-carrier traffic does NOT trigger wide-channel match
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rfcensus.detectors.builtin.lora import (
    LoraDetector,
    _estimate_sf_from_slope,
    _label_variant,
)
from rfcensus.events import (
    ActiveChannelEvent,
    DetectionEvent,
    EventBus,
    WideChannelEvent,
)
from rfcensus.spectrum.wide_channel_aggregator import WideChannelAggregator


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _t(offset_s: float = 0.0) -> datetime:
    base = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset_s)


async def _simulate_burst(
    agg: WideChannelAggregator,
    *,
    center_hz: int,
    bandwidth_hz: int,
    bin_width_hz: int,
    n_bins: int,
    start_time: datetime,
    duration_s: float,
    dongle_id: str = "test-dongle",
) -> None:
    low = center_hz - bandwidth_hz // 2 + bin_width_hz // 2
    freqs = [low + i * bin_width_hz for i in range(n_bins)]
    dt = duration_s / max(1, n_bins)
    for i, freq in enumerate(freqs):
        await agg.observe(
            freq_hz=freq,
            bin_width_hz=bin_width_hz,
            power_dbm=-45.0,
            noise_floor_dbm=-85.0,
            now=start_time + timedelta(seconds=i * dt),
            dongle_id=dongle_id,
        )


# -------------------------------------------------------------------
# End-to-end: aggregator → LoRa detector → DetectionEvent
# -------------------------------------------------------------------


class TestMeshtasticDetectionEndToEnd:
    """Simulate the RF footprint of Meshtastic and LoRaWAN bursts and
    verify a DetectionEvent reaches the bus with correct metadata."""

    @pytest.mark.asyncio
    async def test_meshtastic_longfast_produces_detection_event(self):
        """LongFast: 250 kHz wide, SF11. Without IQ capture (no IQ
        service attached) we can't classify SF, but we should still
        fire a LoRa detection with 250 kHz bandwidth in the US ISM
        band."""
        bus = EventBus()
        detections: list[DetectionEvent] = []

        async def _capture(ev: DetectionEvent) -> None:
            detections.append(ev)

        bus.subscribe(DetectionEvent, _capture)

        # Wire up detector
        detector = LoraDetector()
        detector.attach(bus=bus, session_id=1, iq_service=None)

        # Wire up aggregator
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        # Simulate a LongFast-like packet at 906.875 MHz.
        # v0.5.40: use a tight simulation window that respects the
        # aggregator's simultaneity_window_s — represents rtl_power
        # sweeping through these 10 adjacent bins during a single
        # traversal while the LoRa packet is active.
        center = 906_875_000
        await _simulate_burst(
            agg,
            center_hz=center,
            bandwidth_hz=250_000,
            bin_width_hz=25_000,
            n_bins=10,
            start_time=_t(0),
            duration_s=0.1,
        )
        await bus.drain()

        # At least one LoRa-family detection should have fired
        assert len(detections) >= 1, (
            "expected a DetectionEvent from the LoRa detector for a "
            "LongFast-sized wide-channel composite; got none"
        )
        # Find the best detection (widest bandwidth)
        best = max(detections, key=lambda d: d.bandwidth_hz)
        assert best.detector_name == "lora"
        assert best.technology in ("lora", "lorawan", "meshtastic")
        # Bandwidth should reflect the template that matched
        assert best.bandwidth_hz == 250_000, (
            f"LongFast should produce a 250 kHz detection; "
            f"got {best.bandwidth_hz}"
        )
        # freq should be in the LoRa band (close to 906 MHz)
        assert 902_000_000 <= best.freq_hz <= 928_000_000
        assert abs(best.freq_hz - center) <= 25_000

    @pytest.mark.asyncio
    async def test_meshtastic_mediumfast_produces_detection_event(self):
        """MediumFast: SF9/250kHz. Same bandwidth as LongFast, shorter
        packet duration. Test setup feeds fewer observations but still
        covers the full template width — aggregator should still fire."""
        bus = EventBus()
        detections: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: detections.append(e))

        detector = LoraDetector()
        detector.attach(bus=bus, session_id=1)
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        center = 906_875_000
        await _simulate_burst(
            agg,
            center_hz=center,
            bandwidth_hz=250_000,
            bin_width_hz=25_000,
            # MediumFast is briefer than LongFast — simulate by feeding
            # ~8 bins, still spanning the full template width
            n_bins=8,
            start_time=_t(0),
            duration_s=0.08,
        )
        # Stretch bin placement so they span the full 250k
        # (above used n_bins=8 at bin_width=25k → only 175k span;
        # the aggregator would reject that for 250k template)
        # Instead simulate sparse coverage:
        await bus.drain()

        # Clear events and redo with proper sparse-but-full-span simulation
        detections.clear()
        bus = EventBus()
        bus.subscribe(DetectionEvent, lambda e: detections.append(e))
        detector = LoraDetector()
        detector.attach(bus=bus, session_id=1)
        agg = WideChannelAggregator(
            event_bus=bus, session_id=1,
            adjacency_tolerance=3.0,  # allow skip-bins for sparse coverage
        )

        # 7 bins at positions 0, 1, 3, 5, 6, 8, 9 out of 10 — covers
        # full 250 kHz span with gaps consistent with MediumFast's
        # shorter packet duration
        low = center - 125_000 + 12_500
        positions = [0, 1, 3, 5, 6, 8, 9]
        for pos_idx, pos in enumerate(positions):
            await agg.observe(
                freq_hz=low + pos * 25_000,
                bin_width_hz=25_000,
                power_dbm=-45.0,
                noise_floor_dbm=-85.0,
                now=_t(pos_idx * 0.01),
                dongle_id="test",
            )
        await bus.drain()

        assert len(detections) >= 1, (
            "MediumFast-like sparse-but-full-span activity should "
            "produce a LoRa detection"
        )
        best = max(detections, key=lambda d: d.bandwidth_hz)
        assert best.bandwidth_hz == 250_000
        assert best.technology in ("lora", "lorawan", "meshtastic")

    @pytest.mark.asyncio
    async def test_lorawan_sf7_125khz_produces_detection_event(self):
        """Generic LoRaWAN SF7/125kHz uplink."""
        bus = EventBus()
        detections: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: detections.append(e))

        detector = LoraDetector()
        detector.attach(bus=bus, session_id=1)
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        await _simulate_burst(
            agg,
            center_hz=903_900_000,  # US LoRaWAN uplink channel 0
            bandwidth_hz=125_000,
            bin_width_hz=25_000,
            n_bins=5,
            start_time=_t(0),
            duration_s=0.05,
        )
        await bus.drain()
        assert len(detections) == 1
        assert detections[0].bandwidth_hz == 125_000


# -------------------------------------------------------------------
# Negative cases: don't over-trigger
# -------------------------------------------------------------------


class TestNoFalsePositives:
    @pytest.mark.asyncio
    async def test_narrow_carrier_no_detection(self):
        """A single narrow carrier at 915 MHz should NOT produce a
        LoRa detection. This guards against the aggregator accidentally
        firing on non-LoRa activity like garage remotes (OOK 10 kHz)."""
        bus = EventBus()
        detections: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: detections.append(e))

        detector = LoraDetector()
        detector.attach(bus=bus, session_id=1)
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        # Single active bin — not wide enough for any template
        await agg.observe(
            freq_hz=915_000_000, bin_width_hz=25_000,
            power_dbm=-40.0, noise_floor_dbm=-85.0,
            now=_t(0), dongle_id="d1",
        )
        await bus.drain()
        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_traffic_outside_lora_bands_no_detection(self):
        """A wide-channel composite OUTSIDE any LoRa-allocated band
        (e.g., 600 MHz TV broadcast) should not trigger LoRa detection.
        The detector's coverage check filters by its relevant_freq_ranges."""
        bus = EventBus()
        detections: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: detections.append(e))

        detector = LoraDetector()
        detector.attach(bus=bus, session_id=1)
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        # Simulate 250 kHz activity at 600 MHz (not a LoRa band).
        # Use tight simulation (matches v0.5.40 simultaneity) so the
        # aggregator WOULD fire if this frequency were in-band — the
        # only reason the detection doesn't fire is the LoRa detector's
        # coverage check rejecting out-of-band frequencies.
        await _simulate_burst(
            agg,
            center_hz=600_000_000,
            bandwidth_hz=250_000,
            bin_width_hz=25_000,
            n_bins=10,
            start_time=_t(0),
            duration_s=0.1,
        )
        await bus.drain()
        # No LoRa detection (out of band)
        lora_detections = [d for d in detections if d.detector_name == "lora"]
        assert len(lora_detections) == 0, (
            f"600 MHz activity should not trigger LoRa detection; "
            f"got {len(lora_detections)} detections"
        )


# -------------------------------------------------------------------
# SF estimation from slope
# -------------------------------------------------------------------


class TestSpreadingFactorEstimation:
    """The chirp slope → spreading factor mapping is a closed-form
    calculation: SF = log2(BW² / slope). Verify it classifies the
    Meshtastic presets correctly."""

    def test_longfast_sf11_slope(self):
        """LongFast: BW=250kHz, SF=11. Expected slope: 250000^2 / 2^11
        = 30_517_578 Hz/s ≈ 30.5 MHz/s."""
        slope = 250_000 ** 2 / 2 ** 11
        sf = _estimate_sf_from_slope(
            slope_hz_per_sec=slope, bandwidth_hz=250_000
        )
        assert sf == 11

    def test_mediumfast_sf9_slope(self):
        """MediumFast: BW=250kHz, SF=9. Slope = 250000^2 / 512 = 122 MHz/s."""
        slope = 250_000 ** 2 / 2 ** 9
        sf = _estimate_sf_from_slope(
            slope_hz_per_sec=slope, bandwidth_hz=250_000
        )
        assert sf == 9

    def test_lorawan_sf7_125khz_slope(self):
        """LoRaWAN SF7/125kHz: slope = 125000^2 / 128 = 122 MHz/s
        (coincidentally same slope as MediumFast, but different BW)."""
        slope = 125_000 ** 2 / 2 ** 7
        sf = _estimate_sf_from_slope(
            slope_hz_per_sec=slope, bandwidth_hz=125_000
        )
        assert sf == 7

    def test_variant_labels_are_distinctive(self):
        """The key claim: SF11/250k is unambiguously LongFast, and
        SF9/250k is unambiguously MediumFast. Without this we can't
        tell the two apart in detections."""
        assert _label_variant(sf=11, bandwidth_hz=250_000) == (
            "meshtastic_long_fast"
        )
        assert _label_variant(sf=9, bandwidth_hz=250_000) == (
            "meshtastic_medium_fast"
        )
        # SF7/125 is LoRaWAN, not Meshtastic
        assert _label_variant(sf=7, bandwidth_hz=125_000) == "lorawan_sf7"

    def test_zero_slope_returns_none(self):
        assert (
            _estimate_sf_from_slope(slope_hz_per_sec=0, bandwidth_hz=250_000)
            is None
        )

    def test_implausible_slope_returns_none(self):
        """A slope that'd give SF=50 should be rejected."""
        # SF=50 means slope = BW² / 2^50 — absurdly tiny
        slope = 250_000 ** 2 / 2 ** 50
        result = _estimate_sf_from_slope(
            slope_hz_per_sec=slope, bandwidth_hz=250_000
        )
        assert result is None


# -------------------------------------------------------------------
# Backward compatibility: narrow-bin path still works
# -------------------------------------------------------------------


class TestBackwardCompatibility:
    """v0.5.38 added wide-channel path but kept the narrow-bin path
    as a fallback. Ensure the narrow-bin path still fires correctly
    when it does find matching activity (unusual in practice since
    power-scan bins are narrower than LoRa channels, but the code
    path is exercised in tests and integration)."""

    @pytest.mark.asyncio
    async def test_narrow_bin_path_still_fires(self):
        """Directly fire ActiveChannelEvents with matching bandwidth."""
        bus = EventBus()
        detections: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: detections.append(e))

        detector = LoraDetector()
        detector.attach(bus=bus, session_id=1)

        # Fire 3 matching ActiveChannelEvents (threshold is
        # BURSTS_FOR_DETECTION=3)
        for _ in range(3):
            await bus.publish(
                ActiveChannelEvent(
                    session_id=1,
                    kind="new",
                    freq_center_hz=903_900_000,
                    bandwidth_hz=125_000,
                    classification="pulsed",
                )
            )
        await bus.drain()
        assert len(detections) >= 1
        assert detections[0].detector_name == "lora"
