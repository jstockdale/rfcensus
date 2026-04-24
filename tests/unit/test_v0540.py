"""v0.5.40 regression tests — validates the specific failure modes
observed in the 15:51 scan output.

Three pre-v0.5.40 bugs are tested here:

1. Log spam: every sliding-window composite match emitted at INFO,
   producing hundreds of log lines per band.
2. Sliding-window duplicates: adjacent start_idx positions on a
   continuous active ridge each fire their own composite, when they
   should all be deduped as the same signal.
3. Sweep-induced false positives: in an active ISM band, every
   25 kHz sub-bin sees SOME activity over a 5-second window, so the
   aggregator happily fires composites across the whole band for
   unrelated transmissions.

Each bug has a dedicated test that would have caught the issue
against the v0.5.38 code, and passes against v0.5.40.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from rfcensus.detectors.builtin.lora import LoraDetector
from rfcensus.events import DetectionEvent, EventBus, WideChannelEvent
from rfcensus.spectrum.wide_channel_aggregator import WideChannelAggregator


def _t(offset_s: float = 0.0) -> datetime:
    base = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset_s)


async def _observe_bin(
    agg: WideChannelAggregator,
    *,
    freq_hz: int,
    now: datetime,
    bin_width_hz: int = 25_000,
    power_dbm: float = -45.0,
    noise_floor_dbm: float = -85.0,
    dongle_id: str = "test-dongle",
) -> None:
    await agg.observe(
        freq_hz=freq_hz,
        bin_width_hz=bin_width_hz,
        power_dbm=power_dbm,
        noise_floor_dbm=noise_floor_dbm,
        now=now,
        dongle_id=dongle_id,
    )


# ------------------------------------------------------------------
# Log level fix
# ------------------------------------------------------------------


class TestCompositeLogLevel:
    """Every sliding-window composite match should be logged at DEBUG,
    not INFO. Only the LoRa detector's per-band "detection fired" line
    belongs at INFO."""

    @pytest.mark.asyncio
    async def test_composite_logs_at_debug_not_info(self, caplog):
        bus = EventBus()
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        # Simulate a clean 125 kHz composite
        low = 915_000_000
        with caplog.at_level(logging.INFO, logger="rfcensus.spectrum.wide_channel_aggregator"):
            for i in range(5):
                await _observe_bin(
                    agg,
                    freq_hz=low + i * 25_000,
                    now=_t(i * 0.01),  # tight timing — passes simultaneity
                )
            await bus.drain()

        # At INFO level, the aggregator should have produced ZERO
        # log records, even though it fired at least one composite.
        info_records = [
            r for r in caplog.records
            if r.name == "rfcensus.spectrum.wide_channel_aggregator"
            and r.levelno >= logging.INFO
        ]
        assert len(info_records) == 0, (
            f"aggregator should not log composites at INFO; pre-v0.5.40 "
            f"flooded the log with hundreds of these lines per band. "
            f"Got {len(info_records)} INFO records: "
            f"{[r.message for r in info_records]}"
        )

    @pytest.mark.asyncio
    async def test_composite_still_logs_at_debug(self, caplog):
        """Sanity: DEBUG records ARE still produced — we demoted the
        level, didn't silence the output entirely."""
        bus = EventBus()
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        low = 915_000_000
        with caplog.at_level(
            logging.DEBUG, logger="rfcensus.spectrum.wide_channel_aggregator"
        ):
            for i in range(5):
                await _observe_bin(agg, freq_hz=low + i * 25_000, now=_t(i * 0.01))
            await bus.drain()

        debug_composite_records = [
            r for r in caplog.records
            if "wide channel composite" in r.message
            and r.levelno == logging.DEBUG
        ]
        assert len(debug_composite_records) >= 1


# ------------------------------------------------------------------
# Sliding-window duplicate suppression
# ------------------------------------------------------------------


class TestSlidingWindowDedup:
    """Pre-v0.5.40: the sliding window would fire one composite per
    start position along a continuous ridge of active bins, producing
    many overlapping 125/250/500 kHz events for a single signal.
    v0.5.40 fixes this with (a) tighter same-template overlap threshold
    (30% not 50%) and (b) center-distance dedup."""

    @pytest.mark.asyncio
    async def test_continuous_ridge_deduplicated(self):
        """Feed 20 adjacent active bins, all simultaneous — represents
        a long, continuous signal (e.g., a 500 kHz LoRa burst OR a
        wideband emitter).

        Pre-v0.5.40: the sliding window would fire multiple 125 kHz
        composites at 25-kHz-apart start positions along the ridge,
        producing 10+ overlapping events.

        v0.5.40: center-distance dedup collapses these to a small
        number of distinct detections per template width.
        """
        bus = EventBus()
        events: list[WideChannelEvent] = []

        async def _capture(ev: WideChannelEvent) -> None:
            events.append(ev)

        bus.subscribe(WideChannelEvent, _capture)
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        # 20 adjacent bins spanning 500 kHz
        low = 915_000_000
        for i in range(20):
            await _observe_bin(agg, freq_hz=low + i * 25_000, now=_t(i * 0.005))
        await bus.drain()

        # Count 125 kHz events — each should be at a distinct center
        # (center-distance ≥ 62.5 kHz apart). Pre-v0.5.40, we saw
        # centers 25 kHz apart all firing as separate composites.
        e125 = [e for e in events if e.matched_template_hz == 125_000]
        centers_125 = sorted(e.freq_center_hz for e in e125)
        for i in range(len(centers_125) - 1):
            gap = centers_125[i + 1] - centers_125[i]
            assert gap >= 60_000, (  # template/2 = 62.5kHz, allow small margin
                f"adjacent 125 kHz composites at {centers_125[i]} and "
                f"{centers_125[i+1]} are only {gap/1000:.1f} kHz apart; "
                f"should be ≥ 62 kHz apart after center-distance dedup"
            )

    @pytest.mark.asyncio
    async def test_field_scan_reduction(self):
        """Approximates the 15:51 scan pattern: 20 consecutive active
        bins in the 912-917 MHz region. Field log shed ~60 composite
        events in a few hundred ms. v0.5.40 should shed a handful —
        let's cap at 10 per template width."""
        bus = EventBus()
        events: list[WideChannelEvent] = []
        bus.subscribe(WideChannelEvent, lambda e: events.append(e))
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        low = 912_000_000
        for i in range(20):
            await _observe_bin(agg, freq_hz=low + i * 25_000, now=_t(i * 0.005))
        await bus.drain()

        for template in (125_000, 250_000, 500_000):
            matches = [e for e in events if e.matched_template_hz == template]
            assert len(matches) <= 10, (
                f"v0.5.40 should cap {template // 1000} kHz composites "
                f"per continuous ridge at ≤ 10 (pre-fix emitted 15+); "
                f"got {len(matches)}"
            )


# ------------------------------------------------------------------
# Simultaneity: the big fix
# ------------------------------------------------------------------


class TestSimultaneityCheck:
    """The headline v0.5.40 fix: bins whose last_seen timestamps are
    spread far apart in time do NOT form a valid composite, even
    though they're adjacent in frequency and all within the 5-second
    rolling window. This is what eliminates sweep-induced false
    positives in active ISM bands."""

    @pytest.mark.asyncio
    async def test_spread_out_observations_do_not_form_composite(self):
        """10 adjacent bins, each observed once, at 1-second intervals.

        In a real scan, this is what rtl_power returns for an ACTIVE
        BAND with many unrelated brief transmissions — each bin gets
        hit by some burst at some point over the 5-second window, but
        they're not part of the same signal.

        Pre-v0.5.40: fires 125/250/500 kHz composites.
        v0.5.40: no composites, because last_seen spread (9 seconds)
        exceeds simultaneity_window_s (200 ms default).
        """
        bus = EventBus()
        events: list[WideChannelEvent] = []
        bus.subscribe(WideChannelEvent, lambda e: events.append(e))
        # Longer window to hold all 10 observations in the rolling state
        agg = WideChannelAggregator(
            event_bus=bus, session_id=1, window_s=15.0
        )

        low = 915_000_000
        for i in range(10):
            await _observe_bin(
                agg,
                freq_hz=low + i * 25_000,
                now=_t(i * 1.0),  # 1 second apart — NOT simultaneous
            )
        await bus.drain()

        assert events == [], (
            f"Bins observed 1 second apart should NOT form a composite "
            f"(they represent unrelated transmissions, not a wideband "
            f"signal). Got {len(events)} composite events: "
            f"{[(e.freq_center_hz, e.matched_template_hz) for e in events]}"
        )

    @pytest.mark.asyncio
    async def test_tight_observations_do_form_composite(self):
        """Same 10 bins, but observed within 100 ms of each other —
        the rtl_power sweep-traversal case for a genuinely wideband
        signal. Should form a composite normally."""
        bus = EventBus()
        events: list[WideChannelEvent] = []
        bus.subscribe(WideChannelEvent, lambda e: events.append(e))
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        low = 915_000_000
        for i in range(10):
            await _observe_bin(
                agg,
                freq_hz=low + i * 25_000,
                now=_t(i * 0.01),  # 10 ms apart — one sweep's worth
            )
        await bus.drain()

        assert len(events) >= 1
        templates = {e.matched_template_hz for e in events}
        assert 250_000 in templates, (
            f"10 tight adjacent bins should form at least a 250 kHz "
            f"composite; got templates {templates}"
        )

    @pytest.mark.asyncio
    async def test_simultaneity_window_is_configurable(self):
        """If the operator knows their rtl_power sweep is slower than
        default (say, a wide-band sweep that takes 500 ms to traverse),
        they can loosen simultaneity_window_s and still get detections.
        """
        bus = EventBus()
        events: list[WideChannelEvent] = []
        bus.subscribe(WideChannelEvent, lambda e: events.append(e))
        # Configure 600 ms simultaneity window
        agg = WideChannelAggregator(
            event_bus=bus, session_id=1,
            simultaneity_window_s=0.6,
        )

        low = 915_000_000
        for i in range(10):
            await _observe_bin(
                agg,
                freq_hz=low + i * 25_000,
                now=_t(i * 0.05),  # 500 ms total spread
            )
        await bus.drain()

        assert len(events) >= 1, (
            "with simultaneity_window_s=0.6, 500 ms spread should still "
            "form composites"
        )


# ------------------------------------------------------------------
# IQ-capture failure visibility
# ------------------------------------------------------------------


class TestIQFailureVisibility:
    """v0.5.40 introduced a WARNING for inline IQ capture failures.
    v0.5.41 eliminated inline IQ capture entirely — the LoRa detector
    now defers confirmation to a wave-scheduler-integrated queue.
    These tests verify the new architecture doesn't emit the old
    warning (the failure mode no longer exists) and that detections
    get flagged `needs_iq_confirmation=True` in metadata instead.
    """

    @pytest.mark.asyncio
    async def test_no_inline_iq_warning_in_v0541(self, caplog):
        """v0.5.41: the detector no longer attempts inline IQ capture.
        No IQ-capture-failure WARNING should fire during detection —
        confirmation is fully deferred."""
        from rfcensus.spectrum.iq_capture import IQCaptureError

        class _FailingIQService:
            async def capture(self, **kwargs):
                raise IQCaptureError("should not be called in v0.5.41")

        bus = EventBus()
        detector = LoraDetector()
        detector.attach(bus=bus, session_id=1, iq_service=_FailingIQService())

        with caplog.at_level(logging.WARNING, logger="rfcensus.detectors.builtin.lora"):
            event = WideChannelEvent(
                session_id=1,
                freq_center_hz=906_875_000,
                bandwidth_hz=250_000,
                matched_template_hz=250_000,
                constituent_bin_count=10,
                coverage_ratio=0.90,
            )
            await bus.publish(event)
            await bus.drain()

        # No "IQ confirmation unavailable" warning should fire
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "IQ confirmation unavailable" in r.message
        ]
        assert len(warning_records) == 0, (
            "v0.5.41 should not emit the old inline-IQ WARNING — "
            "confirmation is now deferred to the wave scheduler"
        )

    @pytest.mark.asyncio
    async def test_detection_flagged_for_confirmation(self):
        """v0.5.41: LoRa detections include `needs_iq_confirmation=True`
        in metadata, which the DetectionWriter uses to auto-submit a
        ConfirmationTask."""
        from rfcensus.events import DetectionEvent

        bus = EventBus()
        detections: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: detections.append(e))

        detector = LoraDetector()
        detector.attach(bus=bus, session_id=1, iq_service=None)

        event = WideChannelEvent(
            session_id=1,
            freq_center_hz=906_875_000,
            bandwidth_hz=250_000,
            matched_template_hz=250_000,
            constituent_bin_count=10,
            coverage_ratio=0.90,
        )
        await bus.publish(event)
        await bus.drain()

        assert len(detections) >= 1
        det = detections[0]
        assert det.metadata.get("needs_iq_confirmation") is True, (
            "v0.5.41 LoRa detections should set needs_iq_confirmation "
            "so the DetectionWriter auto-submits a ConfirmationTask"
        )
        # SF classification is deferred — should be None at this point
        assert det.metadata.get("estimated_sf") is None
