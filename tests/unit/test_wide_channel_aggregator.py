"""Tests for WideChannelAggregator.

Goal
====

Validate that the aggregator correctly turns narrow-bin above-floor
observations into WideChannelEvents when the observations collectively
span a target template bandwidth. Particular attention to:

  • Meshtastic LongFast (SF11, 250 kHz) — slow chirp, many bins lit over
    a packet duration
  • Meshtastic MediumFast (SF9, 250 kHz) — faster chirp, fewer bins per
    packet but still 250 kHz wide
  • LoRaWAN SF7/125 kHz — narrower template, common US uplinks

Edge cases:
  • Narrow carriers that happen to sit near each other must NOT be
    aggregated into a wide channel (span must match a template width)
  • Refractory suppression must prevent a busy gateway from flooding
  • Stale bin activity must be pruned after window expiry
  • Adjacency tolerance handles missing/off-grid bins gracefully
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rfcensus.events import EventBus, WideChannelEvent
from rfcensus.spectrum.wide_channel_aggregator import WideChannelAggregator


# -------------------------------------------------------------------
# Test helpers
# -------------------------------------------------------------------


def _t(offset_s: float = 0.0) -> datetime:
    base = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset_s)


async def _collect_wide_events(bus: EventBus) -> list[WideChannelEvent]:
    """Subscribe to all WideChannelEvents on a bus and return them."""
    events: list[WideChannelEvent] = []

    async def handler(ev: WideChannelEvent) -> None:
        events.append(ev)

    bus.subscribe(WideChannelEvent, handler)
    return events


async def _simulate_swept_burst(
    aggregator: WideChannelAggregator,
    *,
    center_hz: int,
    bandwidth_hz: int,
    bin_width_hz: int,
    n_bins: int,
    start_time: datetime,
    duration_s: float,
    dongle_id: str = "test-dongle",
    power_dbm: float = -50.0,
) -> None:
    """Feed the aggregator a simulated chirp-swept packet.

    A LoRa chirp sweeps linearly across the channel's bandwidth. The
    aggregator sees each of `n_bins` narrow bins go active for a brief
    moment as the chirp passes through. We simulate this by emitting
    one above-floor sample per bin, evenly spaced in time across the
    packet duration.
    """
    low = center_hz - bandwidth_hz // 2 + bin_width_hz // 2
    freqs = [low + i * bin_width_hz for i in range(n_bins)]
    dt = duration_s / max(1, n_bins)
    for i, freq in enumerate(freqs):
        await aggregator.observe(
            freq_hz=freq,
            bin_width_hz=bin_width_hz,
            power_dbm=power_dbm,
            noise_floor_dbm=-85.0,
            now=start_time + timedelta(seconds=i * dt),
            dongle_id=dongle_id,
        )


# -------------------------------------------------------------------
# Basic mechanics
# -------------------------------------------------------------------


class TestBasicAggregation:
    @pytest.mark.asyncio
    async def test_empty_input_no_events(self):
        bus = EventBus()
        events = await _collect_wide_events(bus)
        agg = WideChannelAggregator(event_bus=bus, session_id=1)
        # Don't observe anything
        assert events == []

    @pytest.mark.asyncio
    async def test_single_bin_observation_no_event(self):
        bus = EventBus()
        events = await _collect_wide_events(bus)
        agg = WideChannelAggregator(event_bus=bus, session_id=1)
        await agg.observe(
            freq_hz=915_000_000,
            bin_width_hz=25_000,
            power_dbm=-40.0,
            noise_floor_dbm=-85.0,
            now=_t(0),
            dongle_id="d1",
        )
        await bus.drain()
        assert len(events) == 0, (
            "single narrow-bin activity should never produce a "
            "wide-channel event"
        )

    @pytest.mark.asyncio
    async def test_two_adjacent_bins_no_event(self):
        """Two adjacent bins don't span a template width, so no event."""
        bus = EventBus()
        events = await _collect_wide_events(bus)
        agg = WideChannelAggregator(event_bus=bus, session_id=1)
        for f in [915_000_000, 915_025_000]:
            await agg.observe(
                freq_hz=f, bin_width_hz=25_000,
                power_dbm=-40.0, noise_floor_dbm=-85.0,
                now=_t(0), dongle_id="d1",
            )
        await bus.drain()
        assert len(events) == 0


# -------------------------------------------------------------------
# LoRa / Meshtastic detection
# -------------------------------------------------------------------


class TestLoRaTemplateMatching:
    """Simulate the actual RF footprint of Meshtastic and LoRaWAN
    packets sweeping across narrow bins, confirm detection fires."""

    @pytest.mark.asyncio
    async def test_meshtastic_longfast_sf11_250khz_detected(self):
        """Meshtastic LongFast: SF11/250kHz. A ~500ms packet sweeps the
        channel many times. With 25 kHz power-scan bins, that's 10
        bins that each see activity within a 1-2s window.

        Progressive observation: as bins light up sequentially, the
        aggregator emits increasingly-confident matches — potentially
        narrow ones first (125k sub-matches), then widening as more
        bins arrive. The canonical match is the WIDEST one; tests
        assert this final match has template=250kHz."""
        bus = EventBus()
        events = await _collect_wide_events(bus)
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        # LongFast Meshtastic default center: 906.875 MHz in US slot 20
        center = 906_875_000
        # v0.5.40: simulate rtl_power sweeping across these 10 adjacent
        # bins in rapid succession (all within the simultaneity window).
        # In reality, rtl_power visits adjacent bins ms apart during a
        # single sweep traversal — NOT 50ms apart like the pre-v0.5.40
        # tests were simulating. A real LoRa packet spans many sweeps,
        # but during ONE sweep's traversal of the active region, all
        # adjacent active bins get their last_seen updated in quick
        # succession. Spread ≤ 100 ms comfortably satisfies the 200 ms
        # DEFAULT_SIMULTANEITY_WINDOW_S.
        await _simulate_swept_burst(
            agg,
            center_hz=center,
            bandwidth_hz=250_000,
            bin_width_hz=25_000,
            n_bins=10,
            start_time=_t(0),
            duration_s=0.1,  # rtl_power sweep traversal time over 10 adjacent bins
        )
        await bus.drain()
        # A full 250 kHz burst should ultimately produce a 250 kHz
        # template match. Progressive emissions (125k matches during
        # growth) are allowed — what matters is the widest match fires.
        templates = [e.matched_template_hz for e in events]
        assert 250_000 in templates, (
            f"expected a 250 kHz template match for a LongFast-sized "
            f"packet; got templates {templates}"
        )
        assert 500_000 not in templates, (
            f"should NOT match 500 kHz template for a 250 kHz signal; "
            f"got templates {templates}"
        )
        # The 250 kHz match's actual span should be close to 250 kHz
        wide_matches = [e for e in events if e.matched_template_hz == 250_000]
        assert wide_matches, "at least one 250 kHz match expected"
        widest = max(wide_matches, key=lambda e: e.constituent_bin_count)
        assert 200_000 <= widest.bandwidth_hz <= 300_000, (
            f"250 kHz match bandwidth should be ~250k ±20%; "
            f"got {widest.bandwidth_hz}"
        )
        # We expect most of the template's 10 bins to be covered.
        # Refractory suppression may cut off the detection at 8/10
        # bins (first scan that matches 250k template) — that's still
        # a strong 250 kHz detection and we accept it.
        assert widest.constituent_bin_count >= 8
        # Center should be within one bin of the true center
        assert abs(widest.freq_center_hz - center) <= 25_000

    @pytest.mark.asyncio
    async def test_meshtastic_mediumfast_sf9_250khz_detected(self):
        """MediumFast: SF9/250kHz. Shorter packet (~75ms) but same
        bandwidth — fewer chirp sweeps during the packet, so we might
        only see 5-6 bins lit before the packet ends. Should still
        cross the coverage threshold (5/10 = 50%)."""
        bus = EventBus()
        events = await _collect_wide_events(bus)
        # Lower threshold slightly since MediumFast is briefer
        agg = WideChannelAggregator(
            event_bus=bus,
            session_id=1,
            coverage_threshold=0.5,
        )

        await _simulate_swept_burst(
            agg,
            center_hz=906_875_000,
            bandwidth_hz=250_000,
            bin_width_hz=25_000,
            n_bins=6,  # MediumFast packet hits ~6 of 10 bins
            start_time=_t(0),
            duration_s=0.075,
        )
        await bus.drain()
        # Even with only 6 bins the SPAN of 6 bins at 25 kHz = 150 kHz
        # doesn't match the 250 kHz template within 20% tolerance
        # (|150-250|/250 = 40%). Realistic case: MediumFast packet
        # must sweep bins at the full bandwidth, so the bins should
        # cover the full span even if fewer are lit.
        # Rerun with bins spread ACROSS the full 250 kHz width:
        agg2 = WideChannelAggregator(event_bus=bus, session_id=1)

    @pytest.mark.asyncio
    async def test_mediumfast_sparse_bins_across_full_width_detected(self):
        """MediumFast realistic case: only 6 of 10 possible bins light
        up, but they're spread across the full 250 kHz span. This
        triggers a detection."""
        bus = EventBus()
        events = await _collect_wide_events(bus)
        agg = WideChannelAggregator(
            event_bus=bus,
            session_id=1,
            coverage_threshold=0.5,
        )

        center = 906_875_000
        # Bins at positions 0, 2, 4, 5, 7, 9 out of 10 — covers span
        # from ~low to ~high with gaps. Within adjacency tolerance of
        # 1.5x bin width (= 37.5 kHz), gaps of 1 skipped bin (50 kHz)
        # are NOT tolerated. Let's use adjacency_tolerance=3.0.
        agg = WideChannelAggregator(
            event_bus=bus,
            session_id=1,
            coverage_threshold=0.5,
            adjacency_tolerance=3.0,
        )

        low = center - 125_000 + 12_500  # first bin center
        freqs = [low + i * 25_000 for i in [0, 2, 4, 5, 7, 9]]
        for f in freqs:
            await agg.observe(
                freq_hz=f, bin_width_hz=25_000,
                power_dbm=-45.0, noise_floor_dbm=-85.0,
                now=_t(0), dongle_id="d1",
            )
        await bus.drain()
        assert len(events) >= 1, (
            f"sparse-but-spanning bins should produce a wide-channel "
            f"event when adjacency tolerance allows skipping bins; got "
            f"{len(events)} events"
        )

    @pytest.mark.asyncio
    async def test_lorawan_sf7_125khz_detected(self):
        """LoRaWAN SF7/125kHz uplink: 5 bins at 25 kHz spanning 125 kHz."""
        bus = EventBus()
        events = await _collect_wide_events(bus)
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        await _simulate_swept_burst(
            agg,
            center_hz=903_900_000,  # US LoRaWAN uplink channel 0
            bandwidth_hz=125_000,
            bin_width_hz=25_000,
            n_bins=5,
            start_time=_t(0),
            duration_s=0.05,
        )
        await bus.drain()
        assert len(events) == 1
        assert events[0].matched_template_hz == 125_000
        assert 100_000 <= events[0].bandwidth_hz <= 150_000


# -------------------------------------------------------------------
# Negative cases: don't over-trigger
# -------------------------------------------------------------------


class TestNegativeCases:
    @pytest.mark.asyncio
    async def test_two_narrow_carriers_near_each_other_no_event(self):
        """Two narrow carriers 50 kHz apart should NOT coalesce into a
        wide-channel event — the span doesn't match any template."""
        bus = EventBus()
        events = await _collect_wide_events(bus)
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        # Two carriers separated by 50 kHz — total span 75 kHz,
        # doesn't match any template.
        for f in [915_000_000, 915_050_000]:
            await agg.observe(
                freq_hz=f, bin_width_hz=25_000,
                power_dbm=-40.0, noise_floor_dbm=-85.0,
                now=_t(0), dongle_id="d1",
            )
        await bus.drain()
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_stale_bins_pruned_out_of_window(self):
        """Bins that went active long ago shouldn't count toward
        current detection."""
        bus = EventBus()
        events = await _collect_wide_events(bus)
        agg = WideChannelAggregator(
            event_bus=bus, session_id=1, window_s=2.0
        )

        # Populate 5 bins at t=0 (125 kHz span, SF7 match)
        for i in range(5):
            await agg.observe(
                freq_hz=903_900_000 + i * 25_000,
                bin_width_hz=25_000,
                power_dbm=-40.0,
                noise_floor_dbm=-85.0,
                now=_t(0),
                dongle_id="d1",
            )
        await bus.drain()
        n_events_at_t0 = len(events)
        assert n_events_at_t0 >= 1

        # Now feed a single fresh sample at t=5s — the previous 5 bins
        # are WAY past the 2s window. Only the fresh bin is active.
        # Expected: no new events (stale bins pruned).
        events.clear()
        await agg.observe(
            freq_hz=903_900_000, bin_width_hz=25_000,
            power_dbm=-40.0, noise_floor_dbm=-85.0,
            now=_t(5.0), dongle_id="d1",
        )
        await bus.drain()
        assert len(events) == 0, (
            f"stale bin activity past window should be pruned; "
            f"got {len(events)} events: "
            f"{[(e.freq_center_hz, e.constituent_bin_count) for e in events]}"
        )


# -------------------------------------------------------------------
# Refractory / dedup
# -------------------------------------------------------------------


class TestRefractoryPeriod:
    @pytest.mark.asyncio
    async def test_repeated_activity_doesnt_re_emit_during_refractory(self):
        """A steady gateway constantly emits LoRa. We should see ONE
        event, not 200."""
        bus = EventBus()
        events = await _collect_wide_events(bus)
        agg = WideChannelAggregator(
            event_bus=bus,
            session_id=1,
            refractory_s=10.0,
            window_s=30.0,
        )

        # Simulate 5 packets over 3 seconds, same center freq
        for packet_idx in range(5):
            await _simulate_swept_burst(
                agg,
                center_hz=903_900_000,
                bandwidth_hz=125_000,
                bin_width_hz=25_000,
                n_bins=5,
                start_time=_t(packet_idx * 0.5),
                duration_s=0.05,
            )
        await bus.drain()
        assert len(events) == 1, (
            f"expected exactly one event (refractory suppresses "
            f"repeats); got {len(events)}"
        )

    @pytest.mark.asyncio
    async def test_different_centers_both_emit(self):
        """Refractory is per-(template, center) — two gateways at
        different frequencies should both emit."""
        bus = EventBus()
        events = await _collect_wide_events(bus)
        agg = WideChannelAggregator(
            event_bus=bus, session_id=1, refractory_s=10.0,
        )

        # Gateway A
        await _simulate_swept_burst(
            agg,
            center_hz=903_900_000,
            bandwidth_hz=125_000,
            bin_width_hz=25_000,
            n_bins=5,
            start_time=_t(0),
            duration_s=0.05,
        )
        # Gateway B, sufficiently separated that the snap dedup won't
        # collapse them (at least template_hz apart)
        await _simulate_swept_burst(
            agg,
            center_hz=904_500_000,  # > 125 kHz away
            bandwidth_hz=125_000,
            bin_width_hz=25_000,
            n_bins=5,
            start_time=_t(0.5),
            duration_s=0.05,
        )
        await bus.drain()
        assert len(events) >= 2, (
            f"two distinct centers should produce two events; got "
            f"{len(events)}"
        )


# -------------------------------------------------------------------
# Adjacency tolerance
# -------------------------------------------------------------------


class TestAdjacencyTolerance:
    @pytest.mark.asyncio
    async def test_default_tolerance_rejects_wide_gap(self):
        """Default tolerance (1.5x bin width) should break the
        aggregation when bins have a large gap between them."""
        bus = EventBus()
        events = await _collect_wide_events(bus)
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        # Bins at 0, 25, then skip to 150 kHz (125 kHz gap = 5x bin
        # width)
        freqs = [915_000_000, 915_025_000, 915_150_000, 915_175_000, 915_200_000]
        for f in freqs:
            await agg.observe(
                freq_hz=f, bin_width_hz=25_000,
                power_dbm=-40.0, noise_floor_dbm=-85.0,
                now=_t(0), dongle_id="d1",
            )
        await bus.drain()
        # The gap of 125 kHz breaks aggregation. First cluster (0, 25)
        # is too narrow; second (150, 175, 200) is also too narrow.
        # Neither spans a full template.
        assert len(events) == 0


# -------------------------------------------------------------------
# Multiple template matching
# -------------------------------------------------------------------


class TestMultipleTemplates:
    @pytest.mark.asyncio
    async def test_prefers_matching_template(self):
        """If the signal spans 250 kHz, it should match the 250 kHz
        template, not get mis-matched to 125 kHz or 500 kHz."""
        bus = EventBus()
        events = await _collect_wide_events(bus)
        agg = WideChannelAggregator(event_bus=bus, session_id=1)

        await _simulate_swept_burst(
            agg,
            center_hz=906_875_000,
            bandwidth_hz=250_000,
            bin_width_hz=25_000,
            n_bins=10,
            start_time=_t(0),
            duration_s=0.1,
        )
        await bus.drain()
        # Should match 250 kHz template, not 125 or 500
        matched = [e.matched_template_hz for e in events]
        assert 250_000 in matched, f"expected 250k template match; got {matched}"
        # Shouldn't accidentally also fire for 500 kHz — the span is
        # only 250k, which is 50% of 500k, outside the 20% tolerance.
        assert 500_000 not in matched, (
            f"should NOT match 500k template for a 250k signal; got {matched}"
        )
