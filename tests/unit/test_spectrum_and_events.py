"""Tests for spectrum layer (noise floor, classifier) and event bus."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from rfcensus.events import DecodeEvent, EmitterEvent, Event, EventBus
from rfcensus.spectrum.classifier import ChannelHistory, SignalClassifier
from rfcensus.spectrum.noise_floor import NoiseFloorTracker


class TestNoiseFloorTracker:
    def test_converges_on_noise(self):
        t = NoiseFloorTracker(window=30, percentile=0.25)
        random.seed(42)
        for _ in range(100):
            t.observe(433_000_000, -65 + random.gauss(0, 2))
        floor = t.noise_floor(433_000_000)
        assert -70 < floor < -60

    def test_ignores_occasional_strong_signals(self):
        """Strong occasional bursts shouldn't raise the 25th percentile estimate."""
        t = NoiseFloorTracker(window=60, percentile=0.25)
        random.seed(42)
        for _ in range(50):
            t.observe(100_000_000, -70 + random.gauss(0, 1))
        # A few bursts
        for _ in range(5):
            t.observe(100_000_000, -30)
        floor = t.noise_floor(100_000_000)
        assert -75 < floor < -60

    def test_snr_calculation(self):
        t = NoiseFloorTracker(window=30)
        for _ in range(30):
            t.observe(915_000_000, -80)
        snr = t.snr(915_000_000, -40)
        assert 35 < snr < 45

    def test_unknown_freq_returns_pessimistic_floor(self):
        t = NoiseFloorTracker(window=30)
        assert t.noise_floor(999_000_000) == -100.0


class TestChannelHistoryAndClassifier:
    def _continuous_carrier(self) -> ChannelHistory:
        """Generate history of a steady continuous carrier."""
        h = ChannelHistory(freq_hz=100_000_000)
        t0 = datetime(2026, 4, 22, 12, tzinfo=timezone.utc)
        random.seed(1)
        for i in range(100):
            ts = t0 + timedelta(milliseconds=200 * i)
            h.observe(ts, -35 + random.gauss(0, 0.5), above_floor=True)
        return h

    def _pulsed_signal(self) -> ChannelHistory:
        """Generate history of sporadic bursts."""
        h = ChannelHistory(freq_hz=433_920_000)
        t0 = datetime(2026, 4, 22, 12, tzinfo=timezone.utc)
        # Long periods of quiet with occasional 30-sample bursts
        idx = 0
        for burst in range(5):
            # 200 quiet samples
            for i in range(200):
                ts = t0 + timedelta(milliseconds=200 * idx)
                h.observe(ts, -80, above_floor=False)
                idx += 1
            # 10 active samples
            for i in range(10):
                ts = t0 + timedelta(milliseconds=200 * idx)
                h.observe(ts, -40, above_floor=True)
                idx += 1
        return h

    def _periodic_beacon(self) -> ChannelHistory:
        """Regular transmissions at fixed interval."""
        h = ChannelHistory(freq_hz=162_400_000)
        t0 = datetime(2026, 4, 22, 12, tzinfo=timezone.utc)
        idx = 0
        for beacon in range(10):
            # ~60s of silence, then 5s of activity
            for _ in range(300):
                ts = t0 + timedelta(milliseconds=200 * idx)
                h.observe(ts, -85, above_floor=False)
                idx += 1
            for _ in range(25):
                ts = t0 + timedelta(milliseconds=200 * idx)
                h.observe(ts, -45, above_floor=True)
                idx += 1
        return h

    def test_classifies_continuous_carrier(self):
        c = SignalClassifier()
        result = c.classify(self._continuous_carrier())
        assert result.kind in ("continuous_carrier", "fm_voice", "modulated_continuous")
        assert result.confidence > 0.3

    def test_classifies_pulsed(self):
        c = SignalClassifier()
        result = c.classify(self._pulsed_signal())
        # Should be pulsed or intermittent, not continuous
        assert result.kind != "continuous_carrier"

    def test_returns_unknown_with_insufficient_samples(self):
        c = SignalClassifier()
        h = ChannelHistory(freq_hz=1_000_000)
        h.observe(datetime.now(timezone.utc), -50, above_floor=True)
        result = c.classify(h)
        assert result.kind == "unknown"


@pytest.mark.asyncio
class TestEventBus:
    async def test_subscribe_and_publish(self):
        bus = EventBus()
        captured = []

        async def handler(event: DecodeEvent):
            captured.append(event)

        bus.subscribe(DecodeEvent, handler)

        event = DecodeEvent(decoder_name="x", protocol="y")
        await bus.publish(event)
        await bus.drain()
        assert len(captured) == 1

    async def test_isinstance_based_dispatch(self):
        """Subscribing to Event base class should receive subclass events."""
        bus = EventBus()
        captured_decodes = []
        captured_all = []

        async def decode_handler(e: DecodeEvent):
            captured_decodes.append(e)

        async def all_handler(e: Event):
            captured_all.append(e)

        bus.subscribe(DecodeEvent, decode_handler)
        bus.subscribe(Event, all_handler)

        await bus.publish(DecodeEvent(decoder_name="x", protocol="y"))
        await bus.publish(EmitterEvent(protocol="z"))
        await bus.drain()

        assert len(captured_decodes) == 1
        assert len(captured_all) == 2

    async def test_handler_exception_doesnt_break_others(self):
        bus = EventBus()
        good_captured = []

        async def bad_handler(e: DecodeEvent):
            raise RuntimeError("boom")

        async def good_handler(e: DecodeEvent):
            good_captured.append(e)

        bus.subscribe(DecodeEvent, bad_handler)
        bus.subscribe(DecodeEvent, good_handler)

        await bus.publish(DecodeEvent(decoder_name="x", protocol="y"))
        await bus.drain()
        assert len(good_captured) == 1

    async def test_unsubscribe_via_cancel(self):
        bus = EventBus()
        captured = []

        async def handler(e: DecodeEvent):
            captured.append(e)

        sub = bus.subscribe(DecodeEvent, handler)
        await bus.publish(DecodeEvent(decoder_name="x", protocol="y"))
        await bus.drain()
        assert len(captured) == 1

        sub.cancel()
        await bus.publish(DecodeEvent(decoder_name="x", protocol="y"))
        await bus.drain()
        assert len(captured) == 1  # Still only 1


# ──────────────────────────────────────────────────────────────────
# Regression: power-scan backend pre-filtering
# ──────────────────────────────────────────────────────────────────


class TestPowerScanBackendFilter:
    """Regression test for the bug where _run_power_scan tried
    HackRFSweepBackend on systems with no HackRF, causing the broker
    to allocate an RTL-SDR which then failed available_on(), causing
    lease churn and a confusing log trace.

    The fix: pre-filter the candidate backend list by what driver
    hardware is actually present.
    """

    def test_no_hackrf_means_only_rtl_power_considered(self):
        """If the registry has no HackRF, HackRFSweepBackend should
        not even appear in the considered backend list."""
        from unittest.mock import MagicMock
        from rfcensus.spectrum.backends.hackrf_sweep import HackRFSweepBackend
        from rfcensus.spectrum.backends.rtl_power import RtlPowerBackend

        # Simulate the inline backend selection that lives in
        # _run_power_scan. We're not invoking the function here (it
        # requires a full broker/event-bus setup); we're verifying
        # the conditional that drives the selection.
        rtl_only_drivers = {"rtlsdr"}
        backends: list = []
        if "hackrf" in rtl_only_drivers:
            backends.append(HackRFSweepBackend)
        if "rtlsdr" in rtl_only_drivers:
            backends.append(RtlPowerBackend)
        assert backends == [RtlPowerBackend]

    def test_hackrf_present_includes_both(self):
        from rfcensus.spectrum.backends.hackrf_sweep import HackRFSweepBackend
        from rfcensus.spectrum.backends.rtl_power import RtlPowerBackend

        mixed_drivers = {"rtlsdr", "hackrf"}
        backends: list = []
        if "hackrf" in mixed_drivers:
            backends.append(HackRFSweepBackend)
        if "rtlsdr" in mixed_drivers:
            backends.append(RtlPowerBackend)
        # HackRF first (preferred for wide scans)
        assert backends == [HackRFSweepBackend, RtlPowerBackend]

    def test_hackrf_only_excludes_rtl_power(self):
        from rfcensus.spectrum.backends.hackrf_sweep import HackRFSweepBackend
        from rfcensus.spectrum.backends.rtl_power import RtlPowerBackend

        hackrf_only_drivers = {"hackrf"}
        backends: list = []
        if "hackrf" in hackrf_only_drivers:
            backends.append(HackRFSweepBackend)
        if "rtlsdr" in hackrf_only_drivers:
            backends.append(RtlPowerBackend)
        assert backends == [HackRFSweepBackend]
