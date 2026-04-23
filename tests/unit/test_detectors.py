"""Tests for detectors and chirp analysis."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from rfcensus.detectors.builtin.lora import LoraDetector
from rfcensus.detectors.builtin.p25 import P25Detector
from rfcensus.detectors.builtin.wifi_bt import WifiBtDetector
from rfcensus.events import ActiveChannelEvent, DetectionEvent, EventBus
from rfcensus.spectrum.chirp_analysis import analyze_chirps


def _make_active_channel(
    freq_hz: int,
    bandwidth_hz: int,
    classification: str = "pulsed",
    kind: str = "new",
) -> ActiveChannelEvent:
    return ActiveChannelEvent(
        kind=kind,  # type: ignore[arg-type]
        freq_center_hz=freq_hz,
        bandwidth_hz=bandwidth_hz,
        peak_power_dbm=-40.0,
        avg_power_dbm=-50.0,
        noise_floor_dbm=-80.0,
        snr_db=40.0,
        classification=classification,
        confidence=0.7,
    )


@pytest.mark.asyncio
class TestLoraDetector:
    async def test_fires_on_three_qualifying_bursts(self):
        bus = EventBus()
        captured: list[DetectionEvent] = []

        async def capture(e: DetectionEvent) -> None:
            captured.append(e)

        bus.subscribe(DetectionEvent, capture)
        det = LoraDetector()
        det.attach(bus, session_id=1, iq_service=None)

        for _ in range(3):
            await bus.publish(_make_active_channel(
                freq_hz=915_200_000, bandwidth_hz=125_000,
                classification="pulsed",
            ))
        await bus.drain(timeout=2.0)

        assert len(captured) == 1
        assert captured[0].technology == "lora"
        assert captured[0].detector_name == "lora"
        # LoRaWAN needs 3 distinct channels, not just 3 bursts on one channel
        assert captured[0].technology == "lora"

    async def test_fires_lorawan_on_multi_channel_gateway(self):
        """If we see 3 distinct channels before the first announcement,
        it's a LoRaWAN gateway. This requires the 3rd burst on channel A to
        arrive AFTER we've also seen at least 3 distinct channels."""
        bus = EventBus()
        captured: list[DetectionEvent] = []

        async def capture(e: DetectionEvent) -> None:
            captured.append(e)

        bus.subscribe(DetectionEvent, capture)
        det = LoraDetector()
        det.attach(bus, session_id=1, iq_service=None)

        # Interleave 3 channels so all three are seen before announcement
        freqs = (915_200_000, 915_400_000, 915_600_000)
        for _round in range(3):
            for f in freqs:
                await bus.publish(_make_active_channel(f, 125_000))
        await bus.drain(timeout=2.0)

        assert len(captured) >= 1
        # At announcement time, we've seen 3 distinct channels → LoRaWAN
        assert captured[0].technology == "lorawan"

    async def test_ignores_wrong_bandwidth(self):
        bus = EventBus()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))  # type: ignore[arg-type,misc]

        det = LoraDetector()
        det.attach(bus, session_id=1, iq_service=None)

        # 50 kHz bandwidth doesn't match LoRa signatures
        for _ in range(5):
            await bus.publish(_make_active_channel(915_200_000, 50_000))
        await bus.drain(timeout=1.0)
        assert captured == []

    async def test_ignores_out_of_band(self):
        bus = EventBus()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))  # type: ignore[arg-type,misc]

        det = LoraDetector()
        det.attach(bus, session_id=1, iq_service=None)

        # 2.4 GHz isn't a LoRa band
        for _ in range(5):
            await bus.publish(_make_active_channel(2_450_000_000, 125_000))
        await bus.drain(timeout=1.0)
        assert captured == []

    async def test_ignores_continuous_carrier(self):
        """Continuous carriers aren't LoRa (LoRa is bursty)."""
        bus = EventBus()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))  # type: ignore[arg-type,misc]

        det = LoraDetector()
        det.attach(bus, session_id=1, iq_service=None)

        for _ in range(5):
            await bus.publish(_make_active_channel(
                915_200_000, 125_000, classification="continuous_carrier",
            ))
        await bus.drain(timeout=1.0)
        assert captured == []


@pytest.mark.asyncio
class TestP25Detector:
    async def test_fires_on_dedicated_public_safety_band(self):
        bus = EventBus()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))  # type: ignore[arg-type,misc]

        det = P25Detector()
        det.attach(bus, session_id=1)

        # 851 MHz with 12.5 kHz BW continuous carrier → p25
        await bus.publish(_make_active_channel(
            851_200_000, 12_500, classification="continuous_carrier",
        ))
        await bus.drain(timeout=1.0)
        assert len(captured) == 1
        assert captured[0].technology == "p25_narrowband"

    async def test_fires_trunked_on_multiple_channels(self):
        bus = EventBus()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))  # type: ignore[arg-type,misc]

        det = P25Detector()
        det.attach(bus, session_id=1)

        # VHF public safety — 3 narrowband channels = trunked
        for freq in (155_000_000, 155_012_500, 155_025_000):
            await bus.publish(_make_active_channel(
                freq, 12_500, classification="fm_voice",
            ))
        await bus.drain(timeout=1.0)
        assert len(captured) >= 1
        # Should be trunked since ≥3 channels on a shared band (VHF)
        assert captured[0].technology == "p25_trunked_system"


@pytest.mark.asyncio
class TestWifiBtDetector:
    async def test_fires_when_2_channels_seen_in_24ghz(self):
        bus = EventBus()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))  # type: ignore[arg-type,misc]

        det = WifiBtDetector()
        det.attach(bus, session_id=1)

        await bus.publish(_make_active_channel(2_412_000_000, 20_000_000))
        await bus.publish(_make_active_channel(2_437_000_000, 20_000_000))
        await bus.drain(timeout=1.0)

        assert len(captured) == 1
        assert captured[0].technology == "ism_24"
        # Must include specialized tool suggestions
        tools = captured[0].hand_off_tools
        assert any("ubertooth" in t or "nrf" in t or "kismet" in t for t in tools)

    async def test_does_not_fire_on_single_channel(self):
        bus = EventBus()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))  # type: ignore[arg-type,misc]

        det = WifiBtDetector()
        det.attach(bus, session_id=1)

        await bus.publish(_make_active_channel(2_412_000_000, 20_000_000))
        await bus.drain(timeout=1.0)
        assert captured == []


class TestChirpAnalysis:
    def _synthesize_chirp(
        self, duration_s: float = 0.1, sample_rate: int = 1_000_000,
        slope_hz_per_sec: float = 500_000,  # 500 kHz/s, gives 50 kHz sweep over 100ms
        start_freq_hz: float = -25_000,  # endpoints in (-25, +25) kHz, well within Nyquist
    ) -> np.ndarray:
        """Generate a clean linear chirp."""
        n = int(sample_rate * duration_s)
        t = np.arange(n) / sample_rate
        # Phase is integral of frequency; linear chirp → quadratic phase
        phase = 2 * np.pi * (start_freq_hz * t + 0.5 * slope_hz_per_sec * t**2)
        return np.exp(1j * phase).astype(np.complex64)

    def _synthesize_noise(
        self, duration_s: float = 0.1, sample_rate: int = 1_000_000
    ) -> np.ndarray:
        n = int(sample_rate * duration_s)
        rng = np.random.default_rng(seed=42)
        i = rng.standard_normal(n, dtype=np.float32)
        q = rng.standard_normal(n, dtype=np.float32)
        return (i + 1j * q).astype(np.complex64) * 0.3

    def test_detects_clean_chirp(self):
        samples = self._synthesize_chirp()
        result = analyze_chirps(samples, sample_rate=1_000_000)
        assert result.chirp_confidence > 0.5
        assert result.num_chirp_segments >= 1

    def test_does_not_detect_in_noise(self):
        samples = self._synthesize_noise()
        result = analyze_chirps(samples, sample_rate=1_000_000)
        assert result.chirp_confidence < 0.5

    def test_handles_tiny_input(self):
        samples = np.zeros(10, dtype=np.complex64)
        result = analyze_chirps(samples, sample_rate=1_000_000)
        assert result.chirp_confidence == 0.0
        assert "insufficient" in result.reasoning

    def test_pure_tone_not_detected_as_chirp(self):
        """A constant-frequency tone should not be called a chirp."""
        n = 100_000
        t = np.arange(n) / 1_000_000
        # Constant tone at 10 kHz
        samples = np.exp(1j * 2 * np.pi * 10_000 * t).astype(np.complex64)
        result = analyze_chirps(samples, sample_rate=1_000_000)
        # Chirp confidence should be low because slope is ~0 (fails freq_span check)
        assert result.chirp_confidence < 0.5
