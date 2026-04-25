"""Tests for detectors and chirp analysis."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

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

    # ──────────────────────────────────────────────────────────────
    # v0.6.2 — accept intermittent + pulsed classifications.
    #
    # The pre-v0.6.2 filter required active_ratio > 0.9 classifications
    # (continuous_carrier / modulated_continuous / fm_voice). P25 voice
    # channels keyed on/off, and even control channels with bin-level
    # power variability, land in `intermittent` or `pulsed`. Result:
    # the user's east-bay simulcast P25 system at ~770 MHz had 753
    # active carriers visible but the detector NEVER fired because all
    # of them were classified `intermittent`.
    # ──────────────────────────────────────────────────────────────

    async def test_v062_fires_on_intermittent_in_dedicated_band(self):
        """Single intermittent narrowband carrier in 700 MHz public
        safety band → fires (was silently ignored pre-v0.6.2)."""
        bus = EventBus()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))  # type: ignore[arg-type,misc]

        det = P25Detector()
        det.attach(bus, session_id=1)

        # 770.023 MHz — synthesizes the user's east-bay reported
        # control-channel candidate (strongest carrier in their
        # mystery-list at this freq).
        await bus.publish(_make_active_channel(
            770_023_000, 12_500, classification="intermittent",
        ))
        await bus.drain(timeout=1.0)
        assert len(captured) == 1
        assert captured[0].technology == "p25_narrowband"
        assert captured[0].freq_hz == 770_023_000

    async def test_v062_fires_on_pulsed_in_dedicated_band(self):
        """Pulsed (rare-burst) narrowband carrier in 800 MHz public
        safety also fires now."""
        bus = EventBus()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))  # type: ignore[arg-type,misc]

        det = P25Detector()
        det.attach(bus, session_id=1)

        await bus.publish(_make_active_channel(
            857_692_000, 12_500, classification="pulsed",
        ))
        await bus.drain(timeout=1.0)
        assert len(captured) == 1
        assert captured[0].technology == "p25_narrowband"

    async def test_v062_fires_trunked_with_intermittent_channels(self):
        """Three intermittent 12.5 kHz channels in a non-dedicated
        public-safety band → trunked detection still fires."""
        bus = EventBus()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))  # type: ignore[arg-type,misc]

        det = P25Detector()
        det.attach(bus, session_id=1)

        # VHF (non-dedicated) — needs ≥3 channels for trunked detection
        for freq in (155_000_000, 155_012_500, 155_025_000):
            await bus.publish(_make_active_channel(
                freq, 12_500, classification="intermittent",
            ))
        await bus.drain(timeout=1.0)
        assert len(captured) >= 1
        assert captured[0].technology == "p25_trunked_system"

    async def test_v062_still_rejects_unrelated_classifications(self):
        """Loosening to intermittent + pulsed must NOT also let in
        `unknown` or `periodic` — those don't fit the P25 fingerprint
        (control/voice channels aren't periodic, and `unknown` means
        the classifier had insufficient evidence)."""
        bus = EventBus()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))  # type: ignore[arg-type,misc]

        det = P25Detector()
        det.attach(bus, session_id=1)

        # Even at the right freq + bandwidth, these classifications
        # shouldn't fire.
        await bus.publish(_make_active_channel(
            770_000_000, 12_500, classification="periodic",
        ))
        await bus.publish(_make_active_channel(
            770_500_000, 12_500, classification="unknown",
        ))
        await bus.drain(timeout=1.0)
        assert captured == []

    async def test_v062_simulcast_multiple_strong_carriers_fire(self):
        """Synthesize the user's east-bay simulcast pattern: multiple
        strong carriers around 770 MHz. The dedicated-band fast-path
        means we fire on the FIRST matching carrier and dedup the
        band thereafter (one detection per band is intentional — see
        _announced set). Pre-v0.6.2, this silently fired zero times
        because none of these carriers were `continuous_carrier`.

        The detector accumulates all matching carriers in
        `_continuous_channels[band]` even after the announce, which
        could feed a future "update detection with more evidence"
        pathway. For now we just verify the announce happened and that
        downstream observers (the report) can see all four carriers
        had been observed."""
        bus = EventBus()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))  # type: ignore[arg-type,misc]

        det = P25Detector()
        det.attach(bus, session_id=1)

        # The user's reported strongest carriers in p25_700_public_safety
        carriers = (770_023_000, 770_295_000, 770_305_000, 770_858_000)
        for freq in carriers:
            await bus.publish(_make_active_channel(
                freq, 12_500, classification="intermittent",
            ))
        await bus.drain(timeout=1.0)

        # Exactly one detection in the 769-775 MHz band — the band is
        # `_announced` after the first matching carrier so subsequent
        # carriers don't re-fire.
        band_700_detections = [
            d for d in captured
            if 769_000_000 <= d.metadata.get("band_low_hz", 0) < 776_000_000
        ]
        assert len(band_700_detections) == 1
        # Detection fired on the first carrier we sent
        assert band_700_detections[0].freq_hz == 770_023_000
        # All four carriers are accumulated in the detector's internal
        # state regardless of announce dedup — this is the data a
        # future "evidence updater" would surface to the report.
        band = (769_000_000, 775_000_000)
        assert det._continuous_channels[band] == set(carriers)


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
