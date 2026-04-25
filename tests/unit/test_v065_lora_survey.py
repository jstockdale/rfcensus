"""v0.6.5 — LoraSurveyTask tests.

The survey task connects to a fanout, streams IQ, runs energy-gated
chirp detection. We can't stand up a full fanout in tests (real
rtl_tcp + RtlTcpFanout is integration-test territory), so we test
the analytical pieces in isolation:

  • _decode_chunk: u8 interleaved → complex64 conversion
  • Energy gate: noise-floor EMA + threshold logic
  • Refractory: same (freq, bw) suppressed within window
  • LoraSurveyStats: shape + defaults
"""

from __future__ import annotations

import numpy as np
import pytest


# ────────────────────────────────────────────────────────────────────
# 1. LoraSurveyStats shape
# ────────────────────────────────────────────────────────────────────


class TestLoraSurveyStats:
    def test_defaults(self):
        from rfcensus.engine.lora_survey_task import LoraSurveyStats
        s = LoraSurveyStats()
        assert s.chunks_read == 0
        assert s.bytes_read == 0
        assert s.chunks_above_floor == 0
        assert s.analyses_performed == 0
        assert s.detections_emitted == 0
        assert s.suppressed_by_refractory == 0
        assert s.duration_s == 0.0
        assert s.ended_reason == ""
        assert s.errors == []

    def test_field_assignment(self):
        from rfcensus.engine.lora_survey_task import LoraSurveyStats
        s = LoraSurveyStats()
        s.chunks_read = 100
        s.detections_emitted = 3
        s.errors.append("test error")
        assert s.chunks_read == 100
        assert s.detections_emitted == 3
        assert s.errors == ["test error"]


# ────────────────────────────────────────────────────────────────────
# 2. Chunk decoding
# ────────────────────────────────────────────────────────────────────


class TestChunkDecode:
    """rtl_tcp ships unsigned 8-bit interleaved I/Q. Verify the
    survey task's _decode_chunk converts bytes → complex64 with the
    right scaling and DC offset."""

    def _make_task(self):
        # Construct without running — we're only calling _decode_chunk
        from rfcensus.engine.lora_survey_task import LoraSurveyTask
        from unittest.mock import MagicMock
        # Need a band-shaped object with freq_low/freq_high/id
        band = MagicMock()
        band.freq_low = 902_000_000
        band.freq_high = 928_000_000
        band.id = "test_band"
        return LoraSurveyTask(
            broker=MagicMock(),
            event_bus=MagicMock(),
            band=band,
            duration_s=1.0,
        )

    def test_dc_centered_zero_input_decodes_to_zero(self):
        """All bytes = 127 (DC-centered) should produce ~0+0j."""
        task = self._make_task()
        # 127 is just below center 127.5; produces -0.004 + -0.004j
        raw = bytes([127] * 1024)
        samples = task._decode_chunk(raw)
        assert samples.dtype == np.complex64
        assert samples.size == 512  # 1024 bytes / 2 = 512 IQ pairs
        # All samples should be the same small negative value
        assert np.all(np.abs(samples - samples[0]) < 1e-6)
        # Mean magnitude should be tiny
        assert float(np.mean(np.abs(samples))) < 0.01

    def test_max_input_decodes_to_unity(self):
        """All 0xFF bytes should map to ~+1+1j (max amplitude)."""
        task = self._make_task()
        raw = bytes([0xFF] * 1024)
        samples = task._decode_chunk(raw)
        assert samples.size == 512
        # 255 - 127.5 = 127.5; / 127.5 = 1.0
        for s in samples[:10]:
            assert abs(s.real - 1.0) < 0.01
            assert abs(s.imag - 1.0) < 0.01

    def test_min_input_decodes_to_negative_unity(self):
        """All 0x00 bytes should map to ~-1-1j."""
        task = self._make_task()
        raw = bytes([0x00] * 1024)
        samples = task._decode_chunk(raw)
        for s in samples[:10]:
            assert abs(s.real - (-1.0)) < 0.02
            assert abs(s.imag - (-1.0)) < 0.02

    def test_empty_input(self):
        task = self._make_task()
        samples = task._decode_chunk(b"")
        assert samples.size == 0
        assert samples.dtype == np.complex64

    def test_odd_length_truncated(self):
        """Odd byte counts get the trailing byte trimmed (defensive
        against partial reads)."""
        task = self._make_task()
        raw = bytes([127] * 1023)  # odd
        samples = task._decode_chunk(raw)
        # Should produce 511 samples (1022 bytes) not 511.5
        assert samples.size == 511


# ────────────────────────────────────────────────────────────────────
# 3. Refractory suppression
# ────────────────────────────────────────────────────────────────────


class TestRefractorySuppression:
    """First detection of a (freq, bw) tuple announces; subsequent
    matches within the refractory window are suppressed. After the
    window expires, re-announce."""

    def _make_task(self, refractory_s=0.1):
        from rfcensus.engine.lora_survey_task import LoraSurveyTask
        from unittest.mock import MagicMock
        band = MagicMock()
        band.freq_low = 902_000_000
        band.freq_high = 928_000_000
        band.id = "test"
        return LoraSurveyTask(
            broker=MagicMock(), event_bus=MagicMock(),
            band=band, duration_s=1.0,
            refractory_s=refractory_s,
        )

    def _make_hit(self, freq_hz, bw_hz):
        from rfcensus.spectrum.in_window_survey import SurveyHit
        from rfcensus.spectrum.chirp_analysis import ChirpAnalysis
        return SurveyHit(
            freq_hz=freq_hz,
            bandwidth_hz=bw_hz,
            chirp_analysis=ChirpAnalysis(
                chirp_confidence=0.8,
                num_chirp_segments=3,
                mean_segment_length_samples=1024.0,
                mean_slope_hz_per_sec=125e3 / 0.01,
                reasoning="test",
            ),
            snr_db=15.0,
        )

    def test_first_detection_announceable(self):
        task = self._make_task()
        hit = self._make_hit(915_000_000, 125_000)
        assert task._is_announceable(hit) is True

    def test_repeat_within_window_suppressed(self):
        task = self._make_task(refractory_s=10.0)
        hit = self._make_hit(915_000_000, 125_000)
        assert task._is_announceable(hit) is True
        task._mark_refractory(hit)
        # Same hit, immediate retry → suppressed
        assert task._is_announceable(hit) is False

    def test_different_bandwidth_not_suppressed(self):
        task = self._make_task(refractory_s=10.0)
        hit_125 = self._make_hit(915_000_000, 125_000)
        hit_250 = self._make_hit(915_000_000, 250_000)
        task._mark_refractory(hit_125)
        # Different BW → not suppressed
        assert task._is_announceable(hit_250) is True

    def test_far_freq_not_suppressed(self):
        """A new emitter > 100 kHz away from the prior should
        announce, not be suppressed."""
        task = self._make_task(refractory_s=10.0)
        hit_a = self._make_hit(915_000_000, 125_000)
        hit_b = self._make_hit(915_500_000, 125_000)  # 500 kHz away
        task._mark_refractory(hit_a)
        assert task._is_announceable(hit_b) is True

    def test_close_freq_suppressed(self):
        """Within 100 kHz tolerance → considered the same emitter."""
        task = self._make_task(refractory_s=10.0)
        hit_a = self._make_hit(915_000_000, 125_000)
        hit_b = self._make_hit(915_050_000, 125_000)  # 50 kHz away
        task._mark_refractory(hit_a)
        assert task._is_announceable(hit_b) is False

    @pytest.mark.asyncio
    async def test_refractory_expires(self):
        """After refractory_s elapses, re-announcement is allowed."""
        import asyncio
        task = self._make_task(refractory_s=0.05)
        hit = self._make_hit(915_000_000, 125_000)
        task._mark_refractory(hit)
        assert task._is_announceable(hit) is False
        await asyncio.sleep(0.1)
        assert task._is_announceable(hit) is True


# ────────────────────────────────────────────────────────────────────
# 4. Cancellation
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestCancellation:
    async def test_cancel_sets_event(self):
        from rfcensus.engine.lora_survey_task import LoraSurveyTask
        from unittest.mock import MagicMock
        band = MagicMock()
        band.freq_low = 902_000_000
        band.freq_high = 928_000_000
        band.id = "test"
        task = LoraSurveyTask(
            broker=MagicMock(), event_bus=MagicMock(),
            band=band, duration_s=1.0,
        )
        await task.cancel()
        assert task._cancelled.is_set()


# ────────────────────────────────────────────────────────────────────
# 5. SF classification + variant labeling in emitted detections
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestEmissionEnrichment:
    """v0.6.5 added SF + variant labels to LoraSurveyTask detections,
    matching the legacy LoraDetector's enrichment. Verify that the
    same (slope, BW) inputs produce the same SF/variant outputs in
    the survey's emitted DetectionEvent."""

    def _make_task(self):
        from rfcensus.engine.lora_survey_task import LoraSurveyTask
        from unittest.mock import MagicMock
        from rfcensus.events import EventBus

        band = MagicMock()
        band.freq_low = 902_000_000
        band.freq_high = 928_000_000
        band.id = "915_ism"
        bus = EventBus()
        return LoraSurveyTask(
            broker=MagicMock(), event_bus=bus,
            band=band, duration_s=1.0,
        ), bus

    def _make_hit(self, freq_hz, bw_hz, *, sf=None,
                  sf_confidence=3.0, sf_peak=0.18):
        """v0.6.8: SF is now stamped on ChirpAnalysis by survey_iq_window
        (via classify_sf_dechirp), not re-derived from slope. Tests
        construct hits with the SF already specified, mirroring what
        the production survey path produces.
        """
        from rfcensus.spectrum.in_window_survey import SurveyHit
        from rfcensus.spectrum.chirp_analysis import ChirpAnalysis
        return SurveyHit(
            freq_hz=freq_hz,
            bandwidth_hz=bw_hz,
            chirp_analysis=ChirpAnalysis(
                chirp_confidence=0.9,
                num_chirp_segments=4,
                mean_segment_length_samples=2048.0,
                mean_slope_hz_per_sec=0.0,  # not used by emit
                reasoning="test",
                estimated_sf=sf,
                sf_confidence=sf_confidence,
                sf_peak_concentration=sf_peak,
                sf_scores={s: (sf_peak if s == sf else 0.04)
                           for s in range(7, 13)} if sf else None,
            ),
            snr_db=18.0,
        )

    async def test_meshtastic_long_fast_classified(self):
        """SF11 / 250 kHz → meshtastic_long_fast → technology=meshtastic."""
        from rfcensus.engine.lora_survey_task import LoraSurveyStats
        from rfcensus.events import DetectionEvent

        task, bus = self._make_task()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))

        # SF11 / 250 kHz → meshtastic_long_fast → technology=meshtastic
        bw = 250_000
        hit = self._make_hit(915_000_000, bw, sf=11)
        stats = LoraSurveyStats()
        await task._emit_detection(hit, stats)
        await bus.drain(timeout=1.0)

        assert len(captured) == 1
        d = captured[0]
        assert d.technology == "meshtastic"
        assert d.metadata["estimated_sf"] == 11
        assert d.metadata["variant"] == "meshtastic_long_fast"

    async def test_lorawan_125khz_classified(self):
        """SF8 / 125 kHz → lorawan_sf8 → technology=lorawan."""
        from rfcensus.engine.lora_survey_task import LoraSurveyStats
        from rfcensus.events import DetectionEvent

        task, bus = self._make_task()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))

        bw = 125_000
        hit = self._make_hit(902_300_000, bw, sf=8)
        stats = LoraSurveyStats()
        await task._emit_detection(hit, stats)
        await bus.drain(timeout=1.0)

        assert len(captured) == 1
        d = captured[0]
        assert d.technology == "lorawan"
        assert d.metadata["estimated_sf"] == 8
        assert d.metadata["variant"] == "lorawan_sf8"

    async def test_does_not_set_needs_iq_confirmation(self):
        """Survey already DID the IQ analysis — submitting to the
        confirmation queue would just cause a redundant IQ re-capture."""
        from rfcensus.engine.lora_survey_task import LoraSurveyStats
        from rfcensus.events import DetectionEvent

        task, bus = self._make_task()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))

        hit = self._make_hit(915_000_000, 125_000, sf=8)
        stats = LoraSurveyStats()
        await task._emit_detection(hit, stats)
        await bus.drain(timeout=1.0)

        assert len(captured) == 1
        # The flag must not be set — that's the v0.6.5 design point
        # to avoid feeding the now-redundant confirmation queue.
        assert "needs_iq_confirmation" not in captured[0].metadata

    async def test_indeterminate_sf_falls_back_to_generic_lora(self):
        """When the dechirp classifier can't pin down an SF (sf=None on
        the analysis), technology falls back to generic 'lora' with no
        variant. v0.6.8: replaces the old slope-falls-back test which
        exercised slope-based estimation that no longer exists.
        """
        from rfcensus.engine.lora_survey_task import LoraSurveyStats
        from rfcensus.events import DetectionEvent

        task, bus = self._make_task()
        captured: list[DetectionEvent] = []
        bus.subscribe(DetectionEvent, lambda e: captured.append(e))

        # sf=None on the hit's chirp_analysis simulates a dechirp that
        # didn't pass the concentration/confidence gates
        hit = self._make_hit(915_000_000, 125_000, sf=None)
        stats = LoraSurveyStats()
        await task._emit_detection(hit, stats)
        await bus.drain(timeout=1.0)

        d = captured[0]
        assert d.technology == "lora"
        assert d.metadata["estimated_sf"] is None
        assert d.metadata["variant"] is None
