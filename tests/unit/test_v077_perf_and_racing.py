"""v0.7.7: SF racing, RSSI/SNR computation, ring overflow visibility,
keep-up tracking, and aggressive compiler flag wiring.

These tests focus on behavior visible from Python — the C-side
RSSI/SNR computation is exercised via real-capture decoding
since synthesizing valid LoRa preamble IQ in pure Python isn't
worth the effort here (we trust the existing native test_synth
harness for that).
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────
# IqRingBuffer overflow visibility (C in the v0.7.7 plan)
# ─────────────────────────────────────────────────────────────────────


class TestRingBufferOverflowVisibility:
    """v0.7.7 added overflow_events + samples_dropped properties to
    IqRingBuffer so the lazy pipeline can surface CPU-saturation
    diagnostics. Previously the buffer dropped data silently when
    a single write() exceeded capacity."""

    def test_no_overflow_when_writes_fit(self) -> None:
        from rfcensus.utils.iq_ring import IqRingBuffer
        ring = IqRingBuffer(capacity_samples=1000)
        # 500 samples = 1000 bytes, well under capacity
        ring.write(b"\x00" * 1000)
        assert ring.overflow_events == 0
        assert ring.samples_dropped == 0

    def test_no_overflow_on_normal_wrap(self) -> None:
        """Normal ring wrap (writes exceed capacity over time but each
        individual write fits) is NOT an overflow event — that's the
        ring's normal mode of operation."""
        from rfcensus.utils.iq_ring import IqRingBuffer
        ring = IqRingBuffer(capacity_samples=100)
        # Write 5× capacity in chunks that each fit
        for _ in range(10):
            ring.write(b"\x42" * 100)    # 50 samples each, fits
        assert ring.overflow_events == 0
        assert ring.samples_dropped == 0
        # Total written should still be coherent
        assert ring.total_written == 500

    def test_overflow_when_single_write_exceeds_capacity(self) -> None:
        """The specific overflow case: one write() chunk strictly
        larger than the ring. Previously silent; now counted."""
        from rfcensus.utils.iq_ring import IqRingBuffer
        ring = IqRingBuffer(capacity_samples=100)
        # Write 300 samples (600 bytes) into a 100-sample ring
        ring.write(b"\x42" * 600)
        assert ring.overflow_events == 1
        # 300 samples written, 100 fit, 200 dropped
        assert ring.samples_dropped == 200

    def test_overflow_count_accumulates(self) -> None:
        from rfcensus.utils.iq_ring import IqRingBuffer
        ring = IqRingBuffer(capacity_samples=50)
        ring.write(b"\x00" * 200)    # 100 samples → 50 dropped
        ring.write(b"\x00" * 300)    # 150 samples → 100 dropped
        assert ring.overflow_events == 2
        assert ring.samples_dropped == 150


# ─────────────────────────────────────────────────────────────────────
# LazyPipelineStats new counters (B + C in the plan)
# ─────────────────────────────────────────────────────────────────────


class TestLazyPipelineStatsHasV077Counters:
    def test_racing_counters_present(self) -> None:
        from rfcensus.decoders.lazy_pipeline import LazyPipelineStats
        s = LazyPipelineStats()
        assert s.racing_wins == 0
        assert s.racing_losers_killed == 0
        assert s.racing_unresolved == 0

    def test_drop_counters_present(self) -> None:
        from rfcensus.decoders.lazy_pipeline import LazyPipelineStats
        s = LazyPipelineStats()
        assert s.ring_overflows == 0
        assert s.samples_dropped == 0


# ─────────────────────────────────────────────────────────────────────
# Active slot has racing scaffolding
# ─────────────────────────────────────────────────────────────────────


class TestActiveSlotRacingScaffold:
    def test_active_slot_has_race_resolved(self) -> None:
        from rfcensus.decoders.lazy_pipeline import _ActiveSlot
        a = _ActiveSlot(
            freq_hz=915_000_000,
            bandwidth_hz=250_000,
            activated_sample_offset=0,
            next_sample_offset=0,
            feed_start_offset=0,
        )
        # Default: race not yet resolved (until set otherwise)
        assert a.race_resolved is False

    def test_active_slot_race_resolved_can_be_set(self) -> None:
        """Single-preset path marks race resolved at construction."""
        from rfcensus.decoders.lazy_pipeline import _ActiveSlot
        a = _ActiveSlot(
            freq_hz=915_000_000,
            bandwidth_hz=125_000,
            activated_sample_offset=0,
            next_sample_offset=0,
            feed_start_offset=0,
            race_resolved=True,
        )
        assert a.race_resolved is True


# ─────────────────────────────────────────────────────────────────────
# Pipeline construction & keep-up tracking
# ─────────────────────────────────────────────────────────────────────


class TestLazyPipelineKeepupRatio:
    """Lazy pipeline gained a wall-clock vs audio-clock keep-up ratio
    surfaced via the ``keepup_ratio`` property. Used by the standalone
    tool to warn the user when CPU is saturated."""

    def test_keepup_ratio_zero_initially(self) -> None:
        from rfcensus.decoders.lazy_pipeline import LazyMultiPresetPipeline
        from rfcensus.utils.meshtastic_region import default_slot
        from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
        slot = default_slot("US", "MEDIUM_FAST")
        mesh = MeshtasticDecoder("MEDIUM_FAST")
        mesh.add_channel("MediumFast", psk=b"\x01")
        pipe = LazyMultiPresetPipeline(
            sample_rate_hz=1_000_000,
            center_freq_hz=915_000_000,
            candidate_slots=[slot],
            mesh=mesh,
        )
        assert pipe.keepup_ratio == 0.0

    def test_keepup_ratio_populates_after_feed(self) -> None:
        from rfcensus.decoders.lazy_pipeline import LazyMultiPresetPipeline
        from rfcensus.utils.meshtastic_region import default_slot
        from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
        slot = default_slot("US", "MEDIUM_FAST")
        mesh = MeshtasticDecoder("MEDIUM_FAST")
        mesh.add_channel("MediumFast", psk=b"\x01")
        pipe = LazyMultiPresetPipeline(
            sample_rate_hz=1_000_000,
            center_freq_hz=915_000_000,
            candidate_slots=[slot],
            mesh=mesh,
        )
        # Feed 100ms of zeros (no signal, just exercising the loop)
        pipe.feed_cu8(b"\x80" * 200_000)
        # Now keepup_ratio should be > 0 (we did real work)
        assert pipe.keepup_ratio > 0.0
        # And much less than 1.0 — feeding zeros is fast
        assert pipe.keepup_ratio < 1.0


# ─────────────────────────────────────────────────────────────────────
# Ring buffer default bumped 300 → 500ms (F in the plan)
# ─────────────────────────────────────────────────────────────────────


class TestRingBufferDefault:
    def test_lazy_pipeline_default_ring_is_500ms(self) -> None:
        """v0.7.7: bumped from 300 to 500ms for safety against
        bursty consumer scheduling. At 2.4 MS/s that's 2.4 MB
        of cu8 — trivial RAM cost for substantial safety margin."""
        import inspect
        from rfcensus.decoders.lazy_pipeline import LazyMultiPresetPipeline
        sig = inspect.signature(LazyMultiPresetPipeline.__init__)
        ring_param = sig.parameters["ring_buffer_ms"]
        assert ring_param.default == 500.0


# ─────────────────────────────────────────────────────────────────────
# Compiler flag aggressiveness (D in the plan)
# ─────────────────────────────────────────────────────────────────────


class TestCompilerFlagsAggressive:
    """v0.7.7 bumped both native Makefiles to -O3 + arch-specific
    aggressive flags. Verify the Makefiles encode this contract so
    a future well-meaning reviewer doesn't quietly revert it."""

    def test_lora_makefile_uses_o3(self) -> None:
        text = Path(
            "/home/claude/rfcensus/rfcensus/decoders/_native/lora/Makefile"
        ).read_text()
        # -O3 is in the LORA_OPT default
        assert "LORA_OPT ?= -O3" in text
        # -ffast-math + -funroll-loops are on the default opt line
        assert "-ffast-math" in text
        assert "-funroll-loops" in text

    def test_lora_makefile_arch_native(self) -> None:
        text = Path(
            "/home/claude/rfcensus/rfcensus/decoders/_native/lora/Makefile"
        ).read_text()
        # ARM gets -mcpu=native (NEON via implicit), x86 gets
        # -march=native (SSE/AVX)
        assert "-mcpu=native" in text
        assert "-march=native" in text

    def test_meshtastic_makefile_uses_o3(self) -> None:
        text = Path(
            "/home/claude/rfcensus/rfcensus/decoders/_native/meshtastic/Makefile"
        ).read_text()
        assert "MESH_OPT ?= -O3" in text


# ─────────────────────────────────────────────────────────────────────
# RSSI/SNR populated on real packets (A in the plan)
# ─────────────────────────────────────────────────────────────────────


REAL_CAPTURE = Path("/tmp/meshtastic_30s_913_5mhz_1msps.cu8")


@pytest.mark.skipif(
    not REAL_CAPTURE.exists(),
    reason="real-traffic capture required",
)
class TestRssiSnrPopulated:
    """v0.7.7 wrote actual values to the rssi_db / snr_db fields that
    were previously always 0.0. Verify against the real capture that
    the values are sane: RSSI in dBFS (negative for sub-full-scale
    signals, typically -1 to -25 for normal Meshtastic), SNR in dB
    above noise (typically +5 to +25 for clean signals)."""

    def _decode(self):
        from rfcensus.decoders.lora_native import LoraDecoder, LoraConfig
        cfg = LoraConfig(
            sample_rate_hz=1_000_000,
            bandwidth=250_000,
            sf=9,
            sync_word=0x2B,
            mix_freq_hz=375_000,
        )
        dec = LoraDecoder(cfg)
        data = REAL_CAPTURE.read_bytes()
        chunk = 65536
        for i in range(0, len(data), chunk):
            dec.feed_cu8(data[i:i+chunk])
        return list(dec.pop_packets())

    def test_rssi_is_populated_and_sane(self) -> None:
        pkts = self._decode()
        assert len(pkts) > 0, "expected packets in real capture"
        # All packets should have RSSI in a sane range. Strong
        # signals in this capture: ~-1 to -4 dBFS. Allow wider
        # range to absorb future capture variations.
        for p in pkts:
            assert -40.0 < p.rssi_db < 0.0, (
                f"RSSI {p.rssi_db} dBFS out of expected range"
            )

    def test_snr_is_populated_and_sane(self) -> None:
        pkts = self._decode()
        assert len(pkts) > 0
        # SNR should be positive (signal above noise) for valid
        # packets. Typical Meshtastic clean signals: +10 to +25 dB.
        for p in pkts:
            if p.crc_ok:    # only check valid packets
                assert p.snr_db > 5.0, (
                    f"SNR {p.snr_db} dB suspiciously low for valid packet"
                )
                assert p.snr_db < 50.0, (
                    f"SNR {p.snr_db} dB unrealistically high"
                )

    def test_rssi_snr_values_vary_across_packets(self) -> None:
        """Sanity: not all packets have identical RSSI/SNR (which
        would indicate values are hardcoded constants)."""
        pkts = self._decode()
        if len(pkts) < 2:
            pytest.skip("need ≥2 packets for variance check")
        rssi_values = {round(p.rssi_db, 1) for p in pkts}
        snr_values = {round(p.snr_db, 1) for p in pkts if p.crc_ok}
        assert len(rssi_values) > 1, (
            "all packets have identical RSSI — likely hardcoded"
        )
        assert len(snr_values) > 1, (
            "all valid packets have identical SNR — likely hardcoded"
        )


# ─────────────────────────────────────────────────────────────────────
# End-to-end: condition-based racing maintains packet recall
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not REAL_CAPTURE.exists(),
    reason="real-traffic capture required",
)
class TestRacingMaintainsRecall:
    """Critical regression: the v0.7.7 decoder racing must NOT lose
    legitimate packets. The condition-based design (kill losers when
    a winner locks) preserves packet recall — the original time-
    deadline design dropped 7-of-9 at 2× safety. This test pins the
    contract."""

    def _run_pipeline(self, slots_arg="all"):
        """Run the standalone tool against the real capture, return
        list of packet lines (offsets only — full stdout would be
        too large for a test fixture)."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "rfcensus.tools.decode_meshtastic",
             str(REAL_CAPTURE),
             "--frequency", "913500000",
             "--sample-rate", "1000000",
             "--slots", slots_arg,
             "--lazy"],
            capture_output=True, text=True,
            cwd="/home/claude/rfcensus",
            timeout=60,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"decoder failed: {result.stderr}"
            )
        # Packet lines start with @ offset
        lines = [l for l in result.stdout.splitlines()
                 if l.startswith("@")]
        return lines

    def test_lazy_with_racing_finds_at_least_8_packets(self) -> None:
        """The capture has 9 detectable Meshtastic packets. With
        condition-based racing we expect to recover all of them
        (allow 8 as a safety margin against minor timing drift)."""
        lines = self._run_pipeline("all")
        assert len(lines) >= 8, (
            f"racing dropped packets: only {len(lines)}/9 recovered"
        )
