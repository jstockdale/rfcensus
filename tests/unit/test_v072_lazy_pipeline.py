"""Tests for v0.7.2 part 2 — coarse-FFT lazy decoder spawning.

(Companion file to test_v072_full_passband.py which covers the
preset/slot enumeration and dedup parts of v0.7.2. This file covers
the lazy-pipeline parts.)

Three new components:
  • ``IqRingBuffer`` — circular IQ storage indexed by global offset
  • ``PassbandDetector`` — wide-FFT energy detector with per-slot state
    machines that emit activate/deactivate events
  • ``LazyMultiPresetPipeline`` — orchestrates ring + detector +
    on-demand LoraDecoder spawn/teardown

Plus a C-level test of ``lora_compute_symbols_needed()``: the early-
exit helper that lets the demod state machine emit short packets after
collecting only the symbols actually needed (rather than always
waiting for the worst-case 320 — which added 1+ second of latency
for short SF9 packets and broke the lazy pipeline's spawn/teardown
cycle).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


_NATIVE_MESHTASTIC = (
    Path(__file__).parent.parent.parent
    / "rfcensus" / "decoders" / "_native" / "meshtastic"
)
_NATIVE_LORA = (
    Path(__file__).parent.parent.parent
    / "rfcensus" / "decoders" / "_native" / "lora"
)
_REAL_CAPTURE = Path("/tmp/meshtastic_30s_913_5mhz_1msps.cu8")


def _libs_built() -> bool:
    return ((_NATIVE_MESHTASTIC / "libmeshtastic.so").exists() and
            (_NATIVE_LORA / "liblora_demod.so").exists())


# ─────────────────────────────────────────────────────────────────────
# IqRingBuffer
# ─────────────────────────────────────────────────────────────────────

class TestIqRingBuffer:
    def test_basic_write_and_read(self) -> None:
        from rfcensus.utils.iq_ring import IqRingBuffer
        ring = IqRingBuffer(capacity_samples=10)
        ring.write(bytes(range(10)))    # 5 samples
        assert ring.total_written == 5
        assert ring.oldest_offset == 0
        assert ring.newest_offset == 4
        assert ring.read(0, 5) == bytes(range(10))
        assert ring.read(2, 2) == bytes([4, 5, 6, 7])

    def test_wraparound_preserves_offsets(self) -> None:
        """After wrapping, samples are still addressable by the same
        global offset they were written under."""
        from rfcensus.utils.iq_ring import IqRingBuffer
        ring = IqRingBuffer(capacity_samples=10)
        ring.write(bytes(range(10)))           # samples 0..4
        ring.write(bytes(range(100, 116)))     # samples 5..12 → wraps
        assert ring.total_written == 13
        assert ring.oldest_offset == 3
        # Sample 3 should still return its original byte pair (6, 7)
        assert ring.read(3, 1) == bytes([6, 7])
        # Read across the wrap point
        assert ring.read(3, 10) == (
            bytes([6, 7, 8, 9]) + bytes(range(100, 116))
        )

    def test_too_old_returns_none(self) -> None:
        from rfcensus.utils.iq_ring import IqRingBuffer
        ring = IqRingBuffer(capacity_samples=4)
        ring.write(bytes(range(20)))   # 10 samples; only last 4 retained
        assert ring.oldest_offset == 6
        assert ring.read(0, 1) is None   # too old
        assert ring.read(6, 4) == bytes(range(12, 20))

    def test_beyond_newest_returns_none(self) -> None:
        from rfcensus.utils.iq_ring import IqRingBuffer
        ring = IqRingBuffer(capacity_samples=10)
        ring.write(bytes(range(10)))   # 5 samples
        assert ring.read(4, 2) is None   # would need sample 5

    def test_chunk_larger_than_capacity_keeps_tail(self) -> None:
        """When a single write is bigger than the ring, only the tail
        is retained, but the offset arithmetic stays consistent."""
        from rfcensus.utils.iq_ring import IqRingBuffer
        ring = IqRingBuffer(capacity_samples=4)
        ring.write(bytes(range(20)))   # 10 samples in one call
        assert ring.total_written == 10
        # The retained samples are the last 4 (samples 6..9)
        assert ring.read(6, 4) == bytes(range(12, 20))

    def test_exact_capacity_write(self) -> None:
        from rfcensus.utils.iq_ring import IqRingBuffer
        ring = IqRingBuffer(capacity_samples=5)
        ring.write(bytes(range(10)))   # exactly capacity
        assert ring.total_written == 5
        assert ring.read(0, 5) == bytes(range(10))

    def test_read_recent(self) -> None:
        from rfcensus.utils.iq_ring import IqRingBuffer
        ring = IqRingBuffer(capacity_samples=10)
        ring.write(bytes(range(20)))   # 10 samples
        assert ring.read_recent(2) == bytes([16, 17, 18, 19])
        assert ring.read_recent(11) is None   # only 10 written

    def test_odd_byte_write_rejected(self) -> None:
        from rfcensus.utils.iq_ring import IqRingBuffer
        ring = IqRingBuffer(capacity_samples=10)
        with pytest.raises(ValueError):
            ring.write(bytes([1, 2, 3]))

    def test_zero_capacity_rejected(self) -> None:
        from rfcensus.utils.iq_ring import IqRingBuffer
        with pytest.raises(ValueError):
            IqRingBuffer(capacity_samples=0)


# ─────────────────────────────────────────────────────────────────────
# PassbandDetector
# ─────────────────────────────────────────────────────────────────────

def _make_noise_cu8(n_samples: int, scale: float = 10.0,
                    seed: int = 42) -> bytes:
    rng = np.random.default_rng(seed)
    iq = rng.normal(127.5, scale, (n_samples, 2)).clip(0, 255).astype(np.uint8)
    return iq.flatten().tobytes()


def _make_tone_cu8(n_samples: int, freq_hz_offset: float,
                   sample_rate: int = 2_400_000,
                   amp_dbfs: float = -20.0, seed: int = 43) -> bytes:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sample_rate
    amp_lin = 10**(amp_dbfs / 20)
    iq = (np.exp(2j * np.pi * freq_hz_offset * t) * amp_lin * 127
          + rng.normal(0, 5, n_samples) + 1j*rng.normal(0, 5, n_samples))
    iq += 127.5 + 127.5j
    i = iq.real.clip(0, 255).astype(np.uint8)
    q = iq.imag.clip(0, 255).astype(np.uint8)
    return np.stack([i, q], axis=-1).flatten().tobytes()


class TestPassbandDetector:
    def test_no_activations_on_pure_noise(self) -> None:
        """The detector must not fire spurious activations on white
        noise within its bootstrap+release thresholds."""
        from rfcensus.decoders.passband_detector import (
            PassbandDetector, DetectorConfig,
        )
        cfg = DetectorConfig(
            sample_rate_hz=2_400_000, center_freq_hz=915_000_000,
            fft_size=512, hop_samples=256,
            bootstrap_frames=50, drain_frames=50,
        )
        det = PassbandDetector(
            config=cfg,
            slot_freqs_hz=[915_500_000, 914_500_000],
            slot_bandwidths_hz=[250_000, 250_000],
        )
        events: list = []
        for _ in range(3):
            events.extend(det.feed_cu8(_make_noise_cu8(65536)))
        assert events == [], f"unexpected events on noise: {events}"

    def test_activates_on_tone(self) -> None:
        """A tone at slot N's frequency activates that slot, not the
        other one."""
        from rfcensus.decoders.passband_detector import (
            PassbandDetector, DetectorConfig,
        )
        cfg = DetectorConfig(
            sample_rate_hz=2_400_000, center_freq_hz=915_000_000,
            fft_size=512, hop_samples=256,
            bootstrap_frames=20, drain_frames=20,
        )
        det = PassbandDetector(
            config=cfg,
            slot_freqs_hz=[915_500_000, 914_000_000],
            slot_bandwidths_hz=[250_000, 250_000],
        )
        # Bootstrap on noise
        for _ in range(2):
            list(det.feed_cu8(_make_noise_cu8(65536)))
        # Tone at +500kHz → slot 915.5 (+500 from center)
        events = list(det.feed_cu8(_make_tone_cu8(65536, 500_000)))
        activates = [e for e in events if e.kind == "activate"]
        assert any(e.slot_freq_hz == 915_500_000 for e in activates), (
            f"expected activate on 915.5MHz; events={activates}"
        )
        # Far-off slot should NOT activate
        assert not any(e.slot_freq_hz == 914_000_000 for e in activates)

    def test_deactivates_when_signal_goes_away(self) -> None:
        from rfcensus.decoders.passband_detector import (
            PassbandDetector, DetectorConfig,
        )
        cfg = DetectorConfig(
            sample_rate_hz=2_400_000, center_freq_hz=915_000_000,
            fft_size=512, hop_samples=256,
            bootstrap_frames=20, drain_frames=20,
        )
        det = PassbandDetector(
            config=cfg,
            slot_freqs_hz=[915_500_000],
            slot_bandwidths_hz=[250_000],
        )
        list(det.feed_cu8(_make_noise_cu8(65536)))
        list(det.feed_cu8(_make_noise_cu8(65536)))
        # Activate
        events = list(det.feed_cu8(_make_tone_cu8(65536, 500_000)))
        assert any(e.kind == "activate" for e in events)
        # Drain back to silence
        events_after = []
        for _ in range(3):
            events_after.extend(det.feed_cu8(_make_noise_cu8(65536)))
        assert any(e.kind == "deactivate" for e in events_after), (
            f"expected deactivate; got {events_after}"
        )

    def test_sample_offset_matches_input_position(self) -> None:
        """SlotEvent.sample_offset is in INPUT-stream samples (the
        same units as bytes-into-cu8 / 2). Necessary for the lazy
        pipeline to address the ring buffer correctly."""
        from rfcensus.decoders.passband_detector import (
            PassbandDetector, DetectorConfig,
        )
        cfg = DetectorConfig(
            sample_rate_hz=2_400_000, center_freq_hz=915_000_000,
            fft_size=512, hop_samples=256,
            bootstrap_frames=20, drain_frames=200,
        )
        det = PassbandDetector(
            config=cfg,
            slot_freqs_hz=[915_500_000],
            slot_bandwidths_hz=[250_000],
        )
        list(det.feed_cu8(_make_noise_cu8(65536)))
        list(det.feed_cu8(_make_noise_cu8(65536)))
        events = list(det.feed_cu8(_make_tone_cu8(65536, 500_000)))
        activates = [e for e in events if e.kind == "activate"]
        assert activates
        # We've fed 3 × 65536 = 196608 input samples. The activation
        # should fall within the third chunk, i.e. offset ∈ [131072, 196608].
        for e in activates:
            assert 131072 <= e.sample_offset <= 196608 + 512


# ─────────────────────────────────────────────────────────────────────
# lora_compute_symbols_needed (C-level via direct decoder behavior)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _libs_built() or not _REAL_CAPTURE.exists(),
                    reason="needs native libs + real capture")
class TestEarlyEmit:
    def test_short_packet_emits_before_max_syms(self) -> None:
        """The v0.7.2 early-emit fix should let a 49-byte SF9 packet
        emerge with ≤ ~70 symbols collected, not the v0.7.x worst-
        case waterfall of 320 symbols.

        We can't directly observe symbols_collected from Python, but
        we can verify the BEHAVIOR: feed exactly enough samples to
        cover the packet's preamble + payload (with a small margin),
        then check that the packet emerges. If the early-emit weren't
        working, the decoder would still be waiting for samples that
        we never feed and the packet would never come out."""
        from rfcensus.decoders.lora_native import LoraDecoder, LoraConfig

        cfg = LoraConfig(
            sample_rate_hz=1_000_000, bandwidth=250_000, sf=9,
            sync_word=0x2B, mix_freq_hz=375_000,
        )
        # Packet 1 is at file output-sample 113280 (= 0.453s into
        # the 1MS/s stream). With early-emit it should emerge once
        # we've fed ~700k input samples. v0.7.x would need 1.5M+.
        with open(_REAL_CAPTURE, "rb") as f:
            samples = f.read(700_000 * 2)
        dec = LoraDecoder(cfg)
        dec.feed_cu8(samples)
        pkts = list(dec.pop_packets())
        assert len(pkts) >= 1, (
            f"early-emit not working: 700k samples should produce ≥1 "
            f"packet but got {len(pkts)}. stats={dec.stats()}"
        )
        # The first-packet payload is 49 bytes
        assert any(p.crc_ok and p.payload_len == 49 for p in pkts), (
            f"expected the 49-byte CRC-ok packet; got "
            f"{[(p.payload_len, p.crc_ok) for p in pkts]}"
        )

    def test_header_crc_failure_does_not_stall(self) -> None:
        """If the first 8 symbols decode to a header with bad CRC,
        the decoder should bail back to DETECT immediately rather
        than waste samples collecting up to MAX_SYMS. We test this
        indirectly: pure noise should not fill up the demod buffer
        and stall the decoder. After feeding lots of noise, the
        decoder should still be able to find a real packet that
        comes after."""
        from rfcensus.decoders.lora_native import LoraDecoder, LoraConfig

        cfg = LoraConfig(
            sample_rate_hz=1_000_000, bandwidth=250_000, sf=9,
            sync_word=0x2B, mix_freq_hz=375_000,
        )
        # Feed 500k samples of noise, then real data
        rng = np.random.default_rng(99)
        noise = rng.normal(127.5, 10, (500_000, 2)).clip(0, 255).astype(np.uint8)
        noise_bytes = noise.flatten().tobytes()

        with open(_REAL_CAPTURE, "rb") as f:
            real_samples = f.read(2_000_000 * 2)

        dec = LoraDecoder(cfg)
        dec.feed_cu8(noise_bytes)
        dec.feed_cu8(real_samples)
        pkts = list(dec.pop_packets())
        assert len(pkts) >= 1, (
            f"decoder stalled after noise: stats={dec.stats()}"
        )


# ─────────────────────────────────────────────────────────────────────
# LazyMultiPresetPipeline — end-to-end on real capture
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _libs_built() or not _REAL_CAPTURE.exists(),
                    reason="needs native libs + real capture")
class TestLazyPipeline:
    @staticmethod
    def _build_pipe():
        from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
        from rfcensus.decoders.lazy_pipeline import LazyMultiPresetPipeline
        from rfcensus.decoders.passband_detector import DetectorConfig
        from rfcensus.utils.meshtastic_region import (
            enumerate_all_slots_in_passband, default_slot, PRESETS,
            PresetSlot, REGIONS,
        )
        slots = enumerate_all_slots_in_passband(
            "US", 913_500_000, 1_000_000,
        )
        # Add the MEDIUM_FAST default slot (just outside the passband
        # at 1MS/s due to edge guard, but actually present in the
        # capture)
        slots.append(default_slot("US", "MEDIUM_FAST"))
        # Plus other BW=250 presets at 913.125 (the actual MEDIUM_FAST
        # transmit slot)
        for k in ["LONG_FAST", "MEDIUM_SLOW", "SHORT_FAST", "SHORT_SLOW"]:
            slots.append(PresetSlot(
                preset=PRESETS[k], region=REGIONS["US"],
                channel_name="", slot=44, num_slots=104,
                freq_hz=913_125_000,
            ))
        mesh = MeshtasticDecoder("LONG_FAST")
        for k in PRESETS:
            mesh.add_channel(PRESETS[k].display_name,
                             psk=b"\x01", is_primary=False)
        det_cfg = DetectorConfig(
            sample_rate_hz=1_000_000, center_freq_hz=913_500_000,
            fft_size=512, hop_samples=256,
            bootstrap_frames=200, drain_frames=200,
            trigger_threshold_db=8.0, release_threshold_db=4.0,
        )
        return LazyMultiPresetPipeline(
            sample_rate_hz=1_000_000, center_freq_hz=913_500_000,
            candidate_slots=slots, mesh=mesh,
            detector_config=det_cfg, ring_buffer_ms=300.0,
        )

    def test_catches_known_packets(self) -> None:
        """The lazy pipeline should catch at least the 6 known
        decryptable packets (matching the eager-spawn baseline). It
        may catch a few extras due to periodic decoder restarts at
        slot activations recovering from state-machine confusion the
        long-running decoder gets into."""
        from rfcensus.utils.iq_source import FileIQSource
        pipe = self._build_pipe()
        n_decrypted = 0
        with FileIQSource(_REAL_CAPTURE) as src:
            for chunk in src:
                pipe.feed_cu8(chunk)
                for pp in pipe.pop_packets():
                    if pp.decrypted:
                        n_decrypted += 1
        assert n_decrypted >= 6, (
            f"lazy pipeline decrypted only {n_decrypted}, expected ≥6"
        )

    def test_offsets_are_absolute_input_samples(self) -> None:
        """Emitted ``sample_offset`` values must be in absolute INPUT-
        stream samples, not decoder-local OUTPUT samples. We verify
        by checking that one of the known packets (real time t=0.453s
        in the file) emerges with sample_offset close to 453000."""
        from rfcensus.utils.iq_source import FileIQSource
        pipe = self._build_pipe()
        offsets = []
        with FileIQSource(_REAL_CAPTURE) as src:
            for chunk in src:
                pipe.feed_cu8(chunk)
                for pp in pipe.pop_packets():
                    if pp.decrypted:
                        offsets.append(pp.lora.sample_offset)
        assert offsets
        # The first decryptable packet is at file t=0.453s = 453000
        # input samples at 1MS/s. Allow a generous ±100k for detector
        # latency / lookback alignment.
        assert any(350_000 < o < 600_000 for o in offsets), (
            f"first packet's offset doesn't look like input samples: "
            f"{offsets}"
        )

    def test_dedup_collapses_duplicates(self) -> None:
        """Multiple decoders in the same active slot AND adjacent
        slot decoders catching the same physical signal should be
        deduped by ``pop_packets(dedup=True)`` (the default)."""
        from rfcensus.utils.iq_source import FileIQSource
        pipe = self._build_pipe()
        # Force dedup=True (default) and count
        n_with_dedup = 0
        with FileIQSource(_REAL_CAPTURE) as src:
            for chunk in src:
                pipe.feed_cu8(chunk)
                for _ in pipe.pop_packets(dedup=True):
                    n_with_dedup += 1
        # Without dedup we'd see many more (each spawn emits the
        # same packet, and adjacent slots also trigger). With dedup,
        # ~the count of unique physical packets.
        assert 6 <= n_with_dedup <= 20, (
            f"dedup should collapse to ~7-15 packets, got {n_with_dedup}"
        )

    def test_stats_track_spawn_teardown(self) -> None:
        """Spawn/teardown counts should be balanced and non-zero."""
        from rfcensus.utils.iq_source import FileIQSource
        pipe = self._build_pipe()
        with FileIQSource(_REAL_CAPTURE) as src:
            for chunk in src:
                pipe.feed_cu8(chunk)
                for _ in pipe.pop_packets():
                    pass
        s = pipe.lazy_stats
        assert s.slot_activations > 0
        assert s.slot_deactivations > 0
        assert s.decoders_spawned > 0
        # Within a small slack (the final state may have decoders
        # still active when the IQ stream ends)
        assert (s.decoders_spawned - s.decoders_torn_down
                  <= pipe.n_active_decoders + 5)

    def test_stats_method_returns_per_preset(self) -> None:
        """``stats()`` must return per-preset LoraStats matching the
        eager pipeline's interface, so the CLI can use either."""
        from rfcensus.utils.iq_source import FileIQSource
        pipe = self._build_pipe()
        with FileIQSource(_REAL_CAPTURE) as src:
            for chunk in src:
                pipe.feed_cu8(chunk)
                for _ in pipe.pop_packets():
                    pass
        per_preset = pipe.stats()
        assert isinstance(per_preset, dict)
        # MEDIUM_FAST should have decoded packets in this capture
        assert "MEDIUM_FAST" in per_preset
        mf = per_preset["MEDIUM_FAST"]
        assert mf.preambles_found > 0
        assert mf.packets_decoded >= 6
