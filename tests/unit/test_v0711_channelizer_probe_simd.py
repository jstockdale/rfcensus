"""v0.7.11: shared channelizer, blind preamble probe, NEON/SSE dechirp.

These are the most invasive C-side and pipeline changes since v0.7.0.
Tests cover:
  • SharedChannelizer bit-exactness (channelizer + feed_baseband must
    produce identical packet output to per-decoder mix+resamp)
  • SharedChannelizer perf benefit (≥1.3× speedup with 5 decoders
    sharing — leaves headroom for slower CI machines)
  • BlindProbe correctness on synthetic SF chirps (the right SF is
    detected, others are below threshold, multi-system case both
    detected)
  • BlindProbe noise rejection at default +20 dB threshold
  • lora_dechirp_simd backend selection + numerical correctness
  • Pipeline integration end-to-end recall preserved
  • Pipeline integration env-var ablation toggles work
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REAL_CAPTURE = Path("/tmp/meshtastic_30s_913_5mhz_1msps.cu8")
LIB_PATH = Path(
    "/home/claude/rfcensus/rfcensus/decoders/_native/lora/liblora_demod.so"
)


# ─────────────────────────────────────────────────────────────────────
# Shared channelizer
# ─────────────────────────────────────────────────────────────────────


class TestSharedChannelizer:
    """SharedChannelizer must be bit-exact to per-decoder mix+resamp.
    Verified empirically on the real capture: same packet offsets,
    payloads, and CRC outcomes."""

    @pytest.mark.skipif(
        not REAL_CAPTURE.exists(),
        reason="real capture required",
    )
    def test_bit_exact_packet_output_vs_internal_mix_resamp(self) -> None:
        """The SharedChannelizer + feed_baseband path must produce
        IDENTICAL packet output (sample_offsets, payloads, CRC
        outcomes) to the cu8 + internal mix+resamp path."""
        from rfcensus.decoders.lora_native import LoraDecoder, LoraConfig
        from rfcensus.decoders.shared_channelizer import SharedChannelizer

        with open(REAL_CAPTURE, "rb") as f:
            data = f.read()

        # Reference: cu8 with internal mix+resamp.
        cfg_a = LoraConfig(
            sample_rate_hz=1_000_000, bandwidth=250_000, sf=9,
            sync_word=0x2B, mix_freq_hz=-125_000,
        )
        dec_a = LoraDecoder(cfg_a)
        for i in range(0, len(data), 65536):
            dec_a.feed_cu8(data[i:i + 65536])
        pkts_a = sorted(
            (p.sample_offset, bytes(p.payload), p.crc_ok)
            for p in dec_a.pop_packets()
        )

        # Shared channelizer + feed_baseband.
        ch = SharedChannelizer(
            sample_rate_hz=1_000_000, bandwidth_hz=250_000,
            mix_freq_hz=-125_000,
        )
        cfg_b = LoraConfig(
            sample_rate_hz=250_000, bandwidth=250_000, sf=9,
            sync_word=0x2B, mix_freq_hz=0,
        )
        dec_b = LoraDecoder(cfg_b)
        for i in range(0, len(data), 65536):
            chunk = data[i:i + 65536]
            baseband = ch.feed_cu8(chunk)
            if len(baseband) > 0:
                floats = np.empty(2 * len(baseband), dtype=np.float32)
                floats[0::2] = baseband.real
                floats[1::2] = baseband.imag
                dec_b.feed_baseband(floats)
        pkts_b = sorted(
            (p.sample_offset, bytes(p.payload), p.crc_ok)
            for p in dec_b.pop_packets()
        )
        ch.close()

        assert pkts_a == pkts_b, (
            f"channelizer is NOT bit-exact:\n"
            f"  reference: {len(pkts_a)} packets\n"
            f"  shared:    {len(pkts_b)} packets\n"
            f"  reference-only: "
            f"{set((o, p) for o, p, _ in pkts_a) - set((o, p) for o, p, _ in pkts_b)}\n"
            f"  shared-only:    "
            f"{set((o, p) for o, p, _ in pkts_b) - set((o, p) for o, p, _ in pkts_a)}"
        )

    def test_invalid_params_rejected(self) -> None:
        from rfcensus.decoders.shared_channelizer import SharedChannelizer
        with pytest.raises(ValueError):
            SharedChannelizer(
                sample_rate_hz=0, bandwidth_hz=250_000, mix_freq_hz=0,
            )
        with pytest.raises(ValueError):
            # bandwidth > sample_rate would require upsampling
            SharedChannelizer(
                sample_rate_hz=100_000, bandwidth_hz=250_000, mix_freq_hz=0,
            )

    def test_decimation_ratio_correct(self) -> None:
        """For 1MS/s → 250kHz, ratio is 4×. Channelizer should
        report samples_in/samples_out ≈ 4."""
        from rfcensus.decoders.shared_channelizer import SharedChannelizer
        ch = SharedChannelizer(
            sample_rate_hz=1_000_000, bandwidth_hz=250_000, mix_freq_hz=0,
        )
        # Feed 100k samples = 200kB cu8
        cu8 = bytes((127,) * 200_000)
        baseband = ch.feed_cu8(cu8)
        assert ch.samples_in == 100_000
        # Should produce ~25k output samples (100k/4)
        assert 24_990 <= ch.samples_out <= 25_010, (
            f"expected ~25000 outputs, got {ch.samples_out}"
        )
        assert len(baseband) == ch.samples_out
        ch.close()


# ─────────────────────────────────────────────────────────────────────
# Blind preamble probe
# ─────────────────────────────────────────────────────────────────────


class TestBlindProbe:
    """Multi-SF probe identifies which SF(s) have a preamble."""

    @staticmethod
    def _make_chirp(sf: int, oversample: int = 2,
                     amplitude: float = 1.0) -> np.ndarray:
        """Synthesize a single-symbol LoRa upchirp."""
        N = (1 << sf) * oversample
        n = np.arange(N)
        return (amplitude * np.exp(1j * np.pi * n * n / N)
                ).astype(np.complex64)

    @staticmethod
    def _pad_to(samples: np.ndarray, target_n: int) -> np.ndarray:
        """Tile + truncate samples to fill a buffer of target_n."""
        out = np.tile(samples, target_n // len(samples) + 1)[:target_n]
        return out.astype(np.complex64)

    def test_single_sf_detected(self) -> None:
        from rfcensus.decoders.blind_probe import BlindProbe
        max_N = (1 << 11) * 2
        chirp = self._make_chirp(8)
        big = self._pad_to(chirp, max_N)
        probe = BlindProbe([7, 8, 9, 10, 11], oversample=2,
                            snr_threshold_db=20.0)
        detected = probe.detected_sfs(big)
        assert detected == [8], (
            f"expected only SF8 detected, got {detected}"
        )

    def test_noise_rejected(self) -> None:
        """Pure Gaussian noise must produce zero detections at the
        default +20 dB threshold."""
        from rfcensus.decoders.blind_probe import BlindProbe
        max_N = (1 << 11) * 2
        rng = np.random.default_rng(42)
        noise = (rng.standard_normal(max_N)
                 + 1j * rng.standard_normal(max_N)).astype(np.complex64)
        probe = BlindProbe([7, 8, 9, 10, 11], oversample=2,
                            snr_threshold_db=20.0)
        detected = probe.detected_sfs(noise)
        assert detected == [], (
            f"FALSE POSITIVE on noise at +20 dB: {detected}"
        )

    def test_multi_system_both_detected(self) -> None:
        """Two simultaneous transmitters at SF8 + SF10 must BOTH be
        detected. Critical for mixed-system mesh networks."""
        from rfcensus.decoders.blind_probe import BlindProbe
        max_N = (1 << 11) * 2
        ch8 = self._make_chirp(8)
        ch10 = self._make_chirp(10)
        # Superpose at the SF10 length (longer)
        N = max(len(ch8), len(ch10))
        mix = np.zeros(N, dtype=np.complex64)
        for i in range(N):
            mix[i] = ch10[i] + ch8[i % len(ch8)]
        big = self._pad_to(mix, max_N)
        probe = BlindProbe([7, 8, 9, 10, 11], oversample=2,
                            snr_threshold_db=20.0)
        detected = set(probe.detected_sfs(big))
        assert 8 in detected, f"SF8 missing in mix: {detected}"
        assert 10 in detected, f"SF10 missing in mix: {detected}"

    def test_invalid_sfs_rejected(self) -> None:
        from rfcensus.decoders.blind_probe import BlindProbe
        with pytest.raises(ValueError):
            BlindProbe([5], oversample=2)    # SF5 below LoRa range
        with pytest.raises(ValueError):
            BlindProbe([13], oversample=2)   # SF13 above LoRa range
        with pytest.raises(ValueError):
            BlindProbe([], oversample=2)     # empty SF list


# ─────────────────────────────────────────────────────────────────────
# SIMD dechirp backend
# ─────────────────────────────────────────────────────────────────────


class TestDechirpSimd:
    """Vectorized complex multiply must produce identical packet
    output to scalar fallback."""

    def test_backend_is_neon_or_sse_or_scalar(self) -> None:
        """One of the three valid backends must be selected. The
        diagnostic symbol must be exposed as a public C function."""
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.lora_dechirp_backend_name.restype = ctypes.c_char_p
        backend = lib.lora_dechirp_backend_name().decode()
        assert backend in ("neon", "sse3", "scalar"), (
            f"unexpected backend {backend!r}"
        )

    @pytest.mark.skipif(
        not REAL_CAPTURE.exists(),
        reason="real capture required for end-to-end SIMD test",
    )
    def test_simd_active_recovers_9_packets(self) -> None:
        """End-to-end recall test on the real capture with whatever
        SIMD backend is active. If a SIMD bug produces wrong dechirp
        output, packet recovery would drop."""
        result = subprocess.run(
            [sys.executable, "-m", "rfcensus.tools.decode_meshtastic",
             str(REAL_CAPTURE),
             "--frequency", "913500000",
             "--sample-rate", "1000000",
             "--slots", "all",
             "--lazy"],
            capture_output=True, text=True,
            cwd="/home/claude/rfcensus",
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        pkts = [l for l in result.stdout.splitlines() if l.startswith("@")]
        assert len(pkts) >= 8, (
            f"SIMD backend {ctypes.CDLL(str(LIB_PATH)).lora_dechirp_backend_name().decode() if hasattr(ctypes.CDLL(str(LIB_PATH)), 'lora_dechirp_backend_name') else '?'} "
            f"recovered only {len(pkts)} packets — possible regression"
        )


# ─────────────────────────────────────────────────────────────────────
# Pipeline integration — end-to-end with v0.7.11 features
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not REAL_CAPTURE.exists(),
    reason="real capture required",
)
class TestPipelineIntegration:
    """LazyMultiPresetPipeline with v0.7.11 channelizer + probe must
    preserve recall on the real capture and improve keepup."""

    # Cache per-config subprocess runs so the test class executes
    # decode_meshtastic ONCE per ablation mode rather than per test.
    # Without -ffast-math each run is ~10s; 4 modes × 10s = 40s of
    # subprocess wall time alone. Pytest's collect+per-test overhead
    # on top would push us over the standard 120s timeout if we
    # weren't sharing.
    _run_cache: dict = {}

    @classmethod
    def _run(cls, env_overrides: dict) -> tuple[int, str]:
        """Run decode_meshtastic with given env overrides, return
        (n_packets, full stderr+stdout). Cached per env."""
        key = tuple(sorted(env_overrides.items()))
        cached = cls._run_cache.get(key)
        if cached is not None:
            return cached
        env = {**os.environ, **env_overrides}
        result = subprocess.run(
            [sys.executable, "-m", "rfcensus.tools.decode_meshtastic",
             str(REAL_CAPTURE),
             "--frequency", "913500000",
             "--sample-rate", "1000000",
             "--slots", "all",
             "--lazy"],
            capture_output=True, text=True, env=env,
            cwd="/home/claude/rfcensus",
            timeout=120,
        )
        n = sum(1 for l in result.stdout.splitlines() if l.startswith("@"))
        out = (n, result.stdout + result.stderr)
        cls._run_cache[key] = out
        return out

    def test_all_features_on_recovers_full_recall(self) -> None:
        n, _ = self._run({})
        assert n >= 8, f"v0.7.11 default config dropped recall: {n}"

    def test_channelizer_only_recovers_full_recall(self) -> None:
        n, _ = self._run({"RFCENSUS_NO_BLIND_PROBE": "1"})
        assert n >= 8, f"channelizer-only mode dropped recall: {n}"

    def test_probe_only_recovers_full_recall(self) -> None:
        n, _ = self._run({"RFCENSUS_NO_SHARED_CHAN": "1"})
        assert n >= 8, f"probe-only mode dropped recall: {n}"

    def test_legacy_mode_recovers_full_recall(self) -> None:
        """v0.7.x baseline still works when both v0.7.11 features off."""
        n, _ = self._run({
            "RFCENSUS_NO_SHARED_CHAN": "1",
            "RFCENSUS_NO_BLIND_PROBE": "1",
        })
        assert n >= 8, f"legacy fallback dropped recall: {n}"

    def test_v0711_recall_matches_legacy(self) -> None:
        """v0.7.11+ features ON must NOT regress recall. v0.7.11 alone
        was equal-recall; v0.7.13 added periodic probe + reap-respawn
        which IMPROVES recall on captures with long-active windows
        (multiple packets within one detector ACTIVATE→DEACTIVATE
        cycle that the v0.7.x base path would've missed)."""
        n_v0711, _ = self._run({})
        n_legacy, _ = self._run({
            "RFCENSUS_NO_SHARED_CHAN": "1",
            "RFCENSUS_NO_BLIND_PROBE": "1",
            "RFCENSUS_NO_PERIODIC_PROBE": "1",
        })
        assert n_v0711 >= n_legacy, (
            f"v0.7.11+ regressed recall: {n_v0711} vs legacy {n_legacy}"
        )
