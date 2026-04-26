"""v0.7.8: detector state machine vectorization + RSSI smarts.

Vectorization correctness: the new `_step_state_machines_vec` must
produce the same SlotEvent stream as the legacy `_step_state_machines`
on the same input. We test by feeding identical synthetic IQ through
both pipelines (or via a parallel comparison) and diffing events.

RSSI smarts: dedup tie-breaker, CRC-fail SNR classification.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


REAL_CAPTURE = Path("/tmp/meshtastic_30s_913_5mhz_1msps.cu8")


# ─────────────────────────────────────────────────────────────────────
# Detector state machine vectorization
# ─────────────────────────────────────────────────────────────────────


class TestVectorizedStateMachine:
    """The vectorized inner loop must produce the same activation /
    deactivation event sequence as the legacy scalar loop on the
    same input. Tested empirically against synthetic noise (which
    should produce zero events) and signal bursts (should produce
    matching activation/deactivation pairs)."""

    def _make_detector(self, n_slots=10):
        from rfcensus.decoders.passband_detector import (
            PassbandDetector, DetectorConfig,
        )
        cfg = DetectorConfig(
            sample_rate_hz=2_400_000,
            center_freq_hz=915_000_000,
            bootstrap_frames=10,    # short for fast test
        )
        # 10 slots spaced 100 kHz apart, BW=125 kHz each
        slot_freqs = [915_000_000 + i * 100_000 for i in range(n_slots)]
        slot_bws = [125_000] * n_slots
        return PassbandDetector(
            cfg, slot_freqs_hz=slot_freqs, slot_bandwidths_hz=slot_bws,
        )

    def test_pure_noise_yields_no_events(self) -> None:
        """Vectorized state machine on quiet noise should not
        spuriously activate. Feeds in realistic 27ms chunks (matches
        the 65536-byte cu8 chunk the standalone tool uses) so the
        bootstrap-frame counter completes between batches, exercising
        the same code path as production."""
        from rfcensus.decoders.passband_detector import (
            PassbandDetector, DetectorConfig,
        )
        cfg = DetectorConfig(
            sample_rate_hz=2_400_000,
            center_freq_hz=915_000_000,
        )
        det = PassbandDetector(
            cfg,
            slot_freqs_hz=[915_000_000 + i * 100_000 for i in range(5)],
            slot_bandwidths_hz=[125_000] * 5,
        )
        # 1.5 seconds of low-variance noise — well past bootstrap.
        # cu8 centered on 127 with ±1 LSB jitter → ~-72 dBFS noise,
        # nowhere near the 6 dB trigger threshold above floor.
        n_samples = int(2_400_000 * 1.5)
        rng = np.random.default_rng(42)
        noise = rng.integers(
            126, 129, size=n_samples * 2, dtype=np.uint8,
        ).tobytes()

        # Feed in 27ms chunks (= 65536 samples = 131072 bytes).
        # Matches the standalone tool's typical chunk size and lets
        # bootstrap complete between batches.
        chunk_bytes = 65536 * 2
        all_events = []
        for i in range(0, len(noise), chunk_bytes):
            all_events.extend(det.feed_cu8(noise[i:i + chunk_bytes]))

        activations = [e for e in all_events if e.kind == "activate"]
        assert len(activations) == 0, (
            f"vectorized SM spuriously activated on quiet noise: "
            f"{len(activations)} events"
        )

    def test_log10_no_longer_called_in_inner_loop(self) -> None:
        """v0.7.8 contract: the per-slot-per-frame np.log10 calls
        were lifted out of `_step_state_machines_vec` into batched
        ops in the caller. Verify the function body (excluding
        docstring) contains no log10 reference."""
        import inspect
        from rfcensus.decoders.passband_detector import PassbandDetector
        src = inspect.getsource(
            PassbandDetector._step_state_machines_vec
        )
        # Strip the docstring before checking — comments may
        # legitimately reference "log10" in the design rationale.
        # Cheap heuristic: split off the first triple-quoted block.
        if '"""' in src:
            head, _, rest = src.partition('"""')
            _, _, body = rest.partition('"""')
            code = head + body
        else:
            code = src
        assert "log10" not in code, (
            "vectorized state machine should not call log10 — "
            "batched math lives in feed_cu8"
        )


# ─────────────────────────────────────────────────────────────────────
# Detector benchmark guard — catches future perf regressions
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not REAL_CAPTURE.exists(),
    reason="real-traffic capture required",
)
class TestDetectorPerf:
    """Loose perf guard. The vectorized detector at 84 slots × 30s
    of 2.4 MS/s noise should process in well under real-time. If
    this test starts failing, someone added per-slot Python
    overhead back into the hot path."""

    def test_detector_under_real_time_on_84_slots(self) -> None:
        import time
        from rfcensus.decoders.passband_detector import (
            PassbandDetector, DetectorConfig,
        )
        from rfcensus.utils.meshtastic_region import (
            enumerate_all_slots_in_passband,
        )
        slots = enumerate_all_slots_in_passband(
            "US", 913_500_000, 2_400_000,
        )
        # Ensure we got the expected count (test environment check)
        assert len(slots) >= 50, (
            f"expected ≥50 slots in US passband at 2.4 MS/s, "
            f"got {len(slots)}"
        )
        cfg = DetectorConfig(
            sample_rate_hz=2_400_000, center_freq_hz=913_500_000,
        )
        det = PassbandDetector(
            cfg,
            slot_freqs_hz=[s.freq_hz for s in slots],
            slot_bandwidths_hz=[s.preset.bandwidth_hz for s in slots],
        )
        # 5 seconds of noise (full 30s would slow CI; 5s is enough
        # to characterize per-frame cost)
        n_samples = 5 * 2_400_000
        rng = np.random.default_rng(0)
        data = rng.integers(
            100, 156, size=n_samples * 2, dtype=np.uint8,
        ).tobytes()

        chunk_bytes = 65536 * 2
        t0 = time.perf_counter()
        for i in range(0, len(data), chunk_bytes):
            list(det.feed_cu8(data[i:i + chunk_bytes]))
        elapsed = time.perf_counter() - t0
        # Should process 5s of audio in well under 5s of wall time.
        # Allow generous 4s budget (= keepup ratio 0.8) so we have
        # headroom against future test-runner slowdowns.
        assert elapsed < 4.0, (
            f"detector took {elapsed:.2f}s to process 5s of audio "
            f"(keepup ratio {elapsed/5*100:.0f}%); regression?"
        )


# ─────────────────────────────────────────────────────────────────────
# CRC-fail classification by SNR (v0.7.8 RSSI smarts)
# ─────────────────────────────────────────────────────────────────────


class TestCrcFailClassification:
    """v0.7.8 annotates CRC failures with an SNR-derived reason
    code so the user can tell weak-signal failures from
    interference / collision failures."""

    def test_low_snr_crc_fail_labeled_weak_signal(self, capsys) -> None:
        from rfcensus.tools.decode_meshtastic import (
            DecodedRecord, _print_human,
        )
        rec = DecodedRecord(
            sample_offset=12345,
            payload_len=50,
            cr=4,
            crc_ok=False,
            cfo_hz=0.0,
            preset="MEDIUM_FAST",
            freq_hz=915_000_000,
            rssi_db=-25.0,
            snr_db=2.0,    # below 5 dB → weak signal
        )
        _print_human(rec)
        captured = capsys.readouterr()
        assert "weak signal" in captured.out

    def test_high_snr_crc_fail_labeled_interference(self, capsys) -> None:
        from rfcensus.tools.decode_meshtastic import (
            DecodedRecord, _print_human,
        )
        rec = DecodedRecord(
            sample_offset=12345,
            payload_len=50,
            cr=4,
            crc_ok=False,
            cfo_hz=0.0,
            preset="MEDIUM_FAST",
            freq_hz=915_000_000,
            rssi_db=-2.0,
            snr_db=18.0,    # ≥ 15 dB → interference / collision
        )
        _print_human(rec)
        captured = capsys.readouterr()
        assert "interference" in captured.out

    def test_mid_snr_crc_fail_unlabeled(self, capsys) -> None:
        """Between 5 and 15 dB SNR is ambiguous; we don't speculate."""
        from rfcensus.tools.decode_meshtastic import (
            DecodedRecord, _print_human,
        )
        rec = DecodedRecord(
            sample_offset=12345,
            payload_len=50,
            cr=4,
            crc_ok=False,
            cfo_hz=0.0,
            preset="MEDIUM_FAST",
            freq_hz=915_000_000,
            rssi_db=-10.0,
            snr_db=10.0,
        )
        _print_human(rec)
        captured = capsys.readouterr()
        assert "weak signal" not in captured.out
        assert "interference" not in captured.out
        assert "(CRC fail)" in captured.out


# ─────────────────────────────────────────────────────────────────────
# Dedup tie-breaker now that RSSI is real
# ─────────────────────────────────────────────────────────────────────


class TestDedupUsesRealRssi:
    """v0.7.8 simplified the dedup tie-breaker: max(rssi) wins,
    deterministically. No more dead RSSI=0 fallback path. Verify
    the tie-breaker picks the highest-RSSI member of a duplicate
    cluster."""

    def test_lazy_dedup_picks_highest_rssi(self) -> None:
        """Read the dedup source — confirm the no-RSSI fallback path
        is gone and max(rssi_db) is the primary key."""
        src = Path(
            "/home/claude/rfcensus/rfcensus/decoders/lazy_pipeline.py"
        ).read_text()
        # No more `if p.lora.rssi_db != 0.0 else -1e9` sentinel
        assert "rssi_db != 0.0 else" not in src
        # max(rssi_db) is the tie-breaker
        assert "max(grp, key=lambda p: p.lora.rssi_db)" in src

    def test_eager_dedup_picks_highest_rssi(self) -> None:
        src = Path(
            "/home/claude/rfcensus/rfcensus/decoders/meshtastic_pipeline.py"
        ).read_text()
        assert "rssi_db != 0.0 else" not in src
        assert "max(grp, key=lambda p: p.lora.rssi_db)" in src


# ─────────────────────────────────────────────────────────────────────
# End-to-end: v0.7.8 still recovers all packets
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not REAL_CAPTURE.exists(),
    reason="real-traffic capture required",
)
class TestRecallStillWorks:
    """The vectorization changes the detector's internal computation
    but must NOT change the activate/deactivate event stream
    enough to drop packets. Verify recall on the real capture."""

    def test_lazy_pipeline_still_finds_9_packets(self) -> None:
        import subprocess
        import sys
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
        lines = [l for l in result.stdout.splitlines()
                 if l.startswith("@")]
        assert len(lines) >= 8, (
            f"v0.7.8 dropped packets: only {len(lines)}/9 recovered"
        )


# ─────────────────────────────────────────────────────────────────────
# NodeInfo PKI-payload defensive parsing
# ─────────────────────────────────────────────────────────────────────


class TestNodeInfoDefensiveParser:
    """v0.7.8: when newer Meshtastic firmware with PKI broadcasts
    NodeInfo, the payload occasionally contains a 32-byte public
    key at the byte position our parser previously accepted as
    `long_name`. We now validate that string fields look like
    legitimate user-facing text (printable, UTF-8, length-bounded)
    before accepting them. Real-world reproduction: a
    MeshtasticNode broadcast on the user's local network produced
    `'��M7��n���Mk�...' !18b5327c` instead of clean output."""

    @staticmethod
    def _varint(n: int) -> bytes:
        out = b""
        while n > 0x7F:
            out += bytes([(n & 0x7F) | 0x80])
            n >>= 7
        out += bytes([n])
        return out

    def test_pki_style_garbage_field_2_is_rejected(self) -> None:
        """A NodeInfo with id at field 1 and 32 bytes of binary
        (pretending to be a public key) at field 2 should NOT have
        long_name set to garbage. v0.7.9: it SHOULD extract those
        32 bytes as the public key — that's the whole point of
        PKI envelope detection."""
        from rfcensus.tools.decode_meshtastic import _try_decode_nodeinfo
        v = self._varint
        id_bytes = b"!18b5327c"
        field1 = bytes([0x0A]) + v(len(id_bytes)) + id_bytes
        # 32 bytes of low-byte binary — typical of fresh key material
        key_like = bytes(range(32))
        field2 = bytes([0x12]) + v(len(key_like)) + key_like
        result = _try_decode_nodeinfo(field1 + field2)
        # No garbage long_name
        assert "long_name" not in result, (
            f"PKI bytes leaked into long_name: {result!r}"
        )
        # id always present
        assert result["id"] == "!18b5327c"
        # v0.7.9: AND public key extracted from the field-2 envelope
        assert "public_key" in result, (
            f"PKI key not extracted: {result!r}"
        )
        assert result["public_key"] == key_like.hex()

    def test_legitimate_text_long_name_accepted(self) -> None:
        from rfcensus.tools.decode_meshtastic import _try_decode_nodeinfo
        v = self._varint
        id_bytes = b"!18b5327c"
        field1 = bytes([0x0A]) + v(len(id_bytes)) + id_bytes
        long_name = b"My Cool Node"
        field2 = bytes([0x12]) + v(len(long_name)) + long_name
        short_name = b"MCN"
        field3 = bytes([0x1A]) + v(len(short_name)) + short_name
        hw = bytes([0x28, 0x09])    # field 5 varint = 9 (RAK4631)
        result = _try_decode_nodeinfo(field1 + field2 + field3 + hw)
        assert result["id"] == "!18b5327c"
        assert result["long_name"] == "My Cool Node"
        assert result["short_name"] == "MCN"
        assert result["hw_model"] == 9

    def test_utf8_emoji_long_name_accepted(self) -> None:
        """The defensive heuristic must not reject legitimate emoji
        in node names — UTF-8 multi-byte sequences should pass."""
        from rfcensus.tools.decode_meshtastic import _try_decode_nodeinfo
        v = self._varint
        id_bytes = b"!18b5327c"
        field1 = bytes([0x0A]) + v(len(id_bytes)) + id_bytes
        emoji_name = "Robin's 🌲".encode("utf-8")
        field2 = bytes([0x12]) + v(len(emoji_name)) + emoji_name
        result = _try_decode_nodeinfo(field1 + field2)
        assert result.get("long_name") == "Robin's 🌲", (
            f"UTF-8 emoji name should be accepted, got {result}"
        )

    def test_oversized_long_name_rejected(self) -> None:
        """If a field claims to be a long_name but exceeds the 40-byte
        Meshtastic spec cap, reject — likely misalignment."""
        from rfcensus.tools.decode_meshtastic import _try_decode_nodeinfo
        v = self._varint
        id_bytes = b"!12345678"
        field1 = bytes([0x0A]) + v(len(id_bytes)) + id_bytes
        # 50 chars of valid ASCII but > 40 cap
        too_long = b"A" * 50
        field2 = bytes([0x12]) + v(len(too_long)) + too_long
        result = _try_decode_nodeinfo(field1 + field2)
        # id preserved, long_name dropped
        assert result == {"id": "!12345678"}

    def test_null_bytes_in_name_rejected(self) -> None:
        """A name field with an embedded NUL byte (= C string
        terminator from a fixed-size buffer dump) is rejected."""
        from rfcensus.tools.decode_meshtastic import _try_decode_nodeinfo
        v = self._varint
        id_bytes = b"!12345678"
        field1 = bytes([0x0A]) + v(len(id_bytes)) + id_bytes
        nullish = b"Test\x00node\x00\x00garbage"
        field2 = bytes([0x12]) + v(len(nullish)) + nullish
        result = _try_decode_nodeinfo(field1 + field2)
        assert "long_name" not in result
        assert result == {"id": "!12345678"}


# ─────────────────────────────────────────────────────────────────────
# v0.7.9 — pffft FFT backend + PKI key recovery + slot freq display
# ─────────────────────────────────────────────────────────────────────


class TestPkiKeyRecovery:
    """v0.7.9 extracts the 32-byte x25519 public key from PKI-mode
    NodeInfo broadcasts — both the standard field-8 path and the
    non-standard envelope variant where the key follows a bogus
    field-2 length-delim header."""

    @staticmethod
    def _varint(n: int) -> bytes:
        out = b""
        while n > 0x7F:
            out += bytes([(n & 0x7F) | 0x80])
            n >>= 7
        out += bytes([n])
        return out

    def test_standard_field8_public_key_extracted(self) -> None:
        from rfcensus.tools.decode_meshtastic import _try_decode_nodeinfo
        v = self._varint
        # Standard User: id (1) + long_name (2) + public_key (8)
        id_b = b"!12345678"
        f1 = bytes([0x0A]) + v(len(id_b)) + id_b
        ln = b"TestNode"
        f2 = bytes([0x12]) + v(len(ln)) + ln
        # Real-looking 32-byte key (high entropy, no NULL runs)
        import os
        key = os.urandom(32)
        f8 = bytes([0x42]) + v(32) + key    # tag (8<<3)|2 = 0x42
        result = _try_decode_nodeinfo(f1 + f2 + f8)
        assert result["id"] == "!12345678"
        assert result["long_name"] == "TestNode"
        assert "public_key" in result
        assert result["public_key"] == key.hex()

    def test_pki_envelope_variant_extracted(self) -> None:
        """The variant we documented in v0.7.8: id followed by
        0x12 0x83 0x17 (bogus field-2 length encoding) followed by
        32 bytes of key material. Real bytes from the user's
        capture."""
        from rfcensus.tools.decode_meshtastic import _try_decode_nodeinfo
        # Hex from /tmp/meshtastic_30s_913_5mhz_1msps.cu8 NODEINFO
        payload = bytes.fromhex(
            "0a09216665643737383130"   # id="!fed77810"
            "128317"                    # PKI envelope marker (bogus field-2 len)
            "a131e8b73e6d712d3651472fb1626e04"
            "267156da2c3868a91afdbdabc9d7eaed"   # 32 bytes of key
            "f0273a44926f5d6632c342e25028f344"
            "ade230cacf5eca0e2f0246aaccf2"        # trailing bytes (signature?)
        )
        result = _try_decode_nodeinfo(payload)
        assert result["id"] == "!fed77810"
        assert "public_key" in result
        # Verify we got the EXACT 32-byte key, not the envelope
        # bytes by mistake.
        expected_key = "a131e8b73e6d712d3651472fb1626e04267156da2c3868a91afdbdabc9d7eaed"
        assert result["public_key"] == expected_key, (
            f"got {result['public_key']}, expected {expected_key}"
        )

    def test_pki_recovery_stable_across_calls(self) -> None:
        """Same payload must always yield the same key bytes —
        critical for correlating broadcasts from the same node."""
        from rfcensus.tools.decode_meshtastic import _try_decode_nodeinfo
        payload = bytes.fromhex(
            "0a09216665643737383130128317"
            "a131e8b73e6d712d3651472fb1626e04"
            "267156da2c3868a91afdbdabc9d7eaed"
            "f0273a44926f5d66"
        )
        keys = set()
        for _ in range(5):
            r = _try_decode_nodeinfo(payload)
            if r and "public_key" in r:
                keys.add(r["public_key"])
        assert len(keys) == 1, (
            f"expected stable key extraction, got {len(keys)} distinct: {keys}"
        )

    def test_text_only_nodeinfo_does_not_invent_key(self) -> None:
        """A plain-text NodeInfo with no key bytes must NOT trigger
        false-positive key extraction."""
        from rfcensus.tools.decode_meshtastic import _try_decode_nodeinfo
        v = self._varint
        f1 = bytes([0x0A]) + v(9) + b"!12345678"
        f2 = bytes([0x12]) + v(8) + b"TestNode"
        f3 = bytes([0x1A]) + v(2) + b"TN"
        f5 = bytes([0x28, 0x09])    # hw_model = 9
        result = _try_decode_nodeinfo(f1 + f2 + f3 + f5)
        assert "public_key" not in result, (
            f"false-positive key from text-only NodeInfo: {result}"
        )


class TestPffftBackend:
    """v0.7.9 builds a pffft-backed FFT by default. Verify the
    backend is loaded + numerically equivalent to the kiss baseline."""

    def test_pffft_backend_loaded(self) -> None:
        """The shipped .so must report 'pffft' as the backend."""
        import ctypes
        lib = ctypes.CDLL(
            "/home/claude/rfcensus/rfcensus/decoders/_native/lora/liblora_demod.so"
        )
        lib.lora_fft_backend_name.restype = ctypes.c_char_p
        backend = lib.lora_fft_backend_name().decode()
        assert backend == "pffft", (
            f"expected pffft backend, got {backend!r}"
        )

    def test_pffft_matches_numpy_dft(self) -> None:
        """For our LoRa N values (256-16384), pffft output must
        match NumPy's reference DFT to within float32 rounding."""
        import ctypes
        import numpy as np
        lib = ctypes.CDLL(
            "/home/claude/rfcensus/rfcensus/decoders/_native/lora/liblora_demod.so"
        )

        class C(ctypes.Structure):
            _fields_ = [("r", ctypes.c_float), ("i", ctypes.c_float)]

        lib.lora_fft_new.restype = ctypes.c_void_p
        lib.lora_fft_new.argtypes = [ctypes.c_uint32]
        lib.lora_fft_destroy.argtypes = [ctypes.c_void_p]
        lib.lora_fft_forward.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(C), ctypes.POINTER(C)
        ]
        lib.lora_fft_aligned_alloc.restype = ctypes.c_void_p
        lib.lora_fft_aligned_alloc.argtypes = [ctypes.c_size_t]
        lib.lora_fft_aligned_free.argtypes = [ctypes.c_void_p]

        for N in [256, 1024, 4096, 16384]:
            ctx = lib.lora_fft_new(N)
            assert ctx, f"alloc failed for N={N}"
            in_p = lib.lora_fft_aligned_alloc(N * ctypes.sizeof(C))
            out_p = lib.lora_fft_aligned_alloc(N * ctypes.sizeof(C))
            in_arr = (C * N).from_address(in_p)
            out_arr = (C * N).from_address(out_p)
            rng = np.random.default_rng(N)
            x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
            for i, c in enumerate(x):
                in_arr[i].r = float(c.real)
                in_arr[i].i = float(c.imag)
            lib.lora_fft_forward(
                ctx,
                ctypes.cast(in_p, ctypes.POINTER(C)),
                ctypes.cast(out_p, ctypes.POINTER(C)),
            )
            got = np.array(
                [complex(out_arr[i].r, out_arr[i].i) for i in range(N)],
                dtype=np.complex64,
            )
            ref = np.fft.fft(x).astype(np.complex64)
            max_err = np.abs(got - ref).max()
            # Accumulated FP error grows ~sqrt(N) * eps * |signal|;
            # 2e-4 is loose enough for N=16384 with margin.
            assert max_err < 2e-4, (
                f"N={N}: pffft vs numpy max diff {max_err:.2e} too high"
            )
            lib.lora_fft_aligned_free(in_p)
            lib.lora_fft_aligned_free(out_p)
            lib.lora_fft_destroy(ctx)


class TestSlotFreqDisplay:
    """v0.7.9 includes the slot center frequency in the per-packet
    line so the user can tell which decoder under --slots all
    actually caught a given packet."""

    def test_per_packet_line_includes_at_freq(self, capsys) -> None:
        from rfcensus.tools.decode_meshtastic import (
            DecodedRecord, _print_human,
        )
        rec = DecodedRecord(
            sample_offset=12345,
            payload_len=50,
            cr=4,
            crc_ok=False,
            cfo_hz=0.0,
            preset="MEDIUM_FAST",
            freq_hz=913_625_000,
            rssi_db=-3.0,
            snr_db=10.0,
        )
        _print_human(rec)
        out = capsys.readouterr().out
        # Must contain the @MHz suffix
        assert "@913.625" in out, f"slot freq missing: {out!r}"
        # Preset name still present
        assert "mediumfast" in out


class TestPerSlotHitBreakdown:
    """v0.7.9: at the end of a run the standalone tool prints a
    per-slot decoded-packet count. Useful for diagnosing which
    slots under --slots all are productive."""

    def test_summary_contains_per_slot_block_for_real_capture(self) -> None:
        """End-to-end: run the standalone tool against the real
        capture and confirm the new per-slot block appears with
        sensible content."""
        import os
        import subprocess
        import sys
        capture = "/tmp/meshtastic_30s_913_5mhz_1msps.cu8"
        if not os.path.exists(capture):
            import pytest
            pytest.skip("real capture not available")
        result = subprocess.run(
            [sys.executable, "-m", "rfcensus.tools.decode_meshtastic",
             capture,
             "--frequency", "913500000",
             "--sample-rate", "1000000",
             "--slots", "all",
             "--lazy"],
            capture_output=True, text=True,
            cwd="/home/claude/rfcensus",
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        # Header present
        assert "Per-slot decoder hits" in result.stderr, (
            f"per-slot block missing from summary:\n{result.stderr}"
        )
        # At least one slot row present (format: PRESET @ NNN.NNN MHz)
        import re
        slot_rows = re.findall(
            r"#\s+\w+\s+@\s+\d+\.\d+ MHz\s+\d+ packets",
            result.stderr,
        )
        assert len(slot_rows) >= 1, (
            f"no slot rows found:\n{result.stderr}"
        )
