"""Tests for v0.7.0 — Meshtastic-from-SDR pipeline.

Three layers:
  1. PCAP/LoRaTap writer (pure-Python, no native deps required)
  2. MeshtasticDecoder ctypes wrapper (requires libmeshtastic.so)
  3. End-to-end pipeline (requires both .so files + a capture file)

Layer 2 and 3 tests skip if the native libraries aren't built — they
auto-build via the Makefile if invoked through `pytest -v`."""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest


_NATIVE_MESHTASTIC = (
    Path(__file__).parent.parent.parent
    / "rfcensus" / "decoders" / "_native" / "meshtastic"
)
_NATIVE_LORA = (
    Path(__file__).parent.parent.parent
    / "rfcensus" / "decoders" / "_native" / "lora"
)


def _libmeshtastic_built() -> bool:
    return (_NATIVE_MESHTASTIC / "libmeshtastic.so").exists()


def _liblora_built() -> bool:
    return (_NATIVE_LORA / "liblora_demod.so").exists()


# ─────────────────────────────────────────────────────────────────────
# PCAP / LoRaTap writer (pure Python)
# ─────────────────────────────────────────────────────────────────────

class TestPcapLoraTapWriter:
    """Round-trip + spec-compliance tests for the PCAP-LoRaTap writer."""

    def test_global_header_is_24_bytes_le_loratap(self, tmp_path: Path) -> None:
        from rfcensus.utils.pcap_loratap import PcapLoraTapWriter, DLT_LORATAP

        out = tmp_path / "x.pcap"
        with PcapLoraTapWriter(out):
            pass

        data = out.read_bytes()
        assert len(data) == 24, "global header is 24 bytes"

        magic, vmaj, vmin, tz, sigfigs, snaplen, dlt = \
            struct.unpack("<IHHiIII", data)
        assert magic == 0xA1B2C3D4, "LE microsecond magic"
        assert (vmaj, vmin) == (2, 4), "pcap v2.4"
        assert dlt == DLT_LORATAP == 270

    def test_packet_record_layout_matches_spec(self, tmp_path: Path) -> None:
        from rfcensus.utils.pcap_loratap import PcapLoraTapWriter

        payload = bytes(range(20))
        out = tmp_path / "y.pcap"
        with PcapLoraTapWriter(out) as w:
            w.write_packet(
                payload, frequency_hz=915_000_000,
                bandwidth_hz=250_000, sf=9, sync_word=0x2B,
                rssi_dbm=-78, snr_db=4.5,
                timestamp=1700000000.123456,
            )

        data = out.read_bytes()
        # Skip global header (24 bytes), per-packet header (16 bytes)
        rec_hdr = data[24:40]
        ts_s, ts_us, incl, orig = struct.unpack("<IIII", rec_hdr)
        assert ts_s == 1700000000
        # Microsecond rounding: 0.123456 * 1e6 = 123456 (exact-ish)
        assert ts_us == 123456
        # 15-byte LoRaTap header + 20-byte payload = 35 bytes
        assert incl == orig == 35

        loratap = data[40:55]
        assert loratap[0] == 0   # version 0
        assert loratap[1] == 0   # padding
        assert struct.unpack(">H", loratap[2:4])[0] == 15  # length
        assert struct.unpack(">I", loratap[4:8])[0] == 915_000_000
        # bandwidth: 250 kHz / 125 = 2
        assert loratap[8] == 2
        assert loratap[9] == 9   # SF
        # RSSI: -78 dBm + 139 = 61
        assert loratap[10] == 61
        # max_rssi, current_rssi unused → 255
        assert loratap[11] == 255 and loratap[12] == 255
        # SNR: 4.5 * 4 = 18
        assert loratap[13] == 18
        assert loratap[14] == 0x2B   # sync word

        # Payload bytes follow
        assert data[55:75] == payload

    def test_bandwidth_encoding_table(self) -> None:
        """LoRaTap bandwidth is 'in 125 kHz steps': 1=125, 2=250, 4=500."""
        from rfcensus.utils.pcap_loratap import _bw_to_loratap
        assert _bw_to_loratap(125_000) == 1
        assert _bw_to_loratap(250_000) == 2
        assert _bw_to_loratap(500_000) == 4

    def test_rssi_zero_is_na_sentinel(self) -> None:
        """LoRaTap reserves 255 for 'measurement unavailable'."""
        from rfcensus.utils.pcap_loratap import _rssi_to_loratap
        assert _rssi_to_loratap(0.0) == 255

    def test_snr_two_complement_encoding(self) -> None:
        from rfcensus.utils.pcap_loratap import _snr_to_loratap
        assert _snr_to_loratap(0.0) == 0
        # +5 dB → +20 raw (5*4) → 0x14
        assert _snr_to_loratap(5.0) == 20
        # -5 dB → -20 raw → two's complement = 256-20 = 236
        assert _snr_to_loratap(-5.0) == 236

    def test_writes_packets_counter(self, tmp_path: Path) -> None:
        from rfcensus.utils.pcap_loratap import PcapLoraTapWriter
        out = tmp_path / "z.pcap"
        with PcapLoraTapWriter(out) as w:
            for _ in range(7):
                w.write_packet(b"x" * 16,
                    frequency_hz=915_000_000,
                    bandwidth_hz=250_000, sf=9)
            assert w.packets_written == 7

    def test_rejects_invalid_sf(self, tmp_path: Path) -> None:
        from rfcensus.utils.pcap_loratap import PcapLoraTapWriter
        with PcapLoraTapWriter(tmp_path / "a.pcap") as w:
            with pytest.raises(ValueError, match="SF"):
                w.write_packet(b"\0", frequency_hz=915_000_000,
                                bandwidth_hz=250_000, sf=6)
            with pytest.raises(ValueError, match="SF"):
                w.write_packet(b"\0", frequency_hz=915_000_000,
                                bandwidth_hz=250_000, sf=13)

    def test_close_via_context_manager_idempotent(self, tmp_path: Path) -> None:
        """Re-closing or never opening shouldn't crash."""
        from rfcensus.utils.pcap_loratap import PcapLoraTapWriter
        w = PcapLoraTapWriter(tmp_path / "b.pcap")
        w.close()  # never opened
        w.open()
        w.close()
        w.close()  # double close


# ─────────────────────────────────────────────────────────────────────
# Meshtastic decoder ctypes wrapper
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _libmeshtastic_built(),
                    reason="libmeshtastic.so not built — run "
                           "`make` in decoders/_native/meshtastic/")
class TestMeshtasticDecoder:
    def test_library_loads_and_reports_version(self) -> None:
        from rfcensus.decoders.meshtastic_native import library_version
        v = library_version()
        # Must be a non-empty version-ish string
        assert isinstance(v, str) and len(v) > 0
        assert v[0].isdigit(), f"version {v!r} should start with a digit"

    def test_default_channel_hash_for_each_preset(self) -> None:
        """Verify the canonical channel hashes match meshtastic-lite's
        published expectations. These values are deterministic across
        any LoRa implementation that follows the spec — they're an
        XOR-hash of the preset name + the default 16-byte PSK.

        Snapshot taken at v0.7.0; if any of these change the upstream
        Meshtastic project changed either a preset name or the PSK,
        which is a wire-level break and must be handled deliberately."""
        from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
        expected = {
            "LONG_FAST":     0x08,
            "LONG_SLOW":     0x0F,
            "LONG_MODERATE": 0x6E,
            "LONG_TURBO":    0x76,
            "MEDIUM_FAST":   0x1F,   # matches our captured Bay Area packets!
            "MEDIUM_SLOW":   0x18,
            "SHORT_FAST":    0x70,
            "SHORT_SLOW":    0x77,
            "SHORT_TURBO":   0x0E,
        }
        for preset, want in expected.items():
            d = MeshtasticDecoder(preset)
            d.add_default_channel()
            got = d.channel_hash(0)
            assert got == want, (
                f"{preset} default channel hash 0x{got:02X} "
                f"!= expected 0x{want:02X} — has the spec changed?"
            )

    def test_unknown_preset_rejected(self) -> None:
        from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
        with pytest.raises(ValueError, match="unknown preset"):
            MeshtasticDecoder("BOGUS")

    def test_decode_too_short_raises(self) -> None:
        from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
        d = MeshtasticDecoder("MEDIUM_FAST")
        d.add_default_channel()
        with pytest.raises(ValueError, match="too short"):
            d.decode(b"\x00" * 8)  # < 16-byte header

    def test_decode_unknown_channel_returns_parsed_header(self) -> None:
        """Frame with an unknown channel hash should still have its
        header parsed (decrypted=False, plaintext=ciphertext)."""
        from rfcensus.decoders.meshtastic_native import MeshtasticDecoder

        # Hand-built 16-byte header + 8 dummy ciphertext bytes
        # to=BCAST, from=0xCAFE, id=0x1234, flags=0x60 (hop_start=3),
        # channel_hash=0xEE (unknown), next_hop=0, relay=0
        frame = struct.pack("<IIIBBBB",
                             0xFFFFFFFF, 0xCAFE, 0x1234,
                             0x60, 0xEE, 0, 0) + b"abcdefgh"
        d = MeshtasticDecoder("MEDIUM_FAST")
        d.add_default_channel()
        out = d.decode(frame)
        assert out.from_node == 0xCAFE
        assert out.to == 0xFFFFFFFF
        assert out.id == 0x1234
        assert out.channel_hash == 0xEE
        assert out.decrypted is False
        assert out.channel_index == -1


# ─────────────────────────────────────────────────────────────────────
# End-to-end pipeline on real Meshtastic capture
# ─────────────────────────────────────────────────────────────────────

_REAL_CAPTURE = Path("/tmp/meshtastic_30s_913_5mhz_1msps.cu8")


@pytest.mark.skipif(not _libmeshtastic_built() or not _liblora_built(),
                    reason="native libraries not built")
@pytest.mark.skipif(not _REAL_CAPTURE.exists(),
                    reason=f"real capture {_REAL_CAPTURE} not present")
class TestEndToEndPipeline:
    """Validate the FULL pipeline (LoRa demod + Meshtastic decrypt) on
    the 30-second real-world Bay Area Meshtastic capture.

    This is the regression test that would have caught any of the
    v0.6.18 PHY bugs. If decrypt rate drops below 6/6, something is
    broken upstream."""

    def test_decrypts_all_six_known_good_packets(self) -> None:
        from rfcensus.decoders.lora_native import LoraConfig, LoraDecoder
        from rfcensus.decoders.meshtastic_native import MeshtasticDecoder

        lora = LoraDecoder(LoraConfig(
            sample_rate_hz=1_000_000,
            bandwidth=250_000, sf=9,
            sync_word=0x2B, mix_freq_hz=375_000,
        ))
        mesh = MeshtasticDecoder("MEDIUM_FAST")
        mesh.add_default_channel()

        decrypted = 0
        senders: set[int] = set()
        with _REAL_CAPTURE.open("rb") as f:
            while chunk := f.read(1 << 16):
                lora.feed_cu8(chunk)
                for pkt in lora.pop_packets():
                    if not pkt.crc_ok:
                        continue
                    mp = mesh.decode(pkt.payload)
                    if mp.decrypted:
                        decrypted += 1
                        senders.add(mp.from_node)

        s = lora.stats()
        # Regression bounds: the v0.7.0 baseline result on this capture
        # was 8 preambles / 7 headers / 6 packets / 6 decrypted. v0.7.2
        # added the C decoder's early-symbol-count fix
        # (lora_compute_symbols_needed) which catches one additional
        # short packet at end-of-stream that the old MAX_SYMS=320
        # waterfall would have abandoned. So allow ≥ baseline rather
        # than ==.
        assert s.preambles_found >= 8
        assert s.headers_decoded >= 7
        assert s.packets_decoded >= 6
        assert decrypted >= 6, (
            f"only {decrypted} packets decrypted (was 6 in v0.7.0)"
        )
        assert len(senders) >= 6, (
            f"expected at least 6 distinct senders, got {len(senders)}"
        )

    def test_cli_runs_clean(self, tmp_path: Path) -> None:
        """Invoke the CLI like a user would, check exit code + outputs.

        v0.7.1 update: CLI no longer takes --bandwidth/--sf/--mix-freq
        (auto-derived from --preset + --frequency + --region). This
        test now exercises the explicit-preset path."""
        pcap_out = tmp_path / "out.pcap"
        jsonl_out = tmp_path / "out.jsonl"
        result = subprocess.run(
            [sys.executable, "-m", "rfcensus.tools.decode_meshtastic",
             str(_REAL_CAPTURE),
             "--frequency", "913500000",
             "--sample-rate", "1000000",
             "--preset", "MEDIUM_FAST",
             "--region", "US",
             "--quiet",
             "--pcap", str(pcap_out),
             "--jsonl", str(jsonl_out)],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        # Summary should report at least 6 decrypted. v0.7.2 may report
        # more (the early-emit fix can catch one extra short packet
        # at end-of-stream that v0.7.x abandoned).
        import re
        m = re.search(r"decrypted:\s+(\d+)", result.stderr)
        assert m is not None, f"no 'decrypted:' line: {result.stderr}"
        assert int(m.group(1)) >= 6, (
            f"expected ≥6 decrypted, got {m.group(1)}: {result.stderr}"
        )

        assert pcap_out.exists() and pcap_out.stat().st_size > 24
        # JSONL: one line per decoded LoRa packet (whether crc_ok or not)
        lines = jsonl_out.read_text().strip().split("\n")
        assert len(lines) >= 7
        for line in lines:
            d = json.loads(line)
            for key in ("sample_offset", "payload_len", "crc_ok", "cfo_hz"):
                assert key in d
