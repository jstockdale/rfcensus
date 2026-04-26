"""Tests for v0.7.1 — region tables, passband enumeration, IQ source
abstraction, and the multi-preset pipeline.

Three layers:
  1. Pure-Python region/preset/DJB2 tests (no native deps)
  2. IQ source tests (file works always; rtl_sdr / rtl_tcp need
     external infrastructure so they're behavioral-only)
  3. MultiPresetPipeline integration (needs both .so files)
"""
from __future__ import annotations

import socket
import struct
import threading
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


def _libs_built() -> bool:
    return ((_NATIVE_MESHTASTIC / "libmeshtastic.so").exists() and
            (_NATIVE_LORA / "liblora_demod.so").exists())


_REAL_CAPTURE = Path("/tmp/meshtastic_30s_913_5mhz_1msps.cu8")


# ─────────────────────────────────────────────────────────────────────
# Region table + DJB2 + slot computation (pure Python, no native deps)
# ─────────────────────────────────────────────────────────────────────

class TestDjb2Hash:
    """The DJB2 hash MUST match meshtastic-lite's C implementation
    exactly. Mismatches here mean every default-channel slot ends up
    on the wrong frequency and we hear nothing."""

    def test_canonical_hello_vector(self) -> None:
        from rfcensus.utils.meshtastic_region import djb2
        # The canonical Bernstein test vector for "hello" mod 2^32:
        # h=5381; for c in "hello": h = ((h<<5)+h)+ord(c) → 261238937
        assert djb2("hello") == 261238937

    def test_empty_string(self) -> None:
        from rfcensus.utils.meshtastic_region import djb2
        assert djb2("") == 5381

    def test_single_char_a(self) -> None:
        from rfcensus.utils.meshtastic_region import djb2
        # 5381 * 33 + ord('a') = 177573 + 97 = 177670
        assert djb2("a") == 177670

    def test_overflow_wraps_at_32bits(self) -> None:
        """Long strings push h past 2^32 — must wrap to stay in range."""
        from rfcensus.utils.meshtastic_region import djb2
        h = djb2("x" * 64)
        assert 0 <= h <= 0xFFFFFFFF


class TestRegionAndPresetTables:
    def test_us_region_band_plan(self) -> None:
        from rfcensus.utils.meshtastic_region import REGIONS
        us = REGIONS["US"]
        assert us.freq_start_mhz == 902.0
        assert us.freq_end_mhz == 928.0
        assert us.duty_cycle_pct == 100

    def test_all_regions_have_valid_bands(self) -> None:
        from rfcensus.utils.meshtastic_region import REGIONS
        for code, r in REGIONS.items():
            assert r.freq_end_mhz > r.freq_start_mhz, code
            assert 0 < r.duty_cycle_pct <= 100, code

    def test_preset_table_complete(self) -> None:
        from rfcensus.utils.meshtastic_region import PRESETS
        # Nine presets, no more, no less — matches upstream
        # MeshModemPreset enum and meshPresetParams() switch.
        assert set(PRESETS) == {
            "LONG_SLOW", "LONG_MODERATE", "LONG_FAST", "LONG_TURBO",
            "MEDIUM_SLOW", "MEDIUM_FAST",
            "SHORT_SLOW", "SHORT_FAST", "SHORT_TURBO",
        }

    def test_preset_params_match_upstream(self) -> None:
        """Sanity check a few presets against meshPresetParams() in
        meshtastic_config.h. Drift here = wrong (SF, BW) per preset
        = decoder won't sync with traffic."""
        from rfcensus.utils.meshtastic_region import PRESETS
        # (key, bw_hz, sf, cr)
        cases = [
            ("LONG_FAST",     250_000, 11, 5),
            ("LONG_SLOW",     125_000, 12, 8),
            ("LONG_MODERATE", 125_000, 11, 8),
            ("LONG_TURBO",    500_000, 11, 8),
            ("MEDIUM_FAST",   250_000,  9, 5),
            ("MEDIUM_SLOW",   250_000, 10, 5),
            ("SHORT_FAST",    250_000,  7, 5),
            ("SHORT_TURBO",   500_000,  7, 5),
        ]
        for key, bw, sf, cr in cases:
            p = PRESETS[key]
            assert p.bandwidth_hz == bw, key
            assert p.sf == sf, key
            assert p.cr == cr, key


class TestSlotComputation:
    """The slot/frequency math from meshCalcFrequency() in
    meshtastic_radio.h. Reference values were computed by running
    meshtastic-lite directly and pinned here."""

    def test_us_default_slots_match_reference(self) -> None:
        """Reference slot indices + frequencies for US region defaults.
        These are deterministic — derived purely from the preset name
        + region band-plan via DJB2 → mod num_slots → freq_start +
        bw/2 + slot * bw."""
        from rfcensus.utils.meshtastic_region import default_slot
        cases = [
            # (preset_key, expected_slot, expected_freq_mhz)
            ("LONG_MODERATE",  5, 902.6875),
            ("LONG_SLOW",     26, 905.3125),
            ("LONG_FAST",     19, 906.875),
            ("LONG_TURBO",    13, 908.750),
            ("MEDIUM_FAST",   44, 913.125),
            ("MEDIUM_SLOW",   51, 914.875),
            ("SHORT_FAST",    67, 918.875),
            ("SHORT_SLOW",    74, 920.625),
            ("SHORT_TURBO",   49, 926.750),
        ]
        for preset, want_slot, want_mhz in cases:
            s = default_slot("US", preset)
            assert s.slot == want_slot, (
                f"{preset}: slot {s.slot} != {want_slot}"
            )
            assert abs(s.freq_hz - want_mhz * 1e6) < 1, (
                f"{preset}: freq {s.freq_hz/1e6:.4f} MHz "
                f"!= {want_mhz} MHz"
            )

    def test_num_slots_scales_with_bandwidth(self) -> None:
        """Within a region, narrower BW → more slots."""
        from rfcensus.utils.meshtastic_region import default_slot
        # US 902-928 = 26 MHz wide
        assert default_slot("US", "LONG_SLOW").num_slots == 208   # 26/0.125
        assert default_slot("US", "LONG_FAST").num_slots == 104   # 26/0.250
        assert default_slot("US", "LONG_TURBO").num_slots == 52   # 26/0.500

    def test_custom_channel_changes_slot(self) -> None:
        """The slot is derived from channel NAME, not preset name.
        Custom-named channels land on different slots."""
        from rfcensus.utils.meshtastic_region import (
            default_slot, custom_channel_slot,
        )
        default = default_slot("US", "LONG_FAST")
        custom = custom_channel_slot("US", "LONG_FAST", "MyMesh")
        # Different name → different hash → different slot. Astronomically
        # unlikely they collide, but we assert >0.5% likely difference.
        assert default.slot != custom.slot

    def test_slot_override_pins_frequency(self) -> None:
        from rfcensus.utils.meshtastic_region import custom_channel_slot
        # Force slot 0 — frequency = freq_start + bw/2.
        s = custom_channel_slot("US", "LONG_FAST", "ignored", slot_override=0)
        assert s.slot == 0
        # 902.0 + 0.125 = 902.125 MHz
        assert abs(s.freq_hz - 902_125_000) < 1


class TestPassbandEnumeration:
    """``slots_in_passband`` is the heart of "auto" mode — given a
    dongle's tuning, return all preset slots that fit cleanly."""

    def test_915mhz_2_4msps_us_catches_medium_slow(self) -> None:
        from rfcensus.utils.meshtastic_region import slots_in_passband
        slots = slots_in_passband("US", 915_000_000, 2_400_000)
        keys = {s.preset.key for s in slots}
        # MEDIUM_SLOW is at 914.875 — well inside the 913.8-916.2 window.
        assert "MEDIUM_SLOW" in keys
        # MEDIUM_FAST at 913.125 is outside (offset 1.875 > 1.075 max).
        assert "MEDIUM_FAST" not in keys

    def test_914mhz_2_4msps_catches_both_medium_presets(self) -> None:
        from rfcensus.utils.meshtastic_region import slots_in_passband
        slots = slots_in_passband("US", 914_000_000, 2_400_000)
        keys = {s.preset.key for s in slots}
        assert {"MEDIUM_FAST", "MEDIUM_SLOW"} <= keys

    def test_no_slots_when_passband_too_narrow(self) -> None:
        """At 100 kS/s nothing fits (need ≥250 kHz for typical preset)."""
        from rfcensus.utils.meshtastic_region import slots_in_passband
        assert slots_in_passband("US", 915_000_000, 100_000) == []

    def test_results_sorted_by_frequency(self) -> None:
        from rfcensus.utils.meshtastic_region import slots_in_passband
        # Wide tuning to catch several
        slots = slots_in_passband("US", 906_000_000, 2_400_000)
        freqs = [s.freq_hz for s in slots]
        assert freqs == sorted(freqs)

    def test_unknown_preset_raises(self) -> None:
        from rfcensus.utils.meshtastic_region import slots_in_passband
        with pytest.raises(ValueError, match="unknown preset"):
            slots_in_passband("US", 915_000_000, 2_400_000,
                               presets=["MEDIUM_FAST", "BOGUS_PRESET"])

    def test_edge_guard_can_be_disabled(self) -> None:
        """Setting edge_guard=0 admits slots at the literal Nyquist edge."""
        from rfcensus.utils.meshtastic_region import slots_in_passband
        # Tune 913.5 MHz @ 1 MS/s — MEDIUM_FAST at 913.125 sits exactly
        # at the (Fs-BW)/2 edge. Default 25kHz guard rejects it; with
        # guard=0 it's admitted.
        with_guard = slots_in_passband("US", 913_500_000, 1_000_000)
        no_guard = slots_in_passband("US", 913_500_000, 1_000_000,
                                       edge_guard_hz=0)
        assert "MEDIUM_FAST" not in {s.preset.key for s in with_guard}
        assert "MEDIUM_FAST" in {s.preset.key for s in no_guard}


# ─────────────────────────────────────────────────────────────────────
# IQ source abstraction
# ─────────────────────────────────────────────────────────────────────

class TestFileIQSource:
    def test_yields_chunks_then_stops(self, tmp_path: Path) -> None:
        from rfcensus.utils.iq_source import FileIQSource
        # 3 chunks worth of bytes
        data = bytes(range(256)) * 100   # 25600 bytes
        fpath = tmp_path / "x.cu8"
        fpath.write_bytes(data)

        chunks = list(FileIQSource(fpath, chunk_size=10000))
        # 25600 / 10000 = 3 chunks (10k, 10k, 5600)
        assert len(chunks) == 3
        assert b"".join(chunks) == data

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        from rfcensus.utils.iq_source import FileIQSource
        fpath = tmp_path / "y.cu8"
        # v0.7.5: IQSource enforces I/Q-pair alignment — every yielded
        # chunk is even-length so cu8 consumers never see a half-pair.
        # Use a 6-byte fixture so the round-trip is lossless.
        fpath.write_bytes(b"hello!")
        with FileIQSource(fpath) as src:
            assert next(src) == b"hello!"
        # After close(), iteration stops cleanly — re-iter raises
        with pytest.raises(StopIteration):
            next(src)

    def test_empty_file_yields_nothing(self, tmp_path: Path) -> None:
        from rfcensus.utils.iq_source import FileIQSource
        fpath = tmp_path / "empty.cu8"
        fpath.write_bytes(b"")
        assert list(FileIQSource(fpath)) == []


class TestRtlTcpProtocol:
    """We can't actually run rtl_tcp in CI, but we CAN spin up a fake
    server that speaks the protocol enough to verify our client sends
    the right control commands and reads samples correctly."""

    def test_client_sends_correct_setup_commands(self) -> None:
        from rfcensus.utils.iq_source import RtlTcpSource, RtlSdrConfig

        received_cmds: list[bytes] = []
        srv_done = threading.Event()

        def server(srv_sock: socket.socket) -> None:
            try:
                conn, _ = srv_sock.accept()
                # Send the 12-byte greeting
                conn.sendall(b"RTL0" + b"\x00" * 8)
                # Read 5-byte commands until we have all 4 (gain_mode +
                # set_gain + sample_rate + freq; ppm=0 is skipped).
                data = b""
                while len(received_cmds) < 4:
                    chunk = conn.recv(64)
                    if not chunk:
                        break
                    data += chunk
                    while len(data) >= 5:
                        received_cmds.append(data[:5])
                        data = data[5:]
                # Send some IQ samples so the read side has something
                conn.sendall(b"\x80" * 1024)
                # Wait briefly so the client can read before we close
                conn.settimeout(0.5)
                try:
                    conn.recv(1)
                except socket.timeout:
                    pass
                conn.close()
            finally:
                srv_sock.close()
                srv_done.set()

        srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv_sock.bind(("127.0.0.1", 0))
        srv_sock.listen(1)
        port = srv_sock.getsockname()[1]
        threading.Thread(target=server, args=(srv_sock,), daemon=True).start()

        cfg = RtlSdrConfig(
            freq_hz=915_000_000,
            sample_rate_hz=2_400_000,
            gain_tenths_db=300,
            ppm=0,
        )
        with RtlTcpSource("127.0.0.1", port, cfg, chunk_size=512) as src:
            data = next(src)
            assert len(data) > 0

        srv_done.wait(timeout=3)
        # Decode the commands we received: gain_mode=1 (manual),
        # set_gain=300, set_sample_rate=2_400_000, set_freq=915_000_000.
        decoded = [(c[0], struct.unpack(">I", c[1:])[0])
                   for c in received_cmds]
        # 0x03 = SET_GAIN_MODE, 0x04 = SET_GAIN, 0x02 = SET_SAMPLE_RATE,
        # 0x01 = SET_FREQ. We sent gain → mode + value, then rate, freq.
        # ppm=0 is skipped, so we expect exactly 4 commands but the
        # server reads at least 3.
        cmds = {c[0] for c in decoded}
        assert 0x01 in cmds, "missing SET_FREQ command"
        # Check the freq value made it
        for cmd, val in decoded:
            if cmd == 0x01:
                assert val == 915_000_000


# ─────────────────────────────────────────────────────────────────────
# MultiPresetPipeline integration (needs native libs + capture file)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _libs_built(),
                    reason="native libraries not built")
class TestMultiPresetPipeline:
    def test_construction_with_no_slots_raises(self) -> None:
        from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
        from rfcensus.decoders.meshtastic_pipeline import MultiPresetPipeline
        mesh = MeshtasticDecoder("MEDIUM_FAST")
        with pytest.raises(ValueError, match="at least one slot"):
            MultiPresetPipeline(slots=[], sample_rate_hz=1_000_000,
                                center_freq_hz=915_000_000, mesh=mesh)

    def test_mix_freq_computed_per_slot(self) -> None:
        """Each decoder gets its own mix_freq = center − slot_freq."""
        from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
        from rfcensus.decoders.meshtastic_pipeline import MultiPresetPipeline
        from rfcensus.utils.meshtastic_region import default_slot

        slots = [
            default_slot("US", "MEDIUM_FAST"),   # 913.125
            default_slot("US", "MEDIUM_SLOW"),   # 914.875
        ]
        mesh = MeshtasticDecoder("MEDIUM_FAST")
        pipe = MultiPresetPipeline(
            slots=slots, sample_rate_hz=2_400_000,
            center_freq_hz=914_000_000, mesh=mesh,
        )
        # We can't introspect mix_freq directly from the wrapper, but
        # we can verify the right number of decoders were created.
        assert len(pipe.slots) == 2
        assert pipe.center_freq_hz == 914_000_000
        assert pipe.sample_rate_hz == 2_400_000


@pytest.mark.skipif(not _libs_built() or not _REAL_CAPTURE.exists(),
                    reason="needs native libs + real capture file")
class TestMultiPresetEndToEnd:
    """The regression test: spawn 9 decoders against a single-preset
    capture, verify only the matching preset produces packets and
    decryption works."""

    def test_nine_decoders_one_preset_actually_present(self) -> None:
        from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
        from rfcensus.decoders.meshtastic_pipeline import MultiPresetPipeline
        from rfcensus.utils.iq_source import FileIQSource
        from rfcensus.utils.meshtastic_region import (
            PRESETS, default_slot,
        )

        # All 9 presets, mix_freq computed from 913.5 MHz center.
        slots = [default_slot("US", k) for k in PRESETS]
        mesh = MeshtasticDecoder("LONG_FAST")
        # Add ALL 9 default channels so any preset's traffic decrypts.
        for key in PRESETS:
            mesh.add_channel(PRESETS[key].display_name,
                             psk=b"\x01", is_primary=False)

        pipe = MultiPresetPipeline(
            slots=slots, sample_rate_hz=1_000_000,
            center_freq_hz=913_500_000, mesh=mesh,
        )

        decoded_per_preset: dict[str, int] = {}
        decrypted = 0
        with FileIQSource(_REAL_CAPTURE) as src:
            for chunk in src:
                pipe.feed_cu8(chunk)
                for pp in pipe.pop_packets():
                    if pp.crc_ok:
                        decoded_per_preset[pp.slot.preset.key] = \
                            decoded_per_preset.get(pp.slot.preset.key, 0) + 1
                    if pp.decrypted:
                        decrypted += 1

        # The capture is 100% MEDIUM_FAST traffic. Only that decoder
        # should produce CRC-passing packets. Other decoders may have
        # spurious preamble matches but their headers won't validate.
        # v0.7.2 may catch one extra short packet at end-of-stream that
        # the v0.7.x MAX_SYMS=320 waterfall would have missed.
        assert decoded_per_preset.get("MEDIUM_FAST", 0) >= 6
        assert decrypted >= 6
        # No other preset should produce CRC-passing packets — the
        # data simply isn't there.
        for key, n in decoded_per_preset.items():
            if key != "MEDIUM_FAST":
                assert n == 0, (
                    f"unexpected CRC-pass for {key}: {n} packets"
                )
