"""Tests for the meshtastic decoder builtin.

Covers:
  • Registry sees and instantiates the meshtastic decoder
  • capabilities are sane (frequency ranges, sample rates, opt-in flag)
  • check_available works without a binary on PATH
  • end-to-end run() against the real 30s capture (with RtlTcpSource
    patched to FileIQSource so we don't need a live dongle/fanout)
  • DecodeEvents have the right shape and are routed via the EventBus
  • PSK config loading from the decoder_options dict

The end-to-end test is the load-bearing one — it exercises the
async pump, the cross-thread queue handoff, the lazy pipeline,
the meshtastic decoder, and the DecodeEvent emission all at once
on real Bay Area mesh traffic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from rfcensus.decoders.base import DecoderRunSpec
from rfcensus.events import DecodeEvent, EventBus
from rfcensus.hardware.broker import AccessMode, DongleLease
from rfcensus.hardware.dongle import Dongle
from rfcensus.utils.iq_source import FileIQSource


_REAL_CAPTURE = Path("/tmp/meshtastic_30s_913_5mhz_1msps.cu8")
_NATIVE_LORA = (
    Path(__file__).parent.parent.parent
    / "rfcensus" / "decoders" / "_native" / "lora"
)
_NATIVE_MESH = (
    Path(__file__).parent.parent.parent
    / "rfcensus" / "decoders" / "_native" / "meshtastic"
)


def _libs_built() -> bool:
    return ((_NATIVE_LORA / "liblora_demod.so").exists()
            and (_NATIVE_MESH / "libmeshtastic.so").exists())


# ─────────────────────────────────────────────────────────────────────
# Registration & capabilities
# ─────────────────────────────────────────────────────────────────────

class TestRegistration:
    def test_registry_includes_meshtastic(self) -> None:
        from rfcensus.decoders.registry import get_registry, reset_registry
        reset_registry()
        r = get_registry()
        assert "meshtastic" in r.names()

    def test_meshtastic_decoder_class_loadable(self) -> None:
        from rfcensus.decoders.registry import get_registry
        cls = get_registry().get("meshtastic")
        assert cls is not None
        assert cls.capabilities.name == "meshtastic"

    def test_capabilities_cover_us_915_and_eu_868(self) -> None:
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )
        cap = MeshtasticBuiltinDecoder.capabilities
        assert cap.covers(915_000_000)
        assert cap.covers(868_500_000)
        # Outside any Meshtastic ISM band → not covered
        assert not cap.covers(145_000_000)
        assert not cap.covers(2_400_000_000)

    def test_decoder_does_not_require_exclusive_dongle(self) -> None:
        """It connects to the rtl_tcp fanout like rtl_433/rtlamr."""
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )
        cap = MeshtasticBuiltinDecoder.capabilities
        assert not cap.requires_exclusive_dongle
        assert cap.access_mode == AccessMode.SHARED

    def test_decoder_declares_no_external_binary(self) -> None:
        """In-process — no binary check needed."""
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )
        assert MeshtasticBuiltinDecoder.capabilities.external_binary == ""

    def test_routes_for_915_band(self) -> None:
        """Frequency-based decoder selection must route mesh into 915."""
        from rfcensus.decoders.registry import get_registry
        decoders = get_registry().decoders_for_frequency(915_000_000)
        names = [d.capabilities.name for d in decoders]
        assert "meshtastic" in names

    def test_does_not_route_for_2m_amateur(self) -> None:
        from rfcensus.decoders.registry import get_registry
        decoders = get_registry().decoders_for_frequency(145_000_000)
        names = [d.capabilities.name for d in decoders]
        assert "meshtastic" not in names


@pytest.mark.skipif(not _libs_built(), reason="native libs not built")
class TestAvailability:
    def test_available_when_libs_present(self) -> None:
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )
        d = MeshtasticBuiltinDecoder()
        av = asyncio.run(d.check_available())
        assert av.available
        assert av.binary_path == "(in-process)"


# ─────────────────────────────────────────────────────────────────────
# PSK config loading
# ─────────────────────────────────────────────────────────────────────

class TestPskLoading:
    def test_psk_b64_decoded(self) -> None:
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )
        # 16 bytes of \x42 = "QkJCQkJCQkJCQkJCQkJCQg==" in base64
        entry = {"name": "X", "psk_b64": "QkJCQkJCQkJCQkJCQkJCQg=="}
        out = MeshtasticBuiltinDecoder._psk_entry_to_bytes(entry)
        assert out == bytes([0x42] * 16)

    def test_psk_hex_decoded(self) -> None:
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )
        entry = {"name": "X", "psk_hex": "deadbeef"}
        out = MeshtasticBuiltinDecoder._psk_entry_to_bytes(entry)
        assert out == bytes([0xDE, 0xAD, 0xBE, 0xEF])

    def test_psk_short_form(self) -> None:
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )
        entry = {"name": "X", "psk_short": 1}
        out = MeshtasticBuiltinDecoder._psk_entry_to_bytes(entry)
        assert out == bytes([1])

    def test_psk_short_out_of_range_raises(self) -> None:
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )
        with pytest.raises(ValueError):
            MeshtasticBuiltinDecoder._psk_entry_to_bytes(
                {"name": "X", "psk_short": 0},
            )
        with pytest.raises(ValueError):
            MeshtasticBuiltinDecoder._psk_entry_to_bytes(
                {"name": "X", "psk_short": 256},
            )

    def test_psk_entry_with_no_key_returns_none(self) -> None:
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )
        out = MeshtasticBuiltinDecoder._psk_entry_to_bytes({"name": "X"})
        assert out is None


# ─────────────────────────────────────────────────────────────────────
# Slot enumeration
# ─────────────────────────────────────────────────────────────────────

class TestSlotEnumeration:
    def test_us_915_at_2400ksps_yields_many_slots(self) -> None:
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )
        slots = MeshtasticBuiltinDecoder._enumerate_slots(
            region_code="US", slots_mode="all",
            center_freq_hz=915_000_000, sample_rate_hz=2_400_000,
        )
        # US 902-928 has many channels per preset; in a 2.4 MS/s
        # window centered on 915, we expect many tens of slots.
        assert len(slots) >= 30

    def test_default_slots_is_smaller_than_all(self) -> None:
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )
        all_slots = MeshtasticBuiltinDecoder._enumerate_slots(
            region_code="US", slots_mode="all",
            center_freq_hz=915_000_000, sample_rate_hz=2_400_000,
        )
        default_slots = MeshtasticBuiltinDecoder._enumerate_slots(
            region_code="US", slots_mode="default",
            center_freq_hz=915_000_000, sample_rate_hz=2_400_000,
        )
        # Default slots are scattered across the full 902-928 ISM band
        # (each preset's default channel lands at a different freq),
        # so a 2.4 MS/s window around 915 MHz only catches the few
        # whose default freq falls within ±960 kHz of 915. With "all"
        # mode, every (preset, slot) in the same ±960 kHz window is
        # in scope — many tens of pairs.
        assert len(default_slots) >= 1
        assert len(all_slots) > len(default_slots) * 5

    def test_default_slots_at_full_us_passband_catches_more(self) -> None:
        """Sanity: a wider effective passband (centered to span more
        of the band) catches more default-channel slots."""
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )
        # Centered on 915 with 2.4 MS/s: ~914-916 → 1-2 default slots.
        narrow = MeshtasticBuiltinDecoder._enumerate_slots(
            region_code="US", slots_mode="default",
            center_freq_hz=915_000_000, sample_rate_hz=2_400_000,
        )
        # Centered on 906 MHz (where LONG_FAST/LONG_SLOW/LONG_MODERATE
        # cluster) catches more.
        long_band = MeshtasticBuiltinDecoder._enumerate_slots(
            region_code="US", slots_mode="default",
            center_freq_hz=906_000_000, sample_rate_hz=2_400_000,
        )
        assert len(long_band) >= len(narrow)

    def test_unknown_region_raises(self) -> None:
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )
        with pytest.raises(ValueError):
            MeshtasticBuiltinDecoder._enumerate_slots(
                region_code="ZZ", slots_mode="all",
                center_freq_hz=915_000_000, sample_rate_hz=2_400_000,
            )


# ─────────────────────────────────────────────────────────────────────
# End-to-end: run() against real capture via patched IQ source
# ─────────────────────────────────────────────────────────────────────

def _make_lease(host: str = "127.0.0.1", port: int = 1234) -> DongleLease:
    """Build a DongleLease pointing at host:port for the rtl_tcp fanout
    (which we'll have patched to be a FileIQSource so it doesn't matter
    what these values are)."""
    from rfcensus.hardware.dongle import DongleCapabilities
    caps = DongleCapabilities(
        freq_range_hz=(24_000_000, 1_766_000_000),
        max_sample_rate=2_400_000,
        bits_per_sample=8,
        bias_tee_capable=True,
        tcxo_ppm=2.0,
    )
    dongle = Dongle(
        id="test-dongle-0", serial="DONGLE0", model="rtl-sdr-v3",
        driver="rtlsdr", capabilities=caps,
    )
    return DongleLease(
        dongle=dongle, access_mode=AccessMode.SHARED,
        rtl_tcp_host=host, rtl_tcp_port=port,
        consumer="test", _lease_id=1,
    )


@pytest.mark.skipif(not _libs_built() or not _REAL_CAPTURE.exists(),
                    reason="needs native libs + real capture file")
class TestEndToEndDecode:
    """The most important test: end-to-end run on real Bay Area mesh
    capture, with RtlTcpSource patched to FileIQSource so the test
    doesn't need a live dongle."""

    def test_run_emits_decryptable_packets(self) -> None:
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )

        # Capture published events
        bus = EventBus()
        captured: list[DecodeEvent] = []

        async def grab(e: DecodeEvent) -> None:
            captured.append(e)

        bus.subscribe(DecodeEvent, grab)

        # Patch RtlTcpSource → FileIQSource so the decoder reads
        # from the capture file instead of opening a TCP socket.
        # We replace the class only for the duration of this test.
        class _FakeTcpSource:
            def __init__(self, host, port, cfg, chunk_size=65536,
                         connect_timeout=5.0):
                self._inner = FileIQSource(
                    _REAL_CAPTURE, chunk_size=chunk_size,
                )

            def read(self, n: int) -> bytes:
                return self._inner.read(n)

            def close(self) -> None:
                self._inner.close()

            def retune(self, freq_hz: int) -> None:
                pass

        async def _go():
            decoder = MeshtasticBuiltinDecoder()
            spec = DecoderRunSpec(
                lease=_make_lease(),
                freq_hz=913_500_000,
                sample_rate=1_000_000,
                duration_s=60.0,    # long enough; FileIQSource hits EOF
                event_bus=bus,
                session_id=42,
                decoder_options={
                    "meshtastic": {
                        "region": "US",
                        "slots": "all",
                    },
                },
            )
            with patch(
                "rfcensus.decoders.builtin.meshtastic.RtlTcpSource",
                _FakeTcpSource,
            ):
                result = await decoder.run(spec)
            return result

        result = asyncio.run(_go())

        # Pump events the bus may have queued
        # (EventBus runs handlers as background tasks; give them a
        # moment if any are still pending)
        for _ in range(5):
            asyncio.run(asyncio.sleep(0.05))
            if len(captured) >= result.decodes_emitted:
                break

        assert len(result.errors) == 0, f"errors: {result.errors}"
        assert result.decodes_emitted >= 6, (
            f"expected at least 6 decoded packets, got "
            f"{result.decodes_emitted}"
        )

        # All published events should be DecodeEvents for meshtastic
        assert all(e.protocol == "meshtastic" for e in captured)
        assert all(e.decoder_name == "meshtastic" for e in captured)
        assert all(e.session_id == 42 for e in captured)

        # At least 6 should have decrypted plaintext (the public
        # default channel always decrypts in this capture)
        decrypted = [e for e in captured
                     if e.payload.get("decrypted") is True]
        assert len(decrypted) >= 6, (
            f"only {len(decrypted)} decrypted out of "
            f"{len(captured)} CRC-ok packets"
        )

        # At least one should have a UTF-8 text preview ("anyone copy?")
        text_msgs = [e for e in decrypted if "text" in e.payload]
        assert len(text_msgs) >= 1, (
            "expected ≥1 text message in the capture; "
            f"got payloads: {[e.payload for e in decrypted[:3]]}"
        )
        assert any("anyone copy" in e.payload["text"].lower()
                   for e in text_msgs), (
            f"expected to see 'anyone copy?' message; "
            f"got texts: {[e.payload.get('text') for e in text_msgs]}"
        )

    def test_decode_event_has_expected_fields(self) -> None:
        """Each DecodeEvent for a CRC-ok packet should carry the
        full Meshtastic field set in its payload dict."""
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )

        bus = EventBus()
        captured: list[DecodeEvent] = []

        async def grab(e: DecodeEvent) -> None:
            captured.append(e)

        bus.subscribe(DecodeEvent, grab)

        class _FakeTcpSource:
            def __init__(self, host, port, cfg, chunk_size=65536,
                         connect_timeout=5.0):
                self._inner = FileIQSource(
                    _REAL_CAPTURE, chunk_size=chunk_size,
                )

            def read(self, n: int) -> bytes:
                return self._inner.read(n)

            def close(self) -> None:
                self._inner.close()

            def retune(self, freq_hz: int) -> None:
                pass

        async def _go():
            decoder = MeshtasticBuiltinDecoder()
            spec = DecoderRunSpec(
                lease=_make_lease(),
                freq_hz=913_500_000,
                sample_rate=1_000_000,
                duration_s=60.0,
                event_bus=bus,
                session_id=99,
            )
            with patch(
                "rfcensus.decoders.builtin.meshtastic.RtlTcpSource",
                _FakeTcpSource,
            ):
                return await decoder.run(spec)

        asyncio.run(_go())
        # Drain
        for _ in range(5):
            asyncio.run(asyncio.sleep(0.05))

        decrypted = [e for e in captured
                     if e.payload.get("decrypted") is True]
        assert decrypted

        ev = decrypted[0]
        # Field checklist matching what tracker.handle_decode +
        # downstream display expect
        for key in ["preset", "channel_hash", "from_node", "to_node",
                    "packet_id", "hop_limit", "want_ack",
                    "plaintext_hex", "_device_id", "crc_ok"]:
            assert key in ev.payload, (
                f"missing key {key!r} in payload: {ev.payload}"
            )

        # _device_id format is 'meshtastic:0x<NODEID>' — used by
        # tracker.handle_decode to register an emitter
        assert ev.payload["_device_id"].startswith("meshtastic:0x")
        assert ev.decoder_confidence == 1.0
        assert ev.freq_hz > 0

    def test_run_with_exclusive_lease_returns_error(self) -> None:
        """The decoder needs a SHARED lease (the rtl_tcp endpoint is
        what we connect to). An EXCLUSIVE lease should return a
        clean error, not crash."""
        from rfcensus.decoders.builtin.meshtastic import (
            MeshtasticBuiltinDecoder,
        )
        from rfcensus.hardware.dongle import (
            Dongle as _Dongle, DongleCapabilities as _Caps,
        )
        bus = EventBus()
        async def _go():
            d = MeshtasticBuiltinDecoder()
            caps = _Caps(
                freq_range_hz=(24_000_000, 1_766_000_000),
                max_sample_rate=2_400_000, bits_per_sample=8,
                bias_tee_capable=False, tcxo_ppm=2.0,
            )
            lease = DongleLease(
                dongle=_Dongle(id="x", serial="x", model="m",
                                 driver="d", capabilities=caps),
                access_mode=AccessMode.EXCLUSIVE,
                rtl_tcp_host=None, rtl_tcp_port=None,
                _lease_id=2,
            )
            spec = DecoderRunSpec(
                lease=lease, freq_hz=915_000_000, sample_rate=1_000_000,
                duration_s=1.0, event_bus=bus, session_id=1,
            )
            return await d.run(spec)

        result = asyncio.run(_go())
        assert result.ended_reason == "wrong_lease_type"
        assert result.errors
