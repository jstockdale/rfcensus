"""v0.7.16 real-capture fixture regressions.

These tests lock in coverage that synthetic round-trips can't match:

  • SF11 sync gap is constant 128 samples (hardcoded for SF≥10),
    NOT N/4 — the v0.7.15 hypothesis that broke at SF≤9 and BW=125k.
  • LDRO=True must be set for SF11/BW=125k (LongModerate). Wrong
    LDRO → buffer overrun in codec → SIGILL via stack-protector.
  • End-to-end decode through the eager MultiPresetPipeline against
    real-world IQ — RF impairments (CFO, AGC ramps, multipath) that
    test_synth.c can't simulate.

Both fixtures are extracted small windows around verified packets in
John's handheld captures (LongFast 907 MHz, LongModerate 903 MHz).
The window is sized to give the SF11 demod time to chew through any
false-positive sync candidates AND complete the real packet decode —
1.5s is enough at SF9, but SF11/BW=125k packets are ~580ms each, so
3s is the floor for LongModerate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
from rfcensus.decoders.meshtastic_pipeline import MultiPresetPipeline
from rfcensus.utils.meshtastic_region import (
    PRESETS,
    enumerate_all_slots_in_passband,
)


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
LF_FIXTURE = FIXTURE_DIR / "meshtastic_real_907mhz_longfast.cu8"
LM_FIXTURE = FIXTURE_DIR / "meshtastic_real_903mhz_longmoderate.cu8"
SF_FIXTURE = FIXTURE_DIR / "meshtastic_real_918mhz_shortfast.cu8"


def _decode_fixture(
    fixture_path: Path,
    center_freq_hz: int,
    preset_key: str,
    sample_rate_hz: int = 2_400_000,
) -> list:
    """Run the fixture through MultiPresetPipeline, return packets."""
    slots = enumerate_all_slots_in_passband(
        region_code="US",
        center_freq_hz=center_freq_hz,
        sample_rate_hz=sample_rate_hz,
        presets=[preset_key],
    )
    if not slots:
        raise RuntimeError(
            f"no {preset_key} slots fit at center={center_freq_hz}, "
            f"sr={sample_rate_hz} — check fixture parameters",
        )
    mesh = MeshtasticDecoder(preset_key)
    # Register the default channel for this preset using libmeshtastic's
    # short-index PSK (b'\x01' = MESH_DEFAULT_PSK) — same setup that
    # decode_meshtastic.py performs for default-channel traffic.
    mesh.add_channel(
        PRESETS[preset_key].display_name,
        psk=b"\x01",
        is_primary=False,
    )
    pipeline = MultiPresetPipeline(
        slots=slots,
        sample_rate_hz=sample_rate_hz,
        center_freq_hz=center_freq_hz,
        mesh=mesh,
    )
    raw = fixture_path.read_bytes()
    chunk_bytes = 65536
    for i in range(0, len(raw), chunk_bytes):
        pipeline.feed_cu8(raw[i:i + chunk_bytes])
    return list(pipeline.pop_packets())


@pytest.mark.skipif(
    not LF_FIXTURE.exists(),
    reason="fixture meshtastic_real_907mhz_longfast.cu8 not present",
)
def test_longfast_fixture_decodes_howdy_claude():
    """LongFast (SF11/BW=250k, LDRO=False) — a clean text-message
    packet must decode with CRC ✓ and the expected payload string.

    This locks in the SF11 sync-gap fix from v0.7.16: with the prior
    sync_gap = N/4 logic, this fixture decoded ZERO packets despite
    8 valid preambles. The 128-sample constant gap fix produces clean
    decode of John's 'howdy claude' Meshtastic test message.
    """
    packets = _decode_fixture(
        LF_FIXTURE,
        center_freq_hz=907_000_000,
        preset_key="LONG_FAST",
    )
    crc_ok_packets = [p for p in packets if p.lora.crc_ok]
    assert crc_ok_packets, (
        f"expected ≥1 CRC-pass packet; got {len(packets)} total, "
        f"{len(crc_ok_packets)} CRC-pass"
    )
    # At least one decode should match 'howdy claude' (TEXT_MESSAGE_APP)
    matched = [
        p for p in crc_ok_packets
        if p.mesh is not None
        and p.mesh.decrypted
        and b"howdy claude" in p.mesh.plaintext
    ]
    assert matched, (
        f"expected a 'howdy claude' text packet; got "
        f"{len(crc_ok_packets)} CRC-pass packet(s) but none matched. "
        f"Plaintext heads: "
        f"{[p.mesh.plaintext[:60] if p.mesh else None for p in crc_ok_packets[:3]]!r}"
    )


@pytest.mark.skipif(
    not LM_FIXTURE.exists(),
    reason="fixture meshtastic_real_903mhz_longmoderate.cu8 not present",
)
def test_longmoderate_fixture_decodes_this_is_a_test():
    """LongModerate (SF11/BW=125k, LDRO=True) — a clean text-message
    packet must decode with CRC ✓.

    Locks in TWO v0.7.16 fixes simultaneously:
      1. SF11 sync gap = 128 samples (not N/4). Same fix as LongFast.
         Validates the constant-128-samples hypothesis at BW=125k as
         well as BW=250k.
      2. LDRO=True for SF11/BW=125k. Without this, the codec layer's
         buffer-size assumption breaks (different sf_app_pld) and the
         decoder SIGILLs partway through the first packet.

    Both fixes are required to reach this packet — either alone leaves
    the test failing (SF11 fix alone: still SIGILL; LDRO fix alone: 0
    sync matches because gap is wrong).
    """
    packets = _decode_fixture(
        LM_FIXTURE,
        center_freq_hz=903_000_000,
        preset_key="LONG_MODERATE",
    )
    crc_ok_packets = [p for p in packets if p.lora.crc_ok]
    assert crc_ok_packets, (
        f"expected ≥1 CRC-pass packet; got {len(packets)} total, "
        f"{len(crc_ok_packets)} CRC-pass"
    )
    matched = [
        p for p in crc_ok_packets
        if p.mesh is not None
        and p.mesh.decrypted
        and b"this is a test" in p.mesh.plaintext
    ]
    assert matched, (
        f"expected a 'this is a test' text packet; got "
        f"{len(crc_ok_packets)} CRC-pass packet(s) but none matched. "
        f"Plaintext heads: "
        f"{[p.mesh.plaintext[:60] if p.mesh else None for p in crc_ok_packets[:3]]!r}"
    )


@pytest.mark.skipif(
    not SF_FIXTURE.exists(),
    reason="fixture meshtastic_real_918mhz_shortfast.cu8 not present",
)
def test_shortfast_fixture_decodes_hi_claude():
    """ShortFast (SF7/BW=250k, LDRO=False) — a clean text-message packet
    must decode with CRC ✓.

    Coverage rationale: ShortFast is the SMALLEST LoRa configuration
    we support (N=128, 4ms preamble at BW=250k) and has the tightest
    timing margins for the lazy-pipeline lookback + spawn machinery —
    much less slack than SF11. SF7 also has the lowest processing gain
    at +21dB, so probe SNR thresholds and detector-latency budgets
    that are comfortable at SF11 (+33dB gain) can quietly fail here.
    Locking this in protects against future regressions in any of:
      • probe sensitivity (default 20dB threshold vs SF7's marginal SNR)
      • lookback sizing (4ms preamble fits in 1024 BB samples — only)
      • channelizer filter delay
      • detector trigger latency
    """
    packets = _decode_fixture(
        SF_FIXTURE,
        center_freq_hz=918_000_000,
        preset_key="SHORT_FAST",
    )
    crc_ok_packets = [p for p in packets if p.lora.crc_ok]
    assert crc_ok_packets, (
        f"expected ≥1 CRC-pass packet; got {len(packets)} total, "
        f"{len(crc_ok_packets)} CRC-pass"
    )
    matched = [
        p for p in crc_ok_packets
        if p.mesh is not None
        and p.mesh.decrypted
        and b"hi claude" in p.mesh.plaintext
    ]
    assert matched, (
        f"expected a 'hi claude' text packet; got "
        f"{len(crc_ok_packets)} CRC-pass packet(s) but none matched. "
        f"Plaintext heads: "
        f"{[p.mesh.plaintext[:60] if p.mesh else None for p in crc_ok_packets[:3]]!r}"
    )
