"""v0.7.3 follow-up fixes — five user-reported TUI issues.

(1) Border swap: heavy=detail-shown (truly active), double=cursor-only
(2) Tile decode counter relabeled (no more "dec N" — confused with
    decimation in SDR contexts)
(3a) Footer hint string updated (r report + m mesh)
(3b) Meshtastic widget no longer flickers when toggled off (skip
     refresh when display=False)
(4) Per-dongle fanout peer list + recent decodes/detections rings
(5) Per-dongle decode counter uses the dongle's instantaneous
    bandwidth (sample_rate ± 5% rolloff) instead of a 100 kHz
    tolerance — fixes the "global counter shows 10 but per-dongle
    shows 2" inconsistency for wideband decoders like rtl_433
"""

from __future__ import annotations

from datetime import datetime, timezone


# ─────────────────────────────────────────────────────────────────────
# (1) Border swap
# ─────────────────────────────────────────────────────────────────────


def test_border_style_detail_uses_heavy() -> None:
    """The detail-shown tile (truly active) should use heavy.
    The cursor-only-but-not-detail tile should use double."""
    from rfcensus.tui.widgets import dongle_strip
    src = dongle_strip.__doc__ or ""
    src_full = open(dongle_strip.__file__).read()

    # Concrete CSS rule check — a green tile with the -detail class
    # must use heavy
    assert "-tile-green-detail  { border: heavy" in src_full, (
        "v0.7.3 swap not applied: detail tile must use heavy border"
    )
    assert "-tile-green-cursor  { border: double" in src_full, (
        "v0.7.3 swap not applied: cursor-only tile must use double border"
    )


# ─────────────────────────────────────────────────────────────────────
# (2) Tile decode counter wording
# ─────────────────────────────────────────────────────────────────────


def test_tile_decode_label_no_longer_dec() -> None:
    """The wide form of the tile counter must no longer say "dec N"
    (read as decimation in SDR contexts)."""
    src_full = open(
        "/home/claude/rfcensus/rfcensus/tui/widgets/dongle_strip.py"
    ).read()
    # The bare "dec {dec}" pattern must not be in the wide-form code
    assert 'f"dec {dec}"' not in src_full, (
        "tile must not use the confusing 'dec N' label"
    )
    # Should use either 'decodes:' or '↓N' style
    assert ('f"decodes:{dec}"' in src_full or
            'f"↓{dec}"' in src_full)


# ─────────────────────────────────────────────────────────────────────
# (3a) Footer hint
# ─────────────────────────────────────────────────────────────────────


def test_footer_hint_uses_r_report_and_m_mesh() -> None:
    from rfcensus.tui.widgets.footer import FooterBar
    hint = FooterBar.HINT
    # `r report` must be in the visible hint (matches the keystroke)
    assert "r report" in hint
    # `m mesh` (or "m meshtastic") must be in the visible hint
    assert "m mesh" in hint
    # The bare " s report" (with leading space) must NOT appear — `s`
    # is now a hidden alias and the visible label has moved to `r`.
    assert " s report" not in hint


# ─────────────────────────────────────────────────────────────────────
# (3b) Meshtastic toggle: skip refresh when hidden
# ─────────────────────────────────────────────────────────────────────


def test_meshtastic_widget_skips_refresh_when_hidden() -> None:
    """update_entries must not call self.refresh() when display=False
    — that's what was causing the widget to "fight the default log"
    when toggled off."""
    from rfcensus.tui.widgets.meshtastic_recent import MeshtasticRecentWidget
    w = MeshtasticRecentWidget()
    refresh_calls = []
    w.refresh = lambda *a, **k: refresh_calls.append(1)    # type: ignore[method-assign]

    # Simulate hidden state
    w.display = False
    w.update_entries([])
    assert refresh_calls == [], (
        "update_entries must skip refresh when display=False"
    )

    # Simulate visible state
    w.display = True
    w.update_entries([])
    assert refresh_calls == [1], (
        "update_entries must call refresh when display=True"
    )


# ─────────────────────────────────────────────────────────────────────
# (4) Fanout peer list + recent decodes/detections
# ─────────────────────────────────────────────────────────────────────


def test_fanout_peer_set_tracked_on_connect_disconnect() -> None:
    from rfcensus.tui.state import TUIState, _reduce_fanout
    from rfcensus.events import FanoutClientEvent

    state = TUIState()

    e1 = FanoutClientEvent(
        slot_id="fanout[d2]",
        peer_addr="127.0.0.1:51001",
        event_type="connect",
    )
    _reduce_fanout(state, e1)

    e2 = FanoutClientEvent(
        slot_id="fanout[d2]",
        peer_addr="127.0.0.1:51002",
        event_type="connect",
    )
    _reduce_fanout(state, e2)

    d = next(d for d in state.dongles if d.dongle_id == "d2")
    assert d.fanout_clients == 2
    assert d.fanout_client_peers == {
        "127.0.0.1:51001", "127.0.0.1:51002",
    }

    # Disconnect first peer — set shrinks
    e3 = FanoutClientEvent(
        slot_id="fanout[d2]",
        peer_addr="127.0.0.1:51001",
        event_type="disconnect",
    )
    _reduce_fanout(state, e3)
    assert d.fanout_clients == 1
    assert d.fanout_client_peers == {"127.0.0.1:51002"}


def test_recent_decode_pushed_to_dongle_ring() -> None:
    """A decode event in the dongle's passband should add an entry
    to the per-dongle recent_decodes ring (used by the detail pane)."""
    from rfcensus.tui.state import TUIState, DongleState, _reduce_decode
    from rfcensus.events import DecodeEvent

    state = TUIState()
    state.dongles.append(DongleState(
        dongle_id="d2",
        consumer="rtl_433:433_ism",
        freq_hz=433_920_000,
        sample_rate=2_400_000,
    ))

    # Decode at 433.470 — well within ±1.14 MHz of 433.920
    ev = DecodeEvent(
        decoder_name="rtl_433", protocol="tpms",
        freq_hz=433_470_000,
        payload={"msg_type": "TPMS-Schrader", "id": "0x1234"},
        timestamp=datetime.now(timezone.utc),
    )
    _reduce_decode(state, ev)

    d = state.dongles[0]
    assert len(d.recent_decodes) == 1
    entry = d.recent_decodes[0]
    assert entry.protocol == "tpms"
    assert entry.freq_hz == 433_470_000
    # Summary should be the formatter's compact rendering
    assert "TPMS-Schrader" in entry.summary


def test_recent_detection_pushed_to_dongle_ring() -> None:
    """A detection event in a band attributed to a dongle should
    add an entry to the per-dongle recent_detections ring."""
    from rfcensus.tui.state import (
        TUIState, DongleState, _reduce_detection, TaskState,
    )
    from rfcensus.events import DetectionEvent

    state = TUIState()
    state.dongles.append(DongleState(dongle_id="d1"))
    # Wire up the band→dongle attribution path that
    # _reduce_detection uses
    state.active_tasks[(0, "915_ism")] = TaskState(
        band_id="915_ism", dongle_id="d1",
        consumer="lora_survey:915_ism",
        started_at=datetime.now(timezone.utc),
    )

    ev = DetectionEvent(
        detector_name="lora_survey",
        technology="lora",
        freq_hz=915_000_000,
        bandwidth_hz=125_000,
        confidence=0.85,
        evidence="chirp pattern",
        metadata={"band_id": "915_ism"},
        timestamp=datetime.now(timezone.utc),
    )
    _reduce_detection(state, ev)

    d = state.dongles[0]
    assert len(d.recent_detections) == 1
    assert d.recent_detections[0].technology == "lora"
    assert d.recent_detections[0].confidence == 0.85


def test_release_clears_per_band_rings_and_peers() -> None:
    """When a dongle's lease is released, the per-band rings and the
    fanout peer set must be cleared so the next allocation starts
    with fresh state."""
    from rfcensus.tui.state import (
        TUIState, DongleState, _DongleDecodeEntry,
        _reduce_hardware,
    )
    from rfcensus.events import HardwareEvent

    state = TUIState()
    d = DongleState(
        dongle_id="d2",
        consumer="rtl_433:433_ism",
        freq_hz=433_920_000,
        sample_rate=2_400_000,
    )
    d.recent_decodes.append(_DongleDecodeEntry(
        timestamp=datetime.now(timezone.utc),
        freq_hz=433_470_000, protocol="tpms", summary="TPMS X",
    ))
    d.fanout_client_peers.add("127.0.0.1:51001")
    d.fanout_clients = 1
    state.dongles.append(d)

    release = HardwareEvent(
        dongle_id="d2",
        kind="released",
        consumer="rtl_433:433_ism",
        timestamp=datetime.now(timezone.utc),
    )
    _reduce_hardware(state, release)

    assert d.recent_decodes == []
    assert d.fanout_client_peers == set()
    assert d.fanout_clients == 0


# ─────────────────────────────────────────────────────────────────────
# (5) Per-dongle decode count uses bandwidth, not 100 kHz tolerance
# ─────────────────────────────────────────────────────────────────────


def test_decode_attributed_when_within_passband() -> None:
    """The bug: dongle tuned to 433.920 @ 2.4 MS/s should attribute
    decodes anywhere in its 432.7–435.1 MHz instantaneous bandwidth.
    Old code used 100 kHz tolerance — most decodes missed."""
    from rfcensus.tui.state import TUIState, DongleState, _reduce_decode
    from rfcensus.events import DecodeEvent

    state = TUIState()
    d = DongleState(
        dongle_id="d2",
        consumer="rtl_433:433_ism",
        freq_hz=433_920_000,
        sample_rate=2_400_000,
    )
    state.dongles.append(d)

    # Realistic rtl_433 decode frequencies on a 433 ISM dongle
    for freq in (433_470_000, 433_989_000, 433_521_000,
                 433_816_000, 433_489_000):
        _reduce_decode(state, DecodeEvent(
            decoder_name="rtl_433", protocol="tpms",
            freq_hz=freq, payload={"msg_type": "X"},
            timestamp=datetime.now(timezone.utc),
        ))

    # ALL 5 decodes should attribute to dongle d2 (they're within
    # ±1.14 MHz of the 433.920 center). Pre-fix: 0 attributed
    # because 433.470 is 450 kHz from center > 100 kHz tolerance.
    assert d.decodes_in_band == 5, (
        f"expected 5 attributed decodes, got {d.decodes_in_band}"
    )
    assert state.total_decodes == 5


def test_decode_outside_passband_not_attributed() -> None:
    """Decodes well outside the dongle's bandwidth must NOT attribute
    (otherwise the global counter would contaminate cross-band
    counters when multiple dongles are active)."""
    from rfcensus.tui.state import TUIState, DongleState, _reduce_decode
    from rfcensus.events import DecodeEvent

    state = TUIState()
    d_433 = DongleState(
        dongle_id="d2", consumer="rtl_433:433_ism",
        freq_hz=433_920_000, sample_rate=2_400_000,
    )
    state.dongles.append(d_433)

    # Decode at 915 MHz — way outside d_433's passband
    _reduce_decode(state, DecodeEvent(
        decoder_name="rtl_433", protocol="ert_scm",
        freq_hz=915_000_000, payload={"msg_type": "SCM"},
        timestamp=datetime.now(timezone.utc),
    ))

    assert d_433.decodes_in_band == 0
    assert state.total_decodes == 1   # global still bumps


def test_decode_with_no_sample_rate_uses_safe_default() -> None:
    """If sample_rate isn't set yet (event arrives before the first
    HardwareEvent), use a ±1 MHz default — safer than the old
    100 kHz but won't cross-contaminate the typical multi-band
    dongle layout where centers are tens of MHz apart."""
    from rfcensus.tui.state import TUIState, DongleState, _reduce_decode
    from rfcensus.events import DecodeEvent

    state = TUIState()
    d = DongleState(
        dongle_id="d2", consumer="rtl_433:433_ism",
        freq_hz=433_920_000,
        sample_rate=None,    # not yet set
    )
    state.dongles.append(d)

    # Decode at 433.5 MHz (420 kHz from center) — within ±950 kHz
    # default half-bandwidth
    _reduce_decode(state, DecodeEvent(
        decoder_name="rtl_433", protocol="tpms",
        freq_hz=433_500_000, payload={"msg_type": "X"},
        timestamp=datetime.now(timezone.utc),
    ))
    assert d.decodes_in_band == 1
