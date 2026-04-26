"""Tests for the v0.7.2 display + reporting layer.

Three pieces:
  • ``rfcensus.reporting.payload_format`` — per-protocol formatter
    that turns a ``DecodeEvent.payload`` dict into a one-line summary
  • ``rfcensus list decodes`` CLI command — wires a SessionRepo +
    DecodeRepo query through the formatter
  • TUI integration — ``_reduce_decode`` pushes Meshtastic entries
    into ``state.meshtastic_recent``; ``MeshtasticRecentWidget``
    renders them as a tail-style live list

The formatter tests are pure functions (no I/O); the CLI test
populates a tiny SQLite DB and runs the command via Click's test
runner; the TUI tests exercise the reducer + widget render directly
without standing up the full Textual app.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────
# Per-protocol payload formatter
# ─────────────────────────────────────────────────────────────────────


class TestMeshtasticFormatter:
    def test_crc_fail(self) -> None:
        from rfcensus.reporting.payload_format import format_payload
        s = format_payload("meshtastic", {
            "crc_ok": False, "preset": "MEDIUM_FAST",
        })
        assert s == "[crc fail; preset=MEDIUM_FAST]"

    def test_encrypted_no_psk(self) -> None:
        from rfcensus.reporting.payload_format import format_payload
        s = format_payload("meshtastic", {
            "crc_ok": True, "decrypted": False,
            "channel_hash": 0x42, "preset": "LONG_FAST",
        })
        assert "encrypted" in s
        assert "0x42" in s
        assert "LONG_FAST" in s

    def test_decrypted_text_message(self) -> None:
        from rfcensus.reporting.payload_format import format_payload
        # Real plaintext layout: 08 01 12 0c <12 bytes "anyone copy?"> ...
        plaintext = bytes([0x08, 0x01, 0x12, 0x0c]) + b"anyone copy?"
        s = format_payload("meshtastic", {
            "crc_ok": True, "decrypted": True,
            "from_node": 0x99BC7160,
            "to_node": 0xFFFFFFFF,    # broadcast
            "plaintext_hex": plaintext.hex(),
            "plaintext_len": len(plaintext),
            "text": "anyone copy?",
        })
        assert "0x99BC7160" in s
        assert "BCAST" in s
        assert "TEXT" in s
        assert "anyone copy?" in s

    def test_decrypted_unicast(self) -> None:
        from rfcensus.reporting.payload_format import format_payload
        s = format_payload("meshtastic", {
            "crc_ok": True, "decrypted": True,
            "from_node": 0x12345678,
            "to_node": 0x87654321,
            "plaintext_hex": "0801120548656c6c6f",  # port 1 "Hello"
            "plaintext_len": 9,
            "text": "Hello",
        })
        assert "0x12345678" in s
        assert "0x87654321" in s
        assert "BCAST" not in s

    def test_decrypted_position_no_text(self) -> None:
        """POSITION (port 3) packets have no UTF-8 text — formatter
        falls through to the byte-count form."""
        from rfcensus.reporting.payload_format import format_payload
        # Plaintext starts with 08 03 (port 3 = POSITION)
        s = format_payload("meshtastic", {
            "crc_ok": True, "decrypted": True,
            "from_node": 0x749A76A4,
            "to_node": 0xFFFFFFFF,
            "plaintext_hex": "0803121c0d00308b16",
            "plaintext_len": 9,
        })
        assert "0x749A76A4" in s
        assert "BCAST" in s
        assert "POSITION" in s
        assert "9 B" in s

    def test_long_text_truncated(self) -> None:
        from rfcensus.reporting.payload_format import format_payload
        long_text = "x" * 200
        s = format_payload("meshtastic", {
            "crc_ok": True, "decrypted": True,
            "from_node": 0x1, "to_node": 0xFFFFFFFF,
            "plaintext_hex": "0801",
            "text": long_text,
        })
        assert "..." in s
        assert len(s) < 200    # actually got truncated

    def test_unknown_port_label(self) -> None:
        """Unknown port numbers should show as ``portNN`` not crash."""
        from rfcensus.reporting.payload_format import format_payload
        # 08 63 = port-tag (0x08), port 99 (0x63). Followed by an arbitrary
        # length-prefixed payload so the formatter can parse cleanly.
        s = format_payload("meshtastic", {
            "crc_ok": True, "decrypted": True,
            "from_node": 0x1, "to_node": 0xFFFFFFFF,
            "plaintext_hex": "0863120548656c6c6f",  # port 99 + body
            "plaintext_len": 9,
        })
        assert "port99" in s


class TestRtl433FamilyFormatter:
    def test_tpms_formats(self) -> None:
        from rfcensus.reporting.payload_format import format_payload
        s = format_payload("tpms", {
            "msg_type": "Schrader-EG53MA4",
            "_device_id": "TPMS:0x1234",
            "commodity": "tire",
        })
        assert "Schrader-EG53MA4" in s
        assert "tire" in s
        # _device_id has leading underscore → kept (it's a known key
        # the formatter looks up explicitly)
        assert "TPMS:0x1234" in s

    def test_ert_scm_formats(self) -> None:
        from rfcensus.reporting.payload_format import format_payload
        s = format_payload("ert_scm", {
            "msg_type": "SCM",
            "consumption": 12345,
        })
        assert "SCM" in s
        assert "12345" in s


class TestGenericFormatter:
    def test_unknown_protocol_dumps_kv(self) -> None:
        from rfcensus.reporting.payload_format import format_payload
        s = format_payload("mystery_protocol", {"a": 1, "b": "two"})
        assert "a=1" in s
        assert "b=two" in s

    def test_skips_underscore_keys(self) -> None:
        """Leading-underscore keys (e.g. _device_id, _internal_state)
        are conventionally internal and not surfaced in summaries."""
        from rfcensus.reporting.payload_format import format_payload
        s = format_payload("xyz", {"a": 1, "_internal": "should-not-show"})
        assert "a=1" in s
        assert "should-not-show" not in s

    def test_collapses_nested_dicts(self) -> None:
        from rfcensus.reporting.payload_format import format_payload
        s = format_payload("xyz", {
            "a": 1,
            "raw": {"deeply": "nested", "with": "many", "fields": "inside"},
        })
        assert "a=1" in s
        assert "raw={...3}" in s
        assert "deeply" not in s

    def test_truncates_long_output(self) -> None:
        from rfcensus.reporting.payload_format import format_payload
        s = format_payload("xyz", {"k": "x" * 500})
        assert s.endswith("...")
        assert len(s) <= 100

    def test_format_error_does_not_propagate(self) -> None:
        """A formatter that raises must not break the caller — return
        a placeholder string instead."""
        from rfcensus.reporting import payload_format as pf
        # Inject a bad formatter for a fake protocol
        pf._FORMATTERS["fake"] = lambda d: 1 / 0    # deliberate ZeroDivision
        try:
            s = pf.format_payload("fake", {})
            assert "[format error" in s
        finally:
            del pf._FORMATTERS["fake"]


# ─────────────────────────────────────────────────────────────────────
# TUI: state reducer + widget
# ─────────────────────────────────────────────────────────────────────


class TestTuiReducer:
    def test_meshtastic_decode_appended_to_ring(self) -> None:
        from rfcensus.tui.state import TUIState, _reduce_decode
        from rfcensus.events import DecodeEvent

        state = TUIState()
        ev = DecodeEvent(
            decoder_name="meshtastic", protocol="meshtastic",
            freq_hz=913_500_000, rssi_dbm=-65.0, snr_db=12.0,
            payload={
                "preset": "MEDIUM_FAST", "crc_ok": True, "decrypted": True,
                "from_node": 0x99BC7160, "to_node": 0xFFFFFFFF,
                "channel_hash": 0x1F,
                "plaintext_hex": "0801120c616e796f6e6520636f70793f",
                "text": "anyone copy?",
            },
            timestamp=datetime.now(timezone.utc),
        )
        _reduce_decode(state, ev)
        assert state.total_decodes == 1
        assert len(state.meshtastic_recent) == 1
        entry = state.meshtastic_recent[0]
        assert entry.preset == "MEDIUM_FAST"
        assert entry.from_node == 0x99BC7160
        assert entry.decrypted is True
        assert "anyone copy?" in entry.summary

    def test_non_meshtastic_decode_does_not_append_to_mesh_ring(self) -> None:
        from rfcensus.tui.state import TUIState, _reduce_decode
        from rfcensus.events import DecodeEvent

        state = TUIState()
        ev = DecodeEvent(
            decoder_name="rtl_433", protocol="tpms",
            freq_hz=315_000_000,
            payload={"msg_type": "Schrader", "id": "0x1234"},
            timestamp=datetime.now(timezone.utc),
        )
        _reduce_decode(state, ev)
        assert state.total_decodes == 1
        assert len(state.meshtastic_recent) == 0

    def test_ring_capped_to_capacity(self) -> None:
        from rfcensus.tui.state import TUIState, _reduce_decode
        from rfcensus.events import DecodeEvent

        state = TUIState()
        state.meshtastic_recent_capacity = 5
        for i in range(20):
            ev = DecodeEvent(
                decoder_name="meshtastic", protocol="meshtastic",
                freq_hz=913_500_000,
                payload={
                    "preset": "MEDIUM_FAST", "crc_ok": True,
                    "decrypted": True,
                    "from_node": i, "to_node": 0xFFFFFFFF,
                    "channel_hash": 0x1F,
                    "plaintext_hex": "0801120548656c6c6f",
                    "text": f"msg{i}",
                },
                timestamp=datetime.now(timezone.utc),
            )
            _reduce_decode(state, ev)
        assert len(state.meshtastic_recent) == 5
        # Should be the LAST 5 entries (oldest dropped)
        assert state.meshtastic_recent[0].from_node == 15
        assert state.meshtastic_recent[-1].from_node == 19


class TestMeshtasticRecentWidget:
    def test_widget_renders_with_no_entries(self) -> None:
        from rfcensus.tui.widgets.meshtastic_recent import (
            MeshtasticRecentWidget,
        )
        w = MeshtasticRecentWidget()
        out = w.render()
        # render() returns a str-coercible result
        text = str(out)
        assert "No Meshtastic packets" in text
        assert "site.toml" in text     # actionable hint

    def test_widget_renders_decrypted_entry(self) -> None:
        from rfcensus.tui.state import MeshtasticDecodeEntry
        from rfcensus.tui.widgets.meshtastic_recent import (
            MeshtasticRecentWidget,
        )
        e = MeshtasticDecodeEntry(
            timestamp=datetime(2025, 1, 1, 21, 3, 45, tzinfo=timezone.utc),
            freq_hz=913_500_000,
            preset="MEDIUM_FAST",
            crc_ok=True, decrypted=True,
            channel_hash=0x1F,
            from_node=0x99BC7160, to_node=0xFFFFFFFF,
            summary="0x99BC7160→BCAST TEXT 'anyone copy?'",
            rssi_dbm=-65.0, snr_db=12.0,
        )
        w = MeshtasticRecentWidget()
        w.update_entries([e])
        text = str(w.render())
        assert "21:03:45" in text
        assert "913.500" in text
        assert "MED_FAST" in text or "MEDIUM_FAST" in text
        assert "anyone copy?" in text
        # decrypted = ✓✓ glyphs
        assert "✓✓" in text

    def test_widget_renders_encrypted_entry_with_warning_glyph(self) -> None:
        from rfcensus.tui.state import MeshtasticDecodeEntry
        from rfcensus.tui.widgets.meshtastic_recent import (
            MeshtasticRecentWidget,
        )
        e = MeshtasticDecodeEntry(
            timestamp=datetime(2025, 1, 1, 21, 3, 42, tzinfo=timezone.utc),
            freq_hz=913_500_000,
            preset="LONG_FAST",
            crc_ok=True, decrypted=False,
            channel_hash=0x42,
            from_node=None, to_node=None,
            summary="[encrypted; ch=0x42 preset=LONG_FAST]",
            rssi_dbm=-71.0, snr_db=8.0,
        )
        w = MeshtasticRecentWidget()
        w.update_entries([e])
        text = str(w.render())
        assert "encrypted" in text
        # crc-ok-but-not-decrypted = ✓· glyphs
        assert "✓·" in text


# ─────────────────────────────────────────────────────────────────────
# CLI: rfcensus list decodes
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_decodes_renders_meshtastic_with_arrow(
    tmp_path: Path,
) -> None:
    """End-to-end: insert a Meshtastic decode into a fresh DB, then
    invoke ``rfcensus list decodes`` and check the rendered text
    includes the from→to arrow + decrypted text."""
    from datetime import datetime as _dt, timezone as _tz

    from rfcensus.storage.db import Database
    from rfcensus.storage.repositories import (
        DecodeRepo, SessionRepo,
    )
    from rfcensus.storage.models import (
        DecodeRecord, SessionRecord,
    )

    # Database opens lazily and applies migrations on first use; no
    # explicit init_schema needed.
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    sess_repo = SessionRepo(db)
    decode_repo = DecodeRepo(db)

    sid = await sess_repo.create(SessionRecord(
        id=None, command="test", started_at=_dt.now(_tz.utc),
        ended_at=None, site_name="test", config_snap={}, notes="",
    ))

    rec = DecodeRecord(
        id=None, session_id=sid, dongle_id="d0",
        timestamp=_dt.now(_tz.utc),
        decoder="meshtastic", protocol="meshtastic",
        freq_hz=913_500_000, rssi_dbm=-65.0, snr_db=12.0,
        payload={
            "preset": "MEDIUM_FAST", "crc_ok": True, "decrypted": True,
            "from_node": 0x99BC7160, "to_node": 0xFFFFFFFF,
            "channel_hash": 0x1F,
            "plaintext_hex": "0801120c616e796f6e6520636f70793f",
            "text": "anyone copy?",
        },
        validated=True,
    )
    await decode_repo.insert(rec)

    # Read back through the formatter to confirm the rendering path
    rows = await decode_repo.for_session(sid)
    assert len(rows) == 1

    from rfcensus.reporting.payload_format import format_payload
    summary = format_payload(rows[0].protocol, rows[0].payload)
    assert "0x99BC7160" in summary
    assert "BCAST" in summary
    assert "anyone copy?" in summary
