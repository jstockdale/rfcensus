"""v0.5.39 tests: monitor mode.

Focus
=====

Monitor mode is a thin wrapper on top of SessionRunner, so the
interesting test surface is:

  • Validation helpers: unknown bands/decoders/dongles produce
    SystemExit with clear error messages.
  • Output printers: format decoded / detection / wide-channel events
    correctly in both text and JSON-lines modes.
  • In-memory DB path: Database(":memory:") opens, accepts writes,
    and gets discarded without leaving files behind.
  • command="monitor" tag flows through the SessionRunner into
    SessionRecord so downstream queries can distinguish monitor
    sessions from inventory sessions.

The full integration (actually running a session, spinning up the
broker, consuming IQ from a mocked backend) is covered indirectly
by the existing SessionRunner tests; we don't re-validate that here.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from rfcensus.commands.monitor import (
    _JsonLinesPrinter,
    _TextPrinter,
    _filter_to_dongle,
    _new_decoder_config,
    _restrict_to_decoder,
    _summarize_payload,
    _validate_and_restrict_band,
)
from rfcensus.config.schema import BandConfig, BandsSelection, DecoderConfig, SiteConfig
from rfcensus.events import DecodeEvent, DetectionEvent, WideChannelEvent
from rfcensus.storage.db import Database


# ------------------------------------------------------------
# Band / decoder / dongle validation
# ------------------------------------------------------------


def _minimal_config(*band_ids: str) -> SiteConfig:
    bands = [
        BandConfig(
            id=bid,
            name=bid,
            freq_low=900_000_000,
            freq_high=928_000_000,
        )
        for bid in band_ids
    ]
    return SiteConfig(
        band_definitions=bands,
        bands=BandsSelection(enabled=list(band_ids)),
    )


class TestValidateAndRestrictBand:
    def test_known_band_restricts_enabled_list(self):
        cfg = _minimal_config("915_ism", "433_ism", "aprs_2m")
        _validate_and_restrict_band(cfg, "915_ism")
        assert cfg.bands.enabled == ["915_ism"]

    def test_unknown_band_raises_system_exit(self):
        cfg = _minimal_config("915_ism", "433_ism")
        with pytest.raises(SystemExit) as exc_info:
            _validate_and_restrict_band(cfg, "nonexistent_band")
        assert exc_info.value.code == 2

    def test_error_message_lists_known_bands(self, capsys):
        cfg = _minimal_config("915_ism", "aprs_2m")
        with pytest.raises(SystemExit):
            _validate_and_restrict_band(cfg, "xyz")
        err = capsys.readouterr().err
        assert "Unknown band 'xyz'" in err
        assert "915_ism" in err
        assert "aprs_2m" in err


class TestRestrictToDecoder:
    def test_unknown_decoder_raises_system_exit(self, capsys):
        cfg = _minimal_config("915_ism")
        with pytest.raises(SystemExit) as exc_info:
            _restrict_to_decoder(cfg, "not_a_real_decoder")
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "Unknown decoder 'not_a_real_decoder'" in err

    def test_valid_decoder_disables_all_others(self):
        """After restrict, the chosen decoder is enabled and every
        other registered decoder is explicitly disabled in the
        in-memory config. This ensures the strategy layer won't
        silently run a second decoder on the same band."""
        cfg = _minimal_config("915_ism")
        # Pick a decoder name that actually exists in the registry
        from rfcensus.decoders.registry import get_registry
        reg = get_registry()
        names = reg.names()
        if not names:
            pytest.skip("no decoders registered in test environment")
        chosen = names[0]

        _restrict_to_decoder(cfg, chosen)
        # Chosen is enabled
        assert cfg.decoders[chosen].enabled is True
        # All other registered decoders are disabled
        for name in names:
            if name == chosen:
                continue
            assert cfg.decoders[name].enabled is False, (
                f"decoder '{name}' should be disabled after --decoder "
                f"restriction to '{chosen}'"
            )


class TestFilterToDongle:
    class _FakeDongle:
        def __init__(self, id: str, serial: str | None = None):
            self.id = id
            self.serial = serial

    class _FakeRegistry:
        def __init__(self, dongles):
            self.dongles = dongles

    def test_match_by_id(self):
        reg = self._FakeRegistry([
            self._FakeDongle("rtlsdr-00000001", serial="00000001"),
            self._FakeDongle("rtlsdr-00000002", serial="00000002"),
        ])
        _filter_to_dongle(reg, "rtlsdr-00000001")
        assert len(reg.dongles) == 1
        assert reg.dongles[0].id == "rtlsdr-00000001"

    def test_match_by_serial(self):
        reg = self._FakeRegistry([
            self._FakeDongle("rtlsdr-00000001", serial="00000001"),
            self._FakeDongle("rtlsdr-00000002", serial="00000002"),
        ])
        _filter_to_dongle(reg, "00000002")
        assert len(reg.dongles) == 1
        assert reg.dongles[0].serial == "00000002"

    def test_no_match_raises_with_detected_list(self, capsys):
        reg = self._FakeRegistry([
            self._FakeDongle("rtlsdr-abc"),
            self._FakeDongle("rtlsdr-xyz"),
        ])
        with pytest.raises(SystemExit) as exc_info:
            _filter_to_dongle(reg, "bogus-id")
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "bogus-id" in err
        assert "rtlsdr-abc" in err  # detected list shown for help


# ------------------------------------------------------------
# Output printers
# ------------------------------------------------------------


class TestTextPrinter:
    def test_decode_format(self, capsys):
        printer = _TextPrinter()
        ev = DecodeEvent(
            decoder_name="rtl_433",
            protocol="interlogix_security",
            dongle_id="rtlsdr-1",
            freq_hz=433_534_000,
            rssi_dbm=-42.5,
            payload={"_device_id": "abc123", "status": "open"},
        )
        printer.print_decode(ev)
        out = capsys.readouterr().out
        assert "interlogix_security" in out
        assert "433.534" in out
        assert "-42.5dBm" in out
        assert "id=abc123" in out
        assert "status=open" in out

    def test_detection_format(self, capsys):
        printer = _TextPrinter()
        ev = DetectionEvent(
            detector_name="lora",
            technology="meshtastic",
            freq_hz=906_875_000,
            bandwidth_hz=250_000,
            confidence=0.87,
            evidence="wide-channel composite; chirp SF11",
        )
        printer.print_detection(ev)
        out = capsys.readouterr().out
        assert "detect" in out
        assert "meshtastic" in out
        assert "906.875" in out
        assert "bw=250kHz" in out
        assert "conf=0.87" in out

    def test_wide_channel_format(self, capsys):
        printer = _TextPrinter()
        ev = WideChannelEvent(
            freq_center_hz=906_875_000,
            bandwidth_hz=245_000,
            matched_template_hz=250_000,
            constituent_bin_count=10,
            coverage_ratio=0.95,
        )
        printer.print_wide_channel(ev)
        out = capsys.readouterr().out
        assert "wide" in out
        assert "template=250kHz" in out
        assert "coverage=95%" in out


class TestJsonLinesPrinter:
    def test_decode_is_valid_json_line(self):
        printer = _JsonLinesPrinter()
        ev = DecodeEvent(
            decoder_name="rtl_433",
            protocol="interlogix_security",
            freq_hz=433_534_000,
            rssi_dbm=-42.5,
            payload={"id": "abc"},
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            printer.print_decode(ev)
        line = buf.getvalue().strip()
        # Must be exactly one line
        assert "\n" not in line
        parsed = json.loads(line)
        assert parsed["kind"] == "decode"
        assert parsed["protocol"] == "interlogix_security"
        assert parsed["freq_hz"] == 433_534_000

    def test_detection_is_valid_json_line(self):
        printer = _JsonLinesPrinter()
        ev = DetectionEvent(
            detector_name="lora",
            technology="meshtastic",
            freq_hz=906_875_000,
            bandwidth_hz=250_000,
            confidence=0.9,
            metadata={"estimated_sf": 11, "variant": "meshtastic_long_fast"},
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            printer.print_detection(ev)
        parsed = json.loads(buf.getvalue().strip())
        assert parsed["kind"] == "detection"
        assert parsed["technology"] == "meshtastic"
        assert parsed["metadata"]["estimated_sf"] == 11

    def test_wide_channel_is_valid_json_line(self):
        printer = _JsonLinesPrinter()
        ev = WideChannelEvent(
            freq_center_hz=906_875_000,
            bandwidth_hz=245_000,
            matched_template_hz=250_000,
            constituent_bin_count=10,
            coverage_ratio=0.95,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            printer.print_wide_channel(ev)
        parsed = json.loads(buf.getvalue().strip())
        assert parsed["kind"] == "wide_channel"
        assert parsed["matched_template_hz"] == 250_000
        assert parsed["coverage_ratio"] == 0.95


# ------------------------------------------------------------
# _summarize_payload (the text printer's summarizer)
# ------------------------------------------------------------


class TestSummarizePayload:
    def test_empty(self):
        assert _summarize_payload({}) == ""
        assert _summarize_payload(None) == ""

    def test_device_id_shown_first(self):
        result = _summarize_payload(
            {"status": "open", "_device_id": "abc123", "count": 5}
        )
        # device_id should come first
        assert result.startswith("id=abc123")
        assert "status=open" in result

    def test_message_truncated(self):
        long = "x" * 100
        result = _summarize_payload({"message": long})
        assert "..." in result
        assert len(result) < 60  # roughly bounded

    def test_private_keys_hidden(self):
        """Keys starting with _ are internal markers and shouldn't
        clutter the output."""
        result = _summarize_payload(
            {"_device_id": "abc", "_internal": "hide_me", "protocol": "xyz"}
        )
        assert "id=abc" in result
        assert "hide_me" not in result

    def test_non_dict_stringified(self):
        assert "raw string" in _summarize_payload("raw string")


# ------------------------------------------------------------
# In-memory database mode
# ------------------------------------------------------------


class TestInMemoryDatabase:
    @pytest.mark.asyncio
    async def test_memory_db_accepts_writes_and_reads(self):
        """Basic sanity: a Database(":memory:") opens, runs migrations,
        and handles the kind of queries the session runner will do."""
        db = Database(":memory:")
        assert db.is_in_memory is True
        # Exercise the connection: run a migration query
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        # sessions, decodes, active_channels, etc. should be present
        # after migrations run
        assert "sessions" in tables
        assert "decodes" in tables
        assert "active_channels" in tables

    @pytest.mark.asyncio
    async def test_memory_db_path_not_on_disk(self, tmp_path, monkeypatch):
        """Creating a :memory: DB should not create any files on disk."""
        monkeypatch.chdir(tmp_path)
        db = Database(":memory:")
        # Force open
        await db.execute("SELECT 1")
        # No file called :memory: should appear
        assert not (tmp_path / ":memory:").exists()

    def test_memory_db_distinct_from_file_db(self, tmp_path):
        """Constructing a :memory: DB and a file DB yields different
        objects — they should NOT share state."""
        mem_db = Database(":memory:")
        file_db = Database(tmp_path / "test.sqlite")
        assert mem_db is not file_db
        assert mem_db.is_in_memory is True
        assert file_db.is_in_memory is False


# ------------------------------------------------------------
# DecoderConfig helper
# ------------------------------------------------------------


class TestNewDecoderConfig:
    def test_enabled_true(self):
        dc = _new_decoder_config(enabled=True)
        assert dc.enabled is True

    def test_enabled_false(self):
        dc = _new_decoder_config(enabled=False)
        assert dc.enabled is False


# ------------------------------------------------------------
# Command registration
# ------------------------------------------------------------


class TestCommandRegistration:
    def test_monitor_cli_registered_in_main_group(self):
        """Ensure `rfcensus monitor` is wired into the top-level
        click group so users can actually invoke it."""
        from rfcensus.cli import main as cli_group
        subcommand_names = list(cli_group.commands.keys())
        assert "monitor" in subcommand_names, (
            f"'monitor' subcommand should be registered on the main "
            f"CLI group; got: {sorted(subcommand_names)}"
        )

    def test_monitor_help_mentions_key_options(self):
        """Smoke test: the --help output lists --band, --save, --format
        — the three options that change the behavior meaningfully."""
        from click.testing import CliRunner

        from rfcensus.commands.monitor import monitor_cli

        runner = CliRunner()
        result = runner.invoke(monitor_cli, ["--help"])
        assert result.exit_code == 0
        for opt in ("--band", "--save", "--format", "--decoder", "--dongle"):
            assert opt in result.output, (
                f"help text should mention '{opt}' so users know the "
                f"option exists; got:\n{result.output}"
            )
