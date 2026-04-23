"""Parser-level tests for decoder output.

Each decoder's subprocess-output parsing is the version-fragile part.
These tests feed real (or realistic) output lines from each tool and
verify the parser produces the expected DecodeEvent.

No subprocess is invoked; we call `_parse_line` / `_parse_nmea` / etc.
directly. This catches parser drift when rtl_433 / rtlamr / multimon
change their output format.

Where possible, sample lines are taken from actual tool documentation
or observed captures.
"""

from __future__ import annotations

from rfcensus.decoders.builtin.direwolf import _parse_direwolf
from rfcensus.decoders.builtin.multimon import _parse_line as parse_multimon
from rfcensus.decoders.builtin.rtl_433 import _parse_line as parse_rtl_433
from rfcensus.decoders.builtin.rtl_ais import _parse_nmea
from rfcensus.decoders.builtin.rtlamr import _parse_line as parse_rtlamr


DEFAULT_KWARGS = dict(
    freq_hz=433_920_000,
    dongle_id="test",
    session_id=1,
    decoder_name="test_decoder",
)


class TestRtl433Parser:
    def test_parses_standard_tpms_frame(self):
        line = (
            '{"time":"2026-04-22T12:30:00","model":"Toyota-TPMS",'
            '"id":"0123ABC","pressure_kPa":220.5,"temperature_C":24,'
            '"mic":"CHECKSUM","freq":433.92,"rssi":-45.2,"snr":18.1}'
        )
        event = parse_rtl_433(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "tpms"
        assert event.payload["_device_id"] == "0123ABC"
        assert event.rssi_dbm == -45.2
        assert event.snr_db == 18.1
        # Frequency should be converted to Hz
        assert event.freq_hz == 433_920_000

    def test_parses_weather_station(self):
        line = (
            '{"time":"2026-04-22T12:30:00","model":"Acurite-Tower",'
            '"id":12345,"temperature_C":21.3,"humidity":45,'
            '"freq":433.92,"rssi":-55.0}'
        )
        event = parse_rtl_433(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "weather_station"
        assert event.payload["_device_id"] == "12345"

    def test_parses_interlogix_security(self):
        line = (
            '{"time":"2026-04-22T12:30:00","model":"Interlogix-Security",'
            '"subtype":"motion","id":"260820","battery_ok":1,'
            '"freq":319.5,"rssi":-50.0}'
        )
        event = parse_rtl_433(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "interlogix_security"
        # Frequency converted
        assert event.freq_hz == 319_500_000

    def test_returns_none_on_non_json_line(self):
        assert parse_rtl_433("not json", **DEFAULT_KWARGS) is None
        assert parse_rtl_433("", **DEFAULT_KWARGS) is None
        # Banner / status lines from rtl_433
        assert parse_rtl_433("rtl_433 version 23.11", **DEFAULT_KWARGS) is None

    def test_falls_back_to_default_freq_when_not_reported(self):
        line = '{"model":"TPMS","id":"abc","rssi":-40}'
        event = parse_rtl_433(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.freq_hz == DEFAULT_KWARGS["freq_hz"]

    def test_handles_freq_already_in_hz(self):
        line = '{"model":"TPMS","id":"abc","freq_hz":433920000}'
        event = parse_rtl_433(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.freq_hz == 433_920_000

    def test_protocol_classification_by_model(self):
        # Unknown model falls back to generic_ook
        line = '{"model":"SomeWeirdDevice","id":"1234"}'
        event = parse_rtl_433(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "generic_ook"


class TestRtlamrParser:
    def test_parses_scm_message(self):
        line = (
            '{"Time":"2026-04-22T12:30:00.0Z","Type":"SCM",'
            '"Message":{"ID":12345678,"Type":4,"TamperPhy":0,"TamperEnc":0,'
            '"Consumption":100000,"ChecksumVal":1234}}'
        )
        kwargs = {**DEFAULT_KWARGS, "freq_hz": 912_000_000}
        event = parse_rtlamr(line, **kwargs)
        assert event is not None
        assert event.protocol == "ert_scm"
        assert event.payload["_device_id"] == "12345678"

    def test_parses_idm_message(self):
        line = (
            '{"Time":"2026-04-22T12:30:00Z","Type":"IDM",'
            '"Message":{"ERTSerialNumber":98765432,"ConsumptionIntervalCount":42,'
            '"LastConsumptionCount":635437056,'
            '"DifferentialConsumptionIntervals":[430,0,32]}}'
        )
        event = parse_rtlamr(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "ert_idm"
        assert event.payload["_device_id"] == "98765432"

    def test_parses_r900(self):
        line = (
            '{"Time":"2026-04-22T12:30:00Z","Type":"R900",'
            '"Message":{"ID":777888,"Consumption":55000}}'
        )
        event = parse_rtlamr(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "r900"
        assert event.payload["_device_id"] == "777888"

    def test_scm_plus_variant(self):
        line = (
            '{"Time":"2026-04-22T12:30:00Z","Type":"SCM+",'
            '"Message":{"EndpointID":5555,"EndpointType":156,"Consumption":20000}}'
        )
        event = parse_rtlamr(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "ert_scm_plus"
        assert event.payload["_device_id"] == "5555"

    def test_returns_none_on_empty_line(self):
        assert parse_rtlamr("", **DEFAULT_KWARGS) is None

    def test_returns_none_on_banner(self):
        assert parse_rtlamr("rtlamr v0.9", **DEFAULT_KWARGS) is None


class TestRtlAisParser:
    def test_parses_class_a_position_report(self):
        # A real AIS message type 1 (class A position)
        line = "!AIVDM,1,1,,A,13aG?P0P00PD;88MD5MTDww<0<07,0*4A"
        event = _parse_nmea(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "ais_class_a"
        # Should have extracted MMSI
        assert event.payload["mmsi"] != 0
        assert "_device_id" in event.payload

    def test_parses_class_b_position_report(self):
        # Type 18 = Class B position report
        # Real sample with valid checksum
        line = "!AIVDM,1,1,,B,B5NJ;PP005l4ot5Isbl03wsUkP06,0*76"
        event = _parse_nmea(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "ais_class_b"

    def test_returns_none_on_non_ais_line(self):
        assert _parse_nmea("not a sentence", **DEFAULT_KWARGS) is None
        assert _parse_nmea("", **DEFAULT_KWARGS) is None
        # Wrong talker ID
        assert _parse_nmea("$GPGGA,,,,,,0,,,,,,,,,*66", **DEFAULT_KWARGS) is None


class TestMultimonParser:
    def test_parses_pocsag_with_alpha_message(self):
        line = "POCSAG1200: Address: 1234567  Function: 1  Alpha:   HELLO WORLD"
        event = parse_multimon(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "pocsag"
        assert event.payload["address"] == "1234567"
        assert event.payload["baud"] == 1200
        assert "HELLO" in event.payload["message"]
        assert event.payload["_device_id"] == "1234567"

    def test_parses_pocsag_numeric(self):
        line = "POCSAG512: Address: 7654321  Function: 0  Numeric:  1234"
        event = parse_multimon(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "pocsag"
        assert event.payload["baud"] == 512

    def test_parses_flex_message(self):
        line = "FLEX|2021-04-22 12:00:00|1600/2/C|007.004.054|0000001|ALN|text here"
        event = parse_multimon(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "flex"
        assert event.payload["capcode"] == "0000001"

    def test_parses_aprs_line(self):
        line = "APRS: K6ABC-9>APDR16,WIDE1-1:!3748.00N/12226.00W>comment"
        event = parse_multimon(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "aprs"
        assert event.payload["callsign"] == "K6ABC-9"

    def test_parses_dtmf(self):
        line = "DTMF: 5"
        event = parse_multimon(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "dtmf"
        assert event.payload["digit"] == "5"

    def test_ignores_garbage(self):
        assert parse_multimon("random text", **DEFAULT_KWARGS) is None
        assert parse_multimon("", **DEFAULT_KWARGS) is None
        assert parse_multimon("Unknown format here", **DEFAULT_KWARGS) is None


class TestDirewolfParser:
    def test_parses_aprs_frame(self):
        line = "[0] K6ABC-9>APDR16,WIDE1-1:!3748.00N/12226.00W>walking"
        event = _parse_direwolf(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.protocol == "aprs"
        assert event.payload["source"] == "K6ABC-9"
        assert event.payload["destination"] == "APDR16"

    def test_parses_frame_with_no_path(self):
        line = "[0] KJ6ABC>APRS:=3748.00N/12226.00W-testing"
        event = _parse_direwolf(line, **DEFAULT_KWARGS)
        assert event is not None
        assert event.payload["source"] == "KJ6ABC"
        assert event.payload["path"] == ""

    def test_ignores_direwolf_status_lines(self):
        assert _parse_direwolf("Dire Wolf version 1.7", **DEFAULT_KWARGS) is None
        assert _parse_direwolf("", **DEFAULT_KWARGS) is None
        assert _parse_direwolf("Reading from stdin", **DEFAULT_KWARGS) is None
