"""Tests for duplicate-serial detection and disambiguation.

Cheap RTL-SDR boards often ship with serial '00000001' from the
factory. If a user has multiple, we need to:
  1. Give each a unique id by appending the driver index
  2. Surface a clear warning telling the user to fix it permanently
  3. Make `by_serial()` honest about ambiguity
  4. Let `apply_config` give a useful error when the user references
     just the duplicated serial without an id qualifier
"""

from __future__ import annotations

import pytest

from rfcensus.config.schema import DongleConfig, SiteConfig
from rfcensus.hardware.dongle import Dongle, DongleCapabilities, DongleStatus
from rfcensus.hardware.drivers.rtlsdr import _disambiguate_duplicate_ids
from rfcensus.hardware.registry import HardwareRegistry, _duplicate_serial_diagnostics


def _caps() -> DongleCapabilities:
    return DongleCapabilities(
        freq_range_hz=(24_000_000, 1_700_000_000),
        max_sample_rate=2_400_000,
        bits_per_sample=8,
        bias_tee_capable=False,
        tcxo_ppm=10.0,
    )


def _dongle(idx: int, serial: str | None = "00000001", model: str = "rtlsdr_generic") -> Dongle:
    sid = f"rtlsdr-{serial}" if serial else f"rtlsdr-idx{idx}"
    return Dongle(
        id=sid, serial=serial, model=model, driver="rtlsdr",
        capabilities=_caps(), status=DongleStatus.DETECTED, driver_index=idx,
    )


class TestDisambiguation:
    def test_unique_serials_unchanged(self):
        d0 = _dongle(0, "00000001")
        d1 = _dongle(1, "00000043")
        result = _disambiguate_duplicate_ids([d0, d1])
        assert result[0].id == "rtlsdr-00000001"
        assert result[1].id == "rtlsdr-00000043"

    def test_duplicate_serials_get_index_suffix(self):
        d0 = _dongle(0, "00000001")
        d1 = _dongle(1, "00000001")
        result = _disambiguate_duplicate_ids([d0, d1])
        assert result[0].id == "rtlsdr-00000001-idx0"
        assert result[1].id == "rtlsdr-00000001-idx1"

    def test_three_duplicates_all_get_suffixed(self):
        dongles = [_dongle(0), _dongle(1), _dongle(2)]
        result = _disambiguate_duplicate_ids(dongles)
        ids = [d.id for d in result]
        # All ids must be unique
        assert len(set(ids)) == 3
        # All should have the suffix
        assert all("idx" in i for i in ids)

    def test_partial_duplicates(self):
        """Some dongles share a serial, others are unique. Only the
        duplicated ones get suffixed."""
        d0 = _dongle(0, "00000001")
        d1 = _dongle(1, "00000001")
        d2 = _dongle(2, "07262454")
        result = _disambiguate_duplicate_ids([d0, d1, d2])
        assert result[0].id == "rtlsdr-00000001-idx0"
        assert result[1].id == "rtlsdr-00000001-idx1"
        assert result[2].id == "rtlsdr-07262454"  # untouched


class TestDuplicateDiagnostics:
    def test_no_duplicates_no_warning(self):
        dongles = [_dongle(0, "A"), _dongle(1, "B"), _dongle(2, "C")]
        warnings = _duplicate_serial_diagnostics(dongles)
        assert warnings == []

    def test_duplicate_yields_warning(self):
        dongles = [
            _dongle(0, "00000001"),
            _dongle(1, "00000001"),
        ]
        warnings = _duplicate_serial_diagnostics(dongles)
        assert len(warnings) == 1
        w = warnings[0]
        assert "00000001" in w
        assert "rtl_eeprom" in w
        assert w.startswith("⚠")

    def test_warning_lists_all_duplicate_ids(self):
        dongles = [
            _dongle(0, "00000001"),
            _dongle(1, "00000001"),
        ]
        # Run through disambiguator first like the real probe does
        dongles = _disambiguate_duplicate_ids(dongles)
        warnings = _duplicate_serial_diagnostics(dongles)
        assert "rtlsdr-00000001-idx0" in warnings[0]
        assert "rtlsdr-00000001-idx1" in warnings[0]

    def test_dongles_with_no_serial_dont_collide(self):
        """Dongles with serial=None shouldn't be reported as duplicates."""
        d0 = _dongle(0, serial=None)
        d1 = _dongle(1, serial=None)
        warnings = _duplicate_serial_diagnostics([d0, d1])
        assert warnings == []


class TestRegistryByserial:
    def test_unique_serial_returns_dongle(self):
        d = _dongle(0, "uniq")
        reg = HardwareRegistry(dongles=[d])
        assert reg.by_serial("uniq") is d

    def test_duplicate_serial_returns_None(self):
        """When ambiguous, by_serial refuses to guess. Caller must use ids."""
        reg = HardwareRegistry(dongles=[_dongle(0, "dup"), _dongle(1, "dup")])
        assert reg.by_serial("dup") is None

    def test_all_by_serial_returns_all_matches(self):
        reg = HardwareRegistry(dongles=[_dongle(0, "dup"), _dongle(1, "dup")])
        matches = reg.all_by_serial("dup")
        assert len(matches) == 2


class TestApplyConfigWithDuplicates:
    def test_user_can_reference_disambiguated_id(self):
        """If user writes id='rtlsdr-00000001-idx0' in config, it should match."""
        d0 = _dongle(0, "00000001")
        d1 = _dongle(1, "00000001")
        d0.id = "rtlsdr-00000001-idx0"
        d1.id = "rtlsdr-00000001-idx1"
        reg = HardwareRegistry(dongles=[d0, d1])
        config = SiteConfig(
            dongles=[
                DongleConfig(id="rtlsdr-00000001-idx0", serial="00000001",
                             model="rtlsdr_generic", driver="rtlsdr",
                             antenna="whip_915"),
            ]
        )
        warnings = reg.apply_config(config)
        # Should have matched cleanly (the antenna ref will warn since
        # we didn't define one in this minimal config — that's fine)
        assert not any("ambiguous" in w for w in warnings)
        assert not any("not detected" in w for w in warnings)

    def test_ambiguous_serial_reference_yields_clear_warning(self):
        """User wrote just serial='00000001' but two dongles match.
        Warning should explain the situation and suggest using ids."""
        d0 = _dongle(0, "00000001")
        d1 = _dongle(1, "00000001")
        d0.id = "rtlsdr-00000001-idx0"
        d1.id = "rtlsdr-00000001-idx1"
        reg = HardwareRegistry(dongles=[d0, d1])
        config = SiteConfig(
            dongles=[
                DongleConfig(id="some-user-name", serial="00000001",
                             model="rtlsdr_generic", driver="rtlsdr"),
            ]
        )
        warnings = reg.apply_config(config)
        assert any("ambiguous" in w for w in warnings)
        # Warning should list the candidate ids
        assert any("rtlsdr-00000001-idx0" in w for w in warnings)
        assert any("rtlsdr-00000001-idx1" in w for w in warnings)
