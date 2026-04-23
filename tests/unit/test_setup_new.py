"""Tests for `rfcensus setup new` subcommand.

Covers:
  • CLI structure: setup is now a group with a `new` subcommand
  • setup (no subcommand) still works as before
  • _make_preserved_assignment correctly copies existing config into
    a _DongleAssignment record
  • The only-new filter correctly classifies dongles as known vs new
"""

from __future__ import annotations

import pytest

from click.testing import CliRunner


def _caps():
    from rfcensus.hardware.dongle import DongleCapabilities
    return DongleCapabilities(
        freq_range_hz=(24_000_000, 1_700_000_000),
        max_sample_rate=2_400_000,
        bits_per_sample=8,
        bias_tee_capable=False,
        tcxo_ppm=10.0,
    )


def _dongle(serial: str | None, model: str = "rtlsdr_generic"):
    from rfcensus.hardware.dongle import Dongle, DongleStatus
    return Dongle(
        id=f"rtlsdr-{serial or 'none'}",
        serial=serial,
        model=model,
        driver="rtlsdr",
        capabilities=_caps(),
        status=DongleStatus.HEALTHY,
        driver_index=0,
    )


# ──────────────────────────────────────────────────────────────────
# CLI structure
# ──────────────────────────────────────────────────────────────────


class TestSetupGroupStructure:
    def test_setup_is_a_click_group(self):
        from rfcensus.commands.setup import cli
        import click
        assert isinstance(cli, click.Group)

    def test_new_subcommand_registered(self):
        from rfcensus.commands.setup import cli
        assert "new" in cli.commands

    def test_setup_help_mentions_new(self):
        from rfcensus.commands.setup import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "new" in result.output.lower()
        assert "Walk only dongles" in result.output or "setup new" in result.output

    def test_setup_new_help_explains_purpose(self):
        from rfcensus.commands.setup import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["new", "--help"])
        assert result.exit_code == 0
        assert "preserved" in result.output.lower() or "unchanged" in result.output.lower()


# ──────────────────────────────────────────────────────────────────
# _make_preserved_assignment
# ──────────────────────────────────────────────────────────────────


class TestMakePreservedAssignment:
    def test_copies_antenna_from_existing(self):
        from rfcensus.commands.setup import _make_preserved_assignment
        d = _dongle("00000043")
        existing = {"id": "rtlsdr-00000043", "serial": "00000043", "antenna": "whip_433"}
        a = _make_preserved_assignment(d, existing)
        assert a.dongle is d
        assert a.antenna_id == "whip_433"
        assert a.skip is False

    def test_handles_missing_antenna(self):
        from rfcensus.commands.setup import _make_preserved_assignment
        d = _dongle("00000043")
        existing = {"id": "rtlsdr-00000043", "serial": "00000043"}  # no antenna
        a = _make_preserved_assignment(d, existing)
        assert a.antenna_id is None

    def test_preserves_notes(self):
        from rfcensus.commands.setup import _make_preserved_assignment
        d = _dongle("00000043")
        existing = {
            "id": "rtlsdr-00000043", "serial": "00000043",
            "antenna": "whip_433", "notes": "rooftop",
        }
        a = _make_preserved_assignment(d, existing)
        assert a.notes == "rooftop"


# ──────────────────────────────────────────────────────────────────
# Only-new classification logic
# ──────────────────────────────────────────────────────────────────


class TestOnlyNewClassification:
    """Tests the in-line filter logic (replicated here as a unit-test
    since the actual setup function is interactive and hard to drive
    in a unit test)."""

    def _classify(self, detected, existing_dongles_by_serial, existing_dongles_list):
        """Mirror the classification used by _setup() under only_new=True."""
        already_known = []
        new_dongles = []
        for d in detected:
            is_known = (
                d.serial and d.serial in existing_dongles_by_serial
            ) or any(
                exist.get("id") == d.id for exist in existing_dongles_list
            )
            if is_known:
                already_known.append(d)
            else:
                new_dongles.append(d)
        return already_known, new_dongles

    def test_all_unknown_classified_as_new(self):
        d1 = _dongle("00000001")
        d2 = _dongle("00000043")
        known, new = self._classify([d1, d2], {}, [])
        assert known == []
        assert new == [d1, d2]

    def test_known_by_serial(self):
        d1 = _dongle("00000001")
        d2 = _dongle("00000043")
        existing = {"00000001": {"id": "rtlsdr-00000001", "serial": "00000001"}}
        known, new = self._classify([d1, d2], existing, [{"id": "rtlsdr-00000001", "serial": "00000001"}])
        assert known == [d1]
        assert new == [d2]

    def test_known_by_id_when_no_serial(self):
        # Dongle with no serial — classification falls back to id matching
        d1 = _dongle(None)
        d1.id = "custom_id"
        existing_list = [{"id": "custom_id"}]
        known, new = self._classify([d1], {}, existing_list)
        assert known == [d1]
        assert new == []

    def test_partial_overlap(self):
        # 3 dongles, 1 known, 2 new — common case after plugging in 2 more
        d1 = _dongle("00000001")
        d2 = _dongle("00000043")
        d3 = _dongle("07262454")
        existing_serials = {"00000001": {"id": "rtlsdr-00000001", "serial": "00000001"}}
        existing_list = [{"id": "rtlsdr-00000001", "serial": "00000001"}]
        known, new = self._classify([d1, d2, d3], existing_serials, existing_list)
        assert known == [d1]
        assert new == [d2, d3]


# ──────────────────────────────────────────────────────────────────
# serialize → setup new wiring (next steps suggestion)
# ──────────────────────────────────────────────────────────────────


class TestSerializeNextSteps:
    def test_serialize_module_mentions_setup_new(self):
        """The serialize command's done-message should suggest `setup new`
        as the natural follow-on (when a config exists)."""
        import inspect
        from rfcensus.commands import serialize
        source = inspect.getsource(serialize._run)
        assert "setup new" in source
