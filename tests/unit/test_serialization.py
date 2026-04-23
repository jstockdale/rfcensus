"""Tests for RTL-SDR serial reserialization.

Focused on the pure planning logic — the algorithm that decides which
dongles need new serials and what those serials should be. The IO
helpers (backup_eeprom, write_serial, verify_serial) are thin wrappers
around subprocess calls and are tested only at the smoke-test level.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfcensus.commands.serialize import _update_site_toml
from rfcensus.hardware.dongle import Dongle, DongleCapabilities, DongleStatus
from rfcensus.hardware.serialization import (
    SerialAssignment,
    _format_serial,
    plan_reserialization,
)


def _caps() -> DongleCapabilities:
    return DongleCapabilities(
        freq_range_hz=(24_000_000, 1_700_000_000),
        max_sample_rate=2_400_000,
        bits_per_sample=8,
        bias_tee_capable=False,
        tcxo_ppm=10.0,
    )


def _dongle(idx: int, serial: str | None, model: str = "rtlsdr_generic", driver: str = "rtlsdr") -> Dongle:
    return Dongle(
        id=f"{driver}-{serial or 'none'}-idx{idx}",
        serial=serial,
        model=model,
        driver=driver,
        capabilities=_caps(),
        status=DongleStatus.DETECTED,
        driver_index=idx,
    )


# ──────────────────────────────────────────────────────────────────
# Format
# ──────────────────────────────────────────────────────────────────


class TestFormatSerial:
    def test_zero_padded_to_8_digits(self):
        assert _format_serial(1) == "00000001"
        assert _format_serial(42) == "00000042"
        assert _format_serial(99999999) == "99999999"

    def test_zero(self):
        assert _format_serial(0) == "00000000"


# ──────────────────────────────────────────────────────────────────
# Planning: unique serials → empty plan
# ──────────────────────────────────────────────────────────────────


class TestPlanReserialization:
    def test_no_dongles_empty_plan(self):
        plan = plan_reserialization([])
        assert plan.is_empty
        assert plan.assignments == ()

    def test_single_dongle_empty_plan(self):
        plan = plan_reserialization([_dongle(0, "00000001")])
        assert plan.is_empty

    def test_unique_serials_empty_plan(self):
        plan = plan_reserialization([
            _dongle(0, "00000001"),
            _dongle(1, "00000002"),
            _dongle(2, "07262454"),
        ])
        assert plan.is_empty

    def test_hackrf_excluded_from_planning(self):
        """HackRFs have factory-burned 128-bit serials; we never reserialize them."""
        plan = plan_reserialization([
            _dongle(0, "abc123def456", driver="hackrf"),
            _dongle(1, "abc123def456", driver="hackrf"),
        ])
        # Even with "duplicates," HackRF dongles are filtered out
        assert plan.is_empty

    # ──────────────────────────────────────────────────────────────────
    # Worked examples from the design discussion
    # ──────────────────────────────────────────────────────────────────

    def test_three_factory_dongles_at_00000001(self):
        """The classic case: three boards, all serial 00000001."""
        plan = plan_reserialization([
            _dongle(0, "00000001"),
            _dongle(1, "00000001"),
            _dongle(2, "00000001"),
        ])
        # Three assignments: one keeps 01, others get 02 and 03
        assert len(plan.assignments) == 3
        assert plan.assignments[0].new_serial == "00000001"
        assert plan.assignments[0].keeps_original is True
        assert plan.assignments[0].driver_index == 0  # lowest keeps it

        new_serials = [a.new_serial for a in plan.assignments[1:]]
        assert new_serials == ["00000002", "00000003"]
        assert all(not a.keeps_original for a in plan.assignments[1:])

    def test_three_dongles_at_user_chosen_42(self):
        """User previously set all three to 42; we should preserve locality."""
        plan = plan_reserialization([
            _dongle(0, "00000042"),
            _dongle(1, "00000042"),
            _dongle(2, "00000042"),
        ])
        assigned = [a.new_serial for a in plan.assignments]
        assert assigned == ["00000042", "00000043", "00000044"]

    def test_user_42_with_43_and_44_already_taken(self):
        """If 43 and 44 are already used by other attached dongles,
        the duplicates of 42 should skip those values."""
        plan = plan_reserialization([
            _dongle(0, "00000042"),
            _dongle(1, "00000042"),
            _dongle(2, "00000043"),  # already taken (different dongle)
            _dongle(3, "00000044"),  # already taken
        ])
        # Find the assignments for the 42-group only (not the unique 43/44)
        forty_two_assignments = [a for a in plan.assignments if a.original_serial == "00000042"]
        assert len(forty_two_assignments) == 2
        assert forty_two_assignments[0].new_serial == "00000042"
        assert forty_two_assignments[0].keeps_original
        # Duplicate of 42 should jump to 45 (43 and 44 forbidden)
        assert forty_two_assignments[1].new_serial == "00000045"

    def test_partial_collision(self):
        """3× 00000001 + standalone unique dongle with 00000005."""
        plan = plan_reserialization([
            _dongle(0, "00000001"),
            _dongle(1, "00000001"),
            _dongle(2, "00000001"),
            _dongle(3, "00000005"),
        ])
        # Only the duplicates appear in the plan; the unique 00000005 is skipped
        assert len(plan.assignments) == 3
        new_serials = [a.new_serial for a in plan.assignments]
        # 01 (keep), 02, 03 — 05 is unique so it's not in the plan but is forbidden
        assert new_serials == ["00000001", "00000002", "00000003"]

    def test_two_separate_collision_groups(self):
        """3× 01 and 2× 05 — both groups need processing."""
        plan = plan_reserialization([
            _dongle(0, "00000001"),
            _dongle(1, "00000001"),
            _dongle(2, "00000001"),
            _dongle(3, "00000005"),
            _dongle(4, "00000005"),
        ])
        assert len(plan.assignments) == 5

        ones = [a for a in plan.assignments if a.original_serial == "00000001"]
        fives = [a for a in plan.assignments if a.original_serial == "00000005"]
        assert [a.new_serial for a in ones] == ["00000001", "00000002", "00000003"]
        # Group of 5 starts processing AFTER the 1-group has assigned 02 and 03,
        # so those values are now forbidden. 5 keeps, then needs next from 6.
        assert fives[0].new_serial == "00000005"
        assert fives[1].new_serial == "00000006"

    def test_existing_config_serials_are_protected(self):
        """If user's site.toml references an offline dongle's serial,
        we must not reuse that value for a renamed dongle."""
        plan = plan_reserialization(
            [
                _dongle(0, "00000001"),
                _dongle(1, "00000001"),
            ],
            existing_config_serials=frozenset({"00000002", "00000003"}),
        )
        # 01 keeps 01; 01-duplicate must skip past 02 and 03 → 04
        assert len(plan.assignments) == 2
        assert plan.assignments[0].new_serial == "00000001"
        assert plan.assignments[1].new_serial == "00000004"

    def test_lowest_driver_index_keeps_original(self):
        """The dongle that keeps the original serial must be the one with
        lowest driver_index — for determinism."""
        # Pass dongles in non-index order to verify the sort
        plan = plan_reserialization([
            _dongle(2, "00000001"),  # idx 2, listed first
            _dongle(0, "00000001"),  # idx 0
            _dongle(1, "00000001"),  # idx 1
        ])
        # Lowest index (0) should be the keeper
        keeper = next(a for a in plan.assignments if a.keeps_original)
        assert keeper.driver_index == 0

    def test_dongles_without_serial_skipped(self):
        plan = plan_reserialization([
            _dongle(0, None),
            _dongle(1, "00000001"),
            _dongle(2, "00000001"),
        ])
        # Should still find the 1-collision
        assert len(plan.assignments) == 2
        assert all(a.original_serial == "00000001" for a in plan.assignments)

    def test_non_numeric_serial_falls_back_gracefully(self):
        """Some dongles ship with hex or alphanumeric serials. We default
        to numeric base 1 if we can't parse the original."""
        plan = plan_reserialization([
            _dongle(0, "ABCDEF12"),
            _dongle(1, "ABCDEF12"),
        ])
        # First keeps the weird serial; second gets 00000002 (1 is taken because
        # we start from 1+1, but no — base falls back to 1, then candidate 2)
        assert plan.assignments[0].new_serial == "ABCDEF12"
        assert plan.assignments[1].new_serial == "00000002"


# ──────────────────────────────────────────────────────────────────
# Config rewrite
# ──────────────────────────────────────────────────────────────────


class TestUpdateSiteToml:
    def _write_config(self, path: Path, dongles: list[dict]) -> None:
        """Write a minimal site.toml with the given dongle stanzas."""
        import tomli_w
        path.write_text(tomli_w.dumps({
            "site": {"name": "test"},
            "dongles": dongles,
        }))

    def _read_dongles(self, path: Path) -> list[dict]:
        import tomllib
        data = tomllib.loads(path.read_text())
        return data.get("dongles", [])

    def test_renames_serial_in_existing_stanza(self, tmp_path: Path):
        target = tmp_path / "site.toml"
        self._write_config(target, [
            {"id": "rtlsdr-00000001", "serial": "00000001",
             "model": "rtlsdr_generic", "driver": "rtlsdr",
             "antenna": "whip_915"},
        ])
        from rfcensus.hardware.serialization import ReserializationPlan
        plan = ReserializationPlan(assignments=(
            SerialAssignment(
                driver_index=1, original_serial="00000001",
                new_serial="00000002", model="rtlsdr_generic",
            ),
        ))
        n = _update_site_toml(target, plan)
        assert n == 1

        dongles = self._read_dongles(target)
        assert len(dongles) == 1
        assert dongles[0]["serial"] == "00000002"
        assert dongles[0]["id"] == "rtlsdr-00000002"
        assert dongles[0]["antenna"] == "whip_915"  # preserved!

    def test_does_not_touch_unrelated_stanzas(self, tmp_path: Path):
        target = tmp_path / "site.toml"
        self._write_config(target, [
            {"id": "rtlsdr-00000001", "serial": "00000001",
             "model": "rtlsdr_generic", "driver": "rtlsdr",
             "antenna": "whip_915"},
            {"id": "rtlsdr-99999999", "serial": "99999999",
             "model": "rtlsdr_v4", "driver": "rtlsdr",
             "antenna": "discone"},
        ])
        from rfcensus.hardware.serialization import ReserializationPlan
        plan = ReserializationPlan(assignments=(
            SerialAssignment(
                driver_index=1, original_serial="00000001",
                new_serial="00000002", model="rtlsdr_generic",
            ),
        ))
        _update_site_toml(target, plan)
        dongles = self._read_dongles(target)
        assert len(dongles) == 2
        # Find the unchanged stanza
        unchanged = next(d for d in dongles if d.get("model") == "rtlsdr_v4")
        assert unchanged["serial"] == "99999999"
        assert unchanged["antenna"] == "discone"

    def test_keeps_original_id_if_user_customized(self, tmp_path: Path):
        """If the user used a custom id like 'roof_dongle', we shouldn't
        rename it to 'rtlsdr-NEW' — they had a reason for the custom name."""
        target = tmp_path / "site.toml"
        self._write_config(target, [
            {"id": "roof_dongle", "serial": "00000001",
             "model": "rtlsdr_generic", "driver": "rtlsdr",
             "antenna": "whip_915"},
        ])
        from rfcensus.hardware.serialization import ReserializationPlan
        plan = ReserializationPlan(assignments=(
            SerialAssignment(
                driver_index=1, original_serial="00000001",
                new_serial="00000002", model="rtlsdr_generic",
            ),
        ))
        _update_site_toml(target, plan)
        dongles = self._read_dongles(target)
        assert dongles[0]["serial"] == "00000002"
        assert dongles[0]["id"] == "roof_dongle"  # preserved
        assert dongles[0]["antenna"] == "whip_915"

    def test_preserves_other_top_level_sections(self, tmp_path: Path):
        """The site, validation, bands sections must survive a serialize run."""
        import tomli_w
        target = tmp_path / "site.toml"
        target.write_text(tomli_w.dumps({
            "site": {"name": "oakland_basement", "region": "US"},
            "validation": {"min_snr_db": 4.0},
            "bands": {"enabled": ["433_ism"]},
            "dongles": [
                {"id": "rtlsdr-00000001", "serial": "00000001",
                 "model": "rtlsdr_generic", "driver": "rtlsdr"},
            ],
        }))
        from rfcensus.hardware.serialization import ReserializationPlan
        plan = ReserializationPlan(assignments=(
            SerialAssignment(
                driver_index=1, original_serial="00000001",
                new_serial="00000002", model="rtlsdr_generic",
            ),
        ))
        _update_site_toml(target, plan)
        text = target.read_text()
        assert "oakland_basement" in text
        assert "min_snr_db" in text
        assert "433_ism" in text
        assert "00000002" in text


# ──────────────────────────────────────────────────────────────────
# v0.5.3: keeper_overrides and existing_config preference
# ──────────────────────────────────────────────────────────────────


class TestKeeperSelection:
    """Tests for the v0.5.3 priority rules on which dongle keeps the
    original serial when there's a collision:
      1. Explicit user override (keeper_overrides[serial])
      2. Existing config model match (your V4 case)
      3. Lowest driver_index (deterministic fallback)
    """

    def test_explicit_override_wins(self):
        # 3 dongles all at 00000001, override says idx=2 keeps it
        dongles = [
            _dongle(0, "00000001"),
            _dongle(1, "00000001"),
            _dongle(2, "00000001", model="rtlsdr_v4"),
        ]
        plan = plan_reserialization(
            dongles,
            keeper_overrides={"00000001": 2},
        )
        keeper = next(a for a in plan.assignments if a.keeps_original)
        assert keeper.driver_index == 2
        assert keeper.model == "rtlsdr_v4"

    def test_existing_config_model_match_used_as_default(self):
        """Your V4 case: existing config has 00000001 → rtlsdr_v4. Even
        though the V4 is at idx=2 (not lowest), it should be the keeper."""
        dongles = [
            _dongle(0, "00000001", model="rtlsdr_generic"),
            _dongle(1, "00000001", model="rtlsdr_generic"),
            _dongle(2, "00000001", model="rtlsdr_v4"),
        ]
        plan = plan_reserialization(
            dongles,
            existing_config={"00000001": {"model": "rtlsdr_v4", "antenna": "whip_915"}},
        )
        keeper = next(a for a in plan.assignments if a.keeps_original)
        assert keeper.driver_index == 2
        assert keeper.model == "rtlsdr_v4"

    def test_explicit_override_beats_config_match(self):
        """If user provides an explicit override, it wins even over a
        model match in existing config."""
        dongles = [
            _dongle(0, "00000001", model="rtlsdr_generic"),
            _dongle(1, "00000001", model="rtlsdr_v4"),
        ]
        plan = plan_reserialization(
            dongles,
            keeper_overrides={"00000001": 0},  # generic at idx=0
            existing_config={"00000001": {"model": "rtlsdr_v4"}},
        )
        keeper = next(a for a in plan.assignments if a.keeps_original)
        assert keeper.driver_index == 0
        assert keeper.model == "rtlsdr_generic"

    def test_falls_back_to_lowest_index_with_no_hints(self):
        """No override, no model match — lowest driver_index wins
        (preserves prior behavior)."""
        dongles = [
            _dongle(0, "00000001"),
            _dongle(1, "00000001"),
            _dongle(2, "00000001"),
        ]
        plan = plan_reserialization(dongles)
        keeper = next(a for a in plan.assignments if a.keeps_original)
        assert keeper.driver_index == 0

    def test_invalid_override_falls_back_to_default(self):
        """If override targets a driver_index not in the group, fall
        back to the standard heuristic instead of crashing."""
        dongles = [
            _dongle(0, "00000001"),
            _dongle(1, "00000001"),
        ]
        # idx=99 doesn't exist in the group
        plan = plan_reserialization(
            dongles,
            keeper_overrides={"00000001": 99},
        )
        keeper = next(a for a in plan.assignments if a.keeps_original)
        # Falls back to lowest driver_index
        assert keeper.driver_index == 0


class TestDescribeDongleForPicker:
    def test_includes_model_and_index(self):
        from rfcensus.hardware.serialization import describe_dongle_for_picker
        d = _dongle(2, "00000001", model="rtlsdr_v4")
        s = describe_dongle_for_picker(d, existing_config={})
        assert "rtlsdr_v4" in s
        assert "idx=2" in s

    def test_marks_existing_config_match(self):
        from rfcensus.hardware.serialization import describe_dongle_for_picker
        d = _dongle(2, "00000001", model="rtlsdr_v4")
        existing = {"00000001": {"model": "rtlsdr_v4", "antenna": "whip_915"}}
        s = describe_dongle_for_picker(d, existing)
        assert "in config" in s
        assert "whip_915" in s
        assert "✓" in s  # model-match check mark

    def test_no_marker_when_models_differ(self):
        from rfcensus.hardware.serialization import describe_dongle_for_picker
        d = _dongle(2, "00000001", model="rtlsdr_generic")
        existing = {"00000001": {"model": "rtlsdr_v4", "antenna": "whip_915"}}
        s = describe_dongle_for_picker(d, existing)
        # Stanza is shown but without check mark
        assert "in config" in s
        assert "✓" not in s
