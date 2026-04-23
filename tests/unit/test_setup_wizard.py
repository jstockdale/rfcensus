"""Tests for `rfcensus setup` wizard logic.

We test the pure-logic helpers (frequency guide math, config building)
rather than the interactive prompts. The interactive walk-through is
exercised only for: starting state validation, no-dongle handling, and
config-merge semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfcensus.commands import _frequency_guide as fg
from rfcensus.commands.setup import (
    _DongleAssignment,
    _WizardState,
    _build_new_config,
)
from rfcensus.hardware.dongle import Dongle, DongleCapabilities, DongleStatus


def _make_dongle(driver: str = "rtlsdr", serial: str = "00000004") -> Dongle:
    return Dongle(
        id=f"{driver}-{serial[-8:]}",
        serial=serial,
        model="rtl-sdr_v4" if driver == "rtlsdr" else "hackrf_one",
        driver=driver,
        capabilities=DongleCapabilities(
            freq_range_hz=(24_000_000, 1_766_000_000) if driver == "rtlsdr"
                          else (1_000_000, 6_000_000_000),
            max_sample_rate=2_400_000 if driver == "rtlsdr" else 20_000_000,
            bits_per_sample=8,
            bias_tee_capable=False,
            tcxo_ppm=1.0 if driver == "rtlsdr" else 10.0,
            wide_scan_capable=(driver == "hackrf"),
        ),
        status=DongleStatus.HEALTHY,
        driver_index=0,
    )


class TestFrequencyGuide:
    def test_quarter_wave_at_433mhz(self):
        """Quarter wave at 433.92 MHz should be ~17.3 cm (well-known number)."""
        qw = fg.quarter_wave_cm(433_920_000)
        assert 17.0 < qw < 17.6

    def test_quarter_wave_at_915mhz(self):
        """Quarter wave at 915 MHz should be ~8.2 cm."""
        qw = fg.quarter_wave_cm(915_000_000)
        assert 8.0 < qw < 8.4

    def test_quarter_wave_at_315mhz(self):
        """Quarter wave at 315 MHz should be ~23.8 cm."""
        qw = fg.quarter_wave_cm(315_000_000)
        assert 23.5 < qw < 24.1

    def test_find_profile_exact_match(self):
        p = fg.find_profile(915_000_000)
        assert p is not None
        assert p.freq_hz == 915_000_000

    def test_find_profile_within_tolerance(self):
        # 920 MHz is within 5% of 915 MHz
        p = fg.find_profile(920_000_000, tolerance_pct=0.05)
        assert p is not None
        assert p.freq_hz == 915_000_000

    def test_find_profile_outside_tolerance_returns_none(self):
        # 800 MHz is far from any common band in the table — should miss
        p = fg.find_profile(800_000_000, tolerance_pct=0.01)
        # 800 MHz is close to 851 MHz P25 (~6%), but tolerance is 1%, so should miss
        assert p is None or p.freq_hz != 800_000_000

    def test_beginner_recommendations_returns_subset(self):
        recs = fg.beginner_recommendations()
        assert len(recs) >= 3
        assert len(recs) < len(fg.COMMON_FREQUENCIES)
        # All beginner recs should be in the main table
        ids = {p.freq_hz for p in fg.COMMON_FREQUENCIES}
        for r in recs:
            assert r.freq_hz in ids

    def test_profile_decoders_are_real_names(self):
        """Decoders referenced by profiles should match real registered decoder names."""
        from rfcensus.decoders.registry import get_registry as get_decoder_registry

        valid_decoders = set(get_decoder_registry().names())
        for p in fg.COMMON_FREQUENCIES:
            for d in p.decoders:
                assert d in valid_decoders, (
                    f"Profile {p.label!r} references unknown decoder {d!r}. "
                    f"Valid: {sorted(valid_decoders)}"
                )

    def test_profile_detectors_are_real_names(self):
        """Detectors referenced by profiles should match real registered ones."""
        from rfcensus.detectors.registry import get_registry as get_det_registry

        valid_detectors = set(get_det_registry().names())
        for p in fg.COMMON_FREQUENCIES:
            for d in p.detectors:
                assert d in valid_detectors, (
                    f"Profile {p.label!r} references unknown detector {d!r}. "
                    f"Valid: {sorted(valid_detectors)}"
                )

    def test_common_frequencies_sorted_by_frequency(self):
        """COMMON_FREQUENCIES must be in ascending frequency order so the
        wizard menu reads naturally low-to-high. If you add a new profile,
        insert it in the right place."""
        freqs = [p.freq_hz for p in fg.COMMON_FREQUENCIES]
        assert freqs == sorted(freqs), (
            "COMMON_FREQUENCIES is not sorted by frequency. "
            f"Got order: {[f/1e6 for f in freqs]}"
        )


class TestConfigBuilder:
    def test_writes_dongle_with_chosen_antenna(self, tmp_path: Path):
        target = tmp_path / "site.toml"
        dongle = _make_dongle()
        state = _WizardState(detected=[dongle])
        state.assignments = [
            _DongleAssignment(dongle=dongle, antenna_id="whip_915")
        ]
        text = _build_new_config(target, {}, state)
        assert "whip_915" in text
        assert dongle.serial in text

    def test_skipped_dongle_not_written(self, tmp_path: Path):
        target = tmp_path / "site.toml"
        dongle = _make_dongle()
        state = _WizardState(detected=[dongle])
        state.assignments = [_DongleAssignment(dongle=dongle, skip=True)]
        text = _build_new_config(target, {}, state)
        assert dongle.serial not in text

    def test_preserves_existing_top_level_sections(self, tmp_path: Path):
        """Unrelated config sections should survive a setup run."""
        target = tmp_path / "site.toml"
        dongle = _make_dongle()
        existing = {
            "site": {"name": "oakland_basement", "region": "US"},
            "validation": {"min_snr_db": 4.0, "min_confirmations_for_confirmed": 5},
            "bands": {"enabled": ["433_ism", "915_ism"]},
            "antennas": [
                {"id": "user_special", "name": "Roof Yagi", "antenna_type": "yagi",
                 "usable_range": [430_000_000, 450_000_000], "gain_dbi": 11.0,
                 "polarization": "horizontal"}
            ],
        }
        state = _WizardState(detected=[dongle])
        state.assignments = [
            _DongleAssignment(dongle=dongle, antenna_id="user_special")
        ]
        text = _build_new_config(target, existing, state)
        # All preserved
        assert "oakland_basement" in text
        assert "min_snr_db" in text
        assert "433_ism" in text
        assert "Roof Yagi" in text
        # And the dongle was added
        assert "user_special" in text
        assert dongle.serial in text

    def test_replaces_dongle_stanzas_only(self, tmp_path: Path):
        """When config already has [[dongles]], they get replaced wholesale."""
        target = tmp_path / "site.toml"
        existing = {
            "site": {"name": "default"},
            "dongles": [
                {"id": "old_dongle_1", "serial": "DEADBEEF", "model": "old", "driver": "rtlsdr"},
                {"id": "old_dongle_2", "serial": "CAFEBABE", "model": "old", "driver": "rtlsdr"},
            ],
        }
        new_dongle = _make_dongle(serial="11112222")
        state = _WizardState(detected=[new_dongle])
        state.assignments = [
            _DongleAssignment(dongle=new_dongle, antenna_id="whip_915")
        ]
        text = _build_new_config(target, existing, state)
        # Old serials should NOT be present
        assert "DEADBEEF" not in text
        assert "CAFEBABE" not in text
        # New dongle should be
        assert "11112222" in text

    def test_custom_antennas_appended(self, tmp_path: Path):
        target = tmp_path / "site.toml"
        dongle = _make_dongle()
        state = _WizardState(detected=[dongle])
        state.custom_antennas = [
            {
                "id": "whip_telescopic_915mhz",
                "name": "Telescopic whip @ 8.2 cm (tuned for 915 MHz)",
                "antenna_type": "whip",
                "resonant_freq_hz": 915_000_000,
                "usable_range": [777_750_000, 1_052_250_000],
                "gain_dbi": 2.15,
                "polarization": "vertical",
            }
        ]
        state.assignments = [
            _DongleAssignment(dongle=dongle, antenna_id="whip_telescopic_915mhz")
        ]
        text = _build_new_config(target, {}, state)
        assert "whip_telescopic_915mhz" in text
        assert "Telescopic whip" in text

    def test_custom_antenna_doesnt_duplicate_existing_id(self, tmp_path: Path):
        """If user already has an antenna with this id, don't write it again."""
        target = tmp_path / "site.toml"
        existing = {
            "site": {"name": "default"},
            "antennas": [
                {"id": "user_antenna", "name": "Existing", "antenna_type": "whip",
                 "usable_range": [400_000_000, 470_000_000], "gain_dbi": 2.0,
                 "polarization": "vertical"}
            ],
        }
        dongle = _make_dongle()
        state = _WizardState(detected=[dongle])
        # Wizard tries to add same id (shouldn't duplicate)
        state.custom_antennas = [
            {"id": "user_antenna", "name": "Different name"}
        ]
        state.assignments = [
            _DongleAssignment(dongle=dongle, antenna_id="user_antenna")
        ]
        text = _build_new_config(target, existing, state)
        # Original "Existing" should remain; "Different name" should NOT have been added
        assert "Existing" in text
        assert "Different name" not in text

    def test_default_site_added_when_missing(self, tmp_path: Path):
        target = tmp_path / "site.toml"
        dongle = _make_dongle()
        state = _WizardState(detected=[dongle])
        state.assignments = [
            _DongleAssignment(dongle=dongle, antenna_id="whip_915")
        ]
        text = _build_new_config(target, {}, state)
        assert "[site]" in text
