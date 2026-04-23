"""Config loading and antenna matching tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rfcensus.config.loader import load_config, write_default_site_config
from rfcensus.config.schema import AntennaConfig, BandConfig
from rfcensus.hardware.antenna import Antenna, AntennaMatcher


class TestConfigLoader:
    def test_loads_defaults_when_no_user_config(self, tmp_path: Path):
        config = load_config(path=tmp_path / "nonexistent.toml")
        assert config.site.region == "US"
        # We ship a bunch of bands
        assert len(config.band_definitions) > 10
        # And antennas
        assert len(config.antennas) > 5

    def test_user_band_overrides_builtin(self, tmp_path: Path):
        user_cfg = tmp_path / "site.toml"
        user_cfg.write_text(
            """
[site]
name = "test"

[[band_definitions]]
id = "433_ism"
name = "Custom 433 band"
freq_low = 433000000
freq_high = 434500000
strategy = "decoder_only"
"""
        )
        config = load_config(path=user_cfg)
        custom = config.find_band("433_ism")
        assert custom is not None
        assert custom.name == "Custom 433 band"
        assert custom.freq_low == 433_000_000

    def test_enabled_bands_respects_enabled_list(self, tmp_path: Path):
        user_cfg = tmp_path / "site.toml"
        user_cfg.write_text(
            """
[site]
name = "test"

[bands]
enabled = ["433_ism", "915_ism"]
"""
        )
        config = load_config(path=user_cfg)
        enabled_ids = {b.id for b in config.enabled_bands()}
        assert enabled_ids == {"433_ism", "915_ism"}

    def test_enabled_bands_skips_opt_in_by_default(self, tmp_path: Path):
        config = load_config(path=tmp_path / "nonexistent.toml")
        enabled_ids = {b.id for b in config.enabled_bands()}
        # ADS-B is opt-in and shouldn't be enabled by default
        assert "adsb" not in enabled_ids
        # But regular bands should be
        assert "433_ism" in enabled_ids

    def test_write_default_site_config(self, tmp_path: Path):
        target = tmp_path / "config" / "site.toml"
        written = write_default_site_config(target)
        assert written.exists()
        assert "rfcensus site configuration" in written.read_text()

    def test_write_refuses_overwrite_by_default(self, tmp_path: Path):
        target = tmp_path / "site.toml"
        write_default_site_config(target)
        with pytest.raises(Exception):
            write_default_site_config(target)

    def test_write_with_overwrite_flag(self, tmp_path: Path):
        target = tmp_path / "site.toml"
        write_default_site_config(target)
        # Should not raise
        write_default_site_config(target, overwrite=True)


class TestAntennaMatching:
    def test_resonant_antenna_scores_at_resonance(self):
        whip = Antenna.from_config(AntennaConfig(
            id="whip_433", name="test",
            resonant_freq_hz=433_920_000,
            usable_range=(400_000_000, 470_000_000),
        ))
        assert whip.suitability(433_920_000) == pytest.approx(1.0)

    def test_resonant_antenna_scores_lower_far_from_resonance(self):
        whip = Antenna.from_config(AntennaConfig(
            id="whip_433", name="test",
            resonant_freq_hz=433_920_000,
            usable_range=(400_000_000, 470_000_000),
        ))
        assert whip.suitability(460_000_000) < whip.suitability(433_920_000)

    def test_antenna_scores_zero_out_of_range(self):
        whip = Antenna.from_config(AntennaConfig(
            id="whip", name="test",
            resonant_freq_hz=433_920_000,
            usable_range=(400_000_000, 470_000_000),
        ))
        assert whip.suitability(100_000_000) == 0.0
        assert whip.suitability(1_500_000_000) == 0.0

    def test_wideband_antenna_covers_wide_range(self):
        discone = Antenna.from_config(AntennaConfig(
            id="discone", name="test",
            usable_range=(25_000_000, 1_300_000_000),
        ))
        assert discone.covers(100_000_000)
        assert discone.covers(500_000_000)
        assert discone.covers(1_200_000_000)
        assert discone.suitability(500_000_000) > 0.3

    def test_matcher_picks_best_antenna(self):
        good = Antenna.from_config(AntennaConfig(
            id="good", name="test",
            resonant_freq_hz=433_920_000,
            usable_range=(400_000_000, 470_000_000),
        ))
        worse = Antenna.from_config(AntennaConfig(
            id="worse", name="test",
            usable_range=(25_000_000, 1_300_000_000),
        ))
        band = BandConfig(
            id="433_ism",
            name="433",
            freq_low=433_050_000,
            freq_high=434_790_000,
        )
        matcher = AntennaMatcher()
        match = matcher.best_pairing(
            band,
            [("d1", good), ("d2", worse)],
        )
        assert match is not None
        assert match.dongle_id == "d1"

    def test_matcher_returns_none_when_nothing_fits(self):
        band = BandConfig(
            id="x", name="x",
            freq_low=1_500_000_000, freq_high=1_600_000_000,
        )
        tiny = Antenna.from_config(AntennaConfig(
            id="tiny", name="test",
            usable_range=(100_000_000, 200_000_000),
        ))
        matcher = AntennaMatcher(threshold=0.3)
        match = matcher.best_pairing(band, [("d1", tiny)])
        assert match is None
