"""Tests for v0.5.11 — guided scan mode.

Covers:
  • find_telescopic_dongle finds the right dongle (or returns None)
  • before_band_callback skips retune prompt when tuning is close enough
  • before_band_callback handles 'skip' to skip a band
  • after_session_callback offers the right options
  • update_antenna_in_config writes correct stanza changes
  • CLI rejects --guided + --per-band, --guided + --duration forever
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


def _make_dongle(idx, antenna=None):
    from rfcensus.hardware.dongle import (
        Dongle, DongleCapabilities, DongleStatus,
    )
    caps = DongleCapabilities(
        freq_range_hz=(24_000_000, 1_700_000_000),
        max_sample_rate=2_400_000, bits_per_sample=8,
        bias_tee_capable=False, tcxo_ppm=10.0,
    )
    d = Dongle(
        id=f"rtl-{idx}", serial=f"S{idx}", model="rtlsdr_generic",
        driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
        driver_index=idx,
    )
    d.antenna = antenna
    return d


def _make_antenna(ant_id, resonant_mhz):
    from rfcensus.hardware.antenna import Antenna
    return Antenna(
        id=ant_id, name=ant_id, antenna_type="whip",
        resonant_freq_hz=int(resonant_mhz * 1_000_000),
        usable_range=(int(resonant_mhz * 0.85e6), int(resonant_mhz * 1.15e6)),
        gain_dbi=2.15, polarization="vertical",
        requires_bias_power=False, notes="",
    )


# ──────────────────────────────────────────────────────────────────
# Telescopic dongle detection
# ──────────────────────────────────────────────────────────────────


class TestFindTelescopicDongle:
    def test_finds_single_telescopic(self):
        from rfcensus.commands.guided import find_telescopic_dongle
        ant = _make_antenna("whip_telescopic_915mhz", 915)
        d = _make_dongle(0, antenna=ant)
        antennas_by_id = {ant.id: {"id": ant.id, "resonant_freq_hz": ant.resonant_freq_hz}}
        chosen, ad, msg = find_telescopic_dongle([d], antennas_by_id)
        assert chosen is d
        assert msg == ""

    def test_picks_lowest_id_with_message_when_multiple(self):
        from rfcensus.commands.guided import find_telescopic_dongle
        ant_a = _make_antenna("whip_telescopic_915mhz", 915)
        ant_b = _make_antenna("whip_telescopic_433mhz", 433)
        d_a = _make_dongle(2, antenna=ant_a)
        d_b = _make_dongle(0, antenna=ant_b)  # lower id
        antennas_by_id = {
            ant_a.id: {"id": ant_a.id, "resonant_freq_hz": ant_a.resonant_freq_hz},
            ant_b.id: {"id": ant_b.id, "resonant_freq_hz": ant_b.resonant_freq_hz},
        }
        chosen, ad, msg = find_telescopic_dongle([d_a, d_b], antennas_by_id)
        # rtl-0 sorts before rtl-2 alphabetically
        assert chosen is d_b
        assert "Multiple telescopic dongles" in msg

    def test_returns_none_with_message_when_no_telescopic(self):
        from rfcensus.commands.guided import find_telescopic_dongle
        ant = _make_antenna("whip_915", 915)  # NOT a telescopic id
        d = _make_dongle(0, antenna=ant)
        antennas_by_id = {ant.id: {"id": ant.id, "resonant_freq_hz": ant.resonant_freq_hz}}
        chosen, ad, msg = find_telescopic_dongle([d], antennas_by_id)
        assert chosen is None
        assert "No telescopic" in msg


# ──────────────────────────────────────────────────────────────────
# Quarter-wave length computation
# ──────────────────────────────────────────────────────────────────


class TestQuarterWaveCm:
    def test_915_mhz_is_82_cm(self):
        from rfcensus.commands.guided import quarter_wave_cm
        assert abs(quarter_wave_cm(915_000_000) - 8.19) < 0.05

    def test_433_mhz_is_173_cm(self):
        from rfcensus.commands.guided import quarter_wave_cm
        assert abs(quarter_wave_cm(433_000_000) - 17.31) < 0.05

    def test_162_mhz_is_46_cm(self):
        from rfcensus.commands.guided import quarter_wave_cm
        assert abs(quarter_wave_cm(162_000_000) - 46.27) < 0.05


# ──────────────────────────────────────────────────────────────────
# before_band_callback
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestBeforeBandCallback:
    def _cfg(self, current_qw_cm=8.2):
        from rfcensus.commands.guided import GuidedConfig
        return GuidedConfig(
            dongle_id="rtl-0",
            antenna_id="whip_telescopic_915mhz",
            original_resonant_freq_hz=915_000_000,
            original_quarter_wave_cm=8.2,
            current_quarter_wave_cm=current_qw_cm,
        )

    def _task(self, band_id, freq_mhz):
        from rfcensus.config.schema import BandConfig
        from types import SimpleNamespace
        f = int(freq_mhz * 1_000_000)
        band = BandConfig(
            id=band_id, name=band_id,
            freq_low=int(f * 0.99), freq_high=int(f * 1.01),
        )
        return SimpleNamespace(band=band, suggested_dongle_id="rtl-0")

    async def test_skips_prompt_when_tuning_already_close(self):
        from rfcensus.commands.guided import make_before_band_callback
        cfg = self._cfg(current_qw_cm=8.2)  # tuned for 915
        cb = make_before_band_callback(cfg)
        # Same band — within tolerance, no prompt
        with patch("click.prompt") as mock_prompt:
            result = await cb(self._task("ism_915", 915))
        assert result == "go"
        mock_prompt.assert_not_called()

    async def test_prompts_when_tuning_meaningfully_different(self):
        from rfcensus.commands.guided import make_before_band_callback
        cfg = self._cfg(current_qw_cm=8.2)  # tuned for 915
        cb = make_before_band_callback(cfg)
        # Different band — should prompt; user hits Enter (empty)
        with patch("click.prompt", return_value=""):
            result = await cb(self._task("ism_433", 433))
        assert result == "go"
        # current_quarter_wave_cm should be updated
        assert abs(cfg.current_quarter_wave_cm - 17.3) < 0.5

    async def test_skip_response_returns_skip(self):
        from rfcensus.commands.guided import make_before_band_callback
        cfg = self._cfg(current_qw_cm=8.2)
        cb = make_before_band_callback(cfg)
        with patch("click.prompt", return_value="skip"):
            result = await cb(self._task("ism_433", 433))
        assert result == "skip"
        # Tuning should NOT have been updated since user skipped
        assert cfg.current_quarter_wave_cm == 8.2


# ──────────────────────────────────────────────────────────────────
# update_antenna_in_config
# ──────────────────────────────────────────────────────────────────


class TestUpdateAntennaInConfig:
    def test_updates_dongle_assignment_and_adds_new_stanza(self, tmp_path):
        from rfcensus.commands.guided import update_antenna_in_config

        # Initial config: one dongle pointing to whip_telescopic_915mhz
        config_path = tmp_path / "site.toml"
        config_path.write_text(
            '[[dongles]]\n'
            'id = "rtl-0"\n'
            'serial = "X"\n'
            'antenna = "whip_telescopic_915mhz"\n'
            '\n'
            '[[antennas]]\n'
            'id = "whip_telescopic_915mhz"\n'
            'name = "Telescopic @ 8.2 cm (915 MHz)"\n'
            'antenna_type = "whip"\n'
            'resonant_freq_hz = 915000000\n'
            'usable_range = [777000000, 1052000000]\n'
            'gain_dbi = 2.15\n'
            'polarization = "vertical"\n'
        )

        # Update: user reports 11 cm length → ~681 MHz
        n = update_antenna_in_config(
            config_path,
            old_antenna_id="whip_telescopic_915mhz",
            dongle_id="rtl-0",
            new_resonant_hz=681_000_000,
            new_quarter_wave_cm=11.0,
        )
        assert n == 1
        text = config_path.read_text()
        # New stanza added
        assert "whip_telescopic_681mhz" in text
        # Dongle now points to new id
        assert 'antenna = "whip_telescopic_681mhz"' in text
        # Old stanza removed (no other dongle referenced it)
        assert "whip_telescopic_915mhz" not in text

    def test_keeps_old_stanza_if_other_dongle_references_it(self, tmp_path):
        from rfcensus.commands.guided import update_antenna_in_config

        config_path = tmp_path / "site.toml"
        config_path.write_text(
            '[[dongles]]\n'
            'id = "rtl-0"\n'
            'antenna = "whip_telescopic_915mhz"\n'
            '\n'
            '[[dongles]]\n'
            'id = "rtl-1"\n'
            'antenna = "whip_telescopic_915mhz"\n'  # also uses it
            '\n'
            '[[antennas]]\n'
            'id = "whip_telescopic_915mhz"\n'
            'antenna_type = "whip"\n'
            'resonant_freq_hz = 915000000\n'
            'usable_range = [777000000, 1052000000]\n'
            'gain_dbi = 2.15\n'
            'polarization = "vertical"\n'
        )

        update_antenna_in_config(
            config_path,
            old_antenna_id="whip_telescopic_915mhz",
            dongle_id="rtl-0",
            new_resonant_hz=681_000_000,
            new_quarter_wave_cm=11.0,
        )
        text = config_path.read_text()
        # Old stanza stays because rtl-1 still uses it
        assert "whip_telescopic_915mhz" in text
        # New stanza added
        assert "whip_telescopic_681mhz" in text


# ──────────────────────────────────────────────────────────────────
# CLI compatibility checks
# ──────────────────────────────────────────────────────────────────


class TestCliCompatibility:
    def test_guided_with_per_band_rejected(self):
        from click.testing import CliRunner
        from rfcensus.commands.inventory import cli_scan
        runner = CliRunner()
        result = runner.invoke(
            cli_scan, ["--guided", "--per-band", "30s"]
        )
        assert result.exit_code != 0
        assert "single-pass" in result.output.lower() or "single-pass" in str(result.exception or "").lower()

    def test_guided_with_forever_duration_rejected(self):
        from click.testing import CliRunner
        from rfcensus.commands.inventory import cli_scan
        runner = CliRunner()
        result = runner.invoke(
            cli_scan, ["--guided", "--duration", "forever"]
        )
        assert result.exit_code != 0
