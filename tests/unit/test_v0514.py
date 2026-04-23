"""Tests for v0.5.14:
  • Library generic whips widened to ±35% for survey detection work
  • Telescopic custom stanzas widened to ±50%
  • 700/800 MHz P25 public safety bands added
  • rtl_power sidecar deferred so decoders get first dongle claim
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestLibraryWhipWidening:
    def test_whip_915_covers_700_and_800_mhz_p25(self):
        from rfcensus.config.loader import load_config

        config = load_config()
        whip_915 = next(
            (a for a in config.antennas if a.id == "whip_915"), None,
        )
        assert whip_915 is not None
        low, high = whip_915.usable_range
        assert low <= 775_000_000
        assert low <= 860_000_000

    def test_whip_433_covers_neighbor_bands(self):
        from rfcensus.config.loader import load_config

        config = load_config()
        whip_433 = next(
            (a for a in config.antennas if a.id == "whip_433"), None,
        )
        low, high = whip_433.usable_range
        assert low <= 315_000_000
        assert high >= 462_000_000

    def test_whip_315_covers_honeywell_345(self):
        from rfcensus.config.loader import load_config

        config = load_config()
        whip_315 = next(
            (a for a in config.antennas if a.id == "whip_315"), None,
        )
        low, high = whip_315.usable_range
        assert low <= 280_000_000
        assert high >= 345_000_000

    def test_whips_still_score_optimal_at_resonance(self):
        from rfcensus.config.loader import load_config
        from rfcensus.hardware.antenna import Antenna

        config = load_config()
        for ant_id in ("whip_315", "whip_433", "whip_915"):
            cfg = next(a for a in config.antennas if a.id == ant_id)
            ant = Antenna.from_config(cfg)
            assert ant.suitability(ant.resonant_freq_hz) == 1.0


class TestP25Bands:
    def test_p25_700_band_defined(self):
        from rfcensus.config.loader import load_config

        config = load_config()
        p25_700 = next(
            (b for b in config.band_definitions if b.id == "p25_700_public_safety"),
            None,
        )
        assert p25_700 is not None
        assert p25_700.freq_low >= 760_000_000
        assert p25_700.freq_high <= 780_000_000

    def test_p25_800_band_defined(self):
        from rfcensus.config.loader import load_config

        config = load_config()
        p25_800 = next(
            (b for b in config.band_definitions if b.id == "p25_800_public_safety"),
            None,
        )
        assert p25_800 is not None
        assert p25_800.freq_low >= 845_000_000
        assert p25_800.freq_high <= 875_000_000

    def test_p25_bands_auto_enabled(self):
        from rfcensus.config.loader import load_config

        config = load_config()
        for band_id in ("p25_700_public_safety", "p25_800_public_safety"):
            b = next(
                bd for bd in config.band_definitions if bd.id == band_id
            )
            assert b.opt_in is False

    def test_p25_bands_suggest_p25_decoder(self):
        from rfcensus.config.loader import load_config

        config = load_config()
        for band_id in ("p25_700_public_safety", "p25_800_public_safety"):
            b = next(bd for bd in config.band_definitions if bd.id == band_id)
            assert "p25" in b.suggested_decoders

    def test_p25_band_matches_widened_whip_915(self):
        from rfcensus.config.loader import load_config
        from rfcensus.engine.coverage import compute_coverage
        from rfcensus.hardware.antenna import Antenna
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
        )

        config = load_config()
        whip = Antenna.from_config(
            next(a for a in config.antennas if a.id == "whip_915")
        )
        caps = DongleCapabilities(
            freq_range_hz=(24_000_000, 1_700_000_000),
            max_sample_rate=2_400_000, bits_per_sample=8,
            bias_tee_capable=False, tcxo_ppm=10.0,
        )
        d = Dongle(
            id="rtl-0", serial="X", model="rtlsdr_generic",
            driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
            driver_index=0,
        )
        d.antenna = whip

        p25_800 = next(
            bd for bd in config.band_definitions
            if bd.id == "p25_800_public_safety"
        )
        report = compute_coverage([p25_800], [d])
        assert len(report.matched) == 1


class TestTelescopicWidening:
    def test_new_telescopic_stanzas_use_pm_50_percent(self):
        from rfcensus.commands.setup import _flow_telescopic, _WizardState
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
        )

        caps = DongleCapabilities(
            freq_range_hz=(24_000_000, 1_700_000_000),
            max_sample_rate=2_400_000, bits_per_sample=8,
            bias_tee_capable=False, tcxo_ppm=10.0,
        )
        dongle = Dongle(
            id="rtl-0", serial="X", model="rtlsdr_generic",
            driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
            driver_index=0,
        )
        state = _WizardState(detected=[])
        state.library_antennas = []

        responses = iter(["11", ""])
        with patch("click.prompt", side_effect=lambda *a, **kw: next(responses)):
            _flow_telescopic(dongle, state)

        assert len(state.custom_antennas) == 1
        custom = state.custom_antennas[0]
        low, high = custom["usable_range"]
        assert low == int(915_000_000 * 0.5)
        assert high == int(915_000_000 * 1.5)


class TestRtlPowerDeferral:
    def test_deferral_present_in_strategy_source(self):
        import rfcensus.engine.strategy as strategy
        import inspect
        src = inspect.getsource(strategy)
        assert "_deferred_power_scan" in src
        # deferral uses asyncio.sleep with some delay
        assert "await asyncio.sleep(1.0)" in src or "await asyncio.sleep(0.5)" in src
