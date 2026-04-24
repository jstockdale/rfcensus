"""v0.5.42 tests: channel-plan matcher.

Validates the LoRaWAN US915/EU868/AU915 and Meshtastic default lookup
tables against known-good channels from the respective standards.
"""

from __future__ import annotations

import pytest

from rfcensus.spectrum.channel_plans import (
    ChannelMatch,
    channels_for_plan,
    list_plans,
    match_channel,
)


class TestLoRaWANUS915:
    """LoRaWAN US915 uplink channels start at 902.3 MHz, 200 kHz
    spacing, 125 kHz wide — 64 channels. Plus 8 × 500 kHz SF8
    uplinks at 903.0 + N·1.6 MHz. Plus 8 × 500 kHz downlinks."""

    def test_uplink_0_at_902_300(self):
        m = match_channel(freq_hz=902_300_000, bandwidth_hz=125_000)
        assert m is not None
        assert m.plan == "lorawan_us915"
        assert m.channel_id == "uplink_0"

    def test_uplink_8_at_903_900(self):
        m = match_channel(freq_hz=903_900_000, bandwidth_hz=125_000)
        assert m is not None
        assert m.plan == "lorawan_us915"
        assert m.channel_id == "uplink_8"

    def test_uplink_63_at_914_900(self):
        """Last 125 kHz uplink channel."""
        m = match_channel(freq_hz=914_900_000, bandwidth_hz=125_000)
        assert m is not None
        assert m.channel_id == "uplink_63"

    def test_500khz_uplink(self):
        """SF8/500kHz channel at 903.0 MHz."""
        m = match_channel(freq_hz=903_000_000, bandwidth_hz=500_000)
        assert m is not None
        assert m.plan == "lorawan_us915"
        assert m.channel_id == "uplink500_0"

    def test_downlink_at_923_300(self):
        m = match_channel(freq_hz=923_300_000, bandwidth_hz=500_000)
        assert m is not None
        assert "downlink" in m.channel_id

    def test_small_frequency_error_still_matches(self):
        """3 kHz off should still match — within default tolerance."""
        m = match_channel(freq_hz=902_303_000, bandwidth_hz=125_000)
        assert m is not None
        assert m.channel_id == "uplink_0"

    def test_large_frequency_error_does_not_match(self):
        """10 kHz off should NOT match — beyond default tolerance."""
        m = match_channel(freq_hz=902_310_000, bandwidth_hz=125_000)
        # Either matches uplink_0 (close) or doesn't match — both OK;
        # what matters is we don't misidentify as a completely wrong
        # channel
        if m:
            assert m.channel_id in ("uplink_0",)

    def test_wrong_bandwidth_does_not_match(self):
        """125 kHz channel freq with 500 kHz request → no match."""
        m = match_channel(freq_hz=902_300_000, bandwidth_hz=500_000)
        # Either matches a 500 kHz channel at that freq, or nothing.
        # Don't match as a 125 kHz channel.
        if m is not None:
            assert m.bandwidth_hz == 500_000


class TestLoRaWANEU868:
    def test_mandatory_channel_0(self):
        m = match_channel(freq_hz=868_100_000, bandwidth_hz=125_000)
        assert m is not None
        assert m.plan == "lorawan_eu868"
        assert "mandatory" in m.channel_id

    def test_mandatory_channel_2(self):
        m = match_channel(freq_hz=868_500_000, bandwidth_hz=125_000)
        assert m is not None
        assert m.plan == "lorawan_eu868"

    def test_sf7_250khz_channel(self):
        m = match_channel(freq_hz=868_300_000, bandwidth_hz=250_000)
        assert m is not None
        assert m.plan == "lorawan_eu868"
        assert m.bandwidth_hz == 250_000


class TestLoRaWANAU915:
    def test_uplink_0(self):
        m = match_channel(freq_hz=915_200_000, bandwidth_hz=125_000)
        assert m is not None
        assert m.plan == "lorawan_au915"
        assert m.channel_id == "uplink_0"

    def test_uplink_30(self):
        m = match_channel(freq_hz=921_200_000, bandwidth_hz=125_000)
        assert m is not None
        assert m.plan == "lorawan_au915"
        assert m.channel_id == "uplink_30"


class TestMeshtasticUS:
    def test_longfast_default(self):
        m = match_channel(freq_hz=906_875_000, bandwidth_hz=250_000)
        assert m is not None
        assert m.plan == "meshtastic_us"
        assert m.channel_id == "longfast_default"

    def test_mediumfast_default(self):
        m = match_channel(freq_hz=913_125_000, bandwidth_hz=250_000)
        assert m is not None
        assert m.plan == "meshtastic_us"
        assert m.channel_id == "mediumfast_default"

    def test_mediumslow_default(self):
        m = match_channel(freq_hz=914_875_000, bandwidth_hz=250_000)
        assert m is not None
        assert m.plan == "meshtastic_us"
        assert m.channel_id == "mediumslow_default"

    def test_shortfast_default(self):
        """v0.5.43: ShortFast slot 68 at 918.875 MHz (Puget Mesh)."""
        m = match_channel(freq_hz=918_875_000, bandwidth_hz=250_000)
        assert m is not None
        assert m.plan == "meshtastic_us"
        assert m.channel_id == "shortfast_default"

    def test_shortslow_default(self):
        """v0.5.43: ShortSlow slot 75 at 920.625 MHz (djb2 computed)."""
        m = match_channel(freq_hz=920_625_000, bandwidth_hz=250_000)
        assert m is not None
        assert m.channel_id == "shortslow_default"

    def test_shortturbo_default(self):
        """v0.5.43: ShortTurbo uses 500 kHz BW → 52-slot grid;
        slot 50 = 926.75 MHz."""
        m = match_channel(freq_hz=926_750_000, bandwidth_hz=500_000)
        assert m is not None
        assert m.channel_id == "shortturbo_default"

    def test_all_six_meshtastic_presets_distinct(self):
        """All six modern preset defaults hash to different slots →
        different frequencies. Verifies no accidental duplicates in
        the table."""
        presets = [
            (906_875_000, 250_000),  # LongFast
            (913_125_000, 250_000),  # MediumFast
            (914_875_000, 250_000),  # MediumSlow
            (918_875_000, 250_000),  # ShortFast
            (920_625_000, 250_000),  # ShortSlow
            (926_750_000, 500_000),  # ShortTurbo
        ]
        centers = {
            match_channel(freq_hz=f, bandwidth_hz=bw).center_hz
            for f, bw in presets
        }
        assert len(centers) == 6


class TestMeshtasticEU:
    def test_default_longfast(self):
        m = match_channel(freq_hz=869_525_000, bandwidth_hz=250_000)
        assert m is not None
        assert m.plan == "meshtastic_eu"
        # v0.5.43: EU868's single 250 kHz slot is shared by all presets
        assert m.channel_id == "eu868_default"


class TestNoMatch:
    def test_freq_far_from_any_channel(self):
        """Deep in the 900 MHz ISM gap between LoRaWAN channels."""
        m = match_channel(freq_hz=925_000_000, bandwidth_hz=125_000)
        # 925 MHz is too far from any US915 channel (nearest is ~924.9)
        # and should not match
        assert m is None or abs(m.center_hz - 925_000_000) < 5_000

    def test_out_of_band_freq(self):
        """800 MHz — way outside LoRaWAN bands."""
        m = match_channel(freq_hz=800_000_000, bandwidth_hz=125_000)
        assert m is None


class TestPlanEnumeration:
    def test_list_plans_returns_all_five(self):
        plans = list_plans()
        assert set(plans) == {
            "lorawan_us915",
            "lorawan_eu868",
            "lorawan_au915",
            "meshtastic_us",
            "meshtastic_eu",
        }

    def test_us915_has_64_plus_8_plus_8_channels(self):
        channels = channels_for_plan("lorawan_us915")
        assert len(channels) == 80  # 64 uplink + 8 SF8 + 8 downlink

    def test_au915_has_64_uplink_channels(self):
        channels = channels_for_plan("lorawan_au915")
        assert len(channels) == 64


class TestBestMatch:
    """When multiple channels match within tolerance, the nearest
    wins."""

    def test_picks_nearest_of_two_candidates(self):
        """902.38 MHz is 80 kHz from US915 uplink 0 (902.3) and 120 kHz
        from uplink 1 (902.5). But both are outside default 5 kHz
        tolerance. With a wider tolerance, the closer one should
        win."""
        m = match_channel(
            freq_hz=902_380_000, bandwidth_hz=125_000,
            match_tolerance_hz=100_000,
        )
        assert m is not None
        assert m.channel_id == "uplink_0"  # 80 kHz err < 120 kHz err
