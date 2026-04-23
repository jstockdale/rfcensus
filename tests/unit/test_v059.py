"""Tests for v0.5.9 — wizard ergonomics:
  • Back option in sub-menus (recover from wrong choice without Ctrl-C)
  • Actual antenna length input (recompute resonant freq from physics)
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest


# ──────────────────────────────────────────────────────────────────
# Back-option menu helper
# ──────────────────────────────────────────────────────────────────


class TestMenuWithBack:
    def test_back_returns_negative_one(self, monkeypatch):
        from rfcensus.commands.setup import _menu_with_back
        # Three real options + appended "back" → 4 visible items;
        # selecting 4 (back) returns -1
        with patch("click.prompt", return_value="4"):
            choice = _menu_with_back(["A", "B", "C"], prompt=">")
        assert choice == -1

    def test_real_choice_returns_zero_indexed(self, monkeypatch):
        from rfcensus.commands.setup import _menu_with_back
        # Selecting 2 (the second real option) returns index 1
        with patch("click.prompt", return_value="2"):
            choice = _menu_with_back(["A", "B", "C"], prompt=">")
        assert choice == 1

    def test_first_choice_returns_zero(self):
        from rfcensus.commands.setup import _menu_with_back
        with patch("click.prompt", return_value="1"):
            choice = _menu_with_back(["A", "B"], prompt=">")
        assert choice == 0


# ──────────────────────────────────────────────────────────────────
# Beginner flow back-out behavior
# ──────────────────────────────────────────────────────────────────


class TestBeginnerFlowBackOut:
    def test_back_from_category_returns_none(self):
        """User entered the 'help me decide' flow then chose 'back'
        from the category menu — should return None so caller can
        re-show the parent frequency menu."""
        from rfcensus.commands.setup import _suggest_frequency_for_beginner
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
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
        # Category menu has 6 options + back; "7" picks back
        with patch("click.prompt", return_value="7"):
            result = _suggest_frequency_for_beginner(d)
        assert result is None

    def test_back_from_which_one_returns_to_category(self):
        """User picked 'pick something' → got list of frequencies →
        picked back → should loop to category menu, not return None."""
        from rfcensus.commands.setup import _suggest_frequency_for_beginner
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
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
        # Sequence:
        #   Category prompt → "6" (pick something — gives 5 candidates)
        #   Which one prompt → "6" (back, since 5 options + 1 back)
        #   Category prompt → "1" (cars/sensors → 3 candidates)
        #   Which one prompt → "1" (first option → 915 MHz)
        responses = iter(["6", "6", "1", "1"])

        def fake_prompt(*a, **kw):
            return next(responses)

        with patch("click.prompt", side_effect=fake_prompt):
            result = _suggest_frequency_for_beginner(d)
        # Should have eventually returned a profile, not None
        assert result is not None
        # First option in cars/sensors candidates is 915 MHz
        assert result.freq_hz == 915_000_000


# ──────────────────────────────────────────────────────────────────
# Actual antenna length input
# ──────────────────────────────────────────────────────────────────


class TestActualLengthInput:
    """When the user types an actual length different from the
    recommended quarter-wave, we should recompute the resonant freq
    and create a custom antenna stanza with the actual physics."""

    def _make_state(self):
        from rfcensus.commands.setup import _WizardState
        state = _WizardState(detected=[])
        # Pretend the library has whip_915 already
        state.library_antennas = [
            {"id": "whip_915", "name": "915 MHz tuned whip"},
        ]
        return state

    def _make_dongle(self):
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
        )
        caps = DongleCapabilities(
            freq_range_hz=(24_000_000, 1_700_000_000),
            max_sample_rate=2_400_000, bits_per_sample=8,
            bias_tee_capable=False, tcxo_ppm=10.0,
        )
        return Dongle(
            id="rtl-0", serial="X", model="rtlsdr_generic",
            driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
            driver_index=0,
        )

    def test_empty_input_creates_custom_at_suggested_freq(self):
        """Press Enter (empty input) → use the suggested length and
        return the matching library antenna id."""
        from rfcensus.commands.setup import _flow_telescopic
        state = self._make_state()
        dongle = self._make_dongle()

        # Sequence:
        #   _pick_frequency → "11" picks 915 MHz
        #   actual length prompt → "" (just press enter)
        responses = iter(["11", ""])

        def fake_prompt(*a, **kw):
            return next(responses)

        with patch("click.prompt", side_effect=fake_prompt):
            result = _flow_telescopic(dongle, state)

        # v0.5.10: telescopic always creates custom (not library)
        assert result.startswith("whip_telescopic_")
        assert len(state.custom_antennas) == 1
        assert state.custom_antennas[0]["resonant_freq_hz"] == 915_000_000

    def test_custom_length_creates_custom_antenna(self):
        """User says 11 cm for the 915 MHz suggestion → recompute as
        ~681 MHz quarter-wave → create custom antenna with actual freq."""
        from rfcensus.commands.setup import _flow_telescopic
        state = self._make_state()
        dongle = self._make_dongle()

        # Pick 915 MHz, then say actual length is 11 cm
        responses = iter(["11", "11"])

        def fake_prompt(*a, **kw):
            return next(responses)

        with patch("click.prompt", side_effect=fake_prompt):
            result = _flow_telescopic(dongle, state)

        # Should NOT return whip_915 — should be a custom antenna
        assert result != "whip_915"
        assert len(state.custom_antennas) == 1
        custom = state.custom_antennas[0]
        # Quarter-wave at 11 cm: c/(4f) = 11 → f = 7494.8/11 ≈ 681 MHz
        actual_mhz = custom["resonant_freq_hz"] / 1e6
        assert 670 < actual_mhz < 695  # rounding tolerance
        # Name should reflect the actual measurement
        assert "11.0 cm" in custom["name"]

    def test_invalid_length_falls_back_to_suggestion(self):
        """Garbage input → fall back to using the suggested length and
        the library antenna (don't crash, don't pollute config)."""
        from rfcensus.commands.setup import _flow_telescopic
        state = self._make_state()
        dongle = self._make_dongle()

        # Pick 915 MHz, then type garbage
        responses = iter(["11", "this is not a number"])

        def fake_prompt(*a, **kw):
            return next(responses)

        with patch("click.prompt", side_effect=fake_prompt):
            result = _flow_telescopic(dongle, state)

        # v0.5.10: falls back to suggested freq as a custom stanza
        assert result.startswith("whip_telescopic_")
        assert state.custom_antennas[0]["resonant_freq_hz"] == 915_000_000

    def test_negative_length_rejected(self):
        from rfcensus.commands.setup import _flow_telescopic
        state = self._make_state()
        dongle = self._make_dongle()

        responses = iter(["11", "-5"])

        def fake_prompt(*a, **kw):
            return next(responses)

        with patch("click.prompt", side_effect=fake_prompt):
            result = _flow_telescopic(dongle, state)

        # v0.5.10: negative rejected → falls back to suggested 915 MHz custom
        assert result.startswith("whip_telescopic_")
        assert state.custom_antennas[0]["resonant_freq_hz"] == 915_000_000
