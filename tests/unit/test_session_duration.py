"""Tests for session duration budgeting.

The user-facing contract: `--duration N` means N seconds of total wall
time. The session runner divides that budget across waves, since waves
run sequentially.
"""

from __future__ import annotations

import pytest

from rfcensus.engine.session import _compute_per_wave_duration


class TestComputePerWaveDuration:
    def test_one_active_wave_uses_full_budget(self):
        """If there's only one wave, it gets the whole duration."""
        assert _compute_per_wave_duration(600.0, 1) == 600.0

    def test_four_active_waves_split_evenly(self):
        """Standard case: 10-min budget across 4 waves = 150s each."""
        assert _compute_per_wave_duration(600.0, 4) == 150.0

    def test_two_active_waves(self):
        assert _compute_per_wave_duration(600.0, 2) == 300.0

    def test_zero_active_waves_returns_total_safely(self):
        """All waves unassigned — no divide-by-zero. Caller will skip
        everything anyway."""
        assert _compute_per_wave_duration(600.0, 0) == 600.0

    def test_negative_active_waves_returns_total_safely(self):
        """Defensive: negative values shouldn't crash."""
        assert _compute_per_wave_duration(600.0, -1) == 600.0

    def test_minimum_per_wave_floor(self):
        """Pathological short durations get clamped to 1s minimum so
        decoders have a chance to spawn."""
        # 2s / 6 waves = 0.33s per wave — clamps to 1.0
        assert _compute_per_wave_duration(2.0, 6) == 1.0

    def test_no_floor_when_per_wave_above_minimum(self):
        """Floor only kicks in for genuinely tiny per-wave values."""
        # 60s / 4 waves = 15s — well above floor
        assert _compute_per_wave_duration(60.0, 4) == 15.0

    def test_fractional_budget_preserved(self):
        """Don't round — preserve the exact division."""
        # 100s / 3 waves = 33.333...
        result = _compute_per_wave_duration(100.0, 3)
        assert abs(result - 33.333333333) < 0.001
