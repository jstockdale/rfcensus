"""Rolling noise floor tracker.

Maintains a per-frequency noise floor estimate using a sliding percentile
of recent samples. Signals above noise_floor + margin are considered
active. The noise floor itself adapts over time as the RF environment
changes (day/night, FM broadcast variations, etc.).
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _BinHistory:
    window: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    sorted_cache: list[float] = field(default_factory=list)
    dirty: bool = False

    def push(self, value: float) -> None:
        self.window.append(value)
        self.dirty = True

    def percentile(self, p: float) -> float:
        if not self.window:
            return -100.0
        if self.dirty:
            self.sorted_cache = sorted(self.window)
            self.dirty = False
        idx = int(p * (len(self.sorted_cache) - 1))
        return self.sorted_cache[idx]


class NoiseFloorTracker:
    """Tracks a sliding percentile per frequency bin."""

    def __init__(self, *, window: int = 60, percentile: float = 0.25):
        self.window = window
        self.percentile = percentile
        self._bins: dict[int, _BinHistory] = {}

    def observe(self, freq_hz: int, power_dbm: float) -> None:
        bin_hist = self._bins.get(freq_hz)
        if bin_hist is None:
            bin_hist = _BinHistory(window=deque(maxlen=self.window))
            self._bins[freq_hz] = bin_hist
        bin_hist.push(power_dbm)

    def noise_floor(self, freq_hz: int) -> float:
        bin_hist = self._bins.get(freq_hz)
        if bin_hist is None:
            return -100.0
        return bin_hist.percentile(self.percentile)

    def snr(self, freq_hz: int, power_dbm: float) -> float:
        return power_dbm - self.noise_floor(freq_hz)

    def stats(self) -> dict[str, int]:
        return {"tracked_bins": len(self._bins)}

    def neighborhood_floor(self, freq_hz: int, span_hz: int) -> float:
        """Noise floor averaged over nearby bins to avoid self-masking by tones."""
        low = freq_hz - span_hz
        high = freq_hz + span_hz
        nearby = [f for f in self._bins if low <= f <= high]
        if not nearby:
            return self.noise_floor(freq_hz)
        floors = sorted(self.noise_floor(f) for f in nearby)
        # Median of the lower half is a robust "floor of the floor"
        return floors[len(floors) // 4] if floors else -100.0
