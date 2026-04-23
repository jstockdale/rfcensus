"""Spectrum backend abstraction.

A spectrum backend produces an async stream of PowerSample events covering
some frequency range. Backends differ in speed, resolution, and range:

• rtl_power: slow iterative sweep, works on any RTL-SDR
• hackrf_sweep: very fast sweep, only on HackRF, covers 1 MHz - 6 GHz
• soapy_power: slower fallback for non-RTL / non-HackRF SDRs (future)

All backends emit PowerSample records with freq_hz / power_dbm at the
resolution requested via SpectrumSweepSpec.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

from rfcensus.hardware.broker import DongleLease


@dataclass
class SpectrumSweepSpec:
    """What frequencies and at what resolution to sweep."""

    freq_low: int
    freq_high: int
    bin_width_hz: int = 10_000
    dwell_ms: int = 200
    duration_s: float | None = None  # None = until cancelled
    gain: str = "auto"  # "auto" or numeric dB string


@dataclass
class PowerSample:
    timestamp: datetime
    freq_hz: int
    bin_width_hz: int
    power_dbm: float


class SpectrumBackend(ABC):
    """Abstract spectrum scanner."""

    name: str = "abstract"
    # Frequency coverage this backend supports
    max_range: tuple[int, int] = (0, 0)
    # Approximate sweep rate in Hz/sec for planning
    sweep_rate_hz_per_sec: int = 0

    @classmethod
    @abstractmethod
    def available_on(cls, lease: DongleLease) -> bool:
        """True if this backend can run against the given dongle lease."""

    @abstractmethod
    async def sweep(
        self, lease: DongleLease, spec: SpectrumSweepSpec
    ) -> AsyncIterator[PowerSample]:
        """Yield PowerSamples for the requested spec until cancelled or duration elapses."""
