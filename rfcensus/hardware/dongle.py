"""Dongle data model.

A `Dongle` represents one physical SDR device in our universe. It may or
may not be currently plugged in; persistent records are identified by
serial number so we can track "this is the V3 I've been using for weeks"
across reboots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from rfcensus.hardware.antenna import Antenna


class DongleStatus(str, Enum):
    DETECTED = "detected"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class DongleCapabilities:
    """What a dongle hardware can do.

    Populated based on model + driver probe. Some fields are best-effort
    (e.g. we don't actually know a dongle's TCXO quality unless the user
    tells us in their config; we use model defaults).
    """

    freq_range_hz: tuple[int, int]
    max_sample_rate: int
    bits_per_sample: int
    bias_tee_capable: bool
    tcxo_ppm: float
    can_transmit: bool = False
    can_share_via_rtl_tcp: bool = True
    wide_scan_capable: bool = False  # True for HackRF and similar

    def as_dict(self) -> dict[str, Any]:
        return {
            "freq_range_hz": list(self.freq_range_hz),
            "max_sample_rate": self.max_sample_rate,
            "bits_per_sample": self.bits_per_sample,
            "bias_tee_capable": self.bias_tee_capable,
            "tcxo_ppm": self.tcxo_ppm,
            "can_transmit": self.can_transmit,
            "can_share_via_rtl_tcp": self.can_share_via_rtl_tcp,
            "wide_scan_capable": self.wide_scan_capable,
        }


@dataclass
class Dongle:
    """A single SDR. The unit of allocation in the broker."""

    id: str
    serial: str | None
    model: str
    driver: str  # "rtlsdr", "hackrf", "soapy"
    capabilities: DongleCapabilities
    antenna: Antenna | None = None
    status: DongleStatus = DongleStatus.DETECTED
    usb_bus: int | None = None
    usb_address: int | None = None
    usb_port_path: str | None = None
    last_health_check: datetime | None = None
    health_notes: list[str] = field(default_factory=list)
    # Driver-specific handles, e.g. the device index rtl_433 / rtl_power expect
    driver_index: int | None = None

    def covers(self, freq_hz: int) -> bool:
        low, high = self.capabilities.freq_range_hz
        return low <= freq_hz <= high

    def is_usable(self) -> bool:
        return self.status in (DongleStatus.DETECTED, DongleStatus.HEALTHY)

    def describe(self) -> str:
        antenna_desc = self.antenna.name if self.antenna else "no antenna"
        serial = self.serial or "?"
        return f"{self.id} ({self.model}, sn={serial}, {antenna_desc})"
