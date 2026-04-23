"""Hardware abstraction: dongle enumeration, allocation, and health."""

from rfcensus.hardware.antenna import Antenna, AntennaMatcher
from rfcensus.hardware.broker import (
    DongleBroker,
    DongleLease,
    DongleRequirements,
    NoDongleAvailable,
)
from rfcensus.hardware.dongle import Dongle, DongleCapabilities, DongleStatus
from rfcensus.hardware.registry import HardwareRegistry, detect_hardware

__all__ = [
    "Antenna",
    "AntennaMatcher",
    "Dongle",
    "DongleBroker",
    "DongleCapabilities",
    "DongleLease",
    "DongleRequirements",
    "DongleStatus",
    "HardwareRegistry",
    "NoDongleAvailable",
    "detect_hardware",
]
