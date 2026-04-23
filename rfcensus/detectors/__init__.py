"""Detection-only modules.

A detector is the sibling of a decoder. Both observe the RF environment,
but they have different output semantics:

• Decoders produce frames (`DecodeEvent`) that can be aggregated into
  emitters with device IDs.
• Detectors produce pattern matches (`DetectionEvent`) that recognize a
  technology is present but don't attempt to decode individual frames.
  They exist for protocols that are encrypted, proprietary, or best
  served by specialized tools.

Detectors consume `ActiveChannelEvent`s from the spectrum layer and
emit `DetectionEvent`s. They do not control hardware directly; they ride
on top of whatever spectrum scanning is already happening.
"""

from rfcensus.detectors.base import (
    DetectorAvailability,
    DetectorBase,
    DetectorCapabilities,
    DetectorResult,
)
from rfcensus.detectors.registry import DetectorRegistry, get_registry

__all__ = [
    "DetectorAvailability",
    "DetectorBase",
    "DetectorCapabilities",
    "DetectorRegistry",
    "DetectorResult",
    "get_registry",
]
