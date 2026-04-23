"""Detector registry — sibling of DecoderRegistry."""

from __future__ import annotations

from rfcensus.detectors.base import DetectorBase
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


class DetectorRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type[DetectorBase]] = {}

    def register(self, detector_cls: type[DetectorBase]) -> None:
        name = detector_cls.capabilities.name
        if name in self._classes:
            log.debug("overriding existing detector registration for %s", name)
        self._classes[name] = detector_cls

    def names(self) -> list[str]:
        return sorted(self._classes.keys())

    def get(self, name: str) -> type[DetectorBase] | None:
        return self._classes.get(name)

    def all_instances(self) -> list[DetectorBase]:
        """Build one instance of each registered detector."""
        return [cls() for cls in self._classes.values()]


_REGISTRY: DetectorRegistry | None = None


def get_registry() -> DetectorRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = DetectorRegistry()
        from rfcensus.detectors.builtin import lora, p25, wifi_bt  # noqa: F401

        _REGISTRY.register(lora.LoraDetector)
        _REGISTRY.register(p25.P25Detector)
        _REGISTRY.register(wifi_bt.WifiBtDetector)
    return _REGISTRY


def reset_registry() -> None:
    global _REGISTRY
    _REGISTRY = None
