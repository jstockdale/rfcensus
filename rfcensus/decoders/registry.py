"""Decoder registry.

Tracks all DecoderBase subclasses available to rfcensus. Built-in decoders
are registered via explicit imports; third-party decoders can register
by calling `register_decoder()`.
"""

from __future__ import annotations

from rfcensus.config.schema import DecoderConfig, SiteConfig
from rfcensus.decoders.base import DecoderBase
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


class DecoderRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type[DecoderBase]] = {}

    def register(self, decoder_cls: type[DecoderBase]) -> None:
        name = decoder_cls.capabilities.name
        if name in self._classes:
            log.debug("overriding existing decoder registration for %s", name)
        self._classes[name] = decoder_cls

    def names(self) -> list[str]:
        return sorted(self._classes.keys())

    def get(self, name: str) -> type[DecoderBase] | None:
        return self._classes.get(name)

    def decoders_for_frequency(self, freq_hz: int) -> list[type[DecoderBase]]:
        return [
            cls
            for cls in self._classes.values()
            if cls.capabilities.covers(freq_hz)
        ]

    def decoders_for_band(self, freq_low: int, freq_high: int) -> list[type[DecoderBase]]:
        """Decoders whose ranges overlap with the given band."""
        result: list[type[DecoderBase]] = []
        for cls in self._classes.values():
            for low, high in cls.capabilities.freq_ranges:
                if low <= freq_high and high >= freq_low:
                    result.append(cls)
                    break
        return result

    def instantiate(
        self, name: str, site_config: SiteConfig
    ) -> DecoderBase | None:
        cls = self._classes.get(name)
        if cls is None:
            return None
        decoder_config = site_config.decoders.get(name, DecoderConfig())
        if not decoder_config.enabled:
            return None
        return cls(decoder_config)


_REGISTRY: DecoderRegistry | None = None


def get_registry() -> DecoderRegistry:
    """Return the singleton registry, populating built-in decoders on first use."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = DecoderRegistry()
        # Deferred import to avoid circular dependencies
        from rfcensus.decoders.builtin import rtl_433, rtl_ais, rtlamr, multimon, direwolf  # noqa: F401

        _REGISTRY.register(rtl_433.Rtl433Decoder)
        _REGISTRY.register(rtlamr.RtlamrDecoder)
        _REGISTRY.register(rtl_ais.RtlAisDecoder)
        _REGISTRY.register(multimon.MultimonDecoder)
        _REGISTRY.register(direwolf.DirewolfDecoder)
    return _REGISTRY


def reset_registry() -> None:
    global _REGISTRY
    _REGISTRY = None
