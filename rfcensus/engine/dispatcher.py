"""Dispatcher: maps a band to a strategy.

The mapping takes into account the band's default strategy (from its
definition), any per-band override in the user's config, and the
overall default from the `strategies` section.
"""

from __future__ import annotations

from rfcensus.config.schema import BandConfig, SiteConfig, StrategyKind
from rfcensus.engine.strategy import (
    DecoderOnlyStrategy,
    DecoderPrimaryStrategy,
    ExplorationStrategy,
    PowerPrimaryStrategy,
    Strategy,
)


class Dispatcher:
    """Choose a strategy for each band based on config."""

    def __init__(self, config: SiteConfig):
        self.config = config
        self._strategies: dict[StrategyKind, Strategy] = {
            StrategyKind.DECODER_ONLY: DecoderOnlyStrategy(),
            StrategyKind.DECODER_PRIMARY: DecoderPrimaryStrategy(),
            StrategyKind.POWER_PRIMARY: PowerPrimaryStrategy(),
            StrategyKind.EXPLORATION: ExplorationStrategy(),
        }

    def strategy_for(self, band: BandConfig) -> Strategy:
        override = self.config.strategies.overrides.get(band.id)
        kind = override or band.strategy or self.config.strategies.default
        return self._strategies[kind]
