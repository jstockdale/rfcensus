"""Runtime antenna model and matcher.

`Antenna` is the runtime sibling of `AntennaConfig`. The matcher scores
antenna-band pairs and recommends swaps.
"""

from __future__ import annotations

from dataclasses import dataclass

from rfcensus.config.schema import AntennaConfig, BandConfig


@dataclass(frozen=True, slots=True)
class Antenna:
    id: str
    name: str
    antenna_type: str
    resonant_freq_hz: int | None
    usable_range: tuple[int, int]
    gain_dbi: float
    polarization: str
    requires_bias_power: bool
    notes: str

    @classmethod
    def from_config(cls, cfg: AntennaConfig) -> Antenna:
        return cls(
            id=cfg.id,
            name=cfg.name,
            antenna_type=cfg.antenna_type,
            resonant_freq_hz=cfg.resonant_freq_hz,
            usable_range=cfg.usable_range,
            gain_dbi=cfg.gain_dbi,
            polarization=cfg.polarization,
            requires_bias_power=cfg.requires_bias_power,
            notes=cfg.notes,
        )

    def covers(self, freq_hz: int) -> bool:
        low, high = self.usable_range
        return low <= freq_hz <= high

    def suitability(self, freq_hz: int) -> float:
        if not self.covers(freq_hz):
            return 0.0
        if self.resonant_freq_hz is None or self.resonant_freq_hz == 0:
            low, high = self.usable_range
            center = (low + high) / 2
            span = max(1, (high - low) / 2)
            # Wideband antenna: score peaks in the middle, drops near edges
            return max(0.3, 1.0 - abs(freq_hz - center) / span)
        detune = abs(freq_hz - self.resonant_freq_hz) / max(1, self.resonant_freq_hz)
        if detune < 0.05:
            return 1.0
        if detune < 0.15:
            return 0.75
        if detune < 0.30:
            return 0.45
        return 0.2

    def suitability_for_band(self, band: BandConfig) -> float:
        """Score how well this antenna covers a whole band, not just a point.

        We sample at low, center, and high of the band and take the minimum.
        A resonant antenna might be great at center but cover poorly at edges.
        """
        low_score = self.suitability(band.freq_low)
        center_score = self.suitability(band.center_hz)
        high_score = self.suitability(band.freq_high)
        return min(low_score, center_score, high_score)


@dataclass
class AntennaMatch:
    """Result of the matcher: which dongle/antenna to use for a band."""

    band_id: str
    dongle_id: str
    antenna_id: str | None
    score: float
    warnings: list[str]


class AntennaMatcher:
    """Score antenna-dongle-band pairings and pick the best."""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def score(
        self,
        band: BandConfig,
        antenna: Antenna | None,
    ) -> float:
        if antenna is None:
            return 0.1  # Still usable with any antenna, just not great
        return antenna.suitability_for_band(band)

    def best_pairing(
        self,
        band: BandConfig,
        candidates: list[tuple[str, Antenna | None]],
        *,
        ignore_threshold: bool = False,
        dongle_load: dict[str, int] | None = None,
    ) -> AntennaMatch | None:
        """Given (dongle_id, antenna) pairs, pick the best for this band.

        Returns None if no candidate meets the threshold. With
        `ignore_threshold=True`, returns the highest-scoring candidate
        even if it scores below the threshold (used by --all-bands to
        force-include severely detuned bands), with a clear warning
        attached so the user knows the reception will be poor.

        `dongle_load` (optional): map of dongle_id → number of bands
        already assigned to that dongle. Used as a tie-breaker so when
        two equivalent dongles match (e.g. two whip_915 dongles for a
        915 MHz band), we prefer the less-loaded one. This spreads
        bands across waves so all dongles can run in parallel rather
        than one dongle being double-booked.
        """
        load = dongle_load or {}
        scored = [
            (did, ant, self.score(band, ant)) for did, ant in candidates
        ]
        # Sort by (score desc, load asc, dongle_id) so equivalent matches
        # go to the least-loaded dongle, with stable id-based tiebreak
        # for repeatability.
        scored.sort(key=lambda t: (-t[2], load.get(t[0], 0), t[0]))
        if not scored:
            return None
        did, ant, score = scored[0]
        # Hard floor: even with ignore_threshold, an antenna that doesn't
        # physically cover the frequency (score == 0) cannot receive it.
        if score == 0.0:
            return None
        if score < self.threshold and not ignore_threshold:
            return None
        warnings: list[str] = []
        if ant and score < self.threshold:
            warnings.append(
                f"⚠ severely detuned: antenna {ant.id} scored {score:.2f} for "
                f"{band.name} (below {self.threshold:.2f} threshold); "
                f"reception will be poor and decodes may be unreliable"
            )
        elif ant and 0.3 < score < 0.5:
            warnings.append(
                f"antenna {ant.id} is marginal for {band.name}; consider a better match"
            )
        if ant is None:
            warnings.append(
                f"dongle {did} has no antenna declared; results will be noisy"
            )
        return AntennaMatch(
            band_id=band.id,
            dongle_id=did,
            antenna_id=ant.id if ant else None,
            score=score,
            warnings=warnings,
        )
