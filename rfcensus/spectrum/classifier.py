"""Signal classification for active channels.

Given the time-series of power samples for a single frequency bin,
infer what kind of signal it is:

• `continuous_carrier` — power always on, low variance (unmodulated or CW)
• `fm_voice` — continuous with moderate variance (NFM or AM voice)
• `pulsed` — intermittent bursts with long gaps (TPMS, fobs, doorbell)
• `periodic` — regular interval transmissions (weather station, meter)
• `fhss` — continuous presence but frequency jumps (ERT, Bluetooth)
• `unknown` — not enough evidence yet

The classifier is stateful: it accumulates history per bin and updates
classification as evidence accumulates. Low-confidence results return
`unknown`; classifications stabilize as the sample count grows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean, pstdev

from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class _Sample:
    timestamp: datetime
    power_dbm: float
    above_floor: bool


@dataclass
class ChannelHistory:
    """Rolling statistics for one frequency bin."""

    freq_hz: int
    samples: deque[_Sample] = field(default_factory=lambda: deque(maxlen=300))
    active_runs: int = 0  # Number of distinct activity bursts
    total_active_samples: int = 0
    total_samples: int = 0
    last_burst_start: datetime | None = None
    last_burst_end: datetime | None = None
    burst_durations: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    inter_burst_gaps: deque[float] = field(default_factory=lambda: deque(maxlen=30))

    def observe(self, timestamp: datetime, power_dbm: float, above_floor: bool) -> None:
        # Update burst bookkeeping
        prior_active = self.samples[-1].above_floor if self.samples else False
        if above_floor and not prior_active:
            # Burst beginning
            self.active_runs += 1
            if self.last_burst_end is not None:
                gap = (timestamp - self.last_burst_end).total_seconds()
                if 0 <= gap <= 3600:  # ignore pathological gaps
                    self.inter_burst_gaps.append(gap)
            self.last_burst_start = timestamp
        elif not above_floor and prior_active:
            # Burst ending
            if self.last_burst_start is not None:
                dur = (timestamp - self.last_burst_start).total_seconds()
                if 0 <= dur <= 60:
                    self.burst_durations.append(dur)
            self.last_burst_end = timestamp

        self.samples.append(
            _Sample(timestamp=timestamp, power_dbm=power_dbm, above_floor=above_floor)
        )
        self.total_samples += 1
        if above_floor:
            self.total_active_samples += 1


@dataclass
class Classification:
    kind: str
    confidence: float  # 0-1
    reasoning: str = ""


class SignalClassifier:
    """Classifies an active channel based on accumulated ChannelHistory.

    The classifier is stateless per call but returns different results
    as the caller's ChannelHistory accumulates samples.
    """

    # Thresholds chosen based on typical residential signal characteristics.
    # These are heuristics and will be refined as we collect real-world data.
    MIN_SAMPLES_FOR_CLASSIFICATION = 10
    MIN_SAMPLES_FOR_HIGH_CONFIDENCE = 60

    def classify(self, history: ChannelHistory) -> Classification:
        if history.total_samples < self.MIN_SAMPLES_FOR_CLASSIFICATION:
            return Classification(
                kind="unknown", confidence=0.1, reasoning="insufficient samples"
            )

        active_ratio = history.total_active_samples / max(1, history.total_samples)
        power_values = [s.power_dbm for s in history.samples if s.above_floor]
        power_std = pstdev(power_values) if len(power_values) > 2 else 0.0
        confidence_base = min(
            1.0, history.total_samples / self.MIN_SAMPLES_FOR_HIGH_CONFIDENCE
        )

        # Case 1: nearly always on
        if active_ratio > 0.9:
            if power_std < 2.0:
                return Classification(
                    kind="continuous_carrier",
                    confidence=0.8 * confidence_base,
                    reasoning=f"active_ratio={active_ratio:.2f}, power_std={power_std:.1f}",
                )
            if power_std < 6.0:
                return Classification(
                    kind="fm_voice",
                    confidence=0.6 * confidence_base,
                    reasoning=(
                        f"continuous with modulation "
                        f"(active_ratio={active_ratio:.2f}, power_std={power_std:.1f})"
                    ),
                )
            return Classification(
                kind="modulated_continuous",
                confidence=0.5 * confidence_base,
                reasoning=f"high variance ({power_std:.1f} dB) on continuous carrier",
            )

        # Case 2: periodic bursts
        if len(history.inter_burst_gaps) >= 3:
            gaps = list(history.inter_burst_gaps)
            gap_mean = mean(gaps)
            gap_std = pstdev(gaps) if len(gaps) > 1 else 0.0
            # Low variance = periodic; high variance = event-driven
            if gap_mean > 0 and gap_std / gap_mean < 0.3:
                return Classification(
                    kind="periodic",
                    confidence=0.7 * confidence_base,
                    reasoning=(
                        f"regular interval "
                        f"(mean={gap_mean:.1f}s, std={gap_std:.1f}s, "
                        f"bursts={history.active_runs})"
                    ),
                )

        # Case 3: sparse bursts
        if active_ratio < 0.2 and history.active_runs >= 2:
            return Classification(
                kind="pulsed",
                confidence=0.6 * confidence_base,
                reasoning=(
                    f"sparse bursts "
                    f"(active_ratio={active_ratio:.2f}, bursts={history.active_runs})"
                ),
            )

        # Case 4: intermediate — maybe FHSS or just inconsistent
        if 0.2 <= active_ratio <= 0.9:
            return Classification(
                kind="intermittent",
                confidence=0.4 * confidence_base,
                reasoning=(
                    f"active_ratio={active_ratio:.2f}, "
                    f"bursts={history.active_runs}, "
                    f"could be FHSS seen at one channel"
                ),
            )

        return Classification(
            kind="unknown",
            confidence=0.2 * confidence_base,
            reasoning="no strong match",
        )
