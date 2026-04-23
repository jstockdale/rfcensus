"""Occupancy analyzer.

Consumes a stream of PowerSamples and produces ActiveChannel records for
frequencies that exceed the dynamic noise floor by a configurable margin.

Outputs on the event bus:

• PowerSampleEvent for every sample (for the --capture-power writer)
• ActiveChannelEvent (kind=new/updated/gone) as channels rise and fall

Classification is delegated to `SignalClassifier`, which reads from a
`ChannelHistory` per bin. Classifications update over time as more
evidence accumulates.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from rfcensus.events import ActiveChannelEvent, EventBus, PowerSampleEvent
from rfcensus.spectrum.backend import PowerSample
from rfcensus.spectrum.classifier import ChannelHistory, Classification, SignalClassifier
from rfcensus.spectrum.noise_floor import NoiseFloorTracker
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class _ActiveState:
    freq_hz: int
    bin_width_hz: int
    first_seen: datetime
    last_seen: datetime
    peak_power: float
    sum_power: float = 0.0
    sample_count: int = 0
    noise_floor: float = -100.0
    dongle_id: str = ""
    classification: Classification = field(
        default_factory=lambda: Classification(kind="unknown", confidence=0.0)
    )

    def update(self, sample: PowerSample, floor: float) -> None:
        self.last_seen = sample.timestamp
        self.peak_power = max(self.peak_power, sample.power_dbm)
        self.sum_power += sample.power_dbm
        self.sample_count += 1
        self.noise_floor = floor

    @property
    def avg_power(self) -> float:
        return self.sum_power / self.sample_count if self.sample_count else self.peak_power


@dataclass
class OccupancyAnalyzer:
    """Detects active channels from a PowerSample stream."""

    event_bus: EventBus
    session_id: int = 0
    activity_threshold_db: float = 6.0
    new_hold_time: timedelta = field(default_factory=lambda: timedelta(seconds=1))
    stale_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    reclassify_every_n_samples: int = 20
    tracker_window: int = 120
    tracker_percentile: float = 0.25
    emit_power_samples: bool = False

    def __post_init__(self) -> None:
        self._tracker = NoiseFloorTracker(
            window=self.tracker_window, percentile=self.tracker_percentile
        )
        self._active: dict[int, _ActiveState] = {}
        self._emitted_new: set[int] = set()
        self._classifier = SignalClassifier()
        self._histories: dict[int, ChannelHistory] = {}

    async def consume(
        self,
        source: AsyncIterator[PowerSample],
        dongle_id: str = "",
    ) -> int:
        """Consume a spectrum sweep. Returns number of samples processed.

        Callers use the return value to detect silent backend failures —
        a sweep that ends with zero samples almost always means the
        underlying subprocess (rtl_power, hackrf_sweep) failed to open
        the dongle and exited before emitting anything.
        """
        n = 0
        async for sample in source:
            await self._process(sample, dongle_id)
            n += 1
        await self._expire_all()
        return n

    async def _process(self, sample: PowerSample, dongle_id: str) -> None:
        if self.emit_power_samples:
            await self.event_bus.publish(
                PowerSampleEvent(
                    session_id=self.session_id,
                    dongle_id=dongle_id,
                    freq_hz=sample.freq_hz,
                    bin_width_hz=sample.bin_width_hz,
                    power_dbm=sample.power_dbm,
                    timestamp=sample.timestamp,
                )
            )

        self._tracker.observe(sample.freq_hz, sample.power_dbm)
        floor = self._tracker.noise_floor(sample.freq_hz)
        snr = sample.power_dbm - floor
        above_floor = snr >= self.activity_threshold_db
        now = sample.timestamp

        history = self._histories.get(sample.freq_hz)
        if history is None:
            history = ChannelHistory(freq_hz=sample.freq_hz)
            self._histories[sample.freq_hz] = history
        history.observe(now, sample.power_dbm, above_floor)

        if above_floor:
            state = self._active.get(sample.freq_hz)
            if state is None:
                state = _ActiveState(
                    freq_hz=sample.freq_hz,
                    bin_width_hz=sample.bin_width_hz,
                    first_seen=now,
                    last_seen=now,
                    peak_power=sample.power_dbm,
                    noise_floor=floor,
                    dongle_id=dongle_id,
                )
                self._active[sample.freq_hz] = state
            state.update(sample, floor)

            if state.sample_count % self.reclassify_every_n_samples == 0:
                state.classification = self._classifier.classify(history)

            if (
                sample.freq_hz not in self._emitted_new
                and (state.last_seen - state.first_seen) >= self.new_hold_time
            ):
                self._emitted_new.add(sample.freq_hz)
                await self._emit(state, kind="new")
            elif sample.freq_hz in self._emitted_new:
                if state.sample_count % self.reclassify_every_n_samples == 0:
                    await self._emit(state, kind="updated")
        else:
            state = self._active.get(sample.freq_hz)
            if state and (now - state.last_seen) > self.stale_timeout:
                await self._expire(sample.freq_hz)

    async def _emit(self, state: _ActiveState, kind: str) -> None:
        await self.event_bus.publish(
            ActiveChannelEvent(
                session_id=self.session_id,
                kind=kind,  # type: ignore[arg-type]
                dongle_id=state.dongle_id,
                freq_center_hz=state.freq_hz,
                bandwidth_hz=state.bin_width_hz,
                peak_power_dbm=state.peak_power,
                avg_power_dbm=state.avg_power,
                noise_floor_dbm=state.noise_floor,
                snr_db=state.peak_power - state.noise_floor,
                classification=state.classification.kind,
                persistence_ratio=min(1.0, state.sample_count / 60.0),
                confidence=state.classification.confidence,
            )
        )

    async def _expire(self, freq_hz: int) -> None:
        state = self._active.pop(freq_hz, None)
        if state is None or freq_hz not in self._emitted_new:
            return
        self._emitted_new.discard(freq_hz)
        history = self._histories.get(freq_hz)
        if history is not None:
            state.classification = self._classifier.classify(history)
        await self._emit(state, kind="gone")

    async def _expire_all(self) -> None:
        for freq_hz in list(self._active.keys()):
            await self._expire(freq_hz)
