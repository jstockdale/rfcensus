"""DetectorBase abstract class.

A detector subscribes to the event bus (typically ActiveChannelEvents)
and emits DetectionEvents when its fingerprint matches. Detectors don't
hold dongles continuously — they consume what's already being observed.

Detectors that declare `consumes_iq=True` in their capabilities are
given an `IQCaptureService` at attach time, so they can escalate
heuristic suspicions by pulling a brief IQ window for in-depth analysis
(chirp autocorrelation, instantaneous frequency, modulation classifier).
IQ captures are opportunistic: if no dongle is free, the detector
continues on heuristic evidence alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rfcensus.events import ActiveChannelEvent, EventBus

if TYPE_CHECKING:
    from rfcensus.spectrum.iq_capture import IQCaptureService


@dataclass(frozen=True)
class DetectorCapabilities:
    """Static descriptor of what a detector can detect."""

    name: str
    detected_technologies: list[str]
    relevant_freq_ranges: tuple[tuple[int, int], ...]
    consumes_active_channels: bool = True
    consumes_power_samples: bool = False
    # If true, the detector will be given access to IQ capture on attach.
    # Detectors that set this should gracefully handle the service being None
    # (no IQ-capable dongle available) or captures failing.
    consumes_iq: bool = False
    hand_off_tools: tuple[str, ...] = ()
    cpu_cost: str = "cheap"
    description: str = ""

    def covers(self, freq_hz: int) -> bool:
        return any(low <= freq_hz <= high for low, high in self.relevant_freq_ranges)


@dataclass
class DetectorAvailability:
    name: str
    available: bool = True
    reason: str = ""


@dataclass
class DetectorResult:
    name: str
    detections_emitted: int = 0
    evidence_accumulated: dict[str, int] = field(default_factory=dict)


class DetectorBase(ABC):
    """Base class for every detector."""

    capabilities: DetectorCapabilities

    def __init__(self):
        self._session_id: int = 0
        self._event_bus: EventBus | None = None
        self._iq_service: IQCaptureService | None = None
        self._detections_emitted: int = 0

    @property
    def name(self) -> str:
        return self.capabilities.name

    def attach(
        self,
        bus: EventBus,
        session_id: int,
        iq_service: IQCaptureService | None = None,
    ) -> None:
        """Subscribe to the bus and accept optional IQ service."""
        self._event_bus = bus
        self._session_id = session_id
        if self.capabilities.consumes_iq:
            self._iq_service = iq_service
        if self.capabilities.consumes_active_channels:
            bus.subscribe(ActiveChannelEvent, self._handle_channel)

    async def _handle_channel(self, event: ActiveChannelEvent) -> None:
        if not self.capabilities.covers(event.freq_center_hz):
            return
        await self.on_active_channel(event)

    async def on_active_channel(self, event: ActiveChannelEvent) -> None:
        """Override to process active channel events in your detector's band."""

    def check_available(self) -> DetectorAvailability:
        return DetectorAvailability(name=self.name, available=True)

    def finalize(self) -> DetectorResult:
        return DetectorResult(
            name=self.name, detections_emitted=self._detections_emitted
        )
