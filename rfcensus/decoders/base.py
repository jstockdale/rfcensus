"""Decoder base class.

A decoder takes a leased dongle, tunes it to a frequency, runs whatever
subprocess does the actual demodulation, and emits DecodeEvents on the
shared bus.

Each decoder declares its capabilities statically via `DecoderCapabilities`.
The dispatcher and scheduler use these to pick decoders for a given band
and to decide on exclusive vs shared dongle access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pydantic import BaseModel

from rfcensus.config.schema import DecoderConfig
from rfcensus.events import EventBus
from rfcensus.hardware.broker import AccessMode, DongleLease


@dataclass(frozen=True)
class DecoderCapabilities:
    """Static descriptor of what a decoder can do."""

    name: str
    protocols: list[str]
    # Frequencies where this decoder produces meaningful output, in Hz
    freq_ranges: tuple[tuple[int, int], ...]
    min_sample_rate: int = 1_024_000
    preferred_sample_rate: int = 2_400_000
    # If true, decoder wants to open the dongle directly (via -d index)
    # If false, decoder can connect to an rtl_tcp server
    requires_exclusive_dongle: bool = True
    # External tool binary required on PATH
    external_binary: str = ""
    # CPU cost tier: "cheap", "moderate", "expensive"
    cpu_cost: str = "cheap"
    # If true, this decoder is opt-in and not enabled by default
    opt_in: bool = False
    description: str = ""

    def covers(self, freq_hz: int) -> bool:
        return any(low <= freq_hz <= high for low, high in self.freq_ranges)

    @property
    def access_mode(self) -> AccessMode:
        return (
            AccessMode.EXCLUSIVE
            if self.requires_exclusive_dongle
            else AccessMode.SHARED
        )


@dataclass
class DecoderAvailability:
    """Result of checking whether a decoder can run on the current system."""

    name: str
    available: bool
    reason: str = ""
    binary_path: str | None = None
    version: str | None = None


@dataclass
class DecoderResult:
    """Summary returned when a decoder finishes its run."""

    name: str
    decodes_emitted: int = 0
    errors: list[str] = field(default_factory=list)
    ended_reason: str = "completed"


@dataclass
class DecoderRunSpec:
    """Everything a decoder needs to run.

    Passing this dataclass instead of many positional args makes
    signatures stable across decoder additions and removes positional
    ordering hazards.
    """

    lease: "DongleLease"  # forward reference — imported at top
    freq_hz: int
    sample_rate: int
    duration_s: float | None
    event_bus: "EventBus"
    session_id: int
    # Optional tuning hints the strategy can pass in
    rssi_offset_db: float = 0.0
    notes: str = ""
    gain: str = "auto"  # "auto" for AGC, or a numeric dB string like "40"
    # Per-band decoder-specific options. Keyed by decoder name; value
    # is a dict the decoder parses as it sees fit. Lets a BandConfig
    # override decoder behavior without polluting DecoderRunSpec with
    # decoder-specific fields (e.g. rtlamr_msgtype).
    #
    # Example usage (strategy.py):
    #   spec = DecoderRunSpec(
    #       ...,
    #       decoder_options=band.decoder_options,
    #   )
    # Then inside rtlamr decoder:
    #   opts = spec.decoder_options.get("rtlamr", {})
    #   msgtype = opts.get("msgtype", "all")
    decoder_options: dict[str, dict[str, object]] = field(
        default_factory=dict
    )


class DecoderSettings(BaseModel):
    """Runtime tunables applied to a decoder instance.

    Populated from a `DecoderConfig` plus optional per-band overrides.
    """

    binary: str | None = None
    extra_args: list[str] = []


class DecoderBase(ABC):
    """Base class for every decoder.

    Concrete decoders implement:

    • `capabilities` (classvar) – what the decoder supports
    • `check_available()` – verify the external tool is present
    • `run(...)` – run until duration elapses or cancelled, emit events

    The base class wires up subprocess lifecycle so subclasses only need
    to build the command-line and parse output lines.
    """

    capabilities: DecoderCapabilities

    def __init__(self, config: DecoderConfig | None = None):
        self.config = config or DecoderConfig()
        self.settings = DecoderSettings(
            binary=self.config.binary,
            extra_args=list(self.config.extra_args),
        )

    @property
    def name(self) -> str:
        return self.capabilities.name

    @abstractmethod
    async def check_available(self) -> DecoderAvailability:
        ...

    @abstractmethod
    async def run(self, spec: DecoderRunSpec) -> DecoderResult:
        """Run the decoder until `spec.duration_s` elapses or cancellation.

        Implementations emit DecodeEvents via `spec.event_bus` for every
        frame they decode and return a DecoderResult summary on exit.
        """
        ...
