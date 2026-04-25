"""P25 detector.

Identifies P25 (Project 25) public-safety radio activity by channel-level
fingerprint. P25 is the dominant digital standard for US law enforcement,
fire, EMS, and federal agencies.

Fingerprint:

• **Band**: typically 138-174 MHz VHF, 450-520 MHz UHF, or 700/800 MHz
  public safety bands (769-775 MHz, 794-806 MHz, 806-869 MHz)
• **Bandwidth**: ~12.5 kHz for Phase 1 (C4FM), narrower for Phase 2 (TDMA)
• **Control channels**: continuous carrier with rapid deviation (9600 bps
  data stream), classified as `continuous_carrier` or `modulated_continuous`
• **Voice channels**: similar modulation, intermittent (keyed on/off)

We can't tell Phase 1 from Phase 2 from just power samples, nor can we
follow trunking. Detection hands off to SDRTrunk or OP25 for full
monitoring.
"""

from __future__ import annotations

from rfcensus.detectors.base import DetectorBase, DetectorCapabilities
from rfcensus.events import ActiveChannelEvent, DetectionEvent
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


# P25 Phase 1 channels are spaced at 12.5 kHz. We tolerate ±30% because
# the occupancy analyzer's bins may not align perfectly.
P25_BANDWIDTH_HZ = 12_500
P25_BANDWIDTH_TOLERANCE = 0.30


class P25Detector(DetectorBase):
    capabilities = DetectorCapabilities(
        name="p25",
        detected_technologies=["p25_phase1", "p25_phase2", "p25_control_channel"],
        relevant_freq_ranges=(
            # VHF public safety
            (150_800_000, 173_000_000),
            # UHF public safety
            (450_000_000, 520_000_000),
            # 700 MHz public safety narrowband
            (769_000_000, 775_000_000),
            (799_000_000, 806_000_000),
            # 800 MHz public safety trunked
            (806_000_000, 824_000_000),
            (851_000_000, 869_000_000),
        ),
        hand_off_tools=("sdrtrunk", "op25"),
        cpu_cost="cheap",
        description=(
            "Detects P25 public-safety radio activity by 12.5 kHz continuous-carrier fingerprint."
        ),
    )

    CHANNELS_FOR_TRUNKED_DETECTION = 3

    def __init__(self) -> None:
        super().__init__()
        self._continuous_channels: dict[tuple[int, int], set[int]] = {}
        self._announced: set[tuple[int, int]] = set()

    async def on_active_channel(self, event: ActiveChannelEvent) -> None:
        band = self._band_for(event.freq_center_hz)
        if band is None:
            return
        if event.kind == "gone":
            return

        # P25 bandwidth match
        if not self._matches_bandwidth(event.bandwidth_hz):
            return

        # v0.6.2: accept a wider set of classifications.
        #
        # The pre-v0.6.2 filter required `continuous_carrier`,
        # `modulated_continuous`, or `fm_voice` — all of which the
        # SignalClassifier only assigns when active_ratio > 0.9. P25
        # voice channels are keyed on/off (active_ratio typically 0.2-0.7
        # in heavy traffic, much lower in quiet times) so they land in
        # `intermittent` or `pulsed`. Even control channels frequently
        # land in `intermittent` because of bin-level power variability.
        # The result was: in a busy P25 system the detector NEVER
        # FIRED, despite hundreds of P25 carriers visible in the band.
        #
        # New filter: accept `pulsed` and `intermittent` too. Precision
        # is preserved by the bandwidth check (12.5 kHz ±30%), the band
        # check (only public-safety frequency ranges), and the
        # CHANNELS_FOR_TRUNKED_DETECTION threshold for non-dedicated
        # bands. False positives in the 700/800 MHz dedicated bands are
        # still very unlikely because almost nothing else lives there
        # at narrowband 12.5 kHz spacing.
        if event.classification not in (
            "continuous_carrier",
            "modulated_continuous",
            "fm_voice",
            "intermittent",
            "pulsed",
        ):
            return

        self._continuous_channels.setdefault(band, set()).add(event.freq_center_hz)
        channel_count = len(self._continuous_channels[band])

        # Single narrowband channel: ambiguous in non-dedicated bands
        # (could be P25, NXDN, DMR, or analog repeater). Don't announce
        # yet. Multiple 12.5 kHz channels in a public-safety band:
        # trunked system is likely.
        if band in self._announced:
            return
        if channel_count >= self.CHANNELS_FOR_TRUNKED_DETECTION:
            self._announced.add(band)
            await self._announce(band, event, trunked=True)
        elif channel_count >= 1 and self._is_dedicated_p25_band(band):
            # 700/800 MHz public safety bands are dominated by P25; a
            # single narrowband channel at the right spacing is enough
            # evidence to flag it, even if classification was "intermittent"
            # or "pulsed".
            self._announced.add(band)
            await self._announce(band, event, trunked=False)

    async def _announce(
        self, band: tuple[int, int], last_event: ActiveChannelEvent, trunked: bool
    ) -> None:
        if self._event_bus is None:
            return
        channels = sorted(self._continuous_channels[band])
        technology = (
            "p25_trunked_system" if trunked else "p25_narrowband"
        )
        confidence = 0.7 if trunked else 0.5
        evidence = (
            f"{len(channels)} narrowband continuous carrier(s) at "
            f"{', '.join(f'{f/1e6:.3f} MHz' for f in channels[:5])}"
            f"{' + more' if len(channels) > 5 else ''} "
            f"in {band[0]/1e6:.0f}-{band[1]/1e6:.0f} MHz band"
        )
        if trunked:
            evidence += " (pattern consistent with a trunked system)"
        self._detections_emitted += 1
        await self._event_bus.publish(
            DetectionEvent(
                session_id=self._session_id,
                detector_name=self.name,
                technology=technology,
                freq_hz=last_event.freq_center_hz,
                bandwidth_hz=last_event.bandwidth_hz,
                confidence=confidence,
                evidence=evidence,
                hand_off_tools=list(self.capabilities.hand_off_tools),
                metadata={
                    "band_low_hz": band[0],
                    "band_high_hz": band[1],
                    "channels_seen": channels,
                    "trunked": trunked,
                },
            )
        )
        log.info(
            "P25 detection fired at %s: %d channels (trunked=%s)",
            f"{band[0]/1e6:.0f}-{band[1]/1e6:.0f}MHz",
            len(channels), trunked,
        )

    def _band_for(self, freq_hz: int) -> tuple[int, int] | None:
        for low, high in self.capabilities.relevant_freq_ranges:
            if low <= freq_hz <= high:
                return (low, high)
        return None

    def _matches_bandwidth(self, bw_hz: int) -> bool:
        if bw_hz <= 0:
            return False
        return abs(bw_hz - P25_BANDWIDTH_HZ) / P25_BANDWIDTH_HZ <= P25_BANDWIDTH_TOLERANCE

    def _is_dedicated_p25_band(self, band: tuple[int, int]) -> bool:
        """True for bands where P25 dominance makes single-channel detection meaningful."""
        low, _high = band
        # 700/800 MHz public safety
        return 769_000_000 <= low <= 869_000_000
