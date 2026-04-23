"""LoRa detector.

Identifies LoRa / LoRaWAN activity through channel-level fingerprint,
with optional IQ-level chirp confirmation.

Detection pipeline:

1. **Heuristic** (always): observe ActiveChannelEvents. Require LoRa-standard
   bandwidth (125/250/500 kHz ±20%), pulsed classification, and one of the
   common LoRa bands (US 902-928, EU 863-870, 433 ISM).
2. **IQ confirmation** (if available): when heuristic fires, opportunistically
   capture ~500 ms of IQ at the suspect frequency and test for chirp-shaped
   instantaneous frequency. Chirp pattern → high confidence LoRa. No chirp
   but heuristic matched → lower confidence, still reported.
3. **LoRaWAN escalation**: seeing 3+ distinct channels in one band with the
   same fingerprint indicates a gateway, not just a stray device.

Why no full decode? LoRa CSS demodulation requires symbol synchronization,
spreading-factor detection, and Reed-Solomon decoding. That's a specialized
job; we hand off to gr-lora_sdr / chirpstack.
"""

from __future__ import annotations

import asyncio

from rfcensus.detectors.base import DetectorBase, DetectorCapabilities
from rfcensus.events import ActiveChannelEvent, DetectionEvent
from rfcensus.spectrum.chirp_analysis import analyze_chirps
from rfcensus.spectrum.iq_capture import IQCaptureError
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


LORA_BANDWIDTHS_HZ = (125_000, 250_000, 500_000)
LORA_BANDWIDTH_TOLERANCE = 0.20


class LoraDetector(DetectorBase):
    capabilities = DetectorCapabilities(
        name="lora",
        detected_technologies=["lora", "lorawan"],
        relevant_freq_ranges=(
            (902_000_000, 928_000_000),
            (863_000_000, 870_000_000),
            (432_000_000, 434_500_000),
        ),
        consumes_iq=True,
        hand_off_tools=("gr-lora_sdr", "chirpstack", "lorapacketforwarder"),
        cpu_cost="cheap",
        description=(
            "Detects LoRa/LoRaWAN activity by channel fingerprint, with "
            "optional IQ chirp-pattern confirmation when a free dongle is available."
        ),
    )

    BURSTS_FOR_DETECTION = 3
    IQ_CAPTURE_DURATION_S = 0.5
    IQ_SAMPLE_RATE = 1_024_000
    MAX_IQ_CONFIRMATIONS_PER_SESSION = 6

    def __init__(self) -> None:
        super().__init__()
        self._bursts_by_band: dict[tuple[int, int], int] = {}
        self._announced_bands: set[tuple[int, int]] = set()
        self._bws_by_band: dict[tuple[int, int], set[int]] = {}
        self._freqs_by_band: dict[tuple[int, int], set[int]] = {}
        self._chirp_results: dict[tuple[int, int], bool] = {}
        self._iq_lock = asyncio.Lock()
        self._iq_confirmations_done = 0

    async def on_active_channel(self, event: ActiveChannelEvent) -> None:
        band = self._band_for(event.freq_center_hz)
        if band is None:
            return
        if event.kind == "gone":
            return

        matched_bw = self._match_bandwidth(event.bandwidth_hz)
        if matched_bw is None:
            return

        if event.classification not in ("pulsed", "intermittent", "periodic", "unknown"):
            return

        self._bursts_by_band[band] = self._bursts_by_band.get(band, 0) + 1
        self._bws_by_band.setdefault(band, set()).add(matched_bw)
        self._freqs_by_band.setdefault(band, set()).add(event.freq_center_hz)

        if band in self._announced_bands:
            return
        if self._bursts_by_band[band] < self.BURSTS_FOR_DETECTION:
            return

        self._announced_bands.add(band)
        chirp_confirmed: bool | None = None
        if (
            self._iq_service is not None
            and self._iq_confirmations_done < self.MAX_IQ_CONFIRMATIONS_PER_SESSION
        ):
            chirp_confirmed = await self._confirm_with_iq(band, event)
            self._iq_confirmations_done += 1

        await self._announce(band, event, chirp_confirmed=chirp_confirmed)

    async def _confirm_with_iq(
        self, band: tuple[int, int], event: ActiveChannelEvent
    ) -> bool | None:
        """Try to grab IQ and run chirp analysis. Returns None on failure."""
        assert self._iq_service is not None
        async with self._iq_lock:
            try:
                capture = await self._iq_service.capture(
                    freq_hz=event.freq_center_hz,
                    sample_rate=self.IQ_SAMPLE_RATE,
                    duration_s=self.IQ_CAPTURE_DURATION_S,
                    prefer_driver="rtlsdr",
                    timeout_alloc_s=2.0,
                )
            except IQCaptureError as exc:
                log.debug(
                    "lora IQ confirmation skipped for %s: %s",
                    f"{event.freq_center_hz/1e6:.3f}MHz", exc,
                )
                return None

            if capture.bytes_received < 100_000:
                log.debug("lora IQ too short to analyze (%d bytes)", capture.bytes_received)
                return None

            try:
                result = analyze_chirps(capture.samples, capture.sample_rate)
            except Exception:
                log.exception("chirp analysis failed")
                return None

            confirmed = result.chirp_confidence > 0.5 and result.num_chirp_segments >= 1
            self._chirp_results[band] = confirmed
            log.info(
                "lora IQ confirmation at %.3f MHz: confirmed=%s (%s)",
                event.freq_center_hz / 1e6, confirmed, result.reasoning,
            )
            return confirmed

    async def _announce(
        self,
        band: tuple[int, int],
        last_event: ActiveChannelEvent,
        *,
        chirp_confirmed: bool | None,
    ) -> None:
        if self._event_bus is None:
            return
        bursts = self._bursts_by_band[band]
        bws = self._bws_by_band[band]
        freqs = sorted(self._freqs_by_band[band])
        distinct_channels = len(freqs)

        confidence = 0.3 + 0.1 * min(bursts, 5) + 0.1 * min(distinct_channels, 3) + 0.1 * len(bws)
        evidence_parts = [
            f"{bursts} LoRa-bandwidth bursts at {distinct_channels} distinct "
            f"frequencies in {band[0]/1e6:.0f}-{band[1]/1e6:.0f} MHz "
            f"(bandwidths: {', '.join(f'{b//1000}kHz' for b in sorted(bws))})"
        ]
        if chirp_confirmed is True:
            confidence = min(1.0, confidence + 0.3)
            evidence_parts.append("IQ chirp pattern confirmed")
        elif chirp_confirmed is False:
            confidence = max(0.2, confidence - 0.15)
            evidence_parts.append("IQ capture did not confirm chirp pattern")

        if distinct_channels >= 3:
            evidence_parts.append("multi-channel pattern consistent with LoRaWAN gateway")
        confidence = min(1.0, confidence)

        self._detections_emitted += 1
        await self._event_bus.publish(
            DetectionEvent(
                session_id=self._session_id,
                detector_name=self.name,
                technology="lorawan" if distinct_channels >= 3 else "lora",
                freq_hz=last_event.freq_center_hz,
                bandwidth_hz=last_event.bandwidth_hz,
                confidence=confidence,
                evidence="; ".join(evidence_parts),
                hand_off_tools=list(self.capabilities.hand_off_tools),
                metadata={
                    "band_low_hz": band[0],
                    "band_high_hz": band[1],
                    "distinct_channels": distinct_channels,
                    "bandwidths_seen": sorted(bws),
                    "freqs_seen": freqs,
                    "bursts": bursts,
                    "iq_confirmed": chirp_confirmed,
                },
            )
        )
        log.info(
            "LoRa detection fired at %s: %d bursts across %d channels (IQ confirmed=%s)",
            f"{band[0]/1e6:.0f}-{band[1]/1e6:.0f}MHz",
            bursts, distinct_channels, chirp_confirmed,
        )

    def _band_for(self, freq_hz: int) -> tuple[int, int] | None:
        for low, high in self.capabilities.relevant_freq_ranges:
            if low <= freq_hz <= high:
                return (low, high)
        return None

    def _match_bandwidth(self, bw_hz: int) -> int | None:
        if bw_hz <= 0:
            return None
        for standard in LORA_BANDWIDTHS_HZ:
            if abs(bw_hz - standard) / standard <= LORA_BANDWIDTH_TOLERANCE:
                return standard
        return None
