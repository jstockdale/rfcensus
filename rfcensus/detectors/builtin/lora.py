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
from rfcensus.events import ActiveChannelEvent, DetectionEvent, WideChannelEvent
from rfcensus.spectrum.chirp_analysis import analyze_chirps
from rfcensus.spectrum.iq_capture import IQCaptureError
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


LORA_BANDWIDTHS_HZ = (125_000, 250_000, 500_000)
LORA_BANDWIDTH_TOLERANCE = 0.20


class LoraDetector(DetectorBase):
    capabilities = DetectorCapabilities(
        name="lora",
        detected_technologies=["lora", "lorawan", "meshtastic"],
        relevant_freq_ranges=(
            (902_000_000, 928_000_000),
            (863_000_000, 870_000_000),
            (432_000_000, 434_500_000),
        ),
        # v0.5.38: consume WideChannelEvents produced by
        # WideChannelAggregator. This is the PRIMARY detection path;
        # narrow bin events are retained only as a legacy fallback
        # (rarely fires because raw power-scan bins are narrower than
        # LoRa channels — see spectrum/wide_channel_aggregator.py).
        consumes_wide_channels=True,
        consumes_iq=True,
        hand_off_tools=("gr-lora_sdr", "chirpstack", "lorapacketforwarder"),
        cpu_cost="cheap",
        description=(
            "Detects LoRa/LoRaWAN/Meshtastic activity by wide-channel "
            "fingerprint (125/250/500 kHz spans in LoRa bands), with "
            "optional IQ chirp-pattern + spreading-factor estimation "
            "when a free dongle is available."
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
        # v0.5.40: emit one WARNING on the first IQ-capture failure per
        # session (so SF=None in detections is explained), then stay at
        # DEBUG for subsequent failures. Avoids log flooding in dense
        # scans where every LoRa detection would otherwise warn.
        self._iq_failure_warned: bool = False
        # v0.5.38: for wide-channel path only, track the widest template
        # already announced per band. Subsequent events are suppressed
        # UNLESS they carry a strictly wider template (upgrade). This
        # mirrors the aggregator's upgrade semantics — a 250 kHz detection
        # supersedes an earlier 125 kHz one at the same location rather
        # than being blocked by it.
        self._announced_wide_template_by_band: dict[tuple[int, int], int] = {}

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

    async def on_wide_channel(self, event: WideChannelEvent) -> None:
        """v0.5.38 primary path: WideChannelEvent from the aggregator
        represents an already-confirmed wide-bandwidth signal in this
        detector's coverage area. We fire immediately (no burst count
        threshold), opportunistically grab IQ for chirp confirmation,
        and include SF estimation if chirp slope is available.
        """
        band = self._band_for(event.freq_center_hz)
        if band is None:
            return

        # The aggregator's matched_template_hz is already one of our
        # standard widths; record it for evidence.
        matched_bw = event.matched_template_hz
        self._bws_by_band.setdefault(band, set()).add(matched_bw)
        self._freqs_by_band.setdefault(band, set()).add(event.freq_center_hz)
        # Count wide-channel events as "bursts" so that a gateway
        # emitting multiple events over time accumulates evidence for
        # LoRaWAN escalation (distinct_channels >= 3 heuristic).
        self._bursts_by_band[band] = self._bursts_by_band.get(band, 0) + 1

        # If we've already announced at this band with a template
        # AT LEAST this wide, suppress. Allow upgrade to a strictly
        # wider template (so a 125 kHz partial during burst growth
        # doesn't block the correct 250 kHz announcement that comes
        # later as more bins fill in).
        prev_template = self._announced_wide_template_by_band.get(band)
        if prev_template is not None and prev_template >= event.matched_template_hz:
            return

        self._announced_wide_template_by_band[band] = event.matched_template_hz
        self._announced_bands.add(band)

        # v0.5.41: Don't attempt inline IQ confirmation during a scan.
        # All dongles are typically leased to primary decoders + rtl_power
        # during inventory waves (see v0.5.40 investigation), so attempts
        # would fail silently or with WARNING noise. Instead, mark the
        # detection as needing confirmation and let the DetectionWriter
        # auto-submit a ConfirmationTask to the session's queue. The
        # wave planner will schedule confirmation captures into idle wave
        # slots; the user is prompted for an extra wave at session end
        # if any confirmations are still pending.
        await self._announce_wide(
            band=band,
            event=event,
            chirp_confirmed=None,
            chirp_result=None,
            defer_confirmation=True,
        )

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

    async def _capture_and_analyze(self, event: WideChannelEvent):
        """Capture IQ for a brief window centered on this channel and
        return the ChirpAnalysis (or None on failure).

        Distinct from `_confirm_with_iq`: doesn't update announced_bands
        or burst counters — that bookkeeping is done by the caller.
        Caller uses result for confidence AND spreading-factor
        classification.

        v0.5.40: log the FIRST IQ-capture failure per session at
        WARNING level with the underlying cause, so the operator can
        see why SF=None appears in detections. Subsequent failures
        stay at DEBUG to avoid log flooding. The common cause during
        inventory scans is "no dongle available" — all rtlsdr dongles
        are leased to the scanning decoders and rtl_power, and the
        2s allocate timeout passes without one freeing up.
        """
        if self._iq_service is None:
            return None
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
                if not self._iq_failure_warned:
                    self._iq_failure_warned = True
                    log.warning(
                        "lora IQ confirmation unavailable at %.3f MHz: %s. "
                        "SF/variant classification will be skipped for "
                        "this session. Common cause during inventory "
                        "scans: all dongles leased to decoders and "
                        "rtl_power — no free dongle for opportunistic "
                        "IQ capture within the 2s allocate timeout. "
                        "Run `rfcensus monitor` with a single decoder "
                        "to free other dongles for IQ, or reserve a "
                        "scout dongle in a future hybrid-mode config.",
                        event.freq_center_hz / 1e6, exc,
                    )
                else:
                    log.debug(
                        "lora IQ confirmation skipped for %.3f MHz: %s",
                        event.freq_center_hz / 1e6, exc,
                    )
                return None

            if capture.bytes_received < 100_000:
                log.debug(
                    "lora IQ too short to analyze (%d bytes)",
                    capture.bytes_received,
                )
                return None

            try:
                result = analyze_chirps(capture.samples, capture.sample_rate)
            except Exception:
                log.exception("chirp analysis failed")
                return None

            log.info(
                "lora IQ analysis at %.3f MHz: conf=%.2f, %d segments, "
                "mean slope %.1f kHz/s (%s)",
                event.freq_center_hz / 1e6,
                result.chirp_confidence,
                result.num_chirp_segments,
                result.mean_slope_hz_per_sec / 1e3,
                result.reasoning,
            )
            return result

    async def _announce_wide(
        self,
        *,
        band: tuple[int, int],
        event: WideChannelEvent,
        chirp_confirmed: bool | None,
        chirp_result,
        defer_confirmation: bool = False,
    ) -> None:
        """Announce a LoRa-family detection based on a WideChannelEvent.

        Distinct from `_announce` because we have:
          • An already-confirmed wide-channel bandwidth
          • Possible chirp analysis with slope → spreading factor
          • The ability to call out "this is Meshtastic LongFast /
            MediumFast / LoRaWAN SF7" when slope+BW is distinctive

        v0.5.41: if `defer_confirmation=True`, the detection is emitted
        with `needs_iq_confirmation=True` in its metadata. The
        DetectionWriter picks this up after persistence and submits a
        ConfirmationTask to the session queue. The detection fires
        immediately with estimated_sf=None; the confirmation task will
        update the row in-place once it runs.
        """
        if self._event_bus is None:
            return

        bursts = self._bursts_by_band[band]
        bws = self._bws_by_band[band]
        freqs = sorted(self._freqs_by_band[band])
        distinct_channels = len(freqs)

        # Base confidence is higher than the narrow-band path because
        # the aggregator has already confirmed wide-bandwidth activity
        # matching a LoRa template.
        confidence = 0.5 + 0.1 * min(bursts, 3) + 0.1 * min(distinct_channels, 3)

        bw_khz = event.matched_template_hz // 1000
        evidence_parts = [
            f"wide-channel composite at {event.freq_center_hz/1e6:.3f} MHz "
            f"({bw_khz} kHz, {event.constituent_bin_count} bins, "
            f"{event.coverage_ratio*100:.0f}% coverage)"
        ]

        # Spreading-factor / variant classification from chirp slope
        sf_estimate: int | None = None
        variant_label: str | None = None
        if chirp_result is not None and chirp_result.num_chirp_segments >= 1:
            sf_estimate = _estimate_sf_from_slope(
                slope_hz_per_sec=chirp_result.mean_slope_hz_per_sec,
                bandwidth_hz=event.matched_template_hz,
            )
            if sf_estimate is not None:
                variant_label = _label_variant(
                    sf=sf_estimate,
                    bandwidth_hz=event.matched_template_hz,
                )
                evidence_parts.append(
                    f"chirp slope {chirp_result.mean_slope_hz_per_sec/1e3:.0f} "
                    f"kHz/s → estimated SF{sf_estimate}"
                    + (f" ({variant_label})" if variant_label else "")
                )

        if chirp_confirmed is True:
            confidence = min(1.0, confidence + 0.25)
            evidence_parts.append("IQ chirp pattern confirmed")
        elif chirp_confirmed is False:
            confidence = max(0.25, confidence - 0.15)
            evidence_parts.append("IQ capture did not confirm chirp pattern")

        if defer_confirmation:
            evidence_parts.append(
                "SF classification deferred to confirmation wave"
            )

        if distinct_channels >= 3:
            evidence_parts.append(
                "multi-channel pattern consistent with LoRaWAN gateway"
            )
        confidence = min(1.0, confidence)

        # Technology label: meshtastic variants win if SF+BW matches,
        # then LoRaWAN if multi-channel, then generic lora.
        if variant_label and variant_label.startswith("meshtastic"):
            technology = "meshtastic"
        elif distinct_channels >= 3:
            technology = "lorawan"
        else:
            technology = "lora"

        metadata = {
            "band_low_hz": band[0],
            "band_high_hz": band[1],
            "distinct_channels": distinct_channels,
            "bandwidths_seen": sorted(bws),
            "freqs_seen": freqs,
            "bursts": bursts,
            "iq_confirmed": chirp_confirmed,
            "estimated_sf": sf_estimate,
            "variant": variant_label,
            "wide_channel_bins": event.constituent_bin_count,
            "wide_channel_coverage": event.coverage_ratio,
        }
        if defer_confirmation:
            # DetectionWriter reads this and submits a ConfirmationTask
            # (with the freshly-assigned detection_id) to the queue.
            metadata["needs_iq_confirmation"] = True

        self._detections_emitted += 1
        await self._event_bus.publish(
            DetectionEvent(
                session_id=self._session_id,
                detector_name=self.name,
                technology=technology,
                freq_hz=event.freq_center_hz,
                bandwidth_hz=event.matched_template_hz,
                confidence=confidence,
                evidence="; ".join(evidence_parts),
                hand_off_tools=list(self.capabilities.hand_off_tools),
                metadata=metadata,
            )
        )
        log.info(
            "LoRa-family detection fired at %.3f MHz: tech=%s, "
            "BW=%d kHz, SF=%s, variant=%s, conf=%.2f%s",
            event.freq_center_hz / 1e6,
            technology,
            bw_khz,
            sf_estimate,
            variant_label,
            confidence,
            " (confirmation deferred)" if defer_confirmation else "",
        )


# ----------------------------------------------------------------
# Spreading-factor classification
# ----------------------------------------------------------------


def _estimate_sf_from_slope(
    *,
    slope_hz_per_sec: float,
    bandwidth_hz: int,
) -> int | None:
    """Estimate LoRa spreading factor from observed chirp slope.

    For a LoRa chirp: slope = BW² / 2^SF
    Solving for SF: SF = log2(BW² / slope)

    Returns integer SF rounded to nearest, clamped to [5, 12], or None
    if the slope is implausibly low/high (> 1 SF outside the valid
    range, which usually means the chirp analysis picked up something
    that isn't actually LoRa).
    """
    import math

    if slope_hz_per_sec <= 0 or bandwidth_hz <= 0:
        return None
    # Take absolute value — up-chirps and down-chirps have opposite
    # signs; we only care about magnitude.
    slope_abs = abs(slope_hz_per_sec)
    try:
        sf_float = math.log2((bandwidth_hz ** 2) / slope_abs)
    except (ValueError, ZeroDivisionError):
        return None
    sf_int = round(sf_float)
    if sf_int < 4 or sf_int > 13:
        # Implausibly outside LoRa's valid SF range
        return None
    # Clamp to the canonical range
    return max(5, min(12, sf_int))


def _label_variant(*, sf: int, bandwidth_hz: int) -> str | None:
    """Map (SF, bandwidth) to a human-readable variant label where one
    is recognizable. Otherwise None (detection is still reported as
    generic LoRa with the numeric SF in metadata).

    Meshtastic defaults (US region, LoRaPHY_MT):
      • ShortTurbo     SF7  / 500 kHz
      • ShortFast      SF7  / 250 kHz
      • ShortSlow      SF8  / 250 kHz
      • MediumFast     SF9  / 250 kHz
      • MediumSlow     SF10 / 250 kHz
      • LongFast       SF11 / 250 kHz   (default / most common)
      • LongModerate   SF11 / 125 kHz
      • LongSlow       SF12 / 125 kHz

    LoRaWAN US uplink SF7-10/125kHz is common; the bands overlap with
    Meshtastic at (SF7-10, 125kHz). We prefer the Meshtastic label when
    the SF matches a unique default (SF11/250 is unambiguously LongFast
    Meshtastic; SF9/250 is unambiguously MediumFast).
    """
    bw_khz = bandwidth_hz // 1000
    # Meshtastic-distinctive combinations (bw=250 kHz):
    if bw_khz == 250:
        if sf == 7:
            return "meshtastic_short_fast"
        if sf == 8:
            return "meshtastic_short_slow"
        if sf == 9:
            return "meshtastic_medium_fast"
        if sf == 10:
            return "meshtastic_medium_slow"
        if sf == 11:
            return "meshtastic_long_fast"
    if bw_khz == 500 and sf == 7:
        return "meshtastic_short_turbo"
    if bw_khz == 125:
        if sf == 11:
            return "meshtastic_long_moderate_or_lorawan"
        if sf == 12:
            return "meshtastic_long_slow_or_lorawan"
        # Lower SF at 125 kHz is most commonly LoRaWAN
        if 7 <= sf <= 10:
            return f"lorawan_sf{sf}"
    return None
