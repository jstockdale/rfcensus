"""WiFi / Bluetooth 2.4 GHz ISM detector.

The 2.4 GHz ISM band (2400-2500 MHz) is a zoo: WiFi, Bluetooth Classic,
BLE, Zigbee, Thread, proprietary ISM, video senders, microwave oven
leakage, and more. From power samples alone we can't distinguish
individual protocols, so we report activity in this band with hand-off
suggestions rather than attempting classification.

This detector is **informational** — its purpose is to say "2.4 GHz is
busy at your site, here are the specialized tools you might want to
reach for" rather than to identify what specifically is transmitting.

Only HackRF can sweep this range. RTL-SDR tops out at ~1.7 GHz.
"""

from __future__ import annotations

from rfcensus.detectors.base import DetectorBase, DetectorCapabilities
from rfcensus.events import ActiveChannelEvent, DetectionEvent
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


class WifiBtDetector(DetectorBase):
    capabilities = DetectorCapabilities(
        name="wifi_bt_ism",
        detected_technologies=["wifi_2_4ghz", "bluetooth", "ble", "zigbee", "ism_24"],
        relevant_freq_ranges=((2_400_000_000, 2_500_000_000),),
        hand_off_tools=(
            "kismet",           # WiFi
            "ubertooth-rx",     # Bluetooth Classic
            "nrf_sniffer",      # BLE / Zigbee
            "sniffle",          # BLE
            "wireshark",        # general analysis
        ),
        cpu_cost="cheap",
        description=(
            "Flags 2.4 GHz ISM activity. Recommends specialized tools "
            "rather than attempting protocol classification."
        ),
    )

    # 2.4 GHz is busy enough that a single active channel isn't noise.
    MIN_CHANNELS_BEFORE_ANNOUNCE = 2

    def __init__(self) -> None:
        super().__init__()
        self._channels_seen: set[int] = set()
        self._announced: bool = False

    async def on_active_channel(self, event: ActiveChannelEvent) -> None:
        if event.kind == "gone":
            return

        self._channels_seen.add(event.freq_center_hz)
        if self._announced:
            return
        if len(self._channels_seen) < self.MIN_CHANNELS_BEFORE_ANNOUNCE:
            return

        self._announced = True
        if self._event_bus is None:
            return

        channels = sorted(self._channels_seen)
        evidence = (
            f"{len(channels)} active channels detected in 2.4 GHz ISM "
            f"(span: {channels[0]/1e6:.0f}-{channels[-1]/1e6:.0f} MHz). "
            f"Likely a mix of WiFi, Bluetooth, BLE, Zigbee, or other ISM devices."
        )
        self._detections_emitted += 1
        await self._event_bus.publish(
            DetectionEvent(
                session_id=self._session_id,
                detector_name=self.name,
                technology="ism_24",
                freq_hz=channels[0],
                bandwidth_hz=event.bandwidth_hz,
                confidence=0.6,  # We're confident activity exists, not what it is
                evidence=evidence,
                hand_off_tools=list(self.capabilities.hand_off_tools),
                metadata={
                    "channels_seen": channels,
                    "note": "2.4 GHz activity indicated; use specialized tools for protocol ID",
                },
            )
        )
        log.info("2.4 GHz ISM activity detection fired: %d channels", len(channels))
