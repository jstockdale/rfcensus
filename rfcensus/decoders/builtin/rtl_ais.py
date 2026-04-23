"""rtl-ais decoder.

rtl-ais decodes AIS (marine vessel tracking) broadcasts on 161.975 and
162.025 MHz simultaneously. Output is standard NMEA 0183 AIS sentences;
we parse just enough to extract MMSI for emitter tracking.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

from rfcensus.decoders.base import (
    DecoderAvailability,
    DecoderBase,
    DecoderCapabilities,
    DecoderResult,
    DecoderRunSpec,
)
from rfcensus.events import DecodeEvent, EventBus
from rfcensus.hardware.broker import DongleLease
from rfcensus.utils.async_subprocess import (
    BinaryNotFoundError,
    ManagedProcess,
    ProcessConfig,
    which,
)
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


class RtlAisDecoder(DecoderBase):
    capabilities = DecoderCapabilities(
        name="rtl_ais",
        protocols=["ais_class_a", "ais_class_b"],
        freq_ranges=((161_900_000, 162_100_000),),
        min_sample_rate=256_000,
        preferred_sample_rate=256_000,
        requires_exclusive_dongle=True,
        external_binary="rtl_ais",
        cpu_cost="cheap",
        description="Decodes AIS maritime broadcasts on 161.975 + 162.025 MHz",
    )

    async def check_available(self) -> DecoderAvailability:
        binary = self.settings.binary or "rtl_ais"
        path = which(binary)
        if path is None:
            return DecoderAvailability(
                name=self.name,
                available=False,
                reason=f"{binary} not on PATH (install from github.com/dgiardini/rtl-ais)",
            )
        return DecoderAvailability(name=self.name, available=True, binary_path=path)

    async def run(self, spec: DecoderRunSpec) -> DecoderResult:
        lease = spec.lease
        binary = self.settings.binary or "rtl_ais"
        # rtl_ais picks its own sample rate and frequencies; -n emits NMEA on stdout
        args = [binary, "-n"]
        if lease.dongle.driver_index is not None:
            args += ["-d", str(lease.dongle.driver_index)]
        args += list(self.settings.extra_args)

        result = DecoderResult(name=self.name)

        proc = ManagedProcess(
            ProcessConfig(
                name=f"rtl_ais[{lease.dongle.id}]",
                args=args,
                log_stderr=True,
                stderr_log_level="DEBUG",
            )
        )
        try:
            await proc.start()
        except BinaryNotFoundError as exc:
            log.warning(
                "rtl_ais NOT INSTALLED or not on PATH: %s. "
                "Install via `apt install rtl-ais` (Debian/Ubuntu), "
                "`brew install rtl-ais` (macOS), or build from "
                "https://github.com/dgiardini/rtl-ais . Skipping "
                "rtl_ais for this band.",
                exc,
            )
            result.errors.append(str(exc))
            result.ended_reason = "binary_missing"
            return result

        # rtl_ais has no built-in -T flag; enforce duration ourselves
        stop_task: asyncio.Task | None = None
        if spec.duration_s is not None:
            async def _timeout_kill():
                await asyncio.sleep(spec.duration_s)
                await proc.stop()
            stop_task = asyncio.create_task(_timeout_kill())

        try:
            async for line in proc.stdout_lines():
                event = _parse_nmea(
                    line,
                    freq_hz=spec.freq_hz,
                    dongle_id=lease.dongle.id,
                    session_id=spec.session_id,
                    decoder_name=self.name,
                )
                if event is not None:
                    await spec.event_bus.publish(event)
                    result.decodes_emitted += 1
        finally:
            if stop_task and not stop_task.done():
                stop_task.cancel()
            await proc.stop()
        return result


_NMEA_RE = re.compile(r"^!(AIVDM|AIVDO),\d+,\d+,\d*,[AB],([^,]+),\d\*[0-9A-Fa-f]{2}")


def _parse_nmea(
    line: str,
    *,
    freq_hz: int,
    dongle_id: str,
    session_id: int,
    decoder_name: str,
) -> DecodeEvent | None:
    """Minimal NMEA parse: extract MMSI via bit manipulation of the payload."""
    m = _NMEA_RE.match(line)
    if not m:
        return None
    payload_text = m.group(2)
    mmsi = _extract_mmsi(payload_text)
    if mmsi is None:
        return None

    # AIS message type is the first 6 bits of the payload
    msg_type_bits = _sixbit(payload_text[0]) if payload_text else 0
    # Class A typically msg types 1-3, 5; Class B 18, 19, 24
    if msg_type_bits in (18, 19, 24):
        protocol = "ais_class_b"
    else:
        protocol = "ais_class_a"

    return DecodeEvent(
        session_id=session_id,
        decoder_name=decoder_name,
        protocol=protocol,
        dongle_id=dongle_id,
        freq_hz=freq_hz,
        payload={
            "mmsi": mmsi,
            "msg_type": msg_type_bits,
            "_device_id": str(mmsi),
            "raw_nmea": line.strip(),
        },
        timestamp=datetime.now(timezone.utc),
    )


def _sixbit(ch: str) -> int:
    """AIS 6-bit ASCII decode of a single character."""
    v = ord(ch) - 48
    if v > 40:
        v -= 8
    return v & 0x3F


def _extract_mmsi(payload: str) -> int | None:
    """MMSI is a 30-bit field starting at bit 8 of the AIS payload."""
    if len(payload) < 7:
        return None
    bits = 0
    for ch in payload[:7]:  # 7 chars * 6 bits = 42 bits; enough for 8+30
        bits = (bits << 6) | _sixbit(ch)
    # Drop first 8 bits (msg type + repeat indicator), take next 30
    mmsi = (bits >> (42 - 38)) & 0x3FFFFFFF
    if mmsi == 0:
        return None
    return mmsi
