"""direwolf APRS decoder.

direwolf is a higher-quality AFSK1200/AX.25/APRS decoder than
multimon-ng on marginal signals. For APRS specifically we prefer
direwolf if it's installed.

Like multimon, direwolf ingests demodulated audio; we feed it from
rtl_fm. Output is parsed from direwolf's text log format.
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


class DirewolfDecoder(DecoderBase):
    capabilities = DecoderCapabilities(
        name="direwolf",
        protocols=["aprs"],
        freq_ranges=(
            (144_300_000, 144_500_000),  # APRS 2m NA
            (144_700_000, 144_900_000),  # APRS 2m EU (144.8)
        ),
        min_sample_rate=22_050,
        preferred_sample_rate=48_000,
        requires_exclusive_dongle=True,
        external_binary="direwolf",
        cpu_cost="moderate",
        description="High-quality APRS AX.25 decoder via AFSK1200",
    )

    async def check_available(self) -> DecoderAvailability:
        direwolf = which(self.settings.binary or "direwolf")
        rtl_fm = which("rtl_fm")
        if direwolf is None:
            return DecoderAvailability(
                name=self.name,
                available=False,
                reason="direwolf not on PATH",
            )
        if rtl_fm is None:
            return DecoderAvailability(
                name=self.name,
                available=False,
                reason="rtl_fm not on PATH (install rtl-sdr package)",
            )
        return DecoderAvailability(name=self.name, available=True, binary_path=direwolf)

    async def run(self, spec: DecoderRunSpec) -> DecoderResult:
        result = DecoderResult(name=self.name)
        lease = spec.lease
        freq_hz = spec.freq_hz
        rtl_fm = which("rtl_fm")
        direwolf = which(self.settings.binary or "direwolf")
        if rtl_fm is None or direwolf is None:
            result.errors.append("rtl_fm or direwolf not available")
            result.ended_reason = "binary_missing"
            return result

        index = lease.dongle.driver_index if lease.dongle.driver_index is not None else 0
        # Audio path: rtl_fm -> direwolf (reading from stdin)
        rtl_fm_args = [
            rtl_fm, "-d", str(index), "-f", str(freq_hz),
            "-M", "fm", "-s", "48000", "-g", "40",
            "-l", "0", "-E", "deemp", "-",
        ]
        direwolf_args = [
            direwolf,
            "-c", "/dev/null",  # Don't pick up a user direwolf.conf
            "-r", "48000",
            "-B", "1200",
            "-t", "0",  # No color codes
            "-q", "h",  # Suppress heard-direct logging
            "-",
        ]

        shell_cmd = " ".join(
            [_shell_quote(a) for a in rtl_fm_args]
            + ["2>/dev/null", "|"]
            + [_shell_quote(a) for a in direwolf_args]
        )
        proc = ManagedProcess(
            ProcessConfig(
                name=f"direwolf[{lease.dongle.id}@{freq_hz}]",
                args=["sh", "-c", shell_cmd],
                log_stderr=True,
                stderr_log_level="DEBUG",
            )
        )
        try:
            await proc.start()
        except BinaryNotFoundError as exc:
            log.warning(
                "direwolf NOT INSTALLED or not on PATH: %s. "
                "Install via `apt install direwolf` (Debian/Ubuntu) or "
                "`brew install direwolf` (macOS). Skipping direwolf "
                "for this band.",
                exc,
            )
            result.errors.append(str(exc))
            result.ended_reason = "binary_missing"
            return result

        stop_task: asyncio.Task | None = None
        if spec.duration_s is not None:
            async def _timeout_kill():
                await asyncio.sleep(spec.duration_s)
                await proc.stop()
            stop_task = asyncio.create_task(_timeout_kill())

        try:
            async for line in proc.stdout_lines():
                event = _parse_direwolf(
                    line,
                    freq_hz=freq_hz,
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


# direwolf prints APRS frames as:
#
#   [0] CALL-SSID>APRS,PATH:payload
#
# followed by detail lines. We parse the header line only.

_FRAME_RE = re.compile(r"^\[\d+\]\s+([A-Z0-9\-]+)\s*>\s*([A-Z0-9\-]+)(?:,([^:]*))?:(.*)$")


def _parse_direwolf(
    line: str,
    *,
    freq_hz: int,
    dongle_id: str,
    session_id: int,
    decoder_name: str,
) -> DecodeEvent | None:
    m = _FRAME_RE.match(line.strip())
    if not m:
        return None
    source, dest, path, payload = m.groups()
    return DecodeEvent(
        session_id=session_id,
        decoder_name=decoder_name,
        protocol="aprs",
        dongle_id=dongle_id,
        freq_hz=freq_hz,
        payload={
            "source": source,
            "destination": dest,
            "path": path or "",
            "message": payload.strip(),
            "_device_id": source,
        },
        timestamp=datetime.now(timezone.utc),
    )


def _shell_quote(s: str) -> str:
    if not s:
        return "''"
    if any(c in s for c in " '\"\\|&;<>()$`\n\t"):
        return "'" + s.replace("'", "'\\''") + "'"
    return s
