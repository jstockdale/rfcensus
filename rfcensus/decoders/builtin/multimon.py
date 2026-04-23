"""multimon-ng decoder.

multimon-ng decodes a variety of narrowband formats from demodulated
audio: POCSAG (512/1200/2400), FLEX, APRS AX.25, DTMF, EAS, and more.

multimon-ng doesn't talk to SDR hardware directly. We pipe IQ from
`rtl_fm` (or equivalent) through it. This means for each frequency we
want to monitor, we need a separate (rtl_fm -> multimon-ng) pipeline.
In the current broker model we use the exclusive rtl_fm path; shared
access via rtl_tcp is possible but awkward and we defer that.
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


class MultimonDecoder(DecoderBase):
    capabilities = DecoderCapabilities(
        name="multimon",
        protocols=["pocsag", "flex", "aprs", "dtmf", "eas"],
        freq_ranges=(
            (130_000_000, 175_000_000),
            (440_000_000, 470_000_000),
            (929_000_000, 932_000_000),
        ),
        min_sample_rate=22_050,
        preferred_sample_rate=22_050,
        requires_exclusive_dongle=True,
        external_binary="multimon-ng",
        cpu_cost="moderate",
        description="Decodes POCSAG, FLEX, APRS, and other narrowband modes from NFM audio",
    )

    async def check_available(self) -> DecoderAvailability:
        multimon = which(self.settings.binary or "multimon-ng")
        rtl_fm = which("rtl_fm")
        if multimon is None:
            return DecoderAvailability(
                name=self.name,
                available=False,
                reason="multimon-ng not on PATH",
            )
        if rtl_fm is None:
            return DecoderAvailability(
                name=self.name,
                available=False,
                reason="rtl_fm not on PATH (install rtl-sdr package)",
            )
        return DecoderAvailability(name=self.name, available=True, binary_path=multimon)

    async def run(self, spec: DecoderRunSpec) -> DecoderResult:
        result = DecoderResult(name=self.name)
        lease = spec.lease
        freq_hz = spec.freq_hz
        rtl_fm = which("rtl_fm")
        multimon = which(self.settings.binary or "multimon-ng")
        if rtl_fm is None or multimon is None:
            result.errors.append("rtl_fm and/or multimon-ng not available")
            result.ended_reason = "binary_missing"
            return result

        index = lease.dongle.driver_index if lease.dongle.driver_index is not None else 0

        # Build the pipeline: rtl_fm demod NFM at 22050 Hz audio | multimon-ng
        rtl_fm_args = [
            rtl_fm,
            "-d", str(index),
            "-f", str(freq_hz),
            "-M", "fm",
            "-s", "22050",
            "-g", "40",
            "-l", "0",
            "-E", "deemp",
            "-",
        ]
        multimon_args = [
            multimon,
            "-t", "raw",
            "-a", "POCSAG512",
            "-a", "POCSAG1200",
            "-a", "POCSAG2400",
            "-a", "FLEX",
            "-a", "AFSK1200",  # APRS uses AFSK1200
            "-a", "DTMF",
            "-a", "EAS",
            "-f", "alpha",
            "-",
        ]

        # asyncio subprocess piping is awkward; use a shell
        shell_cmd = " ".join(
            [_shell_quote(a) for a in rtl_fm_args]
            + ["2>/dev/null", "|"]
            + [_shell_quote(a) for a in multimon_args]
        )
        proc = ManagedProcess(
            ProcessConfig(
                name=f"multimon[{lease.dongle.id}@{freq_hz}]",
                args=["sh", "-c", shell_cmd],
                log_stderr=True,
                stderr_log_level="DEBUG",
            )
        )
        try:
            await proc.start()
        except BinaryNotFoundError as exc:
            log.warning(
                "multimon-ng NOT INSTALLED or not on PATH: %s. "
                "Install via `apt install multimon-ng` (Debian/Ubuntu) or "
                "`brew install multimon-ng` (macOS). Note multimon also "
                "requires rtl_fm to be available. Skipping multimon for "
                "this band.",
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
                event = _parse_line(
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


# multimon-ng output lines look like:
#
#   POCSAG1200: Address: 123456  Function: 1  Alpha: HELLO WORLD
#   FLEX|2021-04-22 12:00:00|1600/2/C|007.004.054|0000001|ALN|text here
#   APRS: K6ABC-9>APDR16,WIDE1-1:!3748.00N/12226.00W>comment
#   DTMF: 5
#   EAS: ...

_POCSAG_RE = re.compile(r"POCSAG(\d+):\s+Address:\s+(\d+)\s+Function:\s+(\d)(?:\s+(Alpha|Numeric|Short):\s*(.*))?")
_FLEX_RE = re.compile(r"^FLEX\|([^|]*)\|([^|]*)\|([^|]*)\|(\d+)\|(\w+)\|(.*)$")
_APRS_RE = re.compile(r"^APRS:\s+(.+)$")
_DTMF_RE = re.compile(r"^DTMF:\s+(\S+)$")


def _parse_line(
    line: str,
    *,
    freq_hz: int,
    dongle_id: str,
    session_id: int,
    decoder_name: str,
) -> DecodeEvent | None:
    line = line.strip()
    if not line:
        return None

    now = datetime.now(timezone.utc)

    m = _POCSAG_RE.match(line)
    if m:
        baud, address, function, fmt_kind, payload = m.groups()
        return DecodeEvent(
            session_id=session_id,
            decoder_name=decoder_name,
            protocol="pocsag",
            dongle_id=dongle_id,
            freq_hz=freq_hz,
            payload={
                "baud": int(baud),
                "address": address,
                "function": int(function),
                "format": fmt_kind or "",
                "message": payload or "",
                "_device_id": address,
            },
            timestamp=now,
        )

    m = _FLEX_RE.match(line)
    if m:
        ts, protocol_info, freq_info, capcode, msg_kind, payload = m.groups()
        return DecodeEvent(
            session_id=session_id,
            decoder_name=decoder_name,
            protocol="flex",
            dongle_id=dongle_id,
            freq_hz=freq_hz,
            payload={
                "timestamp": ts,
                "protocol_info": protocol_info,
                "frame": freq_info,
                "capcode": capcode,
                "kind": msg_kind,
                "message": payload,
                "_device_id": capcode,
            },
            timestamp=now,
        )

    m = _APRS_RE.match(line)
    if m:
        raw = m.group(1)
        call = raw.split(">", 1)[0].strip()
        return DecodeEvent(
            session_id=session_id,
            decoder_name=decoder_name,
            protocol="aprs",
            dongle_id=dongle_id,
            freq_hz=freq_hz,
            payload={
                "callsign": call,
                "packet": raw,
                "_device_id": call,
            },
            timestamp=now,
        )

    m = _DTMF_RE.match(line)
    if m:
        return DecodeEvent(
            session_id=session_id,
            decoder_name=decoder_name,
            protocol="dtmf",
            dongle_id=dongle_id,
            freq_hz=freq_hz,
            payload={"digit": m.group(1), "_device_id": None},
            timestamp=now,
        )

    return None


def _shell_quote(s: str) -> str:
    """Minimal shell quoting for building a pipeline command."""
    if not s:
        return "''"
    if any(c in s for c in " '\"\\|&;<>()$`\n\t"):
        return "'" + s.replace("'", "'\\''") + "'"
    return s
