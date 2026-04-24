"""direwolf APRS decoder.

direwolf is a higher-quality AFSK1200/AX.25/APRS decoder than
multimon-ng on marginal signals. For APRS specifically we prefer
direwolf if it's installed.

v0.5.37 rewrite: pipe IQ through `rfcensus.tools.fm_bridge` instead of
the exclusive `rtl_fm` binary. This lets direwolf share a dongle with
other decoders (including multimon, which also covers aprs_2m).
"""

from __future__ import annotations

import asyncio
import re
import sys
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


# direwolf prefers 48 kHz PCM for best decode quality on marginal
# signals. fm_bridge can emit this without a fractional resample when
# the IQ input is 2.4 Msps (2,400,000 / 48000 = 50, exact integer
# decimation).
_DIREWOLF_AUDIO_RATE = 48_000


class DirewolfDecoder(DecoderBase):
    capabilities = DecoderCapabilities(
        name="direwolf",
        protocols=["aprs"],
        freq_ranges=(
            (144_300_000, 144_500_000),  # APRS 2m NA
            (144_700_000, 144_900_000),  # APRS 2m EU (144.8)
        ),
        # Expressed in IQ terms since fm_bridge consumes IQ; audio rate
        # is 48 kHz internal to the pipeline.
        min_sample_rate=1_024_000,
        preferred_sample_rate=2_400_000,
        # v0.5.37: shared access via fm_bridge + rtl_tcp.
        requires_exclusive_dongle=False,
        external_binary="direwolf",
        cpu_cost="moderate",
        description=(
            "High-quality APRS AX.25 decoder via AFSK1200 (uses "
            "fm_bridge + rtl_tcp for sharing)"
        ),
    )

    async def check_available(self) -> DecoderAvailability:
        direwolf = which(self.settings.binary or "direwolf")
        if direwolf is None:
            return DecoderAvailability(
                name=self.name,
                available=False,
                reason="direwolf not on PATH",
            )
        return DecoderAvailability(
            name=self.name, available=True, binary_path=direwolf
        )

    async def run(self, spec: DecoderRunSpec) -> DecoderResult:
        result = DecoderResult(name=self.name)
        lease = spec.lease
        freq_hz = spec.freq_hz
        direwolf = which(self.settings.binary or "direwolf")
        if direwolf is None:
            result.errors.append("direwolf not available")
            result.ended_reason = "binary_missing"
            return result

        endpoint = lease.endpoint()
        if endpoint is not None:
            # Shared: fm_bridge | direwolf
            shell_cmd = _build_shared_shell_cmd(
                endpoint=endpoint,
                freq_hz=freq_hz,
                input_rate=spec.sample_rate,
                gain=spec.gain or "auto",
                direwolf_binary=direwolf,
            )
            proc_name = f"direwolf[shared@{lease.dongle.id}@{freq_hz}]"
        else:
            # Exclusive fallback
            rtl_fm = which("rtl_fm")
            if rtl_fm is None:
                result.errors.append(
                    "rtl_fm not available for exclusive direwolf path"
                )
                result.ended_reason = "binary_missing"
                return result
            shell_cmd = _build_exclusive_shell_cmd(
                rtl_fm_binary=rtl_fm,
                driver_index=lease.dongle.driver_index or 0,
                freq_hz=freq_hz,
                direwolf_binary=direwolf,
            )
            proc_name = f"direwolf[exclusive@{lease.dongle.id}@{freq_hz}]"

        proc = ManagedProcess(
            ProcessConfig(
                name=proc_name,
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


# ----------------------------------------------------------------
# Shell pipeline builders
# ----------------------------------------------------------------


def _build_shared_shell_cmd(
    *,
    endpoint: tuple[str, int],
    freq_hz: int,
    input_rate: int,
    gain: str,
    direwolf_binary: str,
) -> str:
    host, port = endpoint
    python = sys.executable
    fm_bridge_args = [
        python, "-m", "rfcensus.tools.fm_bridge",
        "--rtl-tcp", f"{host}:{port}",
        "--freq", str(freq_hz),
        "--input-rate", str(input_rate),
        "--output-rate", str(_DIREWOLF_AUDIO_RATE),
        "--gain", gain,
    ]
    direwolf_args = [
        direwolf_binary,
        "-c", "/dev/null",  # Don't pick up user direwolf.conf
        "-r", str(_DIREWOLF_AUDIO_RATE),
        "-B", "1200",
        "-t", "0",  # No color codes
        "-q", "h",  # Suppress heard-direct logging
        "-",
    ]
    return " ".join(
        [_shell_quote(a) for a in fm_bridge_args]
        + ["2>/dev/null", "|"]
        + [_shell_quote(a) for a in direwolf_args]
    )


def _build_exclusive_shell_cmd(
    *,
    rtl_fm_binary: str,
    driver_index: int,
    freq_hz: int,
    direwolf_binary: str,
) -> str:
    """Legacy exclusive pipeline. Retained for fallback."""
    rtl_fm_args = [
        rtl_fm_binary,
        "-d", str(driver_index),
        "-f", str(freq_hz),
        "-M", "fm",
        "-s", str(_DIREWOLF_AUDIO_RATE),
        "-g", "40",
        "-l", "0",
        "-E", "deemp",
        "-",
    ]
    direwolf_args = [
        direwolf_binary,
        "-c", "/dev/null",
        "-r", str(_DIREWOLF_AUDIO_RATE),
        "-B", "1200",
        "-t", "0",
        "-q", "h",
        "-",
    ]
    return " ".join(
        [_shell_quote(a) for a in rtl_fm_args]
        + ["2>/dev/null", "|"]
        + [_shell_quote(a) for a in direwolf_args]
    )


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
