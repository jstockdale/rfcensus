"""multimon-ng decoder.

multimon-ng decodes a variety of narrowband formats from demodulated
audio: POCSAG (512/1200/2400), FLEX, APRS AX.25, DTMF, EAS, and more.

v0.5.37 rewrite: pipe IQ through `rfcensus.tools.fm_bridge` instead of
the exclusive `rtl_fm` binary. The bridge connects to rtl_tcp (via the
broker's fanout), demodulates FM in Python (numpy/scipy), and emits
int16 PCM to stdout that multimon-ng reads from stdin.

This lets multiple decoders share one dongle via rtl_tcp — multimon
can now coexist with direwolf on the APRS band, with rtl_433 on
paging bands, etc. No more "one decoder per physical USB device"
wave packing bottleneck.

Pipeline:

    python -m rfcensus.tools.fm_bridge \\
        --rtl-tcp HOST:PORT \\
        --freq FREQ_HZ \\
        --input-rate 2400000 \\
        --output-rate 22050 \\
        --gain auto
      | multimon-ng -t raw -a POCSAG512 ... -f alpha -

If we're allocated an EXCLUSIVE lease for whatever reason (e.g., the
broker couldn't set up rtl_tcp, or sharing was disabled by config),
fm_bridge can't connect — so exclusive leases fall through to the
legacy rtl_fm path. This keeps old behavior as a safety net and
ensures multimon still works on hardware setups where rtl_tcp fails.
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


class MultimonDecoder(DecoderBase):
    capabilities = DecoderCapabilities(
        name="multimon",
        protocols=["pocsag", "flex", "aprs", "dtmf", "eas"],
        freq_ranges=(
            (130_000_000, 175_000_000),
            (440_000_000, 470_000_000),
            (929_000_000, 932_000_000),
        ),
        # min_sample_rate is what the BROKER-visible sample rate must be:
        # since fm_bridge consumes IQ from rtl_tcp at whatever rate rtl_tcp
        # is running at (default 2.4 Msps), the capability is expressed in
        # IQ-sample-rate terms, NOT the 22050 Hz audio rate that multimon
        # itself sees downstream of fm_bridge. This matters because the
        # broker compares capabilities to DongleCapabilities.
        min_sample_rate=1_024_000,
        preferred_sample_rate=2_400_000,
        # v0.5.37: shared access. fm_bridge connects to rtl_tcp and
        # decodes IQ to audio in Python, so multimon can coexist with
        # other shared-mode decoders (direwolf, rtl_433, rtlamr) on
        # one dongle.
        requires_exclusive_dongle=False,
        external_binary="multimon-ng",
        cpu_cost="moderate",
        description=(
            "Decodes POCSAG, FLEX, APRS, and other narrowband modes "
            "from NFM audio (uses fm_bridge + rtl_tcp for sharing)"
        ),
    )

    async def check_available(self) -> DecoderAvailability:
        multimon = which(self.settings.binary or "multimon-ng")
        if multimon is None:
            return DecoderAvailability(
                name=self.name,
                available=False,
                reason="multimon-ng not on PATH",
            )
        # fm_bridge is provided by rfcensus itself (numpy/scipy);
        # no external binary beyond multimon is required for the
        # shared-access path. If this lease turns out to be exclusive
        # (no rtl_tcp), we'll fall back to rtl_fm, which is checked
        # at run() time.
        return DecoderAvailability(
            name=self.name, available=True, binary_path=multimon
        )

    async def run(self, spec: DecoderRunSpec) -> DecoderResult:
        result = DecoderResult(name=self.name)
        lease = spec.lease
        freq_hz = spec.freq_hz
        multimon = which(self.settings.binary or "multimon-ng")
        if multimon is None:
            result.errors.append("multimon-ng not available")
            result.ended_reason = "binary_missing"
            return result

        endpoint = lease.endpoint()
        if endpoint is not None:
            # Shared path: fm_bridge | multimon
            shell_cmd = _build_shared_shell_cmd(
                endpoint=endpoint,
                freq_hz=freq_hz,
                input_rate=spec.sample_rate,
                gain=spec.gain or "auto",
                multimon_binary=multimon,
            )
            proc_name = f"multimon[shared@{lease.dongle.id}@{freq_hz}]"
        else:
            # Exclusive path: rtl_fm | multimon (legacy)
            rtl_fm = which("rtl_fm")
            if rtl_fm is None:
                result.errors.append(
                    "rtl_fm not available for exclusive multimon path"
                )
                result.ended_reason = "binary_missing"
                return result
            shell_cmd = _build_exclusive_shell_cmd(
                rtl_fm_binary=rtl_fm,
                driver_index=lease.dongle.driver_index or 0,
                freq_hz=freq_hz,
                multimon_binary=multimon,
            )
            proc_name = f"multimon[exclusive@{lease.dongle.id}@{freq_hz}]"

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
                "multimon-ng NOT INSTALLED or not on PATH: %s. "
                "Install via `apt install multimon-ng` (Debian/Ubuntu) or "
                "`brew install multimon-ng` (macOS). Skipping multimon for "
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


# ----------------------------------------------------------------
# Shell pipeline builders
# ----------------------------------------------------------------


_MULTIMON_AUDIO_RATE = 22_050


def _build_shared_shell_cmd(
    *,
    endpoint: tuple[str, int],
    freq_hz: int,
    input_rate: int,
    gain: str,
    multimon_binary: str,
) -> str:
    """Build the shared-access pipeline:
        python -m rfcensus.tools.fm_bridge ... | multimon-ng ... -

    Uses sys.executable to ensure we use the same Python interpreter
    rfcensus is running under (the `rfcensus.tools` module must be
    importable, so it needs to be the same env).
    """
    host, port = endpoint
    python = sys.executable
    fm_bridge_args = [
        python, "-m", "rfcensus.tools.fm_bridge",
        "--rtl-tcp", f"{host}:{port}",
        "--freq", str(freq_hz),
        "--input-rate", str(input_rate),
        "--output-rate", str(_MULTIMON_AUDIO_RATE),
        "--gain", gain,
    ]
    multimon_args = [
        multimon_binary,
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
    return " ".join(
        [_shell_quote(a) for a in fm_bridge_args]
        + ["2>/dev/null", "|"]
        + [_shell_quote(a) for a in multimon_args]
    )


def _build_exclusive_shell_cmd(
    *,
    rtl_fm_binary: str,
    driver_index: int,
    freq_hz: int,
    multimon_binary: str,
) -> str:
    """Legacy exclusive-access pipeline using rtl_fm. Retained for
    fallback when shared access (rtl_tcp) isn't available."""
    rtl_fm_args = [
        rtl_fm_binary,
        "-d", str(driver_index),
        "-f", str(freq_hz),
        "-M", "fm",
        "-s", str(_MULTIMON_AUDIO_RATE),
        "-g", "40",
        "-l", "0",
        "-E", "deemp",
        "-",
    ]
    multimon_args = [
        multimon_binary,
        "-t", "raw",
        "-a", "POCSAG512",
        "-a", "POCSAG1200",
        "-a", "POCSAG2400",
        "-a", "FLEX",
        "-a", "AFSK1200",
        "-a", "DTMF",
        "-a", "EAS",
        "-f", "alpha",
        "-",
    ]
    return " ".join(
        [_shell_quote(a) for a in rtl_fm_args]
        + ["2>/dev/null", "|"]
        + [_shell_quote(a) for a in multimon_args]
    )


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
