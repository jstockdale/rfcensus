"""HackRF enumeration via `hackrf_info`.

HackRF One covers 1 MHz to 6 GHz with 20 MHz instantaneous bandwidth and
8-bit samples. It's also capable of transmitting, which we don't use but
flag in capabilities. Enumeration is done via `hackrf_info` which prints
serial numbers and board details.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rfcensus.hardware.dongle import Dongle, DongleCapabilities, DongleStatus
from rfcensus.utils.async_subprocess import (
    BinaryNotFoundError,
    ManagedProcess,
    ProcessConfig,
    which,
)
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


_HACKRF_CAPABILITIES = DongleCapabilities(
    freq_range_hz=(1_000_000, 6_000_000_000),
    max_sample_rate=20_000_000,
    bits_per_sample=8,
    bias_tee_capable=True,
    tcxo_ppm=10.0,  # Stock HackRF clock; users often upgrade to external reference
    can_transmit=True,
    can_share_via_rtl_tcp=False,
    wide_scan_capable=True,
)


@dataclass
class HackRfProbeResult:
    dongles: list[Dongle]
    diagnostic: str


async def probe_hackrf() -> HackRfProbeResult:
    """Enumerate attached HackRF devices via `hackrf_info`."""
    if which("hackrf_info") is None:
        return HackRfProbeResult(
            dongles=[],
            diagnostic="hackrf_info not found; install hackrf tools to use HackRF hardware",
        )

    proc = ManagedProcess(
        ProcessConfig(
            name="hackrf_info",
            args=["hackrf_info"],
            log_stderr=False,
            kill_timeout_s=5.0,
        )
    )
    try:
        await proc.start()
    except BinaryNotFoundError as exc:
        return HackRfProbeResult(dongles=[], diagnostic=str(exc))

    lines: list[str] = []
    try:
        async for line in proc.stdout_lines():
            lines.append(line)
            if len(lines) > 100:
                break
    finally:
        await proc.stop()

    dongles = _parse_hackrf_info(lines)
    if not dongles:
        return HackRfProbeResult(
            dongles=[],
            diagnostic=(
                "hackrf_info ran but reported no devices. Check USB connection and "
                "permissions (udev rules)."
            ),
        )
    return HackRfProbeResult(
        dongles=dongles,
        diagnostic=f"found {len(dongles)} HackRF device(s)",
    )


# hackrf_info output format (one block per device):
#
# hackrf_info version: ...
# libhackrf version: ...
#
# Found HackRF
# Index: 0
# Serial number: 0000000000000000abcdef1234567890
# Board ID Number: 2 (HackRF One)
# Firmware Version: 2024.02.1
# Part ID Number: 0xa000cb3c 0x00554f49
#
# We extract Index + Serial number + Board ID.

_INDEX_RE = re.compile(r"^Index:\s+(\d+)")
_SERIAL_RE = re.compile(r"^Serial number:\s+(\S+)")
_BOARD_RE = re.compile(r"^Board ID Number:\s+\d+\s+\((.+?)\)")


def _parse_hackrf_info(lines: list[str]) -> list[Dongle]:
    dongles: list[Dongle] = []
    current: dict[str, str] = {}

    def flush() -> None:
        if "serial" not in current:
            return
        serial = current["serial"]
        index = int(current.get("index", "0"))
        board = current.get("board", "HackRF One")
        dongle_id = f"hackrf-{serial[-8:]}"
        dongles.append(
            Dongle(
                id=dongle_id,
                serial=serial,
                model=_classify_board(board),
                driver="hackrf",
                capabilities=_HACKRF_CAPABILITIES,
                status=DongleStatus.DETECTED,
                driver_index=index,
            )
        )

    for line in lines:
        line = line.strip()
        if line.startswith("Found HackRF"):
            flush()
            current = {}
            continue
        m = _INDEX_RE.match(line)
        if m:
            current["index"] = m.group(1)
            continue
        m = _SERIAL_RE.match(line)
        if m:
            current["serial"] = m.group(1)
            continue
        m = _BOARD_RE.match(line)
        if m:
            current["board"] = m.group(1)
            continue

    flush()
    return dongles


def _classify_board(board: str) -> str:
    board_lower = board.lower()
    if "one" in board_lower:
        return "hackrf_one"
    if "jawbreaker" in board_lower:
        return "hackrf_jawbreaker"
    return "hackrf_generic"
