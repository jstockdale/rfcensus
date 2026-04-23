"""RTL-SDR enumeration via `rtl_test` / `rtl_eeprom`.

We deliberately don't link against librtlsdr via FFI. Shelling out to the
standard CLI tools means rfcensus works with whatever librtlsdr happens
to be installed (v0.6, blog/v4 fork, etc.) without us having to handle
every ABI. Probes are cheap – a couple of hundred milliseconds per call.
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

# Model defaults – these are informed by the device product data sheets
# and are used when we can only identify the chip, not the specific
# board variant.
_MODEL_DEFAULTS: dict[str, DongleCapabilities] = {
    "rtlsdr_generic": DongleCapabilities(
        freq_range_hz=(24_000_000, 1_766_000_000),
        max_sample_rate=3_200_000,
        bits_per_sample=8,
        bias_tee_capable=False,
        tcxo_ppm=50.0,  # Worst case for no-TCXO boards
    ),
    "rtlsdr_v3": DongleCapabilities(
        freq_range_hz=(500_000, 1_766_000_000),  # HF via direct sampling
        max_sample_rate=3_200_000,
        bits_per_sample=8,
        bias_tee_capable=True,
        tcxo_ppm=1.0,
    ),
    "rtlsdr_v4": DongleCapabilities(
        freq_range_hz=(500_000, 1_766_000_000),
        max_sample_rate=3_200_000,
        bits_per_sample=8,
        bias_tee_capable=True,
        tcxo_ppm=0.5,
    ),
    "nesdr_nano3": DongleCapabilities(
        freq_range_hz=(25_000_000, 1_700_000_000),
        max_sample_rate=3_200_000,
        bits_per_sample=8,
        bias_tee_capable=False,
        tcxo_ppm=0.5,
    ),
    "nesdr_smart_v5": DongleCapabilities(
        freq_range_hz=(25_000_000, 1_700_000_000),
        max_sample_rate=3_200_000,
        bits_per_sample=8,
        bias_tee_capable=True,
        tcxo_ppm=0.5,
    ),
}


@dataclass
class RtlSdrProbeResult:
    dongles: list[Dongle]
    diagnostic: str


async def probe_rtlsdr() -> RtlSdrProbeResult:
    """Enumerate all attached RTL-SDR dongles.

    Uses `rtl_test -t` with a short timeout. Returns dongles with serials,
    detected chip/tuner names, and best-guess capabilities.
    """
    if which("rtl_test") is None:
        return RtlSdrProbeResult(
            dongles=[],
            diagnostic="rtl_test not found; install rtl-sdr / librtlsdr-bin",
        )

    proc = ManagedProcess(
        ProcessConfig(
            name="rtl_test",
            args=["rtl_test", "-t"],
            log_stderr=False,
            kill_timeout_s=2.0,
        )
    )
    try:
        await proc.start()
    except BinaryNotFoundError as exc:
        return RtlSdrProbeResult(dongles=[], diagnostic=str(exc))

    lines: list[str] = []
    try:
        # rtl_test -t runs briefly and exits. Collect until EOF.
        async for line in proc.stdout_lines():
            lines.append(line)
            # rtl_test can hang trying to read from a dongle even with -t;
            # stop early if we've seen what we need.
            if len(lines) > 200:
                break
    finally:
        await proc.stop()

    # Also dump stderr explicitly since rtl_test writes the device list there
    # on some versions. We re-run with stderr captured to stdout to be safe.
    combined_proc = ManagedProcess(
        ProcessConfig(
            name="rtl_test-combined",
            args=["sh", "-c", "rtl_test -t 2>&1"],
            log_stderr=False,
            kill_timeout_s=3.0,
        )
    )
    try:
        await combined_proc.start()
        combined_lines: list[str] = []
        async for line in combined_proc.stdout_lines():
            combined_lines.append(line)
            if len(combined_lines) > 200:
                break
    except BinaryNotFoundError:
        combined_lines = lines
    finally:
        await combined_proc.stop()

    all_lines = lines + combined_lines
    dongles = _parse_rtl_test_output(all_lines)

    if not dongles:
        return RtlSdrProbeResult(
            dongles=[],
            diagnostic=(
                "rtl_test ran but found no devices. Is a dongle plugged in? "
                "On Linux, make sure udev rules are installed and the dvb_usb_rtl28xxu "
                "kernel module is blacklisted."
            ),
        )

    return RtlSdrProbeResult(
        dongles=dongles,
        diagnostic=f"found {len(dongles)} RTL-SDR device(s)",
    )


# ------------------------------------------------------------
# Output parsing
# ------------------------------------------------------------
#
# rtl_test -t produces output roughly like:
#
#   Found 3 device(s):
#     0:  RTLSDRBlog, Blog V4, SN: 00000001
#     1:  Realtek, RTL2838UHIDIR, SN: 00000043
#     2:  Nooelec, NESDR SMArt v5, SN: 07262454
#
#   Using device 0: Generic RTL2832U OEM
#   Found Rafael Micro R820T tuner
#   Supported gain values ...
#
# We parse the numbered device list. The "Using device 0" line reflects
# probe-time selection only and doesn't give us per-device details, so
# we rely on the vendor+product string to guess capabilities.

_DEVICE_LINE_RE = re.compile(
    r"^\s*(\d+):\s+(.+?),\s+(.+?),\s+SN:\s+(\S+)\s*$"
)


def _parse_rtl_test_output(lines: list[str]) -> list[Dongle]:
    dongles: list[Dongle] = []
    seen_indices: set[int] = set()
    for line in lines:
        match = _DEVICE_LINE_RE.match(line)
        if not match:
            continue
        index = int(match.group(1))
        if index in seen_indices:
            continue
        seen_indices.add(index)
        vendor = match.group(2).strip()
        product = match.group(3).strip()
        serial = match.group(4).strip()
        model = _classify_model(vendor, product)
        caps = _MODEL_DEFAULTS.get(model, _MODEL_DEFAULTS["rtlsdr_generic"])
        # Build a tentative id; we'll de-duplicate below if multiple
        # dongles share the same serial (common with cheap unprogrammed
        # boards — they all ship as 00000001 from the factory).
        if serial and serial != "00000000":
            tentative_id = f"rtlsdr-{serial}"
        else:
            tentative_id = f"rtlsdr-idx{index}"
        dongles.append(
            Dongle(
                id=tentative_id,
                serial=serial or None,
                model=model,
                driver="rtlsdr",
                capabilities=caps,
                status=DongleStatus.DETECTED,
                driver_index=index,
            )
        )

    # Disambiguate any duplicate ids by appending the USB enumeration
    # index. Two boards with serial "00000001" become
    # "rtlsdr-00000001-idx0" and "rtlsdr-00000001-idx1". The driver_index
    # is stable per-boot but may change if you unplug/replug, so this is
    # a workaround — the proper fix is `rtl_eeprom -d N -s NEW_SERIAL`.
    return _disambiguate_duplicate_ids(dongles)


def _disambiguate_duplicate_ids(dongles: list[Dongle]) -> list[Dongle]:
    """Append `-idx{N}` to any dongle whose id collides with another's."""
    counts: dict[str, int] = {}
    for d in dongles:
        counts[d.id] = counts.get(d.id, 0) + 1
    for d in dongles:
        if counts.get(d.id, 0) > 1 and d.driver_index is not None:
            d.id = f"{d.id}-idx{d.driver_index}"
    return dongles


def _classify_model(vendor: str, product: str) -> str:
    text = f"{vendor} {product}".lower()
    if "rtlsdrblog" in text and "v4" in text:
        return "rtlsdr_v4"
    if "rtlsdrblog" in text or "blog v3" in text:
        return "rtlsdr_v3"
    if "nesdr" in text and "nano 3" in text:
        return "nesdr_nano3"
    if "nesdr" in text and "smart" in text and "v5" in text:
        return "nesdr_smart_v5"
    if "nesdr" in text and "smart" in text:
        return "nesdr_smart_v5"
    if "nesdr" in text:
        return "nesdr_nano3"
    return "rtlsdr_generic"
