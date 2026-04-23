"""Hardware health checks.

Runs short diagnostic probes against each detected dongle and reports
status. Used by `rfcensus doctor` and optionally at the start of an
inventory session.

Design notes:

• These are short, one-shot diagnostics. We use plain asyncio subprocess
  with `communicate()` rather than the `ManagedProcess` streaming wrapper,
  because streaming is overkill and the wrapper's lifecycle has multiple
  cleanup paths that fight a `timeout` wrapper.
• We capture **both stdout and stderr** because the failure modes we care
  about most (device busy, permissions, kernel driver loaded) are reported
  on stderr, not stdout.
• We distinguish "broken" from "in use by another process" because they
  have very different remediation.
"""

from __future__ import annotations

import asyncio
import re
import signal
from dataclasses import dataclass
from datetime import datetime, timezone

from rfcensus.hardware.dongle import Dongle, DongleStatus
from rfcensus.utils.async_subprocess import which
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class HealthReport:
    dongle_id: str
    status: DongleStatus
    notes: list[str]
    ppm_estimate: float | None = None


# How long to let rtl_test / hackrf_info run before signalling.
# rtl_test with -p (PPM mode) runs forever; we let it run long enough
# for the tuner identification, gain table, and a few seconds of samples
# to be processed before clean shutdown. The startup sample-loss blip
# usually self-corrects within ~3 seconds, so 8 seconds gives us a
# representative reading.
_PROBE_TIMEOUT_S = 8.0
# After signalling, give the process this long to flush buffers and exit.
_SHUTDOWN_GRACE_S = 5.0
# A tiny amount of sample loss at startup is normal (USB buffer warmup).
# Don't downgrade the dongle for it.
_SAMPLE_LOSS_TOLERANCE_PPM = 10


# ──────────────────────────────────────────────────────────────────
# RTL-SDR health
# ──────────────────────────────────────────────────────────────────


async def check_rtlsdr(dongle: Dongle) -> HealthReport:
    """Run a short rtl_test probe and parse the result.

    Returns a HealthReport with one of these statuses:

    • HEALTHY  — tuner responded, no concerning warnings
    • DEGRADED — tuner responded but USB sample loss or large PPM
    • BUSY     — device is in use by another process / kernel driver is loaded
    • FAILED   — tuner did not respond, hardware fault, or other unrecoverable
    • UNAVAILABLE — rtl_test binary not installed
    """
    if which("rtl_test") is None:
        return HealthReport(
            dongle_id=dongle.id,
            status=DongleStatus.UNAVAILABLE,
            notes=["rtl_test not installed"],
        )

    index = dongle.driver_index if dongle.driver_index is not None else 0

    # Build the command. We don't need stdbuf wrapping — empirically,
    # asyncio's PIPE handling captures rtl_test output fine without it,
    # as long as we capture cleanly via communicate() (see below).
    args = ["rtl_test", "-d", str(index), "-s", "2048000", "-p"]

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        return HealthReport(
            dongle_id=dongle.id,
            status=DongleStatus.FAILED,
            notes=[f"could not start rtl_test: {exc}"],
        )

    # Run the probe. `rtl_test -p` runs forever, so we must signal it
    # to stop. The naive approach — wait_for(communicate(), timeout) —
    # has a subtle bug: when wait_for cancels communicate(), it also
    # cancels the internal stdout/stderr reader tasks, and any data
    # they had buffered gets discarded. The second communicate() call
    # only sees data written AFTER the signal, missing all the early
    # diagnostic output (tuner identification, gain table, etc.).
    #
    # The correct pattern: call communicate() exactly once, and use a
    # separate scheduled callback to send the signal at the right time.
    # That way the readers run continuously and capture everything the
    # process writes, both before and after shutdown.

    loop = asyncio.get_event_loop()
    signal_handle = loop.call_later(
        _PROBE_TIMEOUT_S, _safe_send_signal, proc, signal.SIGINT
    )
    # Force-kill handle: if rtl_test ignores SIGINT for too long, kill it.
    # The total budget is probe time + shutdown grace.
    kill_handle = loop.call_later(
        _PROBE_TIMEOUT_S + _SHUTDOWN_GRACE_S, _safe_kill, proc
    )

    try:
        stdout_bytes, stderr_bytes = await proc.communicate()
    finally:
        signal_handle.cancel()
        kill_handle.cancel()

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    return _interpret_rtl_test(dongle.id, stdout, stderr)


def _safe_send_signal(proc: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    """Send a signal to the process, ignoring if it's already exited."""
    try:
        proc.send_signal(sig)
    except (ProcessLookupError, AttributeError):
        pass


def _safe_kill(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the process if it's still alive, ignoring errors."""
    try:
        proc.kill()
    except (ProcessLookupError, AttributeError):
        pass


def _interpret_rtl_test(dongle_id: str, stdout: str, stderr: str) -> HealthReport:
    """Convert rtl_test output into a HealthReport. Pure function — easy to test."""
    notes: list[str] = []
    ppm: float | None = None

    combined = stdout + "\n" + stderr

    # Hard-failure patterns first — these mean the dongle isn't usable right now,
    # but the *reason* is useful for the user. Order matters: kernel-driver is
    # more specific than generic busy and shares some wording, so check it first.

    if _KERNEL_DRIVER_PATTERN.search(combined):
        return HealthReport(
            dongle_id=dongle_id,
            status=DongleStatus.BUSY,
            notes=[
                "kernel DVB driver is bound to this device",
                "remediation: blacklist dvb_usb_rtl28xxu, or unload it: "
                "`sudo rmmod dvb_usb_rtl28xxu rtl2832 rtl2830`",
                "permanent fix: add to /etc/modprobe.d/blacklist-rtl.conf",
            ],
        )

    if _BUSY_PATTERN.search(combined):
        return HealthReport(
            dongle_id=dongle_id,
            status=DongleStatus.BUSY,
            notes=[
                "device is in use by another process",
                "common causes: SDR# / SDR++ / GQRX / Cubic SDR / dump1090 / "
                "another rfcensus instance / a long-running rtl_tcp",
                "remediation: stop the other process, or `lsof | grep -i rtl` to find it",
            ],
        )

    if _PERMISSIONS_PATTERN.search(combined):
        return HealthReport(
            dongle_id=dongle_id,
            status=DongleStatus.FAILED,
            notes=[
                "USB permissions error opening the device",
                "remediation: install rtl-sdr udev rules and reconnect the dongle, "
                "or add yourself to the `plugdev` group and re-login",
            ],
        )

    if _NO_DEVICE_PATTERN.search(combined):
        return HealthReport(
            dongle_id=dongle_id,
            status=DongleStatus.FAILED,
            notes=[
                "rtl_test reports no supported devices found at this index",
                "the device may have disconnected since enumeration",
            ],
        )

    # Look for the success markers
    has_tuner = bool(_TUNER_FOUND_PATTERN.search(combined))

    # rtl_test prints a final summary line at clean exit:
    #   "Samples per million lost (minimum): N"
    # That's the canonical sample-loss metric — not the per-buffer
    # "lost at least N bytes" lines, which fire during normal startup
    # when USB buffers warm up.
    final_loss_ppm: int | None = None
    final_match = _SAMPLE_LOSS_FINAL_PATTERN.search(combined)
    if final_match:
        try:
            final_loss_ppm = int(final_match.group(1))
        except ValueError:
            pass

    for line in combined.splitlines():
        if "PLL not locked" in line:
            notes.append("PLL not locked warning (often cosmetic at edges of tuning range)")
        if "cumulative PPM" in line:
            try:
                ppm = float(line.split("cumulative PPM:")[1].strip())
            except (IndexError, ValueError):
                pass

    if not has_tuner:
        # Fallback: something went wrong but none of our patterns matched.
        # Surface a couple of stderr lines to help debugging.
        salient = _salient_stderr_lines(stderr)
        return HealthReport(
            dongle_id=dongle_id,
            status=DongleStatus.FAILED,
            notes=["no tuner response from rtl_test"] + salient,
            ppm_estimate=ppm,
        )

    status = DongleStatus.HEALTHY
    if final_loss_ppm is not None and final_loss_ppm > _SAMPLE_LOSS_TOLERANCE_PPM:
        notes.append(
            f"USB sample loss: {final_loss_ppm} ppm — check powered hub, "
            f"USB bandwidth, or use a USB 3.0 port"
        )
        status = DongleStatus.DEGRADED
    if ppm is not None and abs(ppm) > 100:
        notes.append(
            f"large PPM drift ({ppm:+.0f}); may indicate USB issues or thermal warmup"
        )
        status = DongleStatus.DEGRADED

    return HealthReport(
        dongle_id=dongle_id,
        status=status,
        notes=notes,
        ppm_estimate=ppm,
    )


# Patterns we recognize. These are matched against combined stdout+stderr
# from rtl_test; they're regex because exact wording varies across librtlsdr
# versions.

_BUSY_PATTERN = re.compile(
    r"usb_claim_interface error -6"
    r"|usb_open error -6"
    r"|claimed by second instance"
    r"|Device or resource busy"
    r"|LIBUSB_ERROR_BUSY",
    re.IGNORECASE,
)

_KERNEL_DRIVER_PATTERN = re.compile(
    r"Kernel driver is active"
    r"|kernel driver may be active",
    re.IGNORECASE,
)

_PERMISSIONS_PATTERN = re.compile(
    r"usb_open error -3"
    r"|Please fix the device permissions"
    r"|LIBUSB_ERROR_ACCESS",
    re.IGNORECASE,
)

_NO_DEVICE_PATTERN = re.compile(
    r"No supported devices found"
    r"|No matching devices found",
    re.IGNORECASE,
)

_TUNER_FOUND_PATTERN = re.compile(r"Found .*tuner", re.IGNORECASE)

# rtl_test prints this on clean exit. The number is the canonical
# sample-loss metric, much more reliable than the per-buffer "lost at
# least N bytes" lines that fire during normal startup.
_SAMPLE_LOSS_FINAL_PATTERN = re.compile(
    r"Samples per million lost \(minimum\):\s*(\d+)"
)

# Lines that come from rtl_test's clean-shutdown path. These are caused
# by us sending SIGINT, not by anything wrong with the dongle. Surfacing
# them as diagnostics is misleading.
_SHUTDOWN_NOISE_PATTERN = re.compile(
    r"Signal caught"
    r"|User cancel"
    r"|Samples per million lost"
    r"|exiting\.\.\."
    r"|exiting!",
    re.IGNORECASE,
)


def _salient_stderr_lines(stderr: str, max_lines: int = 3) -> list[str]:
    """Return a few stderr lines that look diagnostic, for surfacing to the user.

    Skips:
    • Empty lines
    • Banner lines ("Found ...", "Using ...")
    • Shutdown messages caused by our own SIGINT — these aren't
      diagnostics, they're an artifact of how we terminate the probe.
    """
    out: list[str] = []
    for line in stderr.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Found ") or line.startswith("Using "):
            continue
        if _SHUTDOWN_NOISE_PATTERN.search(line):
            continue
        out.append(line)
        if len(out) >= max_lines:
            break
    return out


# ──────────────────────────────────────────────────────────────────
# HackRF health
# ──────────────────────────────────────────────────────────────────


async def check_hackrf(dongle: Dongle) -> HealthReport:
    """Probe a HackRF via hackrf_info. Distinguishes busy / missing / healthy."""
    if which("hackrf_info") is None:
        return HealthReport(
            dongle_id=dongle.id,
            status=DongleStatus.UNAVAILABLE,
            notes=["hackrf_info not installed"],
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            "hackrf_info",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        return HealthReport(
            dongle_id=dongle.id,
            status=DongleStatus.FAILED,
            notes=[f"could not start hackrf_info: {exc}"],
        )

    # hackrf_info exits on its own after enumerating devices, so we
    # don't need to signal it. But we still want a hard cap in case it
    # hangs — same single-communicate() pattern as rtl_test for safety.
    loop = asyncio.get_event_loop()
    kill_handle = loop.call_later(_PROBE_TIMEOUT_S, _safe_kill, proc)
    try:
        stdout_bytes, stderr_bytes = await proc.communicate()
    finally:
        kill_handle.cancel()

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    return _interpret_hackrf_info(dongle, stdout, stderr)


def _interpret_hackrf_info(
    dongle: Dongle, stdout: str, stderr: str
) -> HealthReport:
    combined = stdout + "\n" + stderr

    if "HACKRF_ERROR_NOT_FOUND" in combined or "No HackRF boards found" in combined:
        return HealthReport(
            dongle_id=dongle.id,
            status=DongleStatus.FAILED,
            notes=["hackrf_info found no boards (device may have disconnected)"],
        )

    if "Resource busy" in combined or "HACKRF_ERROR_BUSY" in combined:
        return HealthReport(
            dongle_id=dongle.id,
            status=DongleStatus.BUSY,
            notes=[
                "HackRF is in use by another process",
                "common causes: hackrf_sweep, hackrf_transfer, GQRX, GNU Radio flowgraph",
                "remediation: stop the other process and rerun",
            ],
        )

    if "Permission denied" in combined or "LIBUSB_ERROR_ACCESS" in combined:
        return HealthReport(
            dongle_id=dongle.id,
            status=DongleStatus.FAILED,
            notes=[
                "USB permissions error opening HackRF",
                "remediation: install hackrf udev rules and reconnect, "
                "or add yourself to the plugdev group",
            ],
        )

    # Match the dongle's serial in the output if we have one
    if dongle.serial and dongle.serial in combined:
        return HealthReport(
            dongle_id=dongle.id, status=DongleStatus.HEALTHY, notes=[]
        )

    if "Found HackRF" in combined:
        return HealthReport(
            dongle_id=dongle.id,
            status=DongleStatus.HEALTHY,
            notes=["serial not matched in hackrf_info output but device present"],
        )

    return HealthReport(
        dongle_id=dongle.id,
        status=DongleStatus.FAILED,
        notes=["hackrf_info did not list this device"]
        + _salient_stderr_lines(stderr),
    )


# ──────────────────────────────────────────────────────────────────
# Public entrypoint
# ──────────────────────────────────────────────────────────────────


async def check_all(dongles: list[Dongle]) -> list[HealthReport]:
    """Run health checks for every dongle in parallel."""
    tasks: list[asyncio.Task[HealthReport]] = []
    for d in dongles:
        if d.driver == "rtlsdr":
            tasks.append(asyncio.create_task(check_rtlsdr(d)))
        elif d.driver == "hackrf":
            tasks.append(asyncio.create_task(check_hackrf(d)))
        else:
            tasks.append(
                asyncio.create_task(
                    _static_report(d.id, DongleStatus.UNAVAILABLE, ["unknown driver"])
                )
            )
    results = await asyncio.gather(*tasks)
    for dongle, report in zip(dongles, results, strict=True):
        dongle.status = report.status
        dongle.last_health_check = datetime.now(timezone.utc)
        dongle.health_notes = report.notes
    return results


async def _static_report(
    dongle_id: str, status: DongleStatus, notes: list[str]
) -> HealthReport:
    return HealthReport(dongle_id=dongle_id, status=status, notes=notes)
