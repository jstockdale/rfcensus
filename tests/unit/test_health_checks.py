"""Tests for the rtl_test / hackrf_info output interpreters.

We test the pure functions `_interpret_rtl_test` and `_interpret_hackrf_info`
with realistic output samples covering the cases that matter most:
healthy, busy (in use elsewhere), kernel driver loaded, permissions,
device disappeared, and unrecognized errors.
"""

from __future__ import annotations

from rfcensus.hardware.dongle import (
    Dongle,
    DongleCapabilities,
    DongleStatus,
)
from rfcensus.hardware.health import (
    _interpret_hackrf_info,
    _interpret_rtl_test,
)


def _make_rtlsdr(serial: str = "00000001") -> Dongle:
    return Dongle(
        id=f"rtlsdr-{serial[-8:]}",
        serial=serial,
        model="rtl-sdr_v4",
        driver="rtlsdr",
        capabilities=DongleCapabilities(
            freq_range_hz=(24_000_000, 1_766_000_000),
            max_sample_rate=2_400_000,
            bits_per_sample=8,
            bias_tee_capable=True,
            tcxo_ppm=1.0,
        ),
        status=DongleStatus.DETECTED,
        driver_index=0,
    )


def _make_hackrf(serial: str = "0000000000000000c890") -> Dongle:
    return Dongle(
        id=f"hackrf-{serial[-8:]}",
        serial=serial,
        model="hackrf_one",
        driver="hackrf",
        capabilities=DongleCapabilities(
            freq_range_hz=(1_000_000, 6_000_000_000),
            max_sample_rate=20_000_000,
            bits_per_sample=8,
            bias_tee_capable=True,
            tcxo_ppm=10.0,
            wide_scan_capable=True,
        ),
        status=DongleStatus.DETECTED,
    )


# ──────────────────────────────────────────────────────────────────
# rtl_test interpretation
# ──────────────────────────────────────────────────────────────────


class TestInterpretRtlTest:
    def test_healthy_output(self):
        # Typical successful rtl_test output, captured here in test
        stdout = (
            "Found 1 device(s):\n"
            "  0:  Realtek, RTL2838UHIDIR, SN: 00000001\n"
            "\n"
            "Using device 0: Generic RTL2832U OEM\n"
            "Found Rafael Micro R828D tuner\n"
            "Supported gain values (29): ...\n"
            "[R82XX] PLL not locked!\n"
            "Sampling at 2048000 S/s.\n"
            "real sample rate: 2048000 current PPM: 0 cumulative PPM: 0\n"
            "real sample rate: 2048001 current PPM: 0 cumulative PPM: 0\n"
            "Signal caught, exiting!\n"
        )
        stderr = ""
        report = _interpret_rtl_test("rtlsdr-00000001", stdout, stderr)
        assert report.status == DongleStatus.HEALTHY
        assert report.ppm_estimate == 0.0
        # PLL note is present but not status-changing
        assert any("PLL" in n for n in report.notes)

    def test_device_busy_returns_BUSY_status(self):
        """The case the user just hit — dongle in use by another process."""
        stdout = "Found 1 device(s):\n  0:  Realtek, RTL2838UHIDIR, SN: 00000001\n"
        stderr = (
            "usb_claim_interface error -6\n"
            "Failed to open rtlsdr device #0.\n"
        )
        report = _interpret_rtl_test("rtlsdr-00000001", stdout, stderr)
        assert report.status == DongleStatus.BUSY
        # The user needs actionable info, not just "busy"
        joined = " ".join(report.notes).lower()
        assert "another process" in joined
        assert "lsof" in joined or "remediation" in joined

    def test_busy_via_LIBUSB_ERROR_BUSY(self):
        """Some librtlsdr versions report LIBUSB_ERROR_BUSY directly."""
        stderr = "LIBUSB_ERROR_BUSY when claiming interface\n"
        report = _interpret_rtl_test("rtlsdr-x", "", stderr)
        assert report.status == DongleStatus.BUSY

    def test_kernel_driver_active(self):
        """DVB driver bound — still 'busy' from the user's POV but different remediation."""
        stderr = (
            "Kernel driver is active, or device is claimed by second instance of librtlsdr.\n"
            "In the first case, please either detach or blacklist the kernel module\n"
            "(dvb_usb_rtl28xxu)...\n"
        )
        report = _interpret_rtl_test("rtlsdr-x", "", stderr)
        assert report.status == DongleStatus.BUSY
        joined = " ".join(report.notes).lower()
        assert "kernel" in joined
        assert "blacklist" in joined or "rmmod" in joined

    def test_permissions_error_returns_FAILED(self):
        stderr = (
            "usb_open error -3\n"
            "Please fix the device permissions, e.g. by installing the udev rules\n"
            "file rtl-sdr.rules\n"
        )
        report = _interpret_rtl_test("rtlsdr-x", "", stderr)
        assert report.status == DongleStatus.FAILED
        joined = " ".join(report.notes).lower()
        assert "udev" in joined or "permission" in joined

    def test_no_supported_devices(self):
        stdout = "No supported devices found.\n"
        report = _interpret_rtl_test("rtlsdr-x", stdout, "")
        assert report.status == DongleStatus.FAILED
        assert any("disconnected" in n or "no supported" in n.lower() for n in report.notes)

    def test_startup_sample_loss_alone_is_NOT_degraded(self):
        """Per-buffer 'lost at least N bytes' lines fire during normal USB warmup
        and should NOT downgrade the dongle. This was the original false-degrade bug."""
        stdout = (
            "Found Elonics E4000 tuner\n"
            "Sampling at 2048000 S/s.\n"
            "lost at least 64 bytes\n"
            "lost at least 128 bytes\n"
            "Samples per million lost (minimum): 0\n"
        )
        report = _interpret_rtl_test("rtlsdr-x", stdout, "")
        assert report.status == DongleStatus.HEALTHY

    def test_small_final_sample_loss_is_tolerated(self):
        """A few ppm of sample loss is normal startup behavior; don't penalize it."""
        stdout = (
            "Found Rafael Micro R820T tuner\n"
            "Sampling at 2048000 S/s.\n"
            "Samples per million lost (minimum): 2\n"
        )
        report = _interpret_rtl_test("rtlsdr-x", stdout, "")
        assert report.status == DongleStatus.HEALTHY

    def test_significant_final_sample_loss_marks_DEGRADED(self):
        """The canonical sample-loss number, if large, indicates a real problem."""
        stdout = (
            "Found Rafael Micro R820T tuner\n"
            "Sampling at 2048000 S/s.\n"
            "lost at least 1024 bytes\n"
            "lost at least 2048 bytes\n"
            "Samples per million lost (minimum): 250\n"
        )
        report = _interpret_rtl_test("rtlsdr-x", stdout, "")
        assert report.status == DongleStatus.DEGRADED
        assert any("250 ppm" in n for n in report.notes)

    def test_shutdown_messages_not_surfaced_as_diagnostics(self):
        """SIGINT-triggered shutdown messages aren't symptoms — they're how
        we ended the probe. They shouldn't appear in failure notes."""
        # Tuner-found is missing, so we hit the fallback path
        stderr = (
            "Signal caught, exiting!\n"
            "User cancel, exiting...\n"
            "Samples per million lost (minimum): 2\n"
        )
        report = _interpret_rtl_test("rtlsdr-x", "", stderr)
        assert report.status == DongleStatus.FAILED
        joined = " ".join(report.notes)
        # These should NOT appear in the surfaced notes
        assert "Signal caught" not in joined
        assert "User cancel" not in joined
        assert "Samples per million lost" not in joined

    def test_large_ppm_drift_marks_DEGRADED(self):
        stdout = (
            "Found Rafael Micro R820T tuner\n"
            "Sampling at 2048000 S/s.\n"
            "real sample rate: 2048500 current PPM: 250 cumulative PPM: 245\n"
        )
        report = _interpret_rtl_test("rtlsdr-x", stdout, "")
        assert report.status == DongleStatus.DEGRADED
        assert report.ppm_estimate is not None
        assert abs(report.ppm_estimate) > 100

    def test_unrecognized_failure_falls_through_to_FAILED_with_stderr(self):
        """When no pattern matches, surface what stderr said."""
        stderr = "some weird novel error message we haven't seen before\n"
        report = _interpret_rtl_test("rtlsdr-x", "", stderr)
        assert report.status == DongleStatus.FAILED
        assert any("no tuner response" in n for n in report.notes)
        # Should bubble up the salient line for debugging
        assert any("weird novel error" in n for n in report.notes)

    def test_busy_takes_priority_over_other_signals(self):
        """If both 'no devices' and 'busy' appear, busy is the more actionable diagnosis."""
        stderr = (
            "usb_claim_interface error -6\n"
            "No matching devices found.\n"
        )
        report = _interpret_rtl_test("rtlsdr-x", "", stderr)
        assert report.status == DongleStatus.BUSY


# ──────────────────────────────────────────────────────────────────
# hackrf_info interpretation
# ──────────────────────────────────────────────────────────────────


class TestInterpretHackrfInfo:
    def test_healthy_with_matching_serial(self):
        dongle = _make_hackrf(serial="0000000000000000c890acdb12345678")
        stdout = (
            "hackrf_info version: 2023.01.1\n"
            "libhackrf version: 2023.01.1 (0.8)\n"
            "\n"
            "Found HackRF\n"
            "Index: 0\n"
            "Serial number: 0000000000000000c890acdb12345678\n"
            "Board ID Number: 2 (HackRF One)\n"
            "Firmware Version: 2023.01.1\n"
        )
        report = _interpret_hackrf_info(dongle, stdout, "")
        assert report.status == DongleStatus.HEALTHY

    def test_no_boards_found(self):
        dongle = _make_hackrf()
        stdout = "hackrf_info version: 2023.01.1\nNo HackRF boards found.\n"
        report = _interpret_hackrf_info(dongle, stdout, "")
        assert report.status == DongleStatus.FAILED
        assert any("disconnected" in n.lower() or "no boards" in n.lower() for n in report.notes)

    def test_busy(self):
        dongle = _make_hackrf()
        stderr = "hackrf_open() failed: Resource busy (HACKRF_ERROR_BUSY)\n"
        report = _interpret_hackrf_info(dongle, "", stderr)
        assert report.status == DongleStatus.BUSY
        joined = " ".join(report.notes).lower()
        assert "another process" in joined
        assert any(tool in joined for tool in ("hackrf_sweep", "hackrf_transfer", "gqrx"))

    def test_permissions(self):
        dongle = _make_hackrf()
        stderr = "Permission denied opening device\n"
        report = _interpret_hackrf_info(dongle, "", stderr)
        assert report.status == DongleStatus.FAILED
        joined = " ".join(report.notes).lower()
        assert "udev" in joined or "permission" in joined or "plugdev" in joined

    def test_device_present_but_serial_does_not_match(self):
        """We still call this healthy but flag the mismatch."""
        dongle = _make_hackrf(serial="aaaaaaaaaaaaaaaa")
        stdout = (
            "Found HackRF\n"
            "Serial number: zzzzzzzzzzzzzzzz\n"
        )
        report = _interpret_hackrf_info(dongle, stdout, "")
        assert report.status == DongleStatus.HEALTHY
        assert any("serial not matched" in n.lower() for n in report.notes)


# ──────────────────────────────────────────────────────────────────
# Integration-style test for the subprocess capture pipeline
# ──────────────────────────────────────────────────────────────────

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.mark.asyncio
class TestSubprocessCapture:
    """Verify our subprocess pipeline captures all stdio, including data
    written before a SIGINT shutdown. This is the regression test for
    the asyncio.wait_for(communicate()) bug that lost buffered output."""

    async def test_captures_output_written_before_signal(self, tmp_path):
        """A long-running process that writes output, then we signal it
        to stop — we should capture both the early output AND the
        shutdown output. The bug was that we lost the early output."""
        from rfcensus.hardware import health
        import signal

        # Write a fake program that mimics rtl_test: prints diagnostics,
        # runs a loop, catches SIGINT and prints a goodbye.
        fake = tmp_path / "fake_rtl_test.py"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import signal, sys, time\n"
            "print('Found 3 device(s):')\n"
            "print('Using device 0: Generic')\n"
            "print('Found Rafael Micro R820T tuner')\n"
            "print('Sampling at 2048000 S/s.')\n"
            "sys.stdout.flush()\n"
            "running = True\n"
            "def stop(*_):\n"
            "    global running\n"
            "    running = False\n"
            "    print('Signal caught, exiting!', file=sys.stderr)\n"
            "    print('Samples per million lost (minimum): 0', file=sys.stderr)\n"
            "    sys.stderr.flush()\n"
            "signal.signal(signal.SIGINT, stop)\n"
            "while running:\n"
            "    time.sleep(0.1)\n"
        )
        fake.chmod(0o755)

        # Run our scheduled-signal pattern against the fake binary
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(fake),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        loop = asyncio.get_event_loop()
        sig_handle = loop.call_later(
            1.0, health._safe_send_signal, proc, signal.SIGINT
        )
        kill_handle = loop.call_later(
            5.0, health._safe_kill, proc
        )
        try:
            stdout_bytes, stderr_bytes = await proc.communicate()
        finally:
            sig_handle.cancel()
            kill_handle.cancel()

        stdout = stdout_bytes.decode()
        stderr = stderr_bytes.decode()
        # The early-output that the previous bug discarded:
        assert "Found Rafael Micro R820T tuner" in stdout
        assert "Sampling at 2048000" in stdout
        # And the shutdown output:
        assert "Signal caught" in stderr
        assert "Samples per million lost" in stderr


@pytest.mark.asyncio
class TestManagedProcessStopAfterExit:
    """Regression test: ManagedProcess.stop() must not log 'exit status
    already read' warnings when the process has already exited via natural
    EOF (e.g. when the caller drained stdout to completion).
    """

    async def test_stop_after_natural_exit_does_not_warn(self, caplog, tmp_path):
        from rfcensus.utils.async_subprocess import ManagedProcess, ProcessConfig
        import logging

        # Quick-exit script
        script = tmp_path / "quick.py"
        script.write_text("import sys; print('hello'); sys.exit(0)\n")

        proc = ManagedProcess(ProcessConfig(
            name="quick-exit",
            args=[sys.executable, str(script)],
            log_stderr=False,
            kill_timeout_s=2.0,
        ))
        await proc.start()
        # Drain stdout to EOF (mimics what probe_rtlsdr does)
        async for _line in proc.stdout_lines():
            pass

        # The process has now exited naturally. stop() should detect this
        # and skip signalling. If it doesn't, asyncio logs a warning to
        # the asyncio logger.
        caplog.set_level(logging.WARNING, logger="asyncio")
        rc = await proc.stop()
        assert rc == 0
        # No "exit status already read" or similar warnings
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("already read" in m for m in warning_messages), (
            f"Expected no asyncio warnings, got: {warning_messages}"
        )


@pytest.mark.asyncio
class TestSerializationSafeCommunicate:
    """Regression test for the wait_for(communicate()) data-loss bug
    that we fixed in serialization.py. Mirrors test_subprocess_capture
    but using serialization's _safe_communicate helper directly."""

    async def test_safe_communicate_captures_pre_signal_output(self, tmp_path):
        from rfcensus.hardware.serialization import _safe_communicate

        # Fake script that prints output, then runs forever
        fake = tmp_path / "fake_long_running.py"
        fake.write_text(
            "import sys, time\n"
            "print('important output before timeout')\n"
            "sys.stdout.flush()\n"
            "time.sleep(60)\n"
        )

        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(fake),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Timeout shorter than the sleep
        stdout, stderr, timed_out = await _safe_communicate(proc, timeout=1.0)
        assert timed_out is True
        assert b"important output before timeout" in stdout
