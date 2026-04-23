"""Tests for v0.5.18 — rtl_power diagnostic visibility.

A user's rtl_power sidecar on rtlsdr-00000001 was silently failing:
the broker allocated the dongle, rtl_power's subprocess exited fast
(likely USB error with the rtlsdr_v4 hardware model), stdout yielded
zero lines, the sweep loop ended, finally released the lease. Log
showed `allocated ... lease 6 (exclusive)` immediately followed by
`released lease 6` with no error between.

Root cause: `log_stderr=False` in the rtl_power ManagedProcess config
swallowed whatever error the subprocess printed. v0.5.18:
  • Enables stderr logging at WARNING level so failures are visible
  • OccupancyAnalyzer.consume returns a sample count
  • _run_power_scan logs a warning when sweep produces 0 samples
"""

from __future__ import annotations


class TestRtlPowerStderrLogging:
    def test_rtl_power_backend_logs_stderr(self):
        """Failure visibility regression: rtl_power's stderr must NOT
        be suppressed. Without this we can't diagnose USB open errors,
        device-busy conditions, or unrecognized hardware."""
        import inspect
        from rfcensus.spectrum.backends import rtl_power
        src = inspect.getsource(rtl_power)
        # The ProcessConfig for rtl_power must enable stderr logging
        assert "log_stderr=True" in src, (
            "rtl_power backend should set log_stderr=True so silent "
            "subprocess failures become visible"
        )
        assert "log_stderr=False" not in src, (
            "rtl_power backend still has log_stderr=False somewhere"
        )


class TestOccupancyAnalyzerReturnsSampleCount:
    def test_consume_returns_int(self):
        """consume must return a sample count so callers can detect
        silent backend failures (empty sweeps)."""
        import inspect
        from rfcensus.spectrum.occupancy import OccupancyAnalyzer
        sig = inspect.signature(OccupancyAnalyzer.consume)
        # Return annotation — PEP 563 stringified, so match as str or type
        ann = sig.return_annotation
        assert ann is int or ann == "int", (
            f"consume return annotation should be int; got {ann!r}"
        )


class TestPowerScanZeroSamplesWarning:
    def test_run_power_scan_warns_on_zero_samples(self):
        """_run_power_scan should log a warning when sweep returns
        zero samples — near-certain backend failure."""
        import inspect
        from rfcensus.engine import strategy
        src = inspect.getsource(strategy)
        assert "produced 0 samples" in src, (
            "_run_power_scan should log when sweep yields zero samples"
        )
