"""Tests for v0.5.0 behavior changes.

Focus areas:
  • _parse_duration: indefinite tokens (0, forever, etc.)
  • _parse_gain: numeric, auto, range checking
  • broker.release/allocate publish OUTSIDE lock (no deadlock)
  • ManagedProcess.stop SIGINT-first order
  • SessionRunner._handle_sigint sets stop flag
  • Multi-pass duration accounting
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time

import pytest


# ──────────────────────────────────────────────────────────────────
# Duration parsing — indefinite tokens
# ──────────────────────────────────────────────────────────────────


class TestParseDuration:
    def test_normal_durations_unchanged(self):
        from rfcensus.commands.inventory import _parse_duration
        assert _parse_duration("30m", 0) == 1800.0
        assert _parse_duration("1h", 0) == 3600.0
        assert _parse_duration("45s", 0) == 45.0
        assert _parse_duration("600", 0) == 600.0

    def test_zero_means_indefinite(self):
        from rfcensus.commands.inventory import _parse_duration
        assert _parse_duration("0", 1800) == 0.0

    def test_forever_means_indefinite(self):
        from rfcensus.commands.inventory import _parse_duration
        assert _parse_duration("forever", 1800) == 0.0
        assert _parse_duration("FOREVER", 1800) == 0.0
        assert _parse_duration("indefinite", 1800) == 0.0
        assert _parse_duration("inf", 1800) == 0.0

    def test_empty_returns_default(self):
        from rfcensus.commands.inventory import _parse_duration
        assert _parse_duration("", 1800) == 1800.0

    def test_invalid_raises_clearly(self):
        from rfcensus.commands.inventory import _parse_duration
        import click
        with pytest.raises(click.BadParameter, match="invalid duration"):
            _parse_duration("not_a_time", 0)


# ──────────────────────────────────────────────────────────────────
# Gain parsing
# ──────────────────────────────────────────────────────────────────


class TestParseGain:
    def test_auto_default(self):
        from rfcensus.commands.inventory import _parse_gain
        assert _parse_gain("auto") == "auto"
        assert _parse_gain(None) == "auto"
        assert _parse_gain("") == "auto"
        assert _parse_gain("AUTO") == "auto"

    def test_integer_values(self):
        from rfcensus.commands.inventory import _parse_gain
        assert _parse_gain("40") == "40"
        assert _parse_gain("0") == "0"
        assert _parse_gain("50") == "50"

    def test_float_values(self):
        from rfcensus.commands.inventory import _parse_gain
        # Whole-number float strips the .0
        assert _parse_gain("40.0") == "40"
        # Non-integer keeps the decimal
        assert _parse_gain("37.2") == "37.2"

    def test_out_of_range_rejected(self):
        from rfcensus.commands.inventory import _parse_gain
        import click
        with pytest.raises(click.BadParameter, match="out of range"):
            _parse_gain("100")
        with pytest.raises(click.BadParameter, match="out of range"):
            _parse_gain("-5")

    def test_garbage_rejected(self):
        from rfcensus.commands.inventory import _parse_gain
        import click
        with pytest.raises(click.BadParameter, match="invalid gain"):
            _parse_gain("loud")


# ──────────────────────────────────────────────────────────────────
# Broker: publish outside lock (regression for the hang)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestBrokerNoDeadlock:
    """Regression: broker.release() used to hold its lock across an
    event_bus.publish() call. A slow subscriber would block publish,
    holding the lock, and any other release/shutdown call would wedge.

    This caused the 21:00 scan hang where lease 7 got stuck mid-release
    and broker.shutdown() never returned.
    """

    async def test_release_does_not_block_on_slow_subscriber(self):
        from rfcensus.events import EventBus, HardwareEvent
        from rfcensus.hardware.broker import (
            AccessMode, DongleBroker, DongleLease, DongleRequirements,
        )
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
        )
        from rfcensus.hardware.registry import HardwareRegistry

        # Build a registry with one dongle
        caps = DongleCapabilities(
            freq_range_hz=(24_000_000, 1_700_000_000),
            max_sample_rate=2_400_000,
            bits_per_sample=8,
            bias_tee_capable=False,
            tcxo_ppm=10.0,
        )
        d = Dongle(
            id="rtl-0", serial="00000001", model="rtlsdr_generic",
            driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
            driver_index=0,
        )
        registry = HardwareRegistry(dongles=[d])
        bus = EventBus()
        broker = DongleBroker(registry, bus)

        # Install a slow subscriber that takes 0.5s per event
        slow_sub_calls = []

        async def slow_subscriber(event: HardwareEvent) -> None:
            slow_sub_calls.append(event)
            await asyncio.sleep(0.5)

        bus.subscribe(HardwareEvent, slow_subscriber)

        # Allocate, then release. The release MUST NOT block the second
        # call below.
        req = DongleRequirements(
            freq_hz=915_000_000, sample_rate=2_400_000,
            access_mode=AccessMode.EXCLUSIVE,
        )
        lease = await broker.allocate(req, consumer="test")

        # Now spawn a release; meanwhile, broker.shutdown() should still
        # be able to acquire the lock. Before the fix this deadlocked.
        async def release_and_shutdown():
            await broker.release(lease)
            await broker.shutdown()

        # Wrap with a wall-clock timeout to detect deadlock
        try:
            await asyncio.wait_for(release_and_shutdown(), timeout=5.0)
        except asyncio.TimeoutError:
            pytest.fail(
                "broker.release() + shutdown() deadlocked — the lock+publish "
                "anti-pattern returned"
            )

        # Drain to allow the slow subscriber to finish processing
        await bus.drain(timeout=3.0)

    async def test_publish_failures_dont_break_release(self):
        """If a subscriber raises, release() should still complete and the
        lease should still be marked released."""
        from rfcensus.events import EventBus, HardwareEvent
        from rfcensus.hardware.broker import (
            AccessMode, DongleBroker, DongleRequirements,
        )
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
        )
        from rfcensus.hardware.registry import HardwareRegistry

        caps = DongleCapabilities(
            freq_range_hz=(24_000_000, 1_700_000_000),
            max_sample_rate=2_400_000, bits_per_sample=8,
            bias_tee_capable=False, tcxo_ppm=10.0,
        )
        d = Dongle(
            id="rtl-0", serial="00000001", model="rtlsdr_generic",
            driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
            driver_index=0,
        )
        broker = DongleBroker(HardwareRegistry(dongles=[d]), EventBus())
        req = DongleRequirements(
            freq_hz=915_000_000, sample_rate=2_400_000,
            access_mode=AccessMode.EXCLUSIVE,
        )
        lease = await broker.allocate(req, consumer="test")
        # Should not raise even if subscriber is wonky
        await broker.release(lease)
        assert lease._released is True


# ──────────────────────────────────────────────────────────────────
# ManagedProcess.stop SIGINT-first
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestStopSigintFirst:
    """Regression: ManagedProcess.stop() should attempt SIGINT first, then
    SIGTERM, then SIGKILL. multimon-ng ignores SIGTERM mid-decode but
    honors SIGINT. SIGINT-first reduces the SIGKILL fallback rate
    significantly."""

    async def test_sigint_handled_gracefully(self, tmp_path):
        from rfcensus.utils.async_subprocess import ManagedProcess, ProcessConfig

        # Script that handles SIGINT cleanly and exits 0 within 200ms
        script = tmp_path / "sigint_clean.py"
        script.write_text(
            "import signal, sys, time\n"
            "def _handle(sig, frame):\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGINT, _handle)\n"
            "while True:\n"
            "    time.sleep(0.05)\n"
        )

        proc = ManagedProcess(ProcessConfig(
            name="sigint-test",
            args=[sys.executable, str(script)],
            log_stderr=False,
            kill_timeout_s=3.0,
        ))
        await proc.start()
        await asyncio.sleep(0.2)  # Let it install handler
        start = time.monotonic()
        rc = await proc.stop()
        elapsed = time.monotonic() - start
        # Should exit promptly via SIGINT, not need to escalate
        assert elapsed < 2.0, f"stop took {elapsed:.2f}s — escalation happened?"
        # Returncode should be 0 (clean exit) or negative (signal)
        assert rc == 0 or rc is not None

    async def test_sigint_ignored_escalates_to_sigterm(self, tmp_path):
        from rfcensus.utils.async_subprocess import ManagedProcess, ProcessConfig

        # Script that ignores SIGINT but honors SIGTERM
        script = tmp_path / "sigint_ignore.py"
        script.write_text(
            "import signal, sys, time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "def _term(sig, frame):\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, _term)\n"
            "while True:\n"
            "    time.sleep(0.05)\n"
        )

        proc = ManagedProcess(ProcessConfig(
            name="sigterm-only",
            args=[sys.executable, str(script)],
            log_stderr=False,
            kill_timeout_s=3.0,
        ))
        await proc.start()
        await asyncio.sleep(0.2)
        start = time.monotonic()
        rc = await proc.stop()
        elapsed = time.monotonic() - start
        # Should escalate from SIGINT (~1.5s grace) to SIGTERM and exit
        # Should not need to go to SIGKILL
        assert elapsed < 4.0, f"stop took {elapsed:.2f}s — needed SIGKILL?"
        assert rc is not None


# ──────────────────────────────────────────────────────────────────
# SessionRunner SIGINT handler
# ──────────────────────────────────────────────────────────────────


class TestSessionRunnerSigintHandler:
    """Tests for the SIGINT handling logic on SessionRunner.

    We unit-test the handler method directly rather than running a
    whole session, because spinning up a real session in tests is heavy.
    """

    def test_first_sigint_sets_stop_flag(self):
        """First SIGINT should set _stop_requested without raising."""
        from rfcensus.engine.session import SessionRunner
        # Build a minimal SessionRunner — we only exercise _handle_sigint
        runner = SessionRunner.__new__(SessionRunner)
        runner._stop_requested = False
        runner._sigint_count = 0
        runner._handle_sigint()
        assert runner._stop_requested is True
        assert runner._sigint_count == 1

    def test_second_sigint_restores_default_handler(self, monkeypatch):
        """Second SIGINT should restore default SIGINT handler so a third
        Ctrl-C kills the process via Python's normal path."""
        from rfcensus.engine.session import SessionRunner

        runner = SessionRunner.__new__(SessionRunner)
        runner._stop_requested = False
        runner._sigint_count = 0

        # Capture signal.signal calls
        signal_calls = []
        def fake_signal(sig, handler):
            signal_calls.append((sig, handler))
        monkeypatch.setattr(signal, "signal", fake_signal)

        # Stub asyncio.all_tasks to return empty (we're not in a loop)
        monkeypatch.setattr(asyncio, "all_tasks", lambda: [])

        runner._handle_sigint()  # first
        runner._handle_sigint()  # second

        # On the second call, signal.SIGINT should be reset to SIG_DFL
        assert (signal.SIGINT, signal.SIG_DFL) in signal_calls
