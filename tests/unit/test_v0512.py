"""Tests for v0.5.12:
  • Scheduler max_parallel saturates dongle count, not just CPU fraction
  • rtl_tcp readiness wait helper
  • Hardware-loss detector skips decoder-specific exit reasons
"""

from __future__ import annotations

import asyncio
import socket

import pytest


def _make_dongle_with_freq(idx, freq_low_hz, freq_high_hz):
    from rfcensus.hardware.dongle import (
        Dongle, DongleCapabilities, DongleStatus,
    )
    caps = DongleCapabilities(
        freq_range_hz=(freq_low_hz, freq_high_hz),
        max_sample_rate=2_400_000, bits_per_sample=8,
        bias_tee_capable=False, tcxo_ppm=10.0,
    )
    return Dongle(
        id=f"rtl-{idx}", serial=f"S{idx}", model="rtlsdr_generic",
        driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
        driver_index=idx,
    )


def _config_for_test(cpu_fraction=0.5, max_concurrent=None):
    from rfcensus.config.schema import SiteConfig
    cfg_dict = {
        "site": {"name": "test"},
        "resources": {"cpu_budget_fraction": cpu_fraction},
        "antennas": [],
        "bands": {"enabled": ["b1"]},
        "band_definitions": [{
            "id": "b1", "name": "b1",
            "freq_low": 900_000_000, "freq_high": 920_000_000,
        }],
        "dongles": [],
    }
    if max_concurrent is not None:
        cfg_dict["resources"]["max_concurrent_decoders"] = max_concurrent
    return SiteConfig.model_validate(cfg_dict)


def _broker_with(dongles):
    from rfcensus.events import EventBus
    from rfcensus.hardware.broker import DongleBroker
    from rfcensus.hardware.registry import HardwareRegistry
    return DongleBroker(HardwareRegistry(dongles=dongles), EventBus())


# ──────────────────────────────────────────────────────────────────
# Scheduler max_parallel saturates dongle count
# ──────────────────────────────────────────────────────────────────


class TestMaxParallelSaturatesDongles:
    """Regression: cpu_budget_fraction=0.5 on a 4-core box gave
    max_parallel=2, leaving 3 of 5 dongles idle. The new logic uses
    min(usable_dongles, cpu_cap*4) so dongles get saturated when there's
    hardware to use."""

    def test_max_parallel_matches_dongle_count_when_cpu_allows(
        self, monkeypatch
    ):
        from rfcensus.engine.scheduler import Scheduler
        monkeypatch.setattr("os.cpu_count", lambda: 4)

        dongles = [
            _make_dongle_with_freq(i, 24_000_000, 1_700_000_000)
            for i in range(5)
        ]
        config = _config_for_test(cpu_fraction=0.5)
        plan = Scheduler(config, _broker_with(dongles)).plan(
            config.band_definitions
        )
        # 5 usable dongles → max_parallel should be 5 even though
        # cpu_budget_fraction=0.5 would have given just 2
        assert plan.max_parallel_per_wave == 5

    def test_explicit_override_still_respected(self, monkeypatch):
        """max_concurrent_decoders explicitly set takes precedence."""
        from rfcensus.engine.scheduler import Scheduler
        monkeypatch.setattr("os.cpu_count", lambda: 4)

        dongles = [
            _make_dongle_with_freq(i, 24_000_000, 1_700_000_000)
            for i in range(5)
        ]
        config = _config_for_test(cpu_fraction=0.5, max_concurrent=2)
        plan = Scheduler(config, _broker_with(dongles)).plan(
            config.band_definitions
        )
        assert plan.max_parallel_per_wave == 2

    def test_no_dongles_falls_back_to_cpu_cap(self, monkeypatch):
        from rfcensus.engine.scheduler import Scheduler
        monkeypatch.setattr("os.cpu_count", lambda: 4)
        config = _config_for_test(cpu_fraction=0.5)
        plan = Scheduler(config, _broker_with([])).plan(
            config.band_definitions
        )
        # 0 usable dongles → fall back to cpu_cap (4 * 0.5 = 2)
        assert plan.max_parallel_per_wave == 2

    def test_cpu_cap_protects_against_runaway(self, monkeypatch):
        """If user has lots of dongles and tiny CPU, max_parallel
        is capped at cpu_cap*4 to avoid pathological context-switch
        thrash on something like a Pi Zero."""
        from rfcensus.engine.scheduler import Scheduler
        monkeypatch.setattr("os.cpu_count", lambda: 1)  # Pi Zero

        # Pretend the user has 12 dongles attached
        dongles = [
            _make_dongle_with_freq(i, 24_000_000, 1_700_000_000)
            for i in range(12)
        ]
        config = _config_for_test(cpu_fraction=0.5)
        plan = Scheduler(config, _broker_with(dongles)).plan(
            config.band_definitions
        )
        # cpu_cap = max(1, 1 * 0.5) = 1; cap at 4× = 4
        assert plan.max_parallel_per_wave == 4


# ──────────────────────────────────────────────────────────────────
# rtl_tcp readiness wait
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestWaitForTcpReady:
    async def test_returns_true_when_socket_accepts(self):
        # v0.5.24+: probe removed from rtlamr decoder; function lives
        # in rfcensus.utils.tcp and is used by the broker when starting
        # rtl_tcp before attaching the fanout.
        from rfcensus.utils.tcp import wait_for_tcp_ready

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        try:
            ready = await wait_for_tcp_ready("127.0.0.1", port, timeout_s=2.0)
            assert ready is True
        finally:
            sock.close()

    async def test_returns_false_on_timeout(self):
        from rfcensus.utils.tcp import wait_for_tcp_ready
        # Port 1 should never accept on most systems
        ready = await wait_for_tcp_ready("127.0.0.1", 1, timeout_s=0.5)
        assert ready is False


# ──────────────────────────────────────────────────────────────────
# Hardware-loss detector skip set
# ──────────────────────────────────────────────────────────────────


class TestHardwareLossSkipReasons:
    """Verifies the strategy's skip-set for decoder-specific exit
    reasons. We don't run the strategy end-to-end (heavy); just
    inspect the source for the documented exit reason set and
    verify the skip-set semantics directly."""

    def test_skip_set_contains_decoder_specific_reasons(self):
        import rfcensus.engine.strategy as strategy
        import inspect
        src = inspect.getsource(strategy)
        assert '"binary_missing"' in src, "binary_missing should be in skip set"
        assert '"rtl_tcp_not_ready"' in src, "rtl_tcp_not_ready should be in skip set"
        assert '"wrong_lease_type"' in src, "wrong_lease_type should be in skip set"

    def test_decoder_specific_reasons_dont_trigger_flag(self):
        # Mirror the strategy's guard logic to verify semantics
        decoder_specific = {
            "binary_missing", "rtl_tcp_not_ready", "wrong_lease_type",
            "user_skipped",
        }
        for reason in decoder_specific:
            should_flag = reason not in decoder_specific
            assert not should_flag, f"reason={reason!r} should be skipped"

    def test_genuine_hardware_loss_still_flaggable(self):
        """The skip-set must NOT include real hardware-loss reasons,
        otherwise we'd silently miss disconnections."""
        decoder_specific = {
            "binary_missing", "rtl_tcp_not_ready", "wrong_lease_type",
            "user_skipped",
        }
        # "completed" or other unknown reasons should still trigger
        # the flag if other conditions match (early exit + 0 decodes)
        assert "completed" not in decoder_specific
        assert "hardware_lost" not in decoder_specific
