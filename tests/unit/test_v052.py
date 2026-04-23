"""Tests for v0.5.2 functionality:
  • DecoderFailureEvent → SessionRunner queues retry
  • Per-dongle failure cap (3 strikes → permanently failed)
  • Successful run resets the failure counter
  • _is_quiet returns true after the configured window
  • CLI accepts --until-quiet
  • Pending retries don't double-queue for the same band
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest


def _make_runner(*, max_failures: int = 3, until_quiet_s: float | None = None):
    """Build a SessionRunner without going through the full bootstrap.

    Many of the v0.5.2 behaviors don't need an actual session — we can
    test the helper methods directly on a bare instance.
    """
    from rfcensus.engine.session import SessionRunner
    runner = SessionRunner.__new__(SessionRunner)
    runner._stop_requested = False
    runner._sigint_count = 0
    runner._pending_retries = []
    runner._sidecar_tasks = set()
    runner._dongle_failure_counts = {}
    runner._permanently_failed = set()
    runner._last_new_emitter_at = 0.0
    runner._strategy_results = []
    runner._dispatcher = None
    runner._ctx = None
    runner._band_by_id = {}
    runner.max_dongle_failures = max_failures
    runner.until_quiet_s = until_quiet_s
    return runner


def _make_band(band_id: str, freq_mhz: float = 915.0):
    """Synthetic BandConfig for testing."""
    from rfcensus.config.schema import BandConfig
    f = int(freq_mhz * 1_000_000)
    return BandConfig(
        id=band_id, name=band_id,
        freq_low=int(f * 0.99), freq_high=int(f * 1.01),
    )


@dataclass
class _FakeFailureEvent:
    band_id: str
    dongle_id: str
    decoder_name: str = "rtl_433"
    elapsed_s: float = 1.0
    remaining_s: float = 50.0


# ──────────────────────────────────────────────────────────────────
# Failure tracking and retry queueing
# ──────────────────────────────────────────────────────────────────


class TestRecordFailure:
    def test_first_failure_increments_counter_and_queues_retry(self):
        runner = _make_runner()
        band = _make_band("test_band")
        runner._band_by_id["test_band"] = band

        runner._record_failure(_FakeFailureEvent(
            band_id="test_band", dongle_id="rtl-0",
        ))
        assert runner._dongle_failure_counts["rtl-0"] == 1
        assert len(runner._pending_retries) == 1
        assert runner._pending_retries[0].band_id == "test_band"
        assert runner._pending_retries[0].remaining_s == 50.0
        assert "rtl-0" not in runner._permanently_failed

    def test_second_failure_increments_but_doesnt_double_queue_same_band(self):
        runner = _make_runner()
        runner._band_by_id["test_band"] = _make_band("test_band")

        runner._record_failure(_FakeFailureEvent(band_id="test_band", dongle_id="rtl-0"))
        runner._record_failure(_FakeFailureEvent(band_id="test_band", dongle_id="rtl-0"))
        assert runner._dongle_failure_counts["rtl-0"] == 2
        # Still only one queued retry for this band
        assert len(runner._pending_retries) == 1

    def test_third_failure_marks_permanently_failed(self):
        runner = _make_runner(max_failures=3)
        for band_id in ["b1", "b2", "b3"]:
            runner._band_by_id[band_id] = _make_band(band_id)

        # Three failures on the same dongle (different bands so none
        # are deduped)
        runner._record_failure(_FakeFailureEvent(band_id="b1", dongle_id="rtl-0"))
        runner._record_failure(_FakeFailureEvent(band_id="b2", dongle_id="rtl-0"))
        runner._record_failure(_FakeFailureEvent(band_id="b3", dongle_id="rtl-0"))

        assert "rtl-0" in runner._permanently_failed
        # The third failure should NOT have queued a retry (capped)
        retry_ids = [r.band_id for r in runner._pending_retries]
        assert "b3" not in retry_ids
        # First two are queued
        assert "b1" in retry_ids
        assert "b2" in retry_ids

    def test_unknown_band_id_doesnt_queue_retry(self):
        runner = _make_runner()
        runner._record_failure(_FakeFailureEvent(
            band_id="never_heard_of_it", dongle_id="rtl-0",
        ))
        # Counter still increments, but no retry queued
        assert runner._dongle_failure_counts["rtl-0"] == 1
        assert len(runner._pending_retries) == 0


class TestRecordSuccess:
    def test_success_resets_failure_counter(self):
        runner = _make_runner()
        runner._band_by_id["test_band"] = _make_band("test_band")
        runner._record_failure(_FakeFailureEvent(band_id="test_band", dongle_id="rtl-0"))
        runner._record_failure(_FakeFailureEvent(band_id="test_band", dongle_id="rtl-0"))
        assert runner._dongle_failure_counts["rtl-0"] == 2

        runner._record_success("rtl-0")
        assert "rtl-0" not in runner._dongle_failure_counts

    def test_success_on_unrelated_dongle_doesnt_affect_others(self):
        runner = _make_runner()
        runner._dongle_failure_counts["rtl-0"] = 2
        runner._dongle_failure_counts["rtl-1"] = 1

        runner._record_success("rtl-1")
        assert runner._dongle_failure_counts["rtl-0"] == 2
        assert "rtl-1" not in runner._dongle_failure_counts


# ──────────────────────────────────────────────────────────────────
# --until-quiet
# ──────────────────────────────────────────────────────────────────


class TestUntilQuiet:
    def test_disabled_when_until_quiet_s_is_none(self):
        runner = _make_runner(until_quiet_s=None)
        assert not runner._is_quiet()

    def test_disabled_when_zero(self):
        runner = _make_runner(until_quiet_s=0)
        assert not runner._is_quiet()

    def test_returns_false_before_any_emitter(self):
        """If we've never seen an emitter, _is_quiet should NOT trigger
        early exit — otherwise it would fire at startup before anything
        had a chance to be heard."""
        runner = _make_runner(until_quiet_s=600)
        assert not runner._is_quiet()

    @pytest.mark.asyncio
    async def test_returns_true_after_window_elapses(self, monkeypatch):
        runner = _make_runner(until_quiet_s=10.0)

        # Simulate that we saw an emitter, then time advanced past the window
        runner._last_new_emitter_at = 100.0  # arbitrary

        # Patch get_event_loop().time() to return 200 (>= 100 + 10)
        loop = asyncio.get_event_loop()
        monkeypatch.setattr(loop, "time", lambda: 200.0)
        assert runner._is_quiet()

    @pytest.mark.asyncio
    async def test_returns_false_within_window(self, monkeypatch):
        runner = _make_runner(until_quiet_s=10.0)
        runner._last_new_emitter_at = 100.0
        loop = asyncio.get_event_loop()
        monkeypatch.setattr(loop, "time", lambda: 105.0)  # only 5s elapsed
        assert not runner._is_quiet()


class TestHandleEmitter:
    @pytest.mark.asyncio
    async def test_only_new_emitters_update_timestamp(self, monkeypatch):
        runner = _make_runner(until_quiet_s=10.0)

        loop = asyncio.get_event_loop()
        monkeypatch.setattr(loop, "time", lambda: 500.0)

        # "confirmed" event should NOT reset the quiet timer
        @dataclass
        class FakeEmitter:
            kind: str = "confirmed"

        runner._handle_emitter(FakeEmitter(kind="confirmed"))
        assert runner._last_new_emitter_at == 0.0

        # "new" event SHOULD reset the timer
        runner._handle_emitter(FakeEmitter(kind="new"))
        assert runner._last_new_emitter_at == 500.0


# ──────────────────────────────────────────────────────────────────
# CLI flag wiring
# ──────────────────────────────────────────────────────────────────


class TestUntilQuietFlag:
    def test_flag_present_on_inventory(self):
        from rfcensus.commands.inventory import cli_inventory
        opts = {p.name for p in cli_inventory.params}
        assert "until_quiet" in opts

    def test_flag_present_on_scan(self):
        from rfcensus.commands.inventory import cli_scan
        opts = {p.name for p in cli_scan.params}
        assert "until_quiet" in opts

    def test_default_is_none(self):
        from rfcensus.commands.inventory import cli_inventory
        opt = next(p for p in cli_inventory.params if p.name == "until_quiet")
        assert opt.default is None


# ──────────────────────────────────────────────────────────────────
# Reprobe excludes permanently failed
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestReprobeExcludesPermanentlyFailed:
    async def test_permanently_failed_dongle_not_restored(self, monkeypatch):
        """Even if a permanently-failed dongle reappears in detection,
        re-probe should not restore it. This honors the session's
        consecutive-failure cap."""
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
        )
        from rfcensus.hardware.registry import HardwareRegistry, reprobe_for_recovery
        from rfcensus.hardware.drivers.rtlsdr import RtlSdrProbeResult
        from rfcensus.hardware.drivers.hackrf import HackRfProbeResult

        caps = DongleCapabilities(
            freq_range_hz=(24_000_000, 1_700_000_000),
            max_sample_rate=2_400_000, bits_per_sample=8,
            bias_tee_capable=False, tcxo_ppm=10.0,
        )
        d = Dongle(
            id="rtl-0", serial="00000001", model="rtlsdr_generic",
            driver="rtlsdr", capabilities=caps, status=DongleStatus.FAILED,
            driver_index=0,
        )
        registry = HardwareRegistry(dongles=[d])

        # Pretend the dongle reappeared in detection
        async def fake_rtl():
            return RtlSdrProbeResult(dongles=[d], diagnostic="")
        async def fake_hackrf():
            return HackRfProbeResult(dongles=[], diagnostic="")
        monkeypatch.setattr(
            "rfcensus.hardware.drivers.rtlsdr.probe_rtlsdr", fake_rtl,
        )
        monkeypatch.setattr(
            "rfcensus.hardware.drivers.hackrf.probe_hackrf", fake_hackrf,
        )

        # With exclude={rtl-0}, the dongle stays FAILED
        n_back, n_gone = await reprobe_for_recovery(
            registry, exclude={"rtl-0"},
        )
        assert n_back == 0
        assert d.status == DongleStatus.FAILED
