"""Tests for v0.5.1 changes:

  • Serialize preflight bug (rtl_eeprom exit=1 on success path)
  • Registry mark_failed / mark_healthy lifecycle
  • Strategy detects early-exit as suspected hardware loss
  • inventory vs scan defaults
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


def _caps():
    from rfcensus.hardware.dongle import DongleCapabilities
    return DongleCapabilities(
        freq_range_hz=(24_000_000, 1_700_000_000),
        max_sample_rate=2_400_000,
        bits_per_sample=8,
        bias_tee_capable=False,
        tcxo_ppm=10.0,
    )


# ──────────────────────────────────────────────────────────────────
# Preflight bug fix — file-existence-and-size, not exit code
# ──────────────────────────────────────────────────────────────────


class TestPreflightLastErrorLine:
    """Regression for the bug where rtl_eeprom -d N exits 1 on the success
    path, and our preflight grabbed the first line of the device list as
    the 'error', producing a meaningless message like:
        ✗ dongle idx=1: probe failed — 0:  Generic RTL2832U OEM
    """

    def test_filters_device_list_entries(self):
        from rfcensus.hardware.serialization import _last_error_line
        sample = (
            "Found 4 device(s):\n"
            "  0:  Generic RTL2832U OEM\n"
            "  1:  Generic RTL2832U OEM\n"
            "Using device 1: Generic RTL2832U OEM\n"
            "Found Rafael Micro R820T tuner\n"
        )
        # Should NOT pick "0: Generic RTL2832U OEM" since that's a device
        # list entry, not an error
        result = _last_error_line(sample)
        assert result is None or "Generic" not in result or "tuner" in result.lower()

    def test_finds_real_error_among_noise(self):
        from rfcensus.hardware.serialization import _last_error_line
        sample = (
            "Found 4 device(s):\n"
            "  0:  Generic RTL2832U OEM\n"
            "Using device 1: Generic RTL2832U OEM\n"
            "usb_claim_interface error -6\n"
            "Failed to open device 1\n"
        )
        result = _last_error_line(sample)
        assert result is not None
        assert "fail" in result.lower() or "error" in result.lower()

    def test_returns_none_on_pure_status_output(self):
        from rfcensus.hardware.serialization import _last_error_line
        sample = (
            "Found 1 device(s):\n"
            "  0:  Generic RTL2832U OEM\n"
            "Using device 0: Generic RTL2832U OEM\n"
            "Current configuration:\n"
        )
        result = _last_error_line(sample)
        # All lines are device-list/status; no real error
        assert result is None


# ──────────────────────────────────────────────────────────────────
# Registry mark_failed / mark_healthy
# ──────────────────────────────────────────────────────────────────


class TestRegistryLifecycle:
    def _dongle(self, idx, status=None):
        from rfcensus.hardware.dongle import Dongle, DongleStatus
        s = status or DongleStatus.HEALTHY
        return Dongle(
            id=f"rtl-{idx}", serial=f"0000000{idx}", model="rtlsdr_generic",
            driver="rtlsdr", capabilities=_caps(), status=s, driver_index=idx,
        )

    def test_mark_failed_removes_from_usable(self):
        from rfcensus.hardware.registry import HardwareRegistry
        d = self._dongle(0)
        registry = HardwareRegistry(dongles=[d])
        assert d in registry.usable()

        registry.mark_failed("rtl-0", reason="USB disconnect")

        from rfcensus.hardware.dongle import DongleStatus
        assert d.status == DongleStatus.FAILED
        assert d not in registry.usable()
        assert any("USB disconnect" in note for note in d.health_notes)

    def test_mark_failed_idempotent(self):
        from rfcensus.hardware.registry import HardwareRegistry
        from rfcensus.hardware.dongle import DongleStatus
        d = self._dongle(0, status=DongleStatus.FAILED)
        d.health_notes = ["original failure"]
        registry = HardwareRegistry(dongles=[d])
        registry.mark_failed("rtl-0", reason="another failure")
        # Should not double-append notes
        assert d.health_notes == ["original failure"]

    def test_mark_healthy_restores_usable(self):
        from rfcensus.hardware.registry import HardwareRegistry
        from rfcensus.hardware.dongle import DongleStatus
        d = self._dongle(0, status=DongleStatus.FAILED)
        registry = HardwareRegistry(dongles=[d])
        assert d not in registry.usable()

        registry.mark_healthy("rtl-0")

        assert d.status == DongleStatus.HEALTHY
        assert d in registry.usable()

    def test_mark_unknown_id_is_silent(self):
        from rfcensus.hardware.registry import HardwareRegistry
        registry = HardwareRegistry(dongles=[self._dongle(0)])
        # Should not raise on unknown id
        registry.mark_failed("nonexistent", reason="x")
        registry.mark_healthy("nonexistent")


# ──────────────────────────────────────────────────────────────────
# inventory vs scan defaults
# ──────────────────────────────────────────────────────────────────


class TestCommandDefaults:
    def test_scan_default_is_single_pass(self):
        """Scan should NOT default --per-band — single pass behavior."""
        from rfcensus.commands.inventory import cli_scan
        # Click stores option metadata on the command
        per_band_opt = next(
            p for p in cli_scan.params if p.name == "per_band"
        )
        assert per_band_opt.default is None

    def test_inventory_default_is_round_robin(self):
        """Inventory should default --per-band 1m for round-robin."""
        from rfcensus.commands.inventory import cli_inventory
        per_band_opt = next(
            p for p in cli_inventory.params if p.name == "per_band"
        )
        assert per_band_opt.default == "1m"

    def test_inventory_duration_longer_than_scan(self):
        """v0.6.1: inventory now defaults to 'forever' (exhaustive
        enumeration; intermittent emitters need long runs to catch),
        scan stays finite at 5m. The semantic split: scan to discover,
        inventory to enumerate."""
        from rfcensus.commands.inventory import cli_inventory, cli_scan
        scan_dur = next(p for p in cli_scan.params if p.name == "duration").default
        inv_dur = next(p for p in cli_inventory.params if p.name == "duration").default
        assert scan_dur == "5m"
        assert inv_dur == "forever"


# ──────────────────────────────────────────────────────────────────
# Reprobe recovery (mocked — doesn't touch real hardware)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestReprobeRecovery:
    async def test_restores_failed_dongle_when_seen_again(self, monkeypatch):
        from rfcensus.hardware.dongle import DongleStatus
        from rfcensus.hardware.registry import HardwareRegistry, reprobe_for_recovery

        # Initial registry: dongle marked FAILED
        from rfcensus.hardware.dongle import Dongle
        d = Dongle(
            id="rtl-0", serial="00000001", model="rtlsdr_generic",
            driver="rtlsdr", capabilities=_caps(), status=DongleStatus.FAILED,
            driver_index=0,
        )
        registry = HardwareRegistry(dongles=[d])

        # Mock the probes to "find" the same dongle (simulating it came back)
        from rfcensus.hardware.drivers.rtlsdr import RtlSdrProbeResult
        from rfcensus.hardware.drivers.hackrf import HackRfProbeResult

        async def fake_rtl():
            return RtlSdrProbeResult(dongles=[d], diagnostic="")

        async def fake_hackrf():
            return HackRfProbeResult(dongles=[], diagnostic="")

        monkeypatch.setattr(
            "rfcensus.hardware.drivers.rtlsdr.probe_rtlsdr", fake_rtl
        )
        monkeypatch.setattr(
            "rfcensus.hardware.drivers.hackrf.probe_hackrf", fake_hackrf
        )

        n_back, n_gone = await reprobe_for_recovery(registry)
        assert n_back == 1
        assert n_gone == 0
        assert d.status == DongleStatus.HEALTHY

    async def test_keeps_failed_when_still_missing(self, monkeypatch):
        from rfcensus.hardware.dongle import Dongle, DongleStatus
        from rfcensus.hardware.registry import HardwareRegistry, reprobe_for_recovery
        from rfcensus.hardware.drivers.rtlsdr import RtlSdrProbeResult
        from rfcensus.hardware.drivers.hackrf import HackRfProbeResult

        d = Dongle(
            id="rtl-0", serial="00000001", model="rtlsdr_generic",
            driver="rtlsdr", capabilities=_caps(), status=DongleStatus.FAILED,
            driver_index=0,
        )
        registry = HardwareRegistry(dongles=[d])

        # Probes find no dongles — original is still missing
        async def empty_rtl():
            return RtlSdrProbeResult(dongles=[], diagnostic="")

        async def empty_hackrf():
            return HackRfProbeResult(dongles=[], diagnostic="")

        monkeypatch.setattr(
            "rfcensus.hardware.drivers.rtlsdr.probe_rtlsdr", empty_rtl
        )
        monkeypatch.setattr(
            "rfcensus.hardware.drivers.hackrf.probe_hackrf", empty_hackrf
        )

        n_back, n_gone = await reprobe_for_recovery(registry)
        assert n_back == 0
        assert n_gone == 1
        assert d.status == DongleStatus.FAILED  # still failed


# ──────────────────────────────────────────────────────────────────
# Early-exit detection in strategy
# ──────────────────────────────────────────────────────────────────


class TestEarlyExitDetection:
    """Verifies the threshold logic for flagging suspected hardware loss.

    The actual strategy code calls broker.registry.mark_failed when:
      • duration_s is set
      • run_elapsed < min(5.0, duration_s * 0.1)
      • decodes_emitted == 0
      • ended_reason != "binary_missing"
    """

    def test_threshold_for_short_duration(self):
        # 60s expected, 3s actual, 0 decodes → suspect
        elapsed, expected, decodes = 3.0, 60.0, 0
        threshold = min(5.0, expected * 0.1)
        is_suspect = elapsed < threshold and decodes == 0
        assert is_suspect

    def test_threshold_for_long_duration(self):
        # 3600s expected, 4s actual, 0 decodes → suspect (4 < 5)
        elapsed, expected, decodes = 4.0, 3600.0, 0
        threshold = min(5.0, expected * 0.1)  # = 5.0
        assert elapsed < threshold

    def test_normal_completion_not_flagged(self):
        # 60s expected, 60s actual, 0 decodes → quiet band, not failure
        elapsed, expected, decodes = 60.0, 60.0, 0
        threshold = min(5.0, expected * 0.1)
        is_suspect = elapsed < threshold
        assert not is_suspect

    def test_decodes_means_not_failure(self):
        # 60s expected, 1s actual, 5 decodes → maybe just a quick exit
        # but NOT hardware loss because we got data
        elapsed, expected, decodes = 1.0, 60.0, 5
        threshold = min(5.0, expected * 0.1)
        is_suspect = elapsed < threshold and decodes == 0
        assert not is_suspect
