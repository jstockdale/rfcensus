"""Tests for the v0.5.8 single-phase serialize flow.

The two-phase wait_for_unplug + wait_for_replug helpers were replaced
with a single wait_for_serials. The single-phase model supports both
batch (unplug-all-then-replug-all) AND sequential (unplug+replug one
at a time) user patterns — the v0.5.6/.7 two-phase code only worked
for the batch pattern.

Also covers:
  • Batch flow: backup-failure-aborts, partial-write-tolerance
  • TTY check: replug detection fires regardless of --yes when stdin
    is a TTY (this is the v0.5.7 fix preserved)
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import patch

import pytest


# ──────────────────────────────────────────────────────────────────
# wait_for_serials — single-phase, any-pattern detection
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestWaitForSerials:
    """The new single-phase helper. Critical: must work for BOTH
    batch unplug-all-replug-all AND sequential unplug-replug-one-at-a-time
    patterns. The old two-phase helpers required the batch pattern."""

    async def test_batch_pattern_all_serials_appear_at_once(self, monkeypatch):
        """User unplugs all, then plugs all back — all expected serials
        appear simultaneously on the next probe."""
        from rfcensus.hardware import serialization

        responses = [
            ["00000005"],                                          # mid-replug
            ["00000005", "00000002", "00000003"],                  # all back at once
        ]
        call_idx = {"i": 0}

        async def fake_probe():
            i = call_idx["i"]
            call_idx["i"] = min(i + 1, len(responses) - 1)
            return responses[i]

        monkeypatch.setattr(serialization, "_current_rtl_serials", fake_probe)

        arrived: list[str] = []

        def on_arrived(serial, n_seen, n_total):
            arrived.append(serial)

        success, seen, missing = await serialization.wait_for_serials(
            expected_new_serials={"00000002", "00000003"},
            timeout_s=5.0, poll_interval_s=0.01,
            on_arrived=on_arrived,
        )
        assert success is True
        assert seen == {"00000002", "00000003"}
        assert missing == set()
        # Both arrived at the same poll — order in callback is sorted
        assert sorted(arrived) == ["00000002", "00000003"]

    async def test_sequential_pattern_one_serial_at_a_time(self, monkeypatch):
        """User unplugs and replugs one dongle, waits, then does the
        next. This is the case that was BROKEN in v0.5.6/.7 because the
        two-phase model required all dongles unplugged first."""
        from rfcensus.hardware import serialization

        # Stage 1: nothing replugged yet (kernel still has old serials)
        # Stage 2: user unplugged-replugged dongle 1, new serial visible
        # Stage 3: user unplugged-replugged dongle 2, new serial visible
        responses = [
            ["00000005"],                                  # nothing yet
            ["00000005", "00000002"],                      # first new serial
            ["00000005", "00000002"],                      # waiting for second
            ["00000005", "00000002", "00000003"],          # second new serial
        ]
        call_idx = {"i": 0}

        async def fake_probe():
            i = call_idx["i"]
            call_idx["i"] = min(i + 1, len(responses) - 1)
            return responses[i]

        monkeypatch.setattr(serialization, "_current_rtl_serials", fake_probe)

        arrived: list[tuple[str, int, int]] = []

        def on_arrived(serial, n_seen, n_total):
            arrived.append((serial, n_seen, n_total))

        success, seen, missing = await serialization.wait_for_serials(
            expected_new_serials={"00000002", "00000003"},
            timeout_s=5.0, poll_interval_s=0.01,
            on_arrived=on_arrived,
        )
        assert success is True
        assert seen == {"00000002", "00000003"}
        # Each serial fired callback exactly once, in arrival order
        assert arrived == [("00000002", 1, 2), ("00000003", 2, 2)]

    async def test_indefinite_timeout_default(self, monkeypatch):
        """Default timeout is None (indefinite). User may walk away;
        we should keep polling forever rather than timing out at 60s
        or 120s like the old helpers did."""
        from rfcensus.hardware import serialization
        import inspect
        sig = inspect.signature(serialization.wait_for_serials)
        assert sig.parameters["timeout_s"].default is None

    async def test_finite_timeout_works_when_set(self, monkeypatch):
        """If a caller explicitly sets a timeout, it should fire."""
        from rfcensus.hardware import serialization

        async def fake_probe():
            return []  # nothing ever appears

        monkeypatch.setattr(serialization, "_current_rtl_serials", fake_probe)

        success, seen, missing = await serialization.wait_for_serials(
            expected_new_serials={"00000002"},
            timeout_s=0.05, poll_interval_s=0.01,
        )
        assert success is False
        assert missing == {"00000002"}

    async def test_unrelated_serials_dont_count(self, monkeypatch):
        """Only the serials we wrote count toward completion."""
        from rfcensus.hardware import serialization

        async def fake_probe():
            return ["99999999", "12345678"]  # neither expected

        monkeypatch.setattr(serialization, "_current_rtl_serials", fake_probe)

        success, seen, missing = await serialization.wait_for_serials(
            expected_new_serials={"00000002", "00000003"},
            timeout_s=0.05, poll_interval_s=0.01,
        )
        assert success is False
        assert seen == set()
        assert missing == {"00000002", "00000003"}

    async def test_callback_fires_once_per_serial_no_duplicates(self, monkeypatch):
        """Even if a serial stays visible across multiple polls, the
        callback should fire exactly once for it."""
        from rfcensus.hardware import serialization

        responses = [
            ["00000002"],  # first poll: new serial visible
            ["00000002"],  # poll again: still visible (don't re-fire callback)
            ["00000002", "00000003"],  # third: second serial appears
        ]
        call_idx = {"i": 0}

        async def fake_probe():
            i = call_idx["i"]
            call_idx["i"] = min(i + 1, len(responses) - 1)
            return responses[i]

        monkeypatch.setattr(serialization, "_current_rtl_serials", fake_probe)

        arrived: list[str] = []

        def on_arrived(serial, n_seen, n_total):
            arrived.append(serial)

        await serialization.wait_for_serials(
            expected_new_serials={"00000002", "00000003"},
            timeout_s=5.0, poll_interval_s=0.01,
            on_arrived=on_arrived,
        )
        # Each serial callback exactly once
        assert arrived == ["00000002", "00000003"]


# ──────────────────────────────────────────────────────────────────
# _execute_batch flow control — preserved from v0.5.6
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestExecuteBatchFlow:
    async def test_aborts_writes_when_backup_fails(self, monkeypatch, tmp_path):
        """If even one backup fails, no writes should be attempted —
        we don't want to write without a recovery path."""
        from rfcensus.commands import serialize as cmd
        from rfcensus.hardware.serialization import (
            ReserializationPlan, SerialAssignment,
        )

        plan = ReserializationPlan(
            assignments=(
                SerialAssignment(
                    driver_index=1, original_serial="00000001",
                    new_serial="00000002", model="rtlsdr_generic",
                    keeps_original=False,
                ),
                SerialAssignment(
                    driver_index=2, original_serial="00000001",
                    new_serial="00000003", model="rtlsdr_generic",
                    keeps_original=False,
                ),
            ),
            forbidden_serials=frozenset(),
        )

        call_idx = {"i": 0}

        async def fake_backup(idx, serial):
            call_idx["i"] += 1
            if call_idx["i"] == 1:
                f = tmp_path / "ok.bin"
                f.write_bytes(b"\x00" * 256)
                return f
            raise RuntimeError("backup failed: usb error")

        write_called = {"n": 0}

        async def fake_write(idx, serial):
            write_called["n"] += 1
            return True, ""

        monkeypatch.setattr(cmd, "backup_eeprom", fake_backup)
        monkeypatch.setattr(cmd, "write_serial", fake_write)

        outcomes = await cmd._execute_batch(plan, [], yes=True)

        # No writes should have happened — backup failure aborts before write phase
        assert write_called["n"] == 0
        assert outcomes[1].error and "backup" in outcomes[1].error.lower()

    async def test_continues_through_partial_write_failure(
        self, monkeypatch, tmp_path
    ):
        """If one write fails but others succeed, we should still
        proceed (the failed one is just reported in outcomes)."""
        from rfcensus.commands import serialize as cmd
        from rfcensus.hardware.serialization import (
            ReserializationPlan, SerialAssignment,
        )

        plan = ReserializationPlan(
            assignments=(
                SerialAssignment(
                    driver_index=1, original_serial="00000001",
                    new_serial="00000002", model="rtlsdr_generic",
                    keeps_original=False,
                ),
                SerialAssignment(
                    driver_index=2, original_serial="00000001",
                    new_serial="00000003", model="rtlsdr_generic",
                    keeps_original=False,
                ),
            ),
            forbidden_serials=frozenset(),
        )

        async def fake_backup(idx, serial):
            f = tmp_path / f"backup_{idx}.bin"
            f.write_bytes(b"\x00" * 256)
            return f

        async def fake_write(idx, serial):
            if idx == 1:
                return True, ""
            return False, "rtl_eeprom permission denied"

        async def fake_reset(idx):
            return False

        async def fake_serials():
            return ["00000002"]

        monkeypatch.setattr(cmd, "backup_eeprom", fake_backup)
        monkeypatch.setattr(cmd, "write_serial", fake_write)
        monkeypatch.setattr(cmd, "try_software_reset", fake_reset)
        monkeypatch.setattr(cmd, "_current_rtl_serials", fake_serials)

        outcomes = await cmd._execute_batch(plan, [], yes=True)

        assert len(outcomes) == 2
        assert outcomes[0].write_success is True
        assert outcomes[1].write_success is False
        assert "permission denied" in outcomes[1].error


# ──────────────────────────────────────────────────────────────────
# TTY check (v0.5.7 regression preserved)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestReplugDetectionTTYCheck:
    """v0.5.7 fix: --yes must NOT bypass replug detection when stdin
    is a TTY. Replugging is a content/state operation, not a
    yes/no confirmation."""

    async def test_yes_does_not_skip_when_tty(self, monkeypatch, tmp_path):
        from rfcensus.commands import serialize as cmd
        from rfcensus.hardware.serialization import (
            ReserializationPlan, SerialAssignment,
        )

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        plan = ReserializationPlan(
            assignments=(
                SerialAssignment(
                    driver_index=0, original_serial="00000001",
                    new_serial="00000002", model="rtlsdr_generic",
                    keeps_original=False,
                ),
            ),
            forbidden_serials=frozenset(),
        )

        async def fake_backup(idx, serial):
            f = tmp_path / f"backup_{idx}.bin"
            f.write_bytes(b"\x00" * 256)
            return f

        async def fake_write(idx, serial):
            return True, ""

        async def fake_reset(idx):
            return False

        wait_called = {"n": 0}

        async def fake_wait_for_serials(expected, **kw):
            wait_called["n"] += 1
            return True, set(expected), set()

        async def fake_serials():
            return ["00000002"]

        monkeypatch.setattr(cmd, "backup_eeprom", fake_backup)
        monkeypatch.setattr(cmd, "write_serial", fake_write)
        monkeypatch.setattr(cmd, "try_software_reset", fake_reset)
        monkeypatch.setattr(cmd, "_current_rtl_serials", fake_serials)
        monkeypatch.setattr(cmd, "wait_for_serials", fake_wait_for_serials)

        await cmd._execute_batch(plan, [], yes=True)

        assert wait_called["n"] == 1, "v0.5.7 regression: --yes bypassed wait"

    async def test_no_tty_skips_wait(self, monkeypatch, tmp_path):
        from rfcensus.commands import serialize as cmd
        from rfcensus.hardware.serialization import (
            ReserializationPlan, SerialAssignment,
        )

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        plan = ReserializationPlan(
            assignments=(
                SerialAssignment(
                    driver_index=0, original_serial="00000001",
                    new_serial="00000002", model="rtlsdr_generic",
                    keeps_original=False,
                ),
            ),
            forbidden_serials=frozenset(),
        )

        async def fake_backup(idx, serial):
            f = tmp_path / f"backup_{idx}.bin"
            f.write_bytes(b"\x00" * 256)
            return f

        async def fake_write(idx, serial):
            return True, ""

        async def fake_reset(idx):
            return False

        wait_called = {"n": 0}

        async def fake_wait_for_serials(expected, **kw):
            wait_called["n"] += 1
            return True, set(expected), set()

        monkeypatch.setattr(cmd, "backup_eeprom", fake_backup)
        monkeypatch.setattr(cmd, "write_serial", fake_write)
        monkeypatch.setattr(cmd, "try_software_reset", fake_reset)
        monkeypatch.setattr(cmd, "wait_for_serials", fake_wait_for_serials)

        await cmd._execute_batch(plan, [], yes=True)

        assert wait_called["n"] == 0, "no-TTY path should skip wait (legitimate bypass)"
