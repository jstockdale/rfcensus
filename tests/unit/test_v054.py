"""Tests for v0.5.4 fixes:
  • backup_eeprom trusts file existence/size, not exit code
  • write_serial trusts the "successfully written" line, not exit code
  • _has_meaningful_choice correctly identifies when picker should fire
  • Picker fires when there's a meaningful choice (different models)
  • Picker silently uses default when all dongles in group are same model
"""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest


def _dongle(idx, serial, model="rtlsdr_generic"):
    """Synthetic Dongle for testing."""
    from rfcensus.hardware.dongle import Dongle, DongleCapabilities, DongleStatus
    caps = DongleCapabilities(
        freq_range_hz=(24_000_000, 1_700_000_000),
        max_sample_rate=2_400_000, bits_per_sample=8,
        bias_tee_capable=False, tcxo_ppm=10.0,
    )
    return Dongle(
        id=f"rtlsdr-{serial}-idx{idx}", serial=serial, model=model,
        driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
        driver_index=idx,
    )


# ──────────────────────────────────────────────────────────────────
# backup_eeprom trusts the file
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestBackupTrustsFile:
    """Regression: the v0.5.3 backup path treated proc.returncode != 0 as
    failure even when the EEPROM dump was successful. rtl_eeprom on the
    user's system exits 1 even on the read-success path."""

    async def test_success_with_nonzero_exit_code(self, tmp_path, monkeypatch):
        """Simulate rtl_eeprom dumping successfully but exiting 1.
        backup_eeprom should treat this as success because the file
        was written with reasonable size."""
        from rfcensus.hardware import serialization

        async def fake_create_subprocess_exec(*args, **kwargs):
            # The 4th positional arg is the output file path
            # args = ("rtl_eeprom", "-d", "1", "-r", "/path/to/file")
            out_path = Path(args[4])
            out_path.write_bytes(b"\x00" * 256)  # full 256-byte EEPROM dump

            mock_proc = MagicMock()
            mock_proc.returncode = 1  # Non-zero exit despite success
            mock_proc.communicate = AsyncMock(
                return_value=(b"Found 1 device(s):\n  0: Generic\n", b"")
            )
            return mock_proc

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        monkeypatch.setattr(
            serialization, "eeprom_backup_dir", lambda: tmp_path
        )
        monkeypatch.setattr(serialization, "which", lambda _: "/usr/bin/rtl_eeprom")

        # Should NOT raise — file is good even though exit code is 1
        result = await serialization.backup_eeprom(
            driver_index=1, original_serial="00000001"
        )
        assert result.exists()
        assert result.stat().st_size == 256

    async def test_real_failure_raises(self, tmp_path, monkeypatch):
        """If the file ISN'T written (or is empty), a real failure
        occurred and we should raise."""
        from rfcensus.hardware import serialization

        async def fake_create_subprocess_exec(*args, **kwargs):
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(
                return_value=(b"", b"usb_claim_interface error -6\n")
            )
            return mock_proc

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        monkeypatch.setattr(
            serialization, "eeprom_backup_dir", lambda: tmp_path
        )
        monkeypatch.setattr(serialization, "which", lambda _: "/usr/bin/rtl_eeprom")

        with pytest.raises(RuntimeError, match="backup failed"):
            await serialization.backup_eeprom(
                driver_index=1, original_serial="00000001"
            )

    async def test_does_not_show_device_list_as_error(self, tmp_path, monkeypatch):
        """When backup truly fails, the error message should not be a
        device-list line like '0: Generic RTL2832U OEM'."""
        from rfcensus.hardware import serialization

        async def fake_create_subprocess_exec(*args, **kwargs):
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(
                b"Found 4 device(s):\n  0:  Generic RTL2832U OEM\n",
                b"usb_claim_interface error -6\n",
            ))
            return mock_proc

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        monkeypatch.setattr(
            serialization, "eeprom_backup_dir", lambda: tmp_path
        )
        monkeypatch.setattr(serialization, "which", lambda _: "/usr/bin/rtl_eeprom")

        with pytest.raises(RuntimeError) as exc_info:
            await serialization.backup_eeprom(
                driver_index=1, original_serial="00000001"
            )
        # The error message should mention the real error, not the
        # device list noise
        msg = str(exc_info.value)
        assert "Generic RTL2832U OEM" not in msg
        assert "usb_claim_interface" in msg or "error" in msg.lower()


# ──────────────────────────────────────────────────────────────────
# write_serial trusts the success line
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestWriteSerialTrustsSuccessLine:
    async def test_success_line_with_nonzero_exit(self, monkeypatch):
        """rtl_eeprom prints 'Configuration successfully written' but
        exits 1 — should be treated as success."""
        from rfcensus.hardware import serialization

        async def fake_create_subprocess_exec(*args, **kwargs):
            mock_proc = MagicMock()
            mock_proc.returncode = 1  # quirky exit code
            mock_proc.communicate = AsyncMock(return_value=(
                b"Configuration successfully written.\n", b"",
            ))
            return mock_proc

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        monkeypatch.setattr(serialization, "which", lambda _: "/usr/bin/rtl_eeprom")
        # Patch _safe_communicate since write_serial uses it
        async def fake_safe_communicate(proc, *, timeout, input=None):
            stdout, stderr = await proc.communicate()
            return stdout, stderr, False
        monkeypatch.setattr(
            serialization, "_safe_communicate", fake_safe_communicate
        )

        ok, msg = await serialization.write_serial(
            driver_index=1, new_serial="00000002"
        )
        assert ok is True

    async def test_no_success_line_nonzero_exit_is_failure(self, monkeypatch):
        """If we don't see the success line AND exit is non-zero,
        that's a real failure."""
        from rfcensus.hardware import serialization

        async def fake_create_subprocess_exec(*args, **kwargs):
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(
                b"", b"usb_claim_interface error -6\n",
            ))
            return mock_proc

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec
        )
        monkeypatch.setattr(serialization, "which", lambda _: "/usr/bin/rtl_eeprom")
        async def fake_safe_communicate(proc, *, timeout, input=None):
            stdout, stderr = await proc.communicate()
            return stdout, stderr, False
        monkeypatch.setattr(
            serialization, "_safe_communicate", fake_safe_communicate
        )

        ok, msg = await serialization.write_serial(
            driver_index=1, new_serial="00000002"
        )
        assert ok is False
        assert "usb_claim_interface" in msg or "error" in msg.lower()


# ──────────────────────────────────────────────────────────────────
# Picker meaningful-choice detection
# ──────────────────────────────────────────────────────────────────


class TestHasMeaningfulChoice:
    def test_different_models_in_group_is_meaningful(self):
        """User's V4 case: 2 generics + 1 V4 — clearly user wants to
        pick which one keeps the original serial."""
        from rfcensus.commands.serialize import _has_meaningful_choice
        group = [
            _dongle(0, "00000001", model="rtlsdr_generic"),
            _dongle(1, "00000001", model="rtlsdr_generic"),
            _dongle(2, "00000001", model="rtlsdr_v4"),
        ]
        assert _has_meaningful_choice(group, existing_config={})

    def test_all_same_model_no_config_is_not_meaningful(self):
        """All 3 generics with same serial — they're interchangeable;
        no need to bother the user."""
        from rfcensus.commands.serialize import _has_meaningful_choice
        group = [
            _dongle(0, "00000001"),
            _dongle(1, "00000001"),
            _dongle(2, "00000001"),
        ]
        assert not _has_meaningful_choice(group, existing_config={})

    def test_existing_config_makes_choice_meaningful(self):
        """Even if all dongles are same model, if existing config has
        a model preference for this serial, we should ask to confirm
        (since the user has a history with this serial)."""
        from rfcensus.commands.serialize import _has_meaningful_choice
        # All same model, but config previously named this serial
        group = [
            _dongle(0, "00000001", model="rtlsdr_v4"),
            _dongle(1, "00000001", model="rtlsdr_v4"),
        ]
        existing = {"00000001": {"model": "rtlsdr_v4", "antenna": "whip_915"}}
        assert _has_meaningful_choice(group, existing)

    def test_empty_group(self):
        from rfcensus.commands.serialize import _has_meaningful_choice
        assert not _has_meaningful_choice([], existing_config={})


# ──────────────────────────────────────────────────────────────────
# Picker fires when meaningful + ignores --yes for content choices
# ──────────────────────────────────────────────────────────────────


class TestPickerSemantics:
    """Regression: in v0.5.3, --yes was passed by `setup` when invoking
    `serialize` (to skip the EEPROM-write confirmation), and that ALSO
    suppressed the keeper picker — so the user never got to pick which
    dongle keeps the original serial. The fix: --yes only suppresses
    yes/no confirmations, not content choices."""

    def test_yes_does_not_suppress_picker_when_choice_meaningful(
        self, monkeypatch, capsys
    ):
        from rfcensus.commands import serialize as cmd

        # Force TTY so the picker can fire
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        # Auto-answer "1" (pick the first option)
        monkeypatch.setattr("click.prompt", lambda *a, **kw: 1)

        dongles = [
            _dongle(0, "00000001", model="rtlsdr_generic"),
            _dongle(1, "00000001", model="rtlsdr_v4"),
        ]
        # yes=True passed by setup; picker should STILL fire
        overrides = cmd._prompt_for_keepers(
            dongles, existing_config={}, yes=True,
        )
        assert "00000001" in overrides
        # The captured stdout should include picker text
        captured = capsys.readouterr()
        assert "Multiple dongles share serial" in captured.out

    def test_no_tty_falls_back_silently(self, monkeypatch, capsys):
        from rfcensus.commands import serialize as cmd

        # No TTY (e.g. running in a pipe/script)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        dongles = [
            _dongle(0, "00000001", model="rtlsdr_generic"),
            _dongle(1, "00000001", model="rtlsdr_v4"),
        ]
        overrides = cmd._prompt_for_keepers(
            dongles, existing_config={}, yes=False,
        )
        # Should auto-pick without prompting
        assert "00000001" in overrides
        captured = capsys.readouterr()
        assert "Multiple dongles share serial" not in captured.out

    def test_no_meaningful_choice_skips_silently(self, monkeypatch, capsys):
        from rfcensus.commands import serialize as cmd

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        # All same model — no meaningful choice
        dongles = [
            _dongle(0, "00000001"),
            _dongle(1, "00000001"),
        ]
        overrides = cmd._prompt_for_keepers(
            dongles, existing_config={}, yes=False,
        )
        # Auto-default kicks in, no prompt shown
        assert overrides["00000001"] == 0  # lowest driver_index
        captured = capsys.readouterr()
        assert "Multiple dongles share serial" not in captured.out
