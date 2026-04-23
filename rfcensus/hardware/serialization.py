"""RTL-SDR serial-number reserialization.

When multiple RTL-SDR dongles share a serial number (the factory default
'00000001' case is most common), they can't be reliably distinguished by
serial. Driver index works per-boot but shifts on replug, so any config
that references index-based ids breaks when USB enumeration shuffles.

The right fix is to write distinct serials to the EEPROM so each dongle
is permanently identifiable. `rtl_eeprom -d N -s NEW_SERIAL` does this,
but the operation is risky (interrupted writes can brick the EEPROM)
and intimidating to most users.

This module wraps the operation safely:

• Pure planning function (`plan_reserialization`) decides what to write
  without touching hardware. Easy to test exhaustively.
• IO functions (`backup_eeprom`, `write_serial`, `verify_serial`,
  `try_software_reset`) each do one thing, with clear failure modes.
• `execute_plan` orchestrates them with safety checks: backup first,
  verify after, software reset before user-replug, refuse if dongle
  busy. Sequential, not parallel.

This module does NOT decide policy questions like "what about the user's
existing config?" — that lives in the CLI layer (commands/serialize.py
and commands/setup.py). The pure logic here is reusable.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rfcensus.hardware.dongle import Dongle
from rfcensus.utils.async_subprocess import which
from rfcensus.utils.logging import get_logger
from rfcensus.utils.paths import data_dir

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
# Plan dataclasses
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SerialAssignment:
    """A single 'this dongle should get this new serial' decision."""

    driver_index: int
    original_serial: str
    new_serial: str
    model: str
    keeps_original: bool = False  # True if new_serial == original_serial

    @property
    def is_change(self) -> bool:
        return not self.keeps_original


@dataclass(frozen=True)
class ReserializationPlan:
    """The full plan: one assignment per dongle that's part of a colliding group.

    Dongles with already-unique serials are not included — they don't need
    to change. The plan is what we'd execute if the user confirms.
    """

    assignments: tuple[SerialAssignment, ...]
    forbidden_serials: frozenset[str] = field(default_factory=frozenset)

    @property
    def changes(self) -> tuple[SerialAssignment, ...]:
        """Just the dongles whose serial would actually change."""
        return tuple(a for a in self.assignments if a.is_change)

    @property
    def is_empty(self) -> bool:
        return len(self.changes) == 0


# ──────────────────────────────────────────────────────────────────
# Pure planning logic
# ──────────────────────────────────────────────────────────────────


def plan_reserialization(
    detected: list[Dongle],
    existing_config_serials: frozenset[str] = frozenset(),
    *,
    keeper_overrides: dict[str, int] | None = None,
    existing_config: dict[str, dict] | None = None,
) -> ReserializationPlan:
    """Decide which dongles need new serials and what those serials should be.

    Algorithm (per the design discussion):

    1. Group attached dongles by current serial. Skip groups of size 1.
    2. For each group of size k ≥ 2 sharing serial N:
       • One dongle keeps serial N. Selection order:
         (a) `keeper_overrides[N]` — explicit user choice (driver_index)
         (b) Existing config match — if site.toml had this serial mapped
             to a specific model, prefer the dongle of that model
         (c) Lowest driver_index — deterministic fallback
       • Other k-1 dongles get N+1, N+2, ... skipping forbidden values
    3. Forbidden set = (every other attached dongle's serial outside this
       group) ∪ (existing_config_serials) ∪ (already-assigned-this-pass)

    Only RTL-SDR dongles are considered. HackRFs are excluded — their
    factory-burned 128-bit serials don't collide in practice.

    Args:
        keeper_overrides: maps serial → driver_index of the dongle that
            should keep that serial. If absent for a colliding serial,
            falls through to the heuristic default.
        existing_config: maps serial → dongle stanza dict (with 'model',
            'antenna', etc.) from existing site.toml. Used for the model-
            match heuristic.
    """
    keeper_overrides = keeper_overrides or {}
    existing_config = existing_config or {}
    rtlsdr_dongles = [d for d in detected if d.driver == "rtlsdr"]

    # Group by current serial
    by_serial: dict[str, list[Dongle]] = {}
    for d in rtlsdr_dongles:
        if not d.serial:
            continue
        by_serial.setdefault(d.serial, []).append(d)

    # Sort each group by driver_index for deterministic ordering. The
    # actual keeper choice happens below.
    for serial in by_serial:
        by_serial[serial].sort(
            key=lambda d: (d.driver_index if d.driver_index is not None else 999_999)
        )

    # Build base forbidden set: all attached serials + existing config refs.
    # We start with everything; as we assign new serials, we add them too.
    forbidden: set[str] = set(existing_config_serials)
    for d in rtlsdr_dongles:
        if d.serial:
            forbidden.add(d.serial)

    assignments: list[SerialAssignment] = []

    # Process each colliding group in deterministic order (by serial value)
    for serial in sorted(by_serial.keys()):
        group = by_serial[serial]
        if len(group) <= 1:
            continue

        keeper = _choose_keeper(serial, group, keeper_overrides, existing_config)

        # The keeper retains the original serial
        assignments.append(SerialAssignment(
            driver_index=keeper.driver_index if keeper.driver_index is not None else 0,
            original_serial=serial,
            new_serial=serial,
            model=keeper.model,
            keeps_original=True,
        ))
        # `serial` is already in `forbidden`; keeper retaining it is fine

        # Remaining dongles need new serials. Start counting from N+1.
        try:
            base_n = int(serial)
        except ValueError:
            # Non-numeric serial — fall back to 1 as base
            log.warning("non-numeric serial %r; falling back to base 1", serial)
            base_n = 1

        candidate = base_n + 1
        # Process the non-keeper dongles in driver_index order
        non_keepers = [d for d in group if d is not keeper]
        for d in non_keepers:
            # Find next unforbidden value
            while _format_serial(candidate) in forbidden:
                candidate += 1
                if candidate > 99_999_999:
                    raise RuntimeError(
                        "Ran out of 8-digit serial space. This shouldn't happen."
                    )
            new = _format_serial(candidate)
            assignments.append(SerialAssignment(
                driver_index=d.driver_index if d.driver_index is not None else 0,
                original_serial=serial,
                new_serial=new,
                model=d.model,
                keeps_original=False,
            ))
            forbidden.add(new)
            candidate += 1

    return ReserializationPlan(
        assignments=tuple(assignments),
        forbidden_serials=frozenset(forbidden),
    )


def _choose_keeper(
    serial: str,
    group: list[Dongle],
    keeper_overrides: dict[str, int],
    existing_config: dict[str, dict],
) -> Dongle:
    """Pick which dongle in a colliding group keeps the original serial.

    Priority:
      1. Explicit user override (keeper_overrides[serial] = driver_index)
      2. Existing config: if site.toml had `serial=N` mapped to a specific
         model, prefer the dongle whose model matches
      3. Lowest driver_index (deterministic fallback)
    """
    # 1. Explicit user override
    if serial in keeper_overrides:
        target_idx = keeper_overrides[serial]
        for d in group:
            if d.driver_index == target_idx:
                return d
        log.warning(
            "keeper override for serial %s requested driver_index=%d, "
            "but no matching dongle found in group; falling back to default",
            serial, target_idx,
        )

    # 2. Existing config model match
    if serial in existing_config:
        prior_model = existing_config[serial].get("model")
        if prior_model:
            # Prefer a dongle whose model matches; if multiple match, take
            # lowest driver_index among them
            matches = [d for d in group if d.model == prior_model]
            if matches:
                return matches[0]  # group is already sorted by driver_index

    # 3. Lowest driver_index
    return group[0]


def describe_dongle_for_picker(dongle: Dongle, existing_config: dict[str, dict]) -> str:
    """Build a human-readable single-line description of a dongle for the
    interactive serialization picker. Includes model, USB info if any, and
    a "previously configured as ..." note if the dongle's serial appears
    in the existing site.toml.
    """
    parts = [f"{dongle.model:<20s}"]
    parts.append(f"idx={dongle.driver_index}")
    if dongle.usb_bus is not None and dongle.usb_port_path:
        parts.append(f"usb={dongle.usb_bus}:{dongle.usb_port_path}")
    if dongle.serial in existing_config:
        prior = existing_config[dongle.serial]
        ant = prior.get("antenna", "?")
        prior_model = prior.get("model", "")
        match_marker = " ✓" if prior_model == dongle.model else ""
        parts.append(f"(in config: model={prior_model}{match_marker} antenna={ant})")
    return "  ".join(parts)

    return ReserializationPlan(
        assignments=tuple(assignments),
        forbidden_serials=frozenset(forbidden),
    )


def _format_serial(n: int) -> str:
    """8-digit zero-padded decimal: 1 -> '00000001', 42 -> '00000042'."""
    return f"{n:08d}"


# ──────────────────────────────────────────────────────────────────
# IO: backup, write, verify
# ──────────────────────────────────────────────────────────────────


@dataclass
class WriteOutcome:
    """Result of a single write+verify attempt."""

    assignment: SerialAssignment
    backup_path: Path | None = None
    write_success: bool = False
    verify_success: bool = False
    needed_replug: bool = False
    error: str | None = None

    @property
    def fully_succeeded(self) -> bool:
        return self.write_success and self.verify_success


def eeprom_backup_dir() -> Path:
    """Where we keep EEPROM dumps. Created on demand."""
    d = data_dir() / "eeprom_backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def backup_eeprom(driver_index: int, original_serial: str) -> Path:
    """Dump the current EEPROM to a timestamped file. Raises on failure.

    Like preflight_check, this trusts the **file** as the success signal,
    not the exit code. rtl_eeprom can exit 1 even on a successful read
    (confirmed against rtl-sdr 2.0.x — the binary returns the dump's
    own exit status, which isn't always 0). The authoritative signal is
    "file exists with reasonable size."
    """
    if which("rtl_eeprom") is None:
        raise RuntimeError(
            "rtl_eeprom not found on PATH. Install rtl-sdr utilities."
        )

    backup_dir = eeprom_backup_dir()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_serial = re.sub(r"[^A-Za-z0-9]", "_", original_serial or "noserial")
    backup_path = backup_dir / f"{timestamp}_idx{driver_index}_{safe_serial}.bin"

    # rtl_eeprom -d N -r FILE
    proc = await asyncio.create_subprocess_exec(
        "rtl_eeprom", "-d", str(driver_index), "-r", str(backup_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    stdout = stdout_b.decode(errors="replace")
    stderr = stderr_b.decode(errors="replace")

    # Trust the file: if it exists with reasonable size, the dump worked
    # regardless of exit code. RTL-SDR EEPROM is 256 bytes; we allow
    # smaller threshold to catch partial reads worth keeping.
    if backup_path.exists() and backup_path.stat().st_size > 16:
        log.info(
            "backed up EEPROM for idx=%d serial=%s to %s (%d bytes)",
            driver_index, original_serial, backup_path,
            backup_path.stat().st_size,
        )
        return backup_path

    # Real failure — clean up empty file if any, parse stderr for cause
    if backup_path.exists() and backup_path.stat().st_size == 0:
        backup_path.unlink()
    err_text = stderr + "\n" + stdout
    raise RuntimeError(
        f"rtl_eeprom backup failed for index {driver_index}: "
        f"{_last_error_line(err_text) or 'no diagnostic output'}"
    )


async def write_serial(driver_index: int, new_serial: str) -> tuple[bool, str]:
    """Write a new serial to the EEPROM. Returns (success, message).

    rtl_eeprom prints a confirmation prompt that we have to answer 'y' to.
    We pipe 'y\\n' to stdin to bypass the interactive confirmation, since
    we've already taken our own confirmation from the user at the CLI level.
    """
    if which("rtl_eeprom") is None:
        return False, "rtl_eeprom not on PATH"

    proc = await asyncio.create_subprocess_exec(
        "rtl_eeprom", "-d", str(driver_index), "-s", new_serial,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    # rtl_eeprom prompts 'Write new configuration to device [y/n]?' — we
    # pipe 'y\n' to stdin to bypass the interactive confirmation.
    stdout_b, stderr_b, timed_out = await _safe_communicate(
        proc, timeout=10.0, input=b"y\n",
    )
    if timed_out:
        return False, "rtl_eeprom timed out (>10s)"

    stdout = stdout_b.decode(errors="replace")
    stderr = stderr_b.decode(errors="replace")
    combined = stdout + "\n" + stderr

    # Trust the success line over the exit code. rtl_eeprom can exit
    # non-zero on the user's system even when the write succeeded
    # (same exit-code-quirk as the read path). The authoritative signal
    # is the binary's own "successfully written" / "write_success" output.
    if "successfully written" in combined.lower() or "write_success" in combined.lower():
        log.info("wrote new serial %s to dongle idx=%d", new_serial, driver_index)
        return True, ""

    if proc.returncode != 0:
        # No success line AND non-zero exit — real failure
        return False, _last_error_line(combined) or f"exit code {proc.returncode}"

    # Exit 0 with no success line — older rtl_eeprom versions don't
    # always print the line. Trust the exit code.
    log.info(
        "rtl_eeprom for idx=%d exited 0 but no success line; trusting exit code",
        driver_index,
    )
    return True, ""


async def verify_serial_via_rtl_test(
    driver_index: int, expected_serial: str
) -> bool:
    """Re-probe via rtl_test and confirm the serial at this index matches.

    Note: driver_index may shift after a USB re-enumeration. This function
    verifies by index; the caller is responsible for re-probing after a
    replug if indices may have changed.
    """
    if which("rtl_test") is None:
        return False

    proc = await asyncio.create_subprocess_exec(
        "rtl_test", "-t",  # enumerate-only test
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b, timed_out = await _safe_communicate(proc, timeout=8.0)
    if timed_out:
        return False

    output = stdout_b.decode(errors="replace") + stderr_b.decode(errors="replace")
    # Look for "  N:  Vendor, Model, SN: SERIAL"
    for line in output.splitlines():
        m = re.match(r"\s*(\d+):\s*\S.*SN:\s*(\S+)", line)
        if m and int(m.group(1)) == driver_index:
            actual = m.group(2).strip()
            return actual == expected_serial
    return False


async def try_software_reset(driver_index: int) -> bool:
    """Attempt a USB-level reset to trigger re-enumeration.

    Best-effort. If pyusb isn't installed or the reset fails, returns False
    and the caller should fall back to asking the user to physically replug.
    """
    try:
        import usb.core  # type: ignore[import-not-found]
        import usb.util  # type: ignore[import-not-found]
    except ImportError:
        log.debug("pyusb not available; skipping software reset")
        return False

    # RTL-SDR USB IDs (most common): vendor 0x0bda or 0x1d19 or 0x1f4d
    # We can't easily map driver_index→USB device without re-doing enumeration,
    # so just try resetting all attached RTL-SDR-class devices.
    rtl_vendors = [0x0bda, 0x1d19, 0x1f4d, 0x1b80]
    reset_any = False
    for vendor in rtl_vendors:
        try:
            devices = usb.core.find(find_all=True, idVendor=vendor)
            for dev in devices:
                try:
                    dev.reset()
                    reset_any = True
                    log.info("software-reset USB device vendor=0x%04x", vendor)
                except Exception as exc:
                    log.debug("usb reset failed for vendor=0x%04x: %s", vendor, exc)
        except Exception as exc:
            log.debug("usb.core.find failed for vendor=0x%04x: %s", vendor, exc)

    # Give the kernel a moment to re-enumerate
    if reset_any:
        await asyncio.sleep(2.0)
    return reset_any


# ──────────────────────────────────────────────────────────────────
# Hotplug detection (poll-based) for batch unplug + replug flow
# ──────────────────────────────────────────────────────────────────


async def _current_rtl_serials() -> list[str | None]:
    """Snapshot the currently-attached RTL-SDR serial list.

    Returns a list (not a set) because duplicate serials are real and
    we want to count them — e.g. three dongles all reporting 00000001
    pre-serialize should show as a list of length 3.
    """
    from rfcensus.hardware.drivers.rtlsdr import probe_rtlsdr
    try:
        result = await probe_rtlsdr()
        return [d.serial for d in result.dongles]
    except Exception as exc:
        log.debug("probe failed during hotplug poll: %s", exc)
        return []


async def wait_for_serials(
    expected_new_serials: set[str],
    *,
    timeout_s: float | None = None,
    poll_interval_s: float = 0.75,
    on_arrived: callable | None = None,
) -> tuple[bool, set[str], set[str]]:
    """Poll the USB bus until all `expected_new_serials` are present.

    Single-phase model: we don't care about the unplug-replug
    choreography — only that the new serials eventually appear in
    detection. Works for both:

    • Batch flow: user unplugs all dongles, plugs all back, all new
      serials appear within seconds.
    • Sequential flow: user unplugs #1, replugs #1 (we see new serial
      A), unplugs #2, replugs #2 (we see new serial B). Each new serial
      counts as it arrives.
    • Software-reset flow: kernel re-enumerates without physical
      action; new serials appear on the first poll after reset.

    Args:
        expected_new_serials: serials we wrote and expect to see.
        timeout_s: give up after this many seconds. `None` = indefinite
            (caller relies on Ctrl-C as the escape hatch). Default
            indefinite because the user may walk away during a long
            replug session.
        poll_interval_s: how often to re-probe USB.
        on_arrived: optional callback(serial, n_seen, n_total)
            invoked once for each newly-detected expected serial.

    Returns (success, seen_serials, missing_serials).
        success=True when all expected serials are present.
        success=False on timeout (only possible when timeout_s is set).
    """
    start = asyncio.get_event_loop().time()
    seen: set[str] = set()
    n_total = len(expected_new_serials)
    while True:
        if timeout_s is not None:
            if asyncio.get_event_loop().time() - start > timeout_s:
                missing = expected_new_serials - seen
                return False, seen, missing

        current = await _current_rtl_serials()
        current_set = {s for s in current if s}
        new_arrivals = (current_set & expected_new_serials) - seen
        if new_arrivals:
            for s in sorted(new_arrivals):
                seen.add(s)
                if on_arrived:
                    on_arrived(s, len(seen), n_total)
        if seen >= expected_new_serials:
            return True, seen, set()
        await asyncio.sleep(poll_interval_s)


def format_replug_prompt(assignment: SerialAssignment, dongle: Dongle | None = None) -> str:
    """User-facing description of which physical dongle to unplug.

    Uses the most identifying info available: model > USB port path >
    driver index. The dongle parameter, if provided, gives access to
    USB topology info (bus, port path) that's not in the assignment.
    """
    parts = [f"the {assignment.model}"]
    if dongle is not None:
        if dongle.usb_bus is not None and dongle.usb_port_path:
            parts.append(f"at USB bus {dongle.usb_bus} port {dongle.usb_port_path}")
        elif dongle.usb_bus is not None:
            parts.append(f"at USB bus {dongle.usb_bus}")
    if not any("USB" in p for p in parts):
        parts.append(f"(driver index {assignment.driver_index})")
    if assignment.original_serial:
        parts.append(f"with original serial {assignment.original_serial}")
    return " ".join(parts)


def _first_meaningful_line(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("Found ") and not line.startswith("Using "):
            return line
    return None


# ──────────────────────────────────────────────────────────────────
# Subprocess helpers (avoiding the wait_for(communicate()) data-loss bug)
# ──────────────────────────────────────────────────────────────────


async def _safe_communicate(
    proc: asyncio.subprocess.Process,
    *,
    timeout: float,
    input: bytes | None = None,
) -> tuple[bytes, bytes, bool]:
    """Run communicate() with a timeout that doesn't lose buffered output.

    The naive `wait_for(communicate(), timeout)` cancels communicate() on
    timeout, which cancels the internal stdout/stderr reader tasks AND
    discards any data they had buffered. The fix: schedule a kill via
    call_later and let communicate() run to completion.

    Returns (stdout_bytes, stderr_bytes, timed_out).
    """
    loop = asyncio.get_event_loop()
    timed_out_flag = {"value": False}

    def _do_kill() -> None:
        timed_out_flag["value"] = True
        try:
            proc.kill()
        except (ProcessLookupError, AttributeError):
            pass

    kill_handle = loop.call_later(timeout, _do_kill)
    try:
        stdout_b, stderr_b = await proc.communicate(input=input)
    finally:
        kill_handle.cancel()
    return stdout_b, stderr_b, timed_out_flag["value"]


# ──────────────────────────────────────────────────────────────────
# Pre-flight: are all target dongles openable?
# ──────────────────────────────────────────────────────────────────


async def preflight_check(driver_indices: list[int]) -> tuple[bool, list[str]]:
    """Verify each target dongle is openable (no busy / permissions issues)
    BEFORE we start any writes. Returns (all_ok, error_messages).

    Uses a real EEPROM dump to a temp file as the test, not the exit code
    of `rtl_eeprom -d N` (which exits 1 even on success because no
    operation was specified — confirmed against rtl-sdr 2.0.x). The
    success signal is "temp file exists with non-zero size" — the same
    signal we use for the actual backup step, so preflight tests the
    same code path we'll use during the write.
    """
    if which("rtl_eeprom") is None:
        return False, ["rtl_eeprom binary not found; install rtl-sdr utilities"]

    import tempfile

    errors: list[str] = []
    for idx in driver_indices:
        # Real test: dump EEPROM to a temp file. If the file is created
        # with reasonable size, the dongle opened cleanly.
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            tmp_path = Path(tf.name)
        try:
            proc = await asyncio.create_subprocess_exec(
                "rtl_eeprom", "-d", str(idx), "-r", str(tmp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b, timed_out = await _safe_communicate(proc, timeout=5.0)
            if timed_out:
                errors.append(f"dongle idx={idx}: probe timed out")
                continue

            # Success indicator: temp file exists with reasonable size.
            # RTL-SDR EEPROM is 256 bytes; we use a smaller threshold to
            # catch partial reads that should still verify openability.
            if tmp_path.exists() and tmp_path.stat().st_size > 16:
                continue  # All good — the dongle responded with EEPROM data

            # Real failure — parse stderr (where actual errors live) for
            # known causes, falling back to a meaningful line.
            err_text = (
                stderr_b.decode(errors="replace") + "\n"
                + stdout_b.decode(errors="replace")
            )
            if "usb_claim_interface error -6" in err_text or "busy" in err_text.lower():
                errors.append(
                    f"dongle idx={idx}: in use by another process. "
                    f"Stop the other process (`lsof | grep -i rtl`) and rerun."
                )
            elif "usb_open error -3" in err_text or "permission" in err_text.lower():
                errors.append(
                    f"dongle idx={idx}: permission denied. Install rtl-sdr "
                    f"udev rules or add yourself to the plugdev group."
                )
            else:
                last = _last_error_line(err_text) or "unknown error"
                errors.append(f"dongle idx={idx}: probe failed — {last}")
        finally:
            tmp_path.unlink(missing_ok=True)

    return len(errors) == 0, errors


def _last_error_line(text: str) -> str | None:
    """Return the most error-like line in the output.

    Prefers stderr-style lines (containing 'error', 'failed', 'cannot',
    etc.) over informational ones. Falls back to the last non-empty line.
    """
    error_keywords = ("error", "failed", "cannot", "denied", "no such", "unable")
    candidates: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip device-list lines and the "Using device N" status line
        if line.startswith("Found ") or line.startswith("Using ") or line.startswith("Current "):
            continue
        if line.startswith(("0:", "1:", "2:", "3:", "4:", "5:", "6:", "7:", "8:", "9:")):
            continue  # device-list entry like "  0:  Generic RTL2832U OEM"
        candidates.append(line)
    # Prefer lines with error keywords
    for line in reversed(candidates):
        if any(kw in line.lower() for kw in error_keywords):
            return line
    # Fall back to the last meaningful line
    return candidates[-1] if candidates else None
