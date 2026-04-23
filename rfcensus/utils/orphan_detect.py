"""Detection and cleanup of orphan SDR processes.

rfcensus spawns many subprocesses (rtl_tcp, rtl_power, rtl_fm,
multimon-ng, rtlamr, rtl_ais, rtl_433, direwolf). When a session
exits uncleanly — the shell crashes, the user hits the Ctrl-C too
fast, or a shell-pipeline decoder escapes our process group kill —
those subprocesses can linger in the background holding USB
dongles hostage.

The next rfcensus scan then fails mysteriously: rtl_tcp won't claim
the device because someone's already got it open, the fanout times
out waiting for a header, we flag the dongle as dead, and the user
spends 20 minutes debugging a phantom bug.

This module:
  1. Scans `/proc` for processes matching known SDR binary names
     that weren't started by us in this session
  2. Logs them loudly at scan start
  3. Optionally kills them (via --kill-orphans) before proceeding

We use /proc directly rather than `psutil` to avoid pulling in a
dependency for a single-purpose module, and directly rather than
`subprocess.run('pgrep', ...)` to avoid an exec roundtrip and
brittle parsing of pgrep's output across distros.
"""

from __future__ import annotations

import os
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path

from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


# Binary names we care about. These match the names rfcensus spawns
# (via `which()` or absolute paths). Any process whose comm or
# argv[0] matches one of these is a candidate orphan.
#
# We look at basenames only — a user running `/home/jstockdale/go/bin/rtlamr`
# still has `rtlamr` as its comm. This also means we won't falsely
# match `foorrtl_tcp` or `rtl_tcp_new` since comm is usually exact.
SDR_BINARY_NAMES: frozenset[str] = frozenset({
    "rtl_tcp",
    "rtl_power",
    "rtl_fm",
    "rtl_433",
    "rtl_ais",
    "rtl_test",
    "rtl_sdr",
    "rtl_eeprom",
    "rtlamr",
    "multimon-ng",
    "direwolf",
    "hackrf_info",
    "hackrf_transfer",
})


@dataclass(frozen=True)
class OrphanProcess:
    """A running process that looks like an orphan SDR subprocess.

    We capture pid, comm (the kernel's short process name), and the
    full cmdline so diagnostic messages can show the user exactly
    what's hanging around.
    """
    pid: int
    comm: str
    cmdline: str
    age_seconds: float  # seconds since process started, -1 if unknown


def _read_proc_comm(pid: int) -> str | None:
    """Read /proc/<pid>/comm (the short process name, ≤16 chars)."""
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _read_proc_cmdline(pid: int) -> str | None:
    """Read /proc/<pid>/cmdline (null-separated argv) and decode it.

    Returns a space-joined string for human display, or None if the
    process has vanished. Kernel threads have empty cmdline — we
    return None for those too since they're definitely not ours.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if not raw or raw == b"\x00":
        return None
    # argv is null-separated; strip trailing null then replace
    # internal nulls with spaces
    decoded = raw.rstrip(b"\x00").replace(b"\x00", b" ").decode(
        "utf-8", errors="replace"
    )
    return decoded.strip() or None


def _read_proc_age(pid: int) -> float:
    """Approximate age of process <pid> in seconds.

    Uses /proc/<pid>/stat field 22 (starttime in clock ticks since
    boot) and /proc/stat 'btime' (boot time epoch). Returns -1.0 if
    we can't compute it — missing age shouldn't block orphan
    detection.
    """
    try:
        stat_raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, PermissionError, OSError):
        return -1.0
    # stat format has the comm in parens which can contain spaces;
    # everything after the closing paren is space-separated.
    close_paren = stat_raw.rfind(")")
    if close_paren < 0:
        return -1.0
    fields = stat_raw[close_paren + 1:].split()
    # Fields after comm are 1-indexed from state; starttime is
    # field 22 in the man page (so index 22 - 3 = 19 in our split
    # since we chopped comm).
    try:
        starttime_ticks = int(fields[19])
    except (IndexError, ValueError):
        return -1.0
    try:
        clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    except (ValueError, OSError):
        clk_tck = 100  # Conventional default on Linux
    # Boot time
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                btime = int(line.split()[1])
                break
        else:
            return -1.0
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return -1.0
    start_epoch = btime + (starttime_ticks / clk_tck)
    return max(0.0, time.time() - start_epoch)


def find_sdr_orphans(
    *,
    exclude_pids: set[int] | None = None,
    min_age_s: float = 0.0,
) -> list[OrphanProcess]:
    """Scan /proc for SDR-subprocess orphans.

    Parameters
    ----------
    exclude_pids:
        PIDs to ignore (typically our own children). When rfcensus is
        mid-scan and calls this, it should pass its tracked child PIDs
        so it doesn't flag its own live subprocesses as orphans.
    min_age_s:
        Ignore processes younger than this. Useful for startup-time
        checks: a just-spawned subprocess by another tool (e.g. the
        user running `rtl_test` in another terminal 100ms before
        rfcensus started) shouldn't be reported as an orphan until
        it's been alive a while. Pass 0 to report everything.

    Returns
    -------
    List of `OrphanProcess` entries, sorted oldest-first. Empty list
    means no orphans (or we couldn't enumerate /proc, which is
    Non-Linux).
    """
    exclude_pids = exclude_pids or set()
    orphans: list[OrphanProcess] = []

    proc_root = Path("/proc")
    if not proc_root.is_dir():
        # Not Linux — orphan detection is a no-op. macOS users get
        # a best-effort mention but no actual scanning.
        log.debug("orphan detection: /proc not available, skipping")
        return orphans

    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in exclude_pids:
            continue
        if pid == os.getpid():
            continue

        comm = _read_proc_comm(pid)
        if comm is None:
            continue  # process vanished between iterdir and open
        if comm not in SDR_BINARY_NAMES:
            continue

        cmdline = _read_proc_cmdline(pid)
        if cmdline is None:
            continue  # kernel thread or vanished

        age = _read_proc_age(pid)
        if age >= 0.0 and age < min_age_s:
            continue

        orphans.append(OrphanProcess(
            pid=pid, comm=comm, cmdline=cmdline, age_seconds=age,
        ))

    # Oldest first — probably the most likely real orphan (newer
    # matches might be legitimate processes started by the user)
    orphans.sort(key=lambda o: -o.age_seconds if o.age_seconds >= 0 else 0)
    return orphans


def kill_orphans(
    orphans: list[OrphanProcess],
    *,
    sigterm_grace_s: float = 2.0,
) -> tuple[int, int]:
    """SIGTERM each orphan, wait briefly, then SIGKILL survivors.

    Parameters
    ----------
    orphans:
        The list returned by `find_sdr_orphans`.
    sigterm_grace_s:
        How long to wait after SIGTERM before escalating to SIGKILL.
        2 seconds is enough for rtl_tcp/rtl_fm to flush and exit
        cleanly; longer just delays startup.

    Returns
    -------
    Tuple of (killed_cleanly, force_killed) counts.

    Safety notes: we only signal processes whose comm is in our
    SDR_BINARY_NAMES set, so this can't accidentally kill a random
    user process. Raised OSError/PermissionError is logged at
    warning level and counted as skipped — rfcensus may be running
    without root on a multi-user box where the orphans belong to
    another user.
    """
    if not orphans:
        return (0, 0)

    killed_cleanly = 0
    force_killed = 0
    survivors: list[OrphanProcess] = []

    # Phase 1: SIGTERM everyone
    for orphan in orphans:
        try:
            os.kill(orphan.pid, signal.SIGTERM)
            log.info(
                "sent SIGTERM to orphan %s pid=%d (age=%.0fs)",
                orphan.comm, orphan.pid, orphan.age_seconds,
            )
            survivors.append(orphan)
        except ProcessLookupError:
            # Already gone — great
            pass
        except PermissionError:
            log.warning(
                "cannot kill orphan %s pid=%d (permission denied; "
                "likely owned by another user)",
                orphan.comm, orphan.pid,
            )
        except OSError as e:
            log.warning(
                "error killing orphan %s pid=%d: %s",
                orphan.comm, orphan.pid, e,
            )

    # Phase 2: wait, then SIGKILL any survivors
    if survivors:
        time.sleep(sigterm_grace_s)
        for orphan in survivors:
            # Check if it's still there. /proc/<pid>/comm returning
            # None means the process exited.
            if _read_proc_comm(orphan.pid) is None:
                killed_cleanly += 1
                continue
            try:
                os.kill(orphan.pid, signal.SIGKILL)
                force_killed += 1
                log.warning(
                    "SIGKILL'd stubborn orphan %s pid=%d",
                    orphan.comm, orphan.pid,
                )
            except ProcessLookupError:
                killed_cleanly += 1
            except (PermissionError, OSError) as e:
                log.warning(
                    "could not SIGKILL %s pid=%d: %s",
                    orphan.comm, orphan.pid, e,
                )

    return (killed_cleanly, force_killed)


def log_orphans(orphans: list[OrphanProcess]) -> None:
    """Emit a user-facing report about orphan processes.

    Consistent format used by both the scan-startup path and the
    optional doctor command. Empty list logs nothing.
    """
    if not orphans:
        return
    log.warning(
        "⚠ found %d orphan SDR process(es) from a previous session; "
        "they may be holding dongles or ports hostage. Run with "
        "--kill-orphans to clean them up, or kill manually:",
        len(orphans),
    )
    for o in orphans:
        age_str = (
            f"{o.age_seconds:.0f}s" if o.age_seconds >= 0 else "unknown"
        )
        log.warning(
            "  pid=%d comm=%s age=%s cmdline=%s",
            o.pid, o.comm, age_str, o.cmdline[:200],
        )


# Regex to pull the `-d N` device index out of an rtl_* cmdline.
# Matches ` -d 3` (space-separated) or `-d=3` (equals form, though
# rtl_* tools don't actually support this, we accept it defensively).
_RTL_DEVICE_INDEX_RE = re.compile(r"(?:^|\s)-d[ =](\d+)(?:\s|$)")


def guess_orphan_device_indices(orphans: list[OrphanProcess]) -> dict[int, list[OrphanProcess]]:
    """Group orphans by the rtl_* `-d N` device index they reference.

    rtl_tcp / rtl_fm / rtl_power / rtl_433 / rtlamr / rtl_ais all accept
    `-d INDEX` to pick a specific dongle. Parsing this out of the
    cmdline lets us correlate orphans with dongles that failed to
    probe — giving the user a clear "pid 12345 is holding device 3
    (your rtlsdr-00000043)" message instead of an unlinked list.

    Orphans without a parseable `-d N` are placed under key -1
    ("unknown device"). A single orphan can only belong to one key
    (cmdlines normally have only one `-d`).

    Returns a dict mapping device_index → list of orphans.
    """
    by_index: dict[int, list[OrphanProcess]] = {}
    for orphan in orphans:
        m = _RTL_DEVICE_INDEX_RE.search(orphan.cmdline)
        if m:
            idx = int(m.group(1))
        else:
            idx = -1
        by_index.setdefault(idx, []).append(orphan)
    return by_index
