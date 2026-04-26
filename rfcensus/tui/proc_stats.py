"""Cheap process-self resource sampler.

v0.6.14: provides cpu% + RSS for the footer's resource indicator.
Implemented against /proc/self for portability — no psutil dependency,
no shelling out. Two files read per call (~10 µs total on a Pi 5);
called once per tick at 1 Hz, so well under noise.

CPU% is computed as a delta vs the previous sample, so the first call
returns 0.0%. RSS is read fresh each time. Both gracefully degrade to
None on non-Linux systems where /proc/self isn't present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import monotonic


@dataclass
class ProcStats:
    """One snapshot of cpu + memory."""
    cpu_percent: float | None
    rss_bytes: int | None


class ProcSampler:
    """Stateful sampler — keeps the previous CPU jiffies + wall time
    so we can compute a delta for cpu_percent."""

    def __init__(self) -> None:
        self._prev_jiffies: int | None = None
        self._prev_wall: float | None = None
        # Cached page size (constant for the process lifetime)
        try:
            self._page_size = os.sysconf("SC_PAGESIZE")
        except (AttributeError, ValueError, OSError):
            self._page_size = 4096
        # Cached jiffies-per-second
        try:
            self._hz = os.sysconf("SC_CLK_TCK") or 100
        except (AttributeError, ValueError, OSError):
            self._hz = 100

    def sample(self) -> ProcStats:
        """Read /proc/self/stat + /proc/self/statm. Returns a snapshot
        with cpu_percent (None on first call or read failure) and
        rss_bytes (None on read failure)."""
        cpu_percent = self._sample_cpu()
        rss = self._sample_rss()
        return ProcStats(cpu_percent=cpu_percent, rss_bytes=rss)

    def _sample_cpu(self) -> float | None:
        try:
            with open("/proc/self/stat", "rb") as f:
                raw = f.read()
        except OSError:
            return None
        # The 14th and 15th fields are utime + stime (jiffies). The
        # comm field (field 2) can contain spaces and parens, so we
        # split on the trailing ')' to skip past it cleanly.
        try:
            close = raw.rindex(b")")
            tail = raw[close + 2:].split()  # tail[0] = state, tail[11] = utime, tail[12] = stime
            jiffies = int(tail[11]) + int(tail[12])
        except (ValueError, IndexError):
            return None

        now = monotonic()
        if self._prev_jiffies is None or self._prev_wall is None:
            self._prev_jiffies = jiffies
            self._prev_wall = now
            return None

        d_jiffies = jiffies - self._prev_jiffies
        d_wall = now - self._prev_wall
        self._prev_jiffies = jiffies
        self._prev_wall = now
        if d_wall <= 0:
            return None
        # cpu seconds used = d_jiffies / hz; percent = used / d_wall
        return 100.0 * (d_jiffies / self._hz) / d_wall

    def _sample_rss(self) -> int | None:
        try:
            with open("/proc/self/statm", "rb") as f:
                fields = f.read().split()
            # Field 1 (0-indexed) is resident set size in pages
            return int(fields[1]) * self._page_size
        except (OSError, ValueError, IndexError):
            return None


def format_rss(rss_bytes: int | None) -> str:
    """Render bytes as 'NNN MB' or '— MB' if unknown."""
    if rss_bytes is None:
        return "— MB"
    mb = rss_bytes / (1024 * 1024)
    if mb < 1024:
        return f"{mb:.0f} MB"
    return f"{mb / 1024:.1f} GB"


def format_cpu(cpu_percent: float | None) -> str:
    """Render percent as 'NN%' or '—%' if unknown."""
    if cpu_percent is None:
        return "—%"
    return f"{cpu_percent:.0f}%"
