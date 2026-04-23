"""hackrf_sweep spectrum backend.

hackrf_sweep covers 1 MHz to 6 GHz at gigahertz-per-second rates when
run at coarse resolutions. Output format (stdout) is one line per sweep
block:

  YYYY-MM-DD, HH:MM:SS.SSS, hz_low, hz_high, hz_bin_width, num_samples, dB, dB, ...

We emit one PowerSample per bin. HackRF sweeps in 20 MHz chunks across
the requested range, so one full sweep from (say) 100 MHz to 1 GHz
produces ~45 blocks of samples.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone

from rfcensus.hardware.broker import DongleLease
from rfcensus.spectrum.backend import PowerSample, SpectrumBackend, SpectrumSweepSpec
from rfcensus.utils.async_subprocess import (
    BinaryNotFoundError,
    ManagedProcess,
    ProcessConfig,
    which,
)
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


class HackRFSweepBackend(SpectrumBackend):
    name = "hackrf_sweep"
    max_range = (1_000_000, 6_000_000_000)
    # hackrf_sweep at coarse bin widths is ~8 GHz/sec; at finer widths
    # more like 100-500 MHz/sec
    sweep_rate_hz_per_sec = 2_000_000_000

    @classmethod
    def available_on(cls, lease: DongleLease) -> bool:
        return lease.dongle.driver == "hackrf" and which("hackrf_sweep") is not None

    async def sweep(
        self, lease: DongleLease, spec: SpectrumSweepSpec
    ) -> AsyncIterator[PowerSample]:
        binary = which("hackrf_sweep")
        if binary is None:
            raise BinaryNotFoundError("hackrf_sweep not on PATH")

        low_mhz = max(spec.freq_low, self.max_range[0]) // 1_000_000
        high_mhz = min(spec.freq_high, self.max_range[1]) // 1_000_000
        if high_mhz <= low_mhz:
            return

        # hackrf_sweep takes -w for bin width in Hz, -f for freq range in MHz
        args = [
            binary,
            "-f", f"{low_mhz}:{high_mhz}",
            "-w", str(spec.bin_width_hz),
            "-l", "32",  # LNA gain
            "-g", "16",  # VGA gain
        ]
        if spec.duration_s is not None:
            # -N counts number of sweeps; we convert duration to an approximate count
            # based on our estimated sweep rate
            span_hz = (high_mhz - low_mhz) * 1_000_000
            sweep_time_s = max(0.1, span_hz / self.sweep_rate_hz_per_sec)
            num_sweeps = max(1, int(spec.duration_s / sweep_time_s))
            args += ["-N", str(num_sweeps)]

        proc = ManagedProcess(
            ProcessConfig(
                name=f"hackrf_sweep[{lease.dongle.id}]",
                args=args,
                log_stderr=False,
                kill_timeout_s=5.0,
            )
        )
        await proc.start()
        try:
            async for line in proc.stdout_lines():
                for sample in _parse_csv_line(line):
                    yield sample
        finally:
            await proc.stop()


def _parse_csv_line(line: str) -> list[PowerSample]:
    line = line.strip()
    if not line:
        return []
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        return []
    try:
        date = parts[0]
        time = parts[1]
        hz_low = int(parts[2])
        hz_high = int(parts[3])
        hz_step = float(parts[4])
        db_values = [float(p) for p in parts[6:] if p]
    except ValueError:
        return []

    try:
        ts = datetime.fromisoformat(f"{date}T{time}").replace(tzinfo=timezone.utc)
    except ValueError:
        ts = datetime.now(timezone.utc)

    samples: list[PowerSample] = []
    for i, db in enumerate(db_values):
        freq = int(hz_low + hz_step * (i + 0.5))
        if freq > hz_high:
            break
        samples.append(
            PowerSample(
                timestamp=ts,
                freq_hz=freq,
                bin_width_hz=int(hz_step),
                power_dbm=db,
            )
        )
    return samples
