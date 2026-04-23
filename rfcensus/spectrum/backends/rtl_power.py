"""rtl_power spectrum backend.

Wraps the `rtl_power` binary. rtl_power produces CSV output with one row
per sweep: `date, time, Hz low, Hz high, Hz step, samples, dB dB dB ...`.

Each row therefore represents one sweep of the requested range at the
requested bin width. We emit one PowerSample per bin.
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


class RtlPowerBackend(SpectrumBackend):
    name = "rtl_power"
    max_range = (24_000_000, 1_766_000_000)
    sweep_rate_hz_per_sec = 10_000_000  # rough: ~10 MHz/sec at 10 kHz bins

    @classmethod
    def available_on(cls, lease: DongleLease) -> bool:
        return lease.dongle.driver == "rtlsdr" and which("rtl_power") is not None

    async def sweep(
        self, lease: DongleLease, spec: SpectrumSweepSpec
    ) -> AsyncIterator[PowerSample]:
        binary = which("rtl_power")
        if binary is None:
            raise BinaryNotFoundError("rtl_power not on PATH")

        # Clamp to backend range
        low = max(spec.freq_low, self.max_range[0])
        high = min(spec.freq_high, self.max_range[1])
        if high <= low:
            return

        args = [
            binary,
            "-f", f"{low}:{high}:{spec.bin_width_hz}",
            "-i", f"{max(1, spec.dwell_ms // 1000)}",
            "-1" if spec.duration_s and spec.duration_s <= 1 else "-",
        ]
        # rtl_power accepts device index via -d
        if lease.dongle.driver_index is not None:
            args += ["-d", str(lease.dongle.driver_index)]
        # rtl_power -g sets tuner gain in dB; auto/AGC if omitted
        if spec.gain and spec.gain != "auto":
            args += ["-g", str(spec.gain)]
        if spec.duration_s is not None and spec.duration_s > 1:
            args += ["-e", f"{int(spec.duration_s)}s"]
        # Clean up the trailing "-" placeholder
        args = [a for a in args if a != "-"]

        proc = ManagedProcess(
            ProcessConfig(
                name=f"rtl_power[{lease.dongle.id}]",
                args=args,
                # rtl_power prints device enumeration + PLL warnings
                # + tuner config on every invocation. Now that we know
                # the backend works, these are noise at INFO level.
                # Keep them captured at DEBUG so -v still surfaces them
                # for diagnosis.
                log_stderr=True,
                stderr_log_level="DEBUG",
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
    """Parse one row of rtl_power CSV output into PowerSamples."""
    line = line.strip()
    if not line or line.startswith("#"):
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
        # parts[5] is sample count, parts[6:] are dB values
        db_values = [float(p) for p in parts[6:] if p]
    except ValueError:
        return []

    try:
        ts = datetime.fromisoformat(f"{date}T{time}").replace(tzinfo=timezone.utc)
    except ValueError:
        ts = datetime.now(timezone.utc)

    samples: list[PowerSample] = []
    # The span covered by this row is hz_high - hz_low; bin width is hz_step
    for i, db in enumerate(db_values):
        freq = int(hz_low + hz_step * (i + 0.5))  # bin center
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
