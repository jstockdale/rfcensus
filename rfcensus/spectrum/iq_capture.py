"""On-demand IQ capture service.

Detectors that need raw IQ (e.g. for chirp autocorrelation) call
`IQCaptureService.capture()` to get a short burst of complex samples.

• **On-demand, not streaming.** IQ is too voluminous for pub/sub.
• **Brokered dongle allocation.** Uses the same broker everyone else uses.
• **Opportunistic.** Failure is expected and fine — detectors that can't
  get a dongle fall back to heuristic-only confidence.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from rfcensus.hardware.broker import (
    AccessMode,
    DongleBroker,
    DongleRequirements,
    NoDongleAvailable,
)
from rfcensus.utils.async_subprocess import BinaryNotFoundError, which
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class IQCapture:
    samples: np.ndarray  # complex64
    freq_hz: int
    sample_rate: int
    started_at: datetime
    dongle_id: str
    driver: str
    truncated: bool = False
    bytes_received: int = 0

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


class IQCaptureError(RuntimeError):
    pass


class IQCaptureService:
    def __init__(self, broker: DongleBroker):
        self.broker = broker

    async def capture(
        self,
        freq_hz: int,
        sample_rate: int,
        duration_s: float,
        *,
        gain: str = "auto",
        prefer_driver: str | None = None,
        timeout_alloc_s: float = 3.0,
    ) -> IQCapture:
        requirements = DongleRequirements(
            freq_hz=freq_hz,
            sample_rate=sample_rate,
            access_mode=AccessMode.EXCLUSIVE,
            prefer_driver=prefer_driver,
        )
        try:
            lease = await self.broker.allocate(
                requirements, consumer="iq_capture", timeout=timeout_alloc_s
            )
        except NoDongleAvailable as exc:
            raise IQCaptureError(f"no dongle available: {exc}") from exc

        try:
            if lease.dongle.driver == "rtlsdr":
                return await self._capture_rtlsdr(
                    lease, freq_hz, sample_rate, duration_s, gain
                )
            if lease.dongle.driver == "hackrf":
                return await self._capture_hackrf(
                    lease, freq_hz, sample_rate, duration_s, gain
                )
            raise IQCaptureError(f"unsupported driver: {lease.dongle.driver}")
        finally:
            await self.broker.release(lease)

    async def _capture_rtlsdr(
        self, lease, freq_hz, sample_rate, duration_s, gain
    ) -> IQCapture:
        binary = which("rtl_sdr")
        if binary is None:
            raise IQCaptureError("rtl_sdr not on PATH")
        num_samples = int(sample_rate * duration_s)
        num_bytes = num_samples * 2

        args = [
            binary, "-f", str(freq_hz), "-s", str(sample_rate),
            "-n", str(num_samples),
        ]
        if lease.dongle.driver_index is not None:
            args += ["-d", str(lease.dongle.driver_index)]
        args += ["-g", "0" if gain == "auto" else str(gain), "-"]

        started = datetime.now(timezone.utc)
        raw = await _read_subprocess_bytes(args, num_bytes, name="rtl_sdr")

        if len(raw) < 2:
            raise IQCaptureError(f"rtl_sdr produced only {len(raw)} bytes")
        u = np.frombuffer(raw[: (len(raw) // 2) * 2], dtype=np.uint8)
        scaled = (u.astype(np.float32) - 127.5) / 127.5
        samples = (scaled[0::2] + 1j * scaled[1::2]).astype(np.complex64)

        return IQCapture(
            samples=samples, freq_hz=freq_hz, sample_rate=sample_rate,
            started_at=started, dongle_id=lease.dongle.id, driver="rtlsdr",
            truncated=(len(raw) < num_bytes), bytes_received=len(raw),
        )

    async def _capture_hackrf(
        self, lease, freq_hz, sample_rate, duration_s, gain
    ) -> IQCapture:
        binary = which("hackrf_transfer")
        if binary is None:
            raise IQCaptureError("hackrf_transfer not on PATH")
        num_samples = int(sample_rate * duration_s)
        num_bytes = num_samples * 2

        args = [
            binary, "-r", "-", "-f", str(freq_hz), "-s", str(sample_rate),
            "-n", str(num_samples),
        ]
        if gain == "auto":
            args += ["-l", "24", "-g", "16"]
        else:
            args += ["-l", "24", "-g", str(gain)]

        started = datetime.now(timezone.utc)
        raw = await _read_subprocess_bytes(args, num_bytes, name="hackrf_transfer")

        if len(raw) < 2:
            raise IQCaptureError(f"hackrf_transfer produced only {len(raw)} bytes")
        s = np.frombuffer(raw[: (len(raw) // 2) * 2], dtype=np.int8)
        scaled = s.astype(np.float32) / 127.5
        samples = (scaled[0::2] + 1j * scaled[1::2]).astype(np.complex64)

        return IQCapture(
            samples=samples, freq_hz=freq_hz, sample_rate=sample_rate,
            started_at=started, dongle_id=lease.dongle.id, driver="hackrf",
            truncated=(len(raw) < num_bytes), bytes_received=len(raw),
        )


async def _read_subprocess_bytes(args: list[str], num_bytes: int, *, name: str) -> bytes:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise BinaryNotFoundError(str(exc)) from exc

    buffer = bytearray()
    try:
        assert proc.stdout is not None
        while len(buffer) < num_bytes:
            remaining = num_bytes - len(buffer)
            chunk = await proc.stdout.read(min(65536, remaining))
            if not chunk:
                break
            buffer.extend(chunk)
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
    finally:
        if proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
    return bytes(buffer)
