"""v0.5.41 tests: run_batched_confirmation — the task runner that
captures IQ for a cluster, digitally down-converts to each task's
target frequency, runs chirp analysis, and updates the DB row.

These tests use a mock IQ service (no real dongle required) that
returns synthetic IQ with pre-seeded chirp-like content. The DDC
code path is exercised end-to-end against the real chirp analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from rfcensus.engine.confirmation_queue import (
    BatchedConfirmationTask,
    ConfirmationTask,
)
from rfcensus.engine.confirmation_task import run_batched_confirmation


# ------------------------------------------------------------
# Synthetic chirp generator (for mock IQ captures)
# ------------------------------------------------------------


def _synthesize_lora_chirp(
    *,
    sample_rate: int,
    duration_s: float,
    bandwidth_hz: int,
    center_shift_hz: float = 0.0,
    sf: int = 11,
) -> np.ndarray:
    """Generate synthetic LoRa-style chirp bursts the analyzer can detect.

    Each chirp sweeps linearly from -BW/2 to +BW/2 over its duration.
    Between chirps we insert short silent gaps (below the analyzer's
    amplitude threshold) so analyze_chirps() segments the signal into
    individual chirps and fits a linear model to each.

    For SF=11 at 250 kHz BW: one chirp = 8.192 ms. A 0.5 s capture
    holds ~30 full chirps with gaps, plenty for the analyzer to find
    multiple linear segments.
    """
    chirp_duration_s = (2 ** sf) / bandwidth_hz
    slope_hz_per_sec = bandwidth_hz / chirp_duration_s
    gap_duration_s = chirp_duration_s * 0.2  # 20% gap between chirps

    total_samples = int(sample_rate * duration_s)
    signal = np.zeros(total_samples, dtype=np.complex64)

    np.random.seed(42)
    noise_floor = (
        np.random.randn(total_samples) + 1j * np.random.randn(total_samples)
    ).astype(np.complex64) * np.float32(0.02)
    signal += noise_floor

    # Build successive chirps
    chirp_samples = int(sample_rate * chirp_duration_s)
    gap_samples = int(sample_rate * gap_duration_s)
    period = chirp_samples + gap_samples

    idx = 0
    while idx + chirp_samples < total_samples:
        # Linear sweep from -BW/2 + center_shift to +BW/2 + center_shift
        t = np.arange(chirp_samples, dtype=np.float64) / sample_rate
        inst_freq = center_shift_hz + slope_hz_per_sec * t - bandwidth_hz / 2
        phase = 2 * np.pi * np.cumsum(inst_freq) / sample_rate
        burst = np.exp(1j * phase).astype(np.complex64)
        # Superimpose on noise floor (full amplitude during chirp)
        signal[idx : idx + chirp_samples] = burst + noise_floor[
            idx : idx + chirp_samples
        ] * np.float32(0.1)  # weaker noise during chirp
        # Gap region keeps noise floor as-is (below amplitude threshold)
        idx += period

    return signal


# ------------------------------------------------------------
# Mocks
# ------------------------------------------------------------


@dataclass
class _MockCapture:
    samples: np.ndarray
    sample_rate: int
    freq_hz: int
    started_at: datetime
    dongle_id: str = "mock-dongle"
    driver: str = "rtlsdr"
    truncated: bool = False

    @property
    def bytes_received(self) -> int:
        # uint8 IQ = 2 bytes per sample
        return self.samples.size * 2

    @property
    def duration_s(self) -> float:
        return self.samples.size / self.sample_rate


class MockIQService:
    """Returns synthesized LoRa-chirp IQ captures instead of hitting a
    real dongle. Configurable: if `chirps_at_offsets` is set, only
    captures around those offsets produce chirps; others return noise.

    Tracks call count so tests can verify how many captures happened.
    """

    def __init__(
        self,
        *,
        chirps_at_offsets_hz: dict[int, int] | None = None,
        always_chirp: bool = False,
    ):
        # Map of center_shift_hz → sf (for each chirp we want present)
        self.chirps_at_offsets_hz = chirps_at_offsets_hz or {}
        self.always_chirp = always_chirp
        self.capture_count = 0

    async def capture_with_lease(
        self,
        lease,
        freq_hz: int,
        sample_rate: int,
        duration_s: float,
        *,
        gain: str = "auto",
    ) -> _MockCapture:
        self.capture_count += 1

        num_samples = int(sample_rate * duration_s)
        if self.always_chirp:
            # One chirp at DC (task already DDC'd to baseband)
            samples = _synthesize_lora_chirp(
                sample_rate=sample_rate,
                duration_s=duration_s,
                bandwidth_hz=250_000,
                center_shift_hz=0.0,
                sf=11,
            )
        elif self.chirps_at_offsets_hz:
            # Superpose all configured chirps at their respective
            # offsets (taken relative to the capture center).
            samples = np.zeros(num_samples, dtype=np.complex64)
            for offset_hz, sf in self.chirps_at_offsets_hz.items():
                samples += _synthesize_lora_chirp(
                    sample_rate=sample_rate,
                    duration_s=duration_s,
                    bandwidth_hz=250_000,
                    center_shift_hz=float(offset_hz),
                    sf=sf,
                )
        else:
            # Pure noise — no chirps present
            np.random.seed(0)
            samples = (
                np.random.randn(num_samples)
                + 1j * np.random.randn(num_samples)
            ).astype(np.complex64) * np.float32(0.1)

        return _MockCapture(
            samples=samples,
            sample_rate=sample_rate,
            freq_hz=freq_hz,
            started_at=datetime.now(timezone.utc),
        )


class MockDetectionRepo:
    """Records metadata updates without a real DB."""

    def __init__(self):
        self.updates: list[dict[str, Any]] = []

    async def update_metadata(self, *, detection_id: int, **fields):
        self.updates.append({"detection_id": detection_id, **fields})


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------


def _task(
    detection_id: int,
    freq_hz: int,
    bandwidth_hz: int = 250_000,
) -> ConfirmationTask:
    return ConfirmationTask(
        detection_id=detection_id,
        freq_hz=freq_hz,
        bandwidth_hz=bandwidth_hz,
        technology="lora",
        detector_name="lora",
    )


class TestBatchedRunner:
    @pytest.mark.asyncio
    async def test_single_task_success(self):
        """One task, chirp present from the first capture — should
        confirm quickly and exit."""
        center_freq = 906_875_000
        task = _task(1, center_freq)
        cluster = BatchedConfirmationTask(
            tasks=[task],
            max_duration_s=2.0,  # tight budget for test speed
        )

        iq = MockIQService(always_chirp=True)
        repo = MockDetectionRepo()

        confirmed = await run_batched_confirmation(
            cluster=cluster,
            lease=object(),
            iq_service=iq,
            detection_repo=repo,
            session_id=1,
            subcapture_duration_s=0.3,
        )

        assert 1 in confirmed
        # Exactly one update to the DB for detection_id=1
        assert len(repo.updates) == 1
        update = repo.updates[0]
        assert update["detection_id"] == 1
        assert update["iq_confirmed"] is True
        assert update["estimated_sf"] is not None
        # One capture was enough
        assert iq.capture_count == 1

    @pytest.mark.asyncio
    async def test_timeout_when_no_chirp_present(self):
        """If the mock returns noise, the task should time out with
        no confirmation, and the DB is not updated."""
        task = _task(1, 906_875_000)
        cluster = BatchedConfirmationTask(
            tasks=[task],
            max_duration_s=0.3,  # short timeout for test speed
        )

        iq = MockIQService()  # pure noise
        repo = MockDetectionRepo()

        confirmed = await run_batched_confirmation(
            cluster=cluster,
            lease=object(),
            iq_service=iq,
            detection_repo=repo,
            session_id=1,
            subcapture_duration_s=0.1,
        )

        assert confirmed == set()
        assert len(repo.updates) == 0
        # At least one capture attempted
        assert iq.capture_count >= 1

    @pytest.mark.asyncio
    async def test_progress_callback_invoked(self):
        """Progress callback gets called at least once (start + end)."""
        task = _task(1, 906_875_000)
        cluster = BatchedConfirmationTask(
            tasks=[task], max_duration_s=2.0,
        )

        iq = MockIQService(always_chirp=True)
        repo = MockDetectionRepo()

        calls: list[str] = []
        await run_batched_confirmation(
            cluster=cluster, lease=object(), iq_service=iq,
            detection_repo=repo, session_id=1,
            subcapture_duration_s=0.3,
            progress_cb=lambda msg: calls.append(msg),
        )
        assert len(calls) >= 2  # start + at least one confirm
        assert any("listening" in c for c in calls)

    @pytest.mark.asyncio
    async def test_empty_cluster_returns_empty(self):
        """Defensive: a cluster with no tasks is a no-op."""
        cluster = BatchedConfirmationTask(tasks=[], max_duration_s=0.5)
        iq = MockIQService(always_chirp=True)
        repo = MockDetectionRepo()
        confirmed = await run_batched_confirmation(
            cluster=cluster, lease=object(), iq_service=iq,
            detection_repo=repo, session_id=1,
        )
        assert confirmed == set()
        assert iq.capture_count == 0

    @pytest.mark.asyncio
    async def test_iq_capture_error_keeps_trying(self):
        """A transient IQCaptureError on one sub-capture shouldn't fail
        the whole task — the loop retries on next iteration."""
        from rfcensus.spectrum.iq_capture import IQCaptureError

        call_count = {"n": 0}

        class _FlakyIQ:
            async def capture_with_lease(self, lease, **kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise IQCaptureError("transient glitch")
                # Second call succeeds with chirp
                num = int(kwargs["sample_rate"] * kwargs["duration_s"])
                samples = _synthesize_lora_chirp(
                    sample_rate=kwargs["sample_rate"],
                    duration_s=kwargs["duration_s"],
                    bandwidth_hz=250_000, sf=11,
                )
                return _MockCapture(
                    samples=samples,
                    sample_rate=kwargs["sample_rate"],
                    freq_hz=kwargs["freq_hz"],
                    started_at=datetime.now(timezone.utc),
                )

        task = _task(1, 906_875_000)
        cluster = BatchedConfirmationTask(
            tasks=[task], max_duration_s=2.0,
        )
        repo = MockDetectionRepo()
        confirmed = await run_batched_confirmation(
            cluster=cluster, lease=object(), iq_service=_FlakyIQ(),
            detection_repo=repo, session_id=1,
            subcapture_duration_s=0.3,
        )
        assert 1 in confirmed
        assert call_count["n"] >= 2
