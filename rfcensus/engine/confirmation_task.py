"""v0.5.41: batched confirmation task runner.

Takes a BatchedConfirmationTask (a cluster of LoRa detections sharing
one IQ capture window) and runs it against an allocated dongle lease.
For each capture:

  1. Capture ~2 s of IQ at the cluster's center frequency.
  2. For each still-outstanding task in the cluster, DDC to baseband
     at that task's target frequency and low-pass to the task's
     bandwidth.
  3. Run chirp analysis on the extracted signal.
  4. If a chirp is detected, compute SF from slope + variant label,
     update the DB row, mark that task as confirmed.
  5. Loop — remaining outstanding tasks keep listening.

Exits when either all tasks are confirmed or max_duration_s elapses.

Design notes
============

**Why a loop rather than a one-shot capture.** LoRa transmitters are
bursty. A 500 ms capture aimed at a Meshtastic beacon that bursts
every 2 minutes has a ~5% chance of catching the beacon. Looping for
2 minutes with 2 s captures gets the probability to near-certainty
while keeping memory footprint bounded (each capture is ~20 MB at 2.4
Msps complex64 for 2 s, freed between iterations).

**Early exit.** Once all tasks in a cluster report chirps, we return
immediately. Frees the dongle for the next wave's primary tasks.

**Timeout handling.** If some tasks haven't seen a chirp by
max_duration_s, they're marked abandoned with a WARNING. The
DetectionRepo row retains estimated_sf=None; the original detection
(with the wide-channel evidence) stands.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rfcensus.engine.confirmation_queue import BatchedConfirmationTask
from rfcensus.spectrum.chirp_analysis import analyze_chirps
from rfcensus.spectrum.iq_capture import IQCaptureError, IQCaptureService
from rfcensus.tools.dsp import digital_downconvert
from rfcensus.utils.logging import get_logger

if TYPE_CHECKING:
    from rfcensus.storage.repositories import DetectionRepo

log = get_logger(__name__)


# How long each sub-capture inside the listening loop lasts. 2 seconds
# is long enough to catch typical Meshtastic bursts (0.5-1.5 s) with
# margin, short enough that we check every task frequently.
DEFAULT_SUBCAPTURE_DURATION_S = 2.0

# Minimum chirp_confidence from analyze_chirps to accept as a hit.
# Same threshold as the legacy in-line confirmation path.
CHIRP_CONFIDENCE_THRESHOLD = 0.5


async def run_batched_confirmation(
    cluster: BatchedConfirmationTask,
    lease,
    iq_service: IQCaptureService,
    detection_repo: "DetectionRepo",
    *,
    session_id: int,
    subcapture_duration_s: float = DEFAULT_SUBCAPTURE_DURATION_S,
    progress_cb: "callable | None" = None,
) -> set[int]:
    """Execute one batched confirmation cluster. Returns the set of
    detection_ids that were successfully confirmed.

    The caller is responsible for allocating the lease and releasing
    it (the scheduler does this through the same path as primary
    tasks). This function assumes the lease covers the cluster's
    tuning range.

    `progress_cb`, if provided, is called with short text updates
    suitable for emitting to the user's terminal (one call per chirp
    detected, plus start/end).
    """
    from rfcensus.detectors.builtin.lora import (
        _estimate_sf_from_slope,
        _label_variant,
    )

    if not cluster.tasks:
        return set()

    # Map from detection_id → task for fast lookup; drop entries as
    # tasks are confirmed so we don't re-DDC a known transmitter.
    outstanding = {t.detection_id: t for t in cluster.tasks}
    confirmed_ids: set[int] = set()

    start_time = asyncio.get_event_loop().time()
    deadline = start_time + cluster.max_duration_s

    if progress_cb is not None:
        progress_cb(
            f"[confirm] {cluster.describe()} — listening up to "
            f"{cluster.max_duration_s:.0f}s for bursts..."
        )
    log.info(
        "starting batched confirmation: %s (max %.0fs)",
        cluster.describe(), cluster.max_duration_s,
    )

    iteration = 0
    while outstanding and asyncio.get_event_loop().time() < deadline:
        iteration += 1
        # Cap the last sub-capture so we don't overshoot the deadline
        remaining = deadline - asyncio.get_event_loop().time()
        this_capture_s = min(subcapture_duration_s, max(0.1, remaining))

        try:
            capture = await iq_service.capture_with_lease(
                lease=lease,
                freq_hz=cluster.center_freq_hz,
                sample_rate=cluster.sample_rate,
                duration_s=this_capture_s,
            )
        except IQCaptureError as exc:
            # Per-capture failure — log and move on to the next
            # iteration. The whole task doesn't fail; maybe the next
            # capture works. Common causes: transient USB glitch,
            # rtl_sdr momentary buffer underrun.
            log.warning(
                "confirmation capture %d failed for %s: %s — retrying",
                iteration, cluster.describe(), exc,
            )
            await asyncio.sleep(0.5)
            continue

        # Process each outstanding task against this capture
        completed_this_iter: list[int] = []
        for det_id, task in list(outstanding.items()):
            shift_hz = float(task.freq_hz - cluster.center_freq_hz)
            try:
                baseband = digital_downconvert(
                    capture.samples,
                    source_rate=capture.sample_rate,
                    shift_hz=shift_hz,
                    target_bw_hz=task.bandwidth_hz,
                )
            except Exception:
                log.exception(
                    "DDC failed for %.3f MHz in cluster %s",
                    task.freq_hz / 1e6, cluster.describe(),
                )
                continue

            if baseband.size < 1000:
                # Shouldn't happen for ≥0.5s captures, but be defensive
                continue

            # Compute the actual output rate after DDC decimation
            decimated_rate = int(round(baseband.size / this_capture_s))

            try:
                result = analyze_chirps(baseband, decimated_rate)
            except Exception:
                log.exception(
                    "chirp analysis failed for %.3f MHz",
                    task.freq_hz / 1e6,
                )
                continue

            if (
                result.chirp_confidence > CHIRP_CONFIDENCE_THRESHOLD
                and result.num_chirp_segments >= 1
            ):
                sf = _estimate_sf_from_slope(
                    slope_hz_per_sec=result.mean_slope_hz_per_sec,
                    bandwidth_hz=task.bandwidth_hz,
                )
                variant = None
                if sf is not None:
                    variant = _label_variant(
                        sf=sf, bandwidth_hz=task.bandwidth_hz
                    )

                elapsed = asyncio.get_event_loop().time() - start_time
                log.info(
                    "LoRa confirmation success: %.3f MHz (%d kHz) "
                    "→ SF%s %s after %.1fs / iter %d",
                    task.freq_hz / 1e6, task.bandwidth_hz // 1000,
                    sf, variant, elapsed, iteration,
                )
                if progress_cb is not None:
                    progress_cb(
                        f"[confirm] {task.freq_hz/1e6:.3f} MHz → "
                        f"SF{sf}"
                        + (f" ({variant})" if variant else "")
                        + f" after {elapsed:.0f}s"
                    )

                # Update the detection row in place
                try:
                    await detection_repo.update_metadata(
                        detection_id=det_id,
                        estimated_sf=sf,
                        variant=variant,
                        iq_confirmed=True,
                        chirp_confidence=float(result.chirp_confidence),
                        mean_slope_hz_per_sec=float(
                            result.mean_slope_hz_per_sec
                        ),
                    )
                except Exception:
                    log.exception(
                        "failed to update metadata for detection %d",
                        det_id,
                    )

                confirmed_ids.add(det_id)
                completed_this_iter.append(det_id)

        # Remove confirmed tasks from outstanding so we don't re-check
        # their offsets next iteration
        for det_id in completed_this_iter:
            outstanding.pop(det_id, None)

    # Report outcome
    elapsed = asyncio.get_event_loop().time() - start_time
    if outstanding:
        # Some tasks timed out
        skipped_freqs = ", ".join(
            f"{t.freq_hz/1e6:.3f}" for t in outstanding.values()
        )
        log.warning(
            "confirmation timeout after %.1fs for %d tasks in "
            "cluster@%.3f MHz: %s. These transmitters didn't burst "
            "within the listening window; detection rows kept with "
            "estimated_sf=None.",
            elapsed, len(outstanding),
            cluster.center_freq_hz / 1e6, skipped_freqs,
        )
        if progress_cb is not None:
            progress_cb(
                f"[confirm] timeout: {len(outstanding)} task(s) "
                f"didn't burst in {elapsed:.0f}s: {skipped_freqs} MHz"
            )
    else:
        log.info(
            "confirmation cluster complete: all %d task(s) confirmed "
            "in %.1fs (%d iterations)",
            cluster.size, elapsed, iteration,
        )

    return confirmed_ids
