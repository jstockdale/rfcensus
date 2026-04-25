"""Continuous LoRa/LoRaWAN/Meshtastic survey task.

# Why this exists

The original LoRa detection path (rtl_power → WideChannelAggregator →
LoraDetector) never produced detections on real LoRa traffic, even in
known-active ISM bands. The reason is architectural: rtl_power is a
*sweeping* tuner. Adjacent bins are sampled sequentially, ~1 ms apart.
A LoRa chirp lasts 1-10 ms and sweeps across its bandwidth in that time.
The aggregator's simultaneity check (200 ms window across all bins of a
candidate composite) requires multi-bin streaming that rtl_power simply
cannot provide. See `spectrum/wide_channel_aggregator.py` v0.6.5
diagnostic counters for empirical confirmation.

# What this does instead

When a band has `lora_survey = true` and its strategy launches a SHARED
fanout (e.g. 915 ISM with rtl_433 + rtlamr), this task acquires its own
shared lease and connects to the fanout as just another client. It then
reads IQ continuously at the band's full sample rate (2.4 Msps) and runs
chirp-pattern detection on whatever the fanout delivers.

# CPU strategy: cheap energy gate, expensive DSP only when warranted

A naive design would run the full FFT + DDC + chirp analysis on every
chunk of IQ. At 2.4 Msps that's 2-5% CPU continuously even in a quiet
band. The actual design:

  1. Read a chunk (~13 ms of IQ = 64 KB of u8 samples)
  2. Cheap energy gate: compute the chunk's mean power, compare to a
     running noise-floor estimate. If the chunk is quiet (≤ floor + 6 dB),
     drop it and read the next chunk. ~1 µs of work.
  3. Above-floor chunks accumulate into a rolling 250 ms window. When
     the window has enough material (≥ ~600 KB at 2.4 Msps), it's
     handed to `survey_iq_window` for the expensive analysis.
  4. After analysis, the window is cleared and the energy gate
     continues. Refractory list suppresses re-announcement of the
     same (freq, bw) tuple for 60 seconds.

CPU profile target: <2% idle in quiet bands, ~10% during heavy LoRa
traffic. Achievable because the energy gate keeps the expensive DSP
off the hot path 99% of the time.

# Future Meshtastic protocol decoding

This pipeline is the right shape for adding actual Meshtastic packet
decoding later. The DDC step in `survey_iq_window` already produces
baseband samples for each detected chirp; routing those into a
Meshtastic packet decoder (instead of just-detection) adds protocol
work without changing the IO layer. Tracked separately as v0.6.6+ work.
"""

from __future__ import annotations

import asyncio
import contextlib
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import numpy as np

from rfcensus.events import DetectionEvent, EventBus
from rfcensus.spectrum.in_window_survey import (
    DEFAULT_SURVEY_SNR_THRESHOLD_DB,
    SurveyHit,
    survey_iq_window,
)
from rfcensus.utils.logging import get_logger

if TYPE_CHECKING:
    from rfcensus.config.schema import BandConfig
    from rfcensus.hardware.broker import DongleBroker, DongleLease

log = get_logger(__name__)


# ────────────────────────────────────────────────────────────────────
# Tuning constants
# ────────────────────────────────────────────────────────────────────


# Bytes per IQ sample (rtl_tcp ships unsigned 8-bit interleaved I/Q,
# so each sample = 2 bytes).
_BYTES_PER_SAMPLE = 2

# Read chunk size. 64 KB at 2.4 Msps = 32768 IQ samples = ~13.6 ms of
# IQ — a natural unit for the energy gate. Smaller chunks would make
# the gate more responsive but multiply syscall + per-chunk overhead;
# larger chunks would delay the analysis more than necessary.
_READ_CHUNK_BYTES = 64 * 1024

# Analysis window length. survey_iq_window needs ≥ 250 ms (sample_rate / 4
# in its own check) and the chirp analysis benefits from ≥ a few full
# LoRa packets. 250 ms is the floor; we accumulate this much above-floor
# IQ before running the analysis.
_ANALYSIS_WINDOW_S = 0.25

# Energy-gate threshold above the running noise floor (dB). 6 dB ≈ 4×
# power ratio — comfortably above small fluctuations but well below
# any real LoRa burst's peak. Quiet chunks are dropped without further
# work; above-floor chunks accumulate into the analysis window.
_GATE_THRESHOLD_DB = 6.0

# Noise-floor EMA coefficient. 0.02 → ~50-chunk time constant
# (50 × 13.6 ms ≈ 700 ms), so the floor adapts to slow drifts but a
# burst of LoRa traffic doesn't pull it up enough to mask itself.
_NOISE_FLOOR_EMA_ALPHA = 0.02

# Refractory period after a (freq, bw) detection — suppresses
# re-announcement of the same emitter for this many seconds. A LoRa
# gateway transmitting every few seconds would otherwise flood the
# event stream with identical detections.
_REFRACTORY_S = 60.0

# How loose a frequency match counts as "the same emitter" for the
# refractory check. ±100 kHz is generous enough that a slightly drifted
# DDC center matches a previous announcement, tight enough that
# distinct LoRa channels (typically 200+ kHz apart) stay separate.
_REFRACTORY_FREQ_TOLERANCE_HZ = 100_000

# Maximum size of accumulated buffer in samples. If we somehow read
# this much above-floor IQ without ever running analysis (shouldn't
# happen — analysis fires at _ANALYSIS_WINDOW_S worth), drop the
# oldest to prevent unbounded growth. 4× the analysis window is
# defensive headroom.
_MAX_BUFFER_SAMPLES_MULTIPLIER = 4

# Initial noise floor in linear power units. Used until the EMA has
# warmed up. Corresponds to roughly -40 dB normalized power, well
# below any real signal.
_INITIAL_NOISE_FLOOR_LIN = 1e-4

# Logging cadence — emit a "survey running, N chunks, M analyses"
# heartbeat every this many seconds. Helps with operator confidence
# that the task is alive without spamming.
_HEARTBEAT_INTERVAL_S = 60.0

# rtl_tcp command IDs we send on startup
_CMD_SET_FREQ = 0x01
_CMD_SET_SAMPLE_RATE = 0x02

# rtl_tcp greeting header is 12 bytes; we consume but don't use it.
_RTL_TCP_HEADER_SIZE = 12


# ────────────────────────────────────────────────────────────────────
# Stats record (returned from run() for the strategy summary)
# ────────────────────────────────────────────────────────────────────


@dataclass
class LoraSurveyStats:
    """Per-task summary returned from run()."""

    chunks_read: int = 0
    bytes_read: int = 0
    chunks_above_floor: int = 0
    analyses_performed: int = 0
    detections_emitted: int = 0
    suppressed_by_refractory: int = 0
    final_noise_floor_db: float = 0.0
    duration_s: float = 0.0
    ended_reason: str = ""  # "duration", "cancelled", "upstream_eof", "error"
    errors: list[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# The task itself
# ────────────────────────────────────────────────────────────────────


class LoraSurveyTask:
    """Continuous chirp-pattern detector running on a shared fanout.

    Lifecycle:

        task = LoraSurveyTask(
            broker=broker, event_bus=bus,
            band=band, duration_s=720.0,
            session_id=session_id,
        )
        stats = await task.run()

    Cancellable via `await task.cancel()` or by cancelling the
    surrounding asyncio.Task. Cancellation is clean — the lease is
    released and the fanout connection closed.
    """

    def __init__(
        self,
        *,
        broker: "DongleBroker",
        event_bus: EventBus,
        band: "BandConfig",
        duration_s: float,
        session_id: int | None = None,
        sample_rate: int = 2_400_000,
        # Tunables, exposed mainly for tests
        gate_threshold_db: float = _GATE_THRESHOLD_DB,
        analysis_window_s: float = _ANALYSIS_WINDOW_S,
        snr_threshold_db: float = DEFAULT_SURVEY_SNR_THRESHOLD_DB,
        refractory_s: float = _REFRACTORY_S,
        # Allocate timeout — the fanout is already running so this
        # should resolve quickly. 5s tolerates brief contention.
        allocate_timeout_s: float = 5.0,
    ) -> None:
        self.broker = broker
        self.event_bus = event_bus
        self.band = band
        self.duration_s = duration_s
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.gate_threshold_db = gate_threshold_db
        self.analysis_window_s = analysis_window_s
        self.snr_threshold_db = snr_threshold_db
        self.refractory_s = refractory_s
        self.allocate_timeout_s = allocate_timeout_s

        # Center frequency for the survey: the band's center. The
        # fanout is already tuned to this frequency for the primary
        # decoders; we receive the same IQ they do.
        self.center_hz = (band.freq_low + band.freq_high) // 2

        # Internal lifecycle state
        self._cancelled = asyncio.Event()
        self._lease: Optional["DongleLease"] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

        # Refractory list: (center_freq_hz, bandwidth_hz) → monotonic
        # time of last announcement. We don't bother to prune it; LoRa
        # has only 3 standard bandwidths and a finite number of channels
        # in the band, so it stays small.
        self._refractory: dict[tuple[int, int], float] = {}

    async def cancel(self) -> None:
        """Request cooperative shutdown.

        The run() loop checks `_cancelled` between chunks and exits
        cleanly. Lease release happens in run()'s finally block.
        """
        self._cancelled.set()

    async def run(self) -> LoraSurveyStats:
        """Main loop. Acquires lease, connects to fanout, streams IQ,
        runs analysis when energy gate fires. Returns stats summary.

        Exceptions during setup propagate (so the strategy can mark
        the task as failed); exceptions inside the loop are caught,
        logged, and recorded in stats.errors. The caller should treat
        a returned LoraSurveyStats as success/non-fatal even if it
        contains errors — the task ran and reported what happened.
        """
        from rfcensus.hardware.broker import (
            AccessMode,
            DongleRequirements,
            NoDongleAvailable,
        )

        stats = LoraSurveyStats()
        wall_start = time.monotonic()

        # ── Acquire shared lease on the band's dongle ──
        try:
            requirements = DongleRequirements(
                freq_hz=self.center_hz,
                sample_rate=self.sample_rate,
                access_mode=AccessMode.SHARED,
                prefer_driver="rtlsdr",
                require_suitable_antenna=True,
                band_id=self.band.id,
            )
            self._lease = await self.broker.allocate(
                requirements,
                consumer=f"lora_survey:{self.band.id}",
                timeout=self.allocate_timeout_s,
            )
        except NoDongleAvailable as exc:
            stats.ended_reason = "no_dongle"
            stats.errors.append(f"could not acquire shared lease: {exc}")
            log.warning(
                "lora_survey: could not acquire shared lease for %s: %s. "
                "This usually means no fanout was running when the survey "
                "task started — the strategy launches fanouts via the "
                "decoders, so this races on early bands. The survey will "
                "be skipped for this wave.",
                self.band.id, exc,
            )
            return stats

        endpoint = self._lease.endpoint()
        if endpoint is None:
            stats.ended_reason = "wrong_lease_type"
            stats.errors.append(
                "broker returned exclusive lease; survey requires shared"
            )
            log.error(
                "lora_survey: lease for %s is exclusive (no fanout "
                "endpoint). The band's strategy must run a SHARED "
                "fanout via rtl_433 / rtlamr / similar before this task "
                "can attach. Skipping survey.",
                self.band.id,
            )
            await self.broker.release(self._lease)
            self._lease = None
            return stats

        host, port = endpoint
        log.info(
            "lora_survey[%s]: attaching to fanout at %s:%d "
            "(center=%.3f MHz, sr=%d, duration=%.0fs)",
            self.band.id, host, port,
            self.center_hz / 1e6, self.sample_rate, self.duration_s,
        )

        # ── Connect to fanout, send greeting commands, run loop ──
        try:
            await self._connect_and_handshake(host, port)
            await self._stream_loop(stats, wall_start)
        except asyncio.CancelledError:
            stats.ended_reason = stats.ended_reason or "cancelled"
            raise
        except Exception as exc:
            stats.ended_reason = "error"
            stats.errors.append(f"loop crashed: {exc}")
            log.exception(
                "lora_survey[%s]: loop crashed", self.band.id,
            )
        finally:
            stats.duration_s = time.monotonic() - wall_start
            await self._teardown()

        log.info(
            "lora_survey[%s]: ended (%s) — chunks=%d, above_floor=%d, "
            "analyses=%d, detections=%d, suppressed=%d, floor=%.1f dB, "
            "%.1fs",
            self.band.id, stats.ended_reason or "ok",
            stats.chunks_read, stats.chunks_above_floor,
            stats.analyses_performed, stats.detections_emitted,
            stats.suppressed_by_refractory, stats.final_noise_floor_db,
            stats.duration_s,
        )
        return stats

    # ────────────────────────────────────────────────────────────────
    # Internal: connection + handshake
    # ────────────────────────────────────────────────────────────────

    async def _connect_and_handshake(self, host: str, port: int) -> None:
        """Open TCP to the fanout, drain greeting header, send rate +
        freq commands.

        rtl_tcp clients are expected to issue these commands so the
        upstream tuner adopts the requested center freq + sample rate.
        Since the band's primary decoders set these already, our
        commands are mostly redundant — but sending them keeps us a
        well-behaved client and avoids surprises if the fanout's
        upstream gets re-tuned by another consumer mid-stream.
        """
        self._reader, self._writer = await asyncio.open_connection(
            host=host, port=port,
        )
        # Consume the 12-byte greeting header so subsequent bytes are
        # raw IQ samples.
        try:
            await asyncio.wait_for(
                self._reader.readexactly(_RTL_TCP_HEADER_SIZE),
                timeout=5.0,
            )
        except (asyncio.TimeoutError, asyncio.IncompleteReadError) as exc:
            raise RuntimeError(
                f"fanout greeting header timeout/EOF: {exc}"
            ) from exc

        # Send set_sample_rate + set_freq. rtl_tcp commands are 5 bytes:
        # 1-byte command id, 4-byte big-endian value.
        self._writer.write(
            struct.pack(">BI", _CMD_SET_SAMPLE_RATE, self.sample_rate)
        )
        self._writer.write(
            struct.pack(">BI", _CMD_SET_FREQ, self.center_hz)
        )
        await self._writer.drain()

    # ────────────────────────────────────────────────────────────────
    # Internal: main streaming loop with energy gate
    # ────────────────────────────────────────────────────────────────

    async def _stream_loop(
        self, stats: LoraSurveyStats, wall_start: float,
    ) -> None:
        """Continuous IQ streaming with cheap energy gate + accumulating
        analysis window."""
        deadline = wall_start + self.duration_s
        noise_floor_lin = _INITIAL_NOISE_FLOOR_LIN
        gate_threshold_lin_factor = 10.0 ** (self.gate_threshold_db / 10.0)

        # Pre-allocate the reusable scratch buffer for chunk decoding.
        # frombuffer is zero-copy but the float conversion copies; we
        # reuse the destination to avoid per-chunk allocation pressure.
        chunk_samples = _READ_CHUNK_BYTES // _BYTES_PER_SAMPLE
        analysis_window_samples = int(
            self.sample_rate * self.analysis_window_s
        )
        max_buffer_samples = (
            analysis_window_samples * _MAX_BUFFER_SAMPLES_MULTIPLIER
        )

        # Accumulator for above-floor IQ. List of complex64 numpy
        # arrays, concatenated at analysis time.
        accumulator: list[np.ndarray] = []
        accumulated_samples = 0

        last_heartbeat = wall_start

        while True:
            if self._cancelled.is_set():
                stats.ended_reason = "cancelled"
                return

            now_mono = time.monotonic()
            if now_mono >= deadline:
                stats.ended_reason = "duration"
                return

            # Read one chunk with a short timeout — lets us check
            # cancellation + deadline regularly even on a quiet stream.
            try:
                chunk_remaining = max(0.0, deadline - now_mono)
                read_timeout = min(2.0, chunk_remaining + 0.5)
                raw = await asyncio.wait_for(
                    self._reader.readexactly(_READ_CHUNK_BYTES),
                    timeout=read_timeout,
                )
            except asyncio.TimeoutError:
                # Just loop and re-check cancellation/deadline.
                continue
            except asyncio.IncompleteReadError as exc:
                # Upstream closed (fanout torn down, dongle lost, etc.)
                if exc.partial:
                    # Process the tail before giving up.
                    raw = exc.partial
                    stats.chunks_read += 1
                    stats.bytes_read += len(raw)
                stats.ended_reason = "upstream_eof"
                log.info(
                    "lora_survey[%s]: fanout closed (read %d bytes "
                    "in last partial chunk)",
                    self.band.id,
                    len(exc.partial) if exc.partial else 0,
                )
                return

            stats.chunks_read += 1
            stats.bytes_read += len(raw)

            # ── Decode u8-pair → complex64 ──
            samples = self._decode_chunk(raw)
            if samples.size == 0:
                continue

            # ── Cheap energy gate ──
            chunk_power_lin = float(np.mean(np.abs(samples) ** 2))
            # EMA noise floor — small alpha so brief bursts barely move it.
            noise_floor_lin = (
                (1.0 - _NOISE_FLOOR_EMA_ALPHA) * noise_floor_lin
                + _NOISE_FLOOR_EMA_ALPHA * chunk_power_lin
            )
            gate_lin = noise_floor_lin * gate_threshold_lin_factor
            stats.final_noise_floor_db = (
                10.0 * np.log10(max(noise_floor_lin, 1e-20))
            )

            if chunk_power_lin < gate_lin:
                # Quiet chunk — drop it, continue. This is the common
                # case in idle bands and the whole reason the gate is
                # cheap (one mean + abs over 32K complex samples).
                continue

            stats.chunks_above_floor += 1

            # ── Accumulate for the analysis window ──
            accumulator.append(samples)
            accumulated_samples += samples.size

            # Defensive: trim oldest if accumulator grows past sane
            # bounds. Shouldn't happen with the analysis_window_samples
            # threshold below firing first, but guards against runaway
            # loud bands where every chunk is "above floor".
            while accumulated_samples > max_buffer_samples and accumulator:
                dropped = accumulator.pop(0)
                accumulated_samples -= dropped.size

            if accumulated_samples >= analysis_window_samples:
                window = np.concatenate(accumulator)
                accumulator.clear()
                accumulated_samples = 0
                # Run the expensive analysis off the read hot path —
                # we await it but it's bounded (~10-30 ms typical).
                # The await also yields to other coroutines (fanout
                # writers, decoder cleanup), keeping the event loop
                # responsive.
                await self._analyze_window(window, stats)

            # ── Heartbeat ──
            if now_mono - last_heartbeat > _HEARTBEAT_INTERVAL_S:
                last_heartbeat = now_mono
                log.info(
                    "lora_survey[%s]: heartbeat — chunks=%d, "
                    "above_floor=%d, analyses=%d, detections=%d, "
                    "floor=%.1f dB",
                    self.band.id, stats.chunks_read,
                    stats.chunks_above_floor, stats.analyses_performed,
                    stats.detections_emitted, stats.final_noise_floor_db,
                )

    # ────────────────────────────────────────────────────────────────
    # Internal: chunk decoding + analysis dispatch
    # ────────────────────────────────────────────────────────────────

    def _decode_chunk(self, raw: bytes) -> np.ndarray:
        """Convert rtl_tcp's u8 interleaved I/Q to complex64.

        rtl_tcp ships unsigned 8-bit samples centered on 127.5 (so a
        zero-amplitude signal reads as 0x80 on both I and Q). The DC
        offset removal + scale to ±1.0 mirror what survey_iq_window
        and the rest of the spectrum pipeline expect.
        """
        if len(raw) < 2:
            return np.empty(0, dtype=np.complex64)
        # Truncate odd-length tail (shouldn't happen at our chunk
        # size but defensive).
        n = (len(raw) // 2) * 2
        u = np.frombuffer(raw[:n], dtype=np.uint8)
        scaled = (u.astype(np.float32) - 127.5) / 127.5
        return (scaled[0::2] + 1j * scaled[1::2]).astype(np.complex64)

    async def _analyze_window(
        self, window: np.ndarray, stats: LoraSurveyStats,
    ) -> None:
        """Run survey_iq_window on the accumulated above-floor IQ and
        emit a DetectionEvent for each new hit.

        v0.6.8: survey_iq_window does heavy DSP — Welch PSD, candidate
        finding, multiple DDC operations, multiple analyze_chirps and
        classify_sf_dechirp passes. On a 1-second 2.4 MHz IQ window
        (4 MB) this can take 100s of ms to seconds of CPU. If we ran
        it directly on the asyncio event loop, the loop would stall
        long enough that the SDR fanout couldn't read from upstream
        rtl_tcp fast enough — and rtl_433's 3-second async-read
        watchdog would fire, killing every shared rtl_433 client with
        exit code 3 every time the survey emits a detection. We saw
        exactly that pattern in real scans: rtl_433 dying within 1s
        of the first lora_survey hit, fanout reporting 0 dropped
        chunks, lora_survey itself surviving (because it's the cause).

        Solution: asyncio.to_thread. The DSP runs on a worker thread,
        the event loop stays free to service fanout I/O for OTHER
        consumers. We pay one thread-context-switch + the GIL
        contention, both negligible compared to multi-second blocks.
        """
        try:
            hits = await asyncio.to_thread(
                survey_iq_window,
                window,
                sample_rate=self.sample_rate,
                capture_center_hz=self.center_hz,
                snr_threshold_db=self.snr_threshold_db,
            )
        except Exception:
            log.exception(
                "lora_survey[%s]: analysis crashed (continuing)",
                self.band.id,
            )
            return
        finally:
            stats.analyses_performed += 1

        for hit in hits:
            if not self._is_announceable(hit):
                stats.suppressed_by_refractory += 1
                continue
            await self._emit_detection(hit, stats)
            self._mark_refractory(hit)
            # Yield between detections so a burst doesn't block the
            # event loop on bus.publish backpressure.
            await asyncio.sleep(0)

    def _is_announceable(self, hit: SurveyHit) -> bool:
        """Refractory check — true if no recent announcement matches
        within `_REFRACTORY_FREQ_TOLERANCE_HZ` and the same bandwidth.
        """
        now_mono = time.monotonic()
        for (freq, bw), last_t in self._refractory.items():
            if bw != hit.bandwidth_hz:
                continue
            if abs(freq - hit.freq_hz) > _REFRACTORY_FREQ_TOLERANCE_HZ:
                continue
            if (now_mono - last_t) < self.refractory_s:
                return False
        return True

    def _mark_refractory(self, hit: SurveyHit) -> None:
        """Record that we just announced this (freq, bw)."""
        self._refractory[(hit.freq_hz, hit.bandwidth_hz)] = time.monotonic()

    async def _emit_detection(
        self, hit: SurveyHit, stats: LoraSurveyStats,
    ) -> None:
        """Publish a DetectionEvent for this hit.

        v0.6.8: SF now comes from the reference-dechirp classifier
        (classify_sf_dechirp), stamped onto hit.chirp_analysis by
        survey_iq_window. The legacy slope-based estimator is no
        longer consulted — it produced ~SF7 misclassifications for
        SF9 (MediumFast) traffic in real captures because real LoRa
        is contiguous chirps, not chirps-with-gaps.

        The SF confidence (best/second-best dechirp peak ratio) is
        used as a multiplier on the overall detection confidence, so
        weak SF discriminations downgrade the report rather than
        present as confident-but-wrong.
        """
        from rfcensus.spectrum.chirp_analysis import label_variant

        analysis = hit.chirp_analysis

        # Base confidence from the chirp analyzer (SNR-weighted, 0..1)
        base_confidence = float(
            min(1.0, max(0.0, analysis.chirp_confidence))
        )

        # SF + variant from the dechirp classifier
        sf_estimate = analysis.estimated_sf
        variant_label = (
            label_variant(sf=sf_estimate, bandwidth_hz=hit.bandwidth_hz)
            if sf_estimate is not None else None
        )

        # Confidence multiplier from SF discrimination strength.
        # sf_confidence is a ratio: 1.0 = totally indistinguishable
        # from second-best SF, large = unambiguous winner. Cap at
        # 4.0 to prevent unbounded amplification, then map [1.2, 4.0]
        # → [0.7, 1.0]. Below 1.2 we'd have already returned None
        # for SF, so this branch only runs for accepted SF values.
        if sf_estimate is not None:
            sf_conf_clamped = max(1.2, min(4.0, analysis.sf_confidence))
            sf_multiplier = 0.7 + (sf_conf_clamped - 1.2) / 2.8 * 0.3
        else:
            # No SF determined → don't claim Meshtastic/LoRaWAN, just
            # generic LoRa, and downweight confidence.
            sf_multiplier = 0.6

        confidence = float(
            min(1.0, max(0.0, base_confidence * sf_multiplier))
        )

        # Choose technology label by what the variant tells us:
        # Meshtastic > LoRaWAN > generic LoRa. This mirrors the legacy
        # detector's choice so reports group sources of the same
        # technology together.
        if variant_label and variant_label.startswith("meshtastic"):
            technology = "meshtastic"
        elif variant_label and variant_label.startswith("lorawan"):
            technology = "lorawan"
        else:
            technology = "lora"

        evidence_parts = [
            f"chirp pattern at {hit.freq_hz / 1e6:.3f} MHz",
            f"BW={hit.bandwidth_hz // 1000} kHz",
            f"SNR {hit.snr_db:.1f} dB",
        ]
        if sf_estimate is not None:
            evidence_parts.append(
                f"SF{sf_estimate} (dechirp peak {analysis.sf_peak_concentration:.3f}, "
                f"conf {analysis.sf_confidence:.2f})"
            )
        else:
            evidence_parts.append("SF=indeterminate")
        if variant_label is not None:
            evidence_parts.append(f"variant={variant_label}")
        evidence = "; ".join(evidence_parts)

        try:
            await self.event_bus.publish(DetectionEvent(
                session_id=self.session_id,
                detector_name="lora_survey",
                technology=technology,
                freq_hz=hit.freq_hz,
                bandwidth_hz=hit.bandwidth_hz,
                confidence=confidence,
                evidence=evidence,
                hand_off_tools=[
                    "gr-lora_sdr", "chirpstack", "lorapacketforwarder",
                ],
                metadata={
                    "source": "lora_survey_task",
                    "band_id": self.band.id,
                    "snr_db": round(hit.snr_db, 1),
                    "estimated_sf": sf_estimate,
                    "variant": variant_label,
                    # v0.6.8: dechirp-derived classification metadata.
                    # Useful for retrospective analysis (was the SF
                    # confidence high enough to trust this label?).
                    "sf_classification_method": "reference_dechirp",
                    "sf_confidence": round(analysis.sf_confidence, 3),
                    "sf_peak_concentration": round(
                        analysis.sf_peak_concentration, 4,
                    ),
                    "sf_scores": (
                        {str(k): round(v, 4)
                         for k, v in analysis.sf_scores.items()}
                        if analysis.sf_scores else None
                    ),
                },
            ))
            stats.detections_emitted += 1
            log.info(
                "lora_survey[%s]: detected %s at %.3f MHz "
                "(BW=%d kHz, SF=%s, variant=%s, conf=%.2f, "
                "SNR=%.1f dB, sf_conf=%.2f)",
                self.band.id, technology, hit.freq_hz / 1e6,
                hit.bandwidth_hz // 1000, sf_estimate, variant_label,
                confidence, hit.snr_db, analysis.sf_confidence,
            )
        except Exception:
            log.exception(
                "lora_survey[%s]: failed to publish detection",
                self.band.id,
            )

    # ────────────────────────────────────────────────────────────────
    # Internal: teardown
    # ────────────────────────────────────────────────────────────────

    async def _teardown(self) -> None:
        """Close the fanout connection, release the lease. Idempotent."""
        if self._writer is not None:
            with contextlib.suppress(Exception):
                self._writer.close()
                await self._writer.wait_closed()
            self._writer = None
            self._reader = None
        if self._lease is not None:
            with contextlib.suppress(Exception):
                await self.broker.release(self._lease)
            self._lease = None
