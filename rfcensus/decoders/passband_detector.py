"""PassbandDetector — wide-FFT activity detector for Meshtastic slots.

Runs ONE wideband FFT every K samples, estimates per-slot energy by
summing magnitude-squared FFT bins covering each slot's frequency
range, and fires activate/deactivate events when slots cross
energy thresholds.

Architecture role: this lets us avoid spawning expensive per-slot LoRa
decoders eagerly. The eager-spawn approach (one `LoraDecoder` per
(preset, slot) pair) costs ~23 cores for US 2.4 MS/s `--slots all`
because high-SF presets do massive FFTs (78k-point for SF12/125kHz at
2.4 MS/s, even though they only run ~120 of them per second).

The detector itself is fixed-cost and SMALL — at 512-pt FFT with 256-
sample hop on 2.4 MS/s IQ, we run ~9400 FFTs/sec at ~10 µs each =
~10% of one Pi 5 core. Per-slot decoders only spawn when energy says
something is actually transmitting, which is rare (~1-3 active slots
out of 80+ candidates at any moment in normal mesh activity).

State machine per slot:
  IDLE     → ACTIVE   (energy clears trigger threshold for trigger_frames)
  ACTIVE   → DRAINING (energy drops below release threshold)
  DRAINING → IDLE     (still below release after drain_frames)
  DRAINING → ACTIVE   (energy comes back up before drain expires)

The IDLE→ACTIVE debounce avoids spurious activations from noise spikes
or partial preambles that don't lead to a real packet. The DRAINING
state keeps decoders alive between back-to-back transmissions on the
same slot (very common in mesh traffic).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterator, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Configuration + types
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for the passband detector.

    Defaults are tuned for the typical RTL-SDR @ 2.4 MS/s case. For
    other sample rates, scale ``hop_samples`` to maintain ~100 µs per
    FFT-window of time resolution.
    """
    sample_rate_hz: int
    center_freq_hz: int
    fft_size: int = 512
    hop_samples: int = 256
    # How many consecutive "above trigger" frames before a slot
    # transitions IDLE→ACTIVE. Higher = more debounce but slower
    # response. 2-3 frames at 100 µs/frame = 200-300 µs latency.
    trigger_frames: int = 3
    # How many consecutive "below release" frames in DRAINING before
    # we declare the slot truly idle. Should comfortably exceed the
    # gap between back-to-back symbols of any preset. The slowest
    # preset (SF12/BW125) has symbol time 32.8ms; pick drain_frames
    # such that drain_frames × hop_time > 35ms to keep the same slot
    # alive across a multi-symbol packet's brief inter-symbol nulls.
    # At hop=256 samples / 2.4 MS/s = 107 µs per frame, 350 frames =
    # 37.5ms. Set to 400 for a little headroom.
    drain_frames: int = 400
    # Trigger threshold above the per-slot noise floor, in dB.
    # 6 dB is a safe default — well above noise variance, well below
    # typical SNR of an actual LoRa preamble (which is +20 dB or
    # better). Lower = more false positives, more CPU spent on dead
    # decoders.
    trigger_threshold_db: float = 6.0
    # Release threshold (lower than trigger to provide hysteresis).
    release_threshold_db: float = 3.0
    # Noise floor estimator time constant. We use an EMA: noise_floor
    # = α × current + (1-α) × noise_floor, but ONLY when the slot is
    # IDLE (so a real signal doesn't poison the noise estimate).
    # α=0.001 = ~1000-frame time constant = ~100ms, fast enough to
    # track AGC drift, slow enough to not be fooled by gaps in traffic.
    noise_alpha: float = 0.001
    # Bootstrap: until we have this many idle frames, treat noise
    # floor as not-yet-trustworthy and skip triggering. Avoids
    # spurious triggers on the very first FFT before we know what
    # noise looks like.
    bootstrap_frames: int = 100


class SlotState(Enum):
    IDLE = auto()
    ACTIVE = auto()
    DRAINING = auto()


@dataclass
class SlotEnergyState:
    """Runtime energy/state tracking for one candidate slot."""
    slot_freq_hz: int
    bandwidth_hz: int
    fft_bin_lo: int       # inclusive index into FFT output
    fft_bin_hi: int       # exclusive
    n_bins: int           # = fft_bin_hi - fft_bin_lo
    state: SlotState = SlotState.IDLE
    consec_above_trigger: int = 0
    consec_below_release: int = 0
    noise_floor_lin: float = 0.0   # linear (not dB) energy per bin
    idle_frames_seen: int = 0
    # Frame counter at which current ACTIVE / DRAINING phase started
    # (in detector frames since start). Used to compute event timing.
    phase_started_frame: int = 0
    last_energy_lin: float = 0.0


@dataclass
class SlotEvent:
    """An activation or deactivation of a candidate slot."""
    kind: str                    # "activate" or "deactivate"
    slot_freq_hz: int
    bandwidth_hz: int
    sample_offset: int           # global sample offset of event
    energy_db_above_floor: float # for activate; 0 for deactivate
    noise_floor_db: float        # current estimate, for diagnostics


# ─────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────

class PassbandDetector:
    """Wide-FFT passband activity detector.

    Construction takes a list of candidate slots (typically from
    ``enumerate_all_slots_in_passband``) PLUS a detector config. The
    detector pre-computes which FFT bins each slot covers.

    Then ``feed_cu8(samples)`` ingests IQ chunks and yields
    ``SlotEvent`` instances as state transitions occur. The caller
    routes those events to the lazy decoder spawner.
    """

    def __init__(
        self,
        config: DetectorConfig,
        slot_freqs_hz: list[int] | None = None,
        slot_bandwidths_hz: list[int] | None = None,
    ) -> None:
        """Initialize the detector.

        Args:
          config: detector configuration.
          slot_freqs_hz: list of slot center frequencies to monitor (Hz).
          slot_bandwidths_hz: matching list of bandwidths for each slot.
            Same length as slot_freqs_hz. Each entry says how wide of
            an FFT-bin window to integrate for that slot's energy.

        Both lists must be the same length. Multiple "slots" at the
        same frequency but different bandwidths are valid (and common —
        a single RF frequency might be shared by BW=125 / BW=250 /
        BW=500 candidates with different slot-grid alignments).
        """
        self._cfg = config
        if slot_freqs_hz is None or slot_bandwidths_hz is None:
            raise ValueError(
                "PassbandDetector requires slot_freqs_hz and "
                "slot_bandwidths_hz lists"
            )
        if len(slot_freqs_hz) != len(slot_bandwidths_hz):
            raise ValueError(
                f"slot_freqs_hz and slot_bandwidths_hz must be same "
                f"length ({len(slot_freqs_hz)} vs "
                f"{len(slot_bandwidths_hz)})"
            )

        # FFT bin resolution
        self._bin_hz = config.sample_rate_hz / config.fft_size
        self._fft_size = config.fft_size
        self._hop = config.hop_samples

        # Map each (freq, bw) to an FFT-bin range. Center frequency is
        # at FFT bin 0 after fftshift, so a slot at freq F is at bin
        # `round((F - center) / bin_hz) + fft_size/2`. The slot's
        # signal occupies BW Hz around that bin.
        self._slots: list[SlotEnergyState] = []
        for freq, bw in zip(slot_freqs_hz, slot_bandwidths_hz):
            offset_hz = freq - config.center_freq_hz
            center_bin = int(round(offset_hz / self._bin_hz)) + config.fft_size // 2
            half_bins = max(1, int(round(bw / self._bin_hz / 2)))
            bin_lo = max(0, center_bin - half_bins)
            bin_hi = min(config.fft_size, center_bin + half_bins + 1)
            if bin_hi <= bin_lo:
                # Slot frequency is outside the FFT range entirely —
                # shouldn't happen if caller filtered to passband, but
                # be defensive.
                continue
            self._slots.append(SlotEnergyState(
                slot_freq_hz=freq,
                bandwidth_hz=bw,
                fft_bin_lo=bin_lo,
                fft_bin_hi=bin_hi,
                n_bins=bin_hi - bin_lo,
            ))

        # IQ sample accumulator (we need fft_size samples to do an FFT;
        # if a chunk is smaller, we accumulate.) Stored as cu8 bytes.
        self._cu8_buf = bytearray()
        # Total IQ samples consumed (= "global sample offset" of the
        # next sample we'll process).
        self._samples_consumed = 0
        # Frame counter (incremented per FFT computed)
        self._frame_count = 0

        # Window function — Hann reduces spectral leakage between
        # adjacent bins. Important here because Meshtastic slots are
        # only ~26 bins apart at our FFT resolution; without windowing,
        # a strong signal in slot N "smears" into slot N+1's bins.
        self._window = np.hanning(config.fft_size).astype(np.float32)
        # Pre-allocated arrays for the hot path.
        self._cf32_buf = np.zeros(config.fft_size, dtype=np.complex64)

        # Precompute per-slot bin index arrays for vectorized energy
        # estimation. We use a cumulative-sum trick: precompute
        # cumsum(mag_sq), then per-slot energy = (cumsum[hi] -
        # cumsum[lo]) / n_bins. Avoids 1M+ numpy.mean() calls per
        # 30 seconds of capture (which dominate detector CPU when many
        # slots are watched).
        self._slot_lo = np.array(
            [s.fft_bin_lo for s in self._slots], dtype=np.int64,
        )
        self._slot_hi = np.array(
            [s.fft_bin_hi for s in self._slots], dtype=np.int64,
        )
        self._slot_n_bins = np.array(
            [s.n_bins for s in self._slots], dtype=np.float32,
        )

        # v0.7.15: optional native state machine. Lazily initialized
        # on first feed_cu8() call so the import cost (and the C lib
        # load attempt) only happen if the detector is actually used.
        # Falls back to the pure-Python implementation if the .so is
        # not available — see _maybe_init_native_state for the warning.
        self._native_sm = None
        self._native_sm_init_attempted = False

    def _maybe_init_native_state(self) -> None:
        """Lazy-init the native state machine on first frame."""
        if self._native_sm_init_attempted:
            return
        self._native_sm_init_attempted = True
        try:
            from rfcensus.decoders import passband_state_native as psn
        except ImportError:
            return
        if not psn.is_available():
            # Native lib not built — silently fall back. The error is
            # logged once at first use rather than spammed every batch.
            import sys
            err = psn.load_error()
            if err:
                print(f"[passband_detector] native state machine "
                      f"unavailable, using Python fallback: {err}",
                      file=sys.stderr)
            return
        sm = psn.NativeStateMachine(n_slots=len(self._slots))
        sm.update_config(
            noise_alpha=self._cfg.noise_alpha,
            trigger_frames=self._cfg.trigger_frames,
            drain_frames=self._cfg.drain_frames,
            bootstrap_frames=self._cfg.bootstrap_frames,
        )
        # Sync initial slot state into C array.
        sm.sync_in(self._slots)
        self._native_sm = sm

    @property
    def n_slots(self) -> int:
        return len(self._slots)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def samples_consumed(self) -> int:
        return self._samples_consumed

    def slot_state(self, idx: int) -> SlotEnergyState:
        return self._slots[idx]

    def feed_cu8(self, samples: bytes) -> Iterator[SlotEvent]:
        """Ingest a chunk of cu8 IQ. Yield ``SlotEvent`` for any state
        transitions that occur during this chunk.

        Multi-chunk semantics: leftover samples (< fft_size) are kept
        across calls. The detector advances by ``hop_samples`` per
        FFT, so consecutive FFTs see overlapping windows.

        Implementation: we batch-process all the FFT frames in this
        chunk via 2D numpy operations (one ``np.fft.fft`` call over a
        (n_frames, fft_size) matrix), then walk the resulting
        per-slot energy time-series sequentially through the state
        machine. The batch FFT is ~5× faster than calling fft() per
        frame because numpy's per-call overhead amortizes. The state
        machine itself stays sequential because each slot's IDLE/ACTIVE
        transitions depend on the previous frame's state.
        """
        self._cu8_buf.extend(samples)
        bytes_per_fft = 2 * self._fft_size
        bytes_per_hop = 2 * self._hop

        if len(self._cu8_buf) < bytes_per_fft:
            return

        # How many frames can we extract from the current buffer?
        # Each frame consumes hop_samples (then advances), and the
        # final frame additionally needs (fft_size - hop_samples)
        # extra samples for its FFT window.
        usable_bytes = len(self._cu8_buf)
        n_frames = (
            (usable_bytes - bytes_per_fft) // bytes_per_hop + 1
        )
        if n_frames <= 0:
            return

        # Build (n_frames, fft_size) matrix of cf32 samples. Each row
        # is one FFT window; rows overlap by (fft_size - hop_samples)
        # samples relative to neighbors.
        cu8_view = bytes(self._cu8_buf[:bytes_per_fft + (n_frames-1)*bytes_per_hop])
        cu8_arr = np.frombuffer(cu8_view, dtype=np.uint8)
        # Interleaved IQ → complex float32 for the whole chunk
        i = (cu8_arr[0::2].astype(np.float32) - 127.5) / 127.5
        q = (cu8_arr[1::2].astype(np.float32) - 127.5) / 127.5
        full_cf32 = i + 1j * q
        full_cf32 = full_cf32.astype(np.complex64)

        # Build the (n_frames, fft_size) windowed matrix using
        # stride_tricks to avoid copying — each row is a view into
        # full_cf32 at offset k*hop, length fft_size.
        from numpy.lib.stride_tricks import as_strided
        stride_bytes = full_cf32.strides[0]   # element stride
        windows = as_strided(
            full_cf32,
            shape=(n_frames, self._fft_size),
            strides=(self._hop * stride_bytes, stride_bytes),
            writeable=False,
        )
        # Apply window (broadcasts) and FFT all rows at once
        windowed = windows * self._window     # (n_frames, fft_size)
        spectra = np.fft.fft(windowed, axis=1)
        spectra = np.fft.fftshift(spectra, axes=1)
        # v0.7.15: ``np.abs(spectra) ** 2`` is ~2.3× faster than the
        # explicit ``spectra.real * spectra.real + spectra.imag *
        # spectra.imag`` form. NumPy's |z|² kernel hits the SIMD
        # complex-magnitude path directly; the explicit form
        # allocates two intermediate float arrays and adds them.
        # Numerically identical to within ~3e-5 relative (well below
        # any threshold of interest, and the result gets log10'd
        # right after which compresses any small drift).
        mag_sq_all = np.abs(spectra) ** 2     # (n_frames, fft_size)

        # Per-frame, per-slot energy via cumsum (vectorized over frames):
        # cumsum_all[f, k] = sum(mag_sq_all[f, :k]); slot energy =
        # (cumsum[hi] - cumsum[lo]) / n_bins.
        cumsum_all = np.empty(
            (n_frames, self._fft_size + 1), dtype=mag_sq_all.dtype,
        )
        cumsum_all[:, 0] = 0.0
        np.cumsum(mag_sq_all, axis=1, out=cumsum_all[:, 1:])
        # slot_energies has shape (n_frames, n_slots)
        slot_energies = (
            cumsum_all[:, self._slot_hi] - cumsum_all[:, self._slot_lo]
        ) / self._slot_n_bins[None, :]

        # v0.7.8: vectorize the state-machine inner loop's math.
        # Previous design called np.log10 twice per slot per frame
        # (= ~1.6M scalar log10 calls/sec at 9400 fps × 84 slots),
        # which dominated detector CPU time because each scalar call
        # paid full numpy dispatch overhead. By batching the log10s
        # into one vectorized call across (n_frames × n_slots) we
        # cut per-call overhead to amortized zero.
        #
        # noise_floor_lin evolves frame-by-frame in IDLE state via
        # the EMA, but α=0.001 means noise drifts at most ~0.1 dB
        # within a single ~10ms chunk. We snapshot noise_db at
        # batch start and hold it constant across the batch's
        # threshold compares — the resulting trigger-timing error
        # is at most 1-2 frames (sub-millisecond), well below the
        # 3-frame trigger debounce. The linear noise_floor_lin
        # field IS still updated frame-by-frame so longer-term
        # drift tracks correctly across batches.
        log_eps = 1e-30
        # Batch log10 of all energies — single vectorized op replaces
        # n_frames × n_slots scalar np.log10 calls.
        energy_db_all = (10.0 * np.log10(
            slot_energies.astype(np.float32, copy=False) + log_eps
        )).astype(np.float32)
        # Snapshot per-slot noise_db at start of batch.
        noise_lin_at_start = np.array(
            [s.noise_floor_lin for s in self._slots],
            dtype=np.float32,
        )
        # v0.7.8: when noise_floor_lin is 0 (uninitialized — no
        # IDLE frames have passed yet), use +infinity as the
        # placeholder so db_above = energy_db - inf < 0 and the
        # threshold compare always returns False. The EMA below
        # initializes noise_floor_lin on the first IDLE frame, so
        # by the NEXT batch we have a real value. This means the
        # very first batch after detector creation can never
        # trigger — fine, because the bootstrap_frames check would
        # have suppressed it anyway, and a one-batch delay (~27ms
        # in production) is well below the 3-frame trigger debounce.
        # Previously this used -300 as a "tiny noise floor"
        # placeholder, which made db_above huge → false triggers
        # the moment bootstrap_frames cleared mid-batch.
        noise_db_at_start = np.where(
            noise_lin_at_start > 0,
            10.0 * np.log10(noise_lin_at_start + log_eps),
            np.float32("inf"),
        ).astype(np.float32)
        # Vectorized threshold matrices — bool[n_frames, n_slots].
        # The state-machine inner loop reads these by index instead
        # of recomputing per slot.
        db_above_all = energy_db_all - noise_db_at_start[None, :]
        above_trigger_all = (
            db_above_all >= self._cfg.trigger_threshold_db
        )
        below_release_all = (
            db_above_all < self._cfg.release_threshold_db
        )

        # Walk the state machine. v0.7.15: when the native kernel
        # is available, process the whole batch in one C call
        # (eliminates n_frames × n_slots Python iterations). The
        # Python fallback (per-frame call into _step_state_machines_vec)
        # is preserved for environments where the .so isn't built.
        slots_list = self._slots    # bind once for hot-loop speed
        self._maybe_init_native_state()
        if self._native_sm is not None:
            # C path: cast inputs to the right dtypes, sync state in,
            # run kernel, sync state out, yield events.
            # Inputs from the matrix builds above are already float32
            # (energies, db_above, noise_db_at_start) but the bool
            # arrays are numpy bool — convert to uint8 for C.
            energies_f32 = slot_energies.astype(np.float32, copy=False)
            db_above_f32 = db_above_all.astype(np.float32, copy=False)
            noise_db_f32 = noise_db_at_start.astype(np.float32, copy=False)
            above_u8 = above_trigger_all.view(np.uint8)
            below_u8 = below_release_all.view(np.uint8)

            self._native_sm.sync_in(slots_list)
            n_events = self._native_sm.process_batch(
                energies_f32, above_u8, below_u8,
                db_above_f32, noise_db_f32,
                base_sample_offset=self._samples_consumed,
                hop_samples=self._hop,
                fft_size=self._fft_size,
                frame_count=self._frame_count,
            )
            self._native_sm.sync_out(slots_list)

            # Yield events as SlotEvent objects. Using slot_idx to
            # look up the slot's freq/bw (which the C side doesn't
            # know about — we kept those Python-side as immutable
            # attributes).
            for kind_int, slot_idx, sample_offset, e_db, n_db in \
                    self._native_sm.iter_events(n_events):
                slot = slots_list[slot_idx]
                yield SlotEvent(
                    kind="activate" if kind_int == 0 else "deactivate",
                    slot_freq_hz=slot.slot_freq_hz,
                    bandwidth_hz=slot.bandwidth_hz,
                    sample_offset=sample_offset,
                    energy_db_above_floor=e_db,
                    noise_floor_db=n_db,
                )
        else:
            # Python fallback: original per-frame loop.
            for f in range(n_frames):
                sample_offset_at_frame_end = (
                    self._samples_consumed + f * self._hop + self._fft_size
                )
                yield from self._step_state_machines_vec(
                    sample_offset_at_frame_end,
                    slot_energies[f],     # (n_slots,) linear
                    above_trigger_all[f], # (n_slots,) bool
                    below_release_all[f], # (n_slots,) bool
                    db_above_all[f],      # (n_slots,) for event payload
                    noise_db_at_start,    # (n_slots,) for event payload
                )
        # Update last_energy_lin to the latest frame for snapshot
        # API readers (TUI gauges etc.) — they expect to see the
        # most-recent value.
        if n_frames > 0:
            last = slot_energies[-1]
            for i, slot in enumerate(slots_list):
                slot.last_energy_lin = float(last[i])

        # Advance buffer/counters by what we consumed
        consumed_bytes = n_frames * bytes_per_hop
        del self._cu8_buf[:consumed_bytes]
        self._samples_consumed += n_frames * self._hop
        self._frame_count += n_frames

    def _step_state_machines_vec(
        self,
        sample_offset: int,
        energies_lin: np.ndarray,    # (n_slots,) linear energies for this frame
        above_trigger: np.ndarray,    # (n_slots,) bool
        below_release: np.ndarray,    # (n_slots,) bool
        db_above: np.ndarray,         # (n_slots,) signed dB above floor
        noise_db: np.ndarray,         # (n_slots,) noise floor in dB at batch start
    ) -> Iterator[SlotEvent]:
        """v0.7.8: vectorized state-machine inner loop.

        Same per-slot state transitions as ``_step_state_machines``
        (kept for diagnostic comparison + future ABI fallback), but
        all the math (log10, threshold compare) was hoisted out of
        this function into batched numpy ops in the caller. Here we
        just read precomputed bool arrays and mutate per-slot state.

        Saves ~85% of detector CPU at default config — the per-slot
        np.log10 calls were paying full Python→C dispatch overhead
        per call, ~5 µs each. With 84 slots × 9400 fps × 2 logs/slot
        = 1.6M calls/sec = 8 seconds of CPU per second of audio just
        on the log dispatch overhead.
        """
        cfg = self._cfg
        slots = self._slots
        alpha = cfg.noise_alpha
        trigger_frames = cfg.trigger_frames
        drain_frames = cfg.drain_frames
        bootstrap_frames = cfg.bootstrap_frames
        frame_count = self._frame_count

        for i, slot in enumerate(slots):
            energy_lin = float(energies_lin[i])

            if slot.state == SlotState.IDLE:
                # Update noise floor while idle (EMA in linear space).
                # Note: noise_db used for THIS frame's threshold compare
                # was snapshotted at batch start (see caller). The
                # linear noise_floor_lin DOES update per-frame so the
                # next batch sees fresh values — drift across one
                # batch is negligible (α=0.001 × 100 frames = 10%).
                if slot.noise_floor_lin == 0.0:
                    slot.noise_floor_lin = energy_lin
                else:
                    slot.noise_floor_lin += (
                        alpha * (energy_lin - slot.noise_floor_lin)
                    )
                slot.idle_frames_seen += 1

                # Bootstrap: don't trigger until noise estimate is
                # trustworthy.
                if slot.idle_frames_seen < bootstrap_frames:
                    continue

                if above_trigger[i]:
                    slot.consec_above_trigger += 1
                else:
                    slot.consec_above_trigger = 0

                if slot.consec_above_trigger >= trigger_frames:
                    # Promote to ACTIVE
                    slot.state = SlotState.ACTIVE
                    slot.consec_above_trigger = 0
                    slot.consec_below_release = 0
                    slot.phase_started_frame = frame_count
                    yield SlotEvent(
                        kind="activate",
                        slot_freq_hz=slot.slot_freq_hz,
                        bandwidth_hz=slot.bandwidth_hz,
                        sample_offset=sample_offset,
                        energy_db_above_floor=float(db_above[i]),
                        noise_floor_db=float(noise_db[i]),
                    )

            elif slot.state == SlotState.ACTIVE:
                # Don't update noise floor while active (signal would
                # poison the estimate).
                if below_release[i]:
                    slot.state = SlotState.DRAINING
                    slot.consec_below_release = 1
                    slot.phase_started_frame = frame_count
                # else: stay active, keep decoders running

            else:  # DRAINING
                if above_trigger[i]:
                    # Energy came back — back to ACTIVE. No event;
                    # decoders are still alive.
                    slot.state = SlotState.ACTIVE
                    slot.consec_below_release = 0
                else:
                    slot.consec_below_release += 1
                    if slot.consec_below_release >= drain_frames:
                        # Truly idle now. Tear down decoders.
                        slot.state = SlotState.IDLE
                        slot.consec_below_release = 0
                        slot.idle_frames_seen = 0
                        # Reset noise tracking — pick up the current
                        # energy as the new baseline so we don't carry
                        # stale numbers from before the transmission.
                        slot.noise_floor_lin = energy_lin
                        yield SlotEvent(
                            kind="deactivate",
                            slot_freq_hz=slot.slot_freq_hz,
                            bandwidth_hz=slot.bandwidth_hz,
                            sample_offset=sample_offset,
                            energy_db_above_floor=0.0,
                            noise_floor_db=float(noise_db[i]),
                        )

    def _step_state_machines(
        self, sample_offset: int,
    ) -> Iterator[SlotEvent]:
        """Run per-slot state machine for one FFT frame."""
        for slot in self._slots:
            # Convert linear energy to dB. Add tiny epsilon to avoid
            # log(0). The "noise floor" comparison is also in dB-space.
            energy_lin = slot.last_energy_lin
            energy_db = 10.0 * np.log10(energy_lin + 1e-30)
            noise_db = 10.0 * np.log10(slot.noise_floor_lin + 1e-30)

            # Trigger / release thresholds applied to dB-above-floor.
            db_above = energy_db - noise_db

            if slot.state == SlotState.IDLE:
                # Update noise floor while idle (EMA in linear space).
                if slot.noise_floor_lin == 0.0:
                    slot.noise_floor_lin = energy_lin
                else:
                    slot.noise_floor_lin += (
                        self._cfg.noise_alpha
                        * (energy_lin - slot.noise_floor_lin)
                    )
                slot.idle_frames_seen += 1

                # Bootstrap: don't trigger until noise estimate is
                # trustworthy.
                if slot.idle_frames_seen < self._cfg.bootstrap_frames:
                    continue

                if db_above >= self._cfg.trigger_threshold_db:
                    slot.consec_above_trigger += 1
                else:
                    slot.consec_above_trigger = 0

                if (slot.consec_above_trigger
                        >= self._cfg.trigger_frames):
                    # Promote to ACTIVE
                    slot.state = SlotState.ACTIVE
                    slot.consec_above_trigger = 0
                    slot.consec_below_release = 0
                    slot.phase_started_frame = self._frame_count
                    yield SlotEvent(
                        kind="activate",
                        slot_freq_hz=slot.slot_freq_hz,
                        bandwidth_hz=slot.bandwidth_hz,
                        sample_offset=sample_offset,
                        energy_db_above_floor=db_above,
                        noise_floor_db=noise_db,
                    )

            elif slot.state == SlotState.ACTIVE:
                # Don't update noise floor while active (signal would
                # poison the estimate).
                if db_above < self._cfg.release_threshold_db:
                    slot.state = SlotState.DRAINING
                    slot.consec_below_release = 1
                    slot.phase_started_frame = self._frame_count
                # else: stay active, keep decoders running

            else:  # DRAINING
                if db_above >= self._cfg.trigger_threshold_db:
                    # Energy came back — back to ACTIVE. No event;
                    # decoders are still alive.
                    slot.state = SlotState.ACTIVE
                    slot.consec_below_release = 0
                else:
                    slot.consec_below_release += 1
                    if (slot.consec_below_release
                            >= self._cfg.drain_frames):
                        # Truly idle now. Tear down decoders.
                        slot.state = SlotState.IDLE
                        slot.consec_below_release = 0
                        slot.idle_frames_seen = 0
                        # Reset noise tracking — pick up the current
                        # energy as the new baseline so we don't carry
                        # stale numbers from before the transmission.
                        slot.noise_floor_lin = energy_lin
                        yield SlotEvent(
                            kind="deactivate",
                            slot_freq_hz=slot.slot_freq_hz,
                            bandwidth_hz=slot.bandwidth_hz,
                            sample_offset=sample_offset,
                            energy_db_above_floor=0.0,
                            noise_floor_db=noise_db,
                        )

    def snapshot_energies_db(self) -> list[tuple[int, int, float, float]]:
        """For diagnostics: return list of
        (slot_freq_hz, bandwidth_hz, energy_db, noise_floor_db)
        for every monitored slot at the current moment.
        """
        out = []
        for slot in self._slots:
            energy_db = 10.0 * np.log10(slot.last_energy_lin + 1e-30)
            noise_db = 10.0 * np.log10(slot.noise_floor_lin + 1e-30)
            out.append((slot.slot_freq_hz, slot.bandwidth_hz,
                        energy_db, noise_db))
        return out
