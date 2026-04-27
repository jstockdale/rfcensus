"""ctypes wrapper for the native passband state machine.

v0.7.15: ports the per-slot state machine inner loop from
``passband_detector.py:_step_state_machines_vec`` to C. The C
kernel processes a full (n_frames × n_slots) batch in one call,
eliminating ~12% of total CPU spent in Python interpreter
overhead on the per-slot loop (per the v0.7.14 profile).

The library is compiled as a separate small .so (~15 KB)
alongside the LoRa lib. Self-contained with no LoRa or numpy
dependencies on the C side.

Falls back gracefully if the library isn't present or fails to
load — the caller in PassbandDetector checks ``is_available()``
and reverts to the pure-Python implementation. This means the
v0.7.15 perf release works on systems where the user hasn't
rebuilt the natives yet (they get v0.7.14-equivalent performance
plus a one-line warning in stderr).
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────
# Library loading (lazy, with fallback)
# ─────────────────────────────────────────────────────────────────

_lib: ctypes.CDLL | None = None
_lib_load_attempted = False
_lib_load_error: str | None = None


def _find_library() -> Path | None:
    """Locate libpassband_state.so. Returns None if not found."""
    here = Path(__file__).parent / "_native" / "passband"
    candidates = [
        here / "libpassband_state.so",
        here / "libpassband_state.dylib",  # macOS dev builds
    ]
    env = os.environ.get("RFCENSUS_LIBPASSBAND_STATE")
    if env:
        candidates.insert(0, Path(env))
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_library() -> ctypes.CDLL | None:
    """Load the shared lib once. Returns None on failure (caller falls back)."""
    global _lib, _lib_load_attempted, _lib_load_error
    if _lib_load_attempted:
        return _lib
    _lib_load_attempted = True

    lib_path = _find_library()
    if lib_path is None:
        _lib_load_error = (
            "libpassband_state.so not found. Build it with:\n"
            "  cd rfcensus/decoders/_native/passband && make\n"
            "or set RFCENSUS_LIBPASSBAND_STATE=/path/to/libpassband_state.so\n"
            "Falling back to pure-Python state machine (~12% slower)."
        )
        return None

    try:
        lib = ctypes.CDLL(str(lib_path))
    except OSError as e:
        _lib_load_error = f"failed to load {lib_path}: {e}"
        return None

    # int pb_process_batch(
    #     pb_slot_t *slots, size_t n_slots,
    #     const float *energies_lin,
    #     const uint8_t *above_trigger,
    #     const uint8_t *below_release,
    #     const float *db_above,
    #     const float *noise_db_at_start,
    #     size_t n_frames,
    #     int64_t base_sample_offset,
    #     int64_t hop_samples,
    #     int64_t fft_size,
    #     int64_t frame_count,
    #     const pb_config_t *cfg,
    #     pb_event_t *events_out, size_t max_events
    # );
    #
    # Arrays passed as c_void_p (= raw int address) for fast dispatch.
    # See lora_native.py for the rationale on c_void_p vs POINTER(...).
    lib.pb_process_batch.restype = ctypes.c_int
    lib.pb_process_batch.argtypes = [
        ctypes.c_void_p,    # slots
        ctypes.c_size_t,    # n_slots
        ctypes.c_void_p,    # energies_lin
        ctypes.c_void_p,    # above_trigger
        ctypes.c_void_p,    # below_release
        ctypes.c_void_p,    # db_above
        ctypes.c_void_p,    # noise_db_at_start
        ctypes.c_size_t,    # n_frames
        ctypes.c_int64,     # base_sample_offset
        ctypes.c_int64,     # hop_samples
        ctypes.c_int64,     # fft_size
        ctypes.c_int64,     # frame_count
        ctypes.c_void_p,    # cfg
        ctypes.c_void_p,    # events_out
        ctypes.c_size_t,    # max_events
    ]

    _lib = lib
    return _lib


# ─────────────────────────────────────────────────────────────────
# ctypes structure mirrors of the C structs
# ─────────────────────────────────────────────────────────────────

# State enum — must match C pb_slot_state_t and Python SlotState
PB_STATE_IDLE     = 0
PB_STATE_ACTIVE   = 1
PB_STATE_DRAINING = 2

PB_EVENT_ACTIVATE   = 0
PB_EVENT_DEACTIVATE = 1


class _PbSlot(ctypes.Structure):
    """Mirrors `pb_slot_t` in passband_state.h. ABI-stable layout."""
    _fields_ = [
        ("state",                ctypes.c_int32),
        ("consec_above_trigger", ctypes.c_int32),
        ("consec_below_release", ctypes.c_int32),
        ("idle_frames_seen",     ctypes.c_int32),
        ("phase_started_frame",  ctypes.c_int64),
        ("noise_floor_lin",      ctypes.c_float),
        ("last_energy_lin",      ctypes.c_float),
    ]


class _PbConfig(ctypes.Structure):
    """Mirrors `pb_config_t` in passband_state.h."""
    _fields_ = [
        ("noise_alpha",      ctypes.c_float),
        ("trigger_frames",   ctypes.c_int32),
        ("drain_frames",     ctypes.c_int32),
        ("bootstrap_frames", ctypes.c_int32),
    ]


class _PbEvent(ctypes.Structure):
    """Mirrors `pb_event_t` in passband_state.h."""
    _fields_ = [
        ("kind",                  ctypes.c_int32),
        ("slot_idx",              ctypes.c_int32),
        ("sample_offset",         ctypes.c_int64),
        ("energy_db_above_floor", ctypes.c_float),
        ("noise_floor_db",        ctypes.c_float),
    ]


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────


def is_available() -> bool:
    """True iff the native library loaded successfully."""
    return _load_library() is not None


def load_error() -> str | None:
    """Human-readable explanation of why the library failed to load,
    or None if it loaded fine (or hasn't been tried yet)."""
    _load_library()
    return _lib_load_error


class NativeStateMachine:
    """Holds the per-slot ctypes state and event buffer for a detector.

    Lifetime is tied to the PassbandDetector instance: one
    NativeStateMachine per detector. The slot-state ctypes array is
    the authoritative copy of the state machine's mutable state;
    callers mutate Python `SlotEnergyState` objects only at sync
    boundaries (start/end of batch).
    """

    def __init__(self, n_slots: int) -> None:
        if not is_available():
            raise RuntimeError(
                "libpassband_state.so not loaded; cannot create "
                "NativeStateMachine. Call is_available() first."
            )
        self._n_slots = n_slots
        # Allocate slot state and event buffer as ctypes arrays.
        # Slot state is small (84 × 32 bytes = 2.6 KB); event buffer
        # is sized for the worst case (each slot can emit at most
        # 2 events per batch — IDLE→ACTIVE then later DRAINING→IDLE).
        self._slots = (_PbSlot * n_slots)()
        self._max_events = 2 * n_slots + 16   # +16 paranoia headroom
        self._events = (_PbEvent * self._max_events)()
        # Pre-cache addresses (don't recompute every call).
        self._slots_addr  = ctypes.addressof(self._slots)
        self._events_addr = ctypes.addressof(self._events)
        self._cfg = _PbConfig()
        self._cfg_addr = ctypes.addressof(self._cfg)
        # Initialize all slots to IDLE with zero state. The Python
        # SlotEnergyState defaults match.
        for s in self._slots:
            s.state                = PB_STATE_IDLE
            s.consec_above_trigger = 0
            s.consec_below_release = 0
            s.idle_frames_seen     = 0
            s.phase_started_frame  = 0
            s.noise_floor_lin      = 0.0
            s.last_energy_lin      = 0.0

    def update_config(
        self, noise_alpha: float, trigger_frames: int,
        drain_frames: int, bootstrap_frames: int,
    ) -> None:
        self._cfg.noise_alpha      = float(noise_alpha)
        self._cfg.trigger_frames   = int(trigger_frames)
        self._cfg.drain_frames     = int(drain_frames)
        self._cfg.bootstrap_frames = int(bootstrap_frames)

    def sync_in(self, py_slots: list) -> None:
        """Copy mutable state from Python SlotEnergyState objects
        into the ctypes slot array."""
        # SlotState enum → int. The Python enum auto() values are
        # 1, 2, 3 (IDLE, ACTIVE, DRAINING); the C enum is 0, 1, 2.
        # Use the explicit name lookup to be robust.
        from rfcensus.decoders.passband_detector import SlotState
        state_to_int = {
            SlotState.IDLE:     PB_STATE_IDLE,
            SlotState.ACTIVE:   PB_STATE_ACTIVE,
            SlotState.DRAINING: PB_STATE_DRAINING,
        }
        for i, ps in enumerate(py_slots):
            cs = self._slots[i]
            cs.state                = state_to_int[ps.state]
            cs.consec_above_trigger = ps.consec_above_trigger
            cs.consec_below_release = ps.consec_below_release
            cs.idle_frames_seen     = ps.idle_frames_seen
            cs.phase_started_frame  = ps.phase_started_frame
            cs.noise_floor_lin      = ps.noise_floor_lin
            cs.last_energy_lin      = ps.last_energy_lin

    def sync_out(self, py_slots: list) -> None:
        """Copy mutable state from ctypes back into Python objects."""
        from rfcensus.decoders.passband_detector import SlotState
        int_to_state = {
            PB_STATE_IDLE:     SlotState.IDLE,
            PB_STATE_ACTIVE:   SlotState.ACTIVE,
            PB_STATE_DRAINING: SlotState.DRAINING,
        }
        for i, ps in enumerate(py_slots):
            cs = self._slots[i]
            ps.state                = int_to_state[cs.state]
            ps.consec_above_trigger = cs.consec_above_trigger
            ps.consec_below_release = cs.consec_below_release
            ps.idle_frames_seen     = cs.idle_frames_seen
            ps.phase_started_frame  = cs.phase_started_frame
            ps.noise_floor_lin      = cs.noise_floor_lin
            ps.last_energy_lin      = cs.last_energy_lin

    def process_batch(
        self,
        energies_lin: np.ndarray,        # (n_frames, n_slots) float32
        above_trigger: np.ndarray,       # (n_frames, n_slots) uint8
        below_release: np.ndarray,       # (n_frames, n_slots) uint8
        db_above: np.ndarray,            # (n_frames, n_slots) float32
        noise_db_at_start: np.ndarray,   # (n_slots,) float32
        base_sample_offset: int,
        hop_samples: int,
        fft_size: int,
        frame_count: int,
    ) -> int:
        """Run the C kernel. Returns number of events written."""
        # Verify shapes/dtypes — these are easy to get wrong silently
        # and the C side trusts the caller.
        n_frames, n_slots = energies_lin.shape
        assert n_slots == self._n_slots, (
            f"slot count mismatch: configured {self._n_slots}, got {n_slots}")
        assert above_trigger.shape == (n_frames, n_slots)
        assert below_release.shape == (n_frames, n_slots)
        assert db_above.shape == (n_frames, n_slots)
        assert noise_db_at_start.shape == (n_slots,)
        assert energies_lin.dtype == np.float32
        assert above_trigger.dtype == np.uint8
        assert below_release.dtype == np.uint8
        assert db_above.dtype == np.float32
        assert noise_db_at_start.dtype == np.float32

        # All inputs must be C-contiguous.
        energies_lin     = np.ascontiguousarray(energies_lin)
        above_trigger    = np.ascontiguousarray(above_trigger)
        below_release    = np.ascontiguousarray(below_release)
        db_above         = np.ascontiguousarray(db_above)
        noise_db_at_start = np.ascontiguousarray(noise_db_at_start)

        n_events = _lib.pb_process_batch(
            self._slots_addr, n_slots,
            energies_lin.ctypes.data,
            above_trigger.ctypes.data,
            below_release.ctypes.data,
            db_above.ctypes.data,
            noise_db_at_start.ctypes.data,
            n_frames,
            int(base_sample_offset),
            int(hop_samples),
            int(fft_size),
            int(frame_count),
            self._cfg_addr,
            self._events_addr,
            self._max_events,
        )
        if n_events < 0:
            # Overflow — should not happen with our 2*n_slots+16 sizing.
            # Return the truncated count (max_events) so the caller
            # at least gets the events that did fit.
            return self._max_events
        return n_events

    def iter_events(self, n_events: int):
        """Generator yielding (kind, slot_idx, sample_offset,
        energy_db_above_floor, noise_floor_db) tuples for the first
        n_events of the buffer."""
        for i in range(n_events):
            ev = self._events[i]
            yield (
                ev.kind,
                ev.slot_idx,
                ev.sample_offset,
                ev.energy_db_above_floor,
                ev.noise_floor_db,
            )
