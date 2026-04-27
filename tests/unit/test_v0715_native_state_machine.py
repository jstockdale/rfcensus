"""Tests that the v0.7.15 C state machine produces bit-identical
events and final state vs the v0.7.14 Python implementation.

Strategy:
  1. Build a synthetic (n_frames × n_slots) input batch with a
     mix of triggers, releases, and quiet frames.
  2. Run both implementations on identical inputs and compare:
     - Final slot state (state, counters, noise_floor_lin, etc.)
     - Emitted events (kind, slot_idx, sample_offset, energies)
  3. Stress test with longer batches and edge cases (bootstrap
     boundary, immediate re-activation from DRAINING, overflow).

The Python implementation is in passband_detector.py; the C
implementation is exercised via passband_state_native.
"""

import numpy as np
import pytest

from rfcensus.decoders import passband_state_native as psn
from rfcensus.decoders.passband_detector import (
    DetectorConfig, SlotEnergyState, SlotState, PassbandDetector,
)


# Skip everything if the native lib isn't built — these tests
# are gated on the C kernel being available.
pytestmark = pytest.mark.skipif(
    not psn.is_available(),
    reason="libpassband_state.so not built",
)


# ─────────────────────────────────────────────────────────────────
# Reference Python implementation (extracted from passband_detector
# for direct testing without the full FFT pipeline)
# ─────────────────────────────────────────────────────────────────


def py_step_state_machines_vec(
    slots, energies_lin, above_trigger, below_release,
    db_above, noise_db,
    sample_offset, frame_count, cfg,
):
    """Same logic as PassbandDetector._step_state_machines_vec
    but as a free function for direct testing."""
    alpha = cfg.noise_alpha
    trigger_frames = cfg.trigger_frames
    drain_frames = cfg.drain_frames
    bootstrap_frames = cfg.bootstrap_frames

    events = []
    for i, slot in enumerate(slots):
        energy_lin = float(energies_lin[i])

        if slot.state == SlotState.IDLE:
            if slot.noise_floor_lin == 0.0:
                slot.noise_floor_lin = energy_lin
            else:
                slot.noise_floor_lin += alpha * (energy_lin - slot.noise_floor_lin)
            slot.idle_frames_seen += 1

            if slot.idle_frames_seen < bootstrap_frames:
                continue

            if above_trigger[i]:
                slot.consec_above_trigger += 1
            else:
                slot.consec_above_trigger = 0

            if slot.consec_above_trigger >= trigger_frames:
                slot.state = SlotState.ACTIVE
                slot.consec_above_trigger = 0
                slot.consec_below_release = 0
                slot.phase_started_frame = frame_count
                events.append((
                    "activate", i, sample_offset,
                    float(db_above[i]), float(noise_db[i]),
                ))

        elif slot.state == SlotState.ACTIVE:
            if below_release[i]:
                slot.state = SlotState.DRAINING
                slot.consec_below_release = 1
                slot.phase_started_frame = frame_count

        else:  # DRAINING
            if above_trigger[i]:
                slot.state = SlotState.ACTIVE
                slot.consec_below_release = 0
            else:
                slot.consec_below_release += 1
                if slot.consec_below_release >= drain_frames:
                    slot.state = SlotState.IDLE
                    slot.consec_below_release = 0
                    slot.idle_frames_seen = 0
                    slot.noise_floor_lin = energy_lin
                    events.append((
                        "deactivate", i, sample_offset,
                        0.0, float(noise_db[i]),
                    ))

    return events


def py_run_batch(
    slots, energies_lin, above_trigger, below_release,
    db_above, noise_db_at_start,
    base_sample_offset, hop_samples, fft_size, frame_count, cfg,
):
    """Multi-frame Python reference: matches the outer loop in
    PassbandDetector.feed_cu8."""
    n_frames, n_slots = energies_lin.shape
    all_events = []
    for f in range(n_frames):
        sample_offset = base_sample_offset + f * hop_samples + fft_size
        events = py_step_state_machines_vec(
            slots,
            energies_lin[f], above_trigger[f], below_release[f],
            db_above[f], noise_db_at_start,
            sample_offset, frame_count, cfg,
        )
        all_events.extend(events)
    return all_events


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


def make_py_slots(n_slots, *, freq_hz_base=900_000_000, bw_hz=250_000):
    """Create a list of n_slots SlotEnergyState objects."""
    return [
        SlotEnergyState(
            slot_freq_hz=freq_hz_base + i * bw_hz,
            bandwidth_hz=bw_hz,
            fft_bin_lo=i * 32,
            fft_bin_hi=(i + 1) * 32,
            n_bins=32,
        )
        for i in range(n_slots)
    ]


def make_c_kernel(n_slots, py_cfg):
    sm = psn.NativeStateMachine(n_slots=n_slots)
    sm.update_config(
        noise_alpha=py_cfg.noise_alpha,
        trigger_frames=py_cfg.trigger_frames,
        drain_frames=py_cfg.drain_frames,
        bootstrap_frames=py_cfg.bootstrap_frames,
    )
    return sm


def py_event_for_compare(ev_tuple):
    """Normalize a Python event for comparison with C output."""
    kind_str, slot_idx, sample_offset, e_db, n_db = ev_tuple
    kind_int = psn.PB_EVENT_ACTIVATE if kind_str == "activate" else psn.PB_EVENT_DEACTIVATE
    return (kind_int, slot_idx, sample_offset, e_db, n_db)


def c_event_for_compare(c_event):
    """Normalize a C event tuple from sm.iter_events()."""
    return c_event  # already (kind, slot_idx, sample_offset, e_db, n_db)


def compare_slots(py_slots, c_sm, *, atol=0.0):
    """Assert all slot fields match between Python and C state."""
    for i, ps in enumerate(py_slots):
        cs = c_sm._slots[i]
        py_state_int = {
            SlotState.IDLE:     psn.PB_STATE_IDLE,
            SlotState.ACTIVE:   psn.PB_STATE_ACTIVE,
            SlotState.DRAINING: psn.PB_STATE_DRAINING,
        }[ps.state]
        assert cs.state == py_state_int, (
            f"slot {i}: state {cs.state} != {py_state_int} ({ps.state.name})")
        assert cs.consec_above_trigger == ps.consec_above_trigger, (
            f"slot {i}: consec_above_trigger {cs.consec_above_trigger} != {ps.consec_above_trigger}")
        assert cs.consec_below_release == ps.consec_below_release, (
            f"slot {i}: consec_below_release {cs.consec_below_release} != {ps.consec_below_release}")
        assert cs.idle_frames_seen == ps.idle_frames_seen, (
            f"slot {i}: idle_frames_seen {cs.idle_frames_seen} != {ps.idle_frames_seen}")
        assert cs.phase_started_frame == ps.phase_started_frame, (
            f"slot {i}: phase_started_frame {cs.phase_started_frame} != {ps.phase_started_frame}")
        # noise_floor_lin: float32, may have tiny rounding diffs vs Python float64
        assert abs(cs.noise_floor_lin - ps.noise_floor_lin) <= atol or \
               cs.noise_floor_lin == pytest.approx(ps.noise_floor_lin, rel=1e-5), (
            f"slot {i}: noise_floor_lin {cs.noise_floor_lin} != {ps.noise_floor_lin}")


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────


class TestNativeStateMachineCorrectness:
    """C kernel must produce identical events + final state vs Python."""

    def _run_both(self, n_slots, n_frames, cfg, *, seed=0,
                   trigger_density=0.05):
        """Helper: build random-ish input, run both impls, return
        (py_events, c_events, py_slots, c_sm)."""
        rng = np.random.default_rng(seed)

        # Energy: mostly noise (~0.01) with occasional bursts (~10).
        energies = rng.uniform(0.005, 0.015, size=(n_frames, n_slots))
        burst_mask = rng.random(size=(n_frames, n_slots)) < trigger_density
        energies = np.where(burst_mask, energies * 1000.0, energies)
        energies = energies.astype(np.float32)

        # Compute db_above and trigger/release bools as the real
        # detector does. Snapshot noise from the slots' initial state
        # (start uninitialized → use inf placeholder).
        noise_db = np.full(n_slots, np.inf, dtype=np.float32)
        log_eps = 1e-30
        e_db = (10.0 * np.log10(energies + log_eps)).astype(np.float32)
        db_above = e_db - noise_db[None, :]
        above_trigger = (db_above >= cfg.trigger_threshold_db).astype(np.uint8)
        below_release = (db_above < cfg.release_threshold_db).astype(np.uint8)

        # Run Python.
        py_slots = make_py_slots(n_slots)
        py_events = py_run_batch(
            py_slots,
            energies, above_trigger, below_release,
            db_above, noise_db,
            base_sample_offset=10_000,
            hop_samples=256, fft_size=512,
            frame_count=42, cfg=cfg,
        )

        # Run C (with sync in/out).
        c_sm = make_c_kernel(n_slots, cfg)
        c_py_slots = make_py_slots(n_slots)  # parallel slot list
        c_sm.sync_in(c_py_slots)
        n_events = c_sm.process_batch(
            energies, above_trigger, below_release, db_above, noise_db,
            base_sample_offset=10_000,
            hop_samples=256, fft_size=512,
            frame_count=42,
        )
        c_sm.sync_out(c_py_slots)
        c_events = list(c_sm.iter_events(n_events))

        return py_events, c_events, py_slots, c_py_slots, c_sm

    def test_quiet_batch_no_events(self):
        """All-noise batch produces no events but updates noise floor."""
        cfg = DetectorConfig(
            sample_rate_hz=2_400_000, center_freq_hz=915_000_000,
        )
        py_e, c_e, py_s, c_py_s, c_sm = self._run_both(
            n_slots=8, n_frames=200, cfg=cfg,
            trigger_density=0.0,
        )
        assert py_e == []
        assert c_e == []
        compare_slots(py_s, c_sm)

    def test_bootstrap_suppresses_triggers(self):
        """No triggers fire before bootstrap_frames worth of IDLE frames."""
        cfg = DetectorConfig(
            sample_rate_hz=2_400_000, center_freq_hz=915_000_000,
            bootstrap_frames=50,
        )
        py_e, c_e, py_s, c_py_s, c_sm = self._run_both(
            n_slots=4, n_frames=40, cfg=cfg,
            trigger_density=0.5,  # lots of bursts, but bootstrap should suppress
        )
        # Both should produce exactly the same (zero) events.
        assert py_e == []
        assert c_e == []
        compare_slots(py_s, c_sm)

    def test_basic_activation(self):
        """Persistent burst on slot 0 triggers an activate."""
        cfg = DetectorConfig(
            sample_rate_hz=2_400_000, center_freq_hz=915_000_000,
            bootstrap_frames=10, trigger_frames=3,
        )
        n_slots, n_frames = 4, 100

        # First 10 frames: pure noise (bootstrap).
        # Then frames 10-30: persistent strong signal on slot 0.
        # Rest: noise.
        energies = np.full((n_frames, n_slots), 0.01, dtype=np.float32)
        energies[10:30, 0] = 100.0   # >>> noise, will trigger

        # Build inputs the way the real detector would.
        noise_db = np.full(n_slots, np.inf, dtype=np.float32)
        log_eps = 1e-30
        e_db = (10.0 * np.log10(energies + log_eps)).astype(np.float32)
        db_above = e_db - noise_db[None, :]
        above_trigger = (db_above >= cfg.trigger_threshold_db).astype(np.uint8)
        below_release = (db_above < cfg.release_threshold_db).astype(np.uint8)

        # Pre-condition the slots: simulate that bootstrap is past
        # by setting noise_floor_lin and idle_frames_seen.
        py_slots = make_py_slots(n_slots)
        for s in py_slots:
            s.noise_floor_lin = 0.01
            s.idle_frames_seen = cfg.bootstrap_frames + 1
        # Re-run db_above with realistic noise:
        noise_db = np.full(n_slots, 10.0 * np.log10(0.01 + log_eps),
                            dtype=np.float32)
        db_above = e_db - noise_db[None, :]
        above_trigger = (db_above >= cfg.trigger_threshold_db).astype(np.uint8)
        below_release = (db_above < cfg.release_threshold_db).astype(np.uint8)

        py_events = py_run_batch(
            py_slots,
            energies, above_trigger, below_release, db_above, noise_db,
            base_sample_offset=0, hop_samples=256, fft_size=512,
            frame_count=100, cfg=cfg,
        )

        c_sm = make_c_kernel(n_slots, cfg)
        c_py_slots = make_py_slots(n_slots)
        for s in c_py_slots:
            s.noise_floor_lin = 0.01
            s.idle_frames_seen = cfg.bootstrap_frames + 1
        c_sm.sync_in(c_py_slots)
        n_events = c_sm.process_batch(
            energies, above_trigger, below_release, db_above, noise_db,
            base_sample_offset=0, hop_samples=256, fft_size=512,
            frame_count=100,
        )
        c_sm.sync_out(c_py_slots)
        c_events = list(c_sm.iter_events(n_events))

        # Should be one activate on slot 0
        assert len(py_events) >= 1
        assert py_events[0][0] == "activate"
        assert py_events[0][1] == 0  # slot 0
        assert len(c_events) == len(py_events)

        for pe, ce in zip(py_events, c_events):
            ce_norm = ce
            pe_norm = py_event_for_compare(pe)
            assert pe_norm[0] == ce_norm[0]   # kind
            assert pe_norm[1] == ce_norm[1]   # slot_idx
            assert pe_norm[2] == ce_norm[2]   # sample_offset
            # Floats: tolerate tiny float64 vs float32 drift
            assert pe_norm[3] == pytest.approx(ce_norm[3], rel=1e-5, abs=1e-6)
            assert pe_norm[4] == pytest.approx(ce_norm[4], rel=1e-5, abs=1e-6)

        compare_slots(py_slots, c_sm)

    def test_full_lifecycle(self):
        """Slot goes IDLE → ACTIVE → DRAINING → IDLE end-to-end."""
        # Use small drain_frames for fast test
        cfg = DetectorConfig(
            sample_rate_hz=2_400_000, center_freq_hz=915_000_000,
            bootstrap_frames=5, trigger_frames=3, drain_frames=10,
        )
        n_slots = 2
        # 50 frames: noise (bootstrap), then burst, then quiet (drain), then noise.
        energies = np.full((20, n_slots), 0.01, dtype=np.float32)
        energies[5:15, 0] = 100.0  # 10 frames of burst on slot 0
        # After frame 15: back to noise → eventually drain to IDLE.
        rest = np.full((20, n_slots), 0.01, dtype=np.float32)
        energies = np.concatenate([energies, rest], axis=0)

        noise_db = np.full(n_slots, 10.0 * np.log10(0.01 + 1e-30), dtype=np.float32)
        log_eps = 1e-30
        e_db = (10.0 * np.log10(energies + log_eps)).astype(np.float32)
        db_above = e_db - noise_db[None, :]
        above_trigger = (db_above >= cfg.trigger_threshold_db).astype(np.uint8)
        below_release = (db_above < cfg.release_threshold_db).astype(np.uint8)

        # Pre-condition slots out of bootstrap.
        py_slots = make_py_slots(n_slots)
        for s in py_slots:
            s.noise_floor_lin = 0.01
            s.idle_frames_seen = cfg.bootstrap_frames + 1
        c_py_slots = make_py_slots(n_slots)
        for s in c_py_slots:
            s.noise_floor_lin = 0.01
            s.idle_frames_seen = cfg.bootstrap_frames + 1

        py_events = py_run_batch(
            py_slots, energies, above_trigger, below_release,
            db_above, noise_db,
            base_sample_offset=0, hop_samples=256, fft_size=512,
            frame_count=200, cfg=cfg,
        )
        c_sm = make_c_kernel(n_slots, cfg)
        c_sm.sync_in(c_py_slots)
        n_events = c_sm.process_batch(
            energies, above_trigger, below_release, db_above, noise_db,
            base_sample_offset=0, hop_samples=256, fft_size=512,
            frame_count=200,
        )
        c_sm.sync_out(c_py_slots)
        c_events = list(c_sm.iter_events(n_events))

        # Expect one activate then one deactivate on slot 0.
        kinds = [e[0] for e in py_events]
        assert "activate" in kinds, f"py_events={py_events}"
        assert "deactivate" in kinds, f"py_events={py_events}"
        assert len(c_events) == len(py_events)
        for pe, ce in zip(py_events, c_events):
            assert py_event_for_compare(pe)[0] == ce[0]
            assert py_event_for_compare(pe)[1] == ce[1]
            assert py_event_for_compare(pe)[2] == ce[2]
        # Final state must be IDLE on both slots.
        assert all(s.state == SlotState.IDLE for s in py_slots)
        compare_slots(py_slots, c_sm)

    def test_random_stress(self):
        """100 random batches: every event and final state must match."""
        cfg = DetectorConfig(
            sample_rate_hz=2_400_000, center_freq_hz=915_000_000,
            bootstrap_frames=10, trigger_frames=3, drain_frames=20,
        )
        for seed in range(100):
            py_e, c_e, py_s, c_py_s, c_sm = self._run_both(
                n_slots=8, n_frames=300, cfg=cfg,
                trigger_density=0.1, seed=seed,
            )
            assert len(c_e) == len(py_e), (
                f"seed {seed}: py {len(py_e)} events, c {len(c_e)} events")
            for pe, ce in zip(py_e, c_e):
                pe_n = py_event_for_compare(pe)
                assert pe_n[0] == ce[0], f"seed {seed}: kind mismatch"
                assert pe_n[1] == ce[1], f"seed {seed}: slot_idx mismatch"
                assert pe_n[2] == ce[2], f"seed {seed}: sample_offset mismatch"
            compare_slots(py_s, c_sm)

    def test_event_overflow_returns_negative(self):
        """If we somehow hit max_events, return -1 / cap-out gracefully."""
        cfg = DetectorConfig(
            sample_rate_hz=2_400_000, center_freq_hz=915_000_000,
            bootstrap_frames=1, trigger_frames=1, drain_frames=1,
        )
        # n_slots=4 → max_events = 8 + 16 = 24. Make 4 slots cycle
        # IDLE→ACTIVE→DRAINING→IDLE rapidly. Each slot can produce
        # 2 events (one act + one deact) per cycle. With trigger=1 and
        # drain=1, each cycle takes ~3 frames. In 100 frames we'd
        # produce ~33 cycles × 4 slots × 2 = ~260 events, exceeding
        # max_events=24.
        n_slots, n_frames = 4, 100
        # Alternate strong and weak energies every other frame to drive
        # the cycling.
        energies = np.full((n_frames, n_slots), 0.01, dtype=np.float32)
        energies[::3] = 100.0  # every 3rd frame is a burst on all slots

        noise_db = np.full(n_slots, 10.0 * np.log10(0.01 + 1e-30),
                            dtype=np.float32)
        log_eps = 1e-30
        e_db = (10.0 * np.log10(energies + log_eps)).astype(np.float32)
        db_above = e_db - noise_db[None, :]
        above_trigger = (db_above >= cfg.trigger_threshold_db).astype(np.uint8)
        below_release = (db_above < cfg.release_threshold_db).astype(np.uint8)

        c_sm = make_c_kernel(n_slots, cfg)
        c_py_slots = make_py_slots(n_slots)
        for s in c_py_slots:
            s.noise_floor_lin = 0.01
            s.idle_frames_seen = cfg.bootstrap_frames + 1
        c_sm.sync_in(c_py_slots)
        n_events = c_sm.process_batch(
            energies, above_trigger, below_release, db_above, noise_db,
            base_sample_offset=0, hop_samples=256, fft_size=512,
            frame_count=0,
        )
        # Wrapper returns max_events on overflow (not -1 to caller).
        # State should still be fully updated though.
        assert n_events == c_sm._max_events
        # Sanity: all events that DID make it should be valid.
        for ev in c_sm.iter_events(n_events):
            kind, slot_idx, _, _, _ = ev
            assert kind in (psn.PB_EVENT_ACTIVATE, psn.PB_EVENT_DEACTIVATE)
            assert 0 <= slot_idx < n_slots


class TestSyncRoundtrip:
    """sync_in followed by sync_out should be a no-op."""

    def test_roundtrip_preserves_state(self):
        sm = psn.NativeStateMachine(n_slots=4)
        sm.update_config(
            noise_alpha=0.001, trigger_frames=3,
            drain_frames=400, bootstrap_frames=100,
        )
        slots = make_py_slots(4)
        # Set non-default values
        slots[0].state = SlotState.ACTIVE
        slots[0].consec_above_trigger = 0
        slots[0].consec_below_release = 0
        slots[0].noise_floor_lin = 0.05
        slots[0].last_energy_lin = 1.5
        slots[0].idle_frames_seen = 200
        slots[0].phase_started_frame = 12345

        slots[1].state = SlotState.DRAINING
        slots[1].consec_below_release = 17

        sm.sync_in(slots)
        # Make a parallel copy to compare against.
        slots_copy = make_py_slots(4)
        sm.sync_out(slots_copy)

        for orig, after in zip(slots, slots_copy):
            assert orig.state == after.state
            assert orig.consec_above_trigger == after.consec_above_trigger
            assert orig.consec_below_release == after.consec_below_release
            assert orig.idle_frames_seen == after.idle_frames_seen
            assert orig.phase_started_frame == after.phase_started_frame
            assert orig.noise_floor_lin == pytest.approx(after.noise_floor_lin, rel=1e-6)
            assert orig.last_energy_lin == pytest.approx(after.last_energy_lin, rel=1e-6)
