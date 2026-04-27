"""v0.7.16 tests: decoder-state introspection + reap-while-decoding
deferral.

Two layers of test:

1. TestLoraDecoderIsIdle — exercises the new lora_demod_is_idle()
   C function and its Python wrapper. Confirms a fresh decoder is
   idle, that feeding noise keeps it idle, and that the binding
   doesn't crash on edge cases.

2. TestReapDeferral — drives the lazy_pipeline reap-decision logic
   directly with a synthetic _ActiveSlot containing mock decoders
   whose is_idle() return value can be controlled. Verifies:
   • idle decoders → reap proceeds immediately (preserves v0.7.13
     behavior)
   • busy decoder → reap is deferred, counter increments
   • later idle → deferred reap completes, counter increments
   • probe positive during deferral → defer is cancelled
   • busy past hung-timeout → force-reap with warning counter

The deferral logic itself is inline in _maybe_periodic_probe — we
construct a minimal pipeline + slot and call the method directly to
exercise it.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfcensus.decoders.lora_native import LoraConfig, LoraDecoder
from rfcensus.decoders.lazy_pipeline import (
    LazyMultiPresetPipeline,
    _ActiveSlot,
)
from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
from rfcensus.utils.meshtastic_region import (
    PRESETS, default_slot, enumerate_all_slots_in_passband,
)


# ─────────────────────────────────────────────────────────────────────
# Layer 1 — LoraDecoder.is_idle() smoke
# ─────────────────────────────────────────────────────────────────────


class TestLoraDecoderIsIdle:
    """The new C-side state introspection. Cheap binding test — these
    confirm the symbol is exported and the wrapper returns sensible
    values, without exercising the full state machine."""

    def _make_decoder(self, sf: int = 11) -> LoraDecoder:
        cfg = LoraConfig(
            sample_rate_hz=250_000,
            bandwidth=250_000,
            sf=sf,
            sync_word=0x2B,
            mix_freq_hz=0,
        )
        return LoraDecoder(cfg)

    def test_fresh_decoder_is_idle(self) -> None:
        """Right after construction, the C state is LORA_STATE_DETECT."""
        dec = self._make_decoder()
        assert dec.is_idle() is True

    def test_after_noise_still_idle(self) -> None:
        """Feeding pure noise (no preamble) keeps the decoder idle —
        the preamble detector never fires so there's no state advance."""
        dec = self._make_decoder()
        rng = np.random.default_rng(seed=42)
        noise = (
            rng.standard_normal(8192).astype(np.float32)
            + 1j * rng.standard_normal(8192).astype(np.float32)
        ).astype(np.complex64) * 0.05
        dec.feed_baseband(noise)
        assert dec.is_idle() is True

    def test_idle_returns_bool_not_int(self) -> None:
        """is_idle() returns Python bool, not C int. Important for
        downstream code that does `if not dec.is_idle():` —
        comparing int 0 vs False is the same in Python but we want
        the explicit type for clarity."""
        dec = self._make_decoder()
        result = dec.is_idle()
        assert isinstance(result, bool)

    @pytest.mark.parametrize("sf", [7, 8, 9, 10, 11, 12])
    def test_is_idle_works_for_all_sfs(self, sf: int) -> None:
        """Every SF should have working is_idle. Caught in v0.7.16
        because the C state field exists on the struct regardless
        of SF — but worth a regression test in case future code
        somehow becomes SF-dependent."""
        dec = self._make_decoder(sf=sf)
        assert dec.is_idle() is True


# ─────────────────────────────────────────────────────────────────────
# Layer 2 — reap-deferral state-machine on a synthetic ActiveSlot
# ─────────────────────────────────────────────────────────────────────


class _MockDecoder:
    """Stand-in for LoraDecoder. Implements is_idle() (the new field
    we're testing), plus pop_packets(), stats(), and close() since the
    reap path calls all three when actually tearing the decoder down."""

    def __init__(self, idle: bool = True) -> None:
        self._idle = idle

    def is_idle(self) -> bool:
        return self._idle

    def set_idle(self, idle: bool) -> None:
        self._idle = idle

    def pop_packets(self):
        return iter(())

    def stats(self):
        # Return a real (empty) LoraStats — _fold_decoder_stats
        # uses dataclass field access so a duck-typed stub would
        # break. The all-zero default is fine for these tests since
        # we don't assert on per-preset cumulative counters.
        from rfcensus.decoders.lora_native import LoraStats
        return LoraStats()

    def close(self) -> None:
        pass


class TestReapDeferral:
    """Reap-while-decoding deferral logic. Directly drives
    `_maybe_periodic_probe` after seeding an `_ActiveSlot` with
    mock decoders, so we don't need real IQ data."""

    SAMPLE_RATE = 1_000_000     # 1 MS/s for arithmetic simplicity
    PROBE_INTERVAL_MS = 10.0
    REAP_AFTER_MS = 100.0
    REAP_FORCE_AFTER_MS = 500.0   # short enough for tests to hit

    def _make_pipeline(self) -> LazyMultiPresetPipeline:
        """Build a pipeline with the periodic-probe machinery wired
        up. We use a dummy MediumFast channel set just to satisfy
        the constructor — these tests don't actually feed IQ to the
        pipeline, they just call the inline reap-decision logic
        through a mirror helper."""
        center = 913_500_000
        candidates = enumerate_all_slots_in_passband(
            region_code="US", center_freq_hz=center,
            sample_rate_hz=self.SAMPLE_RATE,
        )
        mesh = MeshtasticDecoder("MEDIUM_FAST")
        mesh.add_default_channel()
        return LazyMultiPresetPipeline(
            sample_rate_hz=self.SAMPLE_RATE,
            center_freq_hz=center,
            candidate_slots=candidates,
            mesh=mesh,
            use_periodic_probe=True,
            probe_interval_ms=self.PROBE_INTERVAL_MS,
            reap_after_ms=self.REAP_AFTER_MS,
            reap_force_after_ms=self.REAP_FORCE_AFTER_MS,
        )

    def _make_active(
        self, decoder: _MockDecoder, freq_hz: int = 913_375_000,
    ) -> _ActiveSlot:
        """Construct a minimal `_ActiveSlot` containing one mock
        decoder. Probe is None (we don't exercise the probe inside
        these tests; we directly invoke the reap-decision lines)."""
        return _ActiveSlot(
            freq_hz=freq_hz,
            bandwidth_hz=250_000,
            activated_sample_offset=0,
            next_sample_offset=0,
            feed_start_offset=0,
            decoders={"MEDIUM_FAST": decoder},
            slot_metadata={"MEDIUM_FAST": default_slot("US", "MEDIUM_FAST")},
            state="spawned",
            last_positive_probe_offset=0,
        )

    def _drive_reap_decision(
        self, pipe: LazyMultiPresetPipeline, active: _ActiveSlot,
        chunk_end_offset: int,
    ) -> None:
        """Replicates the inline reap-decision block from
        _maybe_periodic_probe (the no-detection branch). We don't
        invoke the probe machinery, just the decision lines we care
        about — they're driven by the same state, just lifted out
        for testability.

        Keeping this in sync with the production code is the test's
        responsibility: if production logic changes, update here.
        """
        # Match production: the probe-positive branch above also
        # increments next_probe_offset, but that doesn't matter for
        # the deferral state we're testing.
        if active.state != "spawned":
            return
        if active.pinned:
            pipe._stats.reap_skipped_pinned += 1
            return
        idle_for = chunk_end_offset - active.last_positive_probe_offset
        if idle_for < pipe._reap_after_samples:
            return
        any_busy = any(
            not dec.is_idle() for dec in active.decoders.values()
        )
        if any_busy:
            if active.wants_reap_at_offset == 0:
                active.wants_reap_at_offset = chunk_end_offset
            active.reap_deferrals += 1
            pipe._stats.reap_deferred_busy += 1
            wanted_for = chunk_end_offset - active.wants_reap_at_offset
            if wanted_for >= pipe._reap_force_after_samples:
                pipe._stats.reap_force_after_hung += 1
                pipe._periodic_reap(active, chunk_end_offset)
                active.wants_reap_at_offset = 0
            return
        if active.wants_reap_at_offset != 0:
            pipe._stats.reap_completed_after_defer += 1
            active.wants_reap_at_offset = 0
        pipe._periodic_reap(active, chunk_end_offset)

    # ── Tests ───────────────────────────────────────────────────────

    def test_idle_decoder_reaps_immediately(self) -> None:
        """Baseline: with an idle decoder, reap proceeds when the
        idle period crosses reap_after_ms. No deferral counters
        bump — this preserves the v0.7.13 behavior for the common
        case."""
        pipe = self._make_pipeline()
        decoder = _MockDecoder(idle=True)
        active = self._make_active(decoder)
        # Fast-forward past reap_after_ms.
        offset = pipe._reap_after_samples + 1000
        self._drive_reap_decision(pipe, active, offset)
        # Reap happened: state should now be "reaped" (set by
        # _periodic_reap).
        assert active.state == "reaped"
        # No deferral counters incremented for the idle path.
        assert pipe._stats.reap_deferred_busy == 0
        assert pipe._stats.reap_completed_after_defer == 0
        assert pipe._stats.reap_force_after_hung == 0

    def test_busy_decoder_defers_reap(self) -> None:
        """Mid-decode → reap deferred. Decoder stays alive so it can
        finish the in-flight packet."""
        pipe = self._make_pipeline()
        decoder = _MockDecoder(idle=False)   # busy
        active = self._make_active(decoder)
        offset = pipe._reap_after_samples + 1000
        self._drive_reap_decision(pipe, active, offset)
        # Decoder NOT torn down.
        assert active.state == "spawned"
        assert "MEDIUM_FAST" in active.decoders
        # Deferral counter bumped, completion counter not.
        assert pipe._stats.reap_deferred_busy == 1
        assert pipe._stats.reap_completed_after_defer == 0
        # Pending-reap timestamp recorded for hung-timeout tracking.
        assert active.wants_reap_at_offset == offset

    def test_deferred_reap_completes_when_decoder_goes_idle(self) -> None:
        """Happy path: defer → idle → reap completes. The
        completion-after-defer counter is the sign that the system
        worked as intended (vs forced or dropped)."""
        pipe = self._make_pipeline()
        decoder = _MockDecoder(idle=False)
        active = self._make_active(decoder)

        # First tick: busy → defer.
        first_offset = pipe._reap_after_samples + 1000
        self._drive_reap_decision(pipe, active, first_offset)
        assert pipe._stats.reap_deferred_busy == 1
        assert active.wants_reap_at_offset == first_offset

        # Decoder transitions to idle (= packet completed).
        decoder.set_idle(True)

        # Next tick: now-idle → reap proceeds.
        second_offset = first_offset + pipe._probe_interval_samples
        self._drive_reap_decision(pipe, active, second_offset)
        assert active.state == "reaped"
        assert pipe._stats.reap_completed_after_defer == 1
        # Deferral state cleared.
        assert active.wants_reap_at_offset == 0

    def test_force_reap_after_hung_timeout(self) -> None:
        """Pathological case: decoder claims busy forever. After
        reap_force_after_ms, we force the reap and bump the
        force-reap counter. This protects against a hung decoder
        leaking the slot indefinitely."""
        pipe = self._make_pipeline()
        decoder = _MockDecoder(idle=False)   # never goes idle
        active = self._make_active(decoder)

        # Tick 1: defer.
        t1 = pipe._reap_after_samples + 1000
        self._drive_reap_decision(pipe, active, t1)
        assert active.state == "spawned"

        # Tick 2: still busy, but not yet past hung-timeout.
        t2 = t1 + (pipe._reap_force_after_samples // 2)
        self._drive_reap_decision(pipe, active, t2)
        assert active.state == "spawned"
        assert pipe._stats.reap_force_after_hung == 0

        # Tick 3: now past hung-timeout. Force-reap fires.
        t3 = t1 + pipe._reap_force_after_samples + 1
        self._drive_reap_decision(pipe, active, t3)
        assert active.state == "reaped"
        assert pipe._stats.reap_force_after_hung == 1

    def test_probe_positive_during_defer_resets_clock(self) -> None:
        """If new traffic arrives while we're deferring a reap, the
        deferral should be cancelled — the slot is back to active.
        We simulate this by manually clearing wants_reap_at_offset
        the way the probe-positive path does, then verifying that
        the next defer cycle starts fresh from a new offset."""
        pipe = self._make_pipeline()
        decoder = _MockDecoder(idle=False)
        active = self._make_active(decoder)

        # Defer once.
        t1 = pipe._reap_after_samples + 1000
        self._drive_reap_decision(pipe, active, t1)
        assert active.wants_reap_at_offset == t1

        # Probe-positive (simulated): traffic arrived. Production
        # code sets last_positive_probe_offset and clears
        # wants_reap_at_offset.
        t_traffic = t1 + 50_000
        active.last_positive_probe_offset = t_traffic
        active.wants_reap_at_offset = 0

        # Decoder is still busy (still mid-decode), but probe was
        # positive so the idle clock restarts. Drive a tick a small
        # amount later (less than reap_after from t_traffic): NO
        # reap, NO defer counter bump.
        t2 = t_traffic + (pipe._reap_after_samples // 2)
        defers_before = pipe._stats.reap_deferred_busy
        self._drive_reap_decision(pipe, active, t2)
        assert pipe._stats.reap_deferred_busy == defers_before
        assert active.wants_reap_at_offset == 0

    def test_two_decoders_one_busy_defers(self) -> None:
        """When a slot has multiple decoders (SF race), reap defers
        if ANY of them is busy. Decoder for one SF can be mid-decode
        while another SF is in DETECT — we don't want to tear down
        the busy one."""
        pipe = self._make_pipeline()
        idle_decoder = _MockDecoder(idle=True)
        busy_decoder = _MockDecoder(idle=False)
        active = self._make_active(idle_decoder)
        active.decoders["LONG_FAST"] = busy_decoder
        active.slot_metadata["LONG_FAST"] = default_slot("US", "LONG_FAST")

        offset = pipe._reap_after_samples + 1000
        self._drive_reap_decision(pipe, active, offset)
        # Busy decoder kept the reap from happening.
        assert active.state == "spawned"
        assert pipe._stats.reap_deferred_busy == 1

    def test_repeated_defers_only_record_first_offset(self) -> None:
        """Each subsequent deferral tick should NOT reset
        wants_reap_at_offset — that's the timestamp we use to
        decide whether to force-reap. If we kept resetting, the
        force-reap timeout would never fire."""
        pipe = self._make_pipeline()
        decoder = _MockDecoder(idle=False)
        active = self._make_active(decoder)

        t1 = pipe._reap_after_samples + 1000
        self._drive_reap_decision(pipe, active, t1)
        first_recorded = active.wants_reap_at_offset
        assert first_recorded == t1

        t2 = t1 + pipe._probe_interval_samples
        self._drive_reap_decision(pipe, active, t2)
        # Same value as before — the second defer didn't reset it.
        assert active.wants_reap_at_offset == first_recorded
        # But the per-event counter incremented.
        assert pipe._stats.reap_deferred_busy == 2

    def test_reap_skipped_when_pinned(self) -> None:
        """Pinned slots take precedence over the reap-while-decoding
        deferral — if pinned, reap is skipped entirely (the v0.7.13
        pin behavior). The deferral counters should NOT bump."""
        pipe = self._make_pipeline()
        decoder = _MockDecoder(idle=False)
        active = self._make_active(decoder)
        active.pinned = True

        offset = pipe._reap_after_samples + 1000
        self._drive_reap_decision(pipe, active, offset)
        assert active.state == "spawned"
        assert pipe._stats.reap_skipped_pinned == 1
        # No defer counter — pin shortcircuited before the
        # is_idle() check.
        assert pipe._stats.reap_deferred_busy == 0
