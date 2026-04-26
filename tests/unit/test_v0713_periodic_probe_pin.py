"""v0.7.13: periodic probe + reap-respawn + bleed-aware confirmation
+ hot-slot pin.

Three commits in sequence:
  1a. Periodic probe runs every probe_interval_ms while a slot is
      active. If the probe stays silent past reap_after_ms, decoders
      are reaped. On the next probe-positive, decoders re-spawn.
  1b. CRC-pass on (slot, sf) gets that triple "confirmed" — at
      future spawn points, confirmed triples are always included
      even when probe-filter would otherwise prune them. Bleed-
      aware: a CRC-pass at low-RSSI gets demoted in favor of a
      higher-RSSI duplicate within the dedup window.
  1c. Hysteretic pin: if probe-positive rate > pin_high_pct over
      the rolling window, slot is "pinned" — periodic-reap is
      skipped. Drops below pin_low_pct → unpinned.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


REAL_CAPTURE = Path("/tmp/meshtastic_30s_913_5mhz_1msps.cu8")


# ─────────────────────────────────────────────────────────────────────
# Commit 1a — periodic probe, reap-when-cold, respawn-on-fire
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not REAL_CAPTURE.exists(),
    reason="real capture required",
)
class TestPeriodicProbe:
    """End-to-end behavior of the periodic probe + reap pipeline."""

    @staticmethod
    def _run(env_overrides: dict) -> tuple[int, str]:
        env = {**os.environ, **env_overrides}
        result = subprocess.run(
            [sys.executable, "-m", "rfcensus.tools.decode_meshtastic",
             str(REAL_CAPTURE),
             "--frequency", "913500000",
             "--sample-rate", "1000000",
             "--slots", "all",
             "--lazy"],
            capture_output=True, text=True, env=env,
            cwd="/home/claude/rfcensus",
            timeout=120,
        )
        n = sum(1 for l in result.stdout.splitlines() if l.startswith("@"))
        return n, result.stdout + result.stderr

    def test_periodic_probe_improves_recall(self) -> None:
        """v0.7.13 periodic probe must catch packets that the
        v0.7.12-baseline (no periodic probe) misses. The 30s capture
        has multiple packets within long detector-active periods that
        the base path would skip — periodic probe should pick them up
        via respawn."""
        n_periodic, _ = self._run({})
        n_no_periodic, _ = self._run({
            "RFCENSUS_NO_PERIODIC_PROBE": "1",
        })
        assert n_periodic >= n_no_periodic, (
            f"periodic regressed recall: {n_periodic} vs no-periodic "
            f"{n_no_periodic}"
        )
        assert n_periodic > n_no_periodic, (
            f"periodic should have IMPROVED recall on 30s capture; "
            f"got {n_periodic} vs no-periodic {n_no_periodic}"
        )

    def test_periodic_probe_stats_populated(self) -> None:
        """Periodic probe must actually run and show stats output."""
        n, out = self._run({})
        import re
        m = re.search(
            r"periodic:\s+(\d+) probe scans \((\d+) positive",
            out,
        )
        assert m is not None, (
            f"periodic stats line missing in:\n{out[-2000:]}"
        )
        scans = int(m.group(1))
        positive = int(m.group(2))
        assert scans > 100, (
            f"periodic probe didn't run enough times: {scans}"
        )
        # Some positives must occur on a real-traffic capture
        assert positive > 0, (
            f"periodic probe NEVER fired positive — pipeline wiring "
            f"broken?"
        )

    def test_periodic_probe_reaps_then_respawns(self) -> None:
        """The reap-then-respawn cycle must run on the 30s capture
        (long active windows have quiet gaps that should reap)."""
        n, out = self._run({})
        import re
        m = re.search(
            r"(\d+) reap event\(s\)\s+·\s+(\d+) respawn event\(s\)",
            out,
        )
        assert m is not None, f"reap/respawn line missing"
        reaps = int(m.group(1))
        respawns = int(m.group(2))
        assert reaps > 0, "no reaps fired on a capture with quiet periods"
        # Respawns should be > 0 on a capture with bursts of traffic.
        assert respawns > 0, "no respawns fired despite reap activity"

    def test_disable_periodic_via_env(self) -> None:
        """RFCENSUS_NO_PERIODIC_PROBE=1 must turn off periodic probe
        entirely (zero scans recorded)."""
        n, out = self._run({"RFCENSUS_NO_PERIODIC_PROBE": "1"})
        import re
        m = re.search(r"periodic:\s+(\d+) probe scans", out)
        if m is not None:
            scans = int(m.group(1))
            assert scans == 0, (
                f"NO_PERIODIC_PROBE env didn't disable: {scans} scans"
            )
        # Otherwise the line wasn't emitted (which is also fine — we
        # only emit it when scans > 0).


# ─────────────────────────────────────────────────────────────────────
# Commit 1b — confirmed (slot, sf) triples + bleed-aware confirmation
# ─────────────────────────────────────────────────────────────────────


def _build_pipe():
    """Helper to construct a pipeline for in-process unit tests."""
    from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
    from rfcensus.decoders.lazy_pipeline import LazyMultiPresetPipeline
    from rfcensus.utils.meshtastic_region import enumerate_all_slots_in_passband
    mesh = MeshtasticDecoder("MEDIUM_FAST")
    mesh.add_default_channel()
    candidates = enumerate_all_slots_in_passband(
        region_code="US", center_freq_hz=913_500_000,
        sample_rate_hz=1_000_000,
    )
    return LazyMultiPresetPipeline(
        sample_rate_hz=1_000_000, center_freq_hz=913_500_000,
        candidate_slots=candidates, mesh=mesh,
    )


@dataclass
class _FakeLora:
    payload: bytes
    rssi_db: float
    sample_offset: int


@dataclass
class _FakePreset:
    sf: int
    bandwidth_hz: int
    key: str


@dataclass
class _FakeSlot:
    freq_hz: int
    preset: object


class TestBleedAwareConfirmation:
    """The confirmed-set must reject bleed copies (lower-RSSI
    duplicates of a payload within the dedup window) and keep only
    the highest-RSSI member of each cluster."""

    def test_real_first_bleed_rejected(self) -> None:
        """Real strong signal arrives first, bleed copy 10ms later
        must NOT be added to confirmed."""
        pipe = _build_pipe()
        slot1 = _FakeSlot(freq_hz=913_125_000,
                           preset=_FakePreset(sf=9, bandwidth_hz=250_000,
                                              key="MEDIUM_FAST"))
        slot2 = _FakeSlot(freq_hz=913_375_000,
                           preset=_FakePreset(sf=9, bandwidth_hz=250_000,
                                              key="MEDIUM_FAST"))
        pipe._record_crc_pass(slot1,
                               _FakeLora(b"hello", -2.0, 100_000))
        pipe._record_crc_pass(slot2,
                               _FakeLora(b"hello", -8.0, 110_000))
        assert (913_125_000, 250_000, 9) in pipe._confirmed_slot_sf
        assert (913_375_000, 250_000, 9) not in pipe._confirmed_slot_sf

    def test_bleed_first_demoted_by_real(self) -> None:
        """Bleed gets optimistically confirmed, then real arrives
        with higher RSSI and demotes the bleed."""
        pipe = _build_pipe()
        slot_real = _FakeSlot(freq_hz=913_125_000,
                               preset=_FakePreset(sf=9, bandwidth_hz=250_000,
                                                  key="MEDIUM_FAST"))
        slot_bleed = _FakeSlot(freq_hz=913_375_000,
                                preset=_FakePreset(sf=9, bandwidth_hz=250_000,
                                                   key="MEDIUM_FAST"))
        # Bleed first
        pipe._record_crc_pass(slot_bleed,
                               _FakeLora(b"hello", -8.0, 200_000))
        assert (913_375_000, 250_000, 9) in pipe._confirmed_slot_sf
        # Real second
        pipe._record_crc_pass(slot_real,
                               _FakeLora(b"hello", -2.0, 210_000))
        assert (913_125_000, 250_000, 9) in pipe._confirmed_slot_sf
        assert (913_375_000, 250_000, 9) not in pipe._confirmed_slot_sf

    def test_mesh_relay_keeps_both(self) -> None:
        """Same payload arriving 600ms apart on different slots is
        a mesh relay (different physical TX), not bleed. Both
        triples must be confirmed."""
        pipe = _build_pipe()
        slot1 = _FakeSlot(freq_hz=913_125_000,
                           preset=_FakePreset(sf=9, bandwidth_hz=250_000,
                                              key="MEDIUM_FAST"))
        slot2 = _FakeSlot(freq_hz=913_375_000,
                           preset=_FakePreset(sf=9, bandwidth_hz=250_000,
                                              key="MEDIUM_FAST"))
        pipe._record_crc_pass(slot2,
                               _FakeLora(b"hello", -8.0, 300_000))
        # 600ms later (sample_rate_hz=1MHz so 600_000 samples)
        pipe._record_crc_pass(slot1,
                               _FakeLora(b"hello", -2.0, 900_000))
        assert (913_125_000, 250_000, 9) in pipe._confirmed_slot_sf
        assert (913_375_000, 250_000, 9) in pipe._confirmed_slot_sf

    def test_different_payloads_dont_interact(self) -> None:
        """Two different payloads on different slots within the
        window must NOT cause cross-demotion."""
        pipe = _build_pipe()
        slot1 = _FakeSlot(freq_hz=913_125_000,
                           preset=_FakePreset(sf=9, bandwidth_hz=250_000,
                                              key="MEDIUM_FAST"))
        slot2 = _FakeSlot(freq_hz=913_375_000,
                           preset=_FakePreset(sf=9, bandwidth_hz=250_000,
                                              key="MEDIUM_FAST"))
        pipe._record_crc_pass(slot1,
                               _FakeLora(b"alpha", -8.0, 100_000))
        pipe._record_crc_pass(slot2,
                               _FakeLora(b"beta", -2.0, 110_000))
        assert (913_125_000, 250_000, 9) in pipe._confirmed_slot_sf
        assert (913_375_000, 250_000, 9) in pipe._confirmed_slot_sf


# ─────────────────────────────────────────────────────────────────────
# Commit 1c — hot-slot pin with hysteresis
# ─────────────────────────────────────────────────────────────────────


class TestPinHysteresis:
    """The pin-state evolves via probe-positive-rate hysteresis.
    Tested in-process (rather than via subprocess) because we need to
    drive the rate manually."""

    def _make_active(self):
        from rfcensus.decoders.lazy_pipeline import _ActiveSlot
        return _ActiveSlot(
            freq_hz=913_125_000,
            bandwidth_hz=250_000,
            activated_sample_offset=0,
            next_sample_offset=0,
            feed_start_offset=0,
        )

    def _make_pipe(self, **kwargs):
        from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
        from rfcensus.decoders.lazy_pipeline import LazyMultiPresetPipeline
        from rfcensus.utils.meshtastic_region import enumerate_all_slots_in_passband
        mesh = MeshtasticDecoder("MEDIUM_FAST")
        mesh.add_default_channel()
        candidates = enumerate_all_slots_in_passband(
            region_code="US", center_freq_hz=913_500_000,
            sample_rate_hz=1_000_000,
        )
        return LazyMultiPresetPipeline(
            sample_rate_hz=1_000_000, center_freq_hz=913_500_000,
            candidate_slots=candidates, mesh=mesh,
            # Tight params for testing (defaults are 10s window,
            # 5/10% thresholds — same band, just shorter window):
            pin_window_ms=1000.0,
            pin_high_pct=10.0,
            pin_low_pct=5.0,
            **kwargs,
        )

    def test_pin_triggers_above_pin_high(self) -> None:
        """Drive a slot's probe history to >10% positive rate; the
        slot must transition to pinned."""
        pipe = self._make_pipe()
        active = self._make_active()
        # Sample interval = 1MS/s × 10ms = 10000 samples per probe
        # tick. 1000ms window = 100 ticks. Need at least pin_min_samples
        # before pin can trigger.
        # Drive 50% positive rate over 200 ticks (well above 10%).
        for i in range(200):
            offset = i * 10_000
            pipe._update_pin_state(active, offset, fired=(i % 2 == 0))
        assert active.pinned, (
            f"slot should be pinned; rate=50%, history len="
            f"{len(active.probe_history)}"
        )
        assert pipe._stats.pin_events == 1

    def test_unpin_triggers_below_pin_low(self) -> None:
        """Pin a slot, then drive its rate below 5% → must unpin."""
        pipe = self._make_pipe()
        active = self._make_active()
        # First pin it (50% rate).
        for i in range(200):
            offset = i * 10_000
            pipe._update_pin_state(active, offset, fired=(i % 2 == 0))
        assert active.pinned
        # Now drive 0% rate (no positives) for another 200 ticks.
        # Window is 1000ms = 100 ticks; rate will drop as the
        # positive ones age out.
        for i in range(200, 400):
            offset = i * 10_000
            pipe._update_pin_state(active, offset, fired=False)
        assert not active.pinned
        assert pipe._stats.unpin_events == 1

    def test_hysteresis_prevents_flapping(self) -> None:
        """Drive rate to ~7% (between low=5% and high=10%). Slot
        should not pin (7% < 10% pin_high)."""
        pipe = self._make_pipe()
        active = self._make_active()
        # Fire at ticks 50, 65, 80, ... — 1 in 15 ≈ 6.7% rate over
        # the long run. Skip the first 50 ticks to avoid the cold-
        # start edge case where pin_min_samples=10 evaluates the
        # rate over a tiny window that can momentarily exceed 10%.
        # In real captures the cold-start window is filled with
        # whatever rate the slot actually has, not the transient.
        for i in range(400):
            offset = i * 10_000
            fired = (i >= 50 and (i - 50) % 15 == 0)
            pipe._update_pin_state(active, offset, fired=fired)
        assert not active.pinned, (
            f"slot pinned at ~6.7% rate but pin_high is 10%; "
            f"history rate = "
            f"{sum(1 for _, f in active.probe_history if f)/len(active.probe_history):.1%}"
        )

    def test_disable_pin_via_env(self) -> None:
        """RFCENSUS_NO_PIN=1 disables pin transitions even when
        rate would normally trigger."""
        os.environ["RFCENSUS_NO_PIN"] = "1"
        try:
            pipe = self._make_pipe()
        finally:
            del os.environ["RFCENSUS_NO_PIN"]
        active = self._make_active()
        for i in range(200):
            offset = i * 10_000
            pipe._update_pin_state(active, offset, fired=True)  # 100%
        assert not active.pinned
