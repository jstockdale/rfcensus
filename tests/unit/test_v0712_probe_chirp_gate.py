"""v0.7.12: probe SF correctness fix + adaptive threshold + opt-in gate.

The big find this round: the BlindProbe's reference downchirp formula
was incremental phase accumulation, which produced a chirp slope
wrong by the oversample factor. As a result the probe reported SFs
off-by-one (it said SF8 for SF9 signals, etc). Fix: switch to the
closed-form formula `phase[n] = π·(n²/N - n)` matching
`lora_build_chirps` in lora_chirp.c, and use oversample=1 to match
the decoder's internal expectation.

These tests pin the corrected SF identification and verify the new
adaptive-threshold + opt-in gate behavior.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REAL_CAPTURE = Path("/tmp/meshtastic_30s_913_5mhz_1msps.cu8")


# ─────────────────────────────────────────────────────────────────────
# Probe SF correctness regression
# ─────────────────────────────────────────────────────────────────────


class TestProbeChirpFormula:
    """The closed-form chirp formula is what makes probe SF labels
    correct. These tests guard against any future regression that
    re-introduces the incremental-phase formula."""

    @staticmethod
    def _make_real_lora_chirp(sf: int, n_samples: int) -> np.ndarray:
        """Build an upchirp matching the C decoder's lora_build_chirps
        formula EXACTLY (closed-form, π·(n²/N - n)).

        n_samples = N for one symbol period at BW rate."""
        N = n_samples
        n = np.arange(N)
        phase = np.pi * (n * n / N - n)
        return np.exp(1j * phase).astype(np.complex64)

    def test_sf_identification_is_correct(self) -> None:
        """SF9 signal must be identified as SF9 (not SF8 or SF10).
        This is the regression we just fixed: the incremental-phase
        chirp formula produced an off-by-one in SF identification."""
        from rfcensus.decoders.blind_probe import BlindProbe

        # Build a perfect SF9 upchirp at BW rate (oversample=1):
        # N = 2^9 = 512 samples per symbol.
        sf = 9
        N = 1 << sf
        chirp = self._make_real_lora_chirp(sf, N)
        # Pad to max probe N (SF11 = 2048)
        max_N = 1 << 11
        big = np.tile(chirp, max_N // N + 1)[:max_N].astype(np.complex64)

        probe = BlindProbe([7, 8, 9, 10, 11], oversample=1,
                            snr_threshold_db=20.0)
        results = probe.scan(big)
        # The strongest detected SF must be SF9, and no other SF
        # should be detected (their chirp slopes don't match).
        snrs = {r.sf: r.snr_db for r in results}
        detected = [r.sf for r in results if r.detected]
        assert detected == [9], (
            f"SF identification wrong: detected={detected}, "
            f"all SNRs={snrs}"
        )
        assert snrs[9] > 30, f"SF9 SNR should be very high, got {snrs[9]}"
        # All wrong-SF SNRs should be below threshold (well below for
        # adjacent SFs since the chirp slope mismatches).
        for sf_x in [7, 8, 10, 11]:
            assert snrs[sf_x] < 15, (
                f"SF{sf_x} cross-detection too high: {snrs[sf_x]} dB"
            )

    def test_each_sf_is_uniquely_identified(self) -> None:
        """For each SF in 7..11, a clean chirp at that SF must be
        detected as ONLY that SF, not any neighbor."""
        from rfcensus.decoders.blind_probe import BlindProbe

        for true_sf in (7, 8, 9, 10, 11):
            N = 1 << true_sf
            chirp = self._make_real_lora_chirp(true_sf, N)
            max_N = 1 << 11
            big = np.tile(chirp, max_N // N + 1)[:max_N].astype(np.complex64)
            probe = BlindProbe([7, 8, 9, 10, 11], oversample=1,
                                snr_threshold_db=20.0)
            detected = probe.detected_sfs(big)
            assert detected == [true_sf], (
                f"SF{true_sf} chirp identified wrong: detected={detected}"
            )

    @pytest.mark.skipif(
        not REAL_CAPTURE.exists(),
        reason="real capture required",
    )
    def test_probe_agrees_with_decoder_on_real_signal(self) -> None:
        """The first packet in the 30s capture is a known SF9
        MEDIUM_FAST transmission at slot 913.375. Probe must agree."""
        from rfcensus.decoders.shared_channelizer import SharedChannelizer
        from rfcensus.decoders.blind_probe import BlindProbe

        with open(REAL_CAPTURE, "rb") as f:
            data = f.read()
        # Packet at sample offset 453,120, on slot 913.375 (mix=+125k)
        start = 0
        chunk = data[start * 2: 1_000_000 * 2]    # first 1s

        ch = SharedChannelizer(
            sample_rate_hz=1_000_000, bandwidth_hz=250_000,
            mix_freq_hz=125_000,    # 913.500 - 913.375
        )
        baseband = ch.feed_cu8(chunk)
        ch.close()

        probe = BlindProbe([7, 8, 9, 10, 11], oversample=1,
                            snr_threshold_db=20.0)
        min_n = probe.min_samples_required
        # Slide windows across; collect SFs that ever fire
        ever_detected: set[int] = set()
        for off in range(0, len(baseband) - min_n, min_n // 4):
            for r in probe.scan(baseband[off:off + min_n]):
                if r.detected:
                    ever_detected.add(r.sf)
        # The signal IS SF9; probe should detect SF9. Other SFs may
        # ALSO fire occasionally on noise transients but SF9 must be
        # in the set.
        assert 9 in ever_detected, (
            f"SF9 not detected anywhere on a known-SF9 signal: "
            f"detected={sorted(ever_detected)}"
        )


# ─────────────────────────────────────────────────────────────────────
# Probe gating + recall preservation
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not REAL_CAPTURE.exists(),
    reason="real capture required",
)
class TestProbeGate:
    """Pipeline behavior with v0.7.12 probe filtering and the opt-in
    activation gate."""

    _run_cache: dict = {}

    @classmethod
    def _run(cls, env_overrides: dict) -> tuple[int, str]:
        key = tuple(sorted(env_overrides.items()))
        cached = cls._run_cache.get(key)
        if cached is not None:
            return cached
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
        out = (n, result.stdout + result.stderr)
        cls._run_cache[key] = out
        return out

    def test_default_recall_preserved(self) -> None:
        """v0.7.12+ default config (probe filter ON, gate OFF, periodic
        probe ON) must NOT drop recall vs legacy. v0.7.13's periodic
        probe + reap-respawn actually IMPROVES recall on captures with
        long active periods (catches multi-packet sequences where the
        detector emits one ACTIVATE then stays active across packets)."""
        n, _ = self._run({})
        n_legacy, _ = self._run({
            "RFCENSUS_NO_SHARED_CHAN": "1",
            "RFCENSUS_NO_BLIND_PROBE": "1",
            "RFCENSUS_NO_PERIODIC_PROBE": "1",
        })
        assert n >= n_legacy, (
            f"v0.7.12+ default dropped recall: {n} vs legacy {n_legacy}"
        )

    def test_gate_off_is_default(self) -> None:
        """The gate should be OFF by default (must be explicitly
        opted into via RFCENSUS_PROBE_GATE=1) — gating regresses
        recall on activations whose preamble starts AFTER the
        detector fires."""
        n_default, out_default = self._run({})
        # Gate-related stat (probe_rejected) should be 0 in default
        # mode because gate is off. Look for "0 activations rejected"
        # in summary text — that's our diagnostic.
        assert "0 activations rejected by gate" in out_default, (
            "default mode unexpectedly gated activations:\n"
            + out_default[-2000:]
        )

    def test_probe_filter_skipped_some_decoders(self) -> None:
        """Probe should be filtering wrong-SF decoders out of the
        spawn list when it DOES detect something. On the 30s capture
        the lookback usually doesn't contain a preamble at activate
        time (detector fires on leading-edge energy), so the probe
        falls back to spawn-all most of the time. We verify that
        probe scans run at all."""
        n, out = self._run({})
        # Find the "probe scans: N" line.
        import re
        m = re.search(r"probe scans:\s+(\d+)", out)
        assert m is not None, f"probe stats line missing in:\n{out[-2000:]}"
        scans = int(m.group(1))
        assert scans > 0, (
            f"probe never ran — wiring regression? out:\n{out[-2000:]}"
        )

    def test_gate_on_is_an_opt_in_perf_lever(self) -> None:
        """RFCENSUS_PROBE_GATE=1 enables the gate. We verify it
        actually activates the gate path (probe_rejected > 0). We
        do NOT assert recall here because the gate is known to
        regress on captures where activation precedes preamble —
        that's a documented limitation."""
        n, out = self._run({"RFCENSUS_PROBE_GATE": "1"})
        import re
        m = re.search(r"(\d+)\s+activations rejected by gate", out)
        assert m is not None, "no 'rejected by gate' line in output"
        rejected = int(m.group(1))
        assert rejected > 0, (
            f"gate ON but rejected zero activations — env-var path "
            f"may be broken. out:\n{out[-2000:]}"
        )
