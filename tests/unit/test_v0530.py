"""Tests for v0.5.30 — per-band FFT bin resolution + orphan detection.

Motivation:
  • Default rtl_power bin width heuristic (`max(10_000, bw // 256)`)
    produces 81 kHz bins on a 26 MHz ISM band — coarse enough to
    hide every narrowband channel (P25 @ 12.5 kHz, POCSAG @ 12.5 kHz,
    NFM voice @ 12.5/25 kHz). Power scans would produce a smeared
    occupancy map that couldn't distinguish individual channels.
  • Per-band `power_scan_bin_hz` override lets configs set bin width
    appropriate to the signal types in that band.
"""

from __future__ import annotations

import pytest


class TestBandConfigPowerScanBinHz:
    """The new per-band FFT bin override."""

    def test_default_bin_hz_is_none(self):
        """Explicit None default = use the heuristic."""
        from rfcensus.config.schema import BandConfig

        b = BandConfig(
            id="test", name="test",
            freq_low=100_000_000, freq_high=101_000_000,
        )
        assert b.power_scan_bin_hz is None

    def test_effective_bin_hz_falls_back_to_heuristic(self):
        """When no explicit value, `effective_power_scan_bin_hz`
        returns the historical `max(10_000, bw / 256)`."""
        from rfcensus.config.schema import BandConfig

        # 26 MHz span → 26e6 / 256 = 101562, max(10k, 101562) = 101562
        wide = BandConfig(
            id="wide", name="wide",
            freq_low=902_000_000, freq_high=928_000_000,
        )
        assert wide.effective_power_scan_bin_hz == 101562

        # 100 kHz span → 100k / 256 = 390, max(10k, 390) = 10000
        narrow = BandConfig(
            id="narrow", name="narrow",
            freq_low=319_450_000, freq_high=319_550_000,
        )
        assert narrow.effective_power_scan_bin_hz == 10_000

    def test_effective_bin_hz_respects_explicit_override(self):
        """When set, `effective_power_scan_bin_hz` returns exactly
        the configured value — no transformation."""
        from rfcensus.config.schema import BandConfig

        b = BandConfig(
            id="p25", name="p25",
            freq_low=851_000_000, freq_high=869_000_000,
            power_scan_bin_hz=6_250,
        )
        assert b.effective_power_scan_bin_hz == 6_250

    def test_reject_sub_khz_bin(self):
        """Values under 1 kHz aren't physically meaningful at
        rtl_power's typical integration time — enforce a floor."""
        from rfcensus.config.schema import BandConfig

        with pytest.raises(ValueError, match="too fine"):
            BandConfig(
                id="too_fine", name="too_fine",
                freq_low=100_000_000, freq_high=101_000_000,
                power_scan_bin_hz=500,  # 500 Hz — not meaningful
            )

    def test_reject_bin_wider_than_half_bandwidth(self):
        """With fewer than 2 bins per sweep, occupancy detection
        can't work. Reject."""
        from rfcensus.config.schema import BandConfig

        with pytest.raises(ValueError, match="exceeds bandwidth/2"):
            BandConfig(
                id="too_wide", name="too_wide",
                freq_low=100_000_000, freq_high=100_100_000,  # 100 kHz BW
                power_scan_bin_hz=60_000,  # > bw/2 = 50 kHz
            )


class TestStrategyUsesEffectiveBinHz:
    """Regression: strategy.py should consult the new property,
    not compute its own bin width inline (which was the bug)."""

    def test_strategy_source_uses_effective_property(self):
        """Source-level check: the old inline formula is gone."""
        import inspect
        from rfcensus.engine import strategy

        src = inspect.getsource(strategy)
        # The old inline formula must be gone
        assert "max(10_000, band.bandwidth_hz // 256)" not in src, (
            "strategy.py still has the inline bin-width formula. "
            "It must call band.effective_power_scan_bin_hz instead."
        )
        # The new property must be used
        assert "effective_power_scan_bin_hz" in src

    def test_spec_receives_explicit_bin_when_band_configured(self):
        """When a band has an explicit bin_hz, the SpectrumSweepSpec
        constructed for it uses exactly that value."""
        from rfcensus.config.schema import BandConfig
        from rfcensus.spectrum.backend import SpectrumSweepSpec

        band = BandConfig(
            id="pocsag", name="pocsag",
            freq_low=929_000_000, freq_high=932_000_000,
            power_scan_bin_hz=6_250,
        )
        # Build the same spec strategy._run_power_scan builds
        spec = SpectrumSweepSpec(
            freq_low=band.freq_low,
            freq_high=band.freq_high,
            bin_width_hz=band.effective_power_scan_bin_hz,
            dwell_ms=200,
            duration_s=720.0,
        )
        assert spec.bin_width_hz == 6_250


class TestBuiltinBandsHaveSensibleBinWidths:
    """The built-in US bands ship with tuned bin widths. A bad
    edit to bands_us.toml that regressed these defaults would
    quietly degrade power scans — guard against it."""

    def test_narrow_channel_bands_have_fine_bins(self):
        """Bands with 12.5 kHz channels (P25, POCSAG, FRS/GMRS,
        marine VHF, business LMR) must have bins ≤ 12.5 kHz so
        individual channels are resolvable."""
        from rfcensus.config.loader import _load_builtin_bands

        bands = {b.id: b for b in _load_builtin_bands("US")}
        narrow_channel_bands = [
            "pocsag_929",
            "frs_gmrs",
            "marine_vhf",
            "business_vhf",
            "business_uhf",
            "p25_700_public_safety",
            "p25_800_public_safety",
            "ais",
        ]
        for bid in narrow_channel_bands:
            if bid not in bands:
                pytest.skip(f"band {bid} not in default config")
            b = bands[bid]
            bin_hz = b.effective_power_scan_bin_hz
            assert bin_hz <= 12_500, (
                f"{bid}: bin_hz={bin_hz} too coarse for narrow-"
                f"channel detection (12.5 kHz channels). Set "
                f"power_scan_bin_hz in bands_us.toml."
            )

    def test_all_default_bands_have_explicit_bin_hz(self):
        """Ship all built-in POWER-SCANNING bands with an explicit
        bin width so operators can see and adjust the value. Bands
        that rely on the heuristic are a footgun — the default
        formula gives 81 kHz bins on a 26 MHz band, too coarse to
        resolve narrowband channels.

        Decoder-only bands are exempt: they don't run a power scan
        at all, so their bin_hz is never consulted. Opt-in bands
        are exempt: they're experimental and we may not have tuned
        a bin width yet.
        """
        from rfcensus.config.loader import _load_builtin_bands
        from rfcensus.config.schema import StrategyKind

        for band in _load_builtin_bands("US"):
            if band.opt_in:
                continue
            if band.strategy == StrategyKind.DECODER_ONLY:
                # No power scan run for this band; bin_hz unused.
                continue
            assert band.power_scan_bin_hz is not None, (
                f"band {band.id}: no explicit power_scan_bin_hz. "
                f"Add one to bands_us.toml (or set strategy to "
                f"decoder_only / opt_in=true)."
            )

    def test_iq_wide_ism_bands_have_reasonable_bins(self):
        """Wide ISM bands (315/319/345/433/915) should have bins
        in the 25-50 kHz range: fine enough to resolve OOK bursts,
        coarse enough to avoid FFT bloat."""
        from rfcensus.config.loader import _load_builtin_bands

        bands = {b.id: b for b in _load_builtin_bands("US")}
        ism_bands = [
            "315_security",
            "interlogix_security",
            "honeywell_security",
            "433_ism",
            "915_ism",
        ]
        for bid in ism_bands:
            if bid not in bands:
                pytest.skip(f"band {bid} not in default config")
            b = bands[bid]
            bin_hz = b.effective_power_scan_bin_hz
            assert 10_000 <= bin_hz <= 50_000, (
                f"{bid}: bin_hz={bin_hz} outside [10k, 50k] range "
                f"for ISM burst detection."
            )


# ──────────────────────────────────────────────────────────────────
# Orphan process detection (v0.5.30)
# ──────────────────────────────────────────────────────────────────


class TestOrphanDetection:
    """SDR binary orphans from prior uncleanly-exited sessions
    hold USB dongles hostage. We need to find them reliably."""

    def test_sdr_binary_names_covers_expected_set(self):
        """The binary-name registry must cover every subprocess
        rfcensus ever spawns. If a new decoder is added, this test
        reminds the author to update the set."""
        from rfcensus.utils.orphan_detect import SDR_BINARY_NAMES

        must_cover = {
            "rtl_tcp",       # broker shared-slot
            "rtl_power",     # spectrum sweep
            "rtl_fm",        # multimon pipeline source
            "rtl_433",       # rtl_433 decoder
            "rtl_ais",       # rtl_ais decoder
            "rtlamr",        # rtlamr decoder
            "multimon-ng",   # multimon decoder
            "direwolf",      # direwolf decoder
        }
        missing = must_cover - SDR_BINARY_NAMES
        assert not missing, (
            f"SDR_BINARY_NAMES is missing: {missing}. Any binary "
            f"rfcensus can spawn must appear here or orphan detection "
            f"will miss it."
        )

    def test_find_orphans_detects_live_sdr_process(
        self, tmp_path, monkeypatch,
    ):
        """End-to-end: spawn a fake process with a matching comm.
        Achieved by copying /bin/sleep to a file named 'rtl_tcp';
        the kernel's comm field is the basename of argv[0] when
        the process exec'd a real binary (shell scripts show up
        with comm=sh because sh is what actually exec'd)."""
        import os
        import shutil
        import subprocess
        import time
        from rfcensus.utils.orphan_detect import find_sdr_orphans

        sleep_bin = shutil.which("sleep")
        if sleep_bin is None:
            pytest.skip("sleep binary not available")
        fake = tmp_path / "rtl_tcp"
        shutil.copy(sleep_bin, fake)

        proc = subprocess.Popen([str(fake), "5"])
        try:
            time.sleep(0.3)
            orphans = find_sdr_orphans(min_age_s=0.0)
            pids = {o.pid for o in orphans}
            assert proc.pid in pids, (
                f"fake rtl_tcp pid={proc.pid} not found in orphans; "
                f"detected pids: {pids}"
            )
            match = next(o for o in orphans if o.pid == proc.pid)
            assert match.comm == "rtl_tcp"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def test_find_orphans_respects_exclude_pids(self, tmp_path):
        """When we pass a PID in exclude_pids, it must not appear
        in the orphan list. Critical for rfcensus's own children
        during a mid-scan check."""
        import shutil
        import subprocess
        import time
        from rfcensus.utils.orphan_detect import find_sdr_orphans

        sleep_bin = shutil.which("sleep")
        if sleep_bin is None:
            pytest.skip("sleep binary not available")
        fake = tmp_path / "rtl_tcp"
        shutil.copy(sleep_bin, fake)

        proc = subprocess.Popen([str(fake), "5"])
        try:
            time.sleep(0.3)
            orphans = find_sdr_orphans(
                exclude_pids={proc.pid}, min_age_s=0.0,
            )
            pids = {o.pid for o in orphans}
            assert proc.pid not in pids
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def test_find_orphans_respects_min_age(self, tmp_path):
        """Just-spawned processes are excluded when min_age_s
        exceeds their actual age. This prevents false positives
        from user processes started moments before rfcensus."""
        import shutil
        import subprocess
        import time
        from rfcensus.utils.orphan_detect import find_sdr_orphans

        sleep_bin = shutil.which("sleep")
        if sleep_bin is None:
            pytest.skip("sleep binary not available")
        fake = tmp_path / "rtl_tcp"
        shutil.copy(sleep_bin, fake)

        proc = subprocess.Popen([str(fake), "5"])
        try:
            time.sleep(0.2)  # definitely younger than 10s
            orphans = find_sdr_orphans(min_age_s=10.0)
            pids = {o.pid for o in orphans}
            assert proc.pid not in pids, (
                "young process should be excluded by min_age_s"
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def test_find_orphans_ignores_non_sdr_processes(self):
        """We MUST NOT report random /proc entries — only those
        whose comm matches SDR_BINARY_NAMES. Test by confirming
        our own Python test runner isn't in the orphan list."""
        import os
        from rfcensus.utils.orphan_detect import find_sdr_orphans

        orphans = find_sdr_orphans(min_age_s=0.0)
        pids = {o.pid for o in orphans}
        # Our own pid shouldn't appear even if some SDR name
        # overlap coincidence happens — find_sdr_orphans filters
        # os.getpid() explicitly
        assert os.getpid() not in pids

    def test_kill_orphans_sigterms_then_sigkills(self, tmp_path):
        """SIGTERM-then-SIGKILL escalation. Use a Python-based fake
        that installs a SIGTERM ignorer, then copy to 'rtl_tcp' so
        comm matches."""
        import os
        import shutil
        import subprocess
        import sys
        import time
        from rfcensus.utils.orphan_detect import (
            find_sdr_orphans, kill_orphans,
        )

        # We need a process that (a) shows up as 'rtl_tcp' in comm
        # and (b) ignores SIGTERM. Approach: wrap Python with a
        # script that ignores SIGTERM and sleeps, but COPY the
        # Python interpreter to a file named 'rtl_tcp' so comm
        # matches. The interpreter reads -c from argv[1].
        fake = tmp_path / "rtl_tcp"
        shutil.copy(sys.executable, fake)
        code = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "while True: time.sleep(0.1)\n"
        )

        proc = subprocess.Popen([str(fake), "-c", code])
        try:
            time.sleep(0.3)
            orphans = find_sdr_orphans(min_age_s=0.0)
            targets = [o for o in orphans if o.pid == proc.pid]
            assert targets, "setup failed — fake rtl_tcp not detected"

            clean, forced = kill_orphans(targets, sigterm_grace_s=0.5)
            # It ignored SIGTERM so it must have needed SIGKILL
            assert forced >= 1, (
                f"expected SIGKILL escalation; clean={clean} forced={forced}"
            )

            # Confirm the process is actually dead
            time.sleep(0.2)
            returncode = proc.poll()
            assert returncode is not None, "process still alive after kill_orphans"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_kill_orphans_empty_list_is_noop(self):
        """Passing [] must return (0, 0) without error."""
        from rfcensus.utils.orphan_detect import kill_orphans
        assert kill_orphans([]) == (0, 0)


class TestScanCommandHasKillOrphansFlag:
    """The `rfcensus scan` CLI must expose --kill-orphans."""

    def test_scan_cli_accepts_kill_orphans(self):
        from rfcensus.commands import inventory

        scan_cmd = inventory.cli_scan
        param_names = {p.name for p in scan_cmd.params}
        assert "kill_orphans" in param_names, (
            f"--kill-orphans flag missing from `scan` command. "
            f"Got params: {sorted(param_names)}"
        )


# ──────────────────────────────────────────────────────────────────
# v0.5.31: orphan → busy-dongle correlation and interactive UX
# ──────────────────────────────────────────────────────────────────


class TestOrphanDeviceIndexCorrelation:
    """guess_orphan_device_indices parses rtl_* cmdlines for `-d N`
    so we can correlate orphans to specific dongles."""

    def _fake(self, pid, cmdline, comm="rtl_tcp"):
        from rfcensus.utils.orphan_detect import OrphanProcess
        return OrphanProcess(
            pid=pid, comm=comm, cmdline=cmdline, age_seconds=3600.0,
        )

    def test_parses_space_separated_device_flag(self):
        from rfcensus.utils.orphan_detect import guess_orphan_device_indices

        orphans = [
            self._fake(100, "/usr/bin/rtl_tcp -a 127.0.0.1 -p 1234 -d 0 -s 2400000"),
            self._fake(101, "/usr/bin/rtl_fm -d 3 -f 915000000"),
        ]
        indices = guess_orphan_device_indices(orphans)
        assert sorted(indices.keys()) == [0, 3]
        assert indices[0][0].pid == 100
        assert indices[3][0].pid == 101

    def test_multiple_orphans_same_device_grouped(self):
        """A pipeline like rtl_fm | multimon-ng would have rtl_fm on
        `-d 0` and multimon-ng with no `-d`. But if somehow two
        orphans both target `-d 0` they should be grouped."""
        from rfcensus.utils.orphan_detect import guess_orphan_device_indices

        orphans = [
            self._fake(100, "rtl_tcp -d 0"),
            self._fake(101, "rtl_fm -d 0 -f 915e6"),
        ]
        indices = guess_orphan_device_indices(orphans)
        assert len(indices[0]) == 2

    def test_unparseable_cmdline_goes_to_minus_one(self):
        """multimon-ng has no -d flag; rtl_test sometimes doesn't
        either. Group those under key -1."""
        from rfcensus.utils.orphan_detect import guess_orphan_device_indices

        orphans = [
            self._fake(100, "multimon-ng -a POCSAG512 -t raw"),
            self._fake(101, "rtl_test -t"),
        ]
        indices = guess_orphan_device_indices(orphans)
        assert -1 in indices
        assert len(indices[-1]) == 2

    def test_d_flag_not_confused_with_other_d_prefixes(self):
        """`-demod` or `-debug` (hypothetical) must not be read as
        `-d 0`. The regex requires a space between `-d` and the
        number."""
        from rfcensus.utils.orphan_detect import guess_orphan_device_indices

        orphans = [
            # No real rtl tool uses -demod, but test the boundary
            self._fake(100, "rtl_fm -demodopts something"),
        ]
        indices = guess_orphan_device_indices(orphans)
        # Should NOT parse -demod as -d with value "emodopts"
        assert 100 not in [o.pid for lst in indices.values()
                           for o in lst if lst is indices.get(-1, [])] or \
               orphans[0] in indices.get(-1, [])
        # Stronger: the value e/emodopts isn't a valid int, so the
        # regex requires \d+ — this line should end up under -1
        assert orphans[0] in indices[-1]


class TestSessionRunnerSkipHealthCheck:
    """skip_health_check flag lets callers (the inventory command)
    run check_all() themselves earlier, so they can prompt about
    busy dongles before handing control to SessionRunner."""

    def test_skip_health_check_field_exists(self):
        import inspect
        from rfcensus.engine.session import SessionRunner

        sig = inspect.signature(SessionRunner.__init__)
        assert "skip_health_check" in sig.parameters
        param = sig.parameters["skip_health_check"]
        assert param.default is False, (
            "skip_health_check must default to False so existing "
            "callers get health checks automatically"
        )

    def test_session_source_respects_skip_flag(self):
        """The session source must actually gate the check_all call
        on the skip flag. A test that only checks the field exists
        isn't enough — it's easy to add the field and forget to use it."""
        import inspect
        from rfcensus.engine import session

        src = inspect.getsource(session.SessionRunner.run)
        assert "skip_health_check" in src
        assert "check_all" in src

    def test_inventory_passes_skip_health_check(self):
        """The inventory command must pass skip_health_check=True
        because it runs check_all() before constructing SessionRunner.
        If this gets lost in a refactor, every scan would health-check
        twice (and the user-prompt window would widen redundantly)."""
        import inspect
        from rfcensus.commands import inventory

        src = inspect.getsource(inventory)
        assert "skip_health_check=True" in src


class TestInventoryInteractivePrompt:
    """The inventory command prompts for orphan cleanup ONLY when:
      • One or more dongles returned BUSY from the probe
      • One or more orphan SDR processes are running
      • stdout is a TTY (not piped/CI)
      • --kill-orphans wasn't already set

    We can't easily integration-test the prompt path (it needs a TTY
    and hardware), but we can verify the control-flow code is in
    place via source inspection."""

    def test_inventory_has_busy_plus_orphan_correlation_code(self):
        """Ensures the Phase B correlation logic wasn't accidentally
        removed in a refactor."""
        import inspect
        from rfcensus.commands import inventory

        src = inspect.getsource(inventory)
        assert "guess_orphan_device_indices" in src
        assert "DongleStatus.BUSY" in src
        # The three-way prompt must be there
        assert "kill the orphans" in src
        assert "proceed anyway" in src
        # Specifically the Choice list: k/p/q
        assert 'click.Choice(["k", "p", "q"])' in src

    def test_prompt_gated_on_tty(self):
        """Non-TTY stdout must NOT prompt — it's a script or CI.
        Verify the isatty check is present."""
        import inspect
        from rfcensus.commands import inventory

        src = inspect.getsource(inventory)
        # Both stdin and stdout should be checked so the prompt path
        # is definitely interactive on both ends
        assert "isatty()" in src

    def test_kill_orphans_flag_short_circuits_prompt(self):
        """If --kill-orphans is already set, we shouldn't prompt —
        the user's told us what they want. Verify the code path
        uses `kill_orphans` to decide."""
        import inspect
        from rfcensus.commands import inventory

        src = inspect.getsource(inventory)
        # The should_kill local should be initialized from the flag
        assert "should_kill = kill_orphans" in src


class TestTwoPass915MHzCoverage:
    """v0.5.32: added 915_ism_r900 second pass at 912.6 MHz to
    capture R900 water meters (fixed freq) and lower ERT hop
    channels that the primary 915 MHz center misses.

    rtlamr at 2.4 Msps only demodulates ±1.2 MHz around its
    -centerfreq. A single pass at 915 MHz covers 913.8-916.2 MHz
    = 9% of the 26 MHz ISM band. Adding 912.6 MHz as a second
    center nearly doubles capture rate for FHSS meters AND
    gives 100% R900 coverage (R900s transmit at 912.38/912.6 MHz,
    previously outside our window entirely)."""

    def test_both_passes_exist(self):
        from rfcensus.config.loader import _load_builtin_bands
        ids = {b.id for b in _load_builtin_bands("US")}
        assert "915_ism" in ids, "primary 915 MHz pass band missing"
        assert "915_ism_r900" in ids, (
            "R900 pass band (915_ism_r900) missing — the primary "
            "915_ism pass misses R900 meters entirely because they "
            "transmit at 912.38/912.6 MHz, outside the ±1.2 MHz "
            "demod window at -centerfreq=915000000."
        )

    def test_r900_pass_center_is_912_6_mhz(self):
        """Must be exactly 912.6 MHz to match R900 transmit freq."""
        from rfcensus.config.loader import _load_builtin_bands
        bands = {b.id: b for b in _load_builtin_bands("US")}
        r900 = bands["915_ism_r900"]
        assert r900.center_hz == 912_600_000, (
            f"915_ism_r900 center is {r900.center_hz:,} Hz, must be "
            f"912,600,000 Hz (R900 transmit freq). freq_low and "
            f"freq_high in bands_us.toml should average to 912.6 MHz."
        )

    def test_r900_pass_window_matches_rtlamr_demod_bw(self):
        """rtlamr demodulates ±1.2 MHz around -centerfreq at the
        default 2.4 Msps. The band's bandwidth should match this
        so the report accurately reflects what's being covered."""
        from rfcensus.config.loader import _load_builtin_bands
        bands = {b.id: b for b in _load_builtin_bands("US")}
        r900 = bands["915_ism_r900"]
        # Expect 2.4 MHz ± some tolerance (even numbers are hard
        # to hit exactly with discrete config)
        assert 2_300_000 <= r900.bandwidth_hz <= 2_500_000, (
            f"915_ism_r900 bandwidth is {r900.bandwidth_hz:,} Hz; "
            f"expected ~2.4 MHz to match rtlamr's demod window."
        )

    def test_r900_pass_does_not_duplicate_power_scan(self):
        """The primary 915_ism band power-scans 902-928 MHz. The
        R900 pass must NOT also power-scan its narrow window —
        that's redundant work and reports two conflicting
        occupancy maps for the same spectrum."""
        from rfcensus.config.loader import _load_builtin_bands
        from rfcensus.config.schema import StrategyKind
        bands = {b.id: b for b in _load_builtin_bands("US")}
        r900 = bands["915_ism_r900"]
        assert r900.strategy == StrategyKind.DECODER_ONLY, (
            f"915_ism_r900.strategy is {r900.strategy}; must be "
            f"DECODER_ONLY so it doesn't re-run rtl_power over a "
            f"window already covered by 915_ism's primary pass."
        )

    def test_r900_pass_windows_overlap_with_primary_at_boundary(self):
        """The two passes should meet at 913.8 MHz with no gap.
        If they were misaligned (e.g., 915_ism_r900 ended at
        913.7), meters hopping exactly to 913.75 would fall in
        a blind spot between the two passes."""
        from rfcensus.config.loader import _load_builtin_bands
        bands = {b.id: b for b in _load_builtin_bands("US")}
        r900 = bands["915_ism_r900"]
        # rtlamr at 915 MHz center covers 913.8-916.2. r900 pass
        # should end at or beyond 913.8 for seamless coverage.
        assert r900.freq_high >= 913_800_000, (
            f"915_ism_r900 upper edge is {r900.freq_high:,} Hz; "
            f"must be ≥913.8 MHz to meet the primary 915 MHz pass "
            f"(which covers 913.8-916.2 MHz). Otherwise a gap "
            f"forms between the two windows."
        )

    def test_r900_pass_uses_rtlamr_decoder(self):
        """rtlamr is the only decoder that knows R900 protocol.
        rtl_433 is on the primary pass for the ISM burst zoo
        (TPMS, weather stations, doorbells) but doesn't decode
        R900 — no point running it on this narrow pass."""
        from rfcensus.config.loader import _load_builtin_bands
        bands = {b.id: b for b in _load_builtin_bands("US")}
        r900 = bands["915_ism_r900"]
        assert "rtlamr" in r900.suggested_decoders, (
            f"915_ism_r900 must suggest rtlamr; got "
            f"{r900.suggested_decoders}"
        )
