"""Tests for v0.5.21 — rtlamr fast-exit fix + diagnostics.

Diagnosed via fanout instrumentation: rtlamr was connecting to the
shared-slot fanout, receiving the 12-byte rtl_tcp header, and exiting
within 20ms with no stderr output and no commands sent. Cause: our arg
list used `-unique true` (space-separated), but Go's flag package
treats `-unique` as a boolean flag (no value consumed) and then sees
`true` as a positional argument. flag.Parse() stops at the first
positional, so all subsequent flags (-centerfreq, -duration) were
silently ignored. rtlamr ran with default centerfreq + infinite
duration, then bailed for some other reason that we never got to
diagnose because we never saw rtlamr's expected stderr startup output.

Fix: use Go's `-flag=value` syntax for every arg so each token is
unambiguously a flag, never a stray positional.
"""

from __future__ import annotations

import pytest


class TestRtlamrArgsUseEqualsSyntax:
    """rtlamr's CLI uses Go's flag package. Every arg must use
    -flag=value form to prevent flag.Parse() from stopping early
    on a stray positional."""

    def _build_rtlamr_args(self) -> list[str]:
        """Reconstruct what the rtlamr decoder would pass to subprocess
        for a typical 915 MHz scan. Exercises the argument-construction
        code path without actually spawning the binary."""
        from rfcensus.decoders.builtin.rtlamr import RtlamrDecoder
        # Build a minimal fake spec / lease shape that the arg-builder
        # uses. We don't actually call execute() — we just want to
        # snapshot the command-line arg shape.
        # Easier: just inspect the source of execute() for the patterns.
        import inspect
        src = inspect.getsource(RtlamrDecoder)
        return src

    def test_no_space_separated_unique_flag(self):
        """`-unique` is a boolean flag in rtlamr. Using `-unique true`
        causes Go's flag.Parse() to treat 'true' as a positional and
        stop processing subsequent flags, silently dropping
        -centerfreq and -duration."""
        src = self._build_rtlamr_args()
        # Reject space-separated boolean form
        assert '"-unique", "true"' not in src, (
            "rtlamr decoder must NOT use space-separated `-unique true` "
            "form; Go's flag package treats this as `-unique` (bool) "
            "followed by positional `true`, which stops flag.Parse() "
            "and silently ignores all later flags. Use '-unique=true'."
        )
        # Accept the equals form
        assert '"-unique=true"' in src, (
            "rtlamr decoder must use '-unique=true' (equals form) so "
            "Go's flag package parses it correctly."
        )

    def test_centerfreq_uses_equals(self):
        """Same hazard for -centerfreq if any preceding bool flag was
        broken — keep the convention everywhere."""
        src = self._build_rtlamr_args()
        assert '"-centerfreq", str(' not in src, (
            "rtlamr decoder must use '-centerfreq=<hz>' equals form."
        )
        assert "-centerfreq={spec.freq_hz}" in src, (
            "rtlamr decoder must use f-string with '-centerfreq={hz}'."
        )

    def test_duration_uses_equals(self):
        """Same convention for -duration."""
        src = self._build_rtlamr_args()
        assert '"-duration", f"' not in src, (
            "rtlamr decoder must use '-duration=<dur>' equals form."
        )
        assert "-duration=" in src, (
            "rtlamr decoder must use '-duration=...' equals form."
        )

    def test_msgtype_and_format_use_equals(self):
        """All flags follow the Go '-flag=value' convention for
        consistency.

        v0.5.33 note: the msgtype value is now sourced from
        spec.decoder_options["rtlamr"]["msgtype"] with a default of
        "all" (matching v0.5.27's working behavior). See the
        TestRtlamrMsgtypeOverride class below for the plumbing tests.
        """
        src = self._build_rtlamr_args()
        assert '"-msgtype", ' not in src
        assert '"-format", "json"' not in src
        # Default restored to "all" in v0.5.33 after v0.5.31's
        # expanded list (scm,scm+,idm,netidm,r900,r900bcd) caused
        # r900bcd's Parse to double r900's per-block DSP work and
        # pushed rtlamr past the drop threshold under CPU contention.
        assert '-msgtype={msgtype}' in src
        assert 'rtlamr_opts.get("msgtype", "all")' in src
        assert '"-format=json"' in src

    def test_server_uses_equals(self):
        """The -server flag is what wires us to the fanout. Must
        be parsed."""
        src = self._build_rtlamr_args()
        assert '"-server", f"{host}:{port}"' not in src
        assert "-server={host}:{port}" in src

    def test_only_real_rtlamr_flags_used(self):
        """Regression for v0.5.29: we added `-decimation=2` thinking
        it would halve rtlamr's CPU load. `-decimation` is a valid
        flag in rtl_sdr/rtl_power but NOT in rtlamr. Go's flag.Parse
        exits 2 with "flag provided but not defined: -decimation"
        the instant it sees an unknown flag, and rtlamr never
        decodes anything.

        To prevent regressions, every `-flag` we pass to rtlamr
        must be in the known-real set. Expand this set only after
        verifying with `rtlamr -h`.
        """
        import re
        src = self._build_rtlamr_args()

        # Flags rtlamr actually defines, per `rtlamr -h` as of
        # v0.9.1 (Mar 2024). DO NOT expand without running
        # `rtlamr -h` first — Go's flag.Parse rejects unknowns
        # and exits immediately, which looks like a silent rtlamr
        # failure in our logs.
        KNOWN_RTLAMR_FLAGS = {
            # Core
            "duration", "filterid", "filtertype", "format",
            "msgtype", "samplefile", "single", "symbollength",
            "unique", "version",
            # rtltcp-specific
            "agcmode", "centerfreq", "directsampling",
            "freqcorrection", "gainbyindex", "offsettuning",
            "rtlxtalfreq", "samplerate", "server",
            "testmode", "tunergain", "tunergainmode",
            "tunerxtalfreq",
        }

        # Extract flags from strings like "-foo=bar" or "-foo" that
        # appear as complete quoted string literals in the source.
        # This catches both f-strings and plain strings.
        used_flags = set()
        for match in re.finditer(r'"-([a-zA-Z_]+)(?:=|")', src):
            used_flags.add(match.group(1))
        for match in re.finditer(r'f"-([a-zA-Z_]+)=', src):
            used_flags.add(match.group(1))

        unknown = used_flags - KNOWN_RTLAMR_FLAGS
        assert not unknown, (
            f"rtlamr decoder uses flags not in rtlamr's actual flag "
            f"set: {unknown}. Verify with `rtlamr -h` before adding "
            f"new flags — Go's flag.Parse rejects unknowns with "
            f"exit 2, making the decoder appear silently broken."
        )


class TestStrategyResultHasEndedReason:
    """v0.5.21 also fixed a separate bug: StrategyResult was missing
    the ended_reason attribute that the early-exit detector reads.
    Each completed wave threw 4× 'StrategyResult object has no
    attribute ended_reason' errors before this fix."""

    def test_ended_reason_field_exists(self):
        import dataclasses
        from rfcensus.engine.strategy import StrategyResult
        fields = {f.name for f in dataclasses.fields(StrategyResult)}
        assert "ended_reason" in fields

    def test_default_ended_reason_is_empty_string(self):
        """Default must be a string (not None) so `result.ended_reason
        not in decoder_specific_exits` works. Empty string semantically
        means 'completed normally, no specific exit reason recorded.'"""
        from rfcensus.engine.strategy import StrategyResult
        r = StrategyResult(band_id="test")
        assert r.ended_reason == ""
        assert isinstance(r.ended_reason, str)

    def test_can_be_assigned_known_reasons(self):
        """The early-exit detector compares against these well-known
        exit reasons. Verify they're settable on a fresh result."""
        from rfcensus.engine.strategy import StrategyResult
        r = StrategyResult(band_id="test")
        for reason in ("binary_missing", "rtl_tcp_not_ready",
                       "wrong_lease_type", "user_skipped",
                       "hardware_lost"):
            r.ended_reason = reason
            assert r.ended_reason == reason


class TestFanoutDiagnostics:
    """v0.5.21 added rich client-disconnect logging: connection
    duration, bytes sent, dropped chunks, commands forwarded, and
    which side of the connection ended first. This is what let us
    diagnose the rtlamr fast-exit issue."""

    def test_downstream_client_tracks_commands_forwarded(self):
        """Used by the disconnect log to surface how many commands
        a client got to send before bailing out — 0 cmds in <1s
        means the client never even tried to set up the SDR."""
        import dataclasses
        from rfcensus.hardware.rtl_tcp_fanout import _DownstreamClient
        fields = {f.name for f in dataclasses.fields(_DownstreamClient)}
        assert "commands_forwarded" in fields
        assert "bytes_sent" in fields
        assert "dropped_chunks" in fields

    def test_cmd_names_dict_covers_critical_commands(self):
        """The fanout decodes the cmd byte to a human name when
        logging. Critical commands (set_freq, set_sample_rate) must
        be named so log readers can spot them at a glance."""
        from rfcensus.hardware.rtl_tcp_fanout import _CMD_NAMES
        assert _CMD_NAMES[0x01] == "set_freq"
        assert _CMD_NAMES[0x02] == "set_sample_rate"
        assert _CMD_NAMES[0x03] == "set_gain_mode"
        assert _CMD_NAMES[0x04] == "set_gain"


# ──────────────────────────────────────────────────────────────────
# v0.5.22: stderr drain on subprocess fast-exit
# ──────────────────────────────────────────────────────────────────


class TestStderrDrainedOnFastExit:
    """When a subprocess writes to stderr and exits within
    milliseconds (rtlamr's behavior with bad flags), our stderr
    pump task gets cancelled before reading the buffered output.
    The fix: drain to natural EOF first, only cancel as fallback."""

    def test_fast_exit_subprocess_stderr_captured(self, caplog):
        """Run a tiny shell command that prints 4 lines to stderr
        and exits in <50ms. All 4 lines must show up in the log
        (proving stderr was drained to EOF before the pump task
        was torn down)."""
        import asyncio
        import logging
        from rfcensus.utils.async_subprocess import (
            ManagedProcess, ProcessConfig,
        )

        async def _run():
            proc = ManagedProcess(
                ProcessConfig(
                    name="fast-exit-test",
                    args=[
                        "sh", "-c",
                        "echo line1 >&2; echo line2 >&2; "
                        "echo line3 >&2; echo line4 >&2; exit 0",
                    ],
                    log_stderr=True,
                    stderr_log_level="WARNING",
                )
            )
            await proc.start()
            # Drain stdout (none expected) so process closes naturally
            async for _ in proc.stdout_lines():
                pass
            await proc.stop()

        with caplog.at_level(logging.WARNING):
            asyncio.run(_run())

        # All four stderr lines must appear in the log
        stderr_msgs = [
            r.getMessage() for r in caplog.records
            if "fast-exit-test[stderr]" in r.getMessage()
        ]
        for expected in ("line1", "line2", "line3", "line4"):
            assert any(expected in m for m in stderr_msgs), (
                f"stderr line {expected!r} was lost during cleanup. "
                f"All stderr msgs captured: {stderr_msgs}"
            )

    def test_already_exited_process_still_drains_stderr(self, caplog):
        """If the subprocess exits and is reaped before stop() is
        called, stop()'s early-return path must still drain stderr.
        Achieved by sleeping briefly between launch and stop()."""
        import asyncio
        import logging
        from rfcensus.utils.async_subprocess import (
            ManagedProcess, ProcessConfig,
        )

        async def _run():
            proc = ManagedProcess(
                ProcessConfig(
                    name="already-dead-test",
                    args=[
                        "sh", "-c",
                        "echo prelogged >&2; exit 0",
                    ],
                    log_stderr=True,
                    stderr_log_level="WARNING",
                )
            )
            await proc.start()
            # Wait long enough for the subprocess to definitely exit
            # AND for asyncio's child watcher to set returncode. This
            # forces stop() down the early-return path.
            for _ in range(50):
                if proc._proc and proc._proc.returncode is not None:
                    break
                await asyncio.sleep(0.05)
            assert proc._proc.returncode is not None, (
                "test setup failure: subprocess didn't exit in time"
            )
            await proc.stop()

        with caplog.at_level(logging.WARNING):
            asyncio.run(_run())

        stderr_msgs = [
            r.getMessage() for r in caplog.records
            if "already-dead-test[stderr]" in r.getMessage()
        ]
        assert any("prelogged" in m for m in stderr_msgs), (
            f"stderr from already-exited process was lost. "
            f"Captured: {stderr_msgs}"
        )


# ──────────────────────────────────────────────────────────────────
# v0.5.24: full-visibility diagnostics (launch argv at INFO,
# stderr line counter, rtlamr post-mortem classifier)
# ──────────────────────────────────────────────────────────────────


class TestSubprocessLaunchVisibility:
    def test_launch_logs_argv_at_info(self, caplog):
        """The full argv must be logged at INFO (not DEBUG) so
        operators can see what every subprocess was invoked with
        without needing -vv. Critical for diagnosing silent-exit
        bugs like rtlamr's where the only clue is 'what command
        ran'."""
        import asyncio
        import logging
        from rfcensus.utils.async_subprocess import (
            ManagedProcess, ProcessConfig,
        )

        async def _run():
            proc = ManagedProcess(
                ProcessConfig(
                    name="argv-test",
                    args=["sh", "-c", "exit 0"],
                    log_stderr=True,
                )
            )
            await proc.start()
            async for _ in proc.stdout_lines():
                pass
            await proc.stop()

        with caplog.at_level(logging.INFO):
            asyncio.run(_run())

        msgs = [r.getMessage() for r in caplog.records]
        assert any(
            "launching argv-test" in m and "exit 0" in m
            for m in msgs
        ), f"Launch log missing or at wrong level. Captured: {msgs}"

    def test_stderr_lines_logged_counter_tracks_output(self):
        """ManagedProcess.stderr_lines_logged must reflect how many
        lines the pump captured. Lets callers diagnose 'was stderr
        swallowed' vs 'subprocess was silent'."""
        import asyncio
        from rfcensus.utils.async_subprocess import (
            ManagedProcess, ProcessConfig,
        )

        async def _run() -> int:
            proc = ManagedProcess(
                ProcessConfig(
                    name="counter-test",
                    args=[
                        "sh", "-c",
                        "echo a >&2; echo b >&2; echo c >&2; exit 0",
                    ],
                    log_stderr=True,
                )
            )
            await proc.start()
            async for _ in proc.stdout_lines():
                pass
            await proc.stop()
            return proc.stderr_lines_logged

        n = asyncio.run(_run())
        assert n == 3, f"expected 3 stderr lines logged, got {n}"

    def test_silent_subprocess_has_zero_stderr_count(self):
        """A subprocess that writes nothing must leave the counter
        at 0. This is the 'rtlamr genuinely silent' case — the
        operator needs to distinguish this from 'stderr was
        swallowed by cleanup.'"""
        import asyncio
        from rfcensus.utils.async_subprocess import (
            ManagedProcess, ProcessConfig,
        )

        async def _run() -> int:
            proc = ManagedProcess(
                ProcessConfig(
                    name="silent-test",
                    args=["sh", "-c", "exit 0"],
                    log_stderr=True,
                )
            )
            await proc.start()
            async for _ in proc.stdout_lines():
                pass
            await proc.stop()
            return proc.stderr_lines_logged

        n = asyncio.run(_run())
        assert n == 0, f"expected 0 stderr lines for silent subprocess, got {n}"


# ──────────────────────────────────────────────────────────────────
# v0.5.25: binary pre-flight + removed wait_for_tcp_ready ghost client
# ──────────────────────────────────────────────────────────────────


class TestBinaryPreflight:
    """Decoders must warn loudly when their binary is missing AND
    the wait_for_tcp_ready probe (which created ghost fanout
    connections) must be gone from rtlamr."""

    def test_rtlamr_decoder_logs_binary_missing_at_warning(
        self, caplog, monkeypatch,
    ):
        """Behavior test: when rtlamr's binary is missing, the
        decoder emits a WARNING-level log line before returning
        ended_reason='binary_missing'.

        Why a behavior test and not a source grep: the old test
        checked `"NOT INSTALLED" in src`, which locks the exact
        warning wording. Rewriting the message (e.g. to add a
        hint about `apt install rtl-sdr` on Debian) would break
        the test without affecting behavior. caplog-based testing
        is more maintainable and actually exercises the branch."""
        import asyncio
        import logging
        from rfcensus.decoders.base import (
            DecoderConfig, DecoderRunSpec,
        )
        from rfcensus.decoders.builtin.rtlamr import RtlamrDecoder

        # Point at a nonexistent binary so BinaryNotFoundError fires
        cfg = DecoderConfig(binary="/nonexistent/rtlamr_for_test")
        decoder = RtlamrDecoder(cfg)

        # We only need enough of the spec to reach the subprocess
        # launch and fail on binary-missing. A real lease is heavy
        # to construct; use a minimal stub exposing `.endpoint()`.
        class _StubLease:
            _released = False
            class _Dongle:
                id = "test-dongle"
            dongle = _Dongle()
            def endpoint(self):
                return ("127.0.0.1", 1234)

        spec = DecoderRunSpec(
            lease=_StubLease(),
            freq_hz=915_000_000,
            sample_rate=2_400_000,
            duration_s=1.0,
            event_bus=None,
            session_id=1,
            gain="auto",
        )
        with caplog.at_level(logging.WARNING, logger="rfcensus"):
            result = asyncio.run(decoder.run(spec))

        assert result.ended_reason == "binary_missing"
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "rtlamr" in r.getMessage().lower()
        ]
        assert warnings, (
            "rtlamr decoder must emit at least one WARNING when its "
            "binary is missing. caplog captured no matching warnings."
        )

    def test_rtlamr_no_longer_probes_fanout_with_wait_for_tcp_ready(self):
        """The decoder-side wait_for_tcp_ready probe created ghost
        fanout client connections (a 0.03s connect+close that was
        indistinguishable from a real rtlamr fast-exit in logs).
        The broker now guarantees fanout readiness before returning
        a lease, so the probe is redundant. Use the comment-aware
        check so explanatory comments about why we removed it don't
        false-positive."""
        import inspect
        from rfcensus.decoders.builtin.rtlamr import RtlamrDecoder
        src = inspect.getsource(RtlamrDecoder)
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # comments are fine
            assert "wait_for_tcp_ready(" not in stripped, (
                f"rtlamr decoder still calls wait_for_tcp_ready — "
                f"this creates a ghost fanout connection. "
                f"Offending line: {line!r}"
            )


class TestSessionDecoderBinaryPreflight:
    """The session runner should pre-check every registered
    decoder's binary upfront so a missing install is surfaced at
    scan start, not an hour into a run."""

    def test_preflight_method_exists(self):
        from rfcensus.engine.session import SessionRunner
        assert hasattr(SessionRunner, "_log_decoder_binary_preflight")
        assert callable(SessionRunner._log_decoder_binary_preflight)

    def test_preflight_uses_decoder_registry(self):
        """Implementation check: preflight must consult the decoder
        registry so it covers every registered decoder, not a
        hardcoded list."""
        import inspect
        from rfcensus.engine.session import SessionRunner
        src = inspect.getsource(
            SessionRunner._log_decoder_binary_preflight
        )
        assert "get_registry" in src or "DecoderRegistry" in src
        assert "_binary_on_path" in src

    def test_preflight_called_during_run(self):
        """Without the call from run(), the preflight method is
        dead code."""
        import inspect
        from rfcensus.engine.session import SessionRunner
        src = inspect.getsource(SessionRunner.run)
        assert "_log_decoder_binary_preflight" in src


# ──────────────────────────────────────────────────────────────────
# v0.5.25: fallback binary resolution (~/go/bin etc.)
# ──────────────────────────────────────────────────────────────────


class TestBinaryFallbackResolution:
    """Go tools installed via `go install` land in ~/go/bin, which is
    rarely on default PATH. Instead of forcing every operator to edit
    their shell profile, we check a small list of common install dirs
    as fallback. Diagnosed when a user's rtlamr was at ~/go/bin/rtlamr
    but `which rtlamr` returned nothing."""

    def test_which_resolves_absolute_path_unchanged(self, tmp_path):
        import os
        from rfcensus.utils.async_subprocess import which

        # Create a fake executable
        fake = tmp_path / "my_tool"
        fake.write_text("#!/bin/sh\nexit 0\n")
        os.chmod(fake, 0o755)

        # Absolute path should resolve to itself
        assert which(str(fake)) == str(fake)

    def test_which_returns_none_for_missing_absolute_path(self):
        from rfcensus.utils.async_subprocess import which
        assert which("/nonexistent/path/to/tool") is None

    def test_which_falls_back_to_home_go_bin(self, tmp_path, monkeypatch):
        """The key feature: if a Go binary isn't on PATH but lives in
        ~/go/bin, we find it. This unblocks the common `go install X`
        workflow without requiring PATH edits."""
        import os
        from rfcensus.utils import async_subprocess

        # Create a fake Go binary in a simulated ~/go/bin
        fake_home = tmp_path / "fakehome"
        go_bin = fake_home / "go" / "bin"
        go_bin.mkdir(parents=True)
        fake_binary = go_bin / "my_go_tool"
        fake_binary.write_text("#!/bin/sh\nexit 0\n")
        os.chmod(fake_binary, 0o755)

        # Point the fallback dirs at our simulated ~/go/bin
        monkeypatch.setattr(
            async_subprocess,
            "_FALLBACK_BIN_DIRS",
            (str(go_bin),),
        )
        # Ensure PATH doesn't contain our fake dir — we want the
        # fallback to be what resolves it
        monkeypatch.setenv("PATH", "/usr/bin")

        resolved = async_subprocess.which("my_go_tool")
        assert resolved == str(fake_binary), (
            f"fallback resolver should have found {fake_binary}, "
            f"got {resolved}"
        )

    def test_which_prefers_path_over_fallback(
        self, tmp_path, monkeypatch,
    ):
        """If a binary exists BOTH on PATH and in a fallback dir, PATH
        wins. We don't want to silently shadow a user's deliberate
        installation choice."""
        import os
        from rfcensus.utils import async_subprocess

        # Binary on PATH
        path_dir = tmp_path / "path_bin"
        path_dir.mkdir()
        path_binary = path_dir / "my_tool"
        path_binary.write_text("#!/bin/sh\necho path\n")
        os.chmod(path_binary, 0o755)

        # Same name in fallback dir
        fallback_dir = tmp_path / "fallback_bin"
        fallback_dir.mkdir()
        fallback_binary = fallback_dir / "my_tool"
        fallback_binary.write_text("#!/bin/sh\necho fallback\n")
        os.chmod(fallback_binary, 0o755)

        monkeypatch.setattr(
            async_subprocess,
            "_FALLBACK_BIN_DIRS",
            (str(fallback_dir),),
        )
        monkeypatch.setenv("PATH", str(path_dir))

        resolved = async_subprocess.which("my_tool")
        assert resolved == str(path_binary), (
            "PATH entry should take precedence over fallback dir"
        )

    def test_launch_uses_resolved_fallback_path(
        self, tmp_path, monkeypatch, caplog,
    ):
        """End-to-end: if a binary is only in the fallback dir,
        ManagedProcess.start() must rewrite argv[0] to the absolute
        path so execvp actually finds it. Without this the fallback
        lookup would be pointless."""
        import asyncio
        import logging
        import os
        from rfcensus.utils import async_subprocess
        from rfcensus.utils.async_subprocess import (
            ManagedProcess, ProcessConfig,
        )

        fallback_dir = tmp_path / "fallback_bin"
        fallback_dir.mkdir()
        fake = fallback_dir / "fake_tool"
        fake.write_text(
            "#!/bin/sh\necho ran-from-fallback\n"
        )
        os.chmod(fake, 0o755)

        monkeypatch.setattr(
            async_subprocess,
            "_FALLBACK_BIN_DIRS",
            (str(fallback_dir),),
        )
        monkeypatch.setenv("PATH", "/usr/bin")

        async def _run() -> list[str]:
            proc = ManagedProcess(
                ProcessConfig(
                    name="fallback-test",
                    args=["fake_tool"],
                    log_stderr=True,
                )
            )
            await proc.start()
            lines = [line async for line in proc.stdout_lines()]
            await proc.stop()
            return lines

        with caplog.at_level(logging.INFO):
            output = asyncio.run(_run())

        assert "ran-from-fallback" in output, (
            f"binary didn't execute via fallback path. stdout={output}"
        )

        # The launch log should make the resolution transparent
        msgs = [r.getMessage() for r in caplog.records]
        assert any(
            "resolved" in m and "fake_tool" in m and str(fake) in m
            for m in msgs
        ), (
            "launch log should note 'resolved fake_tool → /fallback/path' "
            f"when using the fallback. Captured: {msgs}"
        )


# ──────────────────────────────────────────────────────────────────
# v0.5.26: fanout start retry, removal of wait_for_tcp_ready probe
# ──────────────────────────────────────────────────────────────────


class TestFanoutStartRetry:
    """rtl_tcp is single-client. wait_for_tcp_ready's probe burns
    that slot and blocks the fanout. Fix: don't probe, let fanout
    be the first (and only) client, retry on transient failures."""

    def test_broker_no_longer_probes_rtl_tcp_before_fanout(self):
        """The broker must NOT call wait_for_tcp_ready between spawning
        rtl_tcp and connecting the fanout. Doing so burns the single-
        client slot and caused spurious 5-second header-read timeouts."""
        import inspect
        from rfcensus.hardware.broker import DongleBroker
        src = inspect.getsource(DongleBroker._start_shared_slot)
        # The probe-then-fanout-connect pattern is the anti-pattern.
        # If we see wait_for_tcp_ready used here, it should be clearly
        # guarded/commented as intentional.
        lines = src.split("\n")
        for i, line in enumerate(lines):
            if "wait_for_tcp_ready" in line and not line.lstrip().startswith("#"):
                raise AssertionError(
                    f"_start_shared_slot line {i+1} calls wait_for_tcp_ready "
                    f"without a comment marking it intentional: {line!r}. "
                    "This burns rtl_tcp's single-client slot."
                )

    def test_broker_has_fanout_start_retry_loop(self):
        """After removing the probe, slow rtl_tcp tuner init can cause
        the fanout's first connection attempt to fail. Broker retries
        with backoff."""
        import inspect
        from rfcensus.hardware.broker import DongleBroker
        src = inspect.getsource(DongleBroker._start_shared_slot)
        assert "_FANOUT_START_RETRIES" in src, (
            "broker must define retry count for fanout.start()"
        )
        assert "for attempt in range(" in src, (
            "broker must loop retry attempts"
        )
        assert "await asyncio.sleep(" in src, (
            "broker must back off between retries"
        )


class TestFanoutQueueDepthSized:
    """rtlamr has a 1.7% rate mismatch with rtl_tcp's default 2.4 Msps.
    Tight queue (32 chunks = ~100ms) caused spurious drops. Bumped to
    256 chunks = ~800ms in v0.5.26 to absorb transient jitter."""

    def test_queue_depth_is_at_least_256(self):
        from rfcensus.hardware.rtl_tcp_fanout import _CLIENT_QUEUE_DEPTH
        assert _CLIENT_QUEUE_DEPTH >= 256, (
            f"queue depth {_CLIENT_QUEUE_DEPTH} is too tight; "
            f"rtlamr drops IQ at the v0.5.25 value of 32"
        )


# ──────────────────────────────────────────────────────────────────
# v0.5.27: preflight uses DecoderCapabilities.external_binary
# ──────────────────────────────────────────────────────────────────


class TestPreflightUsesExternalBinary:
    """Prior to v0.5.27, the preflight check was looking up
    decoders by their registration `name` (e.g. "multimon"), which
    diverges from the actual binary name ("multimon-ng") for some
    decoders. This caused false-positive "not found" warnings on
    any system where only the modern binary was installed — which
    is most of them, since `multimon` and `multimon-ng` are
    different programs (the old one is pre-2012 and lacks most
    current modes)."""

    def test_multimon_capabilities_declare_external_binary(self):
        """Regression: the DecoderCapabilities.external_binary is
        the source of truth the preflight consults."""
        from rfcensus.decoders.builtin.multimon import MultimonDecoder
        caps = MultimonDecoder.capabilities
        assert caps.name == "multimon"
        assert caps.external_binary == "multimon-ng"

    def test_all_other_decoders_have_matching_name_and_binary(self):
        """Sanity: multimon is the known outlier. If a new decoder
        gets added with a name/binary mismatch, flag it for a
        conscious choice — maybe they want that, maybe they forgot."""
        from rfcensus.decoders.builtin import (
            direwolf, rtl_433, rtl_ais, rtlamr,
        )
        for mod in (direwolf, rtl_433, rtl_ais, rtlamr):
            # Find the decoder class in each module
            decoder_cls = next(
                v for k, v in vars(mod).items()
                if isinstance(v, type)
                and k.endswith("Decoder")
                and hasattr(v, "capabilities")
            )
            caps = decoder_cls.capabilities
            assert caps.name == caps.external_binary, (
                f"{decoder_cls.__name__}: name={caps.name!r} but "
                f"external_binary={caps.external_binary!r}. If this "
                f"mismatch is intentional, update this test; "
                f"otherwise the preflight will mislead operators."
            )

    def test_preflight_resolves_external_binary_not_decoder_name(
        self, tmp_path, monkeypatch, caplog,
    ):
        """End-to-end: when `multimon-ng` is on PATH but `multimon`
        is not (extremely common on Debian/Ubuntu), the preflight
        must report multimon as PRESENT."""
        import logging
        import os
        from rfcensus.utils import async_subprocess

        # Simulate: only multimon-ng is on PATH, `multimon` is not
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        multimon_ng = fake_bin / "multimon-ng"
        multimon_ng.write_text("#!/bin/sh\nexit 0\n")
        os.chmod(multimon_ng, 0o755)
        # Also need rtl_fm and other decoders to not be there (so
        # preflight reports them missing) and we isolate to see
        # multimon's status specifically
        monkeypatch.setenv("PATH", str(fake_bin))
        monkeypatch.setattr(
            async_subprocess,
            "_FALLBACK_BIN_DIRS",
            (),  # disable fallback so the test is deterministic
        )

        # Reach into session's preflight and run it
        from rfcensus.engine import session as session_mod

        # Build a minimal session-like object with the preflight
        # method accessible. The preflight is a method, but it only
        # uses `self` to access `log` (module-level) and the registry
        # (also global). So we can call the unbound method with a
        # dummy instance.
        class _FakeSelf:
            pass

        fake = _FakeSelf()
        with caplog.at_level(logging.INFO):
            session_mod.SessionRunner._log_decoder_binary_preflight(fake)

        msgs = [r.getMessage() for r in caplog.records]
        # multimon should NOT appear in the missing list because
        # its `multimon-ng` binary IS on PATH
        missing_multimon = [
            m for m in msgs
            if "multimon" in m
            and ("not found" in m or "missing" in m.lower())
            and "binaries:" not in m  # skip the summary line
        ]
        assert not missing_multimon, (
            "preflight should recognize multimon is present via its "
            "external_binary 'multimon-ng'. Missing-multimon log "
            f"messages: {missing_multimon}"
        )


# ──────────────────────────────────────────────────────────────────
# v0.5.28: process group signaling for shell pipelines
# ──────────────────────────────────────────────────────────────────


class TestProcessGroupSignaling:
    """Diagnosed when multimon ran 6 hours past its 720s deadline:
    the `sh -c "rtl_fm | multimon-ng"` pipeline was being SIGTERMed
    at the shell wrapper only, leaving rtl_fm and multimon-ng
    orphaned to PID 1 where they held the SDR dongle hostage. Fix:
    spawn with start_new_session=True, signal the whole group."""

    def test_process_group_config_defaults_true(self):
        """Every subprocess we spawn should end up in its own
        process group by default — it's the safer choice."""
        from rfcensus.utils.async_subprocess import ProcessConfig
        cfg = ProcessConfig(name="test", args=["sh"])
        assert cfg.process_group is True

    def test_shell_pipeline_children_killed_together(self):
        """End-to-end: every process in a shell-pipeline's process
        group must be dead after stop(). Enumerate all pgid members
        via /proc before stop() and verify none remain after.

        Covers both the graceful path (children respect SIGTERM) and
        the escalation path (children ignore SIGTERM → we SIGKILL
        the group) by running the assertion twice with different
        pipeline shapes:

          1. `yes | cat > /dev/null` — children die on SIGTERM, so
             we exercise the normal path.
          2. `sh -c 'trap "" TERM; sleep 30' | cat` — inner sh
             ignores SIGTERM and would survive without killpg
             escalation. This was v0.5.28's specific regression
             case (multimon pipeline ran 6h past deadline because
             the inner processes survived our SIGTERM).
        """
        import asyncio
        import os
        from rfcensus.utils.async_subprocess import (
            ManagedProcess, ProcessConfig,
        )

        async def _run_and_verify(
            shell_cmd: str, kill_timeout_s: float,
        ) -> list[int]:
            """Spawn pipeline, enumerate pgid members, stop(),
            return PIDs that survived."""
            proc = ManagedProcess(
                ProcessConfig(
                    name=f"pipeline-test[{shell_cmd[:30]}]",
                    args=["sh", "-c", shell_cmd],
                    log_stderr=True,
                    kill_timeout_s=kill_timeout_s,
                )
            )
            await proc.start()
            await asyncio.sleep(0.3)  # let all pipeline processes spawn

            sh_pid = proc._proc.pid
            pgid = os.getpgid(sh_pid)

            children_before: list[int] = []
            try:
                for entry in os.listdir("/proc"):
                    if not entry.isdigit():
                        continue
                    try:
                        with open(f"/proc/{entry}/stat") as f:
                            fields = f.read().split()
                        if int(fields[4]) == pgid:
                            children_before.append(int(entry))
                    except (FileNotFoundError, PermissionError,
                            ProcessLookupError):
                        continue
            except FileNotFoundError:
                pytest.skip("/proc not available — Linux-only test")

            assert len(children_before) >= 2, (
                f"setup failed: pipeline should have ≥2 processes "
                f"sharing pgid {pgid}, got {children_before}"
            )

            await proc.stop()

            # Small grace window for SIGCHLD reaping after stop()
            await asyncio.sleep(0.1)

            still_alive = []
            for pid in children_before:
                try:
                    os.kill(pid, 0)
                    still_alive.append(pid)
                except (ProcessLookupError, PermissionError):
                    pass
            return still_alive

        # Case 1: normal path — children die on SIGTERM
        normal_survivors = asyncio.run(_run_and_verify(
            "yes | cat > /dev/null",
            kill_timeout_s=2.0,
        ))
        assert not normal_survivors, (
            f"normal-path pipeline children survived stop(): "
            f"{normal_survivors}"
        )

        # Case 2: escalation path — inner shell ignores SIGTERM.
        # Without killpg(SIGKILL) we'd leave it running as an orphan.
        # This is the v0.5.28 regression case: multimon pipeline
        # survived for 6 hours because we only signaled the wrapper.
        escalation_survivors = asyncio.run(_run_and_verify(
            # Outer sh pipes yes into an inner sh that traps SIGTERM.
            # The inner sh is what would survive a lazy signal
            # attempt.
            "yes | sh -c 'trap \"\" TERM; while true; do sleep 0.1; done'",
            kill_timeout_s=0.5,  # forces SIGKILL escalation quickly
        ))
        assert not escalation_survivors, (
            f"escalation-path children survived stop(): "
            f"{escalation_survivors}. Inner shell ignored SIGTERM "
            f"and we failed to escalate to SIGKILL on the process "
            f"group — the v0.5.28 regression."
        )


# ──────────────────────────────────────────────────────────────────
# v0.5.28: process group kill for shell-pipeline decoders
# ──────────────────────────────────────────────────────────────────


class TestProcessGroupKill:
    """multimon uses `sh -c "rtl_fm ... | multimon-ng ..."`. Signaling
    the sh wrapper is not enough — it exits, leaves rtl_fm and
    multimon-ng orphaned to PID 1, and they keep holding the USB
    dongle indefinitely. Observed in v0.5.27 scan: rtl_fm kept
    streaming for 5+ hours after we logged 'sending SIGKILL'."""

    def test_pgid_captured_at_spawn(self):
        """The pgid must be captured while the leader is alive, not
        lazily at kill time (by which point getpgid may raise).

        Paired with test_shell_pipeline_children_killed_together
        above — that test covers the end-to-end "children die"
        invariant; this test covers the specific fix (pgid cache)
        that made it work. Testing both prevents silent regressions
        where someone refactors away the cache but the end-to-end
        test happens to pass on a fast machine where getpgid() wins
        the race against the leader's death.
        """
        import asyncio
        from rfcensus.utils.async_subprocess import (
            ManagedProcess, ProcessConfig,
        )

        async def _run() -> int | None:
            proc = ManagedProcess(
                ProcessConfig(
                    name="pgid-test",
                    args=["sh", "-c", "sleep 0.5"],
                    process_group=True,
                )
            )
            await proc.start()
            pgid_at_spawn = proc._pgid
            assert pgid_at_spawn is not None, "pgid should be captured"
            # Under start_new_session, pgid == pid of the new leader
            assert pgid_at_spawn == proc._proc.pid
            await proc.stop()
            return pgid_at_spawn

        pgid = asyncio.run(_run())
        assert pgid > 0
