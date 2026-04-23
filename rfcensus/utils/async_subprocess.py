"""Async subprocess management.

Many of our decoder backends (rtl_433, rtlamr, multimon-ng, dump1090, ...) are
external binaries. We spawn them as subprocesses, parse their stdout line by
line, and keep them alive for the duration of a decoder session.

This module provides `ManagedProcess`, an async wrapper that handles:

• Startup with clear error messages if the binary is missing
• Line-buffered stdout reading with a timeout
• Graceful shutdown (SIGTERM, then SIGKILL after timeout)
• Automatic propagation of stderr to logs
• Restart policies for long-running processes
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


class SubprocessError(Exception):
    """Base class for subprocess-related errors."""


class BinaryNotFoundError(SubprocessError):
    """The requested binary could not be found on PATH."""


class ProcessDiedError(SubprocessError):
    """The subprocess exited unexpectedly."""


@dataclass
class ProcessConfig:
    """Configuration for a ManagedProcess."""

    name: str
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None
    # SIGTERM then wait this long before SIGKILL
    kill_timeout_s: float = 5.0
    # Include process stderr in rfcensus log output
    log_stderr: bool = True
    # Log level for stderr lines; set to DEBUG for chatty tools
    stderr_log_level: str = "INFO"
    # Run in its own process group so signals reach all children.
    # REQUIRED for shell pipelines (`sh -c "A | B"`) and any command
    # that forks subprocesses. Without this, we SIGTERM/SIGKILL only
    # the shell, leaving A and B orphaned — diagnosed via the v0.5.27
    # multimon bug where rtl_fm and multimon-ng kept running after
    # we "killed" them, holding the SDR dongle hostage for hours.
    # Default True because it's nearly always the right behavior for
    # long-lived subprocesses with timeouts.
    process_group: bool = True


class ManagedProcess:
    """An async-friendly wrapper around a subprocess.

    Typical usage::

        proc = ManagedProcess(ProcessConfig(name="rtl_433", args=["rtl_433", ...]))
        await proc.start()
        try:
            async for line in proc.stdout_lines():
                ...
        finally:
            await proc.stop()
    """

    def __init__(self, config: ProcessConfig):
        self.config = config
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        # Count of stderr lines logged so the caller can tell
        # "subprocess emitted nothing" from "we lost the output."
        self._stderr_lines_logged: int = 0
        # Captured process group ID. We record this at spawn time
        # because once the group leader exits, `os.getpgid(pid)`
        # raises ProcessLookupError — but other members of the
        # group may still be alive. See _send_signal_to_group.
        self._pgid: int | None = None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def stderr_lines_logged(self) -> int:
        """Total stderr lines the pump captured. Useful for
        diagnosing "was stderr swallowed?" vs "subprocess was
        silent" when a process fast-exits with 0 decodes."""
        return self._stderr_lines_logged

    async def start(self) -> None:
        """Launch the subprocess. Raises BinaryNotFoundError if missing."""
        if self._proc is not None:
            raise SubprocessError(f"{self.config.name} already started")

        bare_binary = self.config.args[0]
        resolved = which(bare_binary)
        if resolved is None:
            raise BinaryNotFoundError(
                f"{bare_binary} not found on PATH or in common install "
                f"directories ({', '.join(_FALLBACK_BIN_DIRS)}). "
                f"Install it, add its directory to PATH, or set the "
                f"binary path explicitly in config."
            )
        # Substitute the resolved path into argv[0]. This is essential
        # when the binary was found via fallback (e.g. ~/go/bin) —
        # otherwise execvp would look only on PATH and fail. Keep a
        # copy of the original args for logging transparency.
        launch_args = [resolved, *self.config.args[1:]]

        env = {**os.environ, **self.config.env}
        # Log the launch at INFO so the full argv is always visible
        # without requiring -vv. Diagnosing decoder exit bugs
        # (rtlamr silent-exit, flag-parse errors, etc.) needs to
        # start from the EXACT command that was run. If we had to
        # resolve via fallback, show that too so the operator knows
        # where we found the binary.
        if resolved != bare_binary:
            log.info(
                "launching %s: %s (resolved %s → %s)",
                self.config.name, " ".join(launch_args),
                bare_binary, resolved,
            )
        else:
            log.info(
                "launching %s: %s",
                self.config.name, " ".join(launch_args),
            )
        self._proc = await asyncio.create_subprocess_exec(
            *launch_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self.config.cwd,
            # Put the child in its own process group (or session) so
            # signals we send it also reach any subprocesses it forks.
            # Essential for shell pipelines like
            # `sh -c "rtl_fm ... | multimon-ng ..."` where we need to
            # kill both sides, not just the shell wrapper.
            start_new_session=self.config.process_group,
        )

        # Capture the process group ID NOW, while the leader
        # (self._proc.pid) is guaranteed alive. If we wait until
        # stop() time and the leader has exited, getpgid() raises
        # ProcessLookupError and we'd fall back to signaling only
        # the (dead) leader — leaving the real workers orphaned.
        # When start_new_session=True, the child's pgid equals its
        # pid. When False (or on Windows), we skip group signaling.
        if self.config.process_group:
            try:
                self._pgid = os.getpgid(self._proc.pid)
            except (ProcessLookupError, OSError) as e:
                # Extremely unlikely — would mean the child died
                # between fork and our first getpgid. Fall back
                # to per-process signaling.
                log.debug(
                    "%s: could not capture pgid at spawn: %s",
                    self.config.name, e,
                )
                self._pgid = None

        if self.config.log_stderr:
            self._stderr_task = asyncio.create_task(
                self._pump_stderr(), name=f"{self.config.name}-stderr"
            )

    async def _pump_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        import logging as _logging

        level = getattr(_logging, self.config.stderr_log_level.upper(), _logging.INFO)
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                log.log(level, "%s[stderr]: %s", self.config.name, line.decode("utf-8", errors="replace").rstrip())
                self._stderr_lines_logged += 1
        except asyncio.CancelledError:
            pass
        except Exception as e:
            # Defensive: don't let unexpected pump errors poison the
            # subprocess cleanup path. We've seen fast-exit edge
            # cases where readline raises weird IOErrors. Log at
            # DEBUG and let the pump die quietly.
            log.debug(
                "stderr pump for %s raised: %s",
                self.config.name, e,
            )

    async def stdout_lines(self) -> AsyncIterator[str]:
        """Yield lines of UTF-8 decoded stdout, stripped of trailing newlines."""
        if self._proc is None or self._proc.stdout is None:
            raise SubprocessError("process not started")
        while True:
            try:
                line = await self._proc.stdout.readline()
            except (BrokenPipeError, ConnectionResetError):
                break
            if not line:
                # EOF – process closed stdout
                break
            yield line.decode("utf-8", errors="replace").rstrip()

    async def stop(self) -> int | None:
        """Send SIGINT, escalate to SIGTERM, then SIGKILL.

        Returns the process return code, or None if it was never started.

        SIGINT-first matters for tools like multimon-ng that ignore SIGTERM
        mid-decode (causing our previous SIGKILL fallback to leave the
        process partially reaped). SIGINT mimics Ctrl-C and is honored by
        almost everything that runs in a terminal. Falling back to SIGTERM
        and then SIGKILL handles processes that ignore SIGINT.

        If the process has already exited (e.g. because stdout EOF told us
        we were done), we detect that and skip signalling — sending signals
        to a corpse causes asyncio to log "exit status already read" warnings.
        """
        if self._proc is None:
            return None
        if self._proc.returncode is not None:
            # Process already reaped — but we MUST still drain stderr.
            # Skipping this was the rtlamr-silent-exit bug: rtlamr
            # exits faster than we get here, returncode is set by the
            # SIGCHLD handler, we early-returned, the stderr pump
            # task got garbage-collected without flushing the pipe-
            # buffered startup banner. Result: no rtlamr stderr in
            # the log even when rtlamr printed it.
            await self._cleanup_stderr_task()
            return self._proc.returncode

        # Give asyncio's child watcher a brief moment to notice if the
        # process has already exited but returncode hasn't been updated
        # yet (common race: stdout EOF means the process is already dead,
        # but the SIGCHLD handler hasn't run by the time we get here).
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=0.1)
            await self._cleanup_stderr_task()
            return self._proc.returncode
        except (asyncio.TimeoutError, TimeoutError):
            pass  # Still running; proceed to terminate

        log.debug("Stopping %s (pid=%s) with SIGINT", self.config.name, self._proc.pid)
        self._send_signal_to_group(signal.SIGINT)

        # Short wait for SIGINT to take effect (graceful, common case)
        sigint_grace = min(1.5, self.config.kill_timeout_s)
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=sigint_grace)
            await self._cleanup_stderr_task()
            return self._proc.returncode
        except (asyncio.TimeoutError, TimeoutError):
            pass

        # SIGINT ignored — escalate to SIGTERM
        log.debug("%s ignored SIGINT, sending SIGTERM", self.config.name)
        self._send_signal_to_group(signal.SIGTERM)
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=self.config.kill_timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            log.warning("%s did not exit on SIGTERM, sending SIGKILL", self.config.name)
            self._send_signal_to_group(signal.SIGKILL)
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=2.0)
            except (asyncio.TimeoutError, TimeoutError):
                log.error("%s refused to die even after SIGKILL", self.config.name)

        # Belt-and-suspenders: send one final SIGKILL to the whole
        # process group. self._proc.wait() above only tells us that
        # the group LEADER (typically `sh` for pipeline decoders)
        # has exited — but the real workers (rtl_fm, multimon-ng)
        # may still be running, and killpg() to a leaderless group
        # still reaches remaining members. This closes the window
        # where we declared victory but rtl_fm kept holding the USB
        # device for hours (observed in v0.5.27 scan: multimon
        # pipeline kept emitting FLEX decodes 5+ hours after we
        # logged "sending SIGKILL").
        if self.config.process_group and self._pgid is not None:
            try:
                os.killpg(self._pgid, signal.SIGKILL)
            except ProcessLookupError:
                # Group is fully dead — perfect
                pass
            except OSError as e:
                log.debug(
                    "%s: final killpg sweep failed: %s",
                    self.config.name, e,
                )

        await self._cleanup_stderr_task()
        return self._proc.returncode

    def _send_signal_to_group(self, sig: int) -> None:
        """Send a signal to the entire process group if we created one,
        otherwise just to the process itself.

        For shell pipelines (`sh -c "A | B"`) created with
        start_new_session=True, signaling the group ensures A and B
        both receive the signal — not just the shell wrapper.
        Without this, `sh` exits and leaves A and B orphaned to PID 1,
        where they continue holding resources (SDR dongles, TCP ports,
        file descriptors) until manually killed.

        Falls back to signaling just the process if getpgid fails
        (process already exited) or if we didn't create a new session.
        """
        if self._proc is None:
            return
        try:
            if self.config.process_group and self._pgid is not None:
                # Signal the entire process group using the pgid we
                # captured at spawn. Works even after the group
                # leader has exited (which makes `os.getpgid(pid)`
                # raise ProcessLookupError) — critical for shell-
                # pipeline decoders like multimon where `sh` is the
                # leader but rtl_fm / multimon-ng are the real
                # workers that need killing.
                try:
                    os.killpg(self._pgid, sig)
                    return
                except ProcessLookupError:
                    # Entire group is gone — success, nothing to do
                    return
                except PermissionError as e:
                    # Shouldn't happen for our own children, but
                    # log and fall through rather than hang.
                    log.debug(
                        "%s: killpg(%d) EPERM: %s",
                        self.config.name, self._pgid, e,
                    )
                except OSError as e:
                    log.debug(
                        "%s: killpg(%d) failed: %s; falling back "
                        "to single-process signal",
                        self.config.name, self._pgid, e,
                    )
            self._proc.send_signal(sig)
        except ProcessLookupError:
            # Process already exited between our checks — normal
            pass
        except Exception as e:
            log.debug(
                "%s: failed to send signal %s: %s",
                self.config.name, sig, e,
            )

    async def _cleanup_stderr_task(self) -> None:
        """Drain the stderr-pump task, then cancel if it doesn't exit
        on its own.

        IMPORTANT: when a subprocess has exited, the stderr pipe will
        EOF on its own, and the pump loop's readline() will return
        b'' and break naturally. We MUST wait for that to happen
        before cancelling, otherwise pipe-buffered stderr lines that
        haven't been consumed by the pump yet are lost.

        This was the rtlamr-silent-exit bug: rtlamr's startup banner
        and any error message are written to stderr, but the binary
        also exits within a few ms. Our pump is in `await readline()`
        when the EOF arrives. If we cancel before the next readline
        completes, several buffered lines never get logged. Result:
        rtlamr's "main.go: error connecting" or "decode.go: ..." lines
        vanish, leaving us with no clue why it exited.

        Order of operations:
          1. Wait briefly for the pump task to finish naturally
             (it will, once it sees pipe EOF after the process exits).
          2. If it doesn't exit within a small grace window, cancel —
             but at this point there's no buffered stderr to lose
             because the wait would have caught it.

        Defensive: this method never raises. Any exception from the
        pump task or the wait/cancel machinery is swallowed at DEBUG
        level. Propagating an exception here would break proc.stop()'s
        return, which in turn would skip subsequent decoder cleanup
        (like the "rtlamr exited with returncode=..." diagnostic).
        """
        if not self._stderr_task or self._stderr_task.done():
            return

        # Phase 1: graceful drain. The pump exits on its own when the
        # OS-side stderr pipe EOFs (which happens when the subprocess
        # closes its stderr fd, normally on exit). 1.0s is generous
        # enough to handle slow flushes; in the common case it returns
        # in microseconds because the pipe has already EOFd.
        try:
            await asyncio.wait_for(self._stderr_task, timeout=1.0)
            return
        except (asyncio.TimeoutError, TimeoutError):
            pass
        except Exception as e:  # noqa: BLE001
            log.debug(
                "stderr drain for %s raised during wait: %s",
                self.config.name, e,
            )
            return

        # Phase 2: forced cancel for stuck tasks. By this point any
        # buffered stderr would have been drained, so cancellation is
        # safe. This handles edge cases like a subprocess that left a
        # forked child holding stderr open.
        self._stderr_task.cancel()
        try:
            await asyncio.wait_for(self._stderr_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, TimeoutError):
            pass
        except Exception as e:  # noqa: BLE001
            log.debug(
                "stderr drain for %s raised during cancel: %s",
                self.config.name, e,
            )

    async def wait(self) -> int:
        """Wait for the process to exit on its own; return the return code."""
        if self._proc is None:
            raise SubprocessError("process not started")
        return await self._proc.wait()

    async def __aenter__(self) -> ManagedProcess:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()


def _binary_on_path(binary: str) -> bool:
    """Return True if `binary` can be located via PATH, common fallback
    locations, or is an absolute path."""
    if os.path.sep in binary:
        return os.path.isfile(binary) and os.access(binary, os.X_OK)
    return which(binary) is not None


# Common install directories checked as fallback when a binary isn't
# on PATH. Go tools default to ~/go/bin, pip --user installs to
# ~/.local/bin, Homebrew on Apple Silicon lives in /opt/homebrew/bin.
# This handles the extremely common case of `go install ...` putting
# a tool in ~/go/bin without the user remembering to add it to PATH.
_FALLBACK_BIN_DIRS = (
    os.path.expanduser("~/go/bin"),
    os.path.expanduser("~/.local/bin"),
    "/usr/local/bin",
    "/opt/homebrew/bin",
)


def which(binary: str) -> str | None:
    """Locate `binary` in PATH, falling back to common install dirs.

    Returns the absolute path to the executable, or None if it cannot
    be found. If `binary` is an absolute path, returns it unchanged
    if executable, else None.
    """
    if os.path.sep in binary:
        return binary if (
            os.path.isfile(binary) and os.access(binary, os.X_OK)
        ) else None

    # Standard PATH first
    resolved = shutil.which(binary)
    if resolved is not None:
        return resolved

    # Fallback directories — check explicitly. Common case: Go tools
    # installed via `go install` land in ~/go/bin without PATH update.
    for bin_dir in _FALLBACK_BIN_DIRS:
        candidate = os.path.join(bin_dir, binary)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return None
