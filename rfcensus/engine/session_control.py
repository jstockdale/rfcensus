"""Session control object — pause / resume / stop signalling.

Why this exists
---------------

The TUI dashboard's `p` hotkey needs to pause an active scanning session.
"Pause" has tiered semantics:

  • Quick pause (0-20s): stop feeding IQ chunks from the fanout to
    downstream queues. Decoders block on their queue.get() naturally.
    Fanouts and decoder processes stay alive — zero restart cost on
    resume.

  • Deep pause (>20s): tear down decoder processes and downstream
    fanout clients. Keep the rtl_tcp upstream + lease alive so we
    still hold the dongles. On resume, restart decoders. Lower idle
    CPU footprint than quick pause, at the cost of ~1-5s decoder
    warmup on resume.

  • Resume after crash: if quick pause's resume_writes() probe finds a
    fanout client has died (decoder process exited, socket broke during
    pause), fall through to the deep-pause restart path immediately
    instead of waiting for the next wave.

Shape
-----

A `SessionControl` is a small bag of asyncio.Event flags + monotonic
timestamps. It's passed by reference into the wave loop, the strategy
context, and the fanout, so each layer can check pause state at the
appropriate granularity:

  • Wave loop: between waves, before launching the next set of tasks
  • Fanout: in the per-client write path, drop chunks while paused
  • Strategy/decoder: optional, if a strategy wants to react during
    a long-running task

Duration accounting
-------------------

Paused wall-clock time doesn't count against `--duration`. Each pause
captures the monotonic time at pause-start; resume adds the elapsed
delta to a running `total_paused_s` counter. The session's main loop
checks `effective_remaining_s = duration_s - (now - start - paused)`
instead of the raw `now - start`.

This lives in its own module (rather than as a session-internal
attribute) because the fanout, strategy, and TUI all need a shared
handle, and bouncing it through three layers' constructors is cleaner
than reaching into session._control everywhere.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


# Threshold between quick-pause and deep-pause behavior. Below this,
# we keep decoders alive and just block fanout writes. Above this, we
# tear them down to release CPU. 20s matches the discussed UX target:
# "I tabbed away briefly" stays cheap, "I walked away" releases load.
DEEP_PAUSE_THRESHOLD_S = 20.0


@dataclass
class SessionControl:
    """Shared pause/resume/stop state for a session.

    All flags are asyncio.Event so any consumer can `await` them
    directly without polling. Monotonic timestamps use time.monotonic()
    so they're insulated from wall-clock skew (NTP correction etc.)
    that would corrupt elapsed-time math.

    Lifecycle:
      • created at session start, passed to scheduler/fanout/strategy
      • pause() called when user requests pause
      • resume() called when user requests resume
      • stop() called for graceful shutdown
      • finalize() called at session end to cap the paused-time tally
    """

    # Lifecycle events. `running` is set while the session is actively
    # processing waves; `paused` is set while the session is paused;
    # `stopped` is set when teardown has been requested. `paused` and
    # `stopped` are mutually exclusive in practice but neither implies
    # the other — a paused session can still be stopped, in which case
    # both flags are set briefly during teardown.
    paused: asyncio.Event = field(default_factory=asyncio.Event)
    stopped: asyncio.Event = field(default_factory=asyncio.Event)

    # Set when transitioning from paused → running, so a coroutine
    # waiting on `wait_not_paused()` wakes up. Cleared on subsequent
    # pause. NOT the inverse of `paused` — we want a one-shot wakeup,
    # not a level-triggered "is running" flag.
    resume_signal: asyncio.Event = field(default_factory=asyncio.Event)

    # Monotonic timestamps for duration accounting.
    pause_started_at: float | None = None  # set when pause() called
    total_paused_s: float = 0.0             # accumulated pause time

    # When True, the next pause should escalate to deep-pause behavior
    # the moment it crosses DEEP_PAUSE_THRESHOLD_S (set automatically
    # by the deep-pause timer). Exposed so the wave loop can check it.
    deep_pause_active: bool = False

    # ── Public API ──────────────────────────────────────────────────

    async def pause(self) -> None:
        """Request pause. Idempotent; already-paused sessions are a no-op."""
        if self.paused.is_set():
            return
        self.pause_started_at = time.monotonic()
        self.paused.set()
        # Clear resume_signal so any prior "spurious" resume doesn't
        # immediately wake new wait_not_paused() callers.
        self.resume_signal.clear()
        log.info("session pause requested")

    async def resume(self) -> None:
        """Resume from pause. Adds the pause duration to total_paused_s
        and signals waiters."""
        if not self.paused.is_set():
            return
        if self.pause_started_at is not None:
            elapsed = time.monotonic() - self.pause_started_at
            self.total_paused_s += elapsed
            log.info(
                "session resumed after %.1fs (total paused: %.1fs)",
                elapsed, self.total_paused_s,
            )
            self.pause_started_at = None
        self.deep_pause_active = False
        self.paused.clear()
        # Pulse the resume signal — any wait_not_paused() callers
        # wake up. We then re-clear it so future pauses can re-pulse.
        self.resume_signal.set()
        # Don't clear immediately — give all waiters time to wake.
        # The next pause() call clears it.

    async def stop(self) -> None:
        """Request graceful shutdown. Wakes any pause-waiters so they
        can exit cleanly rather than blocking forever."""
        self.stopped.set()
        # Pulse resume so paused waiters wake and see stopped flag
        self.resume_signal.set()

    async def wait_not_paused(self) -> None:
        """Block until the session is not paused (or is stopping).

        Returns immediately if not paused. While paused, blocks on
        the resume_signal Event. After waking, re-checks the paused
        flag in case of a spurious wakeup.

        If `stopped` is set, returns immediately regardless of pause
        state — callers should check `stopped.is_set()` after this
        returns to know whether to continue work or tear down.
        """
        while True:
            if self.stopped.is_set():
                return
            if not self.paused.is_set():
                return
            await self.resume_signal.wait()
            # Loop and re-check — handles the case where resume() and
            # then a fresh pause() were called between wait() and
            # this check. We want to block again in that case.

    def is_paused(self) -> bool:
        """Cheap synchronous check. Useful in hot paths like the
        fanout's per-client write loop, where awaiting the Event
        every chunk would add latency."""
        return self.paused.is_set()

    def current_pause_duration_s(self) -> float:
        """Seconds since pause was requested. Returns 0.0 if not
        currently paused. Used by the deep-pause timer to decide
        when to escalate."""
        if not self.paused.is_set() or self.pause_started_at is None:
            return 0.0
        return time.monotonic() - self.pause_started_at

    def effective_total_paused_s(self) -> float:
        """Total paused time accumulated to this moment, including any
        currently-active pause. Used for `--duration` accounting so the
        session's deadline shifts correctly while paused.
        """
        if self.paused.is_set() and self.pause_started_at is not None:
            return self.total_paused_s + (
                time.monotonic() - self.pause_started_at
            )
        return self.total_paused_s

    def finalize(self) -> None:
        """Called at session end to cap any in-progress pause's
        contribution to total_paused_s. After this, total_paused_s
        is final and effective_total_paused_s == total_paused_s."""
        if self.paused.is_set() and self.pause_started_at is not None:
            self.total_paused_s += (
                time.monotonic() - self.pause_started_at
            )
            self.pause_started_at = None


async def deep_pause_watcher(
    control: SessionControl,
    on_deep_pause: callable,
    *,
    threshold_s: float = DEEP_PAUSE_THRESHOLD_S,
    poll_interval_s: float = 1.0,
) -> None:
    """Background watcher: when pause has been active for `threshold_s`,
    invoke `on_deep_pause` callback to trigger decoder teardown.

    This task runs for the lifetime of the session. It sleeps until a
    pause is requested, then wakes every `poll_interval_s` to check the
    elapsed time. Once threshold is crossed, fires `on_deep_pause` and
    sets `control.deep_pause_active`. Resets when pause clears.

    Call site:

        watcher_task = asyncio.create_task(deep_pause_watcher(
            control, on_deep_pause=lambda: do_teardown(),
        ))
        # ... session work ...
        watcher_task.cancel()
        await watcher_task  # awaits cancellation cleanly

    `on_deep_pause` may be sync or async; both are awaited if needed.
    """
    while True:
        # Wait for pause to start.
        if not control.paused.is_set():
            # Cheap polling — pause is rare so we don't need an Event
            # subscription. 1-second poll is plenty responsive.
            await asyncio.sleep(poll_interval_s)
            continue

        # Pause active — watch for threshold or unpause.
        triggered = False
        while control.paused.is_set():
            if control.stopped.is_set():
                return
            if not triggered and control.current_pause_duration_s() >= threshold_s:
                triggered = True
                control.deep_pause_active = True
                log.info(
                    "deep-pause threshold (%.0fs) crossed; triggering teardown",
                    threshold_s,
                )
                try:
                    result = on_deep_pause()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    log.exception(
                        "deep_pause callback failed (continuing watcher)",
                    )
            await asyncio.sleep(poll_interval_s)
