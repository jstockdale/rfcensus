"""rtl_tcp fanout for multi-client shared dongle access.

Osmocom's stock rtl_tcp (and the RTL-SDR Blog fork — confirmed via
their DeepWiki docs: "Single Client Support: Accepts one client
connection at a time") only supports a single IQ consumer. When
two decoders connect to the same rtl_tcp port, the second one gets
no IQ data and exits fast. This was observed in real scans: rtl_433
works fine as the sole consumer, rtlamr joins as second client and
emits 0 decodes before exiting in <1s, which the early-exit detector
then misclassifies as hardware loss.

This module inserts a pure-Python asyncio fanout between rtl_tcp
and our decoder processes:

    rtl_tcp --upstream port-→ RtlTcpFanout ←--downstream port-- decoders...

One upstream connection to rtl_tcp. N downstream client connections,
each receiving the full IQ stream as if they had connected directly
to rtl_tcp. Client commands (freq/gain tuning) are forwarded upstream.

Alternative considered: rtlmux (BSD-3-Clause C program by Stephen
Olesen, 2016) does the same thing. We write it in Python instead so
rfcensus stays pip-installable with no new C dependency. The design
is modeled on rtlmux's buffer handling; credit to that project.

PROTOCOL NOTES
--------------
rtl_tcp protocol is dead simple:
  • Server sends a 12-byte header once at connect time:
      bytes 0-3: magic "RTL0"
      bytes 4-7: tuner type (uint32 big-endian)
      bytes 8-11: tuner gain count (uint32 big-endian)
  • Server streams raw IQ samples as uint8 I/Q sample pairs
  • Client sends 5-byte commands anytime:
      byte 0: command type (uint8)
      bytes 1-4: command value (uint32 big-endian)

The fanout caches the header from upstream and replays it to each new
downstream client before relaying IQ. Commands from any client are
forwarded upstream — rfcensus's shared-slot compatibility predicate
(broker._shared_slot_compatible) ensures only decoders at compatible
frequency+sample_rate share a slot, so command forwarding is safe.

BACKPRESSURE
------------
A slow downstream client cannot stall upstream reads — that would block
ALL clients. Each client has a bounded asyncio.Queue. If the queue
fills, we drop the oldest chunk to make room for the newest, bumping
a per-client drop counter for visibility. Lost IQ samples are
preferable to session-wide stalls for survey-tool workloads.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from rfcensus.utils.logging import get_logger

if TYPE_CHECKING:
    from rfcensus.events import EventBus

log = get_logger(__name__)


# Size of IQ chunks we relay from upstream to downstream queues.
# 16 KB matches rtl_tcp's typical buffer output — larger reads would
# block more, smaller just multiplies overhead.
_RELAY_CHUNK_SIZE = 16 * 1024

# Per-client bounded queue depth. At 2.4 Msps × 2 bytes = 4.8 MB/s,
# 512 chunks × 16 KB = 8 MB represents ~1.7s of IQ. This absorbs
# jitter from CPU-heavy decoders (rtlamr's preamble search burns
# measurable CPU) without spurious drops, even when 3 high-rate
# clients (rtl_433 + rtlamr + lora_survey) share one fanout. A client
# that still falls behind at 1.7s of headroom has a real problem —
# typically either a rate mismatch (rtlamr expects 2.36 Msps but
# rtl_tcp may still be serving 2.4 Msps, so kernel-side pipe
# accumulates ~1.7% per second monotonically) or the machine is
# under-resourced for the workload. Bumped from 256 to 512 in v0.6.6
# after multi-client stress on a Pi 5 showed periodic queue fills
# coinciding with lora_survey's 64KB readexactly cycles.
_CLIENT_QUEUE_DEPTH = 512

# rtl_tcp header: 4-byte magic + 4-byte tuner type + 4-byte gain count
_RTL_TCP_HEADER_SIZE = 12

# rtl_tcp client→server command: 1-byte cmd id + 4-byte value
_RTL_TCP_COMMAND_SIZE = 5

# rtl_tcp command IDs for diagnostic logging. Knowing WHICH commands
# a client sends is often the key to explaining its exit behavior —
# e.g. rtlamr sends set_sample_rate(2392064) on startup because its
# demodulator expects that specific rate. If that command reaches
# upstream rtl_tcp but doesn't land (rtl_tcp was launched with -s and
# ignores post-hoc rate changes on some builds), rtlamr won't decode.
_CMD_NAMES = {
    0x01: "set_freq",
    0x02: "set_sample_rate",
    0x03: "set_gain_mode",      # 0=auto, 1=manual
    0x04: "set_gain",           # tenths of dB
    0x05: "set_freq_correction",
    0x06: "set_if_gain",
    0x07: "set_test_mode",
    0x08: "set_agc_mode",
    0x09: "set_direct_sampling",
    0x0a: "set_offset_tuning",
    0x0b: "set_rtl_xtal",
    0x0c: "set_tuner_xtal",
    0x0d: "set_tuner_gain_by_index",
    0x0e: "set_bias_tee",
}

# Upstream header must arrive within this window, else we give up.
# How long to wait for rtl_tcp's 12-byte device-info header after
# connecting. rtl_tcp can take several seconds to finish libusb
# initialization (especially with multiple concurrent dongles or
# when recovering from a previous session's orphan handle), during
# which the TCP port is already accepting via kernel backlog but
# the accept() loop hasn't run yet. 5s was too aggressive and
# caused spurious fanout failures in v0.5.25. 15s is generous but
# bounded — if rtl_tcp is actually dead we still time out.
_HEADER_TIMEOUT_S = 15.0


@dataclass
class _DownstreamClient:
    """State tracked per connected decoder process."""
    writer: asyncio.StreamWriter
    label: str
    queue: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=_CLIENT_QUEUE_DEPTH)
    )
    dropped_chunks: int = 0
    bytes_sent: int = 0
    commands_forwarded: int = 0
    # v0.6.8: shared-mode command filtering diagnostics.
    commands_dropped_redundant: int = 0  # exact match → no upstream write
    commands_rejected_conflict: int = 0  # mismatch → client disconnected
    disconnected: bool = False


class RtlTcpFanout:
    """Relay one rtl_tcp upstream to many downstream clients.

    Lifecycle:
        fanout = RtlTcpFanout(
            upstream_host="127.0.0.1", upstream_port=20000,
            downstream_host="127.0.0.1", downstream_port=1234,
            slot_label="fanout[rtlsdr-00000003]",
        )
        await fanout.start()       # connect upstream, bind listener
        # ...decoders connect to downstream port and stream...
        await fanout.stop()        # teardown (idempotent)
    """

    def __init__(
        self,
        upstream_host: str,
        upstream_port: int,
        downstream_host: str = "127.0.0.1",
        downstream_port: int = 0,  # 0 = OS-assigned
        slot_label: str = "fanout",
        event_bus: Optional["EventBus"] = None,
        # v0.6.8: shared-mode command filtering. The upstream rtl_tcp
        # was started by the broker with these parameters; clients
        # that try to change them would corrupt the stream for every
        # OTHER client. Setting these locks the fanout into "shared
        # filter mode": exact-match retune commands are silently
        # acknowledged (no upstream write — already at that value);
        # mismatching retune commands cause the offending client to
        # be disconnected so it fails loudly rather than getting wrong
        # data. Pass None for both to disable filtering (legacy
        # passthrough behavior, suitable for exclusive single-client
        # tests but never the broker's runtime path).
        upstream_sample_rate: Optional[int] = None,
        upstream_center_freq_hz: Optional[int] = None,
    ) -> None:
        self._upstream_host = upstream_host
        self._upstream_port = upstream_port
        self._downstream_host = downstream_host
        self._downstream_port = downstream_port
        self._slot_label = slot_label
        # v0.6.4: optional event bus for FanoutClientEvent publishing.
        # When None, the fanout works exactly as before (no events
        # published). The broker passes its bus when constructing
        # fanouts at runtime; tests that don't care about events
        # leave it None.
        self._event_bus = event_bus

        # v0.6.8: locked upstream parameters for command filtering.
        self._upstream_sample_rate = upstream_sample_rate
        self._upstream_center_freq_hz = upstream_center_freq_hz

        # v0.6.5: paused-writes flag. When True, the relay loop drops
        # incoming chunks instead of distributing them to downstream
        # client queues. Decoders block on their queue.get() until the
        # flag clears. Toggled via pause_writes() / resume_writes().
        self._writes_paused: bool = False
        # Counter for diagnostics — how many chunks were dropped
        # while paused, since last resume.
        self._paused_drops: int = 0

        # Populated by start()
        self._upstream_reader: Optional[asyncio.StreamReader] = None
        self._upstream_writer: Optional[asyncio.StreamWriter] = None
        self._dongle_info_header: Optional[bytes] = None
        self._server: Optional[asyncio.base_events.Server] = None
        self._relay_task: Optional[asyncio.Task] = None
        self._clients: list[_DownstreamClient] = []
        self._stopped = False
        # v0.6.6: detached fire-and-forget tasks (currently just slow-
        # client event publishes from the relay loop). Holding strong
        # refs prevents Python from GC'ing pending tasks and emitting
        # the "Task was destroyed but it is pending!" warning. The
        # done-callback drops each task on completion so the set
        # doesn't grow without bound.
        self._bg_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    @property
    def downstream_port(self) -> int:
        """Actual port downstream clients connect to. 0 before start().
        If the constructor was given a specific port, this mirrors it;
        if given 0 (OS-assigned), this returns the bound port after
        start()."""
        if self._server is None:
            return 0
        sockets = self._server.sockets
        if not sockets:
            return 0
        return sockets[0].getsockname()[1]

    @property
    def client_count(self) -> int:
        """How many clients are currently connected (not disconnected)."""
        return sum(1 for c in self._clients if not c.disconnected)

    @property
    def writes_paused(self) -> bool:
        """True if pause_writes() was called and resume_writes() hasn't
        been called yet. Used by the TUI dashboard to show paused state
        and by the wave loop's quick-pause check."""
        return self._writes_paused

    @property
    def paused_drops(self) -> int:
        """Count of chunks dropped while paused since last resume.
        Reset to 0 on resume_writes(). Useful for diagnostics — a high
        drop count over a long pause confirms upstream IQ kept flowing
        (so resume should be quick)."""
        return self._paused_drops

    def pause_writes(self) -> None:
        """v0.6.5 quick-pause primitive: stop distributing IQ chunks
        from upstream to downstream client queues.

        Effect: the relay loop continues reading from rtl_tcp upstream
        (so the dongle keeps producing samples and the TCP buffer
        doesn't back up), but each chunk is dropped instead of being
        queued for clients. Decoder processes connected to downstream
        ports block on their socket read with no data flowing —
        natural pause without killing them.

        Idempotent: calling twice is fine.

        Pairs with `resume_writes()`. The session-level pause/resume
        coordinator (SessionControl) calls these via the broker.
        """
        if not self._writes_paused:
            log.info("[%s] pausing writes (decoders will block)", self._slot_label)
            self._writes_paused = True
            self._paused_drops = 0

    def resume_writes(self) -> bool:
        """v0.6.5 quick-pause primitive: resume distributing IQ chunks.

        Returns True if all downstream clients are still connected
        and ready to receive (the happy quick-pause case). Returns
        False if any client died during the pause — the caller should
        then trigger the deep-pause restart path for this slot.

        We can't actually probe a TCP socket for liveness without
        sending data, but the relay's per-client state already
        records `disconnected` flags set by the writer task when its
        socket write fails. Checking those is enough: a decoder that
        died during pause will have either gracefully closed (which
        the writer task notices on its next attempt — but writer is
        blocked on queue.get() during pause, so it won't notice) or
        the kernel will deliver a RST that we'd see on the next write.
        Therefore we do an explicit zero-byte write probe per client.
        """
        if not self._writes_paused:
            return True

        all_alive = True
        for client in self._clients:
            if client.disconnected:
                continue
            try:
                # Zero-byte write doesn't send anything visible to
                # the peer but does exercise the socket. If the
                # peer has gone away, this raises BrokenPipeError
                # or similar.
                client.writer.write(b"")
                # No drain — that would block on a slow peer. We're
                # only checking for an immediate error.
            except (ConnectionResetError, BrokenPipeError, OSError):
                client.disconnected = True
                all_alive = False
                log.warning(
                    "[%s] client %s died during pause — will need "
                    "restart on resume",
                    self._slot_label, client.label,
                )

        log.info(
            "[%s] resuming writes (dropped %d chunk(s) during pause, "
            "%s)",
            self._slot_label, self._paused_drops,
            "all clients alive" if all_alive else "some clients dead",
        )
        self._writes_paused = False
        self._paused_drops = 0
        return all_alive

    def disconnect_all_clients(self) -> int:
        """v0.6.5 deep-pause primitive: close every downstream client
        connection, forcing decoders to exit on socket EOF.

        The fanout itself stays alive — listening socket open, upstream
        rtl_tcp connection open, internal state preserved. New clients
        could connect afterward (none will, in practice, because this
        is called as part of deep-pause and the strategy tasks owning
        the dead decoders will just exit). On user resume the wave
        loop's normal restart path re-allocates and re-launches.

        Returns the number of clients disconnected. Idempotent in the
        sense that already-disconnected clients are skipped.
        """
        n = 0
        for client in self._clients:
            if client.disconnected:
                continue
            client.disconnected = True
            n += 1
            with contextlib.suppress(Exception):
                client.writer.close()
            with contextlib.suppress(asyncio.QueueFull):
                # Wake-up sentinel so the writer task exits its
                # blocked queue.get() and the per-client handler
                # finishes its disconnect cleanup.
                client.queue.put_nowait(b"")
        if n > 0:
            log.info(
                "[%s] disconnected %d downstream client(s) for deep-pause "
                "teardown (upstream + listener stay alive)",
                self._slot_label, n,
            )
        return n

    async def start(self) -> None:
        """Connect upstream, cache the 12-byte header, bind listener.

        Raises an exception (caller's choice — ConnectionRefusedError,
        OSError, asyncio.IncompleteReadError, asyncio.TimeoutError,
        or RuntimeError) if upstream isn't reachable or doesn't send
        the header. Callers should treat any exception as "fanout
        not usable; tear down the rtl_tcp subprocess that was going
        to feed it."
        """
        # 1. Open upstream connection
        self._upstream_reader, self._upstream_writer = (
            await asyncio.open_connection(
                self._upstream_host, self._upstream_port,
            )
        )

        # 2. Read the 12-byte device-info header within a timeout.
        # Every downstream client sees a copy of this, so decoders
        # see exactly the same protocol preamble they'd see
        # connecting to rtl_tcp directly.
        try:
            self._dongle_info_header = await asyncio.wait_for(
                self._upstream_reader.readexactly(_RTL_TCP_HEADER_SIZE),
                timeout=_HEADER_TIMEOUT_S,
            )
        except (asyncio.TimeoutError, asyncio.IncompleteReadError) as e:
            await self._close_upstream_quietly()
            # Failure mode: TCP connected (so rtl_tcp is at least
            # bound to the port), but it never sent the device-info
            # header. Usually means rtl_tcp is stuck in libusb
            # initialization — the kernel accepted our connect into
            # its socket backlog, but rtl_tcp hasn't reached its
            # accept() loop because it's still trying to claim the
            # USB device. Common triggers: orphan rtl_tcp from a
            # previous session still holding the device, USB hub
            # contention when several dongles power up concurrently,
            # or a dongle that's physically stuck and needs a replug.
            raise RuntimeError(
                f"[{self._slot_label}] upstream {self._upstream_host}:"
                f"{self._upstream_port} did not send device-info header "
                f"within {_HEADER_TIMEOUT_S}s. Likely causes: rtl_tcp "
                f"stuck in libusb init (check the rtl_tcp[...] stderr "
                f"lines above for 'usb_claim_interface' / 'Kernel driver "
                f"is active' / similar), orphan rtl_tcp from previous "
                f"session, or USB re-enumeration race. Try `lsof | grep "
                f"rtlsdr` or unplug/replug the dongle. Original error: {e}"
            ) from e

        log.debug(
            "[%s] upstream header received: magic=%r",
            self._slot_label, self._dongle_info_header[:4],
        )

        # 3. Bind the downstream listener
        try:
            self._server = await asyncio.start_server(
                self._handle_client,
                host=self._downstream_host,
                port=self._downstream_port,
            )
        except OSError:
            await self._close_upstream_quietly()
            raise

        # 4. Start the upstream → downstream relay task
        self._relay_task = asyncio.create_task(
            self._relay_upstream(),
            name=f"rtl-fanout-relay[{self._slot_label}]",
        )

        log.info(
            "[%s] rtl_tcp fanout ready: downstream %s:%d ← upstream %s:%d",
            self._slot_label,
            self._downstream_host, self.downstream_port,
            self._upstream_host, self._upstream_port,
        )

    async def stop(self) -> None:
        """Tear down all connections and the listener.

        Idempotent: calling a second time is a no-op rather than an
        error. Safe to call from cleanup/finally paths.
        """
        if self._stopped:
            return
        self._stopped = True

        # Stop accepting new clients first
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None

        # Cancel the relay task. Its finally block also closes clients,
        # but we back that up below anyway.
        if self._relay_task is not None and not self._relay_task.done():
            self._relay_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._relay_task
            self._relay_task = None

        # Disconnect any remaining clients (belt-and-suspenders)
        for client in list(self._clients):
            client.disconnected = True
            with contextlib.suppress(Exception):
                client.writer.close()
            # Wake up writer loop if blocked on queue.get()
            with contextlib.suppress(asyncio.QueueFull):
                client.queue.put_nowait(b"")

        # Close upstream
        await self._close_upstream_quietly()

        # v0.6.6: drain any in-flight slow-client event tasks. They're
        # short and we already cancelled the relay loop, so this is
        # quick — but do it before we report the stopped log so the
        # event order in the dashboard is consistent.
        if self._bg_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._bg_tasks, return_exceptions=True),
                    timeout=1.0,
                )
            except (asyncio.TimeoutError, TimeoutError):
                for t in self._bg_tasks:
                    t.cancel()

        dropped_total = sum(c.dropped_chunks for c in self._clients)
        log.info(
            "[%s] fanout stopped (clients served: %d, "
            "total dropped chunks: %d)",
            self._slot_label, len(self._clients), dropped_total,
        )

    # ------------------------------------------------------------
    # Internal: upstream reader broadcasts to all client queues
    # ------------------------------------------------------------

    async def _relay_upstream(self) -> None:
        """Read IQ chunks from upstream, fan out to all client queues.

        Runs until upstream closes, errors, or we're cancelled. On
        exit (any path), closes all client writers so downstream
        decoders see EOF cleanly rather than hanging on a silent
        socket.

        v0.6.6 hot-path performance:

          • The per-client distribution loop is fully synchronous —
            no awaits between clients. Awaits in this loop would let
            other tasks run mid-distribution, allowing one client's
            writer to fall behind another's. With multiple high-rate
            decoders (rtl_433 + rtlamr + lora_survey all at 4.5 MB/s
            on the same fanout), a single misplaced await can cascade
            into queue overflows that look like network problems.

          • Slow-client warning publish is fire-and-forget via
            asyncio.create_task. The previous `await
            self._publish_client_event` was the worst offender — even
            though bus.publish doesn't wait for handlers, it still
            yields the loop, which in a hot 3-client distribution
            became a stall vector.

          • Drop-oldest on full queue is a last-ditch measure. With a
            512-deep queue (was 256 in v0.6.5) at 16 KB chunks per
            slot we have ~8 MB / ~1.7s of buffering per client,
            enough slack for routine scheduler jitter without
            drops.
        """
        assert self._upstream_reader is not None
        try:
            while True:
                try:
                    chunk = await self._upstream_reader.read(_RELAY_CHUNK_SIZE)
                except asyncio.CancelledError:
                    raise
                except (ConnectionError, OSError) as e:
                    log.warning(
                        "[%s] upstream read error: %s", self._slot_label, e,
                    )
                    break
                if not chunk:
                    log.warning(
                        "[%s] upstream rtl_tcp closed; shutting down fanout",
                        self._slot_label,
                    )
                    break
                # v0.6.5: if quick-pause is active, drop the chunk
                # without distributing. Decoders block on their
                # queue.get() — natural pause. We continue reading
                # upstream so the dongle's TCP buffer doesn't back up
                # (which would manifest as a fanout slowdown later
                # rather than instantly resuming).
                if self._writes_paused:
                    self._paused_drops += 1
                    continue
                # Distribute to each active client. SYNCHRONOUS LOOP —
                # no awaits. See docstring for why.
                for client in self._clients:
                    if client.disconnected:
                        continue
                    try:
                        client.queue.put_nowait(chunk)
                    except asyncio.QueueFull:
                        # Drop oldest, then put. The two suppress
                        # blocks handle the impossible-but-defensive
                        # case where another coroutine drained the
                        # queue between our get_nowait and put_nowait
                        # (can't actually happen since we never await
                        # in this loop, but leaving the guards costs
                        # nothing and simplifies reasoning).
                        with contextlib.suppress(asyncio.QueueEmpty):
                            client.queue.get_nowait()
                        with contextlib.suppress(asyncio.QueueFull):
                            client.queue.put_nowait(chunk)
                        client.dropped_chunks += 1
                        if client.dropped_chunks % 100 == 1:
                            # Log every 100th drop to surface backpressure
                            # without spamming
                            log.warning(
                                "[%s] client %s slow — dropped %d chunks",
                                self._slot_label, client.label,
                                client.dropped_chunks,
                            )
                            # v0.6.6: fire-and-forget. The previous
                            # `await self._publish_client_event(...)`
                            # blocked the relay until subscribers
                            # processed the event. With multiple
                            # high-rate clients this manifested as
                            # cascading queue fills.
                            self._spawn_client_event(
                                client.label, "slow",
                                bytes_sent=client.bytes_sent,
                            )
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception(
                "[%s] relay task died unexpectedly", self._slot_label,
            )
        finally:
            # Upstream done. Close all clients so their read()s return
            # EOF and decoders exit cleanly. Without this, decoders
            # hang on a silent socket because we'd only been queueing
            # — never writing or closing. Push an empty-bytes sentinel
            # too, to wake any writer stuck on queue.get().
            for client in self._clients:
                if client.disconnected:
                    continue
                client.disconnected = True
                with contextlib.suppress(Exception):
                    client.writer.close()
                with contextlib.suppress(asyncio.QueueFull):
                    client.queue.put_nowait(b"")  # wake-up sentinel

    # ------------------------------------------------------------
    # Internal: per-client connection handling
    # ------------------------------------------------------------

    async def _publish_client_event(
        self,
        peer_addr: str,
        event_type: str,
        bytes_sent: int = 0,
    ) -> None:
        """Publish a FanoutClientEvent if a bus is configured.

        No-op when `_event_bus is None` (the default for tests and
        any caller that doesn't care about the dashboard). Awaits the
        bus.publish call, which itself dispatches handlers as
        background tasks — but the await still yields the event loop,
        so do NOT call this from the hot relay loop. Use
        `_spawn_client_event` instead.
        """
        if self._event_bus is None:
            return
        # Local import keeps the bus import out of the runtime path
        # for fanouts that never publish.
        from rfcensus.events import FanoutClientEvent
        try:
            await self._event_bus.publish(
                FanoutClientEvent(
                    slot_id=self._slot_label,
                    peer_addr=peer_addr,
                    event_type=event_type,  # type: ignore[arg-type]
                    bytes_sent=bytes_sent,
                )
            )
        except Exception:
            # Never let bus issues kill the fanout. Log and continue.
            log.exception(
                "[%s] failed to publish FanoutClientEvent",
                self._slot_label,
            )

    def _spawn_client_event(
        self,
        peer_addr: str,
        event_type: str,
        bytes_sent: int = 0,
    ) -> None:
        """v0.6.6: fire-and-forget variant for the hot relay loop.

        Schedules the publish as a detached task so the relay loop
        never yields waiting for subscribers. We track the task in
        an internal set with a done-callback to avoid the asyncio
        warning about un-awaited tasks; Python would otherwise GC
        the Task while it's pending and emit a noisy log line.
        """
        if self._event_bus is None:
            return
        try:
            task = asyncio.create_task(
                self._publish_client_event(
                    peer_addr, event_type, bytes_sent=bytes_sent,
                ),
                name=f"fanout-event[{self._slot_label}]",
            )
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        except Exception:
            # If the event loop isn't running for some reason, fall
            # back silently. The fanout's job is moving bytes, not
            # publishing telemetry.
            pass

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """One downstream client: replay header, drain queue → client,
        forward client commands → upstream. Exits when either
        direction breaks.
        """
        peer = writer.get_extra_info("peername")
        peer_str = f"{peer[0]}:{peer[1]}" if peer else "?"
        client = _DownstreamClient(writer=writer, label=peer_str)

        # Don't accept clients before we have a header to replay
        if self._dongle_info_header is None:
            log.warning(
                "[%s] client %s connected before upstream header ready",
                self._slot_label, peer_str,
            )
            with contextlib.suppress(Exception):
                writer.close()
            return

        self._clients.append(client)
        log.info(
            "[%s] client connected from %s (now %d client(s))",
            self._slot_label, peer_str, self.client_count,
        )
        # v0.6.4: announce the new client so the TUI can show fanout
        # activity in the dongle detail view.
        await self._publish_client_event(peer_str, "connect")

        # Track how long this client was connected and which subtask
        # ended first — critical for diagnosing fast-exit bugs like
        # rtlamr's 0-decodes regression. If writer ends first with
        # low bytes_sent, client closed its read side immediately
        # after header (likely decoder rejected the stream). If
        # cmd_reader ends first, client sent EOF on its write side
        # (less common for a well-behaved decoder).
        connect_t = asyncio.get_event_loop().time()
        ended_by = "unknown"

        try:
            # Replay cached header so the client sees a normal rtl_tcp
            # greeting
            writer.write(self._dongle_info_header)
            await writer.drain()

            # Run writer (queue → client) and reader (client → upstream
            # commands) concurrently. When either ends, tear down.
            writer_task = asyncio.create_task(
                self._client_writer(client),
                name=f"rtl-fanout-cw[{self._slot_label}:{peer_str}]",
            )
            cmdreader_task = asyncio.create_task(
                self._client_cmd_reader(client, reader),
                name=f"rtl-fanout-cr[{self._slot_label}:{peer_str}]",
            )

            done, pending = await asyncio.wait(
                [writer_task, cmdreader_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Record which subtask finished first
            if writer_task in done and cmdreader_task not in done:
                ended_by = "writer"  # write to client socket failed
            elif cmdreader_task in done and writer_task not in done:
                ended_by = "cmd_reader"  # client closed write side
            elif writer_task in done and cmdreader_task in done:
                ended_by = "both_simultaneously"

            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        except (ConnectionResetError, BrokenPipeError,
                asyncio.IncompleteReadError):
            ended_by = "handshake_error"
        except Exception:
            log.exception(
                "[%s] client %s handler error",
                self._slot_label, peer_str,
            )
        finally:
            client.disconnected = True
            with contextlib.suppress(Exception):
                writer.close()
            duration_s = asyncio.get_event_loop().time() - connect_t
            # If the client exited within a few seconds having received
            # very little IQ, that's almost certainly the rtlamr-style
            # fast-exit bug. Emit at WARNING so it stands out.
            level = (
                log.warning
                if duration_s < 3.0 and client.bytes_sent < 64_000
                else log.info
            )
            level(
                "[%s] client %s disconnected after %.2fs "
                "(sent=%d bytes, dropped=%d chunks, cmds_fwd=%d, "
                "cmds_absorbed=%d, cmds_rejected=%d, "
                "ended_by=%s; %d client(s) remain)",
                self._slot_label, peer_str, duration_s,
                client.bytes_sent, client.dropped_chunks,
                client.commands_forwarded,
                client.commands_dropped_redundant,
                client.commands_rejected_conflict,
                ended_by, self.client_count,
            )
            # v0.6.4: announce disconnect for the dashboard. bytes_sent
            # carries the lifetime byte count for this client so the
            # TUI can show "rtlamr disconnected after 12 MB delivered."
            await self._publish_client_event(
                peer_str, "disconnect", bytes_sent=client.bytes_sent,
            )

    async def _client_writer(self, client: _DownstreamClient) -> None:
        """Drain this client's queue into its socket.

        Exits on any of:
          • empty-bytes sentinel on the queue (pushed when upstream
            closes or stop() is called, to wake a blocked get())
          • client.disconnected flag set
          • write error (client went away)
        """
        while not client.disconnected:
            chunk = await client.queue.get()
            if not chunk or client.disconnected:
                return
            try:
                client.writer.write(chunk)
                await client.writer.drain()
            except (ConnectionResetError, BrokenPipeError,
                    asyncio.CancelledError):
                client.disconnected = True
                return
            except Exception as e:
                log.debug(
                    "[%s] write to %s failed: %s",
                    self._slot_label, client.label, e,
                )
                client.disconnected = True
                return
            client.bytes_sent += len(chunk)

    async def _client_cmd_reader(
        self,
        client: _DownstreamClient,
        reader: asyncio.StreamReader,
    ) -> None:
        """Read 5-byte rtl_tcp commands from this client; in shared
        mode (when the broker locked the upstream tuning), filter
        them so one client can't corrupt the stream for the others.

        v0.6.8 — shared-mode filtering rules:
          • set_freq / set_sample_rate matching the locked upstream
            value: silently absorbed (no upstream write). The client's
            internal state model thinks the command succeeded — fine,
            we ARE at that value.
          • set_freq / set_sample_rate NOT matching the lock: the
            client is asking us to retune the shared dongle, which
            would break every other client. We disconnect this client
            so it sees the EOF and (typically) exits with a clear
            error rather than silently receiving wrong-band data.
          • Every other command (gain, AGC, freq correction, etc.):
            forwarded as-is. These don't change the per-client view of
            the stream in incompatible ways — gain in particular is
            a tuner-wide setting and the broker already picked one.
            Forwarding them is the conservative choice (gain may need
            adjustment for sensitivity), but a future revision could
            also decide to lock these.

        Design note — "why filter rather than just drop everything?":
        we know the upstream parameters from the broker, so we COULD
        drop every tuning command from clients. The argument against:
        if a client requests a value that DIFFERS from the lock (rare,
        but the rtlamr 2,359,296 Hz quirk is real), drop-all silently
        gives them wrong-rate data — their demodulator runs with the
        wrong assumed clock and produces subtly broken decodes with no
        log line, no error, no clue. Disconnecting that client instead
        fails LOUDLY: the operator sees a WARNING explaining exactly
        what happened, and the conflicting consumer either gets
        reconfigured or the band gets reassigned. Loud failure beats
        silent wrong-data every time. Same-value commands ARE just
        absorbed though, so the common case has zero downside.

        Design note — "should we also lock gain / AGC / etc.?":
        currently the broker only sets sample_rate + center_freq at
        rtl_tcp launch (-s and -f); gain stays at rtl_tcp's default
        (AGC enabled). So clients' gain commands are the ONLY way
        gain ever gets set per-band. Once the broker grows a way to
        configure per-band gain and pass it through here, those
        commands could be filtered the same way. For now leaving them
        passthrough is the right tradeoff — last-writer-wins gain
        conflicts are rare in practice and easy to live with.

        When upstream parameters weren't passed in (legacy/test path),
        skip filtering entirely — every command is forwarded as before.
        """
        import struct

        filtering_enabled = (
            self._upstream_sample_rate is not None
            or self._upstream_center_freq_hz is not None
        )

        while not client.disconnected:
            try:
                cmd = await reader.readexactly(_RTL_TCP_COMMAND_SIZE)
            except (asyncio.IncompleteReadError, ConnectionResetError):
                return
            except Exception as e:
                log.debug(
                    "[%s] cmd read from %s failed: %s",
                    self._slot_label, client.label, e,
                )
                return

            cmd_id = cmd[0]
            cmd_value = struct.unpack(">I", cmd[1:5])[0]
            cmd_name = _CMD_NAMES.get(cmd_id, f"cmd_{cmd_id:#04x}")

            # v0.6.8: shared-mode tuning filter.
            if filtering_enabled and cmd_id in (0x01, 0x02):
                # 0x01 = set_freq, 0x02 = set_sample_rate
                locked_value = (
                    self._upstream_center_freq_hz if cmd_id == 0x01
                    else self._upstream_sample_rate
                )
                if locked_value is None:
                    # Only one of the two is locked; this command's
                    # parameter isn't. Treat as legacy passthrough.
                    pass
                elif cmd_value == locked_value:
                    # Idempotent: the client wants what we've already
                    # got. Absorb it — no upstream write. Logged at
                    # DEBUG to keep the normal log clean; an upstream
                    # write would have logged INFO.
                    client.commands_dropped_redundant += 1
                    log.debug(
                        "[%s] %s → %s(%d) absorbed "
                        "(matches locked upstream value)",
                        self._slot_label, client.label,
                        cmd_name, cmd_value,
                    )
                    continue
                else:
                    # Conflict: client is asking us to retune in a way
                    # that would break the other clients. Disconnect
                    # this client so it fails loudly. The fanout
                    # cleanup loop will mark it disconnected and the
                    # relay loop will skip it; the client's process
                    # typically exits cleanly when its rtl_tcp socket
                    # closes.
                    client.commands_rejected_conflict += 1
                    log.warning(
                        "[%s] %s requested %s(%d) but upstream is "
                        "locked at %d — disconnecting client to "
                        "prevent stream corruption for other clients. "
                        "If this is unexpected, check the consumer's "
                        "configured sample_rate/freq matches the "
                        "shared slot's.",
                        self._slot_label, client.label,
                        cmd_name, cmd_value, locked_value,
                    )
                    # Mark disconnected; the writer pump task will
                    # close the socket and the relay loop will skip.
                    client.disconnected = True
                    try:
                        client.writer.close()
                    except Exception:
                        pass
                    return

            if self._upstream_writer is None:
                return
            try:
                self._upstream_writer.write(cmd)
                await self._upstream_writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                return
            except Exception as e:
                log.warning(
                    "[%s] cmd forward upstream failed: %s",
                    self._slot_label, e,
                )
                return
            client.commands_forwarded += 1
            # Decode the command for diagnostics. Log first 5 commands
            # per client at DEBUG, and always log set_sample_rate and
            # set_freq at INFO — those are the commands that meaningfully
            # change the shared stream and often explain cross-client
            # compatibility surprises (e.g. rtlamr's 2392064 Hz quirk).
            if cmd_id in (0x01, 0x02):
                log.info(
                    "[%s] %s → upstream: %s(%d)",
                    self._slot_label, client.label, cmd_name, cmd_value,
                )
            elif client.commands_forwarded <= 5:
                log.debug(
                    "[%s] %s → upstream: %s(%d)",
                    self._slot_label, client.label, cmd_name, cmd_value,
                )

    # ------------------------------------------------------------
    # Internal: cleanup helpers
    # ------------------------------------------------------------

    async def _close_upstream_quietly(self) -> None:
        if self._upstream_writer is not None:
            with contextlib.suppress(Exception):
                self._upstream_writer.close()
                await self._upstream_writer.wait_closed()
            self._upstream_writer = None
            self._upstream_reader = None
