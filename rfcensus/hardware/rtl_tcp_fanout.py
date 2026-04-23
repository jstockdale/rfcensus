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
from typing import Optional

from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


# Size of IQ chunks we relay from upstream to downstream queues.
# 16 KB matches rtl_tcp's typical buffer output — larger reads would
# block more, smaller just multiplies overhead.
_RELAY_CHUNK_SIZE = 16 * 1024

# Per-client bounded queue depth. At 2.4 Msps × 2 bytes = 4.8 MB/s,
# 256 chunks × 16 KB = 4 MB represents ~800 ms of IQ. This absorbs
# jitter from CPU-heavy decoders (rtlamr's preamble search burns
# measurable CPU) without spurious drops. A client that still falls
# behind at 800ms of headroom has a real problem — typically either
# a rate mismatch (rtlamr expects 2.36 Msps but rtl_tcp may still be
# serving 2.4 Msps, so kernel-side pipe accumulates ~1.7% per second
# monotonically) or the machine is under-resourced for the workload.
# Previous value 32 (~100ms) was too aggressive and caused spurious
# drops for rtlamr in v0.5.25.
_CLIENT_QUEUE_DEPTH = 256

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
    ) -> None:
        self._upstream_host = upstream_host
        self._upstream_port = upstream_port
        self._downstream_host = downstream_host
        self._downstream_port = downstream_port
        self._slot_label = slot_label

        # Populated by start()
        self._upstream_reader: Optional[asyncio.StreamReader] = None
        self._upstream_writer: Optional[asyncio.StreamWriter] = None
        self._dongle_info_header: Optional[bytes] = None
        self._server: Optional[asyncio.base_events.Server] = None
        self._relay_task: Optional[asyncio.Task] = None
        self._clients: list[_DownstreamClient] = []
        self._stopped = False

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
                # Distribute to each active client. Drop-oldest on a
                # full queue so slow clients don't stall fast ones.
                for client in self._clients:
                    if client.disconnected:
                        continue
                    try:
                        client.queue.put_nowait(chunk)
                    except asyncio.QueueFull:
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
                "ended_by=%s; %d client(s) remain)",
                self._slot_label, peer_str, duration_s,
                client.bytes_sent, client.dropped_chunks,
                client.commands_forwarded, ended_by,
                self.client_count,
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
        """Read 5-byte rtl_tcp commands from this client, forward
        upstream. Exits when the client disconnects or upstream is
        torn down.
        """
        import struct

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
            cmd_id = cmd[0]
            cmd_name = _CMD_NAMES.get(cmd_id, f"cmd_{cmd_id:#04x}")
            cmd_value = struct.unpack(">I", cmd[1:5])[0]
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
