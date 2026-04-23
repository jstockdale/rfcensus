"""Tests for v0.5.20 — rtl_tcp fanout relay.

Osmocom's rtl_tcp only supports one client. The RTL-SDR Blog fork is
the same. Without a fanout layer, a second decoder connecting to the
shared slot gets no IQ data and exits in <1s (observed in v0.5.17-19
real-world scans with rtl_433 and rtlamr co-tenanting a 915 MHz slot).

RtlTcpFanout is a pure-Python asyncio relay that connects ONCE to
rtl_tcp upstream and broadcasts the IQ stream to N downstream clients.
Each client receives the same 12-byte header + IQ stream as if it had
connected directly to rtl_tcp.

These tests use a mock upstream server (instead of real rtl_tcp) that
emits a known header + deterministic IQ stream, and verify that
multiple downstream clients all receive the same bytes in the same
order, that errors are handled cleanly, and that lifecycle primitives
(stop, disconnect, upstream EOF) don't leak or hang.
"""

from __future__ import annotations

import asyncio
import struct

import pytest


# Magic + tuner_type (R820T=1) + tuner_gain_count (29)
RTL_TCP_HEADER = b"RTL0" + struct.pack(">II", 1, 29)
assert len(RTL_TCP_HEADER) == 12


class _MockRtlTcpServer:
    """Stand-in for real rtl_tcp.

    Sends header immediately on accept, waits for the test to signal
    release, then streams the IQ payload, then accepts client
    commands. The gate lets the test connect downstream clients
    BEFORE upstream starts relaying — otherwise the fanout reads
    bytes into an empty client list and they're dropped (which
    mirrors production behavior but makes deterministic testing
    hard).

    stop() force-closes any active connections so asyncio's
    server.wait_closed() doesn't hang on stuck handler coroutines.
    Models real rtl_tcp crashing or being killed.
    """

    def __init__(self, iq_payload: bytes):
        self.iq_payload = iq_payload
        self.server = None
        self.port = 0
        self.client_commands: list[bytes] = []
        self.release_gate = asyncio.Event()
        self._active_writers: list[asyncio.StreamWriter] = []

    async def start(self) -> None:
        self.server = await asyncio.start_server(
            self._handle, host="127.0.0.1", port=0,
        )
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        self.release_gate.set()
        for w in list(self._active_writers):
            try:
                w.close()
            except Exception:
                pass
        if self.server is not None:
            self.server.close()
            try:
                await self.server.wait_closed()
            except Exception:
                pass

    def release(self) -> None:
        """Test calls this once downstream clients are connected."""
        self.release_gate.set()

    async def _handle(self, reader, writer):
        self._active_writers.append(writer)
        try:
            writer.write(RTL_TCP_HEADER)
            await writer.drain()
            await self.release_gate.wait()
            writer.write(self.iq_payload)
            await writer.drain()
            while True:
                try:
                    cmd = await reader.readexactly(5)
                    self.client_commands.append(cmd)
                except (asyncio.IncompleteReadError, ConnectionResetError):
                    break
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            try:
                self._active_writers.remove(writer)
            except ValueError:
                pass
            try:
                writer.close()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────
# Core behavior
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRtlTcpFanoutBasic:
    async def test_single_client_receives_header_and_stream(self):
        """One client connecting through the fanout sees the same
        header + IQ bytes as it would from rtl_tcp directly."""
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout

        iq_payload = bytes(range(256)) * 32  # 8 KB of deterministic IQ
        upstream = _MockRtlTcpServer(iq_payload)
        await upstream.start()
        try:
            fanout = RtlTcpFanout(
                upstream_host="127.0.0.1",
                upstream_port=upstream.port,
                downstream_host="127.0.0.1",
                downstream_port=0,
                slot_label="test-single",
            )
            await fanout.start()
            try:
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", fanout.downstream_port,
                )
                header = await reader.readexactly(12)
                assert header == RTL_TCP_HEADER

                upstream.release()

                received = b""
                while len(received) < len(iq_payload):
                    chunk = await asyncio.wait_for(
                        reader.read(4096), timeout=5.0,
                    )
                    if not chunk:
                        break
                    received += chunk
                assert received == iq_payload

                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
            finally:
                await fanout.stop()
        finally:
            await upstream.stop()

    async def test_two_clients_both_receive_full_stream(self):
        """Two clients connected to the fanout should both receive
        the full IQ stream — the entire point of this module."""
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout

        iq_payload = bytes(range(256)) * 32
        upstream = _MockRtlTcpServer(iq_payload)
        await upstream.start()
        try:
            fanout = RtlTcpFanout(
                upstream_host="127.0.0.1", upstream_port=upstream.port,
                downstream_host="127.0.0.1", downstream_port=0,
                slot_label="test-two",
            )
            await fanout.start()
            try:
                r1, w1 = await asyncio.open_connection(
                    "127.0.0.1", fanout.downstream_port,
                )
                r2, w2 = await asyncio.open_connection(
                    "127.0.0.1", fanout.downstream_port,
                )
                h1 = await r1.readexactly(12)
                h2 = await r2.readexactly(12)
                assert h1 == RTL_TCP_HEADER
                assert h2 == RTL_TCP_HEADER

                upstream.release()

                async def read_all(rdr, expected_len):
                    received = b""
                    while len(received) < expected_len:
                        chunk = await asyncio.wait_for(
                            rdr.read(4096), timeout=5.0,
                        )
                        if not chunk:
                            break
                        received += chunk
                    return received

                b1, b2 = await asyncio.gather(
                    read_all(r1, len(iq_payload)),
                    read_all(r2, len(iq_payload)),
                )
                assert b1 == iq_payload
                assert b2 == iq_payload

                for w in (w1, w2):
                    w.close()
                    try:
                        await w.wait_closed()
                    except Exception:
                        pass
            finally:
                await fanout.stop()
        finally:
            await upstream.stop()

    async def test_client_command_forwarded_to_upstream(self):
        """When a client sends a 5-byte command, the fanout must
        forward it upstream. Example: set_freq."""
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout

        upstream = _MockRtlTcpServer(b"\x00\x01" * 100)
        await upstream.start()
        try:
            fanout = RtlTcpFanout(
                upstream_host="127.0.0.1", upstream_port=upstream.port,
                downstream_host="127.0.0.1", downstream_port=0,
                slot_label="test-cmd",
            )
            await fanout.start()
            try:
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", fanout.downstream_port,
                )
                await reader.readexactly(12)
                upstream.release()

                # Let the IQ stream partially drain so we're past the
                # header/stream phase
                await reader.readexactly(20)

                # Send a command: cmd_id=1 (set_freq), value=915000000
                cmd = bytes([1]) + struct.pack(">I", 915000000)
                writer.write(cmd)
                await writer.drain()

                # Give the fanout time to forward it upstream
                for _ in range(20):
                    if upstream.client_commands:
                        break
                    await asyncio.sleep(0.05)

                assert upstream.client_commands, (
                    "fanout did not forward client command upstream; "
                    "commands list is empty"
                )
                assert upstream.client_commands[0] == cmd

                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
            finally:
                await fanout.stop()
        finally:
            await upstream.stop()


# ──────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRtlTcpFanoutEdgeCases:
    async def test_start_raises_when_upstream_unreachable(self):
        """If rtl_tcp isn't running at the upstream port, start()
        must fail cleanly rather than hang."""
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout

        fanout = RtlTcpFanout(
            upstream_host="127.0.0.1",
            upstream_port=1,  # reserved, definitely not listening
            downstream_host="127.0.0.1",
            downstream_port=0,
            slot_label="unreachable",
        )
        with pytest.raises((ConnectionRefusedError, OSError)):
            await fanout.start()

    async def test_stop_is_idempotent(self):
        """Calling stop() more than once should be a no-op, not raise."""
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout

        upstream = _MockRtlTcpServer(b"\x00\x01" * 100)
        await upstream.start()
        try:
            fanout = RtlTcpFanout(
                upstream_host="127.0.0.1", upstream_port=upstream.port,
                downstream_host="127.0.0.1", downstream_port=0,
                slot_label="idempotent",
            )
            await fanout.start()
            await fanout.stop()
            await fanout.stop()   # second call must not raise
        finally:
            await upstream.stop()

    async def test_client_disconnect_doesnt_affect_other_clients(self):
        """If one client drops, the other should keep receiving IQ."""
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout

        iq_payload = bytes(range(256)) * 64  # 16 KB
        upstream = _MockRtlTcpServer(iq_payload)
        await upstream.start()
        try:
            fanout = RtlTcpFanout(
                upstream_host="127.0.0.1", upstream_port=upstream.port,
                downstream_host="127.0.0.1", downstream_port=0,
                slot_label="disco",
            )
            await fanout.start()
            try:
                r1, w1 = await asyncio.open_connection(
                    "127.0.0.1", fanout.downstream_port,
                )
                r2, w2 = await asyncio.open_connection(
                    "127.0.0.1", fanout.downstream_port,
                )
                await r1.readexactly(12)
                await r2.readexactly(12)

                upstream.release()

                # Client 1 drops
                w1.close()
                try:
                    await w1.wait_closed()
                except Exception:
                    pass

                # Client 2 should still get full payload
                received = b""
                while len(received) < len(iq_payload):
                    chunk = await asyncio.wait_for(
                        r2.read(4096), timeout=5.0,
                    )
                    if not chunk:
                        break
                    received += chunk
                assert received == iq_payload

                w2.close()
                try:
                    await w2.wait_closed()
                except Exception:
                    pass
            finally:
                await fanout.stop()
        finally:
            await upstream.stop()

    async def test_upstream_disconnect_closes_clients(self):
        """When rtl_tcp goes away, fanout clients should see EOF
        rather than hang on a silent socket."""
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout

        upstream = _MockRtlTcpServer(b"\x00\x01" * 100)
        await upstream.start()
        fanout = RtlTcpFanout(
            upstream_host="127.0.0.1", upstream_port=upstream.port,
            downstream_host="127.0.0.1", downstream_port=0,
            slot_label="upclose",
        )
        await fanout.start()
        try:
            r, w = await asyncio.open_connection(
                "127.0.0.1", fanout.downstream_port,
            )
            await r.readexactly(12)
            upstream.release()

            # Kill upstream
            await upstream.stop()

            # Eventually read() must return EOF rather than hang
            eof_seen = False
            deadline = asyncio.get_event_loop().time() + 5.0
            while asyncio.get_event_loop().time() < deadline:
                try:
                    chunk = await asyncio.wait_for(r.read(4096), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if not chunk:
                    eof_seen = True
                    break
            assert eof_seen, (
                "fanout client didn't see EOF after upstream died; "
                "would cause decoder to hang in production"
            )

            w.close()
            try:
                await w.wait_closed()
            except Exception:
                pass
        finally:
            await fanout.stop()

    async def test_slow_client_drops_not_stalls(self):
        """If one client can't keep up, its queue fills and the fanout
        drops oldest chunks rather than blocking the upstream broadcast.
        A fast client next to the slow one must still receive the full
        stream without being held back."""
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout

        # Large payload so the queue will actually overflow while the
        # slow client reads zero bytes. At 16 KB chunks × 32 queue
        # depth = 512 KB per-client buffer; send 2 MB to blow past it.
        iq_payload = bytes(range(256)) * 8192  # 2 MB
        upstream = _MockRtlTcpServer(iq_payload)
        await upstream.start()
        try:
            fanout = RtlTcpFanout(
                upstream_host="127.0.0.1", upstream_port=upstream.port,
                downstream_host="127.0.0.1", downstream_port=0,
                slot_label="slow",
            )
            await fanout.start()
            try:
                # Fast client
                r_fast, w_fast = await asyncio.open_connection(
                    "127.0.0.1", fanout.downstream_port,
                )
                # Slow client — connects but we intentionally don't read
                r_slow, w_slow = await asyncio.open_connection(
                    "127.0.0.1", fanout.downstream_port,
                )
                await r_fast.readexactly(12)
                await r_slow.readexactly(12)

                upstream.release()

                # Fast client drains the payload; slow client sits
                # completely idle. Fast must still complete in
                # reasonable time.
                received_fast = b""
                while len(received_fast) < len(iq_payload):
                    chunk = await asyncio.wait_for(
                        r_fast.read(65536), timeout=10.0,
                    )
                    if not chunk:
                        break
                    received_fast += chunk

                assert received_fast == iq_payload, (
                    f"fast client got {len(received_fast)}/{len(iq_payload)}; "
                    f"slow client blocked the broadcast"
                )

                # Close slow client
                w_slow.close()
                try:
                    await w_slow.wait_closed()
                except Exception:
                    pass
                w_fast.close()
                try:
                    await w_fast.wait_closed()
                except Exception:
                    pass
            finally:
                await fanout.stop()
        finally:
            await upstream.stop()

    async def test_client_count_reflects_connections(self):
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout

        upstream = _MockRtlTcpServer(b"\x00\x01" * 100)
        await upstream.start()
        fanout = RtlTcpFanout(
            upstream_host="127.0.0.1", upstream_port=upstream.port,
            downstream_host="127.0.0.1", downstream_port=0,
            slot_label="count",
        )
        await fanout.start()
        try:
            assert fanout.client_count == 0

            r1, w1 = await asyncio.open_connection(
                "127.0.0.1", fanout.downstream_port,
            )
            await r1.readexactly(12)
            for _ in range(20):
                if fanout.client_count >= 1:
                    break
                await asyncio.sleep(0.05)
            assert fanout.client_count == 1

            r2, w2 = await asyncio.open_connection(
                "127.0.0.1", fanout.downstream_port,
            )
            await r2.readexactly(12)
            for _ in range(20):
                if fanout.client_count >= 2:
                    break
                await asyncio.sleep(0.05)
            assert fanout.client_count == 2

            w1.close()
            try:
                await w1.wait_closed()
            except Exception:
                pass
            for _ in range(20):
                if fanout.client_count <= 1:
                    break
                await asyncio.sleep(0.05)
            assert fanout.client_count == 1

            w2.close()
            try:
                await w2.wait_closed()
            except Exception:
                pass
        finally:
            await fanout.stop()
            await upstream.stop()


# ──────────────────────────────────────────────────────────────────
# Broker integration
# ──────────────────────────────────────────────────────────────────


class TestSharedSlotHasFanout:
    def test_shared_slot_dataclass_has_fanout_field(self):
        """_SharedSlot must carry a fanout reference so the broker
        can tear it down alongside rtl_tcp."""
        import dataclasses
        from rfcensus.hardware.broker import _SharedSlot
        fields = {f.name for f in dataclasses.fields(_SharedSlot)}
        assert "fanout" in fields
        assert "rtl_tcp_port" in fields

    def test_broker_imports_fanout(self):
        """Broker must import RtlTcpFanout so _start_shared_slot can
        spin one up."""
        import inspect
        from rfcensus.hardware import broker
        src = inspect.getsource(broker)
        assert "RtlTcpFanout" in src
        assert "await fanout.start" in src
