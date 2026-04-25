"""v0.6.8 — shared-mode rtl_tcp command filtering.

The broker spawns rtl_tcp at a specific sample_rate + center_freq and
that lock must hold for every client sharing the slot. Without
filtering, one client (e.g. rtlamr's 2,359,296 Hz quirk) can retune
the upstream and corrupt the IQ stream for every other client.
v0.6.8 wires upstream params through to RtlTcpFanout and:

  • Absorbs set_freq / set_sample_rate that match the lock (no
    upstream write — the client's state machine thinks the command
    succeeded, which is correct since we ARE at that value)
  • Disconnects clients whose set_freq / set_sample_rate conflicts
    with the lock (loud failure beats silent wrong-data)
  • Passes other commands (gain, AGC, etc.) through as before
"""

from __future__ import annotations

import asyncio
import struct

import pytest

from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout, _DownstreamClient


def _cmd(cmd_id: int, value: int) -> bytes:
    """Build a 5-byte rtl_tcp command (id + big-endian uint32)."""
    return bytes([cmd_id]) + struct.pack(">I", value)


SET_FREQ = 0x01
SET_SAMPLE_RATE = 0x02
SET_GAIN = 0x04


class _FakeWriter:
    """Async-compatible writer collecting every command we 'forward
    upstream'. Drain is a no-op."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


class _FakeReader:
    """Yield queued bytes via readexactly. Test pre-populates with cmd
    bytes; readexactly returns them in order."""

    def __init__(self, queued: bytes = b"") -> None:
        self._buf = bytearray(queued)
        self._closed = False

    def feed(self, data: bytes) -> None:
        self._buf.extend(data)

    def close(self) -> None:
        self._closed = True

    async def readexactly(self, n: int) -> bytes:
        # Loop until we have enough bytes or the source is closed.
        while len(self._buf) < n:
            if self._closed:
                raise asyncio.IncompleteReadError(bytes(self._buf), n)
            await asyncio.sleep(0)
        result = bytes(self._buf[:n])
        del self._buf[:n]
        return result


def _make_fanout(*, lock_rate: int | None, lock_freq: int | None) -> RtlTcpFanout:
    """Build a fanout in test mode — no real network, just object
    state. Inject a fake upstream writer so command forwarding is
    observable."""
    f = RtlTcpFanout(
        upstream_host="127.0.0.1",
        upstream_port=0,
        slot_label="test",
        upstream_sample_rate=lock_rate,
        upstream_center_freq_hz=lock_freq,
    )
    f._upstream_writer = _FakeWriter()  # type: ignore[assignment]
    return f


def _make_client(reader: _FakeReader) -> _DownstreamClient:
    """A _DownstreamClient with a fake reader/writer attached."""
    writer = _FakeWriter()
    return _DownstreamClient(writer=writer, label="testclient")  # type: ignore[arg-type]


@pytest.mark.asyncio
class TestFilteringDisabled:
    """When neither upstream param is set, every command passes
    through (legacy/test-only behavior — production always sets both)."""

    async def test_passthrough_all_tuning_commands(self):
        f = _make_fanout(lock_rate=None, lock_freq=None)
        reader = _FakeReader(_cmd(SET_FREQ, 915_000_000) + _cmd(SET_SAMPLE_RATE, 2_400_000))
        reader.close()
        client = _make_client(reader)

        await f._client_cmd_reader(client, reader)  # type: ignore[arg-type]

        upstream = f._upstream_writer.writes  # type: ignore[union-attr]
        assert len(upstream) == 2
        assert client.commands_forwarded == 2
        assert client.commands_dropped_redundant == 0
        assert client.commands_rejected_conflict == 0


@pytest.mark.asyncio
class TestIdempotentAbsorb:
    """When a client requests the EXACT value the upstream is locked
    to, the command is absorbed — no upstream write — and counted as
    'dropped redundant'. The client's view is unchanged (it thinks
    the command succeeded, which is true: we ARE at that value)."""

    async def test_matching_set_sample_rate_absorbed(self):
        f = _make_fanout(lock_rate=2_400_000, lock_freq=915_000_000)
        reader = _FakeReader(_cmd(SET_SAMPLE_RATE, 2_400_000))
        reader.close()
        client = _make_client(reader)

        await f._client_cmd_reader(client, reader)  # type: ignore[arg-type]

        assert f._upstream_writer.writes == []  # type: ignore[union-attr]
        assert client.commands_forwarded == 0
        assert client.commands_dropped_redundant == 1
        assert client.commands_rejected_conflict == 0
        assert not client.disconnected

    async def test_matching_set_freq_absorbed(self):
        f = _make_fanout(lock_rate=2_400_000, lock_freq=915_000_000)
        reader = _FakeReader(_cmd(SET_FREQ, 915_000_000))
        reader.close()
        client = _make_client(reader)

        await f._client_cmd_reader(client, reader)  # type: ignore[arg-type]

        assert f._upstream_writer.writes == []  # type: ignore[union-attr]
        assert client.commands_dropped_redundant == 1
        assert not client.disconnected

    async def test_matching_then_other_commands_still_forwarded(self):
        """Absorbing tuning doesn't break other commands — set_gain
        after a matching set_sample_rate still reaches upstream."""
        f = _make_fanout(lock_rate=2_400_000, lock_freq=915_000_000)
        reader = _FakeReader(
            _cmd(SET_SAMPLE_RATE, 2_400_000)  # absorbed
            + _cmd(SET_GAIN, 280)             # forwarded
        )
        reader.close()
        client = _make_client(reader)

        await f._client_cmd_reader(client, reader)  # type: ignore[arg-type]

        assert client.commands_dropped_redundant == 1
        assert client.commands_forwarded == 1
        assert len(f._upstream_writer.writes) == 1  # type: ignore[union-attr]
        assert f._upstream_writer.writes[0] == _cmd(SET_GAIN, 280)  # type: ignore[union-attr]


@pytest.mark.asyncio
class TestConflictDisconnect:
    """A mismatching set_freq / set_sample_rate disconnects the client
    so it fails LOUDLY (the alternative would be silent wrong-data:
    the client thinks it tuned, but we kept the lock — its
    demodulator runs with the wrong assumed frequency or rate)."""

    async def test_mismatching_set_sample_rate_disconnects(self):
        """The rtlamr 2,359,296 Hz quirk scenario: client requests a
        rate that differs from our 2.4 MHz lock."""
        f = _make_fanout(lock_rate=2_400_000, lock_freq=915_000_000)
        reader = _FakeReader(_cmd(SET_SAMPLE_RATE, 2_359_296))
        reader.close()
        client = _make_client(reader)

        await f._client_cmd_reader(client, reader)  # type: ignore[arg-type]

        # Upstream stream stays untouched — we did NOT corrupt other
        # clients' view by forwarding the conflicting retune.
        assert f._upstream_writer.writes == []  # type: ignore[union-attr]
        assert client.commands_rejected_conflict == 1
        assert client.commands_forwarded == 0
        assert client.disconnected
        # The client's writer should be closed so the client sees EOF
        assert client.writer.closed  # type: ignore[union-attr]

    async def test_mismatching_set_freq_disconnects(self):
        f = _make_fanout(lock_rate=2_400_000, lock_freq=915_000_000)
        reader = _FakeReader(_cmd(SET_FREQ, 433_920_000))
        reader.close()
        client = _make_client(reader)

        await f._client_cmd_reader(client, reader)  # type: ignore[arg-type]

        assert f._upstream_writer.writes == []  # type: ignore[union-attr]
        assert client.commands_rejected_conflict == 1
        assert client.disconnected

    async def test_disconnect_stops_processing_subsequent_commands(self):
        """Once a client conflict-disconnects, subsequent commands in
        the buffer are NOT processed (we returned out of the loop)."""
        f = _make_fanout(lock_rate=2_400_000, lock_freq=915_000_000)
        reader = _FakeReader(
            _cmd(SET_SAMPLE_RATE, 2_359_296)  # conflict → disconnect
            + _cmd(SET_GAIN, 280)             # never seen
        )
        reader.close()
        client = _make_client(reader)

        await f._client_cmd_reader(client, reader)  # type: ignore[arg-type]

        # Only the conflict command was processed; gain not forwarded.
        assert client.commands_forwarded == 0
        assert f._upstream_writer.writes == []  # type: ignore[union-attr]


@pytest.mark.asyncio
class TestNonTuningPassthrough:
    """Filtering is targeted: only set_freq (0x01) and
    set_sample_rate (0x02) are checked. Gain, AGC, freq correction
    etc. always forward (broker doesn't lock them, so client commands
    are the only way they ever get set)."""

    async def test_set_gain_forwards_under_lock(self):
        f = _make_fanout(lock_rate=2_400_000, lock_freq=915_000_000)
        reader = _FakeReader(_cmd(SET_GAIN, 280))
        reader.close()
        client = _make_client(reader)

        await f._client_cmd_reader(client, reader)  # type: ignore[arg-type]

        assert client.commands_forwarded == 1
        assert f._upstream_writer.writes == [_cmd(SET_GAIN, 280)]  # type: ignore[union-attr]

    async def test_freq_correction_forwards_under_lock(self):
        f = _make_fanout(lock_rate=2_400_000, lock_freq=915_000_000)
        # 0x05 = set_freq_correction; ppm offset adjustment
        cmd = _cmd(0x05, 50)
        reader = _FakeReader(cmd)
        reader.close()
        client = _make_client(reader)

        await f._client_cmd_reader(client, reader)  # type: ignore[arg-type]

        assert client.commands_forwarded == 1
        assert f._upstream_writer.writes == [cmd]  # type: ignore[union-attr]


@pytest.mark.asyncio
class TestPartialLock:
    """If only one of the two parameters is locked, the other still
    passes through (defensive — broker always sets both, but the
    fanout shouldn't crash if only one is provided)."""

    async def test_only_rate_locked_freq_passes_through(self):
        f = _make_fanout(lock_rate=2_400_000, lock_freq=None)
        # set_freq has no lock → forwarded; set_sample_rate matches → absorbed
        reader = _FakeReader(
            _cmd(SET_FREQ, 433_000_000)   # no lock → forward
            + _cmd(SET_SAMPLE_RATE, 2_400_000)  # match → absorb
        )
        reader.close()
        client = _make_client(reader)

        await f._client_cmd_reader(client, reader)  # type: ignore[arg-type]

        assert client.commands_forwarded == 1
        assert client.commands_dropped_redundant == 1
        assert f._upstream_writer.writes == [_cmd(SET_FREQ, 433_000_000)]  # type: ignore[union-attr]
