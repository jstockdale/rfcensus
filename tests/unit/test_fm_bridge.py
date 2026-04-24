"""Tests for rfcensus.tools.fm_bridge.

Strategy
========

Two layers:

1. **Unit tests for argument parsing and command packing** — cheap,
   fast, no I/O.

2. **Integration tests with a mock rtl_tcp server** — spin up an
   asyncio TCP server that sends the rtl_tcp header followed by
   synthetic IQ, run the bridge pointed at it, capture the PCM
   output, verify it's sensible (correct size, correct format,
   predictable content from a known IQ input).

The mock server exercises the full protocol handshake and streaming
without needing real hardware.
"""

from __future__ import annotations

import asyncio
import io
import struct

import numpy as np
import pytest

from rfcensus.tools import fm_bridge
from rfcensus.tools.fm_bridge import (
    RTL_TCP_HEADER_LEN,
    RTL_TCP_MAGIC,
    RtlTcpClient,
    _pack_command,
    _parse_rtl_tcp,
    build_parser,
    run_bridge,
)


# ==================================================================
# Command packing
# ==================================================================


class TestPackCommand:
    def test_set_freq(self):
        out = _pack_command(0x01, 144_390_000)
        assert len(out) == 5
        assert out[0] == 0x01
        # big-endian uint32 of the param
        assert struct.unpack(">I", out[1:])[0] == 144_390_000

    def test_set_sample_rate(self):
        out = _pack_command(0x02, 2_400_000)
        assert out[0] == 0x02
        assert struct.unpack(">I", out[1:])[0] == 2_400_000

    def test_set_gain_mode_auto(self):
        out = _pack_command(0x03, 0)
        assert out == b"\x03\x00\x00\x00\x00"

    def test_large_param_is_masked(self):
        """Negative or >32-bit ints should be masked to 32 bits."""
        out = _pack_command(0x01, -1)
        # -1 as 32-bit unsigned = 0xFFFFFFFF
        assert struct.unpack(">I", out[1:])[0] == 0xFFFFFFFF


# ==================================================================
# _parse_rtl_tcp
# ==================================================================


class TestParseRtlTcp:
    def test_valid(self):
        assert _parse_rtl_tcp("127.0.0.1:1234") == ("127.0.0.1", 1234)

    def test_hostname(self):
        assert _parse_rtl_tcp("metatron.local:1234") == ("metatron.local", 1234)

    def test_ipv6_bracketed_not_supported(self):
        """Bracketed IPv6 could be handled but we don't today — if
        someone needs it we can add. This test documents the current
        behavior so it isn't accidentally broken."""
        # "[::1]:1234" — rpartition(":") gives ("[::1]:1234"[:split], "[::1]", "1234")
        # We accept this; not elegant but not wrong.
        host, port = _parse_rtl_tcp("[::1]:1234")
        assert port == 1234

    def test_missing_port_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_rtl_tcp("127.0.0.1")

    def test_bad_port_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_rtl_tcp("127.0.0.1:notanint")

    def test_out_of_range_port_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_rtl_tcp("127.0.0.1:99999")


# ==================================================================
# Argument parser
# ==================================================================


class TestArgParser:
    def test_required_args(self):
        parser = build_parser()
        # All required
        args = parser.parse_args(
            [
                "--rtl-tcp", "127.0.0.1:1234",
                "--freq", "144390000",
                "--input-rate", "2400000",
                "--output-rate", "22050",
            ]
        )
        assert args.rtl_tcp == ("127.0.0.1", 1234)
        assert args.freq == 144_390_000
        assert args.input_rate == 2_400_000
        assert args.output_rate == 22050
        assert args.gain == "auto"
        assert args.deemphasis_us is None

    def test_gain_options(self):
        parser = build_parser()
        # Numeric gain
        args = parser.parse_args([
            "--rtl-tcp", "127.0.0.1:1234",
            "--freq", "1000000",
            "--input-rate", "2400000",
            "--output-rate", "22050",
            "--gain", "40",
        ])
        assert args.gain == "40"

    def test_missing_required_fails(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--freq", "1000"])


# ==================================================================
# Mock rtl_tcp server + end-to-end integration
# ==================================================================


class MockRtlTcpServer:
    """A minimal rtl_tcp server for integration testing.

    Sends the RTL0 header, then streams synthetic IQ bytes. After
    streaming completes, either closes the stream (write_eof) to let
    the client exit naturally, or stays open waiting for the client to
    close on its own.

    Commands from the client are captured in `received_commands` for
    inspection. The server's READ side (for client commands) stays
    open as long as possible so the client can keep sending commands
    even after the IQ stream has finished — this mirrors the rtl_fm
    real-world behavior where a dongle keeps receiving commands while
    (or after) streaming IQ.
    """

    def __init__(
        self,
        *,
        iq_bytes: bytes = b"",
        tuner_type: int = 1,
        num_gains: int = 29,
        client_timeout_s: float = 5.0,
    ):
        self.iq_bytes = iq_bytes
        self.tuner_type = tuner_type
        self.num_gains = num_gains
        self.client_timeout_s = client_timeout_s
        self.received_commands: list[tuple[int, int]] = []
        self.server: asyncio.base_events.Server | None = None
        self.port: int = 0

    async def start(self) -> None:
        self.server = await asyncio.start_server(
            self._handle_client, host="127.0.0.1", port=0
        )
        self.port = self.server.sockets[0].getsockname()[1]

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        cmd_task: asyncio.Task | None = None
        try:
            # 1. Send header
            header = (
                RTL_TCP_MAGIC
                + struct.pack(">I", self.tuner_type)
                + struct.pack(">I", self.num_gains)
            )
            writer.write(header)
            try:
                await writer.drain()
            except (ConnectionError, BrokenPipeError):
                return

            # 2. Start cmd reader BEFORE streaming so commands sent
            # during the streaming phase are captured.
            async def _cmd_reader():
                try:
                    while True:
                        try:
                            cmd_bytes = await reader.readexactly(5)
                        except (asyncio.IncompleteReadError, ConnectionError):
                            return
                        cmd_id = cmd_bytes[0]
                        param = struct.unpack(">I", cmd_bytes[1:5])[0]
                        self.received_commands.append((cmd_id, param))
                except asyncio.CancelledError:
                    return

            cmd_task = asyncio.create_task(_cmd_reader())

            # 3. Stream IQ bytes, yielding between chunks so the
            # cmd_reader task has a chance to pick up commands.
            pos = 0
            while pos < len(self.iq_bytes):
                chunk = self.iq_bytes[pos : pos + 4096]
                try:
                    writer.write(chunk)
                    await writer.drain()
                except (ConnectionError, BrokenPipeError):
                    break
                pos += len(chunk)
                # Small yield — lets cmd_reader run without slowing the
                # overall test too much.
                await asyncio.sleep(0.001)

            # 4. Done streaming. Signal EOF on the SERVER→CLIENT
            # direction so the client's read_iq_chunks loop sees the
            # end of stream and exits. We do NOT close the reader
            # side yet — the client may still be sending configure
            # commands or other traffic, and prematurely closing
            # would cause ConnectionResetError on their drain().
            try:
                if writer.can_write_eof():
                    writer.write_eof()
            except (ConnectionError, OSError, AttributeError):
                pass

            # 5. Wait for the client to close (cmd_task terminates
            # when the reader gets EOF, i.e., the client called
            # close() on its socket). Bounded timeout so a buggy
            # client doesn't hang the test.
            try:
                await asyncio.wait_for(
                    cmd_task, timeout=self.client_timeout_s
                )
            except asyncio.TimeoutError:
                pass
        finally:
            if cmd_task is not None and not cmd_task.done():
                cmd_task.cancel()
                try:
                    await cmd_task
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()


# ------------------------------------------------------------------


class TestRtlTcpClientConnection:
    @pytest.mark.asyncio
    async def test_connect_parses_header(self):
        server = MockRtlTcpServer(
            iq_bytes=b"\x80\x80" * 100, tuner_type=5, num_gains=29
        )
        await server.start()
        try:
            client = RtlTcpClient("127.0.0.1", server.port)
            await client.connect()
            assert client.tuner_type == 5
            assert client.num_gains == 29
            await client.close()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_bad_magic_raises(self):
        """If we connect to something that isn't an rtl_tcp server
        (say, an HTTP port), we should error out with a clear message
        rather than producing garbage audio."""

        async def _serve_bad_header(reader, writer):
            writer.write(b"GARB" + b"\x00" * 8)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(
            _serve_bad_header, host="127.0.0.1", port=0
        )
        port = server.sockets[0].getsockname()[1]
        try:
            client = RtlTcpClient("127.0.0.1", port)
            with pytest.raises(RuntimeError, match="bad magic"):
                await client.connect()
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_short_header_raises(self):
        """Server closes before sending full 12-byte header → clear error."""

        async def _serve_short(reader, writer):
            writer.write(b"RT")  # only 2 bytes
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(
            _serve_short, host="127.0.0.1", port=0
        )
        port = server.sockets[0].getsockname()[1]
        try:
            client = RtlTcpClient("127.0.0.1", port)
            with pytest.raises(RuntimeError, match="before sending complete header"):
                await client.connect()
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_configure_sends_expected_commands(self):
        server = MockRtlTcpServer(iq_bytes=b"\x80\x80" * 1000)
        await server.start()
        try:
            client = RtlTcpClient("127.0.0.1", server.port)
            await client.connect()
            await client.configure(
                freq_hz=144_390_000, sample_rate=2_400_000, gain="auto"
            )
            await asyncio.sleep(0.05)  # let server read the commands

            # Convert to a dict for easy assertions
            cmds_by_id = {cmd: param for cmd, param in server.received_commands}
            assert 0x02 in cmds_by_id  # set_sample_rate
            assert cmds_by_id[0x02] == 2_400_000
            assert 0x01 in cmds_by_id  # set_freq
            assert cmds_by_id[0x01] == 144_390_000
            assert 0x03 in cmds_by_id  # set_gain_mode
            assert cmds_by_id[0x03] == 0  # auto
            await client.close()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_configure_manual_gain_sends_gain_cmd(self):
        server = MockRtlTcpServer(iq_bytes=b"\x80\x80" * 100)
        await server.start()
        try:
            client = RtlTcpClient("127.0.0.1", server.port)
            await client.connect()
            await client.configure(
                freq_hz=144_390_000, sample_rate=2_400_000, gain="40"
            )
            await asyncio.sleep(0.05)

            # Expect set_gain_mode=1 (manual) AND set_gain=400 (40 dB * 10)
            cmds_by_id = {cmd: param for cmd, param in server.received_commands}
            assert cmds_by_id.get(0x03) == 1  # manual
            assert cmds_by_id.get(0x04) == 400  # 40 dB → 400 tenths
            await client.close()
        finally:
            await server.stop()


class TestEndToEndBridge:
    """Integration: run_bridge against a mock server, capture PCM out."""

    @pytest.mark.asyncio
    async def test_run_bridge_produces_pcm(self):
        # Build a small synthetic FM-like IQ stream: all bytes 127,128
        # is effectively DC → after demod the output is near-silence
        # but should still produce samples of the correct length and
        # format.
        iq_bytes = bytes([127, 128] * 50_000)  # 50k IQ samples
        server = MockRtlTcpServer(iq_bytes=iq_bytes)
        await server.start()
        try:
            output = io.BytesIO()
            rc = await run_bridge(
                rtl_tcp_host="127.0.0.1",
                rtl_tcp_port=server.port,
                freq_hz=144_390_000,
                input_rate=50_000,  # low rate for this test
                output_rate=25_000,  # exact 2x
                gain="auto",
                deemphasis_tau_s=None,
                audio_scale=16384.0,
                output_stream=output,
            )
            assert rc == 0
            # Output size should roughly match input_samples / decimation * 2 bytes/sample
            # 50k IQ samples → 25k audio samples → 50k bytes
            assert output.tell() > 10_000  # at least some output
            assert output.tell() % 2 == 0  # int16 samples (even bytes)
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_run_bridge_handles_fm_signal(self):
        """Send a known FM-modulated tone; output should be nontrivial
        audio (not just silence)."""
        fs = 48_000
        f_m = 1_000  # baseband tone
        f_d = 5_000  # FM deviation
        duration_s = 0.2  # 9600 samples of IQ
        n = np.arange(int(fs * duration_s))
        t = n / fs
        phase = (f_d / f_m) * np.sin(2 * np.pi * f_m * t)

        # Convert complex exp to RTL-SDR uint8 bytes:
        # IQ range [-1, 1] → bytes [0, 255] centered on 127.5
        iq = np.exp(1j * phase)
        i_bytes = np.clip(iq.real * 127.5 + 127.5, 0, 255).astype(np.uint8)
        q_bytes = np.clip(iq.imag * 127.5 + 127.5, 0, 255).astype(np.uint8)
        # Interleave I,Q,I,Q
        iq_bytes = np.empty(iq.size * 2, dtype=np.uint8)
        iq_bytes[0::2] = i_bytes
        iq_bytes[1::2] = q_bytes

        server = MockRtlTcpServer(iq_bytes=iq_bytes.tobytes())
        await server.start()
        try:
            output = io.BytesIO()
            rc = await run_bridge(
                rtl_tcp_host="127.0.0.1",
                rtl_tcp_port=server.port,
                freq_hz=100_000_000,
                input_rate=fs,
                output_rate=fs,  # no resampling → simpler to verify
                gain="auto",
                deemphasis_tau_s=None,
                audio_scale=16384.0,
                output_stream=output,
            )
            assert rc == 0
            pcm = np.frombuffer(output.getvalue(), dtype=np.int16)
            # Output should be ~9600 int16 samples
            assert pcm.size > 5000
            # Not silence — check variance
            assert pcm.std() > 100  # meaningful signal energy
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_run_bridge_exits_cleanly_when_server_closes(self):
        """If the rtl_tcp server closes, run_bridge should exit 0."""
        server = MockRtlTcpServer(iq_bytes=bytes([127, 128] * 10))  # very short
        await server.start()
        try:
            output = io.BytesIO()
            rc = await run_bridge(
                rtl_tcp_host="127.0.0.1",
                rtl_tcp_port=server.port,
                freq_hz=1_000_000,
                input_rate=20_000,  # low rate allowed by rtl_tcp
                output_rate=10_000,
                gain="auto",
                deemphasis_tau_s=None,
                audio_scale=16384.0,
                output_stream=output,
            )
            assert rc == 0
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_run_bridge_fails_gracefully_on_bad_port(self):
        """Connecting to a port with no rtl_tcp server returns nonzero."""
        # Pick a port nothing should be listening on
        output = io.BytesIO()
        rc = await run_bridge(
            rtl_tcp_host="127.0.0.1",
            rtl_tcp_port=1,  # port 1 is reserved and won't accept
            freq_hz=144_390_000,
            input_rate=2_400_000,
            output_rate=22_050,
            gain="auto",
            deemphasis_tau_s=None,
            audio_scale=16384.0,
            output_stream=output,
        )
        assert rc != 0


class TestModuleInvocation:
    """python -m rfcensus.tools.fm_bridge should work — confirms the
    module is runnable with minimum surgery."""

    def test_has_main_function(self):
        assert callable(fm_bridge.main)

    def test_parser_help_runs(self):
        """Just verify the help text can be generated without
        exceptions — catches silly argparse config bugs."""
        parser = build_parser()
        help_text = parser.format_help()
        assert "--rtl-tcp" in help_text
        assert "--freq" in help_text
        assert "--input-rate" in help_text
        assert "--output-rate" in help_text
