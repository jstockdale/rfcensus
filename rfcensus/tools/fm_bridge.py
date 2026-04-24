"""FM bridge — connect to rtl_tcp, demodulate IQ to PCM, write to stdout.

Usage
=====

This module is an internal tool invoked by the multimon and direwolf
decoders. It replaces the `rtl_fm` binary in those decoders' pipelines
so the dongle can be shared (via rtl_tcp fanout) instead of held
exclusively.

Typical invocation from a decoder:

    python -m rfcensus.tools.fm_bridge \\
        --rtl-tcp 127.0.0.1:1235 \\
        --freq 144390000 \\
        --input-rate 2400000 \\
        --output-rate 22050 \\
        --gain auto \\
        | multimon-ng -t raw -a AFSK1200 -f alpha -

Inputs
======

  --rtl-tcp HOST:PORT       rtl_tcp server (typically the fanout)
  --freq HZ                 tuning frequency (rtl_tcp set_freq)
  --input-rate HZ           IQ sample rate (rtl_tcp set_sample_rate)
                            If the fanout is already at this rate, no
                            reconfiguration happens. Otherwise, whichever
                            client calls set_sample_rate last wins.
  --output-rate HZ          audio PCM rate (int16, mono, little-endian)
  --gain MODE               'auto' for AGC, or a numeric dB value

rtl_tcp protocol reference
==========================

After connection, rtl_tcp sends a 12-byte header:

    4 bytes  magic  "RTL0"
    4 bytes  tuner type      (big-endian uint32)
    4 bytes  number of gains (big-endian uint32)

Then it streams raw uint8 IQ bytes continuously.

Client commands are 5 bytes, big-endian:
    1 byte   command ID
    4 bytes  parameter

    0x01  set_freq          (Hz)
    0x02  set_sample_rate   (Hz)
    0x03  set_gain_mode     (0=auto, 1=manual)
    0x04  set_gain          (tenths of dB, e.g. 400 = 40 dB)
    0x05  set_freq_correction (ppm)
    0x06  set_tuner_gain_by_index
    0x0d  set_agc_mode      (0=off, 1=on)
    0x0e  set_direct_sampling
    0x0f  set_offset_tuning
    0x10  set_rtl_xtal
    0x11  set_tuner_xtal
    0x12  set_tuner_gain_index
    0x13  set_bias_tee

We only use a small subset (freq, sample_rate, gain_mode, gain, agc).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import struct
import sys

from rfcensus.tools.dsp import FMDemodulator


# ------------------------------------------------------------------
# Constants and logging
# ------------------------------------------------------------------


# Size of each IQ read from the rtl_tcp socket. 16384 balances latency
# against overhead: at 2.4 Msps that's ~3.4 ms of signal per read,
# plenty fine-grained for FM voice while keeping per-read overhead low.
IQ_READ_CHUNK_BYTES = 16_384

# rtl_tcp wire-protocol header is exactly 12 bytes starting with "RTL0".
RTL_TCP_HEADER_LEN = 12
RTL_TCP_MAGIC = b"RTL0"

# Command IDs we actually use
CMD_SET_FREQ = 0x01
CMD_SET_SAMPLE_RATE = 0x02
CMD_SET_GAIN_MODE = 0x03  # 0 = auto/AGC, 1 = manual
CMD_SET_GAIN = 0x04  # tenths of dB; only used when gain_mode=manual
CMD_SET_AGC_MODE = 0x08  # some builds; safe to ignore errors

log = logging.getLogger(__name__)


def _pack_command(cmd_id: int, param: int) -> bytes:
    """Pack a rtl_tcp command: 1-byte cmd_id + 4-byte big-endian param."""
    return bytes([cmd_id]) + struct.pack(">I", param & 0xFFFFFFFF)


# ------------------------------------------------------------------
# rtl_tcp client
# ------------------------------------------------------------------


class RtlTcpClient:
    """Minimal async rtl_tcp client: connect, parse header, read IQ
    bytes, send tune / rate / gain commands.

    Lifecycle: construct → `await connect()` → `await configure(...)` →
    async-iterate `read_iq_chunks()` → `await close()`.

    Not intended for reuse across hosts — one client per target.
    """

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self.tuner_type: int | None = None
        self.num_gains: int | None = None

    async def connect(self, *, timeout_s: float = 15.0) -> None:
        """Open the TCP connection and read the rtl_tcp header.

        The header is fixed-size (12 bytes) and begins with 'RTL0'.
        We validate the magic so that connecting to the wrong port
        (say, HTTP) fails fast with a clear error instead of producing
        garbage 'IQ' samples.
        """
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=timeout_s,
            )
        except (asyncio.TimeoutError, TimeoutError) as e:
            raise RuntimeError(
                f"rtl_tcp connect to {self.host}:{self.port} timed out "
                f"after {timeout_s}s"
            ) from e
        except OSError as e:
            raise RuntimeError(
                f"rtl_tcp connect to {self.host}:{self.port} failed: {e}"
            ) from e

        # Read header strictly. readexactly raises IncompleteReadError
        # if the server disconnects mid-header — we surface that as a
        # RuntimeError with context.
        try:
            header = await asyncio.wait_for(
                self._reader.readexactly(RTL_TCP_HEADER_LEN),
                timeout=timeout_s,
            )
        except asyncio.IncompleteReadError as e:
            raise RuntimeError(
                f"rtl_tcp at {self.host}:{self.port} closed "
                f"before sending complete header ({len(e.partial)} / "
                f"{RTL_TCP_HEADER_LEN} bytes)"
            ) from e
        except (asyncio.TimeoutError, TimeoutError) as e:
            raise RuntimeError(
                f"rtl_tcp at {self.host}:{self.port} didn't send "
                f"header within {timeout_s}s"
            ) from e

        magic = header[0:4]
        if magic != RTL_TCP_MAGIC:
            raise RuntimeError(
                f"rtl_tcp at {self.host}:{self.port} sent bad magic "
                f"{magic!r}; expected {RTL_TCP_MAGIC!r}. "
                f"Is this actually an rtl_tcp server?"
            )
        self.tuner_type = struct.unpack(">I", header[4:8])[0]
        self.num_gains = struct.unpack(">I", header[8:12])[0]
        log.info(
            "rtl_tcp connected to %s:%d (tuner=%d, gains=%d)",
            self.host, self.port, self.tuner_type, self.num_gains,
        )

    async def _send_command(self, cmd_id: int, param: int) -> None:
        if self._writer is None:
            raise RuntimeError("rtl_tcp client not connected")
        self._writer.write(_pack_command(cmd_id, param))
        await self._writer.drain()

    async def configure(
        self,
        *,
        freq_hz: int,
        sample_rate: int,
        gain: str = "auto",
    ) -> None:
        """Send the standard configure sequence: sample_rate, freq, gain.

        Order is deliberate: sample_rate before freq matches what
        rtl_fm does (the dongle re-tunes cleaner this way on some
        chipsets). Gain last because it depends on mode.
        """
        await self._send_command(CMD_SET_SAMPLE_RATE, int(sample_rate))
        await self._send_command(CMD_SET_FREQ, int(freq_hz))

        if gain.strip().lower() == "auto":
            # Auto gain mode = AGC on the tuner.
            await self._send_command(CMD_SET_GAIN_MODE, 0)
        else:
            # Manual gain. Parse as dB, convert to tenths-of-dB param.
            try:
                db = float(gain)
            except ValueError as e:
                raise ValueError(
                    f"gain must be 'auto' or a number in dB; got {gain!r}"
                ) from e
            await self._send_command(CMD_SET_GAIN_MODE, 1)
            await self._send_command(CMD_SET_GAIN, int(round(db * 10)))

    async def read_iq_chunks(self, chunk_bytes: int = IQ_READ_CHUNK_BYTES):
        """Async-iterate IQ byte chunks until the server closes or
        we're cancelled. Each chunk is `chunk_bytes` bytes UNLESS the
        server closed mid-chunk, in which case the final yield is a
        (possibly) partial chunk.

        Guaranteed-even byte counts: even if the server closes at an
        odd byte boundary, the last partial chunk may have odd length.
        Callers are expected to buffer and merge partial data OR drop
        the last byte — the byte-pair boundary is part of the caller's
        concern if odd-length matters to them.
        """
        if self._reader is None:
            raise RuntimeError("rtl_tcp client not connected")
        while True:
            try:
                chunk = await self._reader.read(chunk_bytes)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("rtl_tcp read error: %s; closing", e)
                return
            if not chunk:
                # Server closed the connection
                log.info("rtl_tcp server closed connection")
                return
            yield chunk

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
            self._writer = None
            self._reader = None


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------


async def run_bridge(
    *,
    rtl_tcp_host: str,
    rtl_tcp_port: int,
    freq_hz: int,
    input_rate: int,
    output_rate: int,
    gain: str,
    deemphasis_tau_s: float | None,
    audio_scale: float,
    output_stream=None,  # defaults to sys.stdout.buffer
) -> int:
    """Main pipeline. Returns exit code (0 = success, nonzero = error)."""

    if output_stream is None:
        output_stream = sys.stdout.buffer

    client = RtlTcpClient(rtl_tcp_host, rtl_tcp_port)

    # Use sys.stderr for progress logging so it doesn't mix with PCM
    # on stdout. This is critical: the audio pipeline is on stdout.
    logging.basicConfig(
        level=logging.INFO,
        format="fm_bridge: %(message)s",
        stream=sys.stderr,
    )

    try:
        await client.connect()
    except RuntimeError as e:
        log.error("%s", e)
        return 2

    try:
        await client.configure(
            freq_hz=freq_hz,
            sample_rate=input_rate,
            gain=gain,
        )
    except ValueError as e:
        log.error("configuration failed: %s", e)
        await client.close()
        return 2

    # Handle an odd-byte tail from a previous chunk: rtl_tcp byte
    # stream is I/Q interleaved, so any buffer must be even-length
    # before decoding.
    tail: bytes = b""

    demod = FMDemodulator(
        input_rate=input_rate,
        output_rate=output_rate,
        deemphasis_tau_s=deemphasis_tau_s,
        audio_scale=audio_scale,
    )

    log.info(
        "streaming: %d MHz, in %d Hz, out %d Hz, decimation %dx",
        freq_hz // 1_000_000,
        input_rate,
        output_rate,
        demod._decimation,
    )

    total_iq_bytes = 0
    total_pcm_bytes = 0

    try:
        async for chunk in client.read_iq_chunks():
            if tail:
                chunk = tail + chunk
                tail = b""
            # Ensure even length for IQ byte pairs
            if len(chunk) % 2 != 0:
                tail = chunk[-1:]
                chunk = chunk[:-1]

            pcm = demod.process_iq_bytes(chunk)
            if pcm.size == 0:
                continue

            try:
                output_stream.write(pcm.tobytes())
                output_stream.flush()
            except (BrokenPipeError, IOError) as e:
                # Downstream (multimon/direwolf) exited or closed
                # stdin. Not an error in the fm_bridge sense — our
                # job is done.
                log.info("downstream closed output pipe: %s", e)
                return 0

            total_iq_bytes += len(chunk)
            total_pcm_bytes += pcm.nbytes
    except asyncio.CancelledError:
        log.info("cancelled; shutting down")
        raise
    finally:
        await client.close()
        log.info(
            "done. consumed %.1f MB IQ, produced %.1f MB PCM",
            total_iq_bytes / 1e6,
            total_pcm_bytes / 1e6,
        )

    return 0


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _parse_rtl_tcp(spec: str) -> tuple[str, int]:
    """Parse 'host:port' into (host, port)."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"--rtl-tcp must be HOST:PORT; got {spec!r}"
        )
    host, _, port_str = spec.rpartition(":")
    if not host or not port_str:
        raise argparse.ArgumentTypeError(
            f"--rtl-tcp must be HOST:PORT; got {spec!r}"
        )
    try:
        port = int(port_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--rtl-tcp port must be an integer; got {port_str!r}"
        ) from e
    if not (1 <= port <= 65535):
        raise argparse.ArgumentTypeError(
            f"--rtl-tcp port must be 1-65535; got {port}"
        )
    return host, port


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rfcensus.tools.fm_bridge",
        description=(
            "Demodulate narrowband FM from an rtl_tcp IQ stream and "
            "write PCM audio to stdout. Intended to replace `rtl_fm` "
            "in multimon/direwolf pipelines so dongle access can be "
            "shared via the rtl_tcp fanout."
        ),
    )
    p.add_argument(
        "--rtl-tcp",
        required=True,
        type=_parse_rtl_tcp,
        metavar="HOST:PORT",
        help="rtl_tcp server address (e.g. 127.0.0.1:1235)",
    )
    p.add_argument(
        "--freq",
        required=True,
        type=int,
        metavar="HZ",
        help="tuning frequency in Hz (e.g. 144390000 for APRS)",
    )
    p.add_argument(
        "--input-rate",
        required=True,
        type=int,
        metavar="HZ",
        help="IQ sample rate (e.g. 2400000)",
    )
    p.add_argument(
        "--output-rate",
        required=True,
        type=int,
        metavar="HZ",
        help="audio PCM rate (e.g. 22050 for multimon, 48000 for direwolf)",
    )
    p.add_argument(
        "--gain",
        default="auto",
        metavar="MODE",
        help="'auto' (AGC) or a numeric dB value (default: auto)",
    )
    p.add_argument(
        "--deemphasis-us",
        type=float,
        default=None,
        metavar="US",
        help=(
            "de-emphasis time constant in microseconds (typical: 75 for "
            "US broadcast). Default: none (pass-through). For paging/"
            "AFSK you usually don't want de-emphasis."
        ),
    )
    p.add_argument(
        "--audio-scale",
        type=float,
        default=16384.0,
        help=(
            "scale applied before int16 conversion. Higher = louder / "
            "more clipping. Default 16384 leaves ~6 dB of headroom."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    host, port = args.rtl_tcp
    tau_s = None
    if args.deemphasis_us is not None:
        tau_s = args.deemphasis_us * 1e-6

    # Graceful SIGTERM → cancel the main task → clean shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    main_task = loop.create_task(
        run_bridge(
            rtl_tcp_host=host,
            rtl_tcp_port=port,
            freq_hz=args.freq,
            input_rate=args.input_rate,
            output_rate=args.output_rate,
            gain=args.gain,
            deemphasis_tau_s=tau_s,
            audio_scale=args.audio_scale,
        )
    )

    def _cancel_on_signal():
        if not main_task.done():
            main_task.cancel()

    with contextlib.suppress(NotImplementedError):
        # Windows lacks add_signal_handler; just skip graceful shutdown
        loop.add_signal_handler(signal.SIGTERM, _cancel_on_signal)
        loop.add_signal_handler(signal.SIGINT, _cancel_on_signal)

    try:
        return loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        return 0
    finally:
        loop.close()


if __name__ == "__main__":
    sys.exit(main())
