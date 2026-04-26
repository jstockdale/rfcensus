"""IQ data sources for the standalone Meshtastic decoder.

Three flavors:
  • ``FileIQSource``       — read cu8 from a saved file
  • ``RtlSdrSubprocess``  — spawn ``rtl_sdr`` and read its stdout
  • ``RtlTcpSource``       — connect to a running ``rtl_tcp`` server

All three present the same iterator interface: ``__next__`` returns a
``bytes`` chunk of raw cu8 (interleaved I, Q as uint8 centered on
127.5 — the rtl-sdr default). Use as::

    with FileIQSource(Path("cap.cu8")) as src:
        for chunk in src:
            decoder.feed_cu8(chunk)

We deliberately use subprocess ``rtl_sdr`` rather than librtlsdr via
ctypes for the same reason as the rest of rfcensus: works with
whatever librtlsdr ABI happens to be installed (v0.6, blog/v4 fork,
custom build), no FFI versioning headaches.
"""
from __future__ import annotations

import asyncio
import socket
import struct
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Iterator, Optional


# Default chunk size = 64 KiB. Big enough to keep subprocess overhead
# negligible, small enough that callers can pump packets out in near
# real-time without waiting on a giant buffer to fill. At 1 MS/s cu8
# (= 2 MB/s data rate), 64 KiB = 32 ms per chunk.
DEFAULT_CHUNK_SIZE = 1 << 16


# ─────────────────────────────────────────────────────────────────────
# Base + file source
# ─────────────────────────────────────────────────────────────────────

class IQSource:
    """Abstract IQ source — yields cu8 ``bytes`` chunks.

    Subclasses must implement ``read(n)`` returning up to n bytes (or
    fewer at EOF / less-data-available). Iteration calls ``read``
    until empty bytes are returned, signaling end-of-stream.

    v0.7.5: ``__next__`` now enforces I/Q alignment — every yielded
    chunk is even-length so downstream cu8 consumers (which assume
    each pair of bytes is one complex sample) never see a half-pair.
    Without this, ``RtlTcpSource`` could hand out odd chunks because
    ``socket.recv(n)`` is "up to n", and TCP arbitrarily fragments
    the byte stream — a 16383-byte read followed by a 16385-byte
    read both decode wrong if passed to ``feed_cu8`` raw. We buffer
    the leftover odd byte and prepend it to the next chunk.
    """
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        self._chunk_size = chunk_size
        self._closed = False
        # v0.7.5: holds the leftover odd byte (if any) from the
        # previous read. Always either b"" or a single byte.
        self._iq_remainder: bytes = b""

    def read(self, n: int) -> bytes:
        raise NotImplementedError

    def close(self) -> None:
        self._closed = True

    def __iter__(self) -> "IQSource":
        return self

    def __next__(self) -> bytes:
        if self._closed:
            raise StopIteration
        data = self.read(self._chunk_size)
        if not data:
            # End of stream. If there's a stranded odd byte, drop it —
            # it's half an I/Q pair with no partner coming.
            if self._iq_remainder:
                self._iq_remainder = b""
            raise StopIteration
        # Glue any remainder from last call onto the front, then
        # carve off any new odd byte for next time.
        if self._iq_remainder:
            data = self._iq_remainder + data
            self._iq_remainder = b""
        if len(data) % 2 == 1:
            self._iq_remainder = data[-1:]
            data = data[:-1]
            if not data:
                # Tiny read of exactly 1 byte and empty buffer —
                # return nothing usable but DON'T stop iteration;
                # the next read should bring more bytes.
                return self.__next__()
        return data

    def __enter__(self) -> "IQSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class FileIQSource(IQSource):
    """Read cu8 samples from a saved file. Yields chunk_size bytes per
    iteration until the file is exhausted."""

    def __init__(self, path: Path,
                 chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        super().__init__(chunk_size)
        self._path = path
        self._fp = path.open("rb")

    def read(self, n: int) -> bytes:
        return self._fp.read(n)

    def close(self) -> None:
        if not self._closed:
            self._fp.close()
        super().close()


# ─────────────────────────────────────────────────────────────────────
# rtl_sdr subprocess
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RtlSdrConfig:
    """Tuning configuration for rtl_sdr / rtl_tcp.

    ``device_index`` selects the dongle by USB enumeration order
    (0, 1, 2, ...). Use ``find_device_index_by_serial`` to convert a
    serial number to an index first if you have multiple dongles.

    ``gain_tenths_db`` is in tenths of a dB (rtl_sdr's native unit).
    Pass -1 for AGC. Typical good values for site survey are 200–400
    (20–40 dB), tuned per-dongle to maximize SNR without saturation.

    ``ppm`` is the frequency-correction offset in parts per million,
    needed for cheap dongles that drift. The user's existing rfcensus
    config typically pins this per-dongle from rtl_test calibration.
    """
    freq_hz: int
    sample_rate_hz: int
    device_index: int = 0
    gain_tenths_db: int = -1   # -1 = auto / AGC
    ppm: int = 0


class RtlSdrSubprocess(IQSource):
    """Spawn ``rtl_sdr`` and stream its cu8 output.

    The subprocess runs until ``close()`` is called or the parent
    process exits. Reads block waiting for more data — caller drives
    the pace via the iteration loop or chunk size.

    Example::

        cfg = RtlSdrConfig(freq_hz=915_000_000,
                           sample_rate_hz=2_400_000,
                           device_index=0,
                           gain_tenths_db=300)
        with RtlSdrSubprocess(cfg) as src:
            for chunk in src:
                decoder.feed_cu8(chunk)
    """

    def __init__(self, cfg: RtlSdrConfig,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 binary: str = "rtl_sdr") -> None:
        super().__init__(chunk_size)
        self._cfg = cfg
        self._binary = binary

        if which(binary) is None:
            raise RuntimeError(
                f"{binary!r} not found in PATH. Install librtlsdr "
                f"(rtl-sdr package on Debian/Ubuntu, "
                f"`brew install librtlsdr` on macOS)."
            )

        args = [
            binary,
            "-d", str(cfg.device_index),
            "-f", str(cfg.freq_hz),
            "-s", str(cfg.sample_rate_hz),
        ]
        if cfg.gain_tenths_db >= 0:
            args += ["-g", str(cfg.gain_tenths_db / 10.0)]
        if cfg.ppm != 0:
            args += ["-p", str(cfg.ppm)]
        # Trailing "-" means "write samples to stdout".
        args += ["-"]

        # We capture stderr so we can surface tuning errors. stdout
        # is the IQ pipe — never decoded, just relayed.
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,   # unbuffered so we get bytes promptly
        )

    @property
    def config(self) -> RtlSdrConfig:
        return self._cfg

    def read(self, n: int) -> bytes:
        if self._proc.stdout is None:
            return b""
        # subprocess.PIPE.read returns up to n bytes; with bufsize=0 it
        # returns whatever's currently in the pipe (may be < n).
        return self._proc.stdout.read(n)

    def retune(self, freq_hz: int) -> None:
        """Re-tune to a new frequency by killing and restarting the
        subprocess. There's no IPC channel into rtl_sdr to change
        frequency on the fly — use ``RtlTcpSource`` for that.

        Dead time during retune is typically 200-800 ms (kill + USB
        re-enumerate + new tuner programming). Acceptable for hop
        modes with multi-second dwell times; not for fast hopping.
        """
        # Build new command line with the updated frequency.
        new_cfg = RtlSdrConfig(
            freq_hz=freq_hz,
            sample_rate_hz=self._cfg.sample_rate_hz,
            device_index=self._cfg.device_index,
            gain_tenths_db=self._cfg.gain_tenths_db,
            ppm=self._cfg.ppm,
        )
        # Tear down the old subprocess.
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

        # Spin up a new one. We re-enter __init__'s argv build logic
        # by calling it again with the new config — but __init__ also
        # does the binary-existence check, which is wasteful. Inline
        # the parts we need.
        args = [
            self._binary,
            "-d", str(new_cfg.device_index),
            "-f", str(new_cfg.freq_hz),
            "-s", str(new_cfg.sample_rate_hz),
        ]
        if new_cfg.gain_tenths_db >= 0:
            args += ["-g", str(new_cfg.gain_tenths_db / 10.0)]
        if new_cfg.ppm != 0:
            args += ["-p", str(new_cfg.ppm)]
        args += ["-"]
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._cfg = new_cfg

    def close(self) -> None:
        if self._closed:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        super().close()


# ─────────────────────────────────────────────────────────────────────
# rtl_tcp client — for sharing dongles with other tools
# ─────────────────────────────────────────────────────────────────────

# rtl_tcp command opcodes (RtlTcp.cpp in librtlsdr). Each command is a
# 5-byte message: 1-byte cmd + 4-byte big-endian uint32 parameter.
_RTL_TCP_CMD_SET_FREQ          = 0x01
_RTL_TCP_CMD_SET_SAMPLE_RATE   = 0x02
_RTL_TCP_CMD_SET_GAIN_MODE     = 0x03  # 0 = auto, 1 = manual
_RTL_TCP_CMD_SET_GAIN          = 0x04  # tenths of dB
_RTL_TCP_CMD_SET_FREQ_CORR     = 0x05  # ppm


def _rtl_tcp_command(cmd: int, param: int) -> bytes:
    """Build a 5-byte rtl_tcp command message."""
    return struct.pack(">BI", cmd, param & 0xFFFFFFFF)


class RtlTcpSource(IQSource):
    """Stream cu8 from a running rtl_tcp server.

    The server is typically launched separately (or by rfcensus's
    fanout layer) so multiple clients can share the same dongle. We
    send the standard rtl_tcp control messages on connect to set
    frequency, sample rate, gain — but be aware that whichever client
    sends these LAST wins (rtl_tcp has no per-client state). For the
    standalone tool this means: don't share the dongle if you want
    deterministic tuning, OR coordinate tuning with other clients.
    """

    def __init__(self, host: str, port: int, cfg: RtlSdrConfig,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 connect_timeout: float = 5.0) -> None:
        super().__init__(chunk_size)
        self._cfg = cfg
        self._sock = socket.create_connection((host, port),
                                                timeout=connect_timeout)
        # rtl_tcp sends a 12-byte greeting on connect (magic "RTL0" +
        # tuner type + gain count). Read and discard.
        try:
            greeting = self._sock.recv(12, socket.MSG_WAITALL)
            if len(greeting) < 12 or greeting[:4] != b"RTL0":
                raise RuntimeError(
                    f"rtl_tcp greeting was {greeting!r}, expected b'RTL0...'"
                )
        except socket.timeout:
            self._sock.close()
            raise RuntimeError(f"rtl_tcp at {host}:{port} sent no greeting")

        # Switch to blocking mode for the data pipe (we want to wait
        # for samples, not return immediately on no-data).
        self._sock.settimeout(None)

        # Apply tuning.
        if cfg.gain_tenths_db < 0:
            self._send(_RTL_TCP_CMD_SET_GAIN_MODE, 0)  # AGC on
        else:
            self._send(_RTL_TCP_CMD_SET_GAIN_MODE, 1)
            self._send(_RTL_TCP_CMD_SET_GAIN, cfg.gain_tenths_db)
        self._send(_RTL_TCP_CMD_SET_SAMPLE_RATE, cfg.sample_rate_hz)
        self._send(_RTL_TCP_CMD_SET_FREQ, cfg.freq_hz)
        if cfg.ppm:
            self._send(_RTL_TCP_CMD_SET_FREQ_CORR, cfg.ppm)

    def _send(self, cmd: int, param: int) -> None:
        self._sock.sendall(_rtl_tcp_command(cmd, param))

    def read(self, n: int) -> bytes:
        try:
            return self._sock.recv(n)
        except (ConnectionResetError, BrokenPipeError):
            return b""

    def retune(self, freq_hz: int) -> None:
        """Re-tune the dongle by sending SET_FREQ to rtl_tcp.

        This is "fast hop" — no subprocess restart, just a 5-byte
        command on the open socket. Dead time is just the dongle's
        own PLL settling time (~1-10 ms). Works while other clients
        share the dongle but they'll see the retune too — coordinate
        if that matters.
        """
        self._send(_RTL_TCP_CMD_SET_FREQ, freq_hz)
        self._cfg = RtlSdrConfig(
            freq_hz=freq_hz,
            sample_rate_hz=self._cfg.sample_rate_hz,
            device_index=self._cfg.device_index,
            gain_tenths_db=self._cfg.gain_tenths_db,
            ppm=self._cfg.ppm,
        )

    def close(self) -> None:
        if not self._closed:
            try:
                self._sock.close()
            except OSError:
                pass
        super().close()


# ─────────────────────────────────────────────────────────────────────
# Helper: serial → device index
# ─────────────────────────────────────────────────────────────────────

def find_device_index_by_serial(serial: str,
                                 binary: str = "rtl_test") -> Optional[int]:
    """Map an RTL-SDR serial number to its enumeration index by parsing
    ``rtl_test -t`` output.

    Returns the integer index, or ``None`` if no matching dongle was
    found. Returns ``None`` (not error) if rtl_test isn't installed —
    callers should fall back to passing the index directly.

    The parsing is lenient: rtl_test's output format has changed
    between versions. We look for lines like::

        0:  Realtek, RTL2838UHIDIR, SN: 00000003

    and extract the leading integer + the SN field.
    """
    if which(binary) is None:
        return None

    try:
        result = subprocess.run(
            [binary, "-t"],
            capture_output=True, text=True, timeout=5.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    target = serial.strip()
    # rtl_test prints to stderr in some versions, stdout in others.
    for stream in (result.stdout, result.stderr):
        for line in stream.splitlines():
            # Match e.g. "  0:  Realtek, RTL2838UHIDIR, SN: 00000003"
            stripped = line.strip()
            if not stripped or ":" not in stripped:
                continue
            head, _, rest = stripped.partition(":")
            if not head.strip().isdigit():
                continue
            if "SN:" in rest:
                sn = rest.split("SN:", 1)[1].strip()
                # SN can have trailing comma + tuner info on some versions
                sn = sn.split(",", 1)[0].strip()
                if sn == target:
                    return int(head.strip())

    return None
