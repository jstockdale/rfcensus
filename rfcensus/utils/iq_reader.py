"""IqReader — decouple the IQ ingest from the decode pipeline.

v0.7.16: introduces a reader-thread layer between the raw IQSource
(rtl_tcp, RTL-SDR, file) and the decode pipeline. The reader thread
runs a tight ``source.read() → ring.write()`` loop and the consumer
(main thread, calling ``pipe.feed_cu8()``) reads from the ring at its
own pace.

Why we want this:

  • GC pause robustness. Python GC can pause for 10-100ms, which at
    2.4 MS/s = 240k samples = an entire MediumFast packet. Without a
    decouple buffer those samples vanish (kernel TCP recv fills,
    rtl_tcp drops). With a 2s ring, the consumer catches up on the
    next iteration with zero loss.

  • Visible backpressure. Ring fill level is a real-time gauge of
    how close to overload we are, vs the current "samples_dropped"
    counter that only tells you AFTER it happened.

  • Multi-SDR foundation. PLAN-multi-sdr needs per-source reader
    threads. This is the natural shape; the multi-SDR code becomes
    "instantiate N IqReaders, merge their outputs" instead of
    re-architecting the main loop.

  • Future parallelization. Channelizer + decoder C calls release
    the GIL, so a decoder thread + reader thread genuinely run in
    parallel on Pi 5's 4 cores.

What this is NOT:

  • Not a replacement for the lookback ring inside lazy_pipeline
    (``IqRingBuffer``). That serves a different purpose — letting
    the probe reach back 300ms to catch a preamble that started
    before the detector fired. Both rings exist; they're sized
    differently and serve different consumers.

  • Not a multi-consumer fanout. Single-producer, single-consumer.
    Multi-consumer (e.g. "decoder + recorder both watching this
    stream") is a future addition; the current shape doesn't need
    it.

Drop policy: drop-oldest on overflow. Surfaces ``samples_dropped``
and ``overflow_events`` counters identical in shape to the existing
``IqRingBuffer``. With a 2s ring, drops should be vanishingly rare;
when they happen, the pipeline takes a transient decode glitch and
re-bootstraps on the next IDLE period. The alternative (returning
contiguous-looking data with hidden gaps) would corrupt the lookback
ring's offset semantics, which is much worse.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from rfcensus.utils.iq_source import IQSource, DEFAULT_CHUNK_SIZE


# Default ring capacity in seconds of audio. 2s × 2.4 MS/s × 2 bytes
# = 9.6 MB. Negligible memory, comfortably absorbs any reasonable
# Python GC pause or transient processing stall.
DEFAULT_RING_CAPACITY_SECS = 2.0


@dataclass
class RingStats:
    """Snapshot of decouple-ring health for monitoring."""
    capacity_bytes: int
    bytes_buffered: int           # currently available to read
    bytes_written_total: int      # ever written by the producer
    bytes_read_total: int         # ever consumed by the reader
    samples_dropped: int          # = bytes_dropped // 2
    overflow_events: int          # how many ring-overflow incidents
    fill_pct: float               # bytes_buffered / capacity_bytes × 100


class DecoupleRing:
    """Thread-safe single-producer single-consumer byte ring.

    Drop-oldest on overflow. Blocking reads with optional timeout.
    Writes never block — they may evict old data, in which case
    ``samples_dropped`` and ``overflow_events`` increment.

    The ring stores raw bytes; cu8 sample size (2 bytes per IQ pair)
    is interpreted only when surfacing ``samples_dropped`` (= dropped
    bytes / 2). The ring itself is byte-agnostic.

    Locking: a single ``threading.Lock`` guards (read_pos, write_pos,
    and the buffer). The granularity is per-call, not per-byte.
    Throughput sanity-check: at 2.4 MS/s with 64KB chunks, lock is
    acquired ~75 times/sec by writer and ~75 times/sec by reader.
    Single-mutex contention is invisible at this rate.
    """

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError(
                f"capacity_bytes must be > 0, got {capacity_bytes}"
            )
        self._capacity = capacity_bytes
        self._buf = bytearray(capacity_bytes)
        # Both pointers are MONOTONIC byte counters (total bytes ever
        # written / read). Modulo capacity gives the position within
        # the circular buffer. Difference (write - read) = bytes
        # currently buffered.
        self._write_pos = 0
        self._read_pos = 0
        # Overflow accounting (mirrors IqRingBuffer).
        self._samples_dropped = 0
        self._overflow_events = 0
        # End-of-stream flag — set by close() to wake any blocked
        # reader and signal "no more data coming".
        self._closed = False
        # Lock + condvar for blocking reads.
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    def write(self, data: bytes) -> None:
        """Append ``data`` to the ring. Never blocks. If ``data``
        plus what's already buffered exceeds capacity, the oldest
        bytes are dropped (read_pos advances) to make room.

        Invariant: after this call returns, the ring contains the
        most recent ``min(capacity, total_bytes_written)`` bytes of
        the cumulative byte stream. ``write_pos`` always equals
        the cumulative bytes seen; ``read_pos`` is bumped to
        ``write_pos - capacity`` if needed (with ``samples_dropped``
        incrementing for any bytes the consumer never got to read).
        """
        n = len(data)
        if n == 0:
            return
        with self._not_empty:
            new_write_pos = self._write_pos + n
            # If the new write overflows the ring, the consumer-
            # visible read_pos must advance to the oldest byte still
            # in the ring (= new_write_pos - capacity). If that's
            # past the consumer's read_pos, it means we clobbered
            # bytes the consumer hadn't gotten to — count the drop.
            target_read_pos = max(0, new_write_pos - self._capacity)
            if target_read_pos > self._read_pos:
                dropped = target_read_pos - self._read_pos
                self._samples_dropped += dropped // 2
                self._overflow_events += 1
                self._read_pos = target_read_pos

            # If the chunk itself is larger than the ring, only the
            # LAST `capacity` bytes survive (the prefix would just
            # get immediately overwritten by its own tail).
            if n > self._capacity:
                data = data[n - self._capacity:]
                n = self._capacity

            # Write into the ring. The bytes we're writing represent
            # the last n bytes of the cumulative stream, so they
            # occupy position [(new_write_pos - n) % capacity ..]
            # in the underlying buffer.
            start_in_buf = (new_write_pos - n) % self._capacity
            end_unwrapped = start_in_buf + n
            if end_unwrapped <= self._capacity:
                self._buf[start_in_buf:end_unwrapped] = data
            else:
                first_chunk = self._capacity - start_in_buf
                self._buf[start_in_buf:] = data[:first_chunk]
                self._buf[:n - first_chunk] = data[first_chunk:]

            self._write_pos = new_write_pos
            self._not_empty.notify()

    def read(self, n: int, timeout: Optional[float] = None) -> bytes:
        """Read up to ``n`` bytes from the ring. Blocks until at
        least 1 byte is available or ``timeout`` expires.

        Returns:
          • non-empty bytes (length ≤ n) if data was available
          • empty bytes (b"") if the ring is closed AND empty,
            or if timeout expired with no data

        Returns ≤n bytes, not exactly n — caller can loop if it
        needs an exact count. This matches socket.recv() semantics.
        """
        if n <= 0:
            return b""
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        with self._not_empty:
            while self._write_pos == self._read_pos and not self._closed:
                if deadline is None:
                    self._not_empty.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return b""
                    self._not_empty.wait(timeout=remaining)
            # Either data is available or the ring is closed.
            available = self._write_pos - self._read_pos
            if available == 0:
                # Closed-and-empty.
                return b""
            n = min(n, available)
            start = self._read_pos % self._capacity
            end_unwrapped = start + n
            if end_unwrapped <= self._capacity:
                data = bytes(self._buf[start:end_unwrapped])
            else:
                first_chunk = self._capacity - start
                data = (
                    bytes(self._buf[start:])
                    + bytes(self._buf[:n - first_chunk])
                )
            self._read_pos += n
            return data

    def close(self) -> None:
        """Signal end-of-stream. Subsequent ``read()`` calls return
        any remaining buffered bytes, then b''. Any blocked reader
        wakes up immediately."""
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def stats(self) -> RingStats:
        with self._lock:
            buffered = self._write_pos - self._read_pos
            return RingStats(
                capacity_bytes=self._capacity,
                bytes_buffered=buffered,
                bytes_written_total=self._write_pos,
                bytes_read_total=self._read_pos,
                samples_dropped=self._samples_dropped,
                overflow_events=self._overflow_events,
                fill_pct=(100.0 * buffered / self._capacity)
                            if self._capacity else 0.0,
            )


# ─────────────────────────────────────────────────────────────────


@dataclass
class ReaderStats:
    """Snapshot of IqReader health (= ring stats + reader thread state)."""
    ring: RingStats
    reader_alive: bool
    reader_eof: bool
    bytes_read_from_source: int


class IqReader(IQSource):
    """Wraps an IQSource with a reader thread + decouple ring.

    Subclasses IQSource so it inherits ``__iter__`` + the cu8-pair
    alignment (odd-byte buffering) from the base class. The standalone
    tool's ``for chunk in src:`` loop works unchanged when src is an
    IqReader instead of the underlying IQSource.

    Lifecycle:
        reader = IqReader(source, sample_rate_hz=2_400_000)
        reader.start()
        try:
            for chunk in reader:
                pipe.feed_cu8(chunk)
        finally:
            reader.close()       # joins the reader thread + closes source

    Or as a context manager:
        with IqReader(source, sample_rate_hz=2_400_000) as reader:
            for chunk in reader:
                pipe.feed_cu8(chunk)
    """

    def __init__(
        self,
        source: IQSource,
        sample_rate_hz: int,
        ring_capacity_secs: float = DEFAULT_RING_CAPACITY_SECS,
        source_read_chunk_bytes: int = 65536,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        # IQSource.__init__ sets up self._chunk_size + self._iq_remainder
        # for the iterator's odd-byte handling. The chunk_size we pass
        # here is what the CONSUMER pulls per iteration — independent
        # of source_read_chunk_bytes (which is what the reader thread
        # pulls per source.read() call).
        super().__init__(chunk_size=chunk_size)
        # 2 bytes per cu8 IQ sample.
        capacity_bytes = int(ring_capacity_secs * sample_rate_hz * 2)
        # Round up to ensure at least 2× the source-read chunk fits —
        # otherwise even one full source read could overflow.
        if capacity_bytes < source_read_chunk_bytes * 2:
            capacity_bytes = source_read_chunk_bytes * 2
        self._source = source
        self._ring = DecoupleRing(capacity_bytes)
        self._read_chunk = source_read_chunk_bytes
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._eof = False
        self._bytes_read_from_source = 0
        # Lifecycle protection: don't double-start, don't read after close.
        self._lifecycle_lock = threading.Lock()
        self._started = False

    @property
    def ring(self) -> DecoupleRing:
        """Direct access to the underlying ring (e.g. for stats)."""
        return self._ring

    def start(self) -> None:
        """Spawn the reader thread. Idempotent — calling twice is a no-op."""
        with self._lifecycle_lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name="IqReader",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        """Reader-thread main loop. Pulls from source, writes to ring."""
        try:
            while not self._stop.is_set():
                try:
                    data = self._source.read(self._read_chunk)
                except Exception:
                    # Source raised (e.g. broken connection). Treat as
                    # EOF so the consumer sees b'' and exits cleanly.
                    break
                if not data:
                    break  # natural EOF
                self._bytes_read_from_source += len(data)
                self._ring.write(data)
        finally:
            self._eof = True
            # Wake any blocked reader so they see the EOF.
            self._ring.close()

    def read(self, n: int, timeout: Optional[float] = None) -> bytes:
        """Read up to ``n`` bytes from the decouple ring. Blocks until
        data is available or the source EOFs.

        Matches IQSource.read() semantics (b'' on EOF). The optional
        ``timeout`` parameter is an extension on top of the IQSource
        contract — used for tests and for consumers that want bounded
        blocking. The base class's ``__next__`` calls ``read(n)``
        without the timeout kwarg, which gets the default (unbounded
        blocking until data or EOF).
        """
        return self._ring.read(n, timeout=timeout)

    def close(self) -> None:
        """Stop the reader thread and close the source. Blocks briefly
        while the reader thread joins (up to 2 seconds)."""
        self._stop.set()
        # Closing the source first wakes the reader thread out of any
        # blocking recv(). It'll see "no data" and exit.
        try:
            self._source.close()
        except Exception:
            pass
        self._ring.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # IQSource.close sets self._closed which makes __next__ raise
        # StopIteration on the next iteration.
        super().close()

    def retune(self, freq_hz: int) -> None:
        """Pass-through to the underlying source's retune() (currently
        only RtlTcpSource implements this). Used by hop mode in the
        standalone tool. The reader thread continues running while the
        source retunes — there will be a brief sample gap during the
        hardware PLL settle (~1-10 ms), which the ring absorbs."""
        retune_fn = getattr(self._source, "retune", None)
        if retune_fn is None:
            raise AttributeError(
                f"{type(self._source).__name__} does not support retune()"
            )
        retune_fn(freq_hz)

    def stats(self) -> ReaderStats:
        return ReaderStats(
            ring=self._ring.stats(),
            reader_alive=(self._thread is not None and self._thread.is_alive()),
            reader_eof=self._eof,
            bytes_read_from_source=self._bytes_read_from_source,
        )

    # Context-manager convenience for the standalone tool.
    def __enter__(self) -> "IqReader":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
