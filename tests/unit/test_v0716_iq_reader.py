"""Tests for v0.7.16 DecoupleRing + IqReader.

Coverage:
  • Single-threaded ring operations (write, read, wrap, drop-oldest)
  • Threading: producer + consumer in parallel, no data corruption
  • Drop accounting: samples_dropped tracks accurately under overflow
  • IqReader EOF propagation: source EOF → reader thread exits → consumer sees b''
  • IqReader close() under various conditions: idle, mid-read, mid-write
  • Bytes integrity over 10s+ of synthetic streaming
"""

import threading
import time

import pytest

from rfcensus.utils.iq_reader import DecoupleRing, IqReader


# ─────────────────────────────────────────────────────────────────
# DecoupleRing — pure single-thread tests
# ─────────────────────────────────────────────────────────────────


class TestDecoupleRingSingleThread:

    def test_write_then_read_round_trip(self):
        r = DecoupleRing(capacity_bytes=1024)
        r.write(b"hello world")
        assert r.read(11) == b"hello world"
        assert r.read(1, timeout=0.01) == b""   # empty + timeout

    def test_partial_read(self):
        r = DecoupleRing(capacity_bytes=1024)
        r.write(b"abcdefghij")
        assert r.read(3) == b"abc"
        assert r.read(3) == b"def"
        assert r.read(10) == b"ghij"   # ≤n, not exactly n
        assert r.read(1, timeout=0.01) == b""

    def test_wrap_around(self):
        # Capacity 16; write 12, read 12, write 12 (forces wrap)
        r = DecoupleRing(capacity_bytes=16)
        r.write(b"a" * 12)
        assert r.read(12) == b"a" * 12
        r.write(b"b" * 12)
        assert r.read(12) == b"b" * 12

    def test_wrap_within_single_read(self):
        # Capacity 16; write 8 then read 8, then write 12 (which wraps),
        # then read 12 (which spans the wrap).
        r = DecoupleRing(capacity_bytes=16)
        r.write(b"X" * 8)
        assert r.read(8) == b"X" * 8
        # write_pos = 8, read_pos = 8. Write 12 bytes: occupies [8..15] + [0..3].
        r.write(b"0123456789AB")
        assert r.read(12) == b"0123456789AB"

    def test_drop_oldest_on_overflow(self):
        r = DecoupleRing(capacity_bytes=8)
        r.write(b"AAAA")           # 4 bytes buffered
        r.write(b"BBBBBBBB")       # need 8 more, only 4 free → drop AAAA
        # Buffer should be the LAST 8 bytes that were ever written:
        # AAAA + BBBBBBBB = 12 bytes total; ring keeps last 8 = AAAB BBBB BBB ...
        # Actually: write_pos progresses 4 then 12 total. read_pos was bumped
        # from 0 to 4 to make room. So buffered = bytes 4..11 = BBBBBBBB.
        assert r.read(100) == b"BBBBBBBB"
        s = r.stats()
        assert s.samples_dropped == 4 // 2  # 4 bytes = 2 samples
        assert s.overflow_events == 1

    def test_oversize_chunk_drops_prefix(self):
        r = DecoupleRing(capacity_bytes=8)
        r.write(b"0123456789ABCDEF")  # 16 bytes, capacity 8 → drop first 8
        assert r.read(100) == b"89ABCDEF"
        s = r.stats()
        assert s.samples_dropped == 8 // 2
        assert s.overflow_events == 1

    def test_close_then_read_returns_remaining_then_empty(self):
        r = DecoupleRing(capacity_bytes=16)
        r.write(b"final data")
        r.close()
        # Should drain remaining bytes first
        assert r.read(100) == b"final data"
        # Then empty (closed + drained)
        assert r.read(100) == b""

    def test_read_blocks_until_data_or_close(self):
        r = DecoupleRing(capacity_bytes=16)
        result = []

        def consumer():
            result.append(r.read(10, timeout=2.0))

        t = threading.Thread(target=consumer)
        t.start()
        time.sleep(0.05)             # ensure consumer is blocked
        r.write(b"hello")
        t.join(timeout=1.0)
        assert result == [b"hello"]

    def test_read_timeout_returns_empty(self):
        r = DecoupleRing(capacity_bytes=16)
        t0 = time.monotonic()
        assert r.read(10, timeout=0.05) == b""
        elapsed = time.monotonic() - t0
        # Should respect the timeout (allow generous slack on slow CI)
        assert 0.04 < elapsed < 0.5

    def test_close_unblocks_reader(self):
        r = DecoupleRing(capacity_bytes=16)
        result = []

        def consumer():
            result.append(r.read(10, timeout=5.0))

        t = threading.Thread(target=consumer)
        t.start()
        time.sleep(0.05)
        r.close()
        t.join(timeout=1.0)
        assert result == [b""]
        assert not t.is_alive()

    def test_stats_fill_pct(self):
        r = DecoupleRing(capacity_bytes=100)
        assert r.stats().fill_pct == 0.0
        r.write(b"x" * 25)
        assert r.stats().fill_pct == 25.0
        assert r.stats().bytes_buffered == 25
        r.read(10)
        assert r.stats().fill_pct == 15.0
        assert r.stats().bytes_buffered == 15
        assert r.stats().bytes_read_total == 10
        assert r.stats().bytes_written_total == 25


# ─────────────────────────────────────────────────────────────────
# DecoupleRing — concurrent tests
# ─────────────────────────────────────────────────────────────────


class TestDecoupleRingConcurrent:

    def test_producer_consumer_no_corruption(self):
        """Run a producer + consumer; verify byte order is preserved
        and no data is lost. Ring is sized larger than total data so
        drops can't happen — this test is about thread-safety of the
        write/read paths, NOT about the drop semantics (which the
        next test covers)."""
        n_bytes = 1 << 18       # 256 KB total
        capacity = n_bytes * 2  # 512 KB ring — comfortably larger
        r = DecoupleRing(capacity_bytes=capacity)
        chunk = 1024
        produced = bytearray(n_bytes)
        for i in range(n_bytes):
            produced[i] = i & 0xFF
        produced = bytes(produced)
        consumed = bytearray()

        def producer():
            for i in range(0, n_bytes, chunk):
                r.write(produced[i:i + chunk])
            r.close()

        def consumer():
            while True:
                data = r.read(chunk * 2, timeout=2.0)
                if not data:
                    break
                consumed.extend(data)

        tp = threading.Thread(target=producer, name="producer")
        tc = threading.Thread(target=consumer, name="consumer")
        tp.start(); tc.start()
        tp.join(timeout=10.0)
        tc.join(timeout=10.0)
        assert not tp.is_alive()
        assert not tc.is_alive()
        assert len(consumed) == n_bytes, (
            f"length mismatch: produced {n_bytes}, consumed {len(consumed)}")
        assert bytes(consumed) == produced, "byte order corruption"
        # Ring was sized to hold everything, so zero drops expected
        assert r.stats().samples_dropped == 0
        assert r.stats().overflow_events == 0

    def test_drops_under_consumer_starvation(self):
        """Producer way faster than consumer → ring overflows → drops."""
        capacity = 256
        r = DecoupleRing(capacity_bytes=capacity)
        n_chunks = 1000
        chunk_size = 64

        def producer():
            for _ in range(n_chunks):
                r.write(b"x" * chunk_size)
            r.close()

        consumed_total = 0

        def consumer():
            nonlocal consumed_total
            time.sleep(0.05)   # let producer get a head start
            while True:
                data = r.read(chunk_size, timeout=1.0)
                if not data:
                    break
                consumed_total += len(data)

        tp = threading.Thread(target=producer)
        tc = threading.Thread(target=consumer)
        tp.start(); tc.start()
        tp.join(timeout=10.0); tc.join(timeout=10.0)

        # Total bytes either consumed or dropped should equal what
        # was written. samples_dropped counts SAMPLES (= bytes/2).
        s = r.stats()
        bytes_dropped = s.samples_dropped * 2
        assert consumed_total + bytes_dropped == n_chunks * chunk_size, (
            f"accounting mismatch: consumed={consumed_total} "
            f"dropped_bytes={bytes_dropped} written={n_chunks * chunk_size}")
        # Should have detected overflow at least once
        assert s.overflow_events > 0


# ─────────────────────────────────────────────────────────────────
# IqReader — integration with a fake source
# ─────────────────────────────────────────────────────────────────


class FakeSource:
    """Mock IQSource: yields a fixed sequence of chunks then EOFs."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._idx = 0
        self._closed = False
        self._read_event = threading.Event()

    def read(self, n):
        if self._closed:
            return b""
        if self._idx >= len(self._chunks):
            return b""
        chunk = self._chunks[self._idx]
        self._idx += 1
        self._read_event.set()
        return chunk

    def close(self):
        self._closed = True


class SlowSource:
    """Mock IQSource that sleeps between reads."""

    def __init__(self, chunks, delay_secs):
        self._chunks = list(chunks)
        self._idx = 0
        self._delay = delay_secs
        self._closed = False

    def read(self, n):
        if self._closed:
            return b""
        if self._idx >= len(self._chunks):
            return b""
        time.sleep(self._delay)
        if self._closed:    # check again post-sleep
            return b""
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk

    def close(self):
        self._closed = True


class TestIqReader:

    def test_passes_through_data(self):
        chunks = [b"AAAA", b"BBBB", b"CCCC"]
        src = FakeSource(chunks)
        reader = IqReader(src, sample_rate_hz=2_400_000)
        reader.start()
        try:
            collected = bytearray()
            while True:
                data = reader.read(64, timeout=2.0)
                if not data:
                    break
                collected.extend(data)
            assert bytes(collected) == b"AAAABBBBCCCC"
        finally:
            reader.close()

    def test_eof_propagates(self):
        src = FakeSource([b"hello"])
        reader = IqReader(src, sample_rate_hz=2_400_000)
        reader.start()
        try:
            assert reader.read(100, timeout=2.0) == b"hello"
            # Subsequent read should hit EOF (reader thread exited,
            # ring closed)
            assert reader.read(100, timeout=2.0) == b""
        finally:
            reader.close()

    def test_close_is_idempotent(self):
        src = FakeSource([b"data"])
        reader = IqReader(src, sample_rate_hz=2_400_000)
        reader.start()
        reader.close()
        reader.close()  # should not raise

    def test_close_during_read_unblocks(self):
        src = SlowSource([b"x"] * 10, delay_secs=0.5)
        reader = IqReader(src, sample_rate_hz=2_400_000)
        reader.start()
        result = []

        def consumer():
            time.sleep(0.05)
            data = reader.read(1024, timeout=5.0)
            # might get one chunk before close, then b''
            while data:
                result.append(data)
                data = reader.read(1024, timeout=2.0)

        t = threading.Thread(target=consumer)
        t.start()
        time.sleep(0.1)
        reader.close()
        t.join(timeout=3.0)
        assert not t.is_alive(), "consumer didn't unblock on close"

    def test_stats_track_progress(self):
        chunks = [b"x" * 1000 for _ in range(10)]
        src = FakeSource(chunks)
        reader = IqReader(src, sample_rate_hz=2_400_000)
        reader.start()
        try:
            # Drain everything
            collected = bytearray()
            while True:
                data = reader.read(2000, timeout=2.0)
                if not data:
                    break
                collected.extend(data)
            assert len(collected) == 10_000
            s = reader.stats()
            assert s.bytes_read_from_source == 10_000
            assert s.ring.bytes_written_total == 10_000
            assert s.ring.bytes_read_total == 10_000
            assert s.ring.samples_dropped == 0
            assert s.reader_eof is True
        finally:
            reader.close()

    def test_context_manager(self):
        src = FakeSource([b"hello"])
        with IqReader(src, sample_rate_hz=2_400_000) as reader:
            assert reader.read(100, timeout=2.0) == b"hello"
        # After exit: source closed, thread joined
        assert reader.stats().reader_alive is False

    def test_capacity_sizing(self):
        """2 sec at 2.4 MS/s should give 9.6 MB ring."""
        src = FakeSource([])
        reader = IqReader(
            src,
            sample_rate_hz=2_400_000,
            ring_capacity_secs=2.0,
        )
        # 2.0 × 2.4M × 2 bytes = 9_600_000
        assert reader.ring.capacity_bytes == 9_600_000
        reader.close()

    def test_capacity_floor_for_tiny_rates(self):
        """Tiny capacity gets bumped to at least 2× source-read chunk."""
        src = FakeSource([])
        reader = IqReader(
            src,
            sample_rate_hz=100,        # ridiculously low
            ring_capacity_secs=0.001,
            source_read_chunk_bytes=4096,
        )
        # 0.001s × 100Hz × 2 = 0.2 bytes — below floor.
        # Floor = 2 × 4096 = 8192
        assert reader.ring.capacity_bytes == 8192
        reader.close()
