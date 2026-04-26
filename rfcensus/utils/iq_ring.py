"""IqRingBuffer — fixed-size circular buffer for cu8 IQ samples,
indexed by global sample offset so callers can retroactively read
samples that arrived in the recent past.

Why we need this: the coarse-FFT passband detector identifies that a
LoRa transmission is happening at slot S, but it doesn't catch the
transmission until the preamble has already been on the air for tens
of milliseconds (energy needs to integrate above noise floor through
several detector hops). To decode the packet we need access to IQ
samples from BEFORE the detector fired — far enough back that we can
catch the preamble's start.

Lookback budget per Meshtastic preset (preamble = 8 symbols):
  • SF12/125kHz: 8 × 32.8ms = 262ms (LongSlow)
  • SF11/125kHz: 8 × 16.4ms = 131ms (LongMod)
  • SF11/250kHz: 8 × 8.2ms  = 66ms  (LongFast)
  • SF11/500kHz: 8 × 4.1ms  = 33ms  (LongTurbo)
  • SF10/250kHz: 8 × 4.1ms  = 33ms  (MediumSlow)
  • SF9/250kHz:  8 × 2.0ms  = 16ms  (MediumFast)
  • SF8/250kHz:  8 × 1.0ms  = 8ms   (ShortSlow)
  • SF7/250kHz:  8 × 0.5ms  = 4ms   (ShortFast)
  • SF7/500kHz:  8 × 0.26ms = 2ms   (ShortTurbo)

To catch LongSlow we'd need 262ms of lookback. That's a lot of IQ
(at 2.4 MS/s cu8 = 1.26 MB). The detector triggers on energy in the
WHOLE preamble + payload, so by the time we fire we may already be
into the data — for slow presets that means we've MISSED the preamble
and can't decode that packet anyway.

PRAGMATIC CHOICE: 300ms ring (1.44 MB at 2.4 MS/s) → catches every
preset's preamble if the detector fires within ~30ms of preamble start.
For the slowest two presets (LongSlow, LongMod), trigger latency is
inherently a few preamble symbols, so we accept missing the first
one of any session. Subsequent packets in the same session are caught
because the slot stays ACTIVE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class IqRingBuffer:
    """Circular buffer of cu8 IQ bytes, addressable by global sample
    offset.

    "Sample" = one IQ pair = 2 bytes (cu8 format). Capacity arguments
    and read positions are in SAMPLES, not bytes. Internally we store
    bytes so write/read is straight memcpy-style.

    Thread-safety: NOT thread-safe. Caller serializes write+read.
    """
    capacity_samples: int

    def __post_init__(self) -> None:
        if self.capacity_samples <= 0:
            raise ValueError(
                f"capacity_samples must be > 0, got {self.capacity_samples}"
            )
        # Allocate as bytearray for in-place writes.
        self._capacity_bytes = self.capacity_samples * 2
        self._buf = bytearray(self._capacity_bytes)
        # Total samples ever written. The "global offset" of the most
        # recently written sample is `total_written - 1`. The OLDEST
        # sample still in the buffer is at offset
        # max(0, total_written - capacity_samples).
        self._total_written = 0
        # v0.7.7: silent overflow visibility. When `write()` receives
        # a chunk LARGER than the buffer's total capacity, we
        # previously dropped the prefix silently and bumped
        # `_total_written` past it. That meant the consumer kept
        # reading from a coherent address space but with samples
        # missing — which manifests as missing packets with no
        # explanation. Counters surfaced via the new properties
        # below; lazy_pipeline polls them after each write and rolls
        # them into LazyPipelineStats so the standalone tool can
        # print "samples dropped: N" in its summary block.
        self._overflow_events = 0
        self._samples_dropped = 0

    @property
    def overflow_events(self) -> int:
        """v0.7.7: how many times write() had to drop a prefix
        because the incoming chunk exceeded the ring's capacity.
        Each event corresponds to one `write()` call that lost
        data; the actual sample count lost is in
        `samples_dropped`."""
        return self._overflow_events

    @property
    def samples_dropped(self) -> int:
        """v0.7.7: cumulative count of cu8 IQ samples dropped due
        to overflow in `write()`. Non-zero means the pipeline is
        not keeping up with the input rate — packets were lost in
        whatever frequency range the dropped samples covered."""
        return self._samples_dropped

    @property
    def total_written(self) -> int:
        """Total samples ever ingested (monotonically increasing)."""
        return self._total_written

    @property
    def oldest_offset(self) -> int:
        """Global sample offset of the oldest sample still in buffer.
        0 if buffer hasn't filled yet."""
        return max(0, self._total_written - self.capacity_samples)

    @property
    def newest_offset(self) -> int:
        """Global sample offset of the most recently written sample.
        Returns -1 if no samples have been written yet."""
        return self._total_written - 1

    def write(self, samples: bytes) -> None:
        """Append cu8 samples to the buffer. Overwrites oldest data
        when capacity is exceeded.

        ``samples`` must be cu8 bytes (length = 2 × number of IQ pairs).
        Length must be even.
        """
        n_bytes = len(samples)
        if n_bytes & 1:
            raise ValueError(
                f"cu8 samples must be even-length, got {n_bytes} bytes"
            )
        if n_bytes == 0:
            return
        n_samples = n_bytes // 2

        # If the incoming chunk is STRICTLY bigger than the buffer,
        # only the tail fits. The samples that "land" in the ring are
        # the last `capacity_samples` of the input; we must place them
        # at the byte positions consistent with the ring's offset
        # arithmetic (otherwise subsequent reads via global offset
        # return wrong bytes). Easiest: bump total_written past the
        # dropped prefix, then write the tail through the normal
        # wrap-aware path.
        #
        # v0.7.7: count what we drop. Previously this was silent.
        # NOTE: this branch fires when a SINGLE write() chunk
        # exceeds the ring — typically benign for small ring +
        # huge chunks. The MORE COMMON consumer-falling-behind
        # scenario doesn't trigger this branch — it just overwrites
        # old data on each wrap, which is the ring's NORMAL mode.
        # The lazy pipeline detects that scenario differently (by
        # noticing that a lookback read returns None because the
        # requested start_offset < oldest_offset).
        if n_bytes > self._capacity_bytes:
            tail_start_samples = n_samples - self.capacity_samples
            self._overflow_events += 1
            self._samples_dropped += tail_start_samples
            self._total_written += tail_start_samples
            tail_bytes = samples[tail_start_samples * 2:]
            return self.write(tail_bytes)

        # Compute the starting byte position in the ring for this write.
        # The next-write position is (total_written * 2) % capacity_bytes.
        write_pos_bytes = (self._total_written * 2) % self._capacity_bytes
        end_pos_bytes = write_pos_bytes + n_bytes

        if end_pos_bytes <= self._capacity_bytes:
            # Fits without wrapping
            self._buf[write_pos_bytes:end_pos_bytes] = samples
        else:
            # Splits across the wrap point
            first_part_bytes = self._capacity_bytes - write_pos_bytes
            self._buf[write_pos_bytes:] = samples[:first_part_bytes]
            self._buf[:end_pos_bytes - self._capacity_bytes] = (
                samples[first_part_bytes:]
            )

        self._total_written += n_samples

    def read(self, start_offset: int, n_samples: int) -> Optional[bytes]:
        """Read ``n_samples`` cu8 samples starting at global offset
        ``start_offset``.

        Returns ``None`` if any of the requested range is outside what's
        currently in the buffer (too old or beyond newest written).
        Returns the bytes (length = 2 × n_samples) on success.

        Use ``oldest_offset`` and ``newest_offset`` to check what's
        available before requesting.
        """
        if n_samples <= 0:
            return b""
        if start_offset < self.oldest_offset:
            return None   # too old, overwritten
        if start_offset + n_samples > self._total_written:
            return None   # not yet written

        read_pos_bytes = (start_offset * 2) % self._capacity_bytes
        n_bytes = n_samples * 2
        end_pos_bytes = read_pos_bytes + n_bytes

        if end_pos_bytes <= self._capacity_bytes:
            return bytes(self._buf[read_pos_bytes:end_pos_bytes])
        # Wraps
        first_part_bytes = self._capacity_bytes - read_pos_bytes
        return (bytes(self._buf[read_pos_bytes:])
                + bytes(self._buf[:end_pos_bytes - self._capacity_bytes]))

    def read_recent(self, n_samples: int) -> Optional[bytes]:
        """Read the most recent ``n_samples`` samples.

        Returns None if fewer than n_samples have been written.
        """
        if n_samples > self._total_written:
            return None
        start = self._total_written - n_samples
        return self.read(start, n_samples)

    def __len__(self) -> int:
        """Number of samples currently retained."""
        return min(self._total_written, self.capacity_samples)
