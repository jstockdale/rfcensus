"""PCAP writer with DLT_LORATAP (link-layer type 270) framing.

Writes classic libpcap files (NOT pcap-ng) — both formats are read by
Wireshark, but classic pcap is simpler and has no real downsides for
single-link capture.

Each LoRa frame gets a 15-byte LoRaTap v0 header prepended in front of
the raw PHY payload bytes. Wireshark's built-in LoRaTap dissector
(stable since 2.5.0) parses the header and exposes the frequency, SF,
bandwidth, RSSI and SNR fields. The Meshtastic Wireshark dissector
plugin (community) layers MeshPacket parsing on top of the LoRaTap
payload.

References:
  • https://github.com/eriknl/LoRaTap (header layout, BSD-3-Clause spec)
  • https://www.tcpdump.org/linktypes.html (DLT registry)
  • https://wiki.wireshark.org/Development/LibpcapFileFormat (file format)
"""
from __future__ import annotations

import struct
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, Optional


# ─────────────────────────────────────────────────────────────────────
# Constants — DO NOT modify without consulting the LoRaTap spec
# ─────────────────────────────────────────────────────────────────────

# libpcap link-layer type for LoRaTap encapsulation. Registered in
# tcpdump-group/libpcap pcap/dlt.h:
#   #define DLT_LORATAP             270
DLT_LORATAP = 270

# Classic pcap file magic. We use the LE/microsecond-resolution variant
# (the most common; nanosecond resolution = 0xa1b23c4d if you ever need
# better than 1µs, which we don't for radio frames). Reading with the
# matching magic on any host gets the byte order right automatically.
_PCAP_MAGIC_USEC_LE = 0xA1B2C3D4

# Maximum captured frame size. LoRa max payload is 255 bytes + 15-byte
# LoRaTap header = 270 bytes. We use 4096 to be generous and allow for
# future header extensions (LoRaTap v1 adds an optional `tag` field).
_DEFAULT_SNAPLEN = 4096

# LoRaTap v0 fixed header size (the format we emit). Future LoRaTap v1
# extends this with extra fields; we stick with v0 for maximum tooling
# compatibility — Wireshark's dissector reads both, but anything older
# than ~2.5.0 only knows v0.
_LORATAP_V0_LEN = 15


def _bw_to_loratap(bandwidth_hz: int) -> int:
    """Convert bandwidth in Hz to the LoRaTap "125 kHz steps" encoding.

    LoRaTap stores BW as ``bandwidth_kHz / 125``: 125 kHz → 1, 250 kHz
    → 2, 500 kHz → 4. Values that don't divide evenly are clamped to
    the nearest standard rate."""
    bw_khz = bandwidth_hz // 1000
    # Round to nearest standard LoRa BW (125, 250, 500). The spec
    # technically allows arbitrary multiples but no real radio uses
    # anything else, and Wireshark assumes the standard set.
    if bw_khz <= 187:
        return 1   # 125 kHz
    if bw_khz <= 375:
        return 2   # 250 kHz
    return 4       # 500 kHz


def _rssi_to_loratap(rssi_dbm: float) -> int:
    """Encode a dBm RSSI value to the LoRaTap u8 field.

    LoRaTap RSSI uses an offset/scale that depends on SNR sign:
      • SNR >= 0:  dBm = -139 + packet_rssi
      • SNR < 0:   dBm = -139 + packet_rssi * 0.25

    We pick the +1 step (SNR>=0) encoding because that's the linear
    one and fine for the typical Meshtastic SNR range. Caller-side
    SNR sign is currently always reported as 0 since our decoder
    doesn't measure SNR yet.

    Returns 255 for "N/A" if the input is exactly 0 (our placeholder
    value in lora_decoded_t.rssi_db when measurement isn't available).
    """
    if rssi_dbm == 0.0:
        return 255  # N/A sentinel per LoRaTap spec
    encoded = int(round(rssi_dbm + 139))
    return max(0, min(254, encoded))


def _snr_to_loratap(snr_db: float) -> int:
    """Encode SNR in dB to the LoRaTap u8 field.

    LoRaTap stores SNR as a signed two's-complement int divided by 4:
      dB = (snr_byte_signed) / 4
    So a +5 dB SNR → 20 (raw u8 stored as 0x14), -5 dB → -20 (0xEC).
    """
    raw = int(round(snr_db * 4))
    raw = max(-128, min(127, raw))
    return raw & 0xFF


def build_loratap_header(
    frequency_hz: int,
    bandwidth_hz: int,
    sf: int,
    sync_word: int,
    rssi_dbm: float = 0.0,
    snr_db: float = 0.0,
) -> bytes:
    """Build a 15-byte LoRaTap v0 header.

    Args:
      frequency_hz: LoRa center frequency in Hz (will be encoded BE).
      bandwidth_hz: 125000, 250000, or 500000.
      sf: spreading factor 7..12.
      sync_word: LoRa sync word (Meshtastic = 0x2B, LoRaWAN = 0x34).
      rssi_dbm: optional RSSI; 0.0 → N/A in the encoded header.
      snr_db: optional SNR; 0.0 → N/A.

    All multibyte fields are big-endian per the LoRaTap spec.
    """
    if not (7 <= sf <= 12):
        raise ValueError(f"SF {sf} out of LoRaTap range (7..12)")

    rssi_pkt = _rssi_to_loratap(rssi_dbm)
    snr      = _snr_to_loratap(snr_db)

    return struct.pack(
        ">"           # big-endian
        "B"           # lt_version
        "B"           # lt_padding
        "H"           # lt_length (= 15 for v0)
        "I"           # frequency Hz
        "B"           # bandwidth (in 125 kHz steps)
        "B"           # spreading factor
        "B"           # packet RSSI (255 = N/A)
        "B"           # max RSSI (we don't track; report 255)
        "B"           # current RSSI (idem)
        "B"           # SNR (raw, two's-complement / 4)
        "B",          # sync word
        0,                       # version
        0,                       # padding
        _LORATAP_V0_LEN,         # length
        frequency_hz & 0xFFFFFFFF,
        _bw_to_loratap(bandwidth_hz),
        sf,
        rssi_pkt,
        255,                     # max_rssi: not measured
        255,                     # current_rssi: not measured
        snr,
        sync_word & 0xFF,
    )


# ─────────────────────────────────────────────────────────────────────
# pcap file writer
# ─────────────────────────────────────────────────────────────────────

class PcapLoraTapWriter:
    """Write a classic libpcap file with DLT_LORATAP framing.

    Each ``write_packet`` call emits one packet record: a libpcap
    record header (timestamp + length + length) followed by a LoRaTap
    v0 header followed by the raw LoRa PHY payload bytes.

    Use as a context manager to ensure the file is closed properly::

        with PcapLoraTapWriter("capture.pcap") as w:
            for frame in frames:
                w.write_packet(payload, frequency_hz=915_000_000,
                                bandwidth_hz=250_000, sf=9, sync_word=0x2B)
    """

    def __init__(
        self,
        path: str | Path,
        snaplen: int = _DEFAULT_SNAPLEN,
    ) -> None:
        self._path = Path(path)
        self._snaplen = snaplen
        self._fp: Optional[BinaryIO] = None
        self._packets_written = 0

    def open(self) -> None:
        """Open the file and write the global pcap header."""
        if self._fp is not None:
            raise RuntimeError("already open")
        self._fp = self._path.open("wb")
        # Global pcap header — 24 bytes, native endianness (we pick LE).
        # struct format: magic(I), ver_major(H), ver_minor(H),
        # thiszone(i), sigfigs(I), snaplen(I), network(I)
        self._fp.write(struct.pack(
            "<IHHiIII",
            _PCAP_MAGIC_USEC_LE,
            2, 4,                # pcap version 2.4
            0,                   # thiszone — UTC
            0,                   # sigfigs — always 0
            self._snaplen,
            DLT_LORATAP,
        ))

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def __enter__(self) -> "PcapLoraTapWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def packets_written(self) -> int:
        return self._packets_written

    def write_packet(
        self,
        payload: bytes,
        frequency_hz: int,
        bandwidth_hz: int,
        sf: int,
        sync_word: int = 0x2B,
        rssi_dbm: float = 0.0,
        snr_db: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """Write one LoRa frame to the pcap.

        Args:
          payload: raw LoRa PHY payload bytes (after dewhitening, after
            CRC validation if you have it). For Meshtastic, this is the
            16-byte header + N-byte (still-encrypted) data — Wireshark
            dissectors decrypt downstream, so we keep the wire bytes.
          frequency_hz: center frequency for the LoRaTap header.
          bandwidth_hz: 125000 / 250000 / 500000.
          sf: spreading factor 7..12.
          sync_word: LoRa sync word (default 0x2B for Meshtastic).
          rssi_dbm, snr_db: optional radio metadata.
          timestamp: epoch seconds; defaults to time.time() at call.
        """
        if self._fp is None:
            raise RuntimeError("writer not open — use as context manager")

        if timestamp is None:
            timestamp = time.time()
        ts_sec  = int(timestamp)
        ts_usec = int((timestamp - ts_sec) * 1_000_000)

        loratap = build_loratap_header(
            frequency_hz, bandwidth_hz, sf, sync_word,
            rssi_dbm, snr_db,
        )
        record_data = loratap + payload
        record_len  = len(record_data)
        if record_len > self._snaplen:
            record_data = record_data[: self._snaplen]
            captured_len = self._snaplen
        else:
            captured_len = record_len

        # Per-packet header — 16 bytes:
        # ts_sec(I), ts_usec(I), incl_len(I), orig_len(I)
        self._fp.write(struct.pack(
            "<IIII", ts_sec, ts_usec, captured_len, record_len,
        ))
        self._fp.write(record_data)
        self._packets_written += 1


@contextmanager
def open_pcap(
    path: str | Path,
    snaplen: int = _DEFAULT_SNAPLEN,
) -> Iterator[PcapLoraTapWriter]:
    """Convenience wrapper: ``with open_pcap("foo.pcap") as w: ...``"""
    w = PcapLoraTapWriter(path, snaplen=snaplen)
    w.open()
    try:
        yield w
    finally:
        w.close()
