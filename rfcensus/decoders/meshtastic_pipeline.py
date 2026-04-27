"""Multi-preset Meshtastic pipeline — one IQ stream → N decoders → decrypt.

Wraps ``LoraDecoder`` (PHY) and ``MeshtasticDecoder`` (MAC + decrypt)
into a single object that can decode every Meshtastic preset whose
slot fits in a dongle's passband, simultaneously.

Architecture (from earlier design discussion):
  • ONE dongle = ONE IQ stream
  • Within that stream, N preset slots are visible
  • Spawn one ``LoraDecoder`` per slot, each with its own
    ``mix_freq_hz`` to translate that slot's signal to baseband
  • Feed the SAME bytes to every decoder
  • All decoded packets attempt-decrypt against ONE shared
    ``MeshtasticDecoder`` channel table
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from rfcensus.decoders.lora_native import (
    LoraConfig, LoraDecoder, LoraPacket, LoraStats, ldro_required as _ldro_required,
)
from rfcensus.decoders.meshtastic_native import (
    DecodedPacket, MeshtasticDecoder,
)
from rfcensus.utils.meshtastic_region import PresetSlot


# Sync word for all Meshtastic presets. NOT 0x12 (generic LoRa) —
# Meshtastic firmware sets this to 0x2B in RadioLibInterface.h:84.
_MESHTASTIC_SYNC_WORD = 0x2B


@dataclass
class PipelinePacket:
    """One packet as it falls out of the pipeline.

    Carries the slot it came from so callers know which preset
    decoded it. The Meshtastic-side ``mesh`` field is populated only
    when the LoRa CRC passed; if ``mesh.decrypted`` is True, the
    plaintext is in ``mesh.plaintext``.
    """
    slot: PresetSlot
    lora: LoraPacket
    mesh: Optional[DecodedPacket]   # None if LoRa CRC failed

    @property
    def crc_ok(self) -> bool:
        return self.lora.crc_ok

    @property
    def decrypted(self) -> bool:
        return self.mesh is not None and self.mesh.decrypted


class MultiPresetPipeline:
    """Run multiple LoRa decoders in parallel against one IQ stream.

    Each ``slot`` is decoded by its own ``LoraDecoder`` instance with
    a per-slot downmix to baseband. All instances share the same
    ``MeshtasticDecoder`` for channel matching + decryption.

    Memory: ~100 KB per LoRa decoder + ~20 KB for the MeshtasticDecoder
    channel table. For 9 presets, ~1 MB total.

    CPU: each decoder spends most of its time in DETECT, doing ~one
    FFT per ~half-symbol-window. At Pi 5 speeds this is well under 5%
    of one core per decoder, so 9 decoders = ~30-40% of one core,
    leaving plenty of headroom for the IQ source to pump data.
    """

    def __init__(
        self,
        slots: list[PresetSlot],
        sample_rate_hz: int,
        center_freq_hz: int,
        mesh: MeshtasticDecoder,
    ) -> None:
        if not slots:
            raise ValueError("MultiPresetPipeline needs at least one slot")
        self._slots = list(slots)
        self._sample_rate_hz = sample_rate_hz
        self._center_freq_hz = center_freq_hz
        self._mesh = mesh
        self._decoders: list[tuple[PresetSlot, LoraDecoder]] = []

        for slot in slots:
            # The decoder's mix_freq translates the slot's RF frequency
            # to baseband. Sign convention from lora_demod.h:
            #   mix_freq = (capture_freq - lora_signal_freq)
            # so a positive mix_freq pulls a signal that's BELOW the
            # tuner up to baseband.
            mix_freq = center_freq_hz - slot.freq_hz
            cfg = LoraConfig(
                sample_rate_hz=sample_rate_hz,
                bandwidth=slot.preset.bandwidth_hz,
                sf=slot.preset.sf,
                sync_word=_MESHTASTIC_SYNC_WORD,
                mix_freq_hz=mix_freq,
                # v0.7.16: per Semtech AN1200.13, LDRO required when
                # symbol time ≥ 16 ms. The previous test missed
                # LongModerate (SF11/125k, Tsym = 16.4 ms), which
                # SIGILLs the decoder. The helper covers all such
                # combinations correctly — including hypothetical
                # custom regions that use SF12 at 250 kHz.
                ldro=_ldro_required(slot.preset.sf, slot.preset.bandwidth_hz),
            )
            self._decoders.append((slot, LoraDecoder(cfg)))

    @property
    def slots(self) -> list[PresetSlot]:
        return list(self._slots)

    @property
    def sample_rate_hz(self) -> int:
        return self._sample_rate_hz

    @property
    def center_freq_hz(self) -> int:
        return self._center_freq_hz

    @property
    def mesh(self) -> MeshtasticDecoder:
        return self._mesh

    def feed_cu8(self, samples: bytes) -> int:
        """Pump a chunk of cu8 IQ to every decoder.

        Returns the total number of packets decoded across all
        decoders during this call (those packets are also queued for
        ``pop_packets``).

        The same bytes are passed to each decoder. The C-side ctypes
        layer copies from the Python buffer per call; for N decoders
        that's N copies but the cost is dominated by the actual demod
        FFTs, not the buffer copy. On Pi 5 this is fine for 9 decoders.
        If we ever hit a CPU wall we can refactor to share a single
        cf32 buffer across decoders (one cu8→cf32 conversion + one
        ctypes pass per decoder).
        """
        total = 0
        for _, dec in self._decoders:
            total += dec.feed_cu8(samples)
        return total

    def pop_packets(
        self,
        dedup: bool = True,
        dedup_offset_tolerance: int | None = None,
    ) -> Iterator[PipelinePacket]:
        """Drain queued packets from every decoder, attempt decrypt
        for those with valid CRC, yield with per-pipeline dedup.

        WHY DEDUP:
        Adjacent slot decoders frequently catch the SAME physical
        transmission. LoRa demod has high CFO tolerance — for BW=250
        kHz, ±62 kHz of CFO is decodable, and Meshtastic slot spacing
        in US (BW=250 case) is exactly 250 kHz, so any single
        transmission tends to be decoded by the slot it actually used
        AND by 1-2 adjacent slots. Without dedup, one packet appears
        2-3× in the output stream.

        DEDUP STRATEGY:
        Two packets are "the same" if they share:
          • same payload bytes, AND
          • sample_offset within ``dedup_offset_tolerance`` samples

        v0.7.6: tolerance default raised from 16 samples (~7 µs) to
        sample_rate_hz / 5 (~200 ms). Real parallel-slot duplicates
        land within a few LoRa symbols of each other (multiple ms),
        not microseconds. See LazyMultiPresetPipeline.pop_packets
        for the same fix and full reasoning.

        Among duplicates, we keep the slot with the highest reported
        RSSI as the most likely true transmission frequency.

        Set ``dedup=False`` to see every decoder's view (useful for
        debugging slot attribution). Iteration is chronological by
        sample_offset across all decoders.
        """
        # v0.7.6: derive tolerance from sample rate if not specified
        if dedup_offset_tolerance is None:
            dedup_offset_tolerance = self._sample_rate_hz // 5
        # Collect everything first so we can sort + dedup globally.
        # Per-pop volumes are small (typically <10 packets per
        # iteration), so the O(N) memory pass is fine.
        all_packets: list[PipelinePacket] = []
        for slot, dec in self._decoders:
            for lp in dec.pop_packets():
                mesh_pkt = (self._mesh.decode(lp.payload)
                              if lp.crc_ok else None)
                all_packets.append(PipelinePacket(
                    slot=slot, lora=lp, mesh=mesh_pkt,
                ))

        if not dedup:
            all_packets.sort(key=lambda p: p.lora.sample_offset)
            yield from all_packets
            return

        # Dedup: cluster by payload identity + offset proximity.
        # We sort by sample_offset, then walk the list grouping
        # contiguous packets that match. Within each group, pick the
        # one with the highest RSSI (or, if RSSI tied, the one whose
        # slot frequency is in the middle of the group's slot span —
        # most likely the true frequency).
        all_packets.sort(key=lambda p: p.lora.sample_offset)
        groups: list[list[PipelinePacket]] = []
        for pkt in all_packets:
            placed = False
            for grp in groups:
                head = grp[0]
                if (pkt.lora.payload == head.lora.payload
                    and abs(pkt.lora.sample_offset
                              - head.lora.sample_offset)
                          <= dedup_offset_tolerance):
                    grp.append(pkt)
                    placed = True
                    break
            if not placed:
                groups.append([pkt])

        for grp in groups:
            if len(grp) == 1:
                yield grp[0]
                continue
            # v0.7.8: RSSI is real now (v0.7.7 fixed C-side
            # computation), so just pick max(rssi). Adjacent slots
            # picking up the same transmission via CFO tolerance see
            # 3-15 dB lower RSSI than the slot whose center matches
            # the actual transmission frequency.
            best = max(grp, key=lambda p: p.lora.rssi_db)
            ties = [p for p in grp
                    if p.lora.rssi_db == best.lora.rssi_db]
            if len(ties) > 1:
                ties_sorted = sorted(ties, key=lambda p: p.slot.freq_hz)
                best = ties_sorted[len(ties_sorted) // 2]
            yield best

    def stats(self) -> dict[str, LoraStats]:
        """Per-preset decoder stats keyed by preset key (e.g. 'LONG_FAST')."""
        return {slot.preset.key: dec.stats() for slot, dec in self._decoders}

    def close(self) -> None:
        # Decoders free themselves on __del__, but explicit close lets
        # the caller release native resources deterministically.
        self._decoders.clear()
