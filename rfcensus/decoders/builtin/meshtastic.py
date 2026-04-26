"""Meshtastic decoder.

Connects to a shared rtl_tcp fanout, runs the lazy coarse-FFT
LoRa pipeline in-process, and emits a ``DecodeEvent`` for every
Meshtastic packet whose CRC passes — whether or not it could be
decrypted.

# Why in-process and not subprocess

All of our other decoder builtins (rtl_433, rtlamr, direwolf,
multimon, rtl_ais) are subprocess wrappers — they shell out to a
binary that does the demod and they parse text/JSON off stdout.
Meshtastic is different: there is no stand-alone binary that does
LoRa→Meshtastic decryption from raw IQ. The closest thing in the
wild is gr-lora_sdr, which requires GNU Radio + a GUI flowgraph.

We have a complete in-process pipeline already: the
``lora_native`` C library (the lora_demod state machine) plus the
``meshtastic_native`` C library (AES-CTR decrypt, channel hash
matching) plus the ``LazyMultiPresetPipeline`` orchestrator. So
the cleanest integration is to instantiate that pipeline directly
inside ``run()`` and pump cu8 from a TCP socket connected to the
broker's rtl_tcp fanout.

# Concurrency model

``LazyMultiPresetPipeline.feed_cu8()`` is synchronous CPU-bound
work. To avoid blocking the event loop while the pipeline crunches
samples, the pump runs in a thread via ``asyncio.to_thread``. The
thread receives bytes from a synchronous socket (``RtlTcpSource``)
and feeds them to the pipeline; it puts ``DecodeEvent``s on a
``janus``-style threadsafe queue (we use ``asyncio.Queue`` reached
via ``loop.call_soon_threadsafe`` since adding a janus dep for one
queue isn't worth it).

The async ``run()`` coroutine just drains the queue and publishes
each event on the bus — that side is genuinely async (matches what
the validator/tracker do downstream).

# PSK loading

The decoder accepts ``psks`` in its DecoderConfig (deserialized
from site.toml as ``[decoders.meshtastic] psks = [...]``):

    [decoders.meshtastic]
    enabled = true
    psks = [
      { name = "LongFast",    psk_b64 = "1PG7OiApB1nwvP+rz05pAQ==" },
      { name = "MyPrivate",   psk_hex = "deadbeef..." },
    ]

The well-known default-channel PSK (MESH_DEFAULT_PSK) is always
loaded automatically so public traffic decrypts without any
configuration. Only add custom-channel PSKs if you have them.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import socket
import time
from datetime import datetime, timezone
from typing import Any

from rfcensus.decoders.base import (
    DecoderAvailability,
    DecoderBase,
    DecoderCapabilities,
    DecoderResult,
    DecoderRunSpec,
)
from rfcensus.decoders.lazy_pipeline import LazyMultiPresetPipeline
from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
from rfcensus.decoders.passband_detector import DetectorConfig
from rfcensus.events import DecodeEvent
from rfcensus.utils.iq_source import RtlTcpSource, RtlSdrConfig
from rfcensus.utils.logging import get_logger
from rfcensus.utils.meshtastic_region import (
    PRESETS,
    REGIONS,
    PresetSlot,
    default_slot,
    enumerate_all_slots_in_passband,
)

log = get_logger(__name__)


# Meshtastic uses two ISM bands worldwide that fall inside our
# decoder's coverage. EU 868 MHz (863-870), US 915 MHz (902-928),
# and KR/CN/JP/TW variants. We declare the union — the per-region
# slot enumeration filters down to what's actually allowed at run
# time based on the dongle's tuned center frequency.
_MESHTASTIC_FREQ_RANGES: tuple[tuple[int, int], ...] = (
    (433_000_000, 435_000_000),    # EU 433 ISM (rare)
    (863_000_000, 870_000_000),    # EU 868 ISM
    (902_000_000, 928_000_000),    # US 915 ISM
)


class MeshtasticBuiltinDecoder(DecoderBase):
    """Decode Meshtastic LoRa packets from a shared rtl_tcp fanout."""

    capabilities = DecoderCapabilities(
        name="meshtastic",
        protocols=["meshtastic"],
        freq_ranges=_MESHTASTIC_FREQ_RANGES,
        # 1 MS/s is enough for any single-preset capture and matches
        # what most Meshtastic stations transmit at. 2.4 MS/s is the
        # preferred rate when sharing a slot with rtl_433/rtlamr —
        # gives us the full passband to enumerate every channel.
        min_sample_rate=1_000_000,
        preferred_sample_rate=2_400_000,
        requires_exact_sample_rate=False,
        requires_exclusive_dongle=False,    # we use the rtl_tcp fanout
        external_binary="",                  # in-process, no binary
        cpu_cost="moderate",
        # Off by default until users opt in (the lazy pipeline still
        # spawns short-lived LoRa decoders on every above-threshold
        # passband transient — fine, but worth being explicit).
        opt_in=False,
        description=(
            "Decodes Meshtastic LoRa packets in-process via the lazy "
            "coarse-FFT pipeline. Decrypts the public default channel "
            "automatically; site config can supply additional PSKs."
        ),
    )

    async def check_available(self) -> DecoderAvailability:
        """No external binary; just verify our native libs loaded.

        Importing ``meshtastic_native`` and ``lora_native`` at module
        load time would already have raised ``OSError`` if the .so
        files weren't built. Here we double-check by trying to
        construct a throwaway decoder + add one channel."""
        try:
            mesh = MeshtasticDecoder("LONG_FAST")
            mesh.add_channel(name="LongFast", psk=b"\x01")
            return DecoderAvailability(
                name=self.name,
                available=True,
                binary_path="(in-process)",
            )
        except Exception as exc:    # pragma: no cover - load-time issue
            return DecoderAvailability(
                name=self.name,
                available=False,
                reason=(
                    f"meshtastic native lib not loadable: {exc!r}. "
                    "Build with `cd rfcensus/decoders/_native/meshtastic "
                    "&& make` and `cd rfcensus/decoders/_native/lora "
                    "&& make`."
                ),
            )

    async def run(self, spec: DecoderRunSpec) -> DecoderResult:
        result = DecoderResult(name=self.name)
        lease = spec.lease
        endpoint = lease.endpoint()
        if endpoint is None:
            result.errors.append(
                "meshtastic decoder requires a SHARED rtl_tcp lease "
                "but got an exclusive one"
            )
            result.ended_reason = "wrong_lease_type"
            return result

        host, port = endpoint
        sample_rate = spec.sample_rate

        # Build the Meshtastic decryptor with all configured PSKs
        # plus the public default channel.
        decoder_opts = spec.decoder_options.get("meshtastic", {})
        try:
            mesh = self._build_mesh_decoder(decoder_opts)
        except Exception as exc:
            result.errors.append(f"failed to build mesh decoder: {exc!r}")
            result.ended_reason = "config_error"
            return result

        # Determine which slots to monitor based on the tuner's
        # center frequency and sample rate. We use --slots all by
        # default for the integrated decoder so users see every
        # channel, not just LongFast/MediumFast — the lazy pipeline
        # makes the wide-enumeration cheap.
        region_code = decoder_opts.get("region", "US")
        slots_mode = decoder_opts.get("slots", "all")
        try:
            candidate_slots = self._enumerate_slots(
                region_code=region_code,
                slots_mode=slots_mode,
                center_freq_hz=spec.freq_hz,
                sample_rate_hz=sample_rate,
            )
        except Exception as exc:
            result.errors.append(
                f"slot enumeration failed for region={region_code}: {exc!r}"
            )
            result.ended_reason = "config_error"
            return result

        if not candidate_slots:
            # No Meshtastic activity expected at this center freq;
            # bail cleanly rather than burning CPU on an always-quiet
            # passband.
            log.info(
                "meshtastic: no slots in passband around %.3f MHz "
                "(region=%s); decoder will idle",
                spec.freq_hz / 1e6, region_code,
            )
            result.ended_reason = "no_slots_in_passband"
            return result

        # Build the lazy pipeline. Detector config is a passable
        # default — the cumsum-vectorized hot path handles 1-100
        # slots without sweat.
        det_cfg = DetectorConfig(
            sample_rate_hz=sample_rate,
            center_freq_hz=spec.freq_hz,
            fft_size=512,
            hop_samples=256,
            bootstrap_frames=200,
            drain_frames=200,
            trigger_threshold_db=8.0,
            release_threshold_db=4.0,
        )
        pipe = LazyMultiPresetPipeline(
            sample_rate_hz=sample_rate,
            center_freq_hz=spec.freq_hz,
            candidate_slots=candidate_slots,
            mesh=mesh,
            detector_config=det_cfg,
            ring_buffer_ms=300.0,
        )

        # Channel-name lookup table for payload-formatting.
        channel_names: list[str] = []
        for k in PRESETS:
            channel_names.append(PRESETS[k].display_name)

        # Set up the cross-thread event delivery: the pump thread
        # produces dict-payload tuples; the async loop publishes them.
        loop = asyncio.get_running_loop()
        event_q: asyncio.Queue[DecodeEvent | None] = asyncio.Queue(maxsize=1024)
        stop_evt = asyncio.Event()

        # Track the dongle id for the DecodeEvent (some leases don't
        # always carry one — fall back to "")
        dongle_id = lease.dongle.id if lease.dongle else ""

        decodes_emitted = 0
        deadline = (
            time.monotonic() + spec.duration_s
            if spec.duration_s is not None else None
        )

        def _pump_blocking() -> None:
            """Synchronous pump: connect, read cu8, feed pipeline,
            queue resulting DecodeEvents back to the asyncio loop.

            Runs in a worker thread via ``asyncio.to_thread``."""
            chunk_size = 1 << 16    # 64 KB ≈ 13 ms at 2.4 MS/s
            cfg = RtlSdrConfig(
                freq_hz=spec.freq_hz,
                sample_rate_hz=sample_rate,
                gain_tenths_db=-1,    # AGC
            )
            try:
                source = RtlTcpSource(host, port, cfg,
                                        chunk_size=chunk_size)
            except Exception as exc:
                log.error("meshtastic: rtl_tcp connect failed: %r", exc)
                loop.call_soon_threadsafe(event_q.put_nowait, None)
                return

            try:
                while not stop_evt.is_set():
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    chunk = source.read(chunk_size)
                    if not chunk:
                        # Fanout disconnected; bail.
                        break
                    pipe.feed_cu8(chunk)
                    for pkt in pipe.pop_packets():    # dedup=True default
                        de = self._packet_to_event(
                            pkt=pkt,
                            session_id=spec.session_id,
                            dongle_id=dongle_id,
                            channel_names=channel_names,
                        )
                        # Delivery to async side. Use call_soon_threadsafe
                        # so the queue.put_nowait runs on the event loop.
                        loop.call_soon_threadsafe(event_q.put_nowait, de)
            except Exception as exc:
                log.error("meshtastic pump exception: %r", exc, exc_info=True)
            finally:
                try:
                    source.close()
                except Exception:
                    pass
                # Sentinel signals the consumer side that the pump is done.
                loop.call_soon_threadsafe(event_q.put_nowait, None)

        # Launch the pump in a worker thread. We don't await it
        # directly — instead we pull events off the queue until we
        # see the sentinel (None), then await the pump task to
        # surface any exception.
        pump_task = asyncio.create_task(asyncio.to_thread(_pump_blocking))

        try:
            while True:
                event = await event_q.get()
                if event is None:
                    # Pump finished (clean exit, EOF, or error).
                    break
                await spec.event_bus.publish(event)
                decodes_emitted += 1
        except asyncio.CancelledError:
            # Scheduler is shutting us down — signal the pump to stop
            # and let it drain.
            stop_evt.set()
            try:
                await asyncio.wait_for(pump_task, timeout=2.0)
            except asyncio.TimeoutError:
                log.warning("meshtastic pump did not stop within 2s")
            raise
        finally:
            stop_evt.set()
            # Pump may already be done (we got the sentinel); wait
            # briefly to surface its exception if any.
            try:
                await asyncio.wait_for(pump_task, timeout=2.0)
            except asyncio.TimeoutError:
                log.warning("meshtastic pump did not finish within 2s")
            except Exception as exc:
                # Already logged inside the pump; ensure it surfaces
                # in the result.
                result.errors.append(f"pump: {exc!r}")

        result.decodes_emitted = decodes_emitted
        # Surface lazy-pipeline stats for the per-decoder summary.
        ls = pipe.lazy_stats
        log.info(
            "meshtastic: %d decodes emitted "
            "(spawns=%d, activations=%d, decoded=%d, decrypted=%d)",
            decodes_emitted,
            ls.decoders_spawned, ls.slot_activations,
            ls.packets_decoded, ls.packets_decrypted,
        )
        return result

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _build_mesh_decoder(
        self, opts: dict[str, Any],
    ) -> MeshtasticDecoder:
        """Construct a MeshtasticDecoder pre-loaded with the public
        default channel PSK plus any custom PSKs from site config.

        ``opts`` is the per-band override dict from
        ``DecoderRunSpec.decoder_options['meshtastic']``. Recognized
        keys:

          • ``psks`` — list of {name, psk_b64|psk_hex|psk_short}
            entries. Each becomes a channel in the mesh decoder's
            channel table; on incoming packets the channel hash is
            matched against the table to pick the right PSK.

        We also load all 9 preset display names as zero-payload
        "marker" channels so the channel-hash matcher can identify
        which preset a packet was sent on (the Meshtastic protocol
        encodes the preset name in the channel hash).
        """
        # Decoder is constructed with a default preset; it doesn't
        # actually constrain decryption — channel matching happens
        # by hash, not preset.
        mesh = MeshtasticDecoder("LONG_FAST")

        # The Meshtastic channel hash is derived from (channel_name,
        # PSK) — so the default-channel hash for LongFast differs
        # from the default-channel hash for MediumFast even though
        # both use MESH_DEFAULT_PSK. Pre-populate the channel table
        # with one entry per preset using the preset's display name
        # (e.g. "LongFast", "MediumFast") and the short-PSK form
        # b"\x01" (which the C library expands to MESH_DEFAULT_PSK).
        # This lets us decrypt the public default channel for any
        # preset without per-preset config from the user.
        seen_names: set[str] = set()
        for preset_key, preset in PRESETS.items():
            if preset.display_name in seen_names:
                continue
            try:
                mesh.add_channel(
                    name=preset.display_name,
                    psk=b"\x01",       # short-PSK → MESH_DEFAULT_PSK
                    is_primary=False,
                )
                seen_names.add(preset.display_name)
            except Exception as exc:
                log.warning(
                    "meshtastic: failed to add default channel for "
                    "preset %r: %r", preset_key, exc,
                )

        # User-supplied PSKs from [decoders.meshtastic.psks]
        for entry in opts.get("psks", []) or []:
            if not isinstance(entry, dict):
                log.warning("meshtastic: psk entry is not a dict: %r", entry)
                continue
            name = str(entry.get("name", ""))
            psk_bytes = self._psk_entry_to_bytes(entry)
            if psk_bytes is None:
                log.warning(
                    "meshtastic: psk entry %r missing psk_b64/psk_hex/"
                    "psk_short — skipping", name,
                )
                continue
            try:
                mesh.add_channel(name=name, psk=psk_bytes,
                                 is_primary=False)
                log.info(
                    "meshtastic: loaded PSK for channel %r "
                    "(%d-byte key)", name, len(psk_bytes),
                )
            except Exception as exc:
                log.warning(
                    "meshtastic: add_channel(%r) failed: %r",
                    name, exc,
                )

        return mesh

    @staticmethod
    def _psk_entry_to_bytes(entry: dict[str, Any]) -> bytes | None:
        """Convert one PSK config entry to raw bytes.

        Accepts three forms:
          • ``psk_b64`` (str) — base64-encoded 16- or 32-byte key
          • ``psk_hex`` (str) — hex-encoded key
          • ``psk_short`` (int) — 1-255, expands to MESH_DEFAULT_PSK
            with last byte bumped (Meshtastic's "short PSK" form
            used for preconfigured channels)
        """
        if "psk_b64" in entry:
            return base64.b64decode(str(entry["psk_b64"]))
        if "psk_hex" in entry:
            return bytes.fromhex(str(entry["psk_hex"]))
        if "psk_short" in entry:
            n = int(entry["psk_short"])
            if not 1 <= n <= 255:
                raise ValueError(
                    f"psk_short must be 1..255, got {n}"
                )
            return bytes([n])
        return None

    @staticmethod
    def _enumerate_slots(
        region_code: str,
        slots_mode: str,
        center_freq_hz: int,
        sample_rate_hz: int,
    ) -> list[PresetSlot]:
        """Pick the (preset, slot) pairs to monitor.

        ``slots_mode`` is "default" (each preset's default slot only)
        or "all" (every (preset, slot) in the dongle's instantaneous
        passband). Default is "all" — the lazy pipeline makes wide
        enumeration nearly free at idle.

        We always ALSO include each preset's default-channel slot
        when it falls anywhere within the instantaneous passband
        (using a looser ±0.45 × sample_rate filter than
        enumerate_all_slots_in_passband's tight edge_guard). This
        ensures that a user who tunes to a known default-channel
        frequency (e.g. MEDIUM_FAST at 913.125 MHz) gets coverage
        even if the slot lies within the resampler's edge-rolloff
        guard band that enumerate_all conservatively excludes —
        the LoRa demodulator copes with edge-band signals fine via
        per-slot DDC + resample.
        """
        if region_code not in REGIONS:
            raise ValueError(
                f"unknown region {region_code!r}; "
                f"valid: {sorted(REGIONS)}"
            )

        # First pass: each preset's default-channel slot, filtered to
        # the dongle's instantaneous passband. Loose passband filter:
        # ±0.45 × sample_rate (anti-aliasing rolloff usually ends
        # around the ±0.4 mark, so 0.45 catches default-channel
        # signals at the slightly-rolled-off edges).
        half_bw = int(sample_rate_hz * 0.45)
        defaults_in_band: list[PresetSlot] = []
        for preset_key in PRESETS:
            try:
                s = default_slot(region_code, preset_key)
            except Exception:
                continue
            if abs(s.freq_hz - center_freq_hz) <= half_bw:
                defaults_in_band.append(s)

        if slots_mode == "default":
            return defaults_in_band

        # slots_mode == "all": union the strict-passband enumeration
        # with the default slots (for the case where a default-
        # channel freq lies within the loose-filter band but outside
        # the strict-filter band).
        strict = enumerate_all_slots_in_passband(
            region_code, center_freq_hz, sample_rate_hz,
        )
        seen = {(s.preset.key, s.freq_hz) for s in strict}
        for s in defaults_in_band:
            if (s.preset.key, s.freq_hz) not in seen:
                strict.append(s)
                seen.add((s.preset.key, s.freq_hz))
        return strict

    @staticmethod
    def _packet_to_event(
        pkt,
        session_id: int,
        dongle_id: str,
        channel_names: list[str],
    ) -> DecodeEvent:
        """Convert a ``PipelinePacket`` to a ``DecodeEvent``.

        We always emit when CRC is OK, even if decryption failed.
        That gives the storage layer two tiers of confidence:

          • CRC ok + decrypted → ``decoder_confidence = 1.0``,
            payload contains structured Meshtastic fields
            (from_node, to_node, channel_name, port, plaintext)
          • CRC ok + not decrypted → ``decoder_confidence = 0.6``,
            payload contains the ciphertext as hex plus the
            channel hash for later forensic correlation
        """
        lp = pkt.lora
        slot = pkt.slot
        mp = pkt.mesh

        crc_ok = lp.crc_ok
        decrypted = bool(mp and mp.decrypted)

        payload: dict[str, Any] = {
            "preset": slot.preset.key,
            "preset_display_name": slot.preset.display_name,
            "slot_freq_hz": int(slot.freq_hz),
            "bandwidth_hz": int(slot.preset.bandwidth_hz),
            "sf": int(slot.preset.sf),
            "crc_ok": bool(crc_ok),
            "raw_lora_hex": lp.payload.hex(),
            "raw_lora_len": int(lp.payload_len),
            "sample_offset": int(lp.sample_offset),
        }

        if mp is not None:
            payload["channel_hash"] = int(mp.channel_hash)
            payload["channel_index"] = int(mp.channel_index)
            payload["decrypted"] = decrypted
            if decrypted:
                # Structured Meshtastic fields. Note the C struct uses
                # `to`, `from_node`, `id` (NOT to_node / from_node /
                # packet_id) — match the names in DecodedPacket.
                payload["from_node"] = int(mp.from_node)
                payload["to_node"] = int(mp.to)
                payload["packet_id"] = int(mp.id)
                payload["hop_limit"] = int(mp.hop_limit)
                payload["hop_start"] = int(mp.hop_start)
                payload["want_ack"] = bool(mp.want_ack)
                payload["via_mqtt"] = bool(mp.via_mqtt)
                payload["plaintext_hex"] = mp.plaintext.hex()
                payload["plaintext_len"] = len(mp.plaintext)
                # Try to extract a UTF-8 text preview if this looks
                # like a TEXT_MESSAGE_APP packet (port 1):
                #   varint port-tag (0x08), port (1), payload-tag (0x12),
                #   length, bytes...
                pt = mp.plaintext
                if (len(pt) > 4 and pt[0] == 0x08
                        and pt[1] == 0x01 and pt[2] == 0x12):
                    paylen = pt[3]
                    if 4 + paylen <= len(pt):
                        try:
                            payload["text"] = pt[4:4+paylen].decode(
                                "utf-8", errors="replace",
                            )
                        except Exception:
                            pass
                # device_id used by tracker.handle_decode for emitter
                # tracking — Meshtastic uses node IDs as natural
                # device identifiers.
                payload["_device_id"] = (
                    f"meshtastic:0x{int(mp.from_node):08X}"
                )

        # Validation hint: CRC-failed never confident, decrypted very
        # confident, just-CRC-ok somewhere in between.
        confidence = 0.0
        if crc_ok and decrypted:
            confidence = 1.0
        elif crc_ok:
            confidence = 0.6
        else:
            confidence = 0.2

        return DecodeEvent(
            session_id=session_id,
            decoder_name="meshtastic",
            protocol="meshtastic",
            dongle_id=dongle_id,
            freq_hz=int(slot.freq_hz),
            rssi_dbm=(float(lp.rssi_db)
                       if lp.rssi_db != 0.0 else None),
            snr_db=(float(lp.snr_db)
                     if lp.snr_db != 0.0 else None),
            payload=payload,
            raw_hex=lp.payload.hex(),
            decoder_confidence=confidence,
            timestamp=datetime.now(timezone.utc),
        )
