"""LazyMultiPresetPipeline — coarse-FFT-gated lazy decoder spawning.

This is the v0.7.2 successor to ``MultiPresetPipeline``. Instead of
eagerly running one decoder per (preset, slot) pair, it uses a
``PassbandDetector`` to identify which slot frequencies have RF
energy and spawns LoRa decoders only for those active slots.

Architecture:

    IQ → IqRingBuffer (lookback storage)
       → PassbandDetector (wide FFT, per-slot energy + state machine)
       → SlotEvent (activate/deactivate)
       → spawn/teardown LoraDecoders for the active slot
       → LoraPacket → MeshtasticDecoder → PipelinePacket

Per slot activation we spawn ALL presets matching the slot's BW (so
e.g. when the BW=250 grid at 913.125 MHz becomes active, we spawn 5
decoders: SF7, SF8, SF9, SF10, SF11 all at that frequency). Each
decoder receives the LOOKBACK IQ from the ring buffer first (so it can
see the preamble that triggered the detection), then live IQ until
deactivation.

The lookback amount is chosen per BW — slow presets (SF12/BW125) need
~260ms lookback to see a full preamble; fast presets need only a few
ms. To keep the ring buffer small we use a single lookback equal to
the slowest preset's preamble duration (260ms at SF12/BW125).
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np    # v0.7.11: needed for shared-channelizer fanout buffers

from rfcensus.decoders.lora_native import (
    LoraConfig, LoraDecoder, LoraStats,
)
from rfcensus.decoders.meshtastic_native import MeshtasticDecoder
from rfcensus.decoders.meshtastic_pipeline import PipelinePacket
from rfcensus.decoders.passband_detector import (
    DetectorConfig, PassbandDetector, SlotEvent,
)
# v0.7.11: shared channelizer + multi-SF blind probe.
from rfcensus.decoders.shared_channelizer import SharedChannelizer
from rfcensus.decoders.blind_probe import BlindProbe
from rfcensus.utils.iq_ring import IqRingBuffer
from rfcensus.utils.meshtastic_region import (
    PRESETS, PresetSlot, REGIONS,
)


# Sync word for all Meshtastic presets.
_MESHTASTIC_SYNC_WORD = 0x2B


@dataclass
class _ActiveSlot:
    """Bookkeeping for a slot that the detector has activated.

    We hold one ``_ActiveSlot`` per (slot_freq_hz, bandwidth_hz) the
    detector has fired on. Each contains one LoraDecoder per preset
    matching that bandwidth.

    ``next_sample_offset`` is the global offset of the next sample
    each decoder is ready to consume. Used to slice the current
    feed_cu8 chunk so a decoder spawned mid-chunk doesn't get fed
    samples that were already in its lookback.

    ``feed_start_offset`` is the global INPUT-sample offset of the
    very FIRST cu8 sample fed to each decoder (= activation offset
    minus lookback). Used to convert the C decoder's locally-
    referenced ``sample_offset`` (which is in OUTPUT/post-resampler
    samples and counts from 0 at decoder creation) back to absolute
    capture-stream INPUT samples, so packets can be deduped across
    slots and so the user-visible offset means something meaningful.
    """
    freq_hz: int
    bandwidth_hz: int
    activated_sample_offset: int      # detector fired at this offset
    next_sample_offset: int           # offset of next sample to feed
    feed_start_offset: int            # global input offset of decoder's sample 0
    # Map preset_key → LoraDecoder for the presets we spawned
    decoders: dict[str, LoraDecoder] = field(default_factory=dict)
    # Map preset_key → PresetSlot (for emitting PipelinePacket later)
    slot_metadata: dict[str, PresetSlot] = field(default_factory=dict)
    # v0.7.7: SF racing — when we spawn multiple SFs at the same slot
    # (e.g. all 5 BW=250 presets), kill the losers as soon as one
    # detects a preamble. Saves ~80% of CPU on BW=250 activations
    # because 4-of-5 decoders are running the chirp detector against
    # a signal that doesn't match their SF — pure waste.
    #
    # ``race_resolved`` flips True after the racing window closes
    # (a winner was determined and losers killed). Once True we
    # don't re-evaluate. See _maybe_resolve_race for the
    # condition-based resolution mechanism.
    # v0.7.11: per-slot channelizer. One mix+resample pass shared
    # across all decoders at this (freq, bw) — bit-exact to each
    # decoder doing its own mix+resamp internally, but only paid
    # once instead of once per decoder. None for slots that can't
    # use sharing (e.g. only one decoder, or BW > sample_rate).
    channelizer: object = None    # SharedChannelizer | None
    # Reusable scratch buffer for channelizer output (interleaved
    # float32). Allocated lazily; grown on demand to avoid
    # per-feed numpy allocation.
    _baseband_scratch: object = None    # np.ndarray | None
    race_resolved: bool = False
    # v0.7.13: per-slot probe lifecycle. The probe now runs PERIODICALLY
    # while the slot is active (not just once at activate). Two outcomes
    # we care about:
    #   1. probe fires on a chirp → keep decoders alive, refresh
    #      last_positive_probe_offset
    #   2. probe doesn't fire for M ms → REAP all decoders. Slot stays
    #      active in the detector but our decoder pool is empty until
    #      probe fires again, at which point we re-spawn.
    #
    # `probe` is owned per-slot (built once at first activate, reused
    # across reap/respawn cycles since the candidate SF list doesn't
    # change for a given slot freq+bw).
    #
    # `probe_baseband` is a sliding-window buffer of the most recent
    # baseband samples — concretely the last (probe_window_samples)
    # samples of channelizer output. Probe scans this on each tick.
    #
    # `next_probe_offset` is the input-sample offset at which we'll
    # next run a probe scan (computed = last_probe_offset +
    # probe_interval_samples).
    #
    # `last_positive_probe_offset` is the input-sample offset of the
    # most recent probe-fire. If (now - last_positive) > reap_after,
    # we kill decoders.
    #
    # `state` ∈ {"spawned", "reaped"}. Reaped means decoders dict is
    # empty; we still poll probe to decide when to re-spawn.
    probe: object = None              # BlindProbe | None
    probe_baseband: object = None     # np.ndarray (rolling) | None
    next_probe_offset: int = 0
    last_positive_probe_offset: int = 0
    state: str = "spawned"            # "spawned" | "reaped"
    # v0.7.13 commit 1c: hysteretic pin for hot slots.
    #
    # We track recent probe-positive rate over a rolling window
    # (default 10 seconds). If the rate exceeds PIN_HIGH (15%), the
    # slot becomes "pinned" — the periodic-reap path skips it. If
    # the rate later drops below PIN_LOW (5%), unpin and resume
    # normal reap behavior. The 5/15% hysteresis prevents flapping
    # for slots near the boundary.
    #
    # ``probe_rate_ewma`` is an exponentially-weighted moving
    # average of probe outcomes (1.0 = fired, 0.0 = didn't fire).
    # Smoother than a sliding-window count: no early-window edge
    # spike, no fixed-cost trim per probe. The pin/unpin
    # decisions test this value against the hysteresis thresholds.
    # ``probe_history`` is now just a counter for the warmup
    # gating (we ignore the rate for the first N probes so cold-
    # start doesn't transient-pin).
    probe_rate_ewma: float = 0.0
    probe_history_count: int = 0
    pinned: bool = False


@dataclass
class LazyPipelineStats:
    """Cumulative stats across the lazy pipeline's lifetime."""
    detector_frames: int = 0
    slot_activations: int = 0
    slot_deactivations: int = 0
    decoders_spawned: int = 0
    decoders_torn_down: int = 0
    packets_decoded: int = 0
    packets_decrypted: int = 0
    # v0.7.7: SF racing wins (= losing decoders culled mid-flight).
    # racing_wins is per-event ("a slot resolved with N losers
    # killed"); racing_losers_killed is the cumulative count of
    # decoders ended early. racing_unresolved counts slot
    # deactivations where NO decoder ever detected a preamble (likely
    # a false-positive detector activation — useful diagnostic).
    racing_wins: int = 0
    racing_losers_killed: int = 0
    racing_unresolved: int = 0
    # v0.7.7: ring-buffer overflow counter. Incremented every time
    # the IqRingBuffer.write call had to drop a prefix because the
    # incoming chunk exceeded ring capacity. NON-zero means we are
    # losing samples, which means we are missing packets. Surfaced
    # in the standalone tool's summary so the user knows when their
    # CPU budget is exhausted.
    ring_overflows: int = 0
    samples_dropped: int = 0
    # v0.7.11/12: blind probe diagnostics. probe_decisions = total
    # number of probe scans (= activations where probe ran). Of
    # those, probe_filtered counts SF-decoders that probe pruned
    # off the spawn list; probe_kept_all counts activations where
    # probe found nothing AND fell back to spawn-all (or detected
    # an SF outside our preset list); probe_rejected counts
    # activations the GATE rejected entirely (zero decoders
    # spawned, only when RFCENSUS_PROBE_GATE=1);
    # probe_detected_count counts activations where >=1 SF was
    # detected (whether spawned or filtered).
    probe_decisions: int = 0
    probe_filtered: int = 0
    probe_kept_all: int = 0
    probe_rejected: int = 0
    probe_detected_count: int = 0
    # v0.7.13: periodic-probe lifecycle stats. Each tick of the
    # periodic probe either keeps decoders alive (probe fired) or
    # accumulates toward a reap (probe didn't fire). We surface:
    #   periodic_probe_scans: total periodic probe invocations
    #   periodic_probe_positive: subset that fired on a chirp
    #   periodic_reaps: times we killed all decoders for a slot
    #     because the probe stayed silent past reap_after_ms
    #   periodic_respawns: times we re-spawned decoders after a
    #     reap because the probe fired again on the same active slot
    periodic_probe_scans: int = 0
    periodic_probe_positive: int = 0
    periodic_reaps: int = 0
    periodic_respawns: int = 0
    # v0.7.13 commit 1c: pin lifecycle. pin_events counts
    # IDLE→PINNED transitions (a slot crossed pin_high_pct);
    # unpin_events counts PINNED→IDLE transitions (a slot dropped
    # below pin_low_pct). reap_skipped_pinned counts how many
    # times the periodic-reap path bailed because the slot was
    # pinned — a direct measure of CPU saved by the pin feature.
    pin_events: int = 0
    unpin_events: int = 0
    reap_skipped_pinned: int = 0


class LazyMultiPresetPipeline:
    """Coarse-FFT-gated multi-preset Meshtastic decoder.

    Same external interface as ``MultiPresetPipeline``: ``feed_cu8``
    + ``pop_packets``. Internally uses the ``PassbandDetector`` to
    decide which decoders to spawn.
    """

    # Default preamble symbols per Meshtastic spec (8 upchirps).
    PREAMBLE_SYMBOLS = 8

    def __init__(
        self,
        *,
        sample_rate_hz: int,
        center_freq_hz: int,
        candidate_slots: list[PresetSlot],
        mesh: MeshtasticDecoder,
        detector_config: Optional[DetectorConfig] = None,
        ring_buffer_ms: float = 500.0,
        per_packet_dwell_ms: float = 50.0,
        # v0.7.11: feature toggles for shared channelizer + blind probe.
        # Default both ON — they're proven equivalent (channelizer is
        # bit-exact to per-decoder mix+resamp; probe SNR threshold has
        # generous margin). Set to False to fall back to v0.7.x behavior
        # for A/B testing or if a regression is suspected.
        use_shared_channelizer: bool = True,
        use_blind_probe: bool = True,
        probe_snr_threshold_db: float = 20.0,
        # v0.7.13: periodic re-probe + reap-when-cold.
        # Every probe_interval_ms milliseconds while a slot is active,
        # run the blind probe on the most recent baseband. If the
        # probe doesn't fire for reap_after_ms milliseconds in a row,
        # tear down the decoders for that slot. They're re-spawned on
        # the next positive probe. probe_interval_ms ≤ shortest
        # preamble we care about (4ms for SF7/BW250). reap_after_ms ≥
        # max intra-packet quiet period (≈ 33ms for SF12/BW125 single
        # symbol, so 100ms gives ~3× margin).
        # Set use_periodic_probe=False to disable (v0.7.12 behavior:
        # decoders run from activate to deactivate without periodic
        # culling).
        use_periodic_probe: bool = True,
        probe_interval_ms: float = 10.0,
        reap_after_ms: float = 100.0,
        # v0.7.13 commit 1c: hot-slot pin via probe-positive rate.
        # Track recent probe-positive rate over a rolling window.
        # If the rate exceeds pin_high_pct, the slot becomes pinned
        # — periodic reap is skipped. If the rate drops below
        # pin_low_pct, unpin and resume normal reap. Hysteresis
        # (pin_low < pin_high) prevents flapping for slots near
        # the threshold.
        # Defaults intentionally LOW (5% / 10%). Measured probe-
        # positive rates on real-traffic Meshtastic captures are
        # 1-2% even on busy channels, because most active-window
        # time is between packets (no chirps to dechirp). 5/10%
        # captures truly-sustained channels (constant beacon, multi-
        # node back-to-back TX, co-located transmitters) without
        # over-pinning typical mesh traffic. Quiet channels stay
        # below 5% and continue to reap normally.
        pin_window_ms: float = 10_000.0,
        pin_high_pct: float = 10.0,
        pin_low_pct: float = 5.0,
    ) -> None:
        """Initialize the lazy pipeline.

        Args:
          sample_rate_hz: dongle sample rate (e.g. 2_400_000).
          center_freq_hz: dongle tuner center frequency.
          candidate_slots: list of (preset, slot) pairs the pipeline
            is allowed to monitor. Typically from
            ``enumerate_all_slots_in_passband``. Each slot lives at a
            specific (freq_hz, bandwidth_hz); the detector watches
            the unique (freq, bw) pairs derived from this list.
          mesh: shared MeshtasticDecoder for AES-CTR decrypt.
          detector_config: optional override for detector tuning. If
            None, sensible defaults are used.
          ring_buffer_ms: how much IQ to keep around for lookback.
            Default 300ms easily covers the longest Meshtastic
            preamble (260ms at SF12/BW125).
          per_packet_dwell_ms: how long to keep decoders alive past
            the detector's "deactivate" event. Should be ≥ the longest
            payload duration we care about (a 256-byte SF12 packet at
            BW=125 is ~1.5 sec; we set the default low because the
            detector keeps slots ACTIVE while energy is present, so
            dwell is only the post-deactivation tail).
          use_shared_channelizer: v0.7.11 — share one mix+resample
            across all decoders at the same slot. Bit-exact to per-
            decoder mix+resamp. Frees ~40% of channelization CPU at
            BW=250 (5 decoders sharing). Default True.
          use_blind_probe: v0.7.11 — run a multi-SF preamble probe
            on lookback baseband at activate time to filter the
            spawn list to only matching SFs. Multi-system safe (if
            two transmitters at different SFs are simultaneously
            active, both get spawned). Defensive fallback: if probe
            detects nothing, spawn ALL candidates. Default True.
          probe_snr_threshold_db: SNR threshold for the blind probe
            to count as detected. Default +20 dB — empirically
            measured: real LoRa preambles correlate at +25-+30 dB
            against the matching downchirp; pure noise produces
            +5-+12 dB false peaks. The +20 dB threshold cleanly
            separates the two with no overlap on captured data.
        """
        """Initialize the lazy pipeline.

        Args:
          sample_rate_hz: dongle sample rate (e.g. 2_400_000).
          center_freq_hz: dongle tuner center frequency.
          candidate_slots: list of (preset, slot) pairs the pipeline
            is allowed to monitor. Typically from
            ``enumerate_all_slots_in_passband``. Each slot lives at a
            specific (freq_hz, bandwidth_hz); the detector watches
            the unique (freq, bw) pairs derived from this list.
          mesh: shared MeshtasticDecoder for AES-CTR decrypt.
          detector_config: optional override for detector tuning. If
            None, sensible defaults are used.
          ring_buffer_ms: how much IQ to keep around for lookback.
            Default 300ms easily covers the longest Meshtastic
            preamble (260ms at SF12/BW125).
          per_packet_dwell_ms: how long to keep decoders alive past
            the detector's "deactivate" event. Should be ≥ the longest
            payload duration we care about (a 256-byte SF12 packet at
            BW=125 is ~1.5 sec; we set the default low because the
            detector keeps slots ACTIVE while energy is present, so
            dwell is only the post-deactivation tail).
        """
        self._sample_rate_hz = sample_rate_hz
        self._center_freq_hz = center_freq_hz
        self._mesh = mesh
        # v0.7.11 features
        # v0.7.11 features (env-toggleable for A/B ablation testing).
        # Defaults: shared channelizer + probe ON. Probe filters the
        # spawn list to matching SFs; bypass for adaptive magnitude.
        # Set RFCENSUS_NO_SHARED_CHAN=1 or RFCENSUS_NO_BLIND_PROBE=1
        # to disable for comparison.
        # v0.7.12: probe-as-gate is OPT-IN ONLY (RFCENSUS_PROBE_GATE=1).
        #
        # KNOWN LIMITATION of the gate: the detector emits ACTIVATE when
        # energy crosses threshold, which is often the LEADING EDGE of
        # a transmission's energy ramp — at that point the preamble
        # hasn't started yet, so the lookback is mostly noise.
        # Rejecting based on lookback probe will then drop real
        # packets, because the detector won't re-emit ACTIVATE while
        # it stays in the ACTIVE state (drain_frames keeps it active
        # for the duration of the burst).
        #
        # On the 30s test capture, gate ON drops recall 9/9 → 0/9
        # despite probe correctly identifying all SFs once the
        # preamble fully enters the lookback window — the probe just
        # fires LATER than the detector's activation point, and the
        # detector won't re-arm to give us another chance.
        #
        # Architectural alternatives we could try in v0.7.13+:
        #   • Re-probe on live samples ~50ms after activation (catch
        #     preambles that started AFTER activation), kill decoders
        #     if probe still finds nothing. Requires decoder state
        #     "tentative" → "confirmed" transitions.
        #   • Force-deactivate the detector slot when gate rejects,
        #     so it can re-emit ACTIVATE when energy stays elevated
        #     and the preamble actually appears.
        #
        # Until one of those lands, the gate is OFF by default and
        # the probe is used purely for SF-filtering (which works
        # correctly and saves real spawn cost — measured 53 → 15 SF
        # races on the 60s capture). Probe filtering loses zero
        # packets relative to legacy.
        #
        # Probe FILTERING (skip wrong-SF decoders when probe DOES
        # detect something) remains on by default — this is purely
        # additive (decoders skipped weren't going to decode the
        # signal anyway).
        import os as _os
        if _os.environ.get("RFCENSUS_NO_SHARED_CHAN"):
            use_shared_channelizer = False
        if _os.environ.get("RFCENSUS_NO_BLIND_PROBE"):
            use_blind_probe = False
        if _os.environ.get("RFCENSUS_PROBE_GATE"):
            self._probe_gates_activation = True
        else:
            self._probe_gates_activation = False
        self._use_shared_channelizer = use_shared_channelizer
        self._use_blind_probe = use_blind_probe
        self._probe_snr_threshold_db = probe_snr_threshold_db
        # v0.7.13: periodic re-probe + reap config. Convert ms to
        # input-stream samples up-front since we compare offsets in
        # the live feed loop and don't want to pay a multiply each
        # iteration.
        # Env-var override for ablation testing:
        #   RFCENSUS_NO_PERIODIC_PROBE=1 disables periodic probe
        #   entirely (decoders run from activate to deactivate, the
        #   v0.7.12 behavior).
        if _os.environ.get("RFCENSUS_NO_PERIODIC_PROBE"):
            use_periodic_probe = False
        self._use_periodic_probe = use_periodic_probe
        self._probe_interval_samples = int(
            sample_rate_hz * probe_interval_ms / 1000.0
        )
        self._reap_after_samples = int(
            sample_rate_hz * reap_after_ms / 1000.0
        )
        # v0.7.13 commit 1c: pin config. We store thresholds as
        # FRACTIONS (not %) to avoid repeated divides during the
        # hot-path comparison.
        # Env-var override:
        #   RFCENSUS_NO_PIN=1 disables pin entirely (slots always
        #   reap when probe goes silent, regardless of activity).
        if _os.environ.get("RFCENSUS_NO_PIN"):
            pin_high_pct = 1000.0   # never reach
            pin_low_pct = -1.0       # never reach
        self._pin_window_samples = int(
            sample_rate_hz * pin_window_ms / 1000.0
        )
        self._pin_high_frac = pin_high_pct / 100.0
        self._pin_low_frac = pin_low_pct / 100.0
        # v0.7.13.1: switched from a sliding-window count to EWMA so
        # the cold-start window doesn't transient-pin. The EWMA's
        # effective window length matches pin_window_ms by setting
        # alpha = 2 / (W + 1) where W = pin_window_ms /
        # probe_interval_ms. With defaults (10000ms / 10ms = 1000
        # ticks), alpha ≈ 0.002 — slow but smooth. With shorter
        # windows alpha grows correspondingly.
        ticks_per_window = max(
            1, int(pin_window_ms / probe_interval_ms),
        )
        self._pin_ewma_alpha = 2.0 / (ticks_per_window + 1)
        # Warmup: skip pin/unpin evaluation for the first second of
        # probe activity so the EWMA isn't biased by initial
        # transients. 1 second = ~100 ticks at default 10ms interval.
        self._pin_warmup_ticks = max(
            10, int(1000.0 / probe_interval_ms),
        )
        # v0.7.13 commit 1b: per-(slot_freq, bw, sf) "confirmed" set.
        # An entry lands here when a decoder produces a CRC-pass at
        # that (slot, sf) triple. Once confirmed, the periodic-reap
        # path and the SF-race-loser path NEVER kill that decoder —
        # it stays alive any time the slot is active, ensuring we
        # don't miss future packets at a SF that has demonstrably
        # produced traffic on this exact slot.
        # Lifetime: pipeline-scoped (survives detector deactivate /
        # reactivate cycles for the same slot freq+bw).
        # Granularity is per-(slot, SF), not per-SF — confirming
        # SF9 on slot 913.125 doesn't confirm SF9 on slot 913.625.
        self._confirmed_slot_sf: set[tuple[int, int, int]] = set()
        # v0.7.13 commit 1b: bleed-aware confirmation requires
        # comparing the RSSI of any new CRC-pass against recent
        # CRC-passes for the same payload (within ~one packet
        # duration). The PIPELINE's _pending buffer would normally
        # serve, but it gets drained on each pop_packets/drain_packets
        # call — the comparison would miss bleed copies that arrive
        # between drain calls. Keep a separate sliding-window log:
        # (sample_offset, payload_bytes, rssi_db, slot_freq, bw, sf).
        # Trimmed to last 480ms (= 1 medium-fast packet duration at
        # typical sample rates) on each insert.
        # Used by _record_crc_pass to:
        #   1. Decide whether THIS pass is the highest-RSSI member
        #      of its dedup cluster → if yes, add to confirmed.
        #   2. Demote previously-confirmed (slot, sf) for this
        #      payload if the new pass has higher RSSI.
        self._recent_crc_passes: list[tuple[int, bytes, float, int, int, int]] = []
        # 480ms in input samples — covers the longest mediumfast packet
        # plus margin. Bleed always arrives within tens of ms, so 480
        # is generous.
        self._crc_dedup_window_samples = int(0.480 * sample_rate_hz)
        # Lazy per-slot BlindProbe cache: one probe per (slot_freq, bw)
        # built on first activation. Reused across activations for that
        # slot since the candidate SF list doesn't change.
        self._probes_by_slot: dict[tuple[int, int], object] = {}

        # Build (freq, bw) → list of preset_keys map. When a slot fires
        # we spawn one decoder per preset in the matching list.
        self._presets_by_slot: dict[tuple[int, int], list[PresetSlot]] = (
            defaultdict(list)
        )
        for cs in candidate_slots:
            key = (cs.freq_hz, cs.preset.bandwidth_hz)
            self._presets_by_slot[key].append(cs)

        # IQ ring buffer for lookback
        ring_samples = int(sample_rate_hz * ring_buffer_ms / 1000)
        self._ring = IqRingBuffer(capacity_samples=ring_samples)

        # Build the passband detector
        det_cfg = detector_config or DetectorConfig(
            sample_rate_hz=sample_rate_hz,
            center_freq_hz=center_freq_hz,
        )
        unique_slots = list(self._presets_by_slot.keys())
        self._detector = PassbandDetector(
            config=det_cfg,
            slot_freqs_hz=[s[0] for s in unique_slots],
            slot_bandwidths_hz=[s[1] for s in unique_slots],
        )
        self._detector_config = det_cfg

        # Active-slot tracking, keyed by (freq_hz, bw_hz)
        self._active: dict[tuple[int, int], _ActiveSlot] = {}

        # Pending packets: decoders may emit packets across multiple
        # feed_cu8 calls; we accumulate here and drain via pop_packets.
        self._pending: list[PipelinePacket] = []

        # Stats
        self._stats = LazyPipelineStats()

        # Per-preset accumulator that survives decoder teardown — the
        # ``stats()`` method returns this merged with currently-live
        # decoder stats so callers see the full lifetime view.
        self._per_preset_cumulative: dict[str, "LoraStats"] = {}

        # Total samples consumed (= global sample offset of next sample)
        self._samples_consumed = 0

        # v0.7.7: wall-clock keep-up tracking. We measure how long
        # feed_cu8 takes vs the audio-time the chunk represents. A
        # ratio > 1.0 means we're running slower than real-time and
        # the upstream rtl_tcp socket buffer is filling — eventually
        # the dongle drops samples at the kernel/USB level. Surfaced
        # via stats() so the standalone tool can warn the user.
        # Reported as a sliding mean over the last N feeds.
        from collections import deque
        self._keepup_ratios: deque = deque(maxlen=64)
        self._last_keepup_warning_ts: float = 0.0

    @property
    def keepup_ratio(self) -> float:
        """v0.7.7: rolling mean of (wall_clock_seconds_per_feed /
        audio_seconds_per_feed). 1.0 means processing exactly at
        real-time; > 1.0 means falling behind. Sustained values
        above ~0.85 mean the system is at the edge of its CPU
        budget and should reduce slot count or sample rate."""
        if not self._keepup_ratios:
            return 0.0
        return sum(self._keepup_ratios) / len(self._keepup_ratios)

    @property
    def n_candidate_slots(self) -> int:
        """How many (preset, slot) pairs we'd spawn decoders for in
        the worst case (all slots active simultaneously)."""
        return sum(len(v) for v in self._presets_by_slot.values())

    @property
    def n_unique_slot_frequencies(self) -> int:
        """How many distinct (freq, bw) combinations the detector
        watches."""
        return len(self._presets_by_slot)

    @property
    def n_active_decoders(self) -> int:
        """Current count of running LoRa decoders (varies over time)."""
        return sum(len(a.decoders) for a in self._active.values())

    @property
    def lazy_stats(self) -> LazyPipelineStats:
        """Lazy-pipeline specific stats: detector frames, slot
        activations, decoder spawn/teardown counts. Distinct from
        ``stats()`` which mirrors ``MultiPresetPipeline.stats()`` for
        CLI interchangeability."""
        return self._stats

    # Backward-compat alias for the older ``pipe.stats`` property
    # (was the only stats interface in v0.7.2 dev). New code should use
    # ``lazy_stats`` (above) for lazy-specific counters or ``stats()``
    # (below) for per-preset LoraStats matching the eager pipeline.
    @property
    def pipeline_stats(self) -> LazyPipelineStats:
        return self._stats

    def stats(self) -> dict[str, "LoraStats"]:
        """Per-preset cumulative LoraStats — same shape as
        ``MultiPresetPipeline.stats()``. Lets the CLI use either
        pipeline interchangeably for the per-preset summary table.

        Lazy decoders are short-lived, so asking only the currently-
        active ones would undercount; we accumulate into
        ``_per_preset_cumulative`` at decoder teardown time and merge
        in the still-live ones here on demand."""
        from copy import deepcopy
        out = {k: deepcopy(v) for k, v in self._per_preset_cumulative.items()}
        for active in self._active.values():
            for preset_key, dec in active.decoders.items():
                live = dec.stats()
                if preset_key in out:
                    out[preset_key] = self._merge_stats(out[preset_key], live)
                else:
                    out[preset_key] = deepcopy(live)
        return out

    @staticmethod
    def _merge_stats(a, b):
        """Sum two LoraStats. Float fields take the maximum observed
        value (peak mag is meaningful as a max, not a sum)."""
        from rfcensus.decoders.lora_native import LoraStats
        return LoraStats(
            samples_processed=a.samples_processed + b.samples_processed,
            preambles_found=a.preambles_found + b.preambles_found,
            syncwords_matched=a.syncwords_matched + b.syncwords_matched,
            headers_decoded=a.headers_decoded + b.headers_decoded,
            headers_failed=a.headers_failed + b.headers_failed,
            packets_decoded=a.packets_decoded + b.packets_decoded,
            packets_crc_failed=a.packets_crc_failed + b.packets_crc_failed,
            detect_attempts=a.detect_attempts + b.detect_attempts,
            detect_above_gate=a.detect_above_gate + b.detect_above_gate,
            detect_max_run=max(a.detect_max_run, b.detect_max_run),
            detect_peak_mag_max=max(
                a.detect_peak_mag_max, b.detect_peak_mag_max,
            ),
        )

    def feed_cu8(self, samples: bytes) -> int:
        """Ingest a chunk of cu8 IQ.

        Returns the number of packets that emerged during this call
        (also accumulated in the internal pending queue for
        ``pop_packets``).
        """
        # v0.7.7: wall-clock timing for keep-up tracking. We measure
        # how long the entire chunk takes to process and compare to
        # the audio time it represents. If we're slower than real-
        # time the upstream socket buffer fills and packets drop at
        # the kernel/USB layer with no signal to us — except for
        # the timing ratio.
        import time
        wall_start = time.perf_counter()

        n_packets_before = len(self._pending)
        n_samples = len(samples) // 2
        chunk_start_offset = self._samples_consumed
        chunk_end_offset = chunk_start_offset + n_samples

        # Snapshot ring overflow counters BEFORE write so we can tell
        # if THIS write caused an overflow (vs accumulated history).
        ring_overflow_before = self._ring.overflow_events
        ring_dropped_before = self._ring.samples_dropped

        # Append to ring buffer FIRST so any lookback reads work
        self._ring.write(samples)

        # Propagate any ring drops into pipeline stats.
        if self._ring.overflow_events > ring_overflow_before:
            self._stats.ring_overflows += (
                self._ring.overflow_events - ring_overflow_before
            )
            self._stats.samples_dropped += (
                self._ring.samples_dropped - ring_dropped_before
            )

        # Run the detector and process events. Activation events emit
        # decoders pre-fed with lookback IQ; their next_sample_offset
        # is set to the activation point, so the live feed below
        # picks up from there without duplicating samples.
        for event in self._detector.feed_cu8(samples):
            if event.kind == "activate":
                self._handle_activate(event)
            elif event.kind == "deactivate":
                self._handle_deactivate(event)

        # Feed live samples to all active decoders. Slice the chunk
        # to start at the decoder's next_sample_offset so we don't
        # re-feed lookback samples to a freshly-spawned decoder.
        # v0.7.13: also channelize for REAPED slots (decoders dict
        # empty) so the periodic probe has fresh baseband to scan.
        # The channelizer is the cheap part; decoders are the
        # expensive part. Reaping decoders saves CPU; keeping the
        # channelizer running maintains state for instant respawn.
        for active in self._active.values():
            if active.next_sample_offset >= chunk_end_offset:
                # Decoder is somehow ahead of us — nothing to feed.
                continue
            slice_start = max(0, active.next_sample_offset
                                - chunk_start_offset)
            slice_bytes = samples[slice_start * 2:]
            if slice_bytes:
                # v0.7.11: route through shared channelizer when
                # available — channelize ONCE, fan out to all decoders.
                if active.channelizer is not None:
                    baseband = active.channelizer.feed_cu8(slice_bytes)
                    if len(baseband) > 0:
                        # v0.7.13: append to the probe-baseband ring
                        # so periodic probe sees fresh data.
                        if self._use_periodic_probe:
                            self._append_probe_baseband(active, baseband)
                        if active.decoders:
                            floats = np.empty(2 * len(baseband),
                                               dtype=np.float32)
                            floats[0::2] = baseband.real
                            floats[1::2] = baseband.imag
                            for dec in active.decoders.values():
                                dec.feed_baseband(floats)
                else:
                    # Legacy v0.7.x path: each decoder mix+resamps.
                    # No probe-buffer in this path; periodic probe
                    # needs the channelizer (probe operates on
                    # baseband). Periodic probe is implicitly
                    # disabled when use_shared_channelizer=False.
                    if active.decoders:
                        for dec in active.decoders.values():
                            dec.feed_cu8(slice_bytes)
            active.next_sample_offset = chunk_end_offset
            # v0.7.7: SF racing — once we've fed each decoder past
            # the racing deadline (= activation + 1.5× slowest-
            # preset preamble length), resolve which decoder won
            # and kill the losers. See _maybe_resolve_race for the
            # rationale + safety analysis.
            self._maybe_resolve_race(active)
            # v0.7.13: periodic re-probe — kill decoders that haven't
            # seen probe activity in `reap_after_ms`. Re-spawn on the
            # next positive probe. Cheaper than letting the racing
            # killers' losers run to deactivate, AND lets us free
            # CPU on long-active periods that don't actually contain
            # ongoing chirps.
            if self._use_periodic_probe:
                self._maybe_periodic_probe(active, chunk_end_offset)

        # Drain decoder packet queues. The C decoder reports
        # sample_offset in OUTPUT (post-resampler) samples relative to
        # decoder creation. Rebase to absolute INPUT-stream offset so
        # the user-facing offset is meaningful and so our dedup can
        # cluster the same physical packet caught by multiple decoders.
        for active in self._active.values():
            for preset_key, dec in active.decoders.items():
                slot = active.slot_metadata[preset_key]
                for lp in dec.pop_packets():
                    rebased = self._rebase(lp, active)
                    self._stats.packets_decoded += 1
                    if rebased.crc_ok:
                        # v0.7.13 commit 1b: confirm this (slot, sf)
                        # for spawn-priority on future activations.
                        self._record_crc_pass(slot, rebased)
                    mesh_pkt = (self._mesh.decode(rebased.payload)
                                  if rebased.crc_ok else None)
                    if mesh_pkt and mesh_pkt.decrypted:
                        self._stats.packets_decrypted += 1
                    self._pending.append(PipelinePacket(
                        slot=slot, lora=rebased, mesh=mesh_pkt,
                    ))

        self._samples_consumed += n_samples
        self._stats.detector_frames = self._detector.frame_count

        # v0.7.7: keep-up tracking. wall_dt = how long this feed took;
        # audio_dt = how much real-world time the chunk represents.
        # ratio < 1.0 means we processed faster than real-time (have
        # CPU headroom); > 1.0 means we're falling behind.
        wall_dt = time.perf_counter() - wall_start
        audio_dt = n_samples / self._sample_rate_hz
        if audio_dt > 0:
            self._keepup_ratios.append(wall_dt / audio_dt)

        return len(self._pending) - n_packets_before

    def pop_packets(
        self,
        dedup: bool = True,
        dedup_offset_tolerance: int | None = None,
    ) -> Iterator[PipelinePacket]:
        """Drain pending packets in chronological order, with dedup.

        Duplicates arise when the lazy spawner spawns multiple decoders
        per slot activation (e.g. all 5 BW=250 presets at slot S, even
        though only one preset is actually transmitting) and ALSO when
        adjacent slots both activate from a single physical
        transmission (LoRa CFO tolerance is wide enough that a signal
        at slot N can be decoded by neighboring-slot decoders too).

        Dedup strategy: cluster packets that share payload bytes AND
        sample_offset within ``dedup_offset_tolerance`` input samples.

        v0.7.6: tolerance default raised significantly. The previous
        256-sample default (~100 µs at 2.4 MS/s) only caught
        duplicates that were within ~1/4 of a single LoRa symbol —
        far too tight. Real-world parallel-slot duplicates land
        within a few LoRa SYMBOLS of each other (multiple
        milliseconds) because the slot decoders' filter pipelines
        emit packet metadata at slightly different resampler-output
        offsets. A user reported the same NODEINFO packet emitted
        4× across ~6300 samples (2.6 ms at 2.4 MS/s) on a real
        capture — well outside 256.

        New default: 200 ms expressed in input samples (i.e.
        ``sample_rate_hz / 5``). 200 ms is wider than any single
        LoRa packet's airtime in the Meshtastic preset set (max ~150
        ms for SF12 long-fast at 240 byte payload) but well below
        typical mesh re-broadcast intervals (1-5 seconds), so we
        catch all parallel-slot copies of one transmission without
        merging legitimately distinct re-broadcasts. The
        payload-byte-equality check is the strong invariant — the
        chance of two genuinely distinct packets having byte-identical
        payloads is astronomically small (1 in 2^(8N) for N-byte
        payloads), so the offset window is mostly a sanity bound.

        Within each cluster, we keep the one with highest RSSI (or,
        if RSSI is the placeholder 0.0, keep the median-frequency
        slot). Set ``dedup=False`` to see every decoder's emit
        (useful for debugging slot attribution).

        Pass an explicit ``dedup_offset_tolerance`` (in input samples)
        to override the default.
        """
        if not self._pending:
            return

        # v0.7.6: derive tolerance from sample rate if not specified.
        # 200 ms covers all real Meshtastic packet airtimes with
        # margin while staying well below mesh re-broadcast spacing.
        if dedup_offset_tolerance is None:
            dedup_offset_tolerance = self._sample_rate_hz // 5

        all_packets = self._pending
        self._pending = []
        all_packets.sort(key=lambda p: p.lora.sample_offset)

        if not dedup:
            yield from all_packets
            return

        # Cluster by (payload, offset proximity)
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
            # v0.7.8: RSSI is real (v0.7.7 fixed the C-side
            # computation), so the tie-breaker just picks the
            # highest-RSSI member of the cluster. The slot whose
            # center frequency matches the actual transmission
            # frequency will see the strongest signal — adjacent
            # slots that picked it up via LoRa CFO tolerance see
            # 3-15 dB lower RSSI because the FFT correlator peak
            # is offset from their center bin. Picking max(RSSI)
            # therefore picks the correct slot deterministically.
            #
            # Tie-break (extremely unlikely with float-precision
            # RSSI but defensive): prefer the slot whose center
            # frequency is the median of the cluster's freqs.
            best = max(grp, key=lambda p: p.lora.rssi_db)
            ties = [p for p in grp
                    if p.lora.rssi_db == best.lora.rssi_db]
            if len(ties) > 1:
                ties_sorted = sorted(ties, key=lambda p: p.slot.freq_hz)
                best = ties_sorted[len(ties_sorted) // 2]
            yield best

    def _handle_activate(self, event: SlotEvent) -> None:
        """Spawn decoders for the activated slot.

        Reads lookback IQ from the ring buffer to seed each decoder
        with samples from before the detector fired (so it can see
        the preamble). Then registers the active slot for future
        live-IQ feeding.
        """
        key = (event.slot_freq_hz, event.bandwidth_hz)
        if key in self._active:
            # Already active — should not happen if detector is well-
            # behaved (no double-activate without intervening deactivate).
            # Defensive: just keep the existing decoders running.
            return

        presets = self._presets_by_slot.get(key, [])
        if not presets:
            # Detector fired on a slot we don't have presets for —
            # shouldn't happen because the detector was built from
            # this same list, but be defensive.
            return

        self._stats.slot_activations += 1

        # Compute lookback: how far back in the ring we'll go to feed
        # the decoder enough preamble. The longest preamble in our
        # preset set determines the lookback amount; for narrow-BW /
        # high-SF presets (e.g. SF12/BW125) the preamble is ~260ms.
        presets_for_lookback = presets
        max_preamble_ms = self._max_preamble_ms_for_presets(presets_for_lookback)
        lookback_samples = int(
            self._sample_rate_hz * max_preamble_ms / 1000
        )
        # Plus a pad for detector latency (the energy integration takes
        # trigger_frames × hop_samples before the activate event fires).
        # Use 2× the deterministic latency as safety against slow
        # SNR-rise edges.
        detector_latency_samples = (
            self._detector_config.trigger_frames
            * self._detector_config.hop_samples * 2
        )
        total_lookback = lookback_samples + detector_latency_samples

        # Clamp to what the ring actually holds
        actual_lookback = min(total_lookback, len(self._ring))
        lookback_start = event.sample_offset - actual_lookback
        if lookback_start < self._ring.oldest_offset:
            lookback_start = self._ring.oldest_offset
            actual_lookback = event.sample_offset - lookback_start

        active = _ActiveSlot(
            freq_hz=event.slot_freq_hz,
            bandwidth_hz=event.bandwidth_hz,
            activated_sample_offset=event.sample_offset,
            # The first sample fed to each decoder lives at this
            # global INPUT-stream offset. Used to rebase the C
            # decoder's locally-numbered packet offsets back to the
            # absolute capture position.
            feed_start_offset=lookback_start,
            # After feeding lookback (lookback_start..event.sample_offset),
            # the next live sample to feed is event.sample_offset.
            next_sample_offset=event.sample_offset,
            # If only one preset is candidate, racing is a no-op —
            # mark resolved immediately. (Eager pipeline doesn't go
            # through here, but a slot with a single preset still
            # hits this path for narrow-BW configs.)
            race_resolved=(len(presets) <= 1),
        )

        # Compute mix_freq for this slot
        mix_freq = self._center_freq_hz - event.slot_freq_hz
        bw = event.bandwidth_hz

        # v0.7.11: shared channelizer + blind probe.
        #
        # When sharing is on, we create one SharedChannelizer for this
        # slot and spawn decoders configured to consume baseband (mix=0,
        # sample_rate=bandwidth). The channelizer handles mix+resamp
        # ONCE; each decoder consumes the same baseband stream via
        # feed_baseband. Bit-exact to per-decoder mix+resamp; ~40%
        # cheaper at BW=250 (5 decoders sharing one channelization).
        #
        # When the blind probe is on, we run a one-shot multi-SF probe
        # on the lookback baseband to identify which SF(s) actually
        # have a preamble present. Only matching SFs get spawned.
        # This replaces the v0.7.7 "spawn 5, race" with "look once,
        # spawn matching" — saves the racing CPU entirely on the
        # common case (1 transmitter at 1 SF). Multi-system support:
        # if 2 transmitters at different SFs are simultaneously
        # active, BOTH get spawned. Safety: if the probe detects
        # zero SFs (low-SNR signal that crossed the detector
        # threshold but failed probe), spawn ALL candidates as the
        # fallback so we don't lose a packet to probe over-tightness.
        use_sharing = (
            self._use_shared_channelizer
            and bw <= self._sample_rate_hz
            and len(presets) >= 1    # always safe to share even with 1 decoder
        )
        channelizer = None
        lookback_baseband = None    # complex64 numpy array, set if we have one

        if use_sharing:
            channelizer = SharedChannelizer(
                sample_rate_hz=self._sample_rate_hz,
                bandwidth_hz=bw,
                mix_freq_hz=mix_freq,
            )
            active.channelizer = channelizer
            # Channelize the lookback once so we can both feed it to
            # decoders AND probe it.
            if actual_lookback > 0:
                lookback_iq = self._ring.read(
                    lookback_start, actual_lookback,
                )
                if lookback_iq is not None:
                    lookback_baseband = channelizer.feed_cu8(lookback_iq)

        # v0.7.11: optional blind probe to filter presets.
        # v0.7.12: probe also GATES the activation — if no SF is
        # detected (and no absolute-magnitude bypass triggers), the
        # entire activation is rejected and no decoders spawn at all.
        # This kills the false-positive activations that the energy
        # detector emits on noise transients.
        spawn_presets = list(presets)
        probe_gate_decision = "skipped"   # "skipped" | "detected" | "rejected" | "fell_back"
        if (self._use_blind_probe
                and lookback_baseband is not None
                and len(presets) >= 1):
            # Probe identifies which SF(s) have a preamble in the
            # lookback baseband. Build a probe matching the candidate
            # SFs at this slot (cached across activations because the
            # candidate SFs don't change for a given slot freq+bw).
            probe = self._probes_by_slot.get(key)
            if probe is None:
                # SF list for this slot — sorted, unique
                cand_sfs = sorted({s.preset.sf for s in presets})
                # Probe uses oversample=1 to match the decoder's
                # internal expectation (post-decimation samples are
                # at BW rate, FFT size N = 2^SF, NOT 2^SF × os).
                # An earlier oversample=2 default produced an SF
                # off-by-one because the chirp slope was wrong.
                try:
                    probe = BlindProbe(
                        sfs=cand_sfs,
                        oversample=1,
                        snr_threshold_db=self._probe_snr_threshold_db,
                    )
                    self._probes_by_slot[key] = probe
                except Exception:
                    probe = None
            if probe is not None:
                # Run probe on enough baseband samples to cover the
                # largest candidate SF's symbol size. Sample 4 windows
                # shifted across the LOOKBACK TAIL — that's where the
                # detector said energy crossed threshold, so the
                # preamble is most likely at the end of lookback.
                #
                # For each SF, take the MAXIMUM SNR across windows.
                # Either window catching a preamble suffices ("max"
                # over windows is the right operator because a
                # preamble that aligns with one window's start will
                # be split-and-attenuated in the adjacent window).
                min_n = probe.min_samples_required
                if len(lookback_baseband) >= min_n:
                    self._stats.probe_decisions += 1
                    # Aggregate per-SF: best (highest-SNR) result
                    # across the 4 windows.
                    best_per_sf: dict[int, "ProbeResult"] = {}
                    step = max(1, min_n // 4)
                    n_windows = 4
                    bb_end = len(lookback_baseband)
                    for w in range(n_windows):
                        end = bb_end - w * step
                        start = end - min_n
                        if start < 0:
                            break
                        window = lookback_baseband[start:end]
                        for r in probe.scan(window):
                            cur_best = best_per_sf.get(r.sf)
                            if cur_best is None or r.snr_db > cur_best.snr_db:
                                best_per_sf[r.sf] = r

                    # Decision per SF: in-FFT SNR ≥ threshold → detected.
                    # Real LoRa preambles produce in-FFT SNR of +25 to
                    # +37 dB at oversample=1, well above the +20 dB
                    # default threshold. For weaker signals, lower
                    # probe_snr_threshold_db (e.g. to +12 dB).
                    #
                    # An earlier v0.7.12 draft tried an absolute-
                    # magnitude bypass (peak vs rolling-min noise) but
                    # the cross-SF sqrt(N) bias and per-SF noise
                    # scaling made it net-negative on real captures.
                    # Removed.
                    detected_sfs: set[int] = set()
                    for sf, r in best_per_sf.items():
                        if r.snr_db >= self._probe_snr_threshold_db:
                            detected_sfs.add(sf)

                    if detected_sfs:
                        self._stats.probe_detected_count += 1
                        # Filter presets to detected SF(s) only.
                        # v0.7.13 commit 1b: ALSO include confirmed
                        # (slot, sf) presets — these have decoded a
                        # CRC-pass at this slot in the past, so even
                        # if the probe didn't fire on them this round
                        # we want them spawned to catch the next
                        # packet at this slot's known-good SFs.
                        confirmed_at_slot = self._confirmed_presets_for_slot(
                            event.slot_freq_hz, event.bandwidth_hz,
                        )
                        confirmed_keys = {s.preset.key for s in confirmed_at_slot}
                        filtered_keys = {s.preset.key for s in presets
                                         if s.preset.sf in detected_sfs}
                        union_keys = filtered_keys | confirmed_keys
                        filtered = [s for s in presets
                                    if s.preset.key in union_keys]
                        if filtered:
                            spawned_skipped = len(presets) - len(filtered)
                            self._stats.probe_filtered += spawned_skipped
                            spawn_presets = filtered
                            probe_gate_decision = "detected"
                            # If probe filtered to a single SF, the
                            # legacy v0.7.7 SF race becomes redundant.
                            # Mark resolved immediately so we don't
                            # waste cycles racing the (now lone)
                            # winner against itself.
                            if len(filtered) == 1:
                                active.race_resolved = True
                        else:
                            # detected an SF we don't have a preset
                            # for — fall back to spawn-all
                            self._stats.probe_kept_all += 1
                            probe_gate_decision = "fell_back"
                    elif self._probe_gates_activation:
                        # GATE: probe rejected this activation.
                        # v0.7.13 commit 1b: even with gate ON, spawn
                        # any confirmed (slot, sf) presets. The probe
                        # missed it but historically this slot has
                        # produced packets at these SFs.
                        confirmed_at_slot = self._confirmed_presets_for_slot(
                            event.slot_freq_hz, event.bandwidth_hz,
                        )
                        if confirmed_at_slot:
                            spawn_presets = confirmed_at_slot
                            probe_gate_decision = "detected"
                        else:
                            # Don't spawn anything. The detector fired
                            # on noise that didn't actually contain a
                            # preamble, and there's no historical
                            # confirmation to override.
                            self._stats.probe_rejected += 1
                            probe_gate_decision = "rejected"
                            # Free the channelizer we created — no
                            # decoders consuming it.
                            if channelizer is not None:
                                try:
                                    channelizer.close()
                                except Exception:
                                    pass
                                active.channelizer = None
                            # Don't add to self._active. Early return
                            # equivalent: just skip the spawn loop.
                            spawn_presets = []
                    else:
                        # Gate disabled — spawn all (v0.7.11 fallback).
                        self._stats.probe_kept_all += 1
                        probe_gate_decision = "fell_back"

        # Spawn decoders. Configuration depends on whether we're
        # sharing the channelizer (decoders consume baseband) or
        # not (decoders do their own mix+resamp from cu8 ring).
        for slot in spawn_presets:
            if use_sharing:
                # Decoder consumes already-channelized baseband:
                # rate = bandwidth, mix disabled.
                cfg = LoraConfig(
                    sample_rate_hz=bw,
                    bandwidth=bw,
                    sf=slot.preset.sf,
                    sync_word=_MESHTASTIC_SYNC_WORD,
                    mix_freq_hz=0,
                    ldro=(slot.preset.sf == 12
                          and slot.preset.bandwidth_hz <= 125_000),
                )
            else:
                # Legacy v0.7.x path: per-decoder mix+resamp from cu8.
                cfg = LoraConfig(
                    sample_rate_hz=self._sample_rate_hz,
                    bandwidth=slot.preset.bandwidth_hz,
                    sf=slot.preset.sf,
                    sync_word=_MESHTASTIC_SYNC_WORD,
                    mix_freq_hz=mix_freq,
                    ldro=(slot.preset.sf == 12
                          and slot.preset.bandwidth_hz <= 125_000),
                )
            dec = LoraDecoder(cfg)
            active.decoders[slot.preset.key] = dec
            active.slot_metadata[slot.preset.key] = slot
            self._stats.decoders_spawned += 1

        # Feed the lookback to the newly-spawned decoders.
        if not active.decoders:
            # Probe gate rejected this activation — no decoders to
            # feed, no active slot to track. (channelizer was already
            # freed in the gate branch above.) Just don't register.
            return
        if use_sharing and lookback_baseband is not None and len(lookback_baseband) > 0:
            # Already-channelized — fan out to all decoders.
            floats = np.empty(2 * len(lookback_baseband), dtype=np.float32)
            floats[0::2] = lookback_baseband.real
            floats[1::2] = lookback_baseband.imag
            for dec in active.decoders.values():
                dec.feed_baseband(floats)
        elif actual_lookback > 0:
            # Legacy path: each decoder does its own mix+resamp.
            lookback_iq = self._ring.read(lookback_start, actual_lookback)
            if lookback_iq is not None:
                for dec in active.decoders.values():
                    dec.feed_cu8(lookback_iq)

        self._active[key] = active
        # v0.7.13: initialize periodic-probe lifecycle for this slot.
        # The probe was already created above (cached in _probes_by_slot)
        # for the activation-time check. Reuse that same probe instance
        # — it has the right SF list and threshold for this slot.
        if self._use_periodic_probe and use_sharing:
            active.probe = self._probes_by_slot.get(key)
            # Schedule first probe one interval ahead. The activation
            # path already did a probe-on-lookback; the next periodic
            # probe should target genuinely new baseband.
            active.next_probe_offset = (
                event.sample_offset + self._probe_interval_samples
            )
            # Treat the activation as a "positive probe" for reap
            # accounting — we just spawned because activation said
            # there's energy, so don't reap-on-cold within
            # reap_after_ms of activate. (If activation was a false
            # positive, the next reap_after_ms of probe scans will
            # confirm and reap.)
            active.last_positive_probe_offset = event.sample_offset
            # probe_baseband is allocated lazily by _append_probe_baseband.

    def _spawn_decoder_for_slot(
        self,
        active: _ActiveSlot,
        slot: PresetSlot,
        use_sharing: bool,
        mix_freq: int,
    ) -> LoraDecoder:
        """v0.7.13: build and register a LoraDecoder for a preset slot.

        Centralizes the spawn config so both `_handle_activate` (initial
        spawn) and `_periodic_respawn` (after reap) build decoders the
        same way.
        """
        bw = slot.preset.bandwidth_hz
        if use_sharing:
            cfg = LoraConfig(
                sample_rate_hz=bw,
                bandwidth=bw,
                sf=slot.preset.sf,
                sync_word=_MESHTASTIC_SYNC_WORD,
                mix_freq_hz=0,
                ldro=(slot.preset.sf == 12
                      and slot.preset.bandwidth_hz <= 125_000),
            )
        else:
            cfg = LoraConfig(
                sample_rate_hz=self._sample_rate_hz,
                bandwidth=slot.preset.bandwidth_hz,
                sf=slot.preset.sf,
                sync_word=_MESHTASTIC_SYNC_WORD,
                mix_freq_hz=mix_freq,
                ldro=(slot.preset.sf == 12
                      and slot.preset.bandwidth_hz <= 125_000),
            )
        dec = LoraDecoder(cfg)
        active.decoders[slot.preset.key] = dec
        active.slot_metadata[slot.preset.key] = slot
        self._stats.decoders_spawned += 1
        return dec

    def _append_probe_baseband(
        self, active: _ActiveSlot, baseband,
    ) -> None:
        """v0.7.13: append fresh channelized baseband to the slot's
        rolling probe-buffer.

        We keep enough samples to run a multi-window probe scan: the
        probe needs N = 2^max_SF × oversample samples per scan, and
        we run 4 windows shifted across the buffer. So buffer size =
        ~5× max-N. For SF11 oversample=1 max-N = 2048, so 10k samples
        is plenty (~40ms at BW=250). For SF12/BW125 max-N = 4096, so
        20k samples (~160ms).

        Implementation: simple numpy concatenation with truncation.
        Could be optimized with a true ring buffer if profiling shows
        this dominates — at ~10ms intervals and small buffer sizes,
        the concat cost should be negligible (microseconds per call).
        """
        max_n = self._probe_max_window_samples_for(active)
        # Keep ~5× max_n in the buffer so a 4-window scan with stride
        # max_n/4 always has data to scan. Add a bit of headroom for
        # the channelizer producing >K_ms of samples in a chunk.
        target_capacity = max_n * 5
        existing = active.probe_baseband
        if existing is None:
            active.probe_baseband = baseband.copy()
        else:
            active.probe_baseband = np.concatenate([existing, baseband])
        # Truncate from the front if oversize.
        if len(active.probe_baseband) > target_capacity:
            active.probe_baseband = active.probe_baseband[-target_capacity:]

    def _probe_max_window_samples_for(
        self, active: _ActiveSlot,
    ) -> int:
        """Compute N = 2^max_SF × oversample for the probe at this slot.

        The probe is built with the slot's candidate SF list. Cached
        on the active slot via the probe instance itself. Falls back
        to a conservative 2048 if probe isn't initialized yet.
        """
        if active.probe is None:
            return 2048
        return int(active.probe.min_samples_required)

    def _update_pin_state(
        self, active: _ActiveSlot, chunk_end_offset: int,
        fired: bool,
    ) -> None:
        """v0.7.13.1: EWMA-based hysteretic pin transitions.

        Pin state machine (unchanged from prior version):
          • UNPINNED + ewma_rate ≥ pin_high_pct → PINNED
          • PINNED + ewma_rate < pin_low_pct → UNPINNED
          • Within (pin_low, pin_high): no transition

        Why EWMA over sliding-window count: a fixed-N count is
        susceptible to early-window edge artifacts (when the window
        is half-full, a single fire produces a misleadingly-high
        rate). EWMA smooths that out — initial value 0.0, gradual
        rise as fires accumulate. Combined with a warmup period
        that skips evaluation entirely for the first second, we get
        a stable monotonic rate signal.

        EWMA update: rate ← α·sample + (1−α)·rate, where
        α = 2/(W+1), W = ticks per pin_window_ms.
        """
        # Update EWMA every probe regardless of pin state.
        sample = 1.0 if fired else 0.0
        active.probe_rate_ewma = (
            self._pin_ewma_alpha * sample
            + (1.0 - self._pin_ewma_alpha) * active.probe_rate_ewma
        )
        active.probe_history_count += 1
        # Warmup gate: don't evaluate until we have enough probes
        # for the EWMA to be meaningful.
        if active.probe_history_count < self._pin_warmup_ticks:
            return
        rate = active.probe_rate_ewma
        if active.pinned:
            if rate < self._pin_low_frac:
                active.pinned = False
                self._stats.unpin_events += 1
        else:
            if rate >= self._pin_high_frac:
                active.pinned = True
                self._stats.pin_events += 1

    def _maybe_periodic_probe(
        self, active: _ActiveSlot, chunk_end_offset: int,
    ) -> None:
        """v0.7.13: run the periodic probe if we've passed
        ``next_probe_offset`` and we have enough baseband buffered.

        Outcomes:
          • Probe fires → refresh ``last_positive_probe_offset``.
            If state was REAPED, respawn decoders.
          • Probe doesn't fire AND
            (chunk_end - last_positive) > reap_after AND
            state == SPAWNED → reap all decoders, set state=REAPED.
        """
        if active.probe is None:
            return
        if chunk_end_offset < active.next_probe_offset:
            return
        # Schedule next probe interval regardless of outcome.
        active.next_probe_offset = (
            chunk_end_offset + self._probe_interval_samples
        )
        bb = active.probe_baseband
        min_n = active.probe.min_samples_required
        if bb is None or len(bb) < min_n:
            # Not enough baseband yet — let buffer fill, try again
            # on the next tick.
            return

        self._stats.periodic_probe_scans += 1

        # Multi-window scan over the tail of the buffer (where the
        # most recent samples live). Take the max-SNR per SF across
        # windows; collect the SET of SFs whose best-window SNR
        # crossed threshold (used for respawn-list filtering below).
        step = max(1, min_n // 4)
        n_windows = 4
        bb_end = len(bb)
        detected_sfs: set[int] = set()
        for w in range(n_windows):
            end = bb_end - w * step
            start = end - min_n
            if start < 0:
                break
            for r in active.probe.scan(bb[start:end]):
                if r.detected:
                    detected_sfs.add(r.sf)

        # v0.7.13 commit 1c: append probe outcome to history, trim
        # stale entries, evaluate pin hysteresis. We track on every
        # scan, regardless of detection, so the rate is over time
        # not per-event.
        self._update_pin_state(active, chunk_end_offset, bool(detected_sfs))

        if detected_sfs:
            self._stats.periodic_probe_positive += 1
            active.last_positive_probe_offset = chunk_end_offset
            if active.state == "reaped":
                # Re-spawn decoders for the detected SFs (plus any
                # confirmed (slot, sf) presets per commit 1b). The
                # channelizer has been running the whole time, so the
                # probe-baseband buffer already contains the chirp
                # the probe just detected — feeding it as "lookback"
                # gets the new decoders in sync immediately.
                self._periodic_respawn(
                    active, chunk_end_offset, detected_sfs,
                )
            return

        # No detection. Decide whether to reap.
        if active.state != "spawned":
            return
        # v0.7.13 commit 1c: pinned slots skip reap. The whole
        # point of the pin is to keep decoders alive on slots
        # where activity has been high enough that the spawn-cost
        # of re-spawning would outweigh the running cost.
        if active.pinned:
            self._stats.reap_skipped_pinned += 1
            return
        idle_for = chunk_end_offset - active.last_positive_probe_offset
        if idle_for >= self._reap_after_samples:
            self._periodic_reap(active, chunk_end_offset)

    def _periodic_reap(
        self, active: _ActiveSlot, chunk_end_offset: int,
    ) -> None:
        """v0.7.13: tear down decoders for a slot whose probe has
        been silent past ``reap_after_ms``.

        Channelizer keeps running (for periodic probe). The detector
        still considers this slot active (energy persists or has just
        cleared). On the next probe-positive, we respawn.
        """
        # Drain any pending packets first so we don't lose work the
        # decoder already finished.
        for preset_key, dec in list(active.decoders.items()):
            slot = active.slot_metadata[preset_key]
            for lp in dec.pop_packets():
                rebased = self._rebase(lp, active)
                self._stats.packets_decoded += 1
                if rebased.crc_ok:
                    self._record_crc_pass(slot, rebased)
                mesh_pkt = (self._mesh.decode(rebased.payload)
                             if rebased.crc_ok else None)
                if mesh_pkt and mesh_pkt.decrypted:
                    self._stats.packets_decrypted += 1
                self._pending.append(
                    PipelinePacket(slot=slot, lora=rebased, mesh=mesh_pkt)
                )
            self._fold_decoder_stats(dec, preset_key)
            self._stats.decoders_torn_down += 1
        active.decoders.clear()
        # NB: don't clear slot_metadata — we'll need it on respawn.
        active.state = "reaped"
        active.race_resolved = False    # respawn will re-race
        self._stats.periodic_reaps += 1

    def _periodic_respawn(
        self, active: _ActiveSlot, chunk_end_offset: int,
        detected_sfs: set[int],
    ) -> None:
        """v0.7.13: re-spawn decoders after a reap, triggered by a
        positive probe.

        Spawn list = (probe-detected SFs) ∪ (confirmed (slot, sf)
        presets per commit 1b). The probe-detected SF gets the live
        packet that just started; the confirmed SFs are there in
        case another known-good SF starts a packet within the next
        probe interval. SFs that have never produced a CRC-pass and
        weren't detected this round are NOT respawned — they wait
        for their own probe-fire.

        Feeds the recent probe-baseband as "lookback" so the new
        decoders see the chirp the probe just detected. The
        channelizer has been running continuously, so the buffer
        contains the freshest baseband.
        """
        key = (active.freq_hz, active.bandwidth_hz)
        presets = self._presets_by_slot.get(key, [])
        confirmed_at_slot = self._confirmed_presets_for_slot(
            active.freq_hz, active.bandwidth_hz,
        )
        confirmed_keys = {s.preset.key for s in confirmed_at_slot}
        # Build the spawn list: detected SFs plus confirmed SFs.
        spawn_list = [
            s for s in presets
            if s.preset.sf in detected_sfs
            or s.preset.key in confirmed_keys
        ]
        if not spawn_list:
            # Nothing to spawn — this can happen if the probe detected
            # an SF outside our preset list AND there are no confirmed
            # SFs. Stay reaped; next probe-positive may catch it.
            return
        # mix_freq isn't used in sharing mode but compute defensively.
        mix_freq = self._center_freq_hz - active.freq_hz
        for slot in spawn_list:
            self._spawn_decoder_for_slot(
                active, slot, use_sharing=True, mix_freq=mix_freq,
            )
        # Feed the probe-baseband buffer as the "lookback" so new
        # decoders see the recent chirp. Same shape as
        # _handle_activate's lookback feed.
        bb = active.probe_baseband
        if bb is not None and len(bb) > 0:
            floats = np.empty(2 * len(bb), dtype=np.float32)
            floats[0::2] = bb.real
            floats[1::2] = bb.imag
            for dec in active.decoders.values():
                dec.feed_baseband(floats)
        active.state = "spawned"
        self._stats.periodic_respawns += 1

    def _maybe_resolve_race(self, active: _ActiveSlot) -> None:
        """v0.7.7: SF racing — kill loser decoders once a winner has
        locked a preamble.

        Background: when a wide-BW slot fires (e.g. BW=250 in US),
        we spawn one decoder per matching SF preset because the
        detector can't tell SFs apart from energy alone (they share
        bandwidth). For BW=250 that's 5 decoders per activation
        (LONG_FAST, MEDIUM_SLOW, MEDIUM_FAST, SHORT_SLOW, SHORT_FAST).
        Only ONE of them matches the actual transmission's SF; the
        other 4 burn CPU running their chirp-search loop against a
        signal whose chirp slope doesn't match their reference,
        producing nothing but `detect_attempts`.

        Resolution mechanism (v0.7.7 final):
          1. After EVERY feed, check each decoder's preambles_found.
          2. If ANY decoder has preambles_found > 0, kill all
             decoders that DON'T. Done.
          3. If NO decoder has locked yet, do nothing — wait for
             a future feed.
          4. The slot's natural deactivation handles the case where
             no decoder ever locks (false-positive activation).

        Why no time deadline: the original v0.7.7 attempt used a
        fixed deadline (1.5–3× slowest preset's preamble window).
        That was the wrong abstraction. In real captures, the C
        decoder's chirp-search needs to scan through the lookback
        samples sequentially, which can take 100-300ms of input-
        time worth of advance even for fast SFs. A fixed deadline
        either kills the legitimate winner before it locks
        (deadline too short) or never fires before the slot
        deactivates naturally (deadline too long, no CPU saved).
        Racing-on-condition fires exactly when the answer is known
        with certainty: when the winner has DEMONSTRATED it can
        match the chirp slope.

        Why this is safe: a real Meshtastic transmission is at
        exactly one SF. If the signal is real, exactly one decoder
        will lock its preamble; the others' reference chirps don't
        match the signal's chirp rate, so their dechirp produces
        FFT noise across all bins instead of a clean peak. They
        can't possibly lock 8 consecutive matching upchirps in
        the same bin. So killing them when a winner exists loses
        no packets.
        """
        if active.race_resolved:
            return
        # Check who has locked at least one preamble.
        winners: list[str] = []
        for preset_key, dec in active.decoders.items():
            if dec.stats().preambles_found > 0:
                winners.append(preset_key)
        if not winners:
            # No one has locked yet. Wait for the next feed.
            return
        if len(winners) == len(active.decoders):
            # Everyone locked (extremely rare — would require all
            # SFs at the same center to have valid preambles, which
            # can only happen if multiple co-channel transmissions
            # at different SFs overlap in time. Shouldn't kill
            # anyone in this case.)
            self._stats.racing_wins += 1
            active.race_resolved = True
            return
        # Kill everyone NOT in winners. Drop python refs → C decoder
        # gets freed via __del__ → libc free chain.
        losers = [k for k in active.decoders if k not in winners]
        for k in losers:
            # Fold the loser's lifetime stats before drop so the
            # per-preset accumulator stays accurate. Loser will
            # have detect_attempts > 0 but preambles_found = 0,
            # which is a useful diagnostic for "this SF was never
            # going to match".
            self._fold_decoder_stats(active.decoders[k], k)
            del active.decoders[k]
            del active.slot_metadata[k]
            self._stats.decoders_torn_down += 1
            self._stats.racing_losers_killed += 1
        self._stats.racing_wins += 1
        active.race_resolved = True

    def _fold_decoder_stats(self, dec, preset_key: str) -> None:
        """v0.7.7: extracted helper — fold a decoder's lifetime stats
        into the per-preset cumulative accumulator before it gets
        torn down. Previously inlined in _handle_deactivate; lifted
        out so racing-kill paths can call it too.

        ``preset_key`` is the catalog key (e.g. "MEDIUM_FAST") that
        keys into ``self._per_preset_cumulative`` — must match the
        key used when the decoder was registered."""
        life = dec.stats()
        if preset_key in self._per_preset_cumulative:
            self._per_preset_cumulative[preset_key] = self._merge_stats(
                self._per_preset_cumulative[preset_key], life,
            )
        else:
            from copy import deepcopy
            self._per_preset_cumulative[preset_key] = deepcopy(life)

    def _handle_deactivate(self, event: SlotEvent) -> None:
        """Tear down decoders for the deactivated slot.

        Drains any remaining packets first, then frees the decoders.
        The detector's drain_frames timing should ensure the slot's
        last packet has finished by the time we get this event.
        """
        key = (event.slot_freq_hz, event.bandwidth_hz)
        active = self._active.pop(key, None)
        if active is None:
            return

        self._stats.slot_deactivations += 1

        # Drain remaining packets from each decoder, rebasing offsets,
        # then fold the decoder's per-life stats into our cumulative
        # per-preset accumulator (so the lifetime view via stats()
        # doesn't lose contributions from torn-down decoders).
        for preset_key, dec in active.decoders.items():
            slot = active.slot_metadata[preset_key]
            for lp in dec.pop_packets():
                rebased = self._rebase(lp, active)
                self._stats.packets_decoded += 1
                if rebased.crc_ok:
                    self._record_crc_pass(slot, rebased)
                mesh_pkt = (self._mesh.decode(rebased.payload)
                              if rebased.crc_ok else None)
                if mesh_pkt and mesh_pkt.decrypted:
                    self._stats.packets_decrypted += 1
                self._pending.append(PipelinePacket(
                    slot=slot, lora=rebased, mesh=mesh_pkt,
                ))
            # Fold this decoder's lifetime stats into the cumulative
            # per-preset accumulator before it gets garbage-collected.
            life = dec.stats()
            if preset_key in self._per_preset_cumulative:
                self._per_preset_cumulative[preset_key] = self._merge_stats(
                    self._per_preset_cumulative[preset_key], life,
                )
            else:
                from copy import deepcopy
                self._per_preset_cumulative[preset_key] = deepcopy(life)
            self._stats.decoders_torn_down += 1
        # v0.7.11: free the shared channelizer for this slot.
        if active.channelizer is not None:
            try:
                active.channelizer.close()
            except Exception:
                pass
            active.channelizer = None
        # Decoders are now garbage-collected; their underlying C state
        # is freed via __del__.

    def _record_crc_pass(self, slot: PresetSlot, lora_pkt) -> None:
        """v0.7.13 commit 1b: mark a (slot_freq, bw, sf) triple as
        'confirmed' after a CRC-pass — but ONLY if this pickup is the
        BEST (highest-RSSI) copy of its payload within the dedup
        window.

        Adjacent slot decoders pick up bleed-through from the actual
        TX slot — the bleed copies usually CRC-pass too because the
        chirp is wide enough to dechirp correctly even at ~125kHz
        offset, but their RSSI is 3-15dB lower than the true slot's
        pickup. Confirming the bleed slot would cause us to spawn
        decoders there forever, wasting CPU on a slot that doesn't
        host any actual TX.

        Algorithm:
          1. Find any recent CRC-passes (within
             _crc_dedup_window_samples) with the same payload bytes.
          2. If THIS pass has the highest RSSI in the cluster → add
             this (slot, sf) to confirmed; DEMOTE other cluster
             members from confirmed (if previously added).
          3. If a different cluster member has higher RSSI → don't
             add this (slot, sf). It's bleed.

        The fold-recent-passes window survives drain_packets calls
        (it's stored separately from _pending) so multi-chunk
        bleed comparisons work even when the user drains aggressively.

        Lifetime of confirmed set: pipeline-scoped. Confirmed triples
        survive detector deactivate / reactivate cycles.
        """
        triple = (slot.freq_hz, slot.preset.bandwidth_hz,
                  slot.preset.sf)
        rssi = lora_pkt.rssi_db
        offset = lora_pkt.sample_offset
        payload = lora_pkt.payload

        # Trim recent_crc_passes to the dedup window.
        cutoff = offset - self._crc_dedup_window_samples
        self._recent_crc_passes = [
            r for r in self._recent_crc_passes if r[0] >= cutoff
        ]

        # Find duplicates (same payload, within window).
        cluster = [r for r in self._recent_crc_passes
                   if r[1] == payload]

        # Add THIS pass to recent before deciding (so the cluster
        # snapshot below sees it too if we re-evaluate).
        record = (offset, payload, rssi,
                  slot.freq_hz, slot.preset.bandwidth_hz,
                  slot.preset.sf)
        self._recent_crc_passes.append(record)

        if not cluster:
            # First time seeing this payload — optimistically confirm.
            self._confirmed_slot_sf.add(triple)
            return

        # Compare RSSIs. Find the strongest member of the cluster
        # (including THIS pass).
        full_cluster = cluster + [record]
        best = max(full_cluster, key=lambda r: r[2])
        best_triple = (best[3], best[4], best[5])

        if best_triple == triple:
            # THIS pass is the cluster winner.
            self._confirmed_slot_sf.add(triple)
            # Demote any other (slot, sf) in this cluster that we
            # previously confirmed for this payload.
            for r in cluster:
                other_triple = (r[3], r[4], r[5])
                if other_triple != triple and other_triple in self._confirmed_slot_sf:
                    # Only demote if this OTHER triple has no
                    # OTHER higher-RSSI evidence in the recent log
                    # (i.e. the bleed-loser shouldn't lose its
                    # confirmation if it's the dedup winner of a
                    # DIFFERENT payload).
                    if not self._has_winning_evidence(other_triple):
                        self._confirmed_slot_sf.discard(other_triple)
        else:
            # A different (slot, sf) already in the cluster has
            # higher RSSI. Don't confirm THIS triple. Make sure
            # the WINNING triple is confirmed (in case it wasn't).
            self._confirmed_slot_sf.add(best_triple)
            # Note: we don't discard our own triple here even if
            # it was previously confirmed for THIS payload —
            # _has_winning_evidence will let it stay confirmed if
            # any OTHER payload has named it as winner.

    def _has_winning_evidence(
        self, triple: tuple[int, int, int],
    ) -> bool:
        """Does the recent_crc_passes log contain any payload for
        which `triple` is the highest-RSSI receiver?

        Used by _record_crc_pass to decide whether demoting a
        triple from confirmed is safe — a triple that's still the
        winner of some OTHER payload's cluster should stay
        confirmed.
        """
        # Group the recent log by payload, find the winner of each.
        by_payload: dict[bytes, tuple] = {}
        for r in self._recent_crc_passes:
            cur = by_payload.get(r[1])
            if cur is None or r[2] > cur[2]:
                by_payload[r[1]] = r
        for winner in by_payload.values():
            if (winner[3], winner[4], winner[5]) == triple:
                return True
        return False

    def _confirmed_presets_for_slot(
        self, freq_hz: int, bw_hz: int,
    ) -> list[PresetSlot]:
        """Return the subset of ``self._presets_by_slot[(freq, bw)]``
        whose (freq, bw, sf) triple has been confirmed by a past
        CRC-pass at this exact slot.

        Used at spawn points to ensure confirmed SFs always get a
        decoder, even when the probe-filter would have pruned them.
        """
        presets = self._presets_by_slot.get((freq_hz, bw_hz), [])
        return [
            s for s in presets
            if (freq_hz, bw_hz, s.preset.sf) in self._confirmed_slot_sf
        ]

    def _rebase(self, lp, active: _ActiveSlot):
        """Rebase a LoraPacket's sample_offset from decoder-local
        OUTPUT samples to absolute INPUT-stream samples.

        The C decoder reports ``sample_offset`` in OUTPUT
        (post-resampler) samples, counting from 0 at decoder
        creation. To get back to the absolute capture-stream offset:

          1. Multiply by os_factor = sample_rate / bandwidth to
             convert decoder-local output samples to decoder-local
             input samples.
          2. Add the global INPUT offset of the decoder's first-fed
             sample (``feed_start_offset``).

        Returns a new LoraPacket with the rebased offset; the
        original is left untouched.
        """
        from dataclasses import replace
        os_factor = self._sample_rate_hz // active.bandwidth_hz
        if os_factor < 1:
            os_factor = 1
        decoder_local_input = lp.sample_offset * os_factor
        absolute_input = active.feed_start_offset + decoder_local_input
        return replace(lp, sample_offset=absolute_input)

    def _max_preamble_ms_for_presets(
        self, presets: list[PresetSlot],
    ) -> float:
        """Compute the longest preamble duration in ms among the given
        presets.

        Preamble = 8 symbols × symbol_time_ms = 8 × (2^SF / BW * 1000).
        """
        max_ms = 0.0
        for s in presets:
            symbol_time_ms = (1 << s.preset.sf) / s.preset.bandwidth_hz * 1000
            preamble_ms = self.PREAMBLE_SYMBOLS * symbol_time_ms
            if preamble_ms > max_ms:
                max_ms = preamble_ms
        return max_ms

    def close(self) -> None:
        """Tear down all active decoders."""
        self._active.clear()
        self._pending.clear()
