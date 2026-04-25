"""Strategies for investigating a band.

A strategy knows how to go from "here is a band" to "here are the decodes
and active channels we observed." Different strategies fit different
bands:

• `decoder_only` – just run decoders, assume we know what's there
• `decoder_primary` – run decoders plus a light power scan in parallel
• `power_primary` – power scan first, launch decoders at active channels
• `exploration` – power scan with anomaly reporting; decoders optional

Strategies run asynchronously and use the DongleBroker for hardware.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rfcensus.config.schema import BandConfig, SiteConfig
from rfcensus.decoders.base import DecoderBase
from rfcensus.decoders.registry import DecoderRegistry
from rfcensus.events import EventBus
from rfcensus.hardware.broker import (
    AccessMode,
    DongleBroker,
    DongleLease,
    DongleRequirements,
    NoDongleAvailable,
)
from rfcensus.spectrum.backend import SpectrumBackend, SpectrumSweepSpec
from rfcensus.spectrum.backends.hackrf_sweep import HackRFSweepBackend
from rfcensus.spectrum.backends.rtl_power import RtlPowerBackend
from rfcensus.spectrum.occupancy import OccupancyAnalyzer
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class StrategyContext:
    """Shared dependencies passed to each strategy invocation."""

    config: SiteConfig
    event_bus: EventBus
    broker: DongleBroker
    decoder_registry: DecoderRegistry
    session_id: int
    duration_s: float
    gain: str = "auto"  # "auto" or numeric dB string like "40"
    # When True (set via --all-bands), allocations bypass the broker's
    # antenna-suitability hard-exclusion. The user is opting into known
    # antenna mismatches and accepts likely poor reception.
    all_bands: bool = False


@dataclass
class StrategyResult:
    band_id: str
    decoders_run: list[str] = field(default_factory=list)
    power_scan_performed: bool = False
    decodes_emitted: int = 0
    errors: list[str] = field(default_factory=list)
    # Why the strategy ended. Used by the early-exit detector to
    # distinguish decoder-specific skip reasons ("binary_missing",
    # "rtl_tcp_not_ready", etc.) from real hardware failures. Written
    # by _run_decoder_on_band on various exit paths; default is the
    # empty string meaning "completed normally."
    ended_reason: str = ""


class Strategy(ABC):
    """Investigate one band."""

    @abstractmethod
    async def execute(
        self,
        band: BandConfig,
        ctx: StrategyContext,
        *,
        allowed_decoders: set[str] | None = None,
    ) -> StrategyResult:
        ...


# ------------------------------------------------------------
# Decoder-only
# ------------------------------------------------------------


class DecoderOnlyStrategy(Strategy):
    """Run all suitable decoders on a band. No power scan."""

    async def execute(
        self,
        band: BandConfig,
        ctx: StrategyContext,
        *,
        allowed_decoders: set[str] | None = None,
    ) -> StrategyResult:
        result = StrategyResult(band_id=band.id)
        decoders = _pick_decoders(band, ctx, allowed_decoders=allowed_decoders)
        if not decoders:
            result.errors.append(f"no usable decoders for {band.id}")
            return result

        tasks: list[asyncio.Task] = []
        for decoder in decoders:
            tasks.append(
                asyncio.create_task(
                    _run_decoder_on_band(band, decoder, ctx),
                    name=f"decoder-{decoder.name}-{band.id}",
                )
            )
            result.decoders_run.append(decoder.name)

        # v0.6.10: passive LoRa-survey piggyback. When this band has a
        # shared rtl_tcp fanout running for its decoders (e.g. 915_ism_
        # r900 has rtlamr at 912.6 MHz) AND the band config opts in via
        # `lora_survey = true`, attach a survey sidecar that taps the
        # same fanout. This expands LoRa coverage to wherever we're
        # already scanning in shared mode at zero extra hardware cost.
        # Without this, lora_survey only ran on the bands explicitly
        # using DecoderPrimaryStrategy, missing big chunks of the 915
        # ISM band (the user's metatron deployment used 913.125 MHz
        # which falls in the r900 band's window but not the 915_ism
        # primary window).
        _maybe_attach_lora_survey(band, ctx, tasks)

        decoder_results = await asyncio.gather(*tasks, return_exceptions=True)
        for dr in decoder_results:
            if isinstance(dr, Exception):
                result.errors.append(str(dr))
                continue
            if dr is None:
                continue
            # v0.6.12: distinguish decoder-task results from lora_survey
            # sidecar results. The survey returns a LoraSurveyStats
            # which has detections_emitted (a separate event stream),
            # not decodes_emitted. Without this branch we'd crash with
            # "'LoraSurveyStats' object has no attribute 'decodes_
            # emitted'" any time DecoderOnlyStrategy ran on a band
            # with lora_survey enabled (which is exactly what
            # 915_ism_r900 looks like as of v0.6.10).
            if hasattr(dr, "decodes_emitted"):
                result.decodes_emitted += dr.decodes_emitted
                result.errors.extend(dr.errors)
            elif hasattr(dr, "detections_emitted"):
                # LoraSurveyStats: surface errors but don't count
                # detections as decoder output (they go through the
                # event bus as DetectionEvents instead).
                if getattr(dr, "errors", None):
                    result.errors.extend(dr.errors)
        return result


def _maybe_attach_lora_survey(
    band: BandConfig,
    ctx: StrategyContext,
    tasks: list[asyncio.Task],
) -> None:
    """Attach a deferred lora-survey sidecar to `tasks` if the band
    opts in via `lora_survey = true`. Defers 2s so the primary
    fanout (started by the band's main decoders) has come up before
    the survey tries to allocate a shared lease against it.

    Idempotent — call from any strategy that wants the sidecar
    available; if `band.lora_survey` is False it's a no-op.

    v0.6.10: used by both DecoderPrimaryStrategy (the original 915_ism
    case) and DecoderOnlyStrategy (the new 915_ism_r900 case). When
    Phase 2 active channel-hop survey lands, that gets its own band/
    strategy with explicit dedicated decoders, separate from this
    passive sidecar path.
    """
    if not band.lora_survey:
        return

    async def _deferred_lora_survey():
        await asyncio.sleep(2.0)
        return await _run_lora_survey(band, ctx)

    tasks.append(
        asyncio.create_task(
            _deferred_lora_survey(),
            name=f"lora_survey-{band.id}",
        )
    )


# ------------------------------------------------------------
# Decoder-primary: decoders plus a light power scan
# ------------------------------------------------------------


class DecoderPrimaryStrategy(Strategy):
    """Decoders plus a power scan if a suitable backend is available."""

    async def execute(
        self,
        band: BandConfig,
        ctx: StrategyContext,
        *,
        allowed_decoders: set[str] | None = None,
    ) -> StrategyResult:
        result = StrategyResult(band_id=band.id)
        decoders = _pick_decoders(band, ctx, allowed_decoders=allowed_decoders)

        tasks: list[asyncio.Task] = []
        for decoder in decoders:
            tasks.append(
                asyncio.create_task(
                    _run_decoder_on_band(band, decoder, ctx),
                    name=f"decoder-{decoder.name}-{band.id}",
                )
            )
            result.decoders_run.append(decoder.name)

        if band.power_scan_parallel or _should_power_scan(band, ctx):
            # Defer the power-scan sidecar to give decoders first claim
            # on the wave's dongle pool. Without this, rtl_power can race
            # a shared-mode decoder like rtlamr for the only spare
            # antenna-suitable dongle, and whichever wins locks out the
            # other. Decoders are higher priority than the optional
            # spectrum measurement, so let them allocate first.
            async def _deferred_power_scan():
                await asyncio.sleep(1.0)
                return await _run_power_scan(band, ctx)

            tasks.append(
                asyncio.create_task(
                    _deferred_power_scan(),
                    name=f"powerscan-{band.id}",
                )
            )
            result.power_scan_performed = True

        # v0.6.5: LoRa survey sidecar. Taps the band's shared fanout
        # (already running for rtl_433 / rtlamr) and runs continuous
        # chirp-pattern detection on the streaming IQ. Bypasses the
        # rtl_power-based aggregator entirely — see lora_survey_task.py
        # for why that was structurally broken. Deferred 2s so the
        # primary fanout has come up and the shared lease can attach.
        # v0.6.10: shared launcher used by BOTH DecoderPrimaryStrategy
        # AND DecoderOnlyStrategy so e.g. the 915_ism_r900 band (which
        # uses decoder_only) can also attach a survey to its fanout.
        _maybe_attach_lora_survey(band, ctx, tasks)

        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        for tr in task_results:
            if isinstance(tr, Exception):
                result.errors.append(str(tr))
                continue
            if tr is None:
                continue
            if hasattr(tr, "decodes_emitted"):
                result.decodes_emitted += tr.decodes_emitted
                result.errors.extend(tr.errors)
            elif hasattr(tr, "detections_emitted"):
                # LoraSurveyStats: not a decoder, but its errors are
                # worth surfacing in the strategy result so they end
                # up in the session report.
                if getattr(tr, "errors", None):
                    result.errors.extend(tr.errors)
        return result


# ------------------------------------------------------------
# Power-primary: scan first, decoders on active channels
# ------------------------------------------------------------


class PowerPrimaryStrategy(Strategy):
    """Power scan the band; don't run decoders unless activity is found."""

    async def execute(
        self,
        band: BandConfig,
        ctx: StrategyContext,
        *,
        allowed_decoders: set[str] | None = None,
    ) -> StrategyResult:
        # allowed_decoders has no effect here — this strategy doesn't
        # run decoders directly. Kept in the signature so the session
        # loop can call `strategy.execute(band, ctx, allowed_decoders=...)`
        # uniformly regardless of strategy type.
        result = StrategyResult(band_id=band.id)
        scan_result = await _run_power_scan(band, ctx)
        result.power_scan_performed = True
        if scan_result and scan_result.errors:
            result.errors.extend(scan_result.errors)
        # TODO: inspect OccupancyAnalyzer state to decide which decoders to launch
        return result


# ------------------------------------------------------------
# Exploration: power scan + anomaly attention
# ------------------------------------------------------------


class ExplorationStrategy(Strategy):
    """Power scan with no decoding. For bands we can't meaningfully decode yet."""

    async def execute(
        self,
        band: BandConfig,
        ctx: StrategyContext,
        *,
        allowed_decoders: set[str] | None = None,
    ) -> StrategyResult:
        # allowed_decoders unused (no decoders run). Kept for signature
        # uniformity across Strategy subclasses.
        result = StrategyResult(band_id=band.id)
        scan_result = await _run_power_scan(band, ctx)
        result.power_scan_performed = True
        if scan_result and scan_result.errors:
            result.errors.extend(scan_result.errors)
        return result


# ------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------


def _pick_decoders(
    band: BandConfig,
    ctx: StrategyContext,
    *,
    allowed_decoders: set[str] | None = None,
) -> list[DecoderBase]:
    """Instantiate the decoders suitable for this band, filtered by
    suggested list and enablement.

    If `allowed_decoders` is provided (non-None), restrict the returned
    decoders to exactly that set. This is used by the v0.5.35 plan-time
    splitter to run only a subset of a band's decoders in a given wave
    — the surplus decoders that couldn't fit are run by a separate
    retry task in a later wave.
    """
    suggested = set(band.suggested_decoders)
    chosen: list[DecoderBase] = []
    for name in ctx.decoder_registry.names():
        if suggested and name not in suggested:
            continue
        if allowed_decoders is not None and name not in allowed_decoders:
            continue
        cls = ctx.decoder_registry.get(name)
        if cls is None:
            continue
        caps = cls.capabilities
        if not any(low <= band.freq_high and high >= band.freq_low for low, high in caps.freq_ranges):
            continue
        decoder = ctx.decoder_registry.instantiate(name, ctx.config)
        if decoder is None:
            continue
        chosen.append(decoder)
    return chosen


def _band_shared_sample_rate(
    band: BandConfig, ctx: StrategyContext
) -> int:
    """Decide the sample rate for the band's shared rtl_tcp slot
    deterministically, BEFORE any decoder allocates.

    v0.6.9: previously each decoder requested its own
    preferred_sample_rate and whoever allocated first established the
    slot rate. Late joiners with incompatible rate assumptions either
    silently produced wrong output (rate close enough that broker
    accepted but math wrong — rtlamr at 2,400,000 Hz vs its hardcoded
    2,359,296 Hz constants) or got disconnected by the v0.6.8 fanout
    filter (loud failure, but still a failure). Either way, the
    outcome depended on allocation order, which depended on async
    scheduling — non-deterministic.

    Now: the band knows which decoders WILL run on it. We compute the
    one rate that satisfies all of them up front:

      • If any decoder declares requires_exact_sample_rate, that rate
        wins. (If two such decoders disagree, we'd have an
        unsatisfiable band — currently this can't happen with our
        decoder set; if it ever does, we log loudly and pick one.)
      • Otherwise: max(preferred_sample_rate) across the decoders.
        Higher rate gives more bandwidth, and any decoder happy with
        a lower rate is also happy with a higher one (since
        min_sample_rate is a floor not a ceiling).

    Returns the chosen rate in Hz. Caller is responsible for using
    this rate everywhere — both in the broker allocation request AND
    in the DecoderRunSpec passed to the decoder. That ensures the
    decoder's command-line args (e.g. `rtl_433 -s 2359296`) match
    what the fanout's actually serving.
    """
    decoders = _pick_decoders(band, ctx)
    if not decoders:
        return 2_400_000  # band has no decoders; lora_survey-only path

    exact = [d.capabilities for d in decoders if d.capabilities.requires_exact_sample_rate]
    if exact:
        # All exact-rate decoders must agree, or the band is broken.
        rates = {d.preferred_sample_rate for d in exact}
        if len(rates) > 1:
            # Unsatisfiable — pick the lowest and warn. An admin
            # reading the log will see WHY their decode counts are
            # zero on this band.
            chosen = min(rates)
            log.warning(
                "band %s has %d decoders with conflicting exact-rate "
                "requirements: %s. Using %d Hz; other decoders will "
                "produce broken output. Move them to separate bands "
                "or different dongles.",
                band.id, len(exact), sorted(rates), chosen,
            )
            return chosen
        return rates.pop()

    return max(d.capabilities.preferred_sample_rate for d in decoders)


def _should_power_scan(band: BandConfig, ctx: StrategyContext) -> bool:
    """Should we run a power scan alongside decoders?

    Heuristic: yes if any dongle with HackRF or wide-scan capability is free,
    OR the band is wide enough that there's probably activity we can't decode.
    """
    if band.bandwidth_hz >= 5_000_000:
        return True
    for d in ctx.broker.registry.usable():
        if d.capabilities.wide_scan_capable:
            return True
    return False


async def _run_decoder_on_band(
    band: BandConfig, decoder: DecoderBase, ctx: StrategyContext
):
    # v0.6.9: ALL decoders on this band request the same shared-slot
    # rate, decided up-front from the band's full decoder cohort. Two
    # consequences:
    #   1. The broker creates the slot at this rate the first time it
    #      sees the band — order of allocation no longer matters.
    #   2. The DecoderRunSpec carries this rate too, so the decoder's
    #      command-line args (e.g. `rtl_433 -s 2359296`) match what
    #      the fanout's actually serving. Without this, rtl_433 would
    #      tell the upstream "I want 2,400,000" mid-stream and (a) get
    #      filtered by the v0.6.8 fanout if it conflicts, or (b)
    #      process samples assuming the wrong clock if it doesn't.
    band_rate = _band_shared_sample_rate(band, ctx)
    requirements = DongleRequirements(
        freq_hz=band.center_hz,
        sample_rate=band_rate,
        access_mode=decoder.capabilities.access_mode,
        prefer_driver="rtlsdr",
        require_suitable_antenna=not ctx.all_bands,
        band_id=band.id,
        require_exact_sample_rate=decoder.capabilities.requires_exact_sample_rate,
    )
    try:
        lease = await ctx.broker.allocate(
            requirements, consumer=f"{decoder.name}:{band.id}", timeout=10.0
        )
    except NoDongleAvailable as exc:
        log.warning("no dongle for %s@%s: %s", decoder.name, band.id, exc)
        return None
    try:
        from rfcensus.decoders.base import DecoderRunSpec

        run_spec = DecoderRunSpec(
            lease=lease,
            freq_hz=band.center_hz,
            sample_rate=band_rate,
            duration_s=ctx.duration_s,
            event_bus=ctx.event_bus,
            session_id=ctx.session_id,
            gain=ctx.gain,
            decoder_options=dict(band.decoder_options),
        )
        run_started = asyncio.get_event_loop().time()
        result = await decoder.run(run_spec)
        run_elapsed = asyncio.get_event_loop().time() - run_started

        # v0.6.7: surface ANY meaningfully early exit, not just the
        # < 5s "definitely hardware lost" case. We observed 74-122s
        # exits with -T 720 in v0.6.6 multi-decoder scans and had no
        # log line explaining why. Even if it's not a hardware loss,
        # the user wants to know "your decoder gave up early and
        # here's the elapsed time."
        if (
            ctx.duration_s
            and run_elapsed < ctx.duration_s * 0.5
            and result.ended_reason not in {
                "binary_missing", "rtl_tcp_not_ready",
                "wrong_lease_type", "user_skipped", "hardware_lost",
            }
        ):
            log.warning(
                "decoder %s on %s exited early at %.1fs "
                "(expected %.0fs, %d decode(s) emitted, errors=%s). "
                "Check the decoder's stderr above for clues. Common "
                "causes: rtl_tcp protocol mismatch, sample-rate "
                "negotiation conflict with another shared client, "
                "or rtl_433/rtlamr internal buffer overflow.",
                decoder.name, band.id, run_elapsed,
                ctx.duration_s, result.decodes_emitted, result.errors,
            )

        # Detect suspected hardware loss: decoder exited way before its
        # requested duration, with no decodes emitted. Most often this
        # means the dongle got unplugged or the USB connection died
        # mid-run. Flag the dongle so the broker stops handing it out;
        # the session's periodic re-probe (if active) can restore it
        # later if the user plugs it back in. The session also tracks
        # the failure for sidecar retry when the dongle reconnects.
        # Detect suspected hardware loss, with safeguards against false
        # positives. The early-exit-as-hardware-loss heuristic is right
        # most of the time but has these failure modes:
        #
        #   • binary_missing: decoder binary not installed → not hardware
        #   • rtl_tcp_not_ready: rtlamr couldn't connect to rtl_tcp →
        #     supporting service race, not hardware loss
        #   • wrong_lease_type: configuration mismatch, not hardware
        #
        # Plus: if the broker recently allocated the same dongle to
        # other consumers and those succeeded, the dongle is clearly
        # alive — this is a decoder-specific or band-specific issue.
        decoder_specific_exits = {
            "binary_missing", "rtl_tcp_not_ready", "wrong_lease_type",
            "user_skipped",
        }
        # v0.6.9: live-cohort escape hatch. Even if all the conditions
        # above match, if the dongle currently has OTHER active leases,
        # the dongle is clearly alive — this decoder's early exit is a
        # decoder-specific issue (e.g. shared-mode fanout disconnected
        # this client for a command conflict, while the other client
        # keeps streaming happily).
        #
        # Without this guard, v0.6.8's command-rejection feature
        # produced a regression: when the fanout disconnected rtlamr
        # for requesting set_sample_rate(2359296), rtlamr exited at
        # 0.1s, the heuristic mistakenly marked the dongle FAILED, and
        # subsequent shared-lease requests (e.g. lora_survey trying to
        # join the same fanout 7s later) couldn't find a healthy
        # dongle covering 915 MHz.
        #
        # active_lease_count includes this decoder's own lease (still
        # held until finally: release below), so > 1 means there's at
        # least one OTHER consumer holding a lease on this dongle.
        other_leases_active = (
            ctx.broker.active_lease_count(lease.dongle.id) > 1
        )
        if (
            ctx.duration_s
            and run_elapsed < min(5.0, ctx.duration_s * 0.1)
            and result.decodes_emitted == 0
            and result.ended_reason not in decoder_specific_exits
            and not other_leases_active
        ):
            log.warning(
                "⚠ decoder %s on %s exited after only %.1fs (expected %.0fs); "
                "dongle %s may be disconnected — marking as failed",
                decoder.name, band.id, run_elapsed, ctx.duration_s,
                lease.dongle.id,
            )
            ctx.broker.registry.mark_failed(
                lease.dongle.id,
                reason=f"decoder {decoder.name} exited early on {band.id}",
            )
            result.ended_reason = "hardware_lost"
            # Notify the session so it can queue a retry when the dongle
            # comes back online.
            from rfcensus.events import DecoderFailureEvent
            await ctx.event_bus.publish(DecoderFailureEvent(
                band_id=band.id,
                dongle_id=lease.dongle.id,
                decoder_name=decoder.name,
                elapsed_s=run_elapsed,
                remaining_s=max(0.0, ctx.duration_s - run_elapsed),
            ))
        elif (
            ctx.duration_s
            and run_elapsed < min(5.0, ctx.duration_s * 0.1)
            and result.decodes_emitted == 0
            and other_leases_active
        ):
            # Early exit, but the dongle has other active leases — log
            # the decoder-specific failure WITHOUT marking the dongle.
            # The cohort knowing the dongle is fine prevents the
            # cascade where every subsequent shared-lease allocation
            # for this dongle fails.
            log.warning(
                "decoder %s on %s exited after only %.1fs (expected %.0fs) "
                "but dongle %s has %d other active lease(s) — leaving "
                "dongle healthy. This is typically a decoder-specific "
                "issue (e.g. fanout disconnected this client for a "
                "command conflict; check earlier WARNING lines).",
                decoder.name, band.id, run_elapsed, ctx.duration_s,
                lease.dongle.id,
                ctx.broker.active_lease_count(lease.dongle.id) - 1,
            )

        log.info(
            "decoder %s on %s emitted %d decodes",
            decoder.name,
            band.id,
            result.decodes_emitted,
        )
        return result
    finally:
        await ctx.broker.release(lease)


async def _run_lora_survey(band: BandConfig, ctx: StrategyContext):
    """Run the continuous LoRa-survey sidecar on this band.

    Acquires a shared lease on the band's dongle (joining the existing
    fanout that rtl_433 / rtlamr are using), streams IQ continuously,
    runs an energy-gated chirp-pattern detector. Bypasses the
    rtl_power-based aggregator (which can't catch chirps because
    rtl_power sweeps bins sequentially — see lora_survey_task.py).

    Returns a `LoraSurveyStats` object that the strategy aggregator
    treats as a non-decoder task: stats.errors are surfaced but
    `decodes_emitted` is treated as 0 (LoRa detections are
    DetectionEvents, not decoder output).
    """
    from rfcensus.engine.lora_survey_task import LoraSurveyTask

    # v0.6.9: lora_survey joins the SAME shared slot the band's
    # decoders use, so it must request the same rate they did. Without
    # this, lora_survey would default to 2,400,000 Hz and (a) get
    # refused if the band's actual slot is at 2,359,296 Hz for rtlamr
    # compat, or (b) connect but interpret samples assuming the wrong
    # clock rate, throwing off chirp duration math by ~1.7%.
    band_rate = _band_shared_sample_rate(band, ctx)
    task = LoraSurveyTask(
        broker=ctx.broker,
        event_bus=ctx.event_bus,
        band=band,
        duration_s=ctx.duration_s,
        session_id=ctx.session_id,
        sample_rate=band_rate,
    )
    try:
        return await task.run()
    except asyncio.CancelledError:
        # Cancellation propagates from the wave-loop teardown; let
        # the task's finally block release the lease.
        await task.cancel()
        raise


async def _run_power_scan(band: BandConfig, ctx: StrategyContext):
    """Power scan a band using the best available backend.

    Pre-filters the candidate backend list by what driver hardware is
    actually present in the registry. This avoids the broker handing
    out an RTL-SDR for a HackRF-only backend like hackrf_sweep — that
    just causes lease churn and a confusing log trace.
    """
    backends: list[type[SpectrumBackend]] = []
    available_drivers = {d.driver for d in ctx.broker.registry.dongles}
    if "hackrf" in available_drivers:
        backends.append(HackRFSweepBackend)
    if "rtlsdr" in available_drivers:
        backends.append(RtlPowerBackend)

    for backend_cls in backends:
        try:
            lease = await _allocate_for_backend(backend_cls, band, ctx)
        except NoDongleAvailable:
            continue
        # Belt-and-suspenders: even with pre-filtering, the broker might
        # hand out a non-matching dongle in unusual cases (e.g., HackRF
        # detected but BUSY, broker falls back to RTL-SDR). available_on
        # still gates the actual use.
        if not backend_cls.available_on(lease):
            await ctx.broker.release(lease)
            continue

        backend = backend_cls()
        # v0.5.38: instantiate a WideChannelAggregator alongside the
        # OccupancyAnalyzer. The aggregator watches above-floor samples
        # directly (bypassing the OccupancyAnalyzer's 1-second hold
        # time that would otherwise miss short LoRa bursts), collects
        # per-bin activity in a rolling window, and emits a
        # WideChannelEvent when adjacent bins span a LoRa-standard
        # template width (125/250/500 kHz). This is what lets the
        # LoRa detector actually fire on Meshtastic traffic — without
        # aggregation, narrow 25 kHz bin events never match the LoRa
        # bandwidth heuristic. See spectrum/wide_channel_aggregator.py.
        from rfcensus.spectrum.wide_channel_aggregator import (
            WideChannelAggregator,
        )
        wide_aggregator = WideChannelAggregator(
            event_bus=ctx.event_bus,
            session_id=ctx.session_id,
            # v0.5.44: scan cadence for the background scanner task.
            # The task runs _scan_and_emit every scan_interval_s
            # (0.2 s = 5 Hz), independently of the observe() call rate
            # (which can be 1000+ Hz during rtl_power bursts).
            # Template scanning is CPU-heavy O(n·k) work; keeping it
            # off the observe hot path via start()/stop() + interior
            # yields prevents event-loop starvation that was killing
            # decoder fanout clients pre-v0.5.44.
            scan_interval_s=0.2,
        )
        analyzer = OccupancyAnalyzer(
            event_bus=ctx.event_bus,
            session_id=ctx.session_id,
            wide_aggregator=wide_aggregator,
        )
        spec = SpectrumSweepSpec(
            freq_low=band.freq_low,
            freq_high=band.freq_high,
            bin_width_hz=band.effective_power_scan_bin_hz,
            dwell_ms=200,
            duration_s=ctx.duration_s,
        )
        # v0.5.44: start the background scanner task so the CPU-heavy
        # template matching runs off the sample-processing hot path.
        # Critical for co-existing with decoder fanout writer tasks —
        # without this the scan blocks the event loop for 10-100 ms at
        # a time and downstream decoder clients (rtl_433, rtlamr) hit
        # their internal read timeouts and disconnect. See v0.5.44
        # changelog for the bug that motivated this.
        await wide_aggregator.start()
        try:
            samples_seen = await analyzer.consume(
                backend.sweep(lease, spec), dongle_id=lease.dongle.id
            )
        finally:
            # Stop the scanner task BEFORE releasing the lease — the
            # task may still be running a scan; stop() waits for
            # cancellation to complete before returning.
            await wide_aggregator.stop()
            await ctx.broker.release(lease)
        if samples_seen == 0:
            # Backend exited without emitting any samples — almost
            # always a dongle-open failure (USB error, device busy,
            # permissions) that the subprocess logged to stderr.
            # Surface this so users see WHICH dongle failed, not just
            # "lease released."
            log.warning(
                "%s produced 0 samples on %s for band %s — "
                "check dongle health or USB state",
                backend_cls.name, lease.dongle.id, band.id,
            )
        return type("Result", (), {"decodes_emitted": 0, "errors": []})()

    return None


async def _allocate_for_backend(
    backend_cls: type[SpectrumBackend], band: BandConfig, ctx: StrategyContext
) -> DongleLease:
    preferred_driver = "hackrf" if backend_cls is HackRFSweepBackend else "rtlsdr"
    return await ctx.broker.allocate(
        DongleRequirements(
            freq_hz=band.center_hz,
            sample_rate=2_400_000,
            access_mode=AccessMode.EXCLUSIVE,
            prefer_driver=preferred_driver,
            prefer_wide_scan=backend_cls is HackRFSweepBackend,
            require_suitable_antenna=not ctx.all_bands,
            band_id=band.id,
        ),
        consumer=f"{backend_cls.name}:{band.id}",
        timeout=5.0,
    )
