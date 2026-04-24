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

        decoder_results = await asyncio.gather(*tasks, return_exceptions=True)
        for dr in decoder_results:
            if isinstance(dr, Exception):
                result.errors.append(str(dr))
            elif dr is not None:
                result.decodes_emitted += dr.decodes_emitted
                result.errors.extend(dr.errors)
        return result


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
    requirements = DongleRequirements(
        freq_hz=band.center_hz,
        sample_rate=decoder.capabilities.preferred_sample_rate,
        access_mode=decoder.capabilities.access_mode,
        prefer_driver="rtlsdr",
        require_suitable_antenna=not ctx.all_bands,
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
            sample_rate=decoder.capabilities.preferred_sample_rate,
            duration_s=ctx.duration_s,
            event_bus=ctx.event_bus,
            session_id=ctx.session_id,
            gain=ctx.gain,
            decoder_options=dict(band.decoder_options),
        )
        run_started = asyncio.get_event_loop().time()
        result = await decoder.run(run_spec)
        run_elapsed = asyncio.get_event_loop().time() - run_started

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
        if (
            ctx.duration_s
            and run_elapsed < min(5.0, ctx.duration_s * 0.1)
            and result.decodes_emitted == 0
            and result.ended_reason not in decoder_specific_exits
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

        log.info(
            "decoder %s on %s emitted %d decodes",
            decoder.name,
            band.id,
            result.decodes_emitted,
        )
        return result
    finally:
        await ctx.broker.release(lease)


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
            event_bus=ctx.event_bus, session_id=ctx.session_id
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
        try:
            samples_seen = await analyzer.consume(
                backend.sweep(lease, spec), dongle_id=lease.dongle.id
            )
        finally:
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
        ),
        consumer=f"{backend_cls.name}:{band.id}",
        timeout=5.0,
    )
