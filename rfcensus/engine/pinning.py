"""Decoder→dongle pinning (v0.6.0).

A *pin* dedicates a dongle to one decoder at one frequency for the
entire session lifetime. Pinned dongles take a long-lived exclusive
lease at session bootstrap; the scheduler never sees them. Use this
when you want gap-free coverage of a specific target — e.g. dedicate
one of five RTL-SDRs to rtl_433 @ 433.92 MHz so every weather-station
beacon is captured, while the other four explore the rest of the
spectrum on the normal scan rotation.

Three entry points produce pins:

  • TOML config — persistent across sessions:
      [[dongles]]
      id = "00000043"
      [dongles.pin]
      decoder = "rtl_433"
      freq_hz = 433_920_000

  • CLI flag — one-session override / addition:
      rfcensus inventory --pin 00000043:rtl_433@433.92M
      rfcensus inventory --pin 00000043:rtl_433@433920000:2400000

  • Wizard — `rfcensus pin` (see commands/pin.py).

All three paths produce a list of `PinSpec` objects which is then
validated, allocated, and supervised.

Architecture:

  ┌─────────────────┐
  │ PinSpec (data)  │     parsed from config + CLI
  └────────┬────────┘
           │ validate_pins()
           ▼
  ┌─────────────────┐
  │ Validated pins  │     antenna covers freq, decoder exists,
  └────────┬────────┘     dongle is connected
           │ start_pinned_tasks()
           ▼
  ┌─────────────────┐     broker.allocate() exclusive lease,
  │ PinSupervisor   │     spawn supervisor task running decoder
  │  (asyncio.Task) │     in retry-with-backoff loop
  └────────┬────────┘
           │ session ends or stop_pinned_tasks()
           ▼
  ┌─────────────────┐
  │ Cancel + release│     supervisors cancelled, broker leases
  └─────────────────┘     released

The supervisor's backoff schedule is (1, 2, 5, 10, 60) seconds. After
5 consecutive failures it gives up and logs an error — the dongle stays
held (released only at session end) so the scheduler still won't claim
it; this matches the user's intent ("this dongle is dedicated to X").
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rfcensus.config.schema import PinConfig, SiteConfig
from rfcensus.hardware.broker import (
    AccessMode,
    DongleBroker,
    DongleLease,
    DongleRequirements,
    NoDongleAvailable,
)
from rfcensus.utils.logging import get_logger

if TYPE_CHECKING:
    from rfcensus.decoders.registry import DecoderRegistry
    from rfcensus.events import EventBus
    from rfcensus.hardware.registry import HardwareRegistry

log = get_logger(__name__)


# ────────────────────────────────────────────────────────────────────
# PinSpec — the single in-memory representation
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PinSpec:
    """A validated, ready-to-execute pin.

    Produced by `gather_pins()` from config + CLI flags. Immutable so
    runtime mutations (supervisor state, lease, etc.) live on
    `PinSupervisor` instead.
    """

    dongle_id: str
    decoder: str
    freq_hz: int
    sample_rate: int | None = None  # None → use decoder's preferred at runtime
    access_mode: AccessMode = AccessMode.EXCLUSIVE
    # Provenance — for error messages and the wizard's "where did this
    # come from?" affordances.
    source: str = "config"  # "config" | "cli"

    @property
    def consumer_label(self) -> str:
        """Human-readable identifier used in lease records and logs.

        Format: "pin:<decoder>@<freq_mhz>" — matches the format the
        wizard / list command displays so users can grep logs for it.
        """
        return f"pin:{self.decoder}@{self.freq_hz / 1e6:.3f}M"


# ────────────────────────────────────────────────────────────────────
# Parsing — from config and from CLI
# ────────────────────────────────────────────────────────────────────


# CLI format:
#   <dongle_id>:<decoder>@<freq>[:<sample_rate>]
# where:
#   <dongle_id>     – dongle id from site.toml (or serial — both accepted)
#   <decoder>       – decoder name (e.g. "rtl_433")
#   <freq>          – frequency. Accepts plain Hz (433920000), or with
#                     a unit suffix (433.92M, 162M, 850k). Case-insensitive.
#   <sample_rate>   – optional, plain Hz (2400000) or unit suffix (2.4M)
#
# Examples:
#   00000043:rtl_433@433.92M
#   00000043:rtl_433@433920000
#   00000043:rtl_433@433.92M:2.4M
#   00000043:rtl_433@433920000:2400000
_CLI_PIN_RE = re.compile(
    r"^"
    r"(?P<dongle>[A-Za-z0-9_\-]+)"
    r":"
    r"(?P<decoder>[A-Za-z0-9_]+)"
    r"@"
    r"(?P<freq>[0-9._]+(?:[kKmMgG])?)"
    r"(?::(?P<sample_rate>[0-9._]+(?:[kKmMgG])?))?"
    r"$"
)


def _parse_freq_str(s: str) -> int:
    """Parse a frequency / sample-rate string with optional unit suffix.

    Accepts: plain integer, integer with k/M/G suffix, decimal with
    k/M/G suffix. Underscores allowed as digit separators (matches
    Python int literal style).

    Returns Hz as int.
    """
    s = s.replace("_", "").strip()
    if not s:
        raise ValueError("empty frequency string")
    multipliers = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    suffix = s[-1].lower()
    if suffix in multipliers:
        try:
            value = float(s[:-1])
        except ValueError as e:
            raise ValueError(
                f"can't parse frequency {s!r}: {e}"
            ) from None
        return int(round(value * multipliers[suffix]))
    try:
        return int(s)
    except ValueError as e:
        raise ValueError(
            f"can't parse frequency {s!r}: expected an integer or a "
            f"value with k/M/G suffix"
        ) from None


def parse_cli_pin(spec: str) -> PinSpec:
    """Parse a single --pin CLI argument into a PinSpec.

    Raises ValueError with a helpful message on malformed input.
    """
    m = _CLI_PIN_RE.match(spec.strip())
    if not m:
        raise ValueError(
            f"--pin {spec!r} doesn't match expected format "
            f"'<dongle_id>:<decoder>@<freq>[:<sample_rate>]'. "
            f"Examples: '00000043:rtl_433@433.92M' or "
            f"'00000043:rtl_433@433920000:2400000'"
        )
    freq_hz = _parse_freq_str(m.group("freq"))
    sr_str = m.group("sample_rate")
    sample_rate = _parse_freq_str(sr_str) if sr_str else None
    return PinSpec(
        dongle_id=m.group("dongle"),
        decoder=m.group("decoder"),
        freq_hz=freq_hz,
        sample_rate=sample_rate,
        access_mode=AccessMode.EXCLUSIVE,  # CLI doesn't expose shared
        source="cli",
    )


def gather_pins(
    config: SiteConfig,
    cli_pin_strings: list[str] | None = None,
) -> list[PinSpec]:
    """Merge config-declared pins with CLI flag overrides.

    CLI pins take precedence per-dongle: if the same dongle_id appears
    in both config and CLI, the CLI version wins (and the config one
    is silently dropped). This matches user expectation that ad-hoc
    flags override persistent config.
    """
    pins_by_dongle: dict[str, PinSpec] = {}

    # Config first (lower precedence)
    for dongle_cfg in config.dongles:
        if dongle_cfg.pin is None:
            continue
        spec = _pin_spec_from_config(dongle_cfg.id, dongle_cfg.pin)
        pins_by_dongle[dongle_cfg.id] = spec

    # CLI overrides
    if cli_pin_strings:
        for raw in cli_pin_strings:
            spec = parse_cli_pin(raw)
            if spec.dongle_id in pins_by_dongle:
                log.info(
                    "CLI --pin for dongle %s overrides config pin",
                    spec.dongle_id,
                )
            pins_by_dongle[spec.dongle_id] = spec

    return list(pins_by_dongle.values())


def _pin_spec_from_config(dongle_id: str, pin: PinConfig) -> PinSpec:
    return PinSpec(
        dongle_id=dongle_id,
        decoder=pin.decoder,
        freq_hz=pin.freq_hz,
        sample_rate=pin.sample_rate,
        access_mode=(
            AccessMode.EXCLUSIVE
            if pin.access_mode == "exclusive"
            else AccessMode.SHARED
        ),
        source="config",
    )


# ────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Outcome of validating one pin against runtime context.

    Pins fall into three buckets:

      • ok           → schedule for execution
      • skip         → soft failure (dongle not connected etc.) — log
                       and continue; other pins still run
      • fatal        → hard failure (decoder unknown, antenna mismatch
                       without --allow-pin-antenna-mismatch) — caller
                       should refuse to start the session
    """

    spec: PinSpec
    status: str  # "ok" | "skip" | "fatal"
    reason: str = ""


def validate_pins(
    pins: list[PinSpec],
    registry: "HardwareRegistry",
    decoder_registry: "DecoderRegistry",
    *,
    allow_antenna_mismatch: bool = False,
) -> list[ValidationResult]:
    """Validate every pin against the live runtime.

    Checks per pin:

      1. Dongle exists in registry (skip if not — user might have
         unplugged it, no reason to abort the whole session)
      2. Dongle is healthy (skip if FAILED/UNAVAILABLE)
      3. Decoder is registered (FATAL — typo in config, fail fast so
         the user notices instead of running a session with a silently
         dropped pin)
      4. Dongle's frequency range covers freq_hz (FATAL — pinning to a
         frequency the hardware can't tune is always a bug)
      5. Antenna covers freq_hz (FATAL unless allow_antenna_mismatch=True
         — pinning is intentional, mismatches are almost always typos.
         Override exists for users who genuinely know they have a wide
         enough antenna that AntennaConfig's usable range underreports.)

    Returns one ValidationResult per input pin in the same order.
    Caller decides what to do with skip/fatal.
    """
    results: list[ValidationResult] = []
    known_decoders = set(decoder_registry.names())

    for spec in pins:
        # 1. Dongle exists
        dongle = registry.by_id(spec.dongle_id)
        if dongle is None:
            # Try by serial as a fallback — users sometimes use serials
            # interchangeably with ids.
            dongle = registry.by_serial(spec.dongle_id)
        if dongle is None:
            results.append(ValidationResult(
                spec, "skip",
                f"dongle {spec.dongle_id!r} not connected this session",
            ))
            continue

        # 2. Dongle healthy
        if not dongle.is_usable():
            results.append(ValidationResult(
                spec, "skip",
                f"dongle {spec.dongle_id} status is "
                f"{dongle.status.value}, not usable",
            ))
            continue

        # 3. Decoder exists
        if spec.decoder not in known_decoders:
            results.append(ValidationResult(
                spec, "fatal",
                f"decoder {spec.decoder!r} is not registered; "
                f"available: {sorted(known_decoders)}",
            ))
            continue

        # 4. Hardware frequency range covers
        if not dongle.covers(spec.freq_hz):
            low, high = dongle.capabilities.freq_range_hz
            results.append(ValidationResult(
                spec, "fatal",
                f"dongle {spec.dongle_id} ({dongle.model}) tunes "
                f"{low / 1e6:.0f}–{high / 1e6:.0f} MHz, can't reach "
                f"{spec.freq_hz / 1e6:.3f} MHz",
            ))
            continue

        # 5. Antenna covers (unless override)
        if not allow_antenna_mismatch:
            if dongle.antenna is None:
                results.append(ValidationResult(
                    spec, "fatal",
                    f"dongle {spec.dongle_id} has no antenna assigned; "
                    f"pin would receive nothing. Either assign an "
                    f"antenna or pass --allow-pin-antenna-mismatch.",
                ))
                continue
            if not dongle.antenna.covers(spec.freq_hz):
                results.append(ValidationResult(
                    spec, "fatal",
                    f"dongle {spec.dongle_id}'s antenna "
                    f"{dongle.antenna.name!r} doesn't cover "
                    f"{spec.freq_hz / 1e6:.3f} MHz. Either reassign "
                    f"the antenna, change the pin frequency, or pass "
                    f"--allow-pin-antenna-mismatch if you know better.",
                ))
                continue

        results.append(ValidationResult(spec, "ok"))

    return results


# ────────────────────────────────────────────────────────────────────
# Supervisor — long-running decoder with backoff
# ────────────────────────────────────────────────────────────────────


# Restart delay schedule. Walks through these on consecutive failures
# and then PLATEAUS at the last value forever — never gives up. The
# pin's lease stays held all session regardless of supervisor health,
# so a "give up" mode would just leave a dongle idle. Better to keep
# trying once a minute in case the dongle disappeared and came back
# (USB hiccup, decoder binary upgrade, etc).
_BACKOFF_DELAYS_S = (1.0, 2.0, 5.0, 10.0, 60.0)
# After this many consecutive identical errors, suppress further log
# emissions for the same error. Re-emit on success or on a different
# error type. Prevents a permanently-broken pin from spamming the log.
_DEDUP_AFTER_N_IDENTICAL = 3


@dataclass
class PinSupervisor:
    """Owns a pinned dongle's lease + supervised decoder loop.

    Created by `start_pinned_tasks()`. Holds the broker lease, the
    asyncio.Task running the supervisor loop, and a small amount of
    runtime state (failure count, last error). The supervisor loop
    runs `decoder.run(spec)` repeatedly until the task is cancelled.
    Failures retry forever (plateau at the longest backoff delay) —
    pinned dongles are dedicated; the user's intent is "keep trying."
    """

    spec: PinSpec
    lease: DongleLease
    task: asyncio.Task
    # Mutable counters updated by the supervisor loop
    attempts: int = 0
    successes: int = 0
    last_error: str = ""
    decodes_emitted: int = 0
    # Log-dedup state: if the same error repeats > _DEDUP_AFTER_N_IDENTICAL
    # times in a row, stop logging it. Re-emit on success or when the
    # error type/message changes.
    consecutive_identical_errors: int = 0
    suppression_announced: bool = False


async def _supervisor_loop(
    spec: PinSpec,
    lease: DongleLease,
    decoder_registry: "DecoderRegistry",
    event_bus: "EventBus",
    session_id: int,
    gain: str,
    state: PinSupervisor,
) -> None:
    """The actual restart-with-backoff loop.

    Cancellation: clean exit, no error logged. The caller (session
    teardown) is expected to cancel us.

    Crash handling: log (with dedup), increment attempts, sleep per
    backoff schedule (plateauing at the longest delay), retry forever.
    The lease is NOT released here — the caller releases it at session
    end. We never give up because the dongle is dedicated by user
    intent and an idle pinned dongle is no better than a crashed-and-
    retrying one.

    Log dedup: first `_DEDUP_AFTER_N_IDENTICAL` identical errors are
    logged normally; subsequent identical errors are suppressed (a
    one-time "suppressing further mentions" line is emitted at the
    boundary). Re-emit on success or when the error message changes.
    """
    from rfcensus.decoders.base import DecoderRunSpec

    decoder_cls = decoder_registry.get(spec.decoder)
    if decoder_cls is None:
        # Should have been caught by validate_pins; guard anyway.
        # Unlike runtime crashes, this is a config-level wrongness
        # that retries can't fix — just log and exit the loop.
        log.error(
            "pin supervisor for %s: decoder %r not registered, exiting",
            spec.consumer_label, spec.decoder,
        )
        state.last_error = f"decoder {spec.decoder} not registered"
        return

    decoder = decoder_cls()

    # Resolve sample rate: explicit pin override > decoder preferred
    sample_rate = spec.sample_rate or decoder.capabilities.preferred_sample_rate

    log.info(
        "pin supervisor starting: %s on dongle %s @ %.3f MHz, sr=%d",
        spec.decoder, spec.dongle_id,
        spec.freq_hz / 1e6, sample_rate,
    )

    while True:
        state.attempts += 1
        try:
            run_spec = DecoderRunSpec(
                lease=lease,
                freq_hz=spec.freq_hz,
                sample_rate=sample_rate,
                # duration_s=None → run until cancelled. Decoders that
                # honour None loop indefinitely (matches strategy.py's
                # behaviour for indefinite mode).
                duration_s=None,
                event_bus=event_bus,
                session_id=session_id,
                gain=gain,
                notes=f"pinned: {spec.consumer_label}",
            )
            result = await decoder.run(run_spec)
            state.successes += 1
            state.decodes_emitted += result.decodes_emitted
            # Success resets the dedup state: next failure (if any)
            # gets a fresh log line.
            if state.suppression_announced:
                log.info(
                    "pin supervisor: %s recovered after suppressed errors",
                    spec.consumer_label,
                )
            state.consecutive_identical_errors = 0
            state.suppression_announced = False
            state.last_error = ""
            # If the decoder returned cleanly without being cancelled,
            # it was likely a duration-limited run that fell through.
            # Loop and start it again — pinning means run forever.
            log.info(
                "pin supervisor: %s exited cleanly after %d decodes; "
                "restarting (pin = run forever)",
                spec.consumer_label, result.decodes_emitted,
            )
            # No backoff on clean exits; restart immediately.
            await asyncio.sleep(0.1)
            continue

        except asyncio.CancelledError:
            log.info(
                "pin supervisor: %s cancelled cleanly (%d decodes total)",
                spec.consumer_label, state.decodes_emitted,
            )
            raise

        except Exception as exc:
            error_str = f"{type(exc).__name__}: {exc}"

            # Dedup logic
            if error_str == state.last_error:
                state.consecutive_identical_errors += 1
            else:
                # New / different error: reset dedup, and if we were
                # previously suppressing, note the recovery into the
                # different-error mode so the user knows progress was
                # made (or that things changed in a different way).
                if state.suppression_announced:
                    log.info(
                        "pin supervisor: %s error type changed",
                        spec.consumer_label,
                    )
                state.consecutive_identical_errors = 1
                state.suppression_announced = False
                state.last_error = error_str

            if state.consecutive_identical_errors <= _DEDUP_AFTER_N_IDENTICAL:
                log.warning(
                    "pin supervisor: %s crashed (attempt %d, "
                    "consec-identical %d): %s",
                    spec.consumer_label, state.attempts,
                    state.consecutive_identical_errors, error_str,
                )
            elif not state.suppression_announced:
                log.warning(
                    "pin supervisor: %s — same error %d× in a row, "
                    "suppressing further mentions until success or new "
                    "error (will keep retrying every %.0fs)",
                    spec.consumer_label,
                    state.consecutive_identical_errors,
                    _BACKOFF_DELAYS_S[-1],
                )
                state.suppression_announced = True
            # else: suppressed, no log

            # Pick backoff delay. Walk through schedule on first N
            # consecutive failures, then plateau at the last value
            # forever. Use consecutive failures (= attempts - successes)
            # so a long-running success-then-fail stays at the short
            # end of the backoff.
            consecutive_failures = state.attempts - state.successes
            idx = min(consecutive_failures - 1, len(_BACKOFF_DELAYS_S) - 1)
            delay = _BACKOFF_DELAYS_S[idx]
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                # Cancellation during backoff — don't try the next
                # iteration, exit cleanly.
                raise


# ────────────────────────────────────────────────────────────────────
# Lifecycle: start + stop
# ────────────────────────────────────────────────────────────────────


@dataclass
class PinningOutcome:
    """Summary of what start_pinned_tasks did.

    Used by SessionRunner to log the outcome and decide whether to
    abort (fatal validation failures) or continue with degraded set.
    """

    supervisors: list[PinSupervisor] = field(default_factory=list)
    skipped: list[ValidationResult] = field(default_factory=list)
    fatal: list[ValidationResult] = field(default_factory=list)

    @property
    def has_fatal(self) -> bool:
        return bool(self.fatal)


async def start_pinned_tasks(
    pins: list[PinSpec],
    *,
    registry: "HardwareRegistry",
    broker: DongleBroker,
    decoder_registry: "DecoderRegistry",
    event_bus: "EventBus",
    session_id: int,
    gain: str = "auto",
    allow_antenna_mismatch: bool = False,
) -> PinningOutcome:
    """Validate, allocate, and supervise every pin.

    Returns immediately after spawning supervisor tasks (does not
    await them). Caller is expected to keep the returned PinningOutcome
    alive for the session's lifetime, then call `stop_pinned_tasks()`
    on it during teardown.

    On fatal validation, raises NoDongleAvailable with a consolidated
    message — no supervisors are started, no leases are held.
    """
    results = validate_pins(
        pins, registry, decoder_registry,
        allow_antenna_mismatch=allow_antenna_mismatch,
    )

    fatal = [r for r in results if r.status == "fatal"]
    if fatal:
        # Consolidated error: list every fatal failure so the user can
        # fix them all in one pass instead of whack-a-mole.
        msg = "Pin validation failed:\n" + "\n".join(
            f"  • {r.spec.dongle_id} → {r.spec.decoder}@"
            f"{r.spec.freq_hz / 1e6:.3f}M: {r.reason}"
            for r in fatal
        )
        raise NoDongleAvailable(msg)

    outcome = PinningOutcome()

    for r in results:
        if r.status == "skip":
            log.warning(
                "skipping pin %s → %s@%.3fM: %s",
                r.spec.dongle_id, r.spec.decoder,
                r.spec.freq_hz / 1e6, r.reason,
            )
            outcome.skipped.append(r)
            continue

        spec = r.spec
        # Resolve sample rate now so we can pass it to the broker
        decoder_cls = decoder_registry.get(spec.decoder)
        sample_rate = (
            spec.sample_rate
            or decoder_cls().capabilities.preferred_sample_rate
        )

        # Pins don't belong to a band — they're an explicit
        # decoder+freq pairing. Synthesize a `pin:<decoder>` tag so
        # the TUI's per-dongle tile can still display something
        # meaningful in its band slot rather than a blank.
        synthetic_band_id = f"pin:{spec.decoder}"

        requirements = DongleRequirements(
            freq_hz=spec.freq_hz,
            sample_rate=sample_rate,
            access_mode=spec.access_mode,
            prefer_driver=None,  # don't restrict; pin specifies the dongle
            require_suitable_antenna=False,  # already validated above
            band_id=synthetic_band_id,
        )
        # Force allocation onto the specific dongle. The broker doesn't
        # currently support a "must be this dongle" flag (we explored
        # adding require_dongle_id; not needed for MVP because the only
        # dongle satisfying these requirements that we want is the
        # pinned one — any other dongle covering this freq is fair
        # game for the scheduler). To guarantee the pinned dongle is
        # the one actually leased, we manually call the broker's
        # internal `_lease()` after asserting candidate availability.
        target = registry.by_id(spec.dongle_id) or registry.by_serial(spec.dongle_id)
        if target is None:
            # Should have been caught by validate; guard anyway.
            outcome.skipped.append(ValidationResult(
                spec, "skip", f"dongle {spec.dongle_id} disappeared",
            ))
            continue
        async with broker._lock:
            # Re-check availability under the lock so a parallel
            # bootstrap call (shouldn't happen in practice but cheap)
            # can't double-allocate.
            from rfcensus.hardware.dongle import DongleStatus
            if target.status in (DongleStatus.FAILED, DongleStatus.UNAVAILABLE):
                outcome.skipped.append(ValidationResult(
                    spec, "skip",
                    f"dongle {spec.dongle_id} became unusable before "
                    f"lease (status={target.status.value})",
                ))
                continue
            if target.id in broker._exclusive_holders:
                outcome.skipped.append(ValidationResult(
                    spec, "skip",
                    f"dongle {spec.dongle_id} already exclusively held "
                    f"(other consumer claimed it before pin bootstrap?)",
                ))
                continue
            if target.id in broker._shared_slots:
                outcome.skipped.append(ValidationResult(
                    spec, "skip",
                    f"dongle {spec.dongle_id} already running a shared "
                    f"slot — release it before pinning",
                ))
                continue
            lease = await broker._lease(
                target, requirements, consumer=spec.consumer_label,
            )

        # v0.6.4: publish the structured allocation event so the TUI's
        # DongleStrip sees pinned dongles light up. The broker's
        # `allocate()` path publishes this automatically; pinning
        # bypasses `allocate()` (it calls `_lease()` directly to
        # guarantee the specific dongle is leased) so we duplicate
        # the publish here. Outside the lock, like the broker does,
        # to avoid deadlocking subscribers.
        try:
            from rfcensus.events import HardwareEvent
            await event_bus.publish(HardwareEvent(
                dongle_id=target.id,
                kind="allocated",
                detail=f"pin lease {lease._lease_id} for {spec.consumer_label}",
                freq_hz=spec.freq_hz,
                sample_rate=sample_rate,
                consumer=spec.consumer_label,
                band_id=synthetic_band_id,
            ))
        except Exception:
            log.exception(
                "pin allocation event publish failed for %s", spec.dongle_id,
            )

        # Mark for UI / list / session attach metadata
        target.pin_holder = spec.consumer_label

        # Spawn supervisor task. We use create_task and stash the task
        # on the supervisor itself so the caller can cancel it via
        # `stop_pinned_tasks()`. State dataclass is shared between the
        # supervisor coroutine (which mutates it) and external code
        # (which reads it for metrics).
        state = PinSupervisor(spec=spec, lease=lease, task=None)  # type: ignore[arg-type]
        task = asyncio.create_task(
            _supervisor_loop(
                spec=spec, lease=lease,
                decoder_registry=decoder_registry,
                event_bus=event_bus,
                session_id=session_id,
                gain=gain,
                state=state,
            ),
            name=f"pin_supervisor:{spec.consumer_label}",
        )
        state.task = task
        outcome.supervisors.append(state)

        log.info(
            "pin started: %s → %s @ %.3f MHz "
            "(lease=%d, sr=%d, %s)",
            spec.dongle_id, spec.decoder,
            spec.freq_hz / 1e6, lease._lease_id, sample_rate,
            spec.access_mode.value,
        )

    return outcome


async def stop_pinned_tasks(
    outcome: PinningOutcome, broker: DongleBroker,
    *, timeout_s: float = 10.0,
) -> None:
    """Cancel supervisor tasks, await their exit, release leases.

    Idempotent — safe to call multiple times. Should be called during
    session teardown regardless of how the session ended (normal exit,
    SIGINT, exception).
    """
    for sup in outcome.supervisors:
        if sup.task and not sup.task.done():
            sup.task.cancel()

    # Await all cancellations with a bounded wait so a stuck supervisor
    # can't block teardown forever.
    if outcome.supervisors:
        pending = [s.task for s in outcome.supervisors if s.task]
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                log.warning(
                    "pin teardown: %d supervisor task(s) didn't exit in "
                    "%.0fs after cancellation",
                    sum(1 for s in outcome.supervisors
                        if s.task and not s.task.done()),
                    timeout_s,
                )

    # Release leases regardless of supervisor outcome — held leases
    # would prevent the broker from cleaning up rtl_tcp slots etc.
    for sup in outcome.supervisors:
        try:
            await broker.release(sup.lease)
        except Exception:
            log.exception(
                "pin teardown: error releasing lease %d for %s",
                sup.lease._lease_id, sup.spec.consumer_label,
            )
        # Clear UI marker
        sup.lease.dongle.pin_holder = None


# ────────────────────────────────────────────────────────────────────
# Helpers for callers (CLI commands, wizard, list output)
# ────────────────────────────────────────────────────────────────────


def summarize_pinning_outcome(outcome: PinningOutcome) -> list[str]:
    """Format the pinning outcome as a list of user-facing lines.

    Caller decides where to print (stderr, log, etc.). Returned lines
    are without trailing newlines.
    """
    lines: list[str] = []
    if outcome.supervisors:
        lines.append(
            f"Pinned {len(outcome.supervisors)} dongle(s) for the session:"
        )
        for sup in outcome.supervisors:
            lines.append(
                f"  • {sup.spec.dongle_id} → {sup.spec.decoder} @ "
                f"{sup.spec.freq_hz / 1e6:.3f} MHz"
                + (
                    f" (sample rate {sup.spec.sample_rate / 1e6:.2f} MHz)"
                    if sup.spec.sample_rate else ""
                )
            )
    if outcome.skipped:
        lines.append(f"Skipped {len(outcome.skipped)} pin(s):")
        for r in outcome.skipped:
            lines.append(
                f"  • {r.spec.dongle_id} → {r.spec.decoder}@"
                f"{r.spec.freq_hz / 1e6:.3f}M: {r.reason}"
            )
    return lines


def warn_if_all_dongles_pinned(
    outcome: PinningOutcome,
    registry: "HardwareRegistry",
) -> str | None:
    """Return a warning string if every usable dongle is pinned, else None.

    Allowed configuration ("I know exactly what I want, no exploration")
    but worth flagging at startup so users notice when they accidentally
    pin everything and wonder why nothing else is being scanned.
    """
    total = len(registry.usable()) + len(outcome.supervisors)
    if total == 0:
        return None
    if len(outcome.supervisors) >= total and outcome.supervisors:
        return (
            f"All {total}/{total} usable dongles are pinned — no "
            f"dongles available for the scheduler. Exploration scan "
            f"will be empty. (If that's intended, you're good.)"
        )
    return None
