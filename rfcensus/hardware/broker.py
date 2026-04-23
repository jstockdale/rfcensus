"""Dongle allocation broker.

Decoders and spectrum backends request dongles via the broker rather than
opening devices directly. This lets us:

• Track which dongles are in use
• Support exclusive access (dump1090, rtl_power) vs shared access via rtl_tcp
  (rtlamr, some rtl_433 configurations)
• Preempt lower-priority consumers if needed
• Clean up gracefully on crash

rtl_tcp sharing model
---------------------

When two or more consumers want to share a single RTL-SDR, the broker
spawns one `rtl_tcp` process bound to localhost on a unique port and
hands each consumer a `DongleLease` that points at that TCP endpoint.
The rtl_tcp server remains alive as long as at least one lease is
active. Sample rate and center frequency are coordinated: the server
runs at the max required sample rate, and consumers software-filter to
their individual sub-bands.

Exclusive leases (e.g. dump1090 opening the USB device directly) cannot
coexist with rtl_tcp; requesting one while the other is active fails.
"""

from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from rfcensus.events import EventBus, HardwareEvent
from rfcensus.hardware.dongle import Dongle, DongleStatus
from rfcensus.hardware.registry import HardwareRegistry
from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout
from rfcensus.utils.async_subprocess import ManagedProcess, ProcessConfig, which
from rfcensus.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


class AccessMode(str, Enum):
    EXCLUSIVE = "exclusive"  # Consumer opens device directly
    SHARED = "shared"  # Consumer connects via rtl_tcp


class NoDongleAvailable(RuntimeError):
    """Raised when no dongle satisfies requirements in the allowed time."""


@dataclass
class DongleRequirements:
    """What a consumer needs from a dongle."""

    freq_hz: int
    sample_rate: int = 2_400_000
    access_mode: AccessMode = AccessMode.EXCLUSIVE
    bias_tee: bool = False
    prefer_driver: str | None = None
    prefer_wide_scan: bool = False
    # If true, broker tries to select a dongle whose current antenna covers freq_hz
    require_suitable_antenna: bool = True


@dataclass
class DongleLease:
    """Handle returned by `broker.allocate`."""

    dongle: Dongle
    access_mode: AccessMode
    # For SHARED leases:
    rtl_tcp_host: str | None = None
    rtl_tcp_port: int | None = None
    # Consumer identity for logging
    consumer: str = ""
    # Internal tracking
    _lease_id: int = 0
    _released: bool = field(default=False, init=False)

    def endpoint(self) -> tuple[str, int] | None:
        if self.rtl_tcp_host and self.rtl_tcp_port:
            return self.rtl_tcp_host, self.rtl_tcp_port
        return None


@dataclass
class _SharedSlot:
    """Per-dongle state when rtl_tcp is running for shared access.

    Tracks center_freq_hz because rtl_tcp tunes a SINGLE frequency.
    Multiple clients can connect to the same rtl_tcp server but they
    all see IQ centered on this frequency with the sample_rate's
    instantaneous bandwidth. A new joining client whose frequency
    falls outside that window cannot usefully share — its decoder
    would operate on data from a different part of the spectrum.

    host/port point at our ASYNCIO FANOUT (RtlTcpFanout), not at
    the raw rtl_tcp socket. Osmocom's stock rtl_tcp only services one
    client at a time, so we insert a Python fanout relay that accepts
    many clients and broadcasts the single upstream stream to all of
    them. See rfcensus.hardware.rtl_tcp_fanout for the implementation.
    """

    process: ManagedProcess
    host: str                      # downstream (fanout-facing) host
    port: int                      # downstream (fanout-facing) port
    sample_rate: int
    center_freq_hz: int
    # Production slots ALWAYS have a fanout (set by _start_shared_slot).
    # None only in unit tests that construct slots directly to exercise
    # the compatibility predicate. Runtime code can rely on it being set.
    fanout: "RtlTcpFanout | None" = None
    rtl_tcp_port: int = 0
    # Leases currently using this slot
    lease_ids: set[int] = field(default_factory=set)


# Fraction of sample_rate usable for shared-slot frequency matching.
# 2.4 MHz SR has ~2.4 MHz nominal bandwidth, but anti-aliasing filter
# rolloff eats some at each edge. Using 0.8 × SR / 2 gives a safe
# usable window of ±960 kHz for a typical 2.4 MHz SR slot. New shared
# requests must fall within this window of the slot's center.
SHARED_SLOT_BANDWIDTH_FRACTION = 0.8


def _shared_slot_compatible(
    slot: "_SharedSlot", req_freq_hz: int, req_sample_rate: int
) -> bool:
    """Can a new shared consumer at req_freq_hz join this slot?

    Must satisfy both:
      • Slot's sample rate >= requested (we can't provide more resolution
        than what rtl_tcp was started with)
      • Requested frequency within slot's usable bandwidth window
    """
    if slot.sample_rate < req_sample_rate:
        return False
    half_window = SHARED_SLOT_BANDWIDTH_FRACTION * slot.sample_rate / 2
    return abs(req_freq_hz - slot.center_freq_hz) <= half_window


class DongleBroker:
    """Central allocator. One instance per rfcensus session."""

    def __init__(self, registry: HardwareRegistry, event_bus: EventBus):
        self.registry = registry
        self.event_bus = event_bus
        self._lock = asyncio.Lock()
        self._leases: dict[int, DongleLease] = {}
        self._shared_slots: dict[str, _SharedSlot] = {}  # by dongle.id
        self._exclusive_holders: dict[str, int] = {}  # dongle.id → lease_id
        self._next_lease_id = 1
        self._next_tcp_port = 1234

    async def allocate(
        self,
        requirements: DongleRequirements,
        consumer: str = "unknown",
        timeout: float = 30.0,
    ) -> DongleLease:
        """Allocate a dongle matching `requirements` for `consumer`.

        Raises `NoDongleAvailable` if no dongle can be allocated within timeout.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            event_payload = None
            allocated_lease = None
            async with self._lock:
                candidates = self._find_candidates(requirements)
                if candidates:
                    dongle = candidates[0]
                    allocated_lease = await self._lease(dongle, requirements, consumer)
                    event_payload = HardwareEvent(
                        dongle_id=dongle.id,
                        kind="allocated",
                        detail=f"lease {allocated_lease._lease_id} for {consumer}",
                    )
            # Publish OUTSIDE the lock to avoid the deadlock pattern
            # described in release().
            if allocated_lease is not None:
                if event_payload is not None:
                    await self.event_bus.publish(event_payload)
                return allocated_lease
            # Nothing available; wait and retry
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise NoDongleAvailable(
                    f"no dongle available for {consumer} matching {requirements}"
                )
            await asyncio.sleep(min(0.5, remaining))

    def _find_candidates(self, req: DongleRequirements) -> list[Dongle]:
        result: list[Dongle] = []
        # For SHARED requests, a BUSY dongle running a compatible shared
        # slot IS a candidate — joining that slot is free. `usable()`
        # excludes BUSY so we filter status ourselves and rely on the
        # exclusive_holders / shared_slots maps for actual availability.
        from rfcensus.hardware.dongle import DongleStatus
        for dongle in self.registry.dongles:
            if dongle.status in (DongleStatus.FAILED, DongleStatus.UNAVAILABLE):
                continue
            if not dongle.covers(req.freq_hz):
                continue
            if req.prefer_driver and dongle.driver != req.prefer_driver:
                # Not a hard filter, but deprioritize; we'll sort later
                pass
            if req.bias_tee and not dongle.capabilities.bias_tee_capable:
                continue
            if req.prefer_wide_scan and not dongle.capabilities.wide_scan_capable:
                pass  # Also deprioritized
            if req.access_mode == AccessMode.EXCLUSIVE:
                if dongle.id in self._exclusive_holders:
                    continue
                if dongle.id in self._shared_slots:
                    continue
            else:  # SHARED
                if dongle.id in self._exclusive_holders:
                    continue
                if not dongle.capabilities.can_share_via_rtl_tcp:
                    continue
                # If a shared slot already exists on this dongle, the
                # joining request must be compatible with it — both
                # sample rate AND frequency. Otherwise the slot is
                # effectively locked to its original user.
                slot = self._shared_slots.get(dongle.id)
                if slot and not _shared_slot_compatible(
                    slot, req.freq_hz, req.sample_rate
                ):
                    continue
            if req.require_suitable_antenna and dongle.antenna is not None:
                if not dongle.antenna.covers(req.freq_hz):
                    # Hard-exclude. Earlier this was a soft "deprioritize"
                    # but that defeated the entire flag — the broker would
                    # still hand out unsuitable dongles when nothing better
                    # was free, stealing them from tasks that genuinely
                    # needed them. Callers who want to bypass this (e.g.
                    # --all-bands) should pass require_suitable_antenna=False.
                    continue
            result.append(dongle)

        # Sort: best antenna suitability first, then wider capabilities
        def score(d: Dongle) -> tuple[int, float, int, int]:
            antenna_score = (
                d.antenna.suitability(req.freq_hz)
                if d.antenna
                else 0.2
            )
            driver_score = 1 if (req.prefer_driver and d.driver == req.prefer_driver) else 0
            wide_score = 1 if (req.prefer_wide_scan and d.capabilities.wide_scan_capable) else 0
            # Joining an existing shared slot is strictly better than
            # starting a new rtl_tcp server. For SHARED requests, rank
            # dongles with a matching existing slot FIRST — this is what
            # makes co-location (e.g. rtl_433 + rtlamr on same dongle
            # at same freq) actually work. Without this the broker
            # treats all candidates equally and allocates rtl_433 and
            # rtlamr to different dongles, defeating the point.
            has_matching_slot = 0
            if req.access_mode == AccessMode.SHARED:
                slot = self._shared_slots.get(d.id)
                # Prefer dongles with an already-running COMPATIBLE shared
                # slot so we don't spin up duplicate rtl_tcp instances
                # when one could serve multiple decoders. Only compatible
                # counts — a slot at a different frequency doesn't help.
                if slot and _shared_slot_compatible(
                    slot, req.freq_hz, req.sample_rate
                ):
                    has_matching_slot = 1
            return (has_matching_slot, antenna_score, driver_score, wide_score)

        result.sort(key=score, reverse=True)
        return result

    async def _lease(
        self, dongle: Dongle, req: DongleRequirements, consumer: str
    ) -> DongleLease:
        lease_id = self._next_lease_id
        self._next_lease_id += 1
        dongle.status = DongleStatus.BUSY

        if req.access_mode == AccessMode.EXCLUSIVE:
            self._exclusive_holders[dongle.id] = lease_id
            lease = DongleLease(
                dongle=dongle,
                access_mode=AccessMode.EXCLUSIVE,
                consumer=consumer,
                _lease_id=lease_id,
            )
        else:
            slot = self._shared_slots.get(dongle.id)
            if slot is None:
                # First shared consumer — start rtl_tcp tuned to their
                # frequency. Subsequent joiners must be compatible with
                # this center frequency (enforced in _find_candidates).
                slot = await self._start_shared_slot(
                    dongle, req.sample_rate, req.freq_hz,
                )
                self._shared_slots[dongle.id] = slot
            slot.lease_ids.add(lease_id)
            lease = DongleLease(
                dongle=dongle,
                access_mode=AccessMode.SHARED,
                rtl_tcp_host=slot.host,
                rtl_tcp_port=slot.port,
                consumer=consumer,
                _lease_id=lease_id,
            )

        self._leases[lease_id] = lease
        log.info(
            "allocated %s to %s as lease %d (%s)",
            dongle.id,
            consumer,
            lease_id,
            req.access_mode.value,
        )
        return lease

    async def _start_shared_slot(
        self, dongle: Dongle, sample_rate: int, center_freq_hz: int,
    ) -> _SharedSlot:
        if which("rtl_tcp") is None:
            raise NoDongleAvailable(
                "rtl_tcp binary not found; required for shared access"
            )

        # Allocate two ports: one for rtl_tcp (internal) and one for
        # the fanout (downstream-facing, seen by decoders).
        rtl_tcp_port = _pick_free_port(self._next_tcp_port)
        self._next_tcp_port = rtl_tcp_port + 1
        fanout_port = _pick_free_port(self._next_tcp_port)
        self._next_tcp_port = fanout_port + 1

        args = ["rtl_tcp", "-a", "127.0.0.1", "-p", str(rtl_tcp_port)]
        if dongle.driver_index is not None:
            args += ["-d", str(dongle.driver_index)]
        args += ["-s", str(sample_rate)]
        # Lock the initial tuning so joining clients see a predictable
        # center frequency. Without this, clients could each try to
        # retune and fight each other.
        args += ["-f", str(center_freq_hz)]

        proc = ManagedProcess(
            ProcessConfig(
                name=f"rtl_tcp[{dongle.id}:{rtl_tcp_port}]",
                args=args,
                log_stderr=True,
                # Bumped from DEBUG to INFO in v0.5.26. rtl_tcp's
                # startup output ("Found N device(s):", "Using device X",
                # "Tuner gain set to automatic.", "listening...") is
                # low-volume but ESSENTIAL for diagnosing fanout-header
                # timeouts. If rtl_tcp can't open the device or gets
                # stuck in libusb initialization, we need to see what
                # it says to know whether to blame the tuner, the
                # kernel, an orphan process, or our own logic.
                stderr_log_level="INFO",
            )
        )
        await proc.start()

        # NOTE (v0.5.26): previously we called wait_for_tcp_ready()
        # here to confirm rtl_tcp was accepting before the fanout
        # connected. That turned out to be actively harmful — osmocom
        # rtl_tcp is strictly single-client: it serializes accept()
        # calls through a single handler thread. wait_for_tcp_ready
        # opens a TCP connection (the "first client"), closes it
        # immediately, but rtl_tcp has already started processing it.
        # The fanout's subsequent connection queues behind that
        # ghost session, sometimes for 5+ seconds depending on how
        # rtl_tcp handles the half-closed state. Result: the fanout's
        # 5-second header-read timeout fires and the slot fails.
        # Diagnosed by the "did not send device-info header within
        # 5.0s" errors on first attempt that mysteriously resolved
        # on retry.
        #
        # The fix is to skip the probe entirely and let the fanout
        # be rtl_tcp's first and only client. rtl_tcp's tuner init
        # takes 1-3 seconds on typical hardware; if the fanout's
        # first connection attempt fails (because rtl_tcp isn't
        # listening yet), we retry with backoff up to _FANOUT_START_
        # RETRIES times. This gives us the wait-for-ready semantics
        # without burning the single-client slot.

        fanout = RtlTcpFanout(
            upstream_host="127.0.0.1",
            upstream_port=rtl_tcp_port,
            downstream_host="127.0.0.1",
            downstream_port=fanout_port,
            slot_label=f"fanout[{dongle.id}]",
        )
        last_err: Exception | None = None
        _FANOUT_START_RETRIES = 4
        _FANOUT_RETRY_BACKOFF_S = (0.5, 1.0, 2.0, 3.0)
        for attempt in range(_FANOUT_START_RETRIES):
            try:
                await fanout.start()
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < _FANOUT_START_RETRIES - 1:
                    backoff = _FANOUT_RETRY_BACKOFF_S[attempt]
                    log.info(
                        "fanout[%s] attempt %d/%d failed (%s); "
                        "retrying in %.1fs",
                        dongle.id, attempt + 1, _FANOUT_START_RETRIES,
                        e, backoff,
                    )
                    await asyncio.sleep(backoff)
        if last_err is not None:
            log.exception(
                "failed to start fanout for %s after %d attempts; "
                "tearing down rtl_tcp",
                dongle.id, _FANOUT_START_RETRIES,
            )
            try:
                await proc.stop()
            except Exception:
                pass
            raise NoDongleAvailable(
                f"fanout for {dongle.id} failed to start after "
                f"{_FANOUT_START_RETRIES} attempts: {last_err}"
            )

        actual_fanout_port = fanout.downstream_port

        slot = _SharedSlot(
            process=proc,
            fanout=fanout,
            host="127.0.0.1",
            port=actual_fanout_port,
            rtl_tcp_port=rtl_tcp_port,
            sample_rate=sample_rate,
            center_freq_hz=center_freq_hz,
        )
        log.info(
            "started rtl_tcp for %s: upstream=%s:%d, downstream=%s:%d "
            "(freq=%d Hz, sample_rate=%d)",
            dongle.id,
            slot.host, rtl_tcp_port,
            slot.host, actual_fanout_port,
            center_freq_hz,
            sample_rate,
        )
        return slot

    async def release(self, lease: DongleLease) -> None:
        if lease._released:
            return
        # Capture event payload while holding the lock; publish AFTER releasing
        # the lock. Holding a mutex across an await is an anti-pattern — if a
        # subscriber is slow, publish blocks, the lock is held forever, and
        # every other release/shutdown call blocks behind it. We saw this
        # cause sessions to hang at the end of a wave when multimon got
        # SIGKILL'd: lease 7's release entered the lock, blocked on publish,
        # broker.shutdown() then waited on the lock indefinitely.
        async with self._lock:
            lease._released = True
            lease_id = lease._lease_id
            dongle = lease.dongle

            if lease.access_mode == AccessMode.EXCLUSIVE:
                if self._exclusive_holders.get(dongle.id) == lease_id:
                    self._exclusive_holders.pop(dongle.id, None)
            else:
                slot = self._shared_slots.get(dongle.id)
                if slot:
                    slot.lease_ids.discard(lease_id)
                    if not slot.lease_ids:
                        log.info(
                            "stopping rtl_tcp + fanout for %s "
                            "(last lease released)", dongle.id,
                        )
                        # Fanout first so it stops feeding clients,
                        # then rtl_tcp so the USB handle is freed.
                        if slot.fanout is not None:
                            try:
                                await slot.fanout.stop()
                            except Exception:
                                log.exception(
                                    "error stopping fanout for %s", dongle.id,
                                )
                        await slot.process.stop()
                        self._shared_slots.pop(dongle.id, None)

            self._leases.pop(lease_id, None)
            if (
                dongle.id not in self._exclusive_holders
                and dongle.id not in self._shared_slots
            ):
                dongle.status = DongleStatus.HEALTHY

            event_payload = HardwareEvent(
                dongle_id=dongle.id,
                kind="released",
                detail=f"lease {lease_id} from {lease.consumer}",
            )

        # Publish OUTSIDE the lock so a slow subscriber can't deadlock us.
        await self.event_bus.publish(event_payload)
        log.info("released lease %d on %s", lease_id, dongle.id)

    async def shutdown(self) -> None:
        """Release everything. Called at session end."""
        async with self._lock:
            for slot in list(self._shared_slots.values()):
                if slot.fanout is not None:
                    try:
                        await slot.fanout.stop()
                    except Exception:
                        log.exception("error stopping fanout during shutdown")
                await slot.process.stop()
            self._shared_slots.clear()
            self._exclusive_holders.clear()
            self._leases.clear()


def _pick_free_port(start: int = 1234) -> int:
    """Find a free TCP port starting at `start`. Best-effort; racy by nature."""
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise NoDongleAvailable("no free TCP port for rtl_tcp in range")
