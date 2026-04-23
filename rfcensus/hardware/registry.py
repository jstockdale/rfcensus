"""Hardware registry: consolidates probes across drivers.

Call `detect_hardware()` once at session start. The registry caches the
result for the duration of the process; re-probe with `detect_hardware(force=True)`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from rfcensus.config.schema import SiteConfig
from rfcensus.hardware.antenna import Antenna
from rfcensus.hardware.dongle import Dongle, DongleStatus
from rfcensus.hardware.drivers.hackrf import probe_hackrf
from rfcensus.hardware.drivers.rtlsdr import probe_rtlsdr
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class HardwareRegistry:
    """Holds detected hardware and tracks which dongles are configured."""

    dongles: list[Dongle] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    detected_at: datetime | None = None

    def by_id(self, dongle_id: str) -> Dongle | None:
        return next((d for d in self.dongles if d.id == dongle_id), None)

    def by_serial(self, serial: str) -> Dongle | None:
        """Return the unique dongle with this serial, or None.

        Returns None (not the first match) if multiple dongles share the
        serial — the caller must disambiguate by id.
        """
        matches = [d for d in self.dongles if d.serial == serial]
        if len(matches) == 1:
            return matches[0]
        return None

    def all_by_serial(self, serial: str) -> list[Dongle]:
        """Return every dongle with this serial. Useful for surfacing
        ambiguity to the user."""
        return [d for d in self.dongles if d.serial == serial]

    def usable(self) -> list[Dongle]:
        return [d for d in self.dongles if d.is_usable()]

    def mark_failed(self, dongle_id: str, reason: str = "") -> None:
        """Flag a dongle as failed so the broker stops allocating it.

        Used when a decoder exits unexpectedly and we suspect the device
        was disconnected or otherwise stopped responding. The dongle stays
        in the registry (so a future re-probe can restore it) but moves
        out of the `usable()` set.
        """
        for d in self.dongles:
            if d.id == dongle_id:
                if d.status == DongleStatus.FAILED:
                    return
                log.warning(
                    "marking dongle %s (serial=%s) as FAILED: %s",
                    dongle_id, d.serial or "?", reason or "no reason given",
                )
                d.status = DongleStatus.FAILED
                d.health_notes.append(reason or "marked failed")
                return
        log.debug("mark_failed: dongle %s not in registry", dongle_id)

    def mark_healthy(self, dongle_id: str) -> None:
        """Restore a dongle to HEALTHY status. Called by re-probe when a
        previously-failed dongle reappears in detection."""
        for d in self.dongles:
            if d.id == dongle_id:
                if d.status == DongleStatus.HEALTHY:
                    return
                log.info(
                    "restoring dongle %s (serial=%s) to HEALTHY (was %s)",
                    dongle_id, d.serial or "?", d.status.value,
                )
                d.status = DongleStatus.HEALTHY
                d.health_notes.append("restored to healthy after re-probe")
                return

    def covering(self, freq_hz: int) -> list[Dongle]:
        return [d for d in self.usable() if d.covers(freq_hz)]

    def apply_config(self, config: SiteConfig) -> list[str]:
        """Merge user's config into the detected dongle list.

        • Matches declared dongles to detected hardware by serial
        • Attaches antennas based on declared antenna refs
        • Returns human-readable warnings for mismatches
        """
        warnings: list[str] = []
        antennas_by_id = {a.id: Antenna.from_config(a) for a in config.antennas}

        for declared in config.dongles:
            detected = None
            ambiguous = False
            if declared.serial:
                detected = self.by_serial(declared.serial)
                if detected is None and self.all_by_serial(declared.serial):
                    # Multiple dongles share this serial; caller must use id.
                    ambiguous = True
            if detected is None:
                # Fall back to matching by id (works for disambiguated
                # duplicates like "rtlsdr-00000001-idx0", and useful for
                # testing / manual setup)
                detected = self.by_id(declared.id)
            if detected is None:
                if ambiguous:
                    matches = self.all_by_serial(declared.serial)
                    warnings.append(
                        f"declared dongle '{declared.id}' (serial={declared.serial}) "
                        f"is ambiguous — {len(matches)} dongles share that serial: "
                        f"{', '.join(d.id for d in matches)}. "
                        f"Update your config to use one of those ids instead of just the serial."
                    )
                else:
                    warnings.append(
                        f"declared dongle '{declared.id}' (serial={declared.serial}) not detected"
                    )
                continue

            # Rename detected dongle to use the declared id for consistency in logs
            detected.id = declared.id
            if declared.antenna:
                antenna = antennas_by_id.get(declared.antenna)
                if antenna is None:
                    warnings.append(
                        f"dongle '{declared.id}' references unknown antenna "
                        f"'{declared.antenna}'"
                    )
                else:
                    detected.antenna = antenna

            # Apply tcxo_ppm override from user config if meaningful
            if declared.tcxo_ppm and declared.tcxo_ppm > 0:
                # We can't mutate frozen DongleCapabilities so we wrap it
                new_caps = detected.capabilities
                if new_caps.tcxo_ppm != declared.tcxo_ppm:
                    from dataclasses import replace

                    detected.capabilities = replace(new_caps, tcxo_ppm=declared.tcxo_ppm)

        return warnings


_REGISTRY: HardwareRegistry | None = None


async def detect_hardware(force: bool = False) -> HardwareRegistry:
    """Probe attached hardware, populate the registry. Cached until `force=True`."""
    global _REGISTRY
    if _REGISTRY is not None and not force:
        return _REGISTRY

    log.info("detecting SDR hardware")
    rtl_result, hackrf_result = await asyncio.gather(
        probe_rtlsdr(), probe_hackrf()
    )
    all_dongles = [*rtl_result.dongles, *hackrf_result.dongles]
    diagnostics = [rtl_result.diagnostic, hackrf_result.diagnostic]

    # Detect duplicate serials and add a diagnostic. The probe layer has
    # already disambiguated the ids, but the user should know the serials
    # collide so they can program distinct ones via `rtl_eeprom`.
    diagnostics.extend(_duplicate_serial_diagnostics(all_dongles))

    now = datetime.now(timezone.utc)
    for d in all_dongles:
        d.last_health_check = now

    registry = HardwareRegistry(
        dongles=all_dongles,
        diagnostics=[d for d in diagnostics if d],
        detected_at=now,
    )
    _REGISTRY = registry
    return registry


async def reprobe_for_recovery(
    registry: HardwareRegistry,
    *,
    exclude: set[str] | None = None,
) -> tuple[int, int]:
    """Re-probe attached SDR hardware and reconcile against the registry.

    Used in long-running sessions to recover from transient USB issues —
    if a dongle was previously marked FAILED but now appears in detection,
    restore it to HEALTHY. Newly-attached dongles aren't added (we don't
    want to surprise a running session with new hardware mid-flight).

    `exclude` is a set of dongle ids that should NOT be restored even if
    they reappear in detection — used to honor "permanently failed"
    decisions made by the session's failure-cap logic.

    Returns (n_restored, n_still_missing).
    """
    from rfcensus.hardware.drivers.rtlsdr import probe_rtlsdr
    from rfcensus.hardware.drivers.hackrf import probe_hackrf

    exclude = exclude or set()
    rtl_result, hackrf_result = await asyncio.gather(
        probe_rtlsdr(), probe_hackrf()
    )
    detected_serials = {d.serial for d in rtl_result.dongles + hackrf_result.dongles if d.serial}
    detected_ids = {d.id for d in rtl_result.dongles + hackrf_result.dongles}

    n_restored = 0
    n_still_missing = 0
    for known in registry.dongles:
        if known.status != DongleStatus.FAILED:
            continue
        if known.id in exclude:
            # Permanently failed for this session — don't restore even
            # if the device reappears (likely flapping hardware)
            continue
        # Match by serial first (more stable across USB re-enumeration);
        # fall back to id for dongles without serials.
        is_back = (known.serial and known.serial in detected_serials) or (
            known.id in detected_ids
        )
        if is_back:
            registry.mark_healthy(known.id)
            n_restored += 1
        else:
            n_still_missing += 1
    return n_restored, n_still_missing


def _duplicate_serial_diagnostics(dongles: list[Dongle]) -> list[str]:
    """Build human-readable warnings for any serials that appear more than once.

    Cheap RTL-SDRs ship with serial '00000001' from the factory. If the
    user has multiple, they collide and we can't reliably tell which is
    which across reboots — driver_index is only stable per-boot.
    """
    by_serial: dict[str, list[Dongle]] = {}
    for d in dongles:
        if d.serial:
            by_serial.setdefault(d.serial, []).append(d)

    out: list[str] = []
    for serial, group in by_serial.items():
        if len(group) <= 1:
            continue
        ids = ", ".join(d.id for d in group)
        out.append(
            f"⚠ {len(group)} dongles share serial '{serial}': {ids}. "
            f"Driver index is stable per-boot but may shift if you replug. "
            f"To fix permanently: stop all SDR processes, then run "
            f"`rtl_eeprom -d N -s NEW_SERIAL` for each (use unique values "
            f"like 00000001, 00000002, ...). Power-cycle USB after writing."
        )
    return out


def reset_registry() -> None:
    """Drop cached hardware registry. Used by tests and `rfcensus doctor --rescan`."""
    global _REGISTRY
    _REGISTRY = None
