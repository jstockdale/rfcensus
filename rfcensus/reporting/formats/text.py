"""Plain text inventory report generator.

Produces a human-readable summary of a completed session: emitters found,
their classification + confidence, any anomalies, detections of known
technologies, active channels the power scan lit up without a decode,
and warnings.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from rfcensus.engine.session import SessionResult
from rfcensus.reporting.privacy import scrub_emitter
from rfcensus.storage.models import (
    ActiveChannelRecord,
    AnomalyRecord,
    DetectionRecord,
    EmitterRecord,
)


# How many "mystery carrier" lines to print per band before truncating.
# 10 is plenty to notice a pattern (e.g., "lots of activity at 915 that
# we can't decode") without drowning the report in noise. Long scans in
# busy RF environments easily generate 50-100 bins lit above threshold
# per band — showing them all makes the section longer than the actual
# emitter data, which buries the lede.
_MAX_UNTRACKED_CHANNELS_PER_BAND = 10

# Tolerance when matching an active channel against a known emitter
# frequency. The channel's bandwidth already gives us one envelope but
# we pad by this amount to handle cases where the power scan and the
# decoder report frequencies with slightly different conventions (e.g.,
# channel center vs actual carrier, or binning misalignment). 20 kHz
# comfortably covers typical narrowband channels without over-collapsing
# genuinely distinct carriers.
_FREQ_MATCH_TOLERANCE_HZ = 20_000


def render_text_report(
    result: SessionResult,
    emitters: list[EmitterRecord],
    anomalies: list[AnomalyRecord],
    detections: list[DetectionRecord] | None = None,
    active_channels: list[ActiveChannelRecord] | None = None,
    *,
    include_ids: bool = False,
    site_name: str = "default",
    previously_known_ids: set[int] | None = None,
) -> str:
    detections = detections or []
    active_channels = active_channels or []
    lines: list[str] = []
    lines.append("═" * 72)
    lines.append(f" rfcensus inventory report — session {result.session_id}")
    lines.append(f" site: {site_name}")
    lines.append(f" started: {_fmt(result.started_at)}")
    lines.append(f" ended:   {_fmt(result.ended_at)}")
    duration = (result.ended_at - result.started_at).total_seconds()
    lines.append(f" duration: {_humanize_duration(duration)}")
    lines.append("═" * 72)

    if not include_ids:
        lines.append("")
        lines.append(
            "Device IDs below are hashed. Re-run with --include-ids to see raw values."
        )

    lines.append("")
    lines.append(f"Plan: {len(result.plan.waves)} wave(s), "
                 f"{len(result.plan.tasks)} band task(s)")
    if result.plan.unassigned:
        lines.append(
            f"Unassigned bands (no dongle coverage): "
            f"{', '.join(result.plan.unassigned)}"
        )
    if result.plan.warnings:
        lines.append("Warnings:")
        for w in result.plan.warnings:
            lines.append(f"  • {w}")

    # Emitters grouped by protocol
    lines.append("")
    lines.append("─" * 72)
    lines.append(" Emitters detected")
    lines.append("─" * 72)

    if not emitters:
        lines.append("  (none)")
    else:
        by_protocol: dict[str, list[EmitterRecord]] = defaultdict(list)
        for e in emitters:
            by_protocol[e.protocol].append(e)

        confirmed_threshold = 3
        new_ids = previously_known_ids or set()

        for protocol in sorted(by_protocol.keys()):
            emitter_list = sorted(
                by_protocol[protocol], key=lambda r: r.confidence, reverse=True
            )
            confirmed = [
                e for e in emitter_list if e.observation_count >= confirmed_threshold
            ]
            tentative = [
                e for e in emitter_list if e.observation_count < confirmed_threshold
            ]
            lines.append("")
            lines.append(
                f"  {protocol}: {len(confirmed)} confirmed, {len(tentative)} tentative"
            )
            for e in emitter_list:
                badge = "✓" if e.observation_count >= confirmed_threshold else "?"
                new_mark = " [new]" if (e.id and e.id not in new_ids) else ""
                display = scrub_emitter(e, include_raw_ids=include_ids)
                rssi = (
                    f"{display.typical_rssi_dbm:+.1f} dBm"
                    if display.typical_rssi_dbm is not None
                    else "no RSSI"
                )
                freq_mhz = (
                    display.typical_freq_hz / 1_000_000 if display.typical_freq_hz else 0
                )
                lines.append(
                    f"    {badge} {display.device_id}"
                    f"  conf={display.confidence:.2f}"
                    f"  obs={display.observation_count}"
                    f"  {freq_mhz:.3f} MHz  {rssi}"
                    f"  [{display.classification or 'unclassified'}]"
                    f"{new_mark}"
                )

    # Detections (technologies identified for hand-off)
    if detections:
        lines.append("")
        lines.append("─" * 72)
        lines.append(" Technologies detected (hand off for deeper analysis)")
        lines.append("─" * 72)
        for d in detections:
            tools = ", ".join(d.hand_off_tools) if d.hand_off_tools else ""
            freq = f"{d.freq_hz / 1_000_000:.3f} MHz" if d.freq_hz else ""
            lines.append(
                f"  • {d.technology:20s} {freq}  conf={d.confidence:.2f}"
            )
            if d.evidence:
                lines.append(f"      evidence: {d.evidence}")
            if tools:
                lines.append(f"      suggested tools: {tools}")

    # Anomalies
    if anomalies:
        lines.append("")
        lines.append("─" * 72)
        lines.append(" Anomalies (worth investigating)")
        lines.append("─" * 72)
        for a in anomalies:
            freq = f"{a.freq_hz / 1_000_000:.3f} MHz" if a.freq_hz else ""
            lines.append(
                f"  • {a.kind:20s} {freq}  {a.description or ''}"
            )

    # Active channels without a decode — "mystery carriers"
    # v0.5.36: these are frequencies where the power scan saw activity
    # above the noise floor for long enough to be worth noticing, but
    # no decoder produced output and no detector fired a classification.
    # This closes the reporting gap users noticed where power_scan=yes
    # produced no visible output. See _select_untracked_channels below
    # for the exact filter.
    untracked = _select_untracked_channels(active_channels, emitters, detections)
    if untracked:
        lines.append("")
        lines.append("─" * 72)
        lines.append(" Mystery carriers (active, but nothing decoded)")
        lines.append("─" * 72)
        lines.append(
            "  Frequencies that lit up above the noise floor during "
            "power scans but produced no decoder output and no detector"
        )
        lines.append(
            "  classification. Persistent carriers here are the most "
            "interesting — they suggest a signal that's present but"
        )
        lines.append(
            "  that our current decoder/detector set doesn't recognize. "
            f"(Showing top {_MAX_UNTRACKED_CHANNELS_PER_BAND} per band "
            f"by persistence.)"
        )

        # Group by band. An active channel belongs to whichever band
        # was actually scanned (covers its center freq). Bands come
        # from the session plan so we only show ones the user knows
        # were scanned; anything outside known bands is bucketed under
        # "(outside scanned bands)" — unusual, but possible if a noisy
        # off-band bin leaked into the scan.
        bands_by_id = {}
        for task in result.plan.tasks:
            bands_by_id[task.band.id] = task.band

        by_band: dict[str, list[ActiveChannelRecord]] = defaultdict(list)
        for ch in untracked:
            band_id = _band_id_for_freq(bands_by_id.values(), ch.freq_center_hz)
            by_band[band_id or "(outside scanned bands)"].append(ch)

        for band_id in sorted(by_band.keys()):
            channels = by_band[band_id]
            # Sort by persistence_ratio desc (most interesting first),
            # falling back to peak power when persistence is None
            channels.sort(
                key=lambda c: (
                    c.persistence_ratio if c.persistence_ratio is not None else -1,
                    c.peak_power_dbm if c.peak_power_dbm is not None else -999,
                ),
                reverse=True,
            )
            lines.append("")
            lines.append(f"  {band_id}  ({len(channels)} active)")
            shown = channels[:_MAX_UNTRACKED_CHANNELS_PER_BAND]
            for ch in shown:
                lines.append("    " + _format_active_channel(ch))
            if len(channels) > len(shown):
                omitted = len(channels) - len(shown)
                lines.append(
                    f"    … {omitted} more in {band_id} not shown"
                )

    # Strategy summary
    lines.append("")
    lines.append("─" * 72)
    lines.append(" Execution summary")
    lines.append("─" * 72)
    for sr in result.strategy_results:
        lines.append(
            f"  {sr.band_id}: decoders={','.join(sr.decoders_run) or 'none'}"
            f"  power_scan={'yes' if sr.power_scan_performed else 'no'}"
            f"  decodes={sr.decodes_emitted}"
        )
        for err in sr.errors:
            lines.append(f"    ! {err}")

    lines.append("")
    lines.append(f"Total validated decodes: {result.total_decodes}")
    lines.append("═" * 72)

    return "\n".join(lines) + "\n"


def _fmt(dt: datetime) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _humanize_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _select_untracked_channels(
    active_channels: list[ActiveChannelRecord],
    emitters: list[EmitterRecord],
    detections: list[DetectionRecord],
) -> list[ActiveChannelRecord]:
    """Filter active_channels down to those NOT already surfaced as
    an emitter (from a decoder) or a detection (from a detector).

    Matching logic: a channel is "already covered" if any
    emitter/detection has a frequency falling within the channel's
    bandwidth (plus a small tolerance for binning/measurement drift).
    We intentionally use a generous envelope because a power-scan bin
    and a decoder-reported frequency are rarely pixel-perfect aligned
    — but two widely separated carriers won't match by accident at the
    tolerances used here.

    Example: an Interlogix sensor decoded at 433.534 MHz and the
    matching active-channel bin at 433.525 MHz with 10 kHz bandwidth
    will match (433.525 ± 5 kHz + 20 kHz tolerance = 433.500–433.550
    covers 433.534).
    """
    known_freqs: list[int] = []
    for e in emitters:
        if e.typical_freq_hz is not None:
            known_freqs.append(int(e.typical_freq_hz))
    for d in detections:
        if d.freq_hz is not None:
            known_freqs.append(int(d.freq_hz))

    if not known_freqs:
        return list(active_channels)

    untracked: list[ActiveChannelRecord] = []
    for ch in active_channels:
        half_bw = ch.bandwidth_hz // 2
        low = ch.freq_center_hz - half_bw - _FREQ_MATCH_TOLERANCE_HZ
        high = ch.freq_center_hz + half_bw + _FREQ_MATCH_TOLERANCE_HZ
        if any(low <= f <= high for f in known_freqs):
            continue
        untracked.append(ch)
    return untracked


def _band_id_for_freq(bands, freq_hz: int) -> str | None:
    """Return the id of the (first) band whose span covers `freq_hz`,
    or None if no band covers it. Used to group active channels by the
    band the user actually scanned."""
    for b in bands:
        if b.freq_low <= freq_hz <= b.freq_high:
            return b.id
    return None


def _format_active_channel(ch: ActiveChannelRecord) -> str:
    """One-line formatted representation of an active channel for the
    'Mystery carriers' section."""
    freq_mhz = ch.freq_center_hz / 1_000_000
    peak = (
        f"{ch.peak_power_dbm:+.1f} dBm"
        if ch.peak_power_dbm is not None else "? dBm"
    )
    floor = (
        f"{ch.noise_floor_dbm:+.1f} dBm"
        if ch.noise_floor_dbm is not None else "? dBm"
    )
    persist = (
        f"{ch.persistence_ratio * 100:.0f}%"
        if ch.persistence_ratio is not None else "?"
    )
    duration_s = max(0.0, (ch.last_seen - ch.first_seen).total_seconds())
    classification = ch.classification or "unclassified"
    return (
        f"{freq_mhz:10.3f} MHz  peak={peak}  floor={floor}  "
        f"persist={persist}  seen={_humanize_duration(duration_s)}  "
        f"[{classification}]"
    )
