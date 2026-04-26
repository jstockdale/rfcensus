"""Plain text inventory report generator.

Produces a human-readable summary of a completed session: emitters found,
their classification + confidence, any anomalies, detections of known
technologies, active channels the power scan lit up without a decode,
and warnings.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
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

# v0.6.2 — display tunables for the Mystery carriers section. Earlier
# iterations of this section were prone to two failure modes that
# combined to make the output unreadable in busy RF environments:
#
#   1. Bin spillover. A single carrier whose energy spans 3-5 adjacent
#      FFT bins produced 3-5 near-duplicate ActiveChannelRecord rows.
#      The "top 10 by persistence" list then showed the SAME carrier
#      under three slightly different frequencies. Fix: cluster
#      adjacent records within _ADJACENT_CLUSTER_WINDOW_HZ.
#
#   2. Saturated bands. Bands like business_uhf, frs_gmrs, and the P25
#      voice channels carry hundreds of legitimate (just-undecoded)
#      transmissions per scan. Listing the "top 10 mystery carriers"
#      from a band with 900 active channels surfaces nothing useful —
#      they're all 100% persistence at similar SNR, and they're not
#      mysterious anyway, just unsupported. Fix: when post-clustering
#      count exceeds _SATURATED_BAND_THRESHOLD, replace per-row
#      enumeration with a one-paragraph "this band is saturated"
#      summary that reports the band's overall character.

# Cluster window: ActiveChannelRecords whose centers fall within this
# distance of each other are treated as the same physical carrier
# scattered across adjacent FFT bins. Picked at 25 kHz because:
#   • Most narrowband ISM/OOK channels are 10-15 kHz wide; 25 kHz
#     comfortably covers center-of-bin drift across 2-3 adjacent bins
#     of a typical 5-10 kHz rtl_power sweep.
#   • Two genuinely-distinct OOK key fobs in the wild rarely sit
#     within 25 kHz of each other (frequency-agile transmitters spread
#     across the band).
#   • If we DO accidentally merge two real signals into one cluster
#     entry, the cluster's 'count' field is honest signal that
#     something interesting is there for the user to investigate.
_ADJACENT_CLUSTER_WINDOW_HZ = 25_000

# Above this many post-clustering carriers per band, we stop
# enumerating individual rows and switch to summary mode. Pulled from
# the empirical observation that bands with > ~50 active carriers are
# almost always carrying normal radio service activity (business radio,
# trunked voice, etc.) rather than genuinely mysterious signals.
_SATURATED_BAND_THRESHOLD = 50

# In summary mode, how many strongest-by-peak carriers to mention.
_MAX_STRONGEST_IN_SUMMARY = 5

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
    command_name: str = "inventory",
) -> str:
    detections = detections or []
    active_channels = active_channels or []
    lines: list[str] = []
    lines.append("═" * 72)
    # v0.6.15: name the report after the command that produced it.
    # Was hardcoded "inventory report" which was wrong/confusing for
    # `rfcensus scan` and `rfcensus hybrid` runs that share this code.
    lines.append(
        f" rfcensus {command_name} report — session {result.session_id}"
    )
    lines.append(f" site: {site_name}")
    lines.append(f" started: {_fmt(result.started_at)}")
    lines.append(f" ended:   {_fmt(result.ended_at)}")
    duration = (result.ended_at - result.started_at).total_seconds()
    lines.append(f" duration: {_humanize_duration(duration)}")
    # v0.7.4: surface early-termination prominently in the report
    # header so the user knows that absent emitters / undecided bands
    # are due to the scan being cut short, not confirmed-silent.
    if getattr(result, "stopped_early", False):
        lines.append("═" * 72)
        lines.append(" ⚠ INCOMPLETE — session stopped before plan finished")
        n_skipped = getattr(result, "tasks_skipped_due_to_stop", 0)
        n_total = len(result.plan.tasks)
        n_done = n_total - n_skipped
        lines.append(
            f" {n_done}/{n_total} planned task(s) executed; "
            f"{n_skipped} skipped due to early stop"
        )
        if n_skipped > 0:
            # List the bands that didn't run so the user knows what's
            # missing from the report (vs. truly silent).
            executed_bands = {
                r.band_id for r in result.strategy_results
            }
            skipped_bands = sorted({
                t.band.id for t in result.plan.tasks
                if t.band.id not in executed_bands
            })
            if skipped_bands:
                lines.append(
                    " skipped band(s): " + ", ".join(skipped_bands)
                )
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

    # v0.6.15: Diagnostics block. Surfaces fixable failure modes
    # near the TOP of the report so the user notices them before
    # scrolling through detected emitters / mystery carriers. Each
    # section corresponds to a different remediation path; if a
    # section is empty we omit it entirely so an all-clean report
    # stays compact.
    diag_lines = _render_diagnostics(result.strategy_results)
    if diag_lines:
        lines.append("")
        lines.append("─" * 72)
        lines.append(" Diagnostics")
        lines.append("─" * 72)
        lines.extend(diag_lines)

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
            "  classification. Adjacent FFT bins from the same carrier "
            "are clustered; bands with too many active carriers to be"
        )
        lines.append(
            "  meaningfully \"mysterious\" (typically business radio, "
            "trunked voice) are summarized rather than enumerated."
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
            lines.append("")
            band = bands_by_id.get(band_id)
            band_name = band.name if band is not None else None
            lines.extend(_render_band_mystery_section(
                band_id=band_id,
                band_name=band_name,
                channels=channels,
            ))

    # Strategy summary
    lines.append("")
    lines.append("─" * 72)
    lines.append(" Execution summary")
    lines.append("─" * 72)
    for sr in result.strategy_results:
        lines.append(
            f"  {sr.band_id}: power_scan={'yes' if sr.power_scan_performed else 'no'}"
            f"  decodes={sr.decodes_emitted}"
        )
        # v0.6.15: per-decoder breakdown so the user can tell
        # "everything ran fine, band was silent" from "one decoder
        # is missing its binary." Each line: name, decode count, why
        # it ended.
        if sr.decoder_runs:
            for dr in sr.decoder_runs:
                # ended_reason interpretation:
                #   ""               — completed normally (ran the wave)
                #   "duration"       — same: ran the wave, nothing decoded
                #   "binary_missing" — decoder binary not installed
                #   "wrong_lease_type" — lease/decoder mismatch (config bug)
                #   "error"          — exception / bad data
                #   "hardware_lost"  — USB unplug / dongle reset
                #   "cancelled"      — user-requested stop
                #   "upstream_eof"   — fanout died (often = upstream crash)
                reason = dr.ended_reason or "completed"
                marker = " " if dr.decodes_emitted > 0 else "·"
                # Flag fixable failure modes with ! so they jump out
                if reason in (
                    "binary_missing", "wrong_lease_type",
                    "error", "hardware_lost", "upstream_eof",
                ):
                    marker = "!"
                lines.append(
                    f"    {marker} {dr.name:<10s} decodes={dr.decodes_emitted}"
                    f"  ended={reason}"
                )
        elif sr.decoders_run:
            # Strategy didn't fill in the per-decoder breakdown
            # (older code path or unusual strategy). Fall back to the
            # bare decoder list so we don't lose info.
            lines.append(
                f"      decoders: {','.join(sr.decoders_run)}"
            )
        else:
            lines.append("      (no decoders configured)")
        for err in sr.errors:
            lines.append(f"    ! {err}")

    lines.append("")
    lines.append(f"Total validated decodes: {result.total_decodes}")
    lines.append("═" * 72)

    return "\n".join(lines) + "\n"


def _fmt(dt: datetime) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _render_diagnostics(strategy_results) -> list[str]:
    """v0.6.15: build the Diagnostics section.

    Three buckets, each gets its own paragraph if non-empty:
      • Decoders that couldn't start (binary missing / config bug).
        Fix path: install the binary or correct the band config.
      • Decoders that crashed mid-run (errors, hardware loss, eof).
        Fix path: investigate logs or hardware.
      • Bands where everything ran fine but produced 0 decodes.
        These are the "genuinely silent / nothing in the air" cases —
        not necessarily a bug, but worth knowing so you don't add
        decoders trying to fix what isn't broken.

    Empty list ⇒ caller skips the whole section header.
    """
    cant_start: list[tuple[str, str, str]] = []  # (band, decoder, reason)
    crashed: list[tuple[str, str, str, list[str]]] = []
    silent_bands: list[str] = []

    silent_reasons = {"", "completed", "duration", "cancelled"}
    cant_start_reasons = {"binary_missing", "wrong_lease_type"}
    crashed_reasons = {"error", "hardware_lost", "upstream_eof"}

    for sr in strategy_results:
        for dr in sr.decoder_runs:
            if dr.ended_reason in cant_start_reasons:
                cant_start.append((sr.band_id, dr.name, dr.ended_reason))
            elif dr.ended_reason in crashed_reasons:
                crashed.append(
                    (sr.band_id, dr.name, dr.ended_reason, dr.errors)
                )
        # A band counts as "silent" if it had decoders that all ran
        # to completion (no can't-start, no crashed) AND produced
        # zero decodes. We exclude bands that detected things via
        # detectors (lora_survey etc.) — silent means decoder-silent.
        if sr.decoders_run and sr.decodes_emitted == 0 and not any(
            dr.ended_reason in (cant_start_reasons | crashed_reasons)
            for dr in sr.decoder_runs
        ):
            if all(
                dr.ended_reason in silent_reasons
                for dr in sr.decoder_runs
            ) and sr.decoder_runs:
                silent_bands.append(sr.band_id)

    out: list[str] = []
    if cant_start:
        out.append("  Decoders that couldn't start "
                   "(install the binary or fix config):")
        for band_id, name, reason in cant_start:
            out.append(f"    ! {band_id}/{name} — {reason}")
        out.append("")
    if crashed:
        out.append("  Decoders that crashed mid-run "
                   "(check logs / hardware):")
        for band_id, name, reason, errors in crashed:
            err_summary = errors[0] if errors else "(no error message)"
            out.append(f"    ! {band_id}/{name} — {reason} — {err_summary}")
        out.append("")
    if silent_bands:
        out.append("  Bands silent for the full wave "
                   "(decoders ran fine, nothing in the air):")
        # Wrap to ~60 chars per line so this stays readable for a
        # site like metatron with 8+ silent bands
        chunk = "    " + ", ".join(silent_bands)
        out.append(chunk)
        out.append("")
    # Trim trailing blank line
    while out and out[-1] == "":
        out.pop()
    return out


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


# ────────────────────────────────────────────────────────────────────
# v0.6.2 — clustering, scoring, and saturated-band display
# ────────────────────────────────────────────────────────────────────


@dataclass
class _ChannelCluster:
    """Group of ActiveChannelRecords merged because they're within
    `_ADJACENT_CLUSTER_WINDOW_HZ` of each other.

    Represents what's almost certainly one physical carrier whose
    energy spilled across multiple adjacent FFT bins. The cluster's
    aggregate stats (max peak, min floor, max persistence) describe
    the underlying signal more accurately than any single bin row.
    """

    members: list[ActiveChannelRecord] = field(default_factory=list)

    @property
    def freq_low_hz(self) -> int:
        return min(m.freq_center_hz - m.bandwidth_hz // 2 for m in self.members)

    @property
    def freq_high_hz(self) -> int:
        return max(m.freq_center_hz + m.bandwidth_hz // 2 for m in self.members)

    @property
    def representative_freq_hz(self) -> int:
        """Use the highest-peak member's frequency as the representative.
        Most likely the actual transmitter center; the spillover bins
        sit in the skirt."""
        # `key=` so we don't fall through to comparing ActiveChannelRecords
        # when peaks tie (records aren't orderable).
        winner = max(
            self.members,
            key=lambda m: m.peak_power_dbm if m.peak_power_dbm is not None else -999,
        )
        return winner.freq_center_hz

    @property
    def peak_power_dbm(self) -> float | None:
        peaks = [m.peak_power_dbm for m in self.members if m.peak_power_dbm is not None]
        return max(peaks) if peaks else None

    @property
    def noise_floor_dbm(self) -> float | None:
        # Min floor = quietest neighbour, best estimate of the true floor
        # under this carrier (some bins have higher floors due to the
        # carrier itself raising the local noise estimate).
        floors = [m.noise_floor_dbm for m in self.members if m.noise_floor_dbm is not None]
        return min(floors) if floors else None

    @property
    def persistence_ratio(self) -> float | None:
        ratios = [m.persistence_ratio for m in self.members if m.persistence_ratio is not None]
        return max(ratios) if ratios else None

    @property
    def sample_count(self) -> int | None:
        """Max sample_count across members — the best evidence any
        merged bin has. Returns None if no member tracks sample_count
        (e.g. rows from a pre-v0.6.3 database, or synthetic records
        in tests that didn't set the field)."""
        counts = [m.sample_count for m in self.members if m.sample_count is not None]
        return max(counts) if counts else None

    @property
    def first_seen(self) -> datetime:
        return min(m.first_seen for m in self.members)

    @property
    def last_seen(self) -> datetime:
        return max(m.last_seen for m in self.members)

    @property
    def classification(self) -> str | None:
        """Most common non-None classification across members."""
        cls = [m.classification for m in self.members if m.classification]
        if not cls:
            return None
        counts = Counter(cls)
        return counts.most_common(1)[0][0]

    @property
    def count(self) -> int:
        """How many raw bins were merged. count > 1 = bin spillover."""
        return len(self.members)


def _cluster_adjacent_channels(
    channels: list[ActiveChannelRecord],
    *,
    window_hz: int = _ADJACENT_CLUSTER_WINDOW_HZ,
) -> list[_ChannelCluster]:
    """Merge ActiveChannelRecords whose centers fall within `window_hz`
    of each other into shared clusters.

    O(n log n): sort by frequency, then walk linearly, growing the
    current cluster while the next channel's center is within
    `window_hz` of the previous channel's center.

    Note that this clusters by CENTER-to-CENTER distance only; we
    don't try to be clever about asymmetric spillover. A 25 kHz
    window comfortably covers the typical 2-3 adjacent rtl_power bins
    that one narrowband carrier produces.
    """
    if not channels:
        return []

    sorted_chs = sorted(channels, key=lambda c: c.freq_center_hz)
    clusters: list[_ChannelCluster] = []
    current = _ChannelCluster(members=[sorted_chs[0]])
    for ch in sorted_chs[1:]:
        prev = current.members[-1]
        if ch.freq_center_hz - prev.freq_center_hz <= window_hz:
            current.members.append(ch)
        else:
            clusters.append(current)
            current = _ChannelCluster(members=[ch])
    clusters.append(current)
    return clusters


# v0.6.3 — sample-count threshold below which persistence ratios
# are considered unreliable. The mystery_score multiplies in a
# low-confidence factor when the cluster's best member has fewer
# than this many observations, so a bin that popped once and got
# a coincidental 100% persistence doesn't outrank a sustained
# carrier that was observed for thousands of sweeps.
_MYSTERY_MIN_CONFIDENT_N = 30


def _mystery_score(cluster: _ChannelCluster) -> float:
    """Composite "interestingness" score for ranking mystery carriers.

    persistence × min(snr, 30) / 30 × confidence_from_n

    Persistence-only ranking ties at 100% across most strong carriers
    in busy bands and is therefore useless. Combining persistence with
    SNR (capped at 30 dB so a +50 dBm intermod product doesn't dominate
    a +20 dBm legitimate carrier) gives a usable ordering: consistent
    AND clearly above the floor scores high; a brief weak burst or a
    long-running carrier just above the floor scores low.

    The sample-count factor (v0.6.3) further protects against
    coincidental high-persistence scores on bins that were only
    observed a handful of times — sample_count below _MYSTERY_MIN_CONFIDENT_N
    scales the score toward zero so well-observed real carriers
    outrank briefly-seen anomalies.
    """
    p = cluster.persistence_ratio or 0.0
    peak = cluster.peak_power_dbm
    floor = cluster.noise_floor_dbm
    if peak is None or floor is None:
        snr_factor = 0.5  # treat unknown SNR as middling
    else:
        snr = peak - floor
        snr_factor = max(0.0, min(snr, 30.0)) / 30.0

    # Confidence factor based on observation count. If sample_count is
    # unavailable (None from pre-v0.6.3 data or from tests), assume
    # middling confidence so scores don't collapse across the board.
    n = cluster.sample_count
    if n is None:
        n_factor = 1.0  # unknown → don't penalize (backward-compatible)
    else:
        n_factor = min(1.0, n / _MYSTERY_MIN_CONFIDENT_N)

    return p * snr_factor * n_factor


def _format_cluster(cluster: _ChannelCluster) -> str:
    """One-line representation of a clustered carrier for the report.

    Single-bin clusters look like the v0.5.36 single-channel output
    (so simple cases stay readable). Multi-bin clusters add a span
    annotation and the bin count.
    """
    rep = cluster.representative_freq_hz / 1_000_000
    peak = (
        f"{cluster.peak_power_dbm:+.1f} dBm"
        if cluster.peak_power_dbm is not None else "? dBm"
    )
    floor = (
        f"{cluster.noise_floor_dbm:+.1f} dBm"
        if cluster.noise_floor_dbm is not None else "? dBm"
    )
    persist = (
        f"{cluster.persistence_ratio * 100:.0f}%"
        if cluster.persistence_ratio is not None else "?"
    )
    # v0.6.3: show sample count when available so users can distinguish
    # "100% persistence from 3 samples" (weak) from "100% persistence
    # from 600 samples" (real always-on carrier). Pre-v0.6.3 databases
    # return sample_count=None; we omit the annotation in that case.
    n = cluster.sample_count
    persist_annotated = f"{persist} (n={n})" if n is not None else persist
    duration_s = max(0.0, (cluster.last_seen - cluster.first_seen).total_seconds())
    classification = cluster.classification or "unclassified"

    # Span annotation only for multi-bin clusters — single-bin keeps
    # the original v0.5.36 line shape so people who learned the format
    # still parse it.
    if cluster.count > 1:
        span_khz = (cluster.freq_high_hz - cluster.freq_low_hz) / 1000
        span = f"  ±{span_khz / 2:.0f}kHz×{cluster.count}bins"
    else:
        span = ""

    return (
        f"{rep:10.3f} MHz  peak={peak}  floor={floor}  "
        f"persist={persist_annotated}  seen={_humanize_duration(duration_s)}  "
        f"[{classification}]{span}"
    )


def _format_saturated_band_summary(
    band_id: str,
    band_name: str | None,
    clusters: list[_ChannelCluster],
) -> list[str]:
    """Render a saturated-band entry instead of enumerating mysteries.

    When a band has more carriers than _SATURATED_BAND_THRESHOLD even
    after clustering, listing "top 10 mysteries" is the wrong
    response — they're not mysteries, they're the band's normal
    activity that rfcensus doesn't decode. Show the band's character
    and point at follow-up tools instead.
    """
    n = len(clusters)
    lines: list[str] = []

    label = band_id
    if band_name and band_name != band_id:
        label = f"{band_id} – {band_name}"
    lines.append(
        f"  {label}  ({n} carriers, band saturated – "
        f"individual mystery enumeration suppressed)"
    )

    # Aggregate band character: peak / median / strongest few
    peaks = [c.peak_power_dbm for c in clusters if c.peak_power_dbm is not None]
    if peaks:
        peaks_sorted = sorted(peaks, reverse=True)
        max_peak = peaks_sorted[0]
        median = peaks_sorted[len(peaks_sorted) // 2]
        strong_count = sum(1 for p in peaks if p > median + 10)
        lines.append(
            f"    band character: max peak {max_peak:+.1f} dBm, "
            f"median {median:+.1f} dBm, "
            f"{strong_count} carrier(s) > median+10 dB"
        )

    # Strongest by peak — useful even when persistence ranking is useless
    strongest = sorted(
        clusters,
        key=lambda c: c.peak_power_dbm if c.peak_power_dbm is not None else -999,
        reverse=True,
    )[:_MAX_STRONGEST_IN_SUMMARY]
    if strongest:
        names = []
        for c in strongest:
            freq_mhz = c.representative_freq_hz / 1_000_000
            peak = c.peak_power_dbm
            if peak is None:
                names.append(f"{freq_mhz:.3f}")
            else:
                names.append(f"{freq_mhz:.3f} ({peak:+.1f} dBm)")
        lines.append(
            f"    strongest {len(strongest)}: {', '.join(names)}"
        )

    lines.append(
        f"    investigate live with: rfcensus monitor {band_id}"
    )
    return lines


def _render_band_mystery_section(
    band_id: str,
    band_name: str | None,
    channels: list[ActiveChannelRecord],
) -> list[str]:
    """Top-level renderer for one band's slice of the Mystery section.

    Performs clustering, decides between detailed-list and
    saturated-summary modes, and returns the rendered lines.
    """
    clusters = _cluster_adjacent_channels(channels)
    n = len(clusters)
    lines: list[str] = []

    if n > _SATURATED_BAND_THRESHOLD:
        lines.extend(_format_saturated_band_summary(band_id, band_name, clusters))
        return lines

    # Detailed list. Sort by composite score, fall back on peak power
    # when scores are zero (e.g. all members have no SNR data).
    clusters.sort(
        key=lambda c: (
            _mystery_score(c),
            c.peak_power_dbm if c.peak_power_dbm is not None else -999,
        ),
        reverse=True,
    )

    label = band_id
    if band_name and band_name != band_id:
        label = f"{band_id} – {band_name}"
    raw_count = sum(c.count for c in clusters)
    if raw_count == n:
        header = f"  {label}  ({n} active)"
    else:
        # Note the bin merging so users understand why their 1245-bin
        # band is now showing 200 entries instead.
        header = (
            f"  {label}  ({n} carriers from {raw_count} raw bins after "
            f"adjacent-bin clustering)"
        )
    lines.append(header)

    shown = clusters[:_MAX_UNTRACKED_CHANNELS_PER_BAND]
    for c in shown:
        lines.append("    " + _format_cluster(c))
    if len(clusters) > len(shown):
        omitted = len(clusters) - len(shown)
        lines.append(
            f"    … {omitted} more in {band_id} not shown"
        )
    return lines
