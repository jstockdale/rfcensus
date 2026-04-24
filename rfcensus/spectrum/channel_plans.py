"""v0.5.42: channel-plan inference for LoRa-family detections.

Given a refined center frequency and matched bandwidth, identify
which standardized channel (if any) this corresponds to. Lets reports
label detections like "LoRaWAN US channel 4 (904.1 MHz)" or
"Meshtastic LongFast slot 20 (US default)" instead of raw freq.

Channel plans implemented
=========================

  • LoRaWAN US915  — 64 uplink + 8 downlink channels in 902-928 MHz
  • LoRaWAN EU868  — 3 mandatory + optional uplink channels in 863-870 MHz
  • LoRaWAN AU915  — 64 uplink channels (subset of US grid + offset)
  • Meshtastic US  — 8 LongFast / MediumFast / etc. slots in 902-928 MHz
  • Meshtastic EU  — slots in 869.x MHz EU868 sub-band

These are public, well-documented standards. References:

  • LoRaWAN Regional Parameters RP002-1.0.4 (LoRa Alliance)
  • Meshtastic firmware src/mesh/RadioInterface.cpp regional defaults
    (https://github.com/meshtastic/firmware)

Matching
========

A detection matches a channel if:
  • Its bandwidth is within 20% of the channel's nominal width
  • Its refined center is within `match_tolerance_hz` of the channel's
    center (default 5 kHz — well inside any tolerance these standards
    publish, but loose enough to handle our refinement uncertainty)

Returns the BEST match (lowest center-frequency error) or None.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelMatch:
    """A standardized channel identification for a detection."""

    plan: str  # e.g. "lorawan_us915"
    channel_id: str  # e.g. "uplink_4" or "longfast_slot_20"
    center_hz: int
    bandwidth_hz: int
    description: str  # human-readable label for reports


# ---------------------------------------------------------------------
# Channel plan tables
# ---------------------------------------------------------------------


def _lorawan_us915_uplink_channels() -> list[ChannelMatch]:
    """LoRaWAN US915 sub-bands 1-8: 64 uplink channels.

    Channel 0 = 902.3 MHz, spacing 200 kHz, 125 kHz wide.
    Plus 8 SF8/500kHz uplinks at 903.0 + N·1.6 MHz.
    """
    out: list[ChannelMatch] = []
    # 64 × 125 kHz uplinks
    for n in range(64):
        center = 902_300_000 + n * 200_000
        out.append(
            ChannelMatch(
                plan="lorawan_us915",
                channel_id=f"uplink_{n}",
                center_hz=center,
                bandwidth_hz=125_000,
                description=f"LoRaWAN US915 uplink ch.{n} ({center/1e6:.3f} MHz, 125 kHz)",
            )
        )
    # 8 × 500 kHz SF8 uplinks
    for n in range(8):
        center = 903_000_000 + n * 1_600_000
        out.append(
            ChannelMatch(
                plan="lorawan_us915",
                channel_id=f"uplink500_{n}",
                center_hz=center,
                bandwidth_hz=500_000,
                description=f"LoRaWAN US915 SF8/500kHz uplink ch.{n} ({center/1e6:.3f} MHz)",
            )
        )
    # 8 × 500 kHz downlink (RX1) channels
    for n in range(8):
        center = 923_300_000 + n * 600_000
        out.append(
            ChannelMatch(
                plan="lorawan_us915",
                channel_id=f"downlink_{n}",
                center_hz=center,
                bandwidth_hz=500_000,
                description=f"LoRaWAN US915 downlink ch.{n} ({center/1e6:.3f} MHz, 500 kHz)",
            )
        )
    return out


def _lorawan_eu868_channels() -> list[ChannelMatch]:
    """LoRaWAN EU868: 3 mandatory channels + 5 commonly-allocated optional.

    All 125 kHz wide. The 3 mandatory channels (868.1, 868.3, 868.5) are
    used by every EU868 device; the optional ones are allocated by the
    network server at join. Including the most common defaults.
    """
    centers_mhz = [868.1, 868.3, 868.5, 867.1, 867.3, 867.5, 867.7, 867.9]
    out: list[ChannelMatch] = []
    for i, mhz in enumerate(centers_mhz):
        center = int(mhz * 1_000_000)
        kind = "mandatory" if i < 3 else "optional"
        out.append(
            ChannelMatch(
                plan="lorawan_eu868",
                channel_id=f"{kind}_{i}",
                center_hz=center,
                bandwidth_hz=125_000,
                description=f"LoRaWAN EU868 {kind} ch.{i} ({mhz:.1f} MHz, 125 kHz)",
            )
        )
    # Plus the 250 kHz SF7/250kHz channel at 868.3
    out.append(
        ChannelMatch(
            plan="lorawan_eu868",
            channel_id="sf7bw250",
            center_hz=868_300_000,
            bandwidth_hz=250_000,
            description="LoRaWAN EU868 SF7/250kHz (868.3 MHz, 250 kHz)",
        )
    )
    return out


def _lorawan_au915_uplink_channels() -> list[ChannelMatch]:
    """LoRaWAN AU915: 64 × 125 kHz uplinks at 915.2 + N·200 kHz."""
    out: list[ChannelMatch] = []
    for n in range(64):
        center = 915_200_000 + n * 200_000
        out.append(
            ChannelMatch(
                plan="lorawan_au915",
                channel_id=f"uplink_{n}",
                center_hz=center,
                bandwidth_hz=125_000,
                description=f"LoRaWAN AU915 uplink ch.{n} ({center/1e6:.3f} MHz, 125 kHz)",
            )
        )
    return out


def _meshtastic_us_channels() -> list[ChannelMatch]:
    """Meshtastic US (902-928 MHz): default frequency for each modem
    preset, computed via Meshtastic's actual channel-name hash.

    Formula (Meshtastic firmware src/mesh/Channels.cpp hashName + the
    RadioInterface slot→frequency conversion):

        slot_0_indexed = djb2(channel_name) % num_slots
        center_hz = 902_000_000 + slot * bw_hz + bw_hz // 2

    where num_slots = 26_000_000 / bw_hz — 104 slots for 250 kHz BW
    presets, 52 for 500 kHz ShortTurbo, 208 for 125 kHz long-range.
    djb2 is the Bernstein hash: h = 5381; for c in name: h = h*33 + c
    (unsigned 32-bit).

    Default channel name for each preset equals the preset name
    itself (LONG_FAST → "LongFast") when using the out-of-box public
    channel. Users who rename their primary channel land on different
    slots; SF+BW signature still identifies the preset.

    Values verified v0.5.43 against documented slots:
      • LongFast slot 20 (906.875 MHz) — meshtastic.org/docs
      • MediumFast slot 45 (913.125 MHz) — mtnme.sh/mediumfast,
        freq51.net (HAM-band conflict advisory)
      • MediumSlow slot 52 (914.875 MHz) — heypete's calculator,
        BayMe.sh deployment
      • ShortFast slot 68 (918.875 MHz) — pugetmesh.org/may2025
      • ShortSlow slot 75 (920.625 MHz) — djb2 computed,
        not independently verified by operator deployment
      • ShortTurbo slot 50 (926.75 MHz, 52-slot grid for 500 kHz) —
        djb2 computed
    """
    return [
        ChannelMatch(
            plan="meshtastic_us",
            channel_id="longfast_default",
            center_hz=906_875_000,
            bandwidth_hz=250_000,
            description=(
                "Meshtastic US LongFast default channel "
                "(906.875 MHz, 250 kHz, SF11) – slot 20"
            ),
        ),
        ChannelMatch(
            plan="meshtastic_us",
            channel_id="mediumslow_default",
            center_hz=914_875_000,
            bandwidth_hz=250_000,
            description=(
                "Meshtastic US MediumSlow default channel "
                "(914.875 MHz, 250 kHz, SF10) – slot 52"
            ),
        ),
        ChannelMatch(
            plan="meshtastic_us",
            channel_id="mediumfast_default",
            center_hz=913_125_000,
            bandwidth_hz=250_000,
            description=(
                "Meshtastic US MediumFast default channel "
                "(913.125 MHz, 250 kHz, SF9) – slot 45"
            ),
        ),
        ChannelMatch(
            plan="meshtastic_us",
            channel_id="shortslow_default",
            center_hz=920_625_000,
            bandwidth_hz=250_000,
            description=(
                "Meshtastic US ShortSlow default channel "
                "(920.625 MHz, 250 kHz, SF8) – slot 75"
            ),
        ),
        ChannelMatch(
            plan="meshtastic_us",
            channel_id="shortfast_default",
            center_hz=918_875_000,
            bandwidth_hz=250_000,
            description=(
                "Meshtastic US ShortFast default channel "
                "(918.875 MHz, 250 kHz, SF7) – slot 68"
            ),
        ),
        ChannelMatch(
            plan="meshtastic_us",
            channel_id="shortturbo_default",
            center_hz=926_750_000,
            bandwidth_hz=500_000,
            description=(
                "Meshtastic US ShortTurbo default channel "
                "(926.75 MHz, 500 kHz, SF7) – slot 50"
            ),
        ),
        # Long-range 125 kHz presets (less common in modern firmware
        # but still present in the enum and occasionally observed).
        ChannelMatch(
            plan="meshtastic_us",
            channel_id="longslow_default",
            center_hz=905_312_500,
            bandwidth_hz=125_000,
            description=(
                "Meshtastic US LongSlow default channel "
                "(905.3125 MHz, 125 kHz, SF12)"
            ),
        ),
        ChannelMatch(
            plan="meshtastic_us",
            channel_id="longmoderate_default",
            center_hz=912_812_500,
            bandwidth_hz=125_000,
            description=(
                "Meshtastic US LongModerate default channel "
                "(912.8125 MHz, 125 kHz, SF11)"
            ),
        ),
        ChannelMatch(
            plan="meshtastic_us",
            channel_id="vlongslow_default",
            center_hz=908_062_500,
            bandwidth_hz=125_000,
            description=(
                "Meshtastic US VLongSlow default channel "
                "(908.0625 MHz, 125 kHz, SF12)"
            ),
        ),
    ]


def _meshtastic_eu_channels() -> list[ChannelMatch]:
    """Meshtastic EU868: 869.4-869.65 MHz 10%-duty sub-band.

    The EU allocation is only 250 kHz wide, so only ONE 250 kHz slot
    fits — all 250 kHz presets (LongFast/MediumSlow/MediumFast/
    ShortSlow/ShortFast) use 869.525 MHz regardless of channel name.
    500 kHz ShortTurbo does NOT fit the EU band. 125 kHz presets have
    2 possible slots but default channel names all hash to the same.

    Verified v0.5.43 against meshtastic.org/docs/overview/radio-settings/:
      "After factory reset the radio will be set to frequency slot 1
       with a center frequency of 869.525 MHz."
    """
    return [
        ChannelMatch(
            plan="meshtastic_eu",
            channel_id="eu868_default",
            center_hz=869_525_000,
            bandwidth_hz=250_000,
            description=(
                "Meshtastic EU868 default channel "
                "(869.525 MHz, 250 kHz) – all 250 kHz presets share this slot"
            ),
        ),
    ]


def _all_channels() -> list[ChannelMatch]:
    """All channels from all known plans, concatenated. Cached by the
    module-level constant below for efficient lookup."""
    return (
        _lorawan_us915_uplink_channels()
        + _lorawan_eu868_channels()
        + _lorawan_au915_uplink_channels()
        + _meshtastic_us_channels()
        + _meshtastic_eu_channels()
    )


# Module-level cache — channel tables are static
_CHANNEL_CATALOG: list[ChannelMatch] | None = None


def _catalog() -> list[ChannelMatch]:
    global _CHANNEL_CATALOG
    if _CHANNEL_CATALOG is None:
        _CHANNEL_CATALOG = _all_channels()
    return _CHANNEL_CATALOG


# ---------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------


def match_channel(
    *,
    freq_hz: int,
    bandwidth_hz: int,
    match_tolerance_hz: int = 5_000,
    bw_tolerance: float = 0.20,
) -> ChannelMatch | None:
    """Return the best ChannelMatch for (freq_hz, bandwidth_hz) or None.

    A channel matches if:
      • bandwidth is within bw_tolerance (fraction) of nominal width
      • center is within match_tolerance_hz of nominal center

    Of all matching channels, the one with the smallest center error
    wins. None if no channel matches.

    `match_tolerance_hz` should be larger than the IQ-based refinement
    uncertainty (~1-5 kHz for our chirp/FFT methods) but smaller than
    the channel spacing (~200 kHz minimum for LoRaWAN US, 250 kHz for
    Meshtastic). 5 kHz is comfortably in that range.
    """
    best: ChannelMatch | None = None
    best_err: int | None = None
    for ch in _catalog():
        bw_err = abs(ch.bandwidth_hz - bandwidth_hz) / ch.bandwidth_hz
        if bw_err > bw_tolerance:
            continue
        err = abs(ch.center_hz - freq_hz)
        if err > match_tolerance_hz:
            continue
        if best_err is None or err < best_err:
            best = ch
            best_err = err
    return best


def list_plans() -> list[str]:
    """All known plan identifiers."""
    return sorted({c.plan for c in _catalog()})


def channels_for_plan(plan: str) -> list[ChannelMatch]:
    """All channels in a named plan. Useful for tests and reports."""
    return [c for c in _catalog() if c.plan == plan]
