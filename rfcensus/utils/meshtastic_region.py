"""Meshtastic region tables + preset frequency-slot computation.

Direct Python port of the relevant parts of meshtastic-lite's
``src/meshtastic_radio.h`` and ``src/meshtastic_config.h``. The DJB2
hash and the slot arithmetic match meshtastic-lite byte-for-byte
(cross-checked by ``test_v071_region_passband.py``) so the two
implementations always agree on which RF frequency a given
``(region, preset, channel_name)`` lands on.

Why a Python port instead of FFI through libmeshtastic? The math is
tiny — DJB2 is 4 lines, the slot calc is 3 — and we need to call it
many times to enumerate "which preset slots fall in this dongle's
passband". Doing that through ctypes adds friction without speed gain.
The C library remains the authoritative implementation; this module
is just a convenience.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────
# Region table — mirrors MESH_REGIONS in meshtastic_config.h:105
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Region:
    """A regulatory band-plan entry.

    ``freq_start_mhz`` / ``freq_end_mhz`` are the LoRa-usable edges
    of the band, in MHz (matching meshtastic-lite's ``RegionDef``
    representation). ``power_limit_dbm`` is informational here — only
    relevant for TX, which we never do.
    """
    code: str
    freq_start_mhz: float
    freq_end_mhz: float
    duty_cycle_pct: int
    power_limit_dbm: int
    description: str


REGIONS: dict[str, Region] = {
    "US":      Region("US",      902.0,  928.0,   100, 30, "United States"),
    "EU_433":  Region("EU_433",  433.0,  434.0,   10,  10, "Europe 433"),
    "EU_868":  Region("EU_868",  869.4,  869.65,  10,  27, "Europe 868"),
    "CN":      Region("CN",      470.0,  510.0,   100, 19, "China"),
    "JP":      Region("JP",      920.5,  923.5,   100, 13, "Japan"),
    "ANZ":     Region("ANZ",     915.0,  928.0,   100, 30, "Aus/NZ 915"),
    "ANZ_433": Region("ANZ_433", 433.05, 434.79,  100, 14, "Aus/NZ 433"),
    "KR":      Region("KR",      920.0,  923.0,   100, 23, "Korea"),
    "TW":      Region("TW",      920.0,  925.0,   100, 27, "Taiwan"),
    "IN":      Region("IN",      865.0,  867.0,   100, 30, "India"),
    "NZ_865":  Region("NZ_865",  864.0,  868.0,   100, 36, "New Zealand 865"),
    "TH":      Region("TH",      920.0,  925.0,   100, 16, "Thailand"),
    "RU":      Region("RU",      868.7,  869.2,   100, 20, "Russia"),
}


# ─────────────────────────────────────────────────────────────────────
# Modem presets — mirrors meshPresetParams() in meshtastic_config.h
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Preset:
    """A Meshtastic modem preset.

    ``key`` is the rfcensus identifier (uppercase with underscores).
    ``display_name`` is the string Meshtastic uses for both the human-
    readable preset label AND for DJB2 hashing into the channel slot
    when no custom channel name is configured. The two MUST match the
    upstream meshPresetName() exactly — if they drift, default-channel
    traffic lands on the wrong slot and we won't see anything.
    """
    key: str
    display_name: str
    bandwidth_hz: int
    sf: int
    cr: int


# Order matters for human-friendly enumerations (slowest → fastest).
PRESETS: dict[str, Preset] = {
    "LONG_SLOW":     Preset("LONG_SLOW",     "LongSlow",   125_000, 12, 8),
    "LONG_MODERATE": Preset("LONG_MODERATE", "LongMod",    125_000, 11, 8),
    "LONG_FAST":     Preset("LONG_FAST",     "LongFast",   250_000, 11, 5),
    "LONG_TURBO":    Preset("LONG_TURBO",    "LongTurbo",  500_000, 11, 8),
    "MEDIUM_SLOW":   Preset("MEDIUM_SLOW",   "MediumSlow", 250_000, 10, 5),
    "MEDIUM_FAST":   Preset("MEDIUM_FAST",   "MediumFast", 250_000,  9, 5),
    "SHORT_SLOW":    Preset("SHORT_SLOW",    "ShortSlow",  250_000,  8, 5),
    "SHORT_FAST":    Preset("SHORT_FAST",    "ShortFast",  250_000,  7, 5),
    "SHORT_TURBO":   Preset("SHORT_TURBO",   "ShortTurbo", 500_000,  7, 5),
}


# ─────────────────────────────────────────────────────────────────────
# DJB2 hash — exact port of meshDjb2Hash() in meshtastic_radio.h
# ─────────────────────────────────────────────────────────────────────

def djb2(s: str) -> int:
    """Bernstein DJB2 hash, exactly matching meshtastic-lite's C version.

    The C version operates on ``uint32_t`` and lets the multiplications
    wrap. We mask after each step to keep the arithmetic identical.
    """
    h = 5381
    for ch in s:
        # ((h << 5) + h) + ord(ch) — the canonical DJB2 step
        h = (((h << 5) + h) + ord(ch)) & 0xFFFFFFFF
    return h


# ─────────────────────────────────────────────────────────────────────
# Slot computation
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PresetSlot:
    """One Meshtastic preset placed at its actual RF frequency.

    A "slot" is the integer channel number within the region's grid;
    ``freq_hz`` is the RF center frequency of that slot.

    The slot index depends on the channel NAME (DJB2 hash mod the
    region's channel count). For a default channel (empty user-side
    name), the name used is the preset's ``display_name`` —
    "LongFast", "MediumFast", etc. For a user-named channel, the
    user's name takes over.

    All numeric fields are in canonical units: Hz for frequencies,
    Hz for bandwidth (NOT kHz like the upstream C code).
    """
    preset: Preset
    region: Region
    channel_name: str       # the name actually used for the DJB2 hash
    slot: int               # channel index within region
    num_slots: int          # total slots available for this BW in this region
    freq_hz: int            # RF center frequency in Hz


def _calc_freq(
    region: Region,
    preset: Preset,
    channel_name: str,
    slot_override: int | None = None,
) -> PresetSlot:
    """Port of ``meshCalcFrequency`` from meshtastic_radio.h.

    The math (matching upstream exactly):
      • num_slots = floor((freq_end - freq_start) / (bw_mhz))
      • slot     = djb2(channel_name) mod num_slots, OR override
      • freq_mhz = freq_start + bw_mhz/2 + slot * bw_mhz

    Returns a ``PresetSlot`` with the resolved frequency in Hz.
    """
    bw_mhz = preset.bandwidth_hz / 1_000_000.0
    span_mhz = region.freq_end_mhz - region.freq_start_mhz
    num_slots = int(math.floor(span_mhz / bw_mhz))
    if num_slots == 0:
        num_slots = 1   # match meshtastic-lite's defensive clamp

    if slot_override is not None and 0 <= slot_override < num_slots:
        slot = slot_override
    else:
        slot = djb2(channel_name) % num_slots

    # Center of slot N: freq_start + bw/2 + N * bw
    freq_mhz = region.freq_start_mhz + (bw_mhz / 2.0) + (slot * bw_mhz)
    freq_hz = int(round(freq_mhz * 1_000_000))

    return PresetSlot(
        preset=preset,
        region=region,
        channel_name=channel_name,
        slot=slot,
        num_slots=num_slots,
        freq_hz=freq_hz,
    )


def default_slot(region_code: str, preset_key: str) -> PresetSlot:
    """Get the default-channel frequency slot for a preset in a region.

    "Default channel" means the user hasn't configured a custom name,
    so the preset's display name is used in the DJB2 hash. This is
    where 95%+ of Meshtastic traffic lives in any given region.
    """
    region = REGIONS[region_code]
    preset = PRESETS[preset_key]
    return _calc_freq(region, preset, preset.display_name)


def custom_channel_slot(
    region_code: str,
    preset_key: str,
    channel_name: str,
    slot_override: int | None = None,
) -> PresetSlot:
    """Get the frequency slot for a custom-named channel.

    Use this when the user has configured a non-default channel name
    (which changes the slot via the DJB2 hash). The ``slot_override``
    parameter mirrors Meshtastic's ``config.lora.channel_num`` —
    forces a specific slot regardless of name hashing.
    """
    region = REGIONS[region_code]
    preset = PRESETS[preset_key]
    return _calc_freq(region, preset, channel_name, slot_override)


# ─────────────────────────────────────────────────────────────────────
# Passband enumeration — "which presets can this dongle hear?"
# ─────────────────────────────────────────────────────────────────────

def slots_in_passband(
    region_code: str,
    center_freq_hz: int,
    sample_rate_hz: int,
    presets: list[str] | None = None,
    edge_guard_hz: int = 25_000,
) -> list[PresetSlot]:
    """Find every preset's default slot whose signal fits in a dongle's
    passband.

    A passband is the frequency range a dongle can receive given a
    chosen tuner center + sample rate. The signal must fit entirely
    within ``[center − Fs/2, center + Fs/2]`` to avoid aliasing — which
    means the signal's CENTER must lie within ``±(Fs − BW)/2`` of the
    tuner center.

    ``edge_guard_hz`` reserves a small margin from the absolute Nyquist
    edge to stay clear of the RTL-SDR's anti-alias filter roll-off
    (typical AGC/decimation chain has noticeable degradation in the
    last ~25 kHz). Set to 0 to use the full theoretical passband.

    Returns slots sorted by frequency. Slots whose default frequency
    lies outside the passband are silently omitted — the caller checks
    ``len()`` to know how many decoders to spawn.
    """
    if presets is None:
        presets = list(PRESETS.keys())

    out: list[PresetSlot] = []
    for key in presets:
        if key not in PRESETS:
            raise ValueError(f"unknown preset {key!r}; "
                             f"valid: {', '.join(PRESETS)}")
        slot = default_slot(region_code, key)
        bw = slot.preset.bandwidth_hz
        # Signal occupies bw centered at slot.freq_hz. For it to fit:
        #   slot.freq_hz - bw/2 ≥ center - Fs/2 + edge_guard
        #   slot.freq_hz + bw/2 ≤ center + Fs/2 - edge_guard
        # Combined: |slot.freq_hz - center| ≤ (Fs - bw)/2 - edge_guard
        max_offset = (sample_rate_hz - bw) // 2 - edge_guard_hz
        if max_offset <= 0:
            # Sample rate doesn't even fit one signal of this BW — skip.
            continue
        if abs(slot.freq_hz - center_freq_hz) <= max_offset:
            out.append(slot)

    out.sort(key=lambda s: s.freq_hz)
    return out


def all_default_slots(region_code: str) -> list[PresetSlot]:
    """List every preset's default slot in a region, sorted by frequency.

    Useful for ``--preset all --hop`` mode where the tool needs to
    cover the entire spread. Also handy for ``--list-slots`` diagnostics.
    """
    out = [default_slot(region_code, key) for key in PRESETS]
    out.sort(key=lambda s: s.freq_hz)
    return out


def enumerate_all_slots_in_passband(
    region_code: str,
    center_freq_hz: int,
    sample_rate_hz: int,
    presets: list[str] | None = None,
    edge_guard_hz: int = 25_000,
) -> list[PresetSlot]:
    """Enumerate EVERY (preset, slot_index) pair whose RF frequency
    falls in the dongle's passband.

    Unlike ``slots_in_passband`` (which returns only each preset's
    DEFAULT-channel slot), this returns the full Cartesian product of
    presets × candidate slot indices that fit. Use this for "catch
    every preset on every frequency" coverage where any custom-named
    channel could be transmitting.

    For the US region at 2.4 MS/s, this returns ~80 slots: 18 per
    125-kHz preset × 2 such presets, 9 per 250-kHz preset × 5, 4 per
    500-kHz preset × 2.

    The returned ``PresetSlot.channel_name`` is empty for non-default
    slots (we don't know what name hashes to a non-default slot — could
    be any of millions). Callers that need to decrypt traffic on these
    slots must either supply ``--psk NAME:HEX`` for each known channel
    name, or accept that they'll see encrypted-but-unparsed packets.

    Slots are sorted by ``(preset.bandwidth_hz, freq_hz)`` so callers
    can pump same-BW decoders together (modest cache benefit).
    """
    region = REGIONS[region_code]
    if presets is None:
        presets = list(PRESETS.keys())

    out: list[PresetSlot] = []
    for key in presets:
        if key not in PRESETS:
            raise ValueError(f"unknown preset {key!r}; "
                             f"valid: {', '.join(PRESETS)}")
        preset = PRESETS[key]
        bw = preset.bandwidth_hz
        bw_mhz = bw / 1_000_000.0
        span_mhz = region.freq_end_mhz - region.freq_start_mhz
        num_slots = int(math.floor(span_mhz / bw_mhz))
        if num_slots == 0:
            continue

        # Compute the range of slot indices that fit in the passband.
        # slot N freq = freq_start + bw/2 + N*bw (in MHz). We want
        # |freq - center| ≤ (Fs - bw)/2 - edge_guard.
        max_offset = (sample_rate_hz - bw) // 2 - edge_guard_hz
        if max_offset <= 0:
            continue

        # Solve for N: |center - (freq_start*1e6 + bw/2 + N*bw)| ≤ max_offset
        center_mhz = center_freq_hz / 1_000_000.0
        slot_min = max(0, int(math.ceil(
            (center_mhz - region.freq_start_mhz - bw_mhz/2 - max_offset/1e6)
            / bw_mhz
        )))
        slot_max = min(num_slots - 1, int(math.floor(
            (center_mhz - region.freq_start_mhz - bw_mhz/2 + max_offset/1e6)
            / bw_mhz
        )))

        for slot_idx in range(slot_min, slot_max + 1):
            freq_mhz = region.freq_start_mhz + (bw_mhz / 2.0) + (slot_idx * bw_mhz)
            freq_hz = int(round(freq_mhz * 1_000_000))
            # Final passband check (defensive against off-by-one in
            # the ceil/floor solve above).
            if abs(freq_hz - center_freq_hz) > max_offset:
                continue
            out.append(PresetSlot(
                preset=preset,
                region=region,
                channel_name="",   # unknown for non-default slots
                slot=slot_idx,
                num_slots=num_slots,
                freq_hz=freq_hz,
            ))

    out.sort(key=lambda s: (s.preset.bandwidth_hz, s.freq_hz))
    return out
