"""Frequency guide for the setup wizard.

A small reference table that maps common frequencies to:

• Quarter-wavelength in cm (helps users tune telescopic antennas)
• What's typically on the band (helps users orient)
• Which decoders/detectors care about this frequency
• Region where this band is meaningful

Used by the wizard's "what frequency?" prompt and the "I don't know"
branch (which presents this as a guided suggestion rather than just
refusing to help).
"""

from __future__ import annotations

from dataclasses import dataclass

# Speed of light in m/s (for quarter-wave math)
C_M_PER_S = 299_792_458


@dataclass(frozen=True)
class FrequencyProfile:
    """Description of a common frequency / band."""

    label: str
    freq_hz: int
    region: str  # "US", "EU", "global"
    typical_traffic: str
    decoders: tuple[str, ...]
    detectors: tuple[str, ...] = ()
    suggested_antenna_id: str | None = None
    notes: str = ""

    @property
    def quarter_wave_cm(self) -> float:
        return (C_M_PER_S / self.freq_hz) * 100 / 4

    @property
    def half_wave_cm(self) -> float:
        return (C_M_PER_S / self.freq_hz) * 100 / 2


# Ordered strictly by ascending frequency. Maintainers: keep this sorted.
# `test_common_frequencies_sorted_by_frequency` will fail if you don't.
COMMON_FREQUENCIES: tuple[FrequencyProfile, ...] = (
    FrequencyProfile(
        label="144 MHz — 2m amateur (APRS, voice repeaters)",
        freq_hz=144_390_000,
        region="global",
        typical_traffic=(
            "APRS at 144.39 MHz (NA) or 144.80 MHz (EU), 2m FM voice "
            "repeaters, amateur satellites."
        ),
        decoders=("multimon", "direwolf"),
        suggested_antenna_id="discone",
        notes="A discone or 2m-tuned vertical works well here.",
    ),
    FrequencyProfile(
        label="156-174 MHz — Marine VHF + business/public-safety VHF",
        freq_hz=156_800_000,
        region="global",
        typical_traffic=(
            "Marine VHF voice (156.0-162.0 MHz), business and public-safety "
            "VHF voice and trunked systems."
        ),
        decoders=(),
        detectors=("p25",),
        suggested_antenna_id="marine_vhf",
    ),
    FrequencyProfile(
        label="162 MHz — AIS (marine vessel tracking) + NOAA Weather Radio",
        freq_hz=162_000_000,
        region="global",
        typical_traffic=(
            "AIS at 161.975/162.025 MHz (vessel tracking, useful near coast/rivers), "
            "NOAA weather radio at 162.4-162.55 MHz."
        ),
        decoders=("rtl_ais",),
        suggested_antenna_id="marine_vhf",
    ),
    FrequencyProfile(
        label="315 MHz — TPMS, older car keyfobs, legacy security",
        freq_hz=315_000_000,
        region="US",
        typical_traffic=(
            "tire pressure sensors (US vehicles), some garage door openers, "
            "older home security sensors, key fobs"
        ),
        decoders=("rtl_433",),
        suggested_antenna_id="whip_315",
    ),
    FrequencyProfile(
        label="319.5 MHz — GE/Interlogix security",
        freq_hz=319_500_000,
        region="US",
        typical_traffic=(
            "GE/Interlogix wireless home security (door/window sensors, "
            "motion detectors). Common in suburban neighborhoods."
        ),
        decoders=("rtl_433",),
        suggested_antenna_id="whip_315",
    ),
    FrequencyProfile(
        label="345 MHz — Honeywell 5800 security",
        freq_hz=345_000_000,
        region="US",
        typical_traffic=(
            "Honeywell 5800-series wireless security sensors. "
            "Common with ADT and similar systems."
        ),
        decoders=("rtl_433",),
        suggested_antenna_id="whip_315",
    ),
    FrequencyProfile(
        label="433 MHz — 433 ISM (weather, sensors, fobs, EU TPMS)",
        freq_hz=433_920_000,
        region="global",
        typical_traffic=(
            "weather stations, soil sensors, low-power telemetry, doorbells, "
            "EU TPMS, EU LoRa, garage remotes. Very busy band."
        ),
        decoders=("rtl_433",),
        detectors=(),  # v0.6.6: lora detector removed; LoraSurveyTask runs as a sidecar on lora_survey-enabled bands
        suggested_antenna_id="whip_433",
    ),
    FrequencyProfile(
        label="450-470 MHz — UHF business / public safety / GMRS",
        freq_hz=460_000_000,
        region="US",
        typical_traffic=(
            "UHF business radio, GMRS/FRS family radios, public safety "
            "trunked systems, some amateur."
        ),
        decoders=(),
        detectors=("p25",),
        suggested_antenna_id="whip_433",
    ),
    FrequencyProfile(
        label="700/800 MHz — public safety trunked (P25)",
        freq_hz=851_000_000,
        region="US",
        typical_traffic=(
            "Police/fire/EMS P25 trunked systems. Encrypted in many "
            "jurisdictions but the control channel is always in the clear."
        ),
        decoders=(),
        detectors=("p25",),
        suggested_antenna_id="magmount_800_900",
        notes=(
            "For decoding voice, hand off to SDRTrunk. rfcensus will detect "
            "presence and identify the system, not decode it."
        ),
    ),
    FrequencyProfile(
        label="868 MHz — EU ISM (LoRaWAN, weather, smart-home)",
        freq_hz=868_000_000,
        region="EU",
        typical_traffic=(
            "EU LoRaWAN, EU smart meters, KNX/Z-Wave, weather stations. "
            "EU equivalent of the US 915 MHz band."
        ),
        decoders=("rtl_433",),
        detectors=(),  # v0.6.6: lora detector removed; LoraSurveyTask runs as a sidecar on lora_survey-enabled bands
        suggested_antenna_id="whip_915",
        notes="If you're in the US, you probably want 915 MHz instead.",
    ),
    FrequencyProfile(
        label="915 MHz — US ISM (LoRaWAN, electric/gas meters, sensors)",
        freq_hz=915_000_000,
        region="US",
        typical_traffic=(
            "LoRaWAN gateways, Itron ERT utility meters (gas/water/some "
            "electric), 900 MHz cordless phones, industrial telemetry, "
            "amateur experimentation."
        ),
        decoders=("rtl_433", "rtlamr"),
        detectors=(),  # v0.6.6: lora detector removed; LoraSurveyTask runs as a sidecar on lora_survey-enabled bands
        suggested_antenna_id="whip_915",
    ),
    FrequencyProfile(
        label="929 MHz — pager band (POCSAG, FLEX)",
        freq_hz=929_500_000,
        region="US",
        typical_traffic=(
            "Hospital and commercial pagers. Less traffic than it used to "
            "have, but still active in many cities."
        ),
        decoders=("multimon",),
        suggested_antenna_id="whip_915",
    ),
    FrequencyProfile(
        label="1090 MHz — ADS-B (aircraft transponders)",
        freq_hz=1_090_000_000,
        region="global",
        typical_traffic=(
            "Aircraft ADS-B position/identity broadcasts. Range is "
            "line-of-sight, typically 100-300 km from a rooftop antenna."
        ),
        decoders=(),
        suggested_antenna_id="dipole_1090",
        notes=(
            "Use a 1090-tuned dipole or filter+LNA combo for best results. "
            "Generic whips work but range is reduced."
        ),
    ),
    FrequencyProfile(
        label="2.4 GHz — WiFi/Bluetooth/Zigbee (HackRF only)",
        freq_hz=2_440_000_000,
        region="global",
        typical_traffic=(
            "WiFi (channels 1-14), Bluetooth Classic + BLE, Zigbee, Thread, "
            "lots of proprietary ISM. RTL-SDR cannot tune here; HackRF can."
        ),
        decoders=(),
        detectors=("wifi_bt_ism",),
        suggested_antenna_id=None,
        notes=(
            "rfcensus will tell you 2.4 GHz is busy; for protocol-specific "
            "decoding use kismet (WiFi), nRF Sniffer (BLE), or ubertooth (BT)."
        ),
    ),
)


def quarter_wave_cm(freq_hz: int) -> float:
    """Quarter-wavelength in cm for a given frequency."""
    return (C_M_PER_S / freq_hz) * 100 / 4


def find_profile(freq_hz: int, tolerance_pct: float = 0.05) -> FrequencyProfile | None:
    """Find a profile near the given frequency (within tolerance)."""
    for p in COMMON_FREQUENCIES:
        if abs(p.freq_hz - freq_hz) / p.freq_hz <= tolerance_pct:
            return p
    return None


def beginner_recommendations() -> tuple[FrequencyProfile, ...]:
    """The frequencies a new user should consider first.

    These are the highest-payoff bands for beginners — they're easy to
    catch (lots of traffic), interesting (real devices doing real
    things), and educational (you'll see how RF actually works in your
    neighborhood).
    """
    target_freqs = (433_920_000, 915_000_000, 162_000_000, 144_390_000)
    return tuple(
        p for p in COMMON_FREQUENCIES if p.freq_hz in target_freqs
    )
