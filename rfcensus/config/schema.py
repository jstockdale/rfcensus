"""Pydantic schema for rfcensus configuration.

Configs are layered:

  built-in defaults  ←  user site config  ←  CLI overrides

Each layer validates against these models. User config is a TOML file at
~/.config/rfcensus/site.toml. Built-in defaults ship with the package.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrategyKind(str, Enum):
    DECODER_ONLY = "decoder_only"
    DECODER_PRIMARY = "decoder_primary"
    POWER_PRIMARY = "power_primary"
    EXPLORATION = "exploration"


class AntennaConfig(BaseModel):
    """Describes one antenna the user owns."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    antenna_type: str = "whip"
    resonant_freq_hz: int | None = None
    usable_range: tuple[int, int]
    gain_dbi: float = 2.15
    # Free-form string. Common values: vertical, horizontal, rhcp, lhcp, unknown.
    polarization: str = "vertical"
    requires_bias_power: bool = False
    notes: str = ""

    @field_validator("usable_range")
    @classmethod
    def _usable_range_ordered(cls, v: tuple[int, int]) -> tuple[int, int]:
        low, high = v
        if low >= high:
            raise ValueError("usable_range low must be less than high")
        return v

    def covers(self, freq_hz: int) -> bool:
        """True if this antenna is usable at the given frequency."""
        low, high = self.usable_range
        return low <= freq_hz <= high

    def suitability(self, freq_hz: int) -> float:
        """0.0 (unusable) to 1.0 (resonant) suitability score for a frequency."""
        if not self.covers(freq_hz):
            return 0.0
        if self.resonant_freq_hz is None:
            # Wideband antenna: score by how centered the freq is in the range
            low, high = self.usable_range
            center = (low + high) / 2
            span = (high - low) / 2
            return max(0.3, 1.0 - abs(freq_hz - center) / span)
        # Resonant antenna: score by closeness to resonance
        detune = abs(freq_hz - self.resonant_freq_hz) / self.resonant_freq_hz
        if detune < 0.05:
            return 1.0
        if detune < 0.15:
            return 0.7
        if detune < 0.30:
            return 0.4
        return 0.2


class DongleConfig(BaseModel):
    """Describes one SDR dongle the user has declared."""

    model_config = ConfigDict(extra="forbid")

    id: str
    serial: str | None = None
    model: str  # "rtlsdr_v3", "rtlsdr_v4", "nesdr_nano3", "nesdr_smart_v5", "hackrf_one"
    driver: Literal["rtlsdr", "hackrf", "soapy"] = "rtlsdr"
    antenna: str | None = None  # References AntennaConfig.id
    bias_tee: bool = False
    tcxo_ppm: float = 1.0
    notes: str = ""


# ------------------------------------------------------------
# Bands and strategies
# ------------------------------------------------------------


class BandConfig(BaseModel):
    """Describes one band segment we might scan or decode."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    freq_low: int
    freq_high: int
    region: str = "US"
    expected_signals: list[str] = Field(default_factory=list)
    suggested_decoders: list[str] = Field(default_factory=list)
    strategy: StrategyKind = StrategyKind.DECODER_PRIMARY
    power_scan_parallel: bool = False
    typical_activity: str = "intermittent"  # "constant", "bursty", "rare"
    opt_in: bool = False  # Requires explicit user enabling
    notes: str = ""
    # FFT bin width for rtl_power sweeps of this band. None = use the
    # default heuristic (see `effective_power_scan_bin_hz`).
    #
    # Why this exists: the default heuristic (`bandwidth_hz // 256`)
    # gives 81 kHz bins on a 26 MHz band, which is roughly 6× too
    # coarse for most narrowband signals (P25 is 12.5 kHz, POCSAG is
    # 12.5 kHz, NFM voice is 12.5 or 25 kHz). A signal spanning one
    # bin gets averaged with neighboring empty channels, so occupancy
    # detection misses it. Bands with known narrow channels should
    # set this to `channel_spacing / 2` or finer.
    #
    # Tradeoff: halving bin width doubles FFT work (O(N log N) with
    # a constant-rate sweep, but N rises linearly, and rtl_power's
    # reporting cadence is fixed so more bins = more CPU). Typical
    # modern x86 handles 5 kHz bins across 30 MHz fine; Pi Zero may
    # struggle below 50 kHz. Values below 1000 Hz are almost always
    # a mistake (IQ rate limits prevent rtl_power from delivering
    # meaningful narrower resolution).
    power_scan_bin_hz: int | None = None
    # Per-decoder options. Keyed by decoder name (e.g. "rtlamr"); each
    # value is a dict the decoder interprets as it sees fit. Lets a
    # band tune decoder behavior without adding decoder-specific fields
    # to BandConfig itself.
    #
    # Currently used by:
    #   rtlamr.msgtype — comma-separated protocol list (default: "all")
    #     Example: 915_ism_r900 sets msgtype="r900,r900bcd,idm,netidm"
    #     so the second-pass rtlamr focuses on R900 variants.
    #
    # We keep this to ONE level of nesting (decoder_name → key → value)
    # rather than arbitrary depth. If a decoder needs more structure
    # it can encode it as a single dict value.
    decoder_options: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _freq_ordered(self) -> BandConfig:
        if self.freq_low > self.freq_high:
            raise ValueError(f"band {self.id}: freq_low > freq_high")
        return self

    @model_validator(mode="after")
    def _bin_hz_sane(self) -> BandConfig:
        """Sanity-check an explicit bin width if set.

        The hard lower bound (1000 Hz) is about as fine as rtl_power
        can deliver at typical 1-second integration — below that,
        variance dominates and you're just averaging noise. The
        upper bound (bandwidth / 2) ensures there are at least 2
        bins per sweep, which is the minimum for occupancy detection
        to detect ANY signal at all.
        """
        if self.power_scan_bin_hz is not None:
            if self.power_scan_bin_hz < 1000:
                raise ValueError(
                    f"band {self.id}: power_scan_bin_hz={self.power_scan_bin_hz} "
                    f"is too fine; rtl_power can't meaningfully deliver "
                    f"sub-1kHz bins at its typical integration time"
                )
            bandwidth = self.freq_high - self.freq_low
            if self.power_scan_bin_hz > bandwidth // 2:
                raise ValueError(
                    f"band {self.id}: power_scan_bin_hz={self.power_scan_bin_hz} "
                    f"exceeds bandwidth/2 ({bandwidth // 2}); would yield "
                    f"fewer than 2 bins per sweep"
                )
        return self

    @property
    def center_hz(self) -> int:
        return (self.freq_low + self.freq_high) // 2

    @property
    def bandwidth_hz(self) -> int:
        return self.freq_high - self.freq_low

    @property
    def effective_power_scan_bin_hz(self) -> int:
        """Bin width to actually use for rtl_power sweeps.

        If `power_scan_bin_hz` is set explicitly (via config), use it.
        Otherwise fall back to the historical heuristic that scales
        with bandwidth: `max(10 kHz, bandwidth / 256)`.

        The heuristic is wrong for narrow-channel bands (produces
        81 kHz bins for 915_ism's 26 MHz span, coarse enough to hide
        individual channels). New bands should set `power_scan_bin_hz`
        explicitly; the heuristic is kept for backward compatibility
        with user configs that don't know about the new field.
        """
        if self.power_scan_bin_hz is not None:
            return self.power_scan_bin_hz
        return max(10_000, self.bandwidth_hz // 256)


# ------------------------------------------------------------
# Decoder configuration
# ------------------------------------------------------------


class DecoderConfig(BaseModel):
    """Per-decoder configuration overrides."""

    model_config = ConfigDict(extra="allow")  # Decoder-specific extras

    enabled: bool = True
    binary: str | None = None  # Override path to binary
    extra_args: list[str] = Field(default_factory=list)


# ------------------------------------------------------------
# Privacy and policy
# ------------------------------------------------------------


class PrivacyConfig(BaseModel):
    """Privacy behavior defaults."""

    model_config = ConfigDict(extra="forbid")

    hash_device_ids: bool = True
    hash_salt: str = "auto"  # "auto" means generate + save
    include_ids_in_export: bool = False
    include_ids_in_report: bool = False


class ValidationConfig(BaseModel):
    """Thresholds for decode validation."""

    model_config = ConfigDict(extra="forbid")

    min_snr_db: float = 3.0
    min_rssi_dbm: float = -120.0
    max_rssi_dbm: float = -0.5  # Above this is compressed/saturated
    min_confirmations_for_confirmed: int = 3
    suspicious_id_patterns: list[str] = Field(
        default_factory=lambda: [
            "ffffff",
            "000000",
            "aaaaaa",
            "555555",
        ]
    )
    max_decodes_per_minute_per_decoder: int = 600


class ResourceConfig(BaseModel):
    """CPU / memory budgets."""

    model_config = ConfigDict(extra="forbid")

    # Fraction of logical cores to use, 0.0-1.0
    cpu_budget_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    # Override cpu_budget_fraction if set
    max_concurrent_decoders: int | None = None
    # Storage growth controls
    power_sample_retention_days: int = 7
    decode_retention_days: int = 90


# ------------------------------------------------------------
# Site (top-level user config)
# ------------------------------------------------------------


class LocationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float | None = None
    lon: float | None = None


class SiteMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    region: str = "US"
    location: LocationConfig = Field(default_factory=LocationConfig)
    timezone: str | None = None


class BandsSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: list[str] = Field(default_factory=list)
    disabled: list[str] = Field(default_factory=list)


class StrategiesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: StrategyKind = StrategyKind.DECODER_PRIMARY
    # Per-band overrides: {band_id: strategy_kind}
    overrides: dict[str, StrategyKind] = Field(default_factory=dict)


class SiteConfig(BaseModel):
    """Top-level merged configuration."""

    model_config = ConfigDict(extra="forbid")

    site: SiteMetadata = Field(default_factory=SiteMetadata)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    dongles: list[DongleConfig] = Field(default_factory=list)
    antennas: list[AntennaConfig] = Field(default_factory=list)
    bands: BandsSelection = Field(default_factory=BandsSelection)
    band_definitions: list[BandConfig] = Field(default_factory=list)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    decoders: dict[str, DecoderConfig] = Field(default_factory=dict)

    def find_antenna(self, antenna_id: str) -> AntennaConfig | None:
        return next((a for a in self.antennas if a.id == antenna_id), None)

    def find_band(self, band_id: str) -> BandConfig | None:
        return next((b for b in self.band_definitions if b.id == band_id), None)

    def find_dongle(self, dongle_id: str) -> DongleConfig | None:
        return next((d for d in self.dongles if d.id == dongle_id), None)

    def enabled_bands(self) -> list[BandConfig]:
        """Return the BandConfigs that should actually be scanned."""
        explicitly_enabled = set(self.bands.enabled)
        explicitly_disabled = set(self.bands.disabled)
        result: list[BandConfig] = []
        for band in self.band_definitions:
            if band.id in explicitly_disabled:
                continue
            if band.opt_in and band.id not in explicitly_enabled:
                continue
            if explicitly_enabled and band.id not in explicitly_enabled:
                # If user specified an enabled list, only those count
                continue
            result.append(band)
        return result
