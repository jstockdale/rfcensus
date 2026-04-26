"""Config loading.

Merge order, lowest to highest precedence:

1. Built-in band definitions (`rfcensus/config/builtin/bands_*.toml`)
2. Built-in antenna library   (`rfcensus/config/builtin/antennas.toml`)
3. User site config           (`~/.config/rfcensus/site.toml`)
4. Explicit overrides from the CLI or programmatic API

We don't do a full-object deep merge; users override by adding their own
entries to the lists (with matching IDs replacing defaults) or by
setting scalar fields at the top level of their site.toml.
"""

from __future__ import annotations

import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rfcensus.config.schema import (
    AntennaConfig,
    BandConfig,
    MeshtasticDecoderConfig,
    SiteConfig,
)
from rfcensus.utils.logging import get_logger
from rfcensus.utils.paths import site_config_path

log = get_logger(__name__)


class ConfigError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc


def _load_builtin_bands(region: str = "US") -> list[BandConfig]:
    """Load shipped band definitions for a region."""
    resource_name = f"bands_{region.lower()}.toml"
    try:
        text = files("rfcensus.config.builtin").joinpath(resource_name).read_bytes()
    except FileNotFoundError:
        log.warning("no builtin band file for region %s, using US", region)
        text = files("rfcensus.config.builtin").joinpath("bands_us.toml").read_bytes()
    data = tomllib.loads(text.decode("utf-8"))
    return [BandConfig(**entry) for entry in data.get("band", [])]


def _load_builtin_antennas() -> list[AntennaConfig]:
    text = files("rfcensus.config.builtin").joinpath("antennas.toml").read_bytes()
    data = tomllib.loads(text.decode("utf-8"))
    return [AntennaConfig(**entry) for entry in data.get("antenna", [])]


def _merge_lists_by_id(
    defaults: list[Any], user: list[Any], label: str
) -> list[Any]:
    """Merge two lists of items that have an `id` field.

    User items replace defaults with matching IDs. New user items are appended.
    """
    by_id: dict[str, Any] = {item.id: item for item in defaults}
    for user_item in user:
        if user_item.id in by_id:
            log.debug("%s %s overridden by user config", label, user_item.id)
        by_id[user_item.id] = user_item
    return list(by_id.values())


def load_config(path: Path | None = None, region: str | None = None) -> SiteConfig:
    """Load and validate the user's config, merged with built-in defaults.

    Parameters
    ----------
    path:
        Path to user's site.toml. Defaults to ~/.config/rfcensus/site.toml.
        If the file doesn't exist, returns defaults.
    region:
        Override the region for built-in band selection (e.g. "US", "EU").
        If not supplied, uses the region declared in the user config, or "US".
    """
    path = path or site_config_path()

    user_raw: dict[str, Any] = {}
    if path.exists():
        log.debug("loading user config from %s", path)
        user_raw = _load_toml(path)

    # Determine region before full validation to pick the right builtin bands
    effective_region = (
        region
        or user_raw.get("site", {}).get("region")
        or "US"
    )

    builtin_bands = _load_builtin_bands(effective_region)
    builtin_antennas = _load_builtin_antennas()

    # Normalize user top-level keys we manipulate specially
    user_raw = dict(user_raw)
    user_band_raw = user_raw.pop("band_definitions", [])
    user_antennas_raw = user_raw.pop("antennas", [])

    try:
        user_bands = [BandConfig(**b) for b in user_band_raw]
        user_antennas = [AntennaConfig(**a) for a in user_antennas_raw]
    except ValidationError as exc:
        raise ConfigError(f"invalid band or antenna definition: {exc}") from exc

    merged_bands = _merge_lists_by_id(builtin_bands, user_bands, "band")
    merged_antennas = _merge_lists_by_id(builtin_antennas, user_antennas, "antenna")

    # Rebuild the dict the SiteConfig validator will see
    user_raw["band_definitions"] = [b.model_dump() for b in merged_bands]
    user_raw["antennas"] = [a.model_dump() for a in merged_antennas]

    try:
        cfg = SiteConfig(**user_raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid site config at {path}: {exc}") from exc

    # If the user has a [decoders.meshtastic] section, re-validate it
    # through the typed MeshtasticDecoderConfig so PSK entries get
    # checked at config-load time (wrong base64 padding, wrong key
    # length, missing name etc. surface here instead of silently
    # failing at decode time). The dict-form survived the loose
    # DecoderConfig pass thanks to extra="allow"; we now upgrade it
    # in place.
    mesh_raw = user_raw.get("decoders", {}).get("meshtastic")
    if mesh_raw is not None:
        try:
            cfg.decoders["meshtastic"] = MeshtasticDecoderConfig(**mesh_raw)
        except ValidationError as exc:
            raise ConfigError(
                f"invalid [decoders.meshtastic] config in {path}: {exc}"
            ) from exc

    return cfg


def write_default_site_config(
    path: Path | None = None, *, overwrite: bool = False
) -> Path:
    """Write a starter site.toml to the user's config directory."""
    path = path or site_config_path()
    if path.exists() and not overwrite:
        raise ConfigError(f"{path} already exists; pass overwrite=True to replace")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_DEFAULT_SITE_CONFIG, encoding="utf-8")
    return path


_DEFAULT_SITE_CONFIG = """\
# rfcensus site configuration
#
# This file describes the hardware you have, the bands you care about,
# and your privacy / policy preferences. Anything you don't set here
# falls back to rfcensus' built-in defaults.
#
# Regenerate with `rfcensus init --overwrite`.

[site]
name = "default"
region = "US"
# location = { lat = 37.80, lon = -122.27 }   # Optional, enables satellite pass prediction

[privacy]
hash_device_ids = true
hash_salt = "auto"
include_ids_in_export = false
include_ids_in_report = false

[validation]
min_snr_db = 6.0
min_confirmations_for_confirmed = 3

[resources]
cpu_budget_fraction = 0.5
power_sample_retention_days = 7
decode_retention_days = 90

# Declare your dongles. Serials should match what `rfcensus doctor` detects.
# [[dongles]]
# id = "v3_main"
# serial = "00000043"
# model = "rtlsdr_v3"
# antenna = "whip_915"
# tcxo_ppm = 1.0

# [[dongles]]
# id = "hackrf_main"
# serial = ""
# model = "hackrf_one"
# driver = "hackrf"
# antenna = "discone"

# Pick which bands to scan. Omit the `enabled` list to scan everything
# that isn't opt-in. Use `disabled` to skip specific ones.
[bands]
# enabled = ["433_ism", "915_ism", "pocsag_929", "ais"]
# disabled = ["frs_gmrs"]

[strategies]
default = "decoder_primary"
# overrides = { "915_ism" = "decoder_primary", "business_vhf" = "power_primary" }

# Decoder overrides. Disable a decoder here if you don't want to run it,
# or point to a specific binary if you've built a custom one.
# [decoders.rtl_433]
# enabled = true
# binary = "/usr/local/bin/rtl_433"

# Meshtastic decoder (in-process LoRa demodulation + AES-CTR decrypt).
# The public default channel decrypts automatically — no PSKs needed
# for vanilla mesh traffic. Add entries here only for custom-named
# channels whose PSK you have (export from the Meshtastic app via
# share-channel QR; the URL contains a base64-encoded PSK).
# [decoders.meshtastic]
# enabled = true
# region = "US"           # US, EU_868, EU_433, CN, JP, KR, TW, RU, ...
# slots = "all"           # "all" = every (preset, slot) in passband (recommended);
#                         # "default" = only each preset's default channel
#
# [[decoders.meshtastic.psks]]
# name = "Bay Area Mesh"
# psk_b64 = "1PG7OiApB1nwvP+rz05pAQ=="
#
# [[decoders.meshtastic.psks]]
# name = "Family Group"
# psk_hex = "deadbeefcafebabe..."   # 16 or 32 bytes
"""
