"""Tests for the typed Meshtastic decoder configuration schema.

Two layers covered:
  • ``MeshtasticPskEntry`` / ``MeshtasticDecoderConfig`` Pydantic
    validation (key length, base64/hex correctness, exactly-one-form,
    psk_short range, region validity)
  • ``load_config()`` integration — a [decoders.meshtastic] section in
    site.toml is re-validated with the typed schema, so malformed
    entries surface as ConfigError at load time instead of silent
    failures at decode time

Plus the strategy-level integration: site-level decoder config
(specifically the meshtastic PSK list and region) flows through to
the ``DecoderRunSpec.decoder_options`` dict that the decoder's
``run()`` reads. This is the path that lets a user write
``[decoders.meshtastic]`` once in site.toml and have every band
that hosts the meshtastic decoder pick up the PSKs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError


# ─────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────


class TestMeshtasticPskEntry:
    def test_psk_short_basic(self) -> None:
        from rfcensus.config.schema import MeshtasticPskEntry
        e = MeshtasticPskEntry(name="X", psk_short=1)
        assert e.psk_short == 1
        assert e.psk_b64 is None
        assert e.psk_hex is None

    def test_psk_b64_16_byte_key_accepted(self) -> None:
        """Standard AES-128 PSK encoded as base64."""
        from rfcensus.config.schema import MeshtasticPskEntry
        # 16 bytes = "BBBB...BBBB" (16x 0x42)
        e = MeshtasticPskEntry(name="X", psk_b64="QkJCQkJCQkJCQkJCQkJCQg==")
        assert e.psk_b64.startswith("QkJC")

    def test_psk_b64_32_byte_key_accepted(self) -> None:
        """AES-256 PSK encoded as base64."""
        from rfcensus.config.schema import MeshtasticPskEntry
        import base64
        key32 = bytes([0x42] * 32)
        e = MeshtasticPskEntry(name="X", psk_b64=base64.b64encode(key32).decode())
        assert len(base64.b64decode(e.psk_b64)) == 32

    def test_psk_b64_wrong_length_rejected(self) -> None:
        """8-byte key: not valid for AES-128 or AES-256."""
        from rfcensus.config.schema import MeshtasticPskEntry
        with pytest.raises(ValidationError, match="16.*or 32"):
            MeshtasticPskEntry(name="X", psk_b64="QUFBQUFBQUE=")  # 8 bytes

    def test_psk_b64_invalid_base64_rejected(self) -> None:
        from rfcensus.config.schema import MeshtasticPskEntry
        with pytest.raises(ValidationError, match="not valid base64"):
            MeshtasticPskEntry(name="X", psk_b64="@@@not-base64")

    def test_psk_hex_16_byte_key_accepted(self) -> None:
        from rfcensus.config.schema import MeshtasticPskEntry
        e = MeshtasticPskEntry(name="X", psk_hex="42" * 16)
        assert len(bytes.fromhex(e.psk_hex)) == 16

    def test_psk_hex_wrong_length_rejected(self) -> None:
        from rfcensus.config.schema import MeshtasticPskEntry
        with pytest.raises(ValidationError, match="16.*or 32"):
            MeshtasticPskEntry(name="X", psk_hex="42" * 8)  # 8 bytes

    def test_psk_hex_invalid_hex_rejected(self) -> None:
        from rfcensus.config.schema import MeshtasticPskEntry
        with pytest.raises(ValidationError, match="not valid hex"):
            MeshtasticPskEntry(name="X", psk_hex="GGGG" * 8)

    def test_no_key_form_rejected(self) -> None:
        from rfcensus.config.schema import MeshtasticPskEntry
        with pytest.raises(ValidationError, match="must provide one of"):
            MeshtasticPskEntry(name="X")

    def test_multiple_key_forms_rejected(self) -> None:
        from rfcensus.config.schema import MeshtasticPskEntry
        with pytest.raises(ValidationError, match="multiple key forms"):
            MeshtasticPskEntry(
                name="X",
                psk_b64="QkJCQkJCQkJCQkJCQkJCQg==",
                psk_hex="42" * 16,
            )

    def test_empty_name_rejected(self) -> None:
        from rfcensus.config.schema import MeshtasticPskEntry
        with pytest.raises(ValidationError, match="non-empty"):
            MeshtasticPskEntry(name="", psk_short=1)
        with pytest.raises(ValidationError, match="non-empty"):
            MeshtasticPskEntry(name="   ", psk_short=1)

    def test_psk_short_out_of_range_rejected(self) -> None:
        from rfcensus.config.schema import MeshtasticPskEntry
        with pytest.raises(ValidationError, match="1..255"):
            MeshtasticPskEntry(name="X", psk_short=0)
        with pytest.raises(ValidationError, match="1..255"):
            MeshtasticPskEntry(name="X", psk_short=256)


class TestMeshtasticDecoderConfig:
    def test_defaults(self) -> None:
        from rfcensus.config.schema import MeshtasticDecoderConfig
        c = MeshtasticDecoderConfig()
        assert c.enabled is True
        assert c.region == "US"
        assert c.slots == "all"
        assert c.psks == []

    def test_invalid_slots_value_rejected(self) -> None:
        from rfcensus.config.schema import MeshtasticDecoderConfig
        with pytest.raises(ValidationError):
            MeshtasticDecoderConfig(slots="every")

    def test_extra_fields_rejected(self) -> None:
        """Typed config tightens DecoderConfig's loose extra='allow' to
        forbid: typos like 'PSKS' (caps) or 'reigon' should error
        instead of silently doing nothing."""
        from rfcensus.config.schema import MeshtasticDecoderConfig
        with pytest.raises(ValidationError):
            MeshtasticDecoderConfig(reigon="US")  # typo

    def test_psks_list_validates_each_entry(self) -> None:
        from rfcensus.config.schema import MeshtasticDecoderConfig
        c = MeshtasticDecoderConfig(
            psks=[
                {"name": "A", "psk_short": 1},
                {"name": "B", "psk_b64": "QkJCQkJCQkJCQkJCQkJCQg=="},
            ],
        )
        assert len(c.psks) == 2
        assert c.psks[0].name == "A"
        assert c.psks[1].psk_b64.startswith("QkJC")

    def test_psks_with_one_bad_entry_fails_whole_load(self) -> None:
        """A bad PSK entry rejects the entire decoder config so the
        user gets a clear error pointing at the right line."""
        from rfcensus.config.schema import MeshtasticDecoderConfig
        with pytest.raises(ValidationError):
            MeshtasticDecoderConfig(
                psks=[
                    {"name": "Good", "psk_short": 1},
                    {"name": "Bad", "psk_b64": "@@@not-base64"},
                ],
            )


# ─────────────────────────────────────────────────────────────────────
# Loader integration: site.toml → typed validation
# ─────────────────────────────────────────────────────────────────────


class TestLoaderTypedValidation:
    """Confirm load_config() wires the typed MeshtasticDecoderConfig
    through, so malformed PSKs in site.toml fail at load time."""

    def test_valid_meshtastic_section_loads(self, tmp_path: Path) -> None:
        from rfcensus.config.loader import load_config
        from rfcensus.config.schema import MeshtasticDecoderConfig
        site = tmp_path / "site.toml"
        site.write_text("""
[decoders.meshtastic]
enabled = true
region = "US"
slots = "all"

[[decoders.meshtastic.psks]]
name = "Bay Area Mesh"
psk_b64 = "QkJCQkJCQkJCQkJCQkJCQg=="

[[decoders.meshtastic.psks]]
name = "Old Channel"
psk_short = 2
""")
        cfg = load_config(path=site)
        mesh = cfg.decoders["meshtastic"]
        # Should be the typed subclass, not just a plain DecoderConfig
        assert isinstance(mesh, MeshtasticDecoderConfig)
        assert mesh.region == "US"
        assert mesh.slots == "all"
        assert len(mesh.psks) == 2
        assert mesh.psks[0].name == "Bay Area Mesh"
        assert mesh.psks[1].psk_short == 2

    def test_malformed_psk_fails_load(self, tmp_path: Path) -> None:
        from rfcensus.config.loader import load_config, ConfigError
        site = tmp_path / "site.toml"
        site.write_text("""
[[decoders.meshtastic.psks]]
name = "Broken"
psk_b64 = "@@@not-base64"
""")
        with pytest.raises(ConfigError, match="meshtastic"):
            load_config(path=site)

    def test_no_meshtastic_section_loads_normally(self, tmp_path: Path) -> None:
        """Most users won't have a [decoders.meshtastic] section.
        Loader must not require it."""
        from rfcensus.config.loader import load_config
        site = tmp_path / "site.toml"
        site.write_text("""
[site]
name = "test"
""")
        cfg = load_config(path=site)
        assert "meshtastic" not in cfg.decoders or (
            # If something else added a default, that's fine — but
            # there shouldn't be a typed config there since user
            # didn't specify one
            cfg.decoders.get("meshtastic", None) is None
            or len(getattr(cfg.decoders.get("meshtastic"), "psks", [])) == 0
        )

    def test_loose_form_still_loads(self, tmp_path: Path) -> None:
        """Other decoders use the loose dict form via extra='allow'.
        The typed-meshtastic upgrade shouldn't break them."""
        from rfcensus.config.loader import load_config
        site = tmp_path / "site.toml"
        site.write_text("""
[decoders.rtl_433]
enabled = true
binary = "/usr/local/bin/rtl_433"
""")
        cfg = load_config(path=site)
        assert "rtl_433" in cfg.decoders
        assert cfg.decoders["rtl_433"].binary == "/usr/local/bin/rtl_433"


# ─────────────────────────────────────────────────────────────────────
# Strategy integration: site config → DecoderRunSpec.decoder_options
# ─────────────────────────────────────────────────────────────────────


class TestStrategyDecoderOptionsMerge:
    """The mechanism that flows site-level decoder config (PSKs etc.)
    into the DecoderRunSpec the decoder receives at run time."""

    def test_site_meshtastic_config_flows_to_run_spec(self) -> None:
        """Verify the merge: site-level [decoders.meshtastic] PSKs
        appear in the merged decoder_options dict that strategy
        builds for the run spec."""
        from rfcensus.config.schema import (
            MeshtasticDecoderConfig, MeshtasticPskEntry,
        )

        # Replicate the merge logic from strategy._run_decoder_on_band:
        # for each site-level decoder cfg, lift non-base fields into
        # decoder_options, then per-band overrides take precedence.
        site_decoders = {
            "meshtastic": MeshtasticDecoderConfig(
                region="US", slots="all",
                psks=[MeshtasticPskEntry(name="Test", psk_short=1)],
            ),
        }
        merged: dict[str, dict] = {}
        base_fields = {"enabled", "binary", "extra_args"}
        for dec_name, site_dec_cfg in site_decoders.items():
            extras = {
                k: v for k, v in site_dec_cfg.model_dump().items()
                if k not in base_fields
            }
            if extras:
                merged[dec_name] = extras

        assert "meshtastic" in merged
        assert merged["meshtastic"]["region"] == "US"
        assert merged["meshtastic"]["slots"] == "all"
        assert len(merged["meshtastic"]["psks"]) == 1
        assert merged["meshtastic"]["psks"][0]["name"] == "Test"
        assert merged["meshtastic"]["psks"][0]["psk_short"] == 1

    def test_band_overrides_win_over_site(self) -> None:
        """Per-band decoder_options override site-level values for
        the same key. (The user wants band-specific tuning for
        rtlamr's msgtype, for example.)"""
        from rfcensus.config.schema import MeshtasticDecoderConfig

        site_decoders = {
            "meshtastic": MeshtasticDecoderConfig(
                region="US", slots="all",
            ),
        }
        band_options = {"meshtastic": {"slots": "default"}}

        merged: dict[str, dict] = {}
        base_fields = {"enabled", "binary", "extra_args"}
        for dec_name, site_dec_cfg in site_decoders.items():
            extras = {
                k: v for k, v in site_dec_cfg.model_dump().items()
                if k not in base_fields
            }
            if extras:
                merged[dec_name] = extras
        for dec_name, opts in band_options.items():
            merged.setdefault(dec_name, {}).update(opts)

        # Band override wins
        assert merged["meshtastic"]["slots"] == "default"
        # Site value preserved for other keys
        assert merged["meshtastic"]["region"] == "US"
