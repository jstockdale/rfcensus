"""v0.5.33 regression tests: rtlamr msgtype CPU regression fix.

Background
==========

v0.5.31 switched rtlamr's --msgtype from the shorthand "all" (which
expands internally to scm, scm+, idm, r900) to the explicit list
"scm,scm+,idm,netidm,r900,r900bcd". The intent was to add netidm and
r900bcd coverage, which the "all" alias does NOT include.

A real-world 60-minute scan then showed drop rate had jumped from
~0.8% (v0.5.27) to ~53% — rtlamr was consuming ~107% CPU across its
Go runtime threads and the fanout was falling behind.

Root cause was traced by reading rtlamr's source (rtlamr-master).
The relevant pattern is in r900/r900.go's Parse function:

    func (p *Parser) Parse(pkts, msgCh, wg) {
        p.once.Do(func() { /* allocate */ })
        copy(p.signal, p.signal[cfg.BlockSize:])
        copy(p.signal[cfg.PacketLength:], p.Decoder.Signal[...])
        p.filter()    // O(BufferLength) work
        p.quantize()  // O(BufferLength) work
        for _, pkt := range pkts { ... }  // only if matches
        wg.Done()
    }

r900's Parse unconditionally runs a full filter + quantize pass on
every IQ block — even when no preamble matched. Most other protocol
Parse functions (scm, scm+, idm, netidm) return fast when pkts is
empty.

r900bcd/r900bcd.go wraps r900 by spawning an inner goroutine that
calls r900.Parse AGAIN:

    func (p Parser) Parse(pkts, msgCh, wg) {
        ...
        go p.Parser.Parse(pkts, localMsgCh, localWg)  // r900.Parse
        ...
    }

So adding r900bcd to the msgtype list effectively DOUBLES the r900
filter+quantize work per block. This was the CPU regression.

The fix (v0.5.33)
=================

1. Default msgtype reverts to "all" — matches v0.5.27's working
   config (scm, scm+, idm, r900).

2. Per-band override via BandConfig.decoder_options["rtlamr"]["msgtype"]
   lets a band supply a different msgtype list without polluting
   the generic BandConfig schema with decoder-specific fields.

3. 915_ism_r900 (second-pass band at 912.6 MHz added in v0.5.32)
   uses decoder_options to request msgtype="r900,r900bcd,idm,netidm".
   This rtlamr instance runs alone on its dongle in a later wave,
   so it has CPU budget to spare for the r900bcd double-work.

This test suite asserts:
  • The rtlamr decoder defaults to msgtype=all
  • decoder_options.rtlamr.msgtype overrides the default
  • DecoderRunSpec carries decoder_options through from strategy
  • BandConfig.decoder_options round-trips from TOML
  • 915_ism_r900 has the expected override
  • 915_ism does NOT have an override (uses default)
  • Strategy populates DecoderRunSpec.decoder_options from the band
"""

from __future__ import annotations

import inspect
from dataclasses import fields
from typing import Any
from unittest.mock import MagicMock

import pytest


class TestDecoderRunSpecHasDecoderOptions:
    """DecoderRunSpec is how a strategy hands context to a decoder.

    v0.5.33 added a decoder_options dict to support per-band,
    decoder-specific overrides without per-decoder fields.
    """

    def test_decoder_run_spec_has_decoder_options_field(self):
        from rfcensus.decoders.base import DecoderRunSpec

        field_names = {f.name for f in fields(DecoderRunSpec)}
        assert "decoder_options" in field_names, (
            "DecoderRunSpec must expose decoder_options so strategies "
            "can pass per-band decoder-specific tuning (like rtlamr's "
            "msgtype) without new dataclass fields per decoder."
        )

    def test_decoder_options_defaults_to_empty_dict(self):
        """Unset → empty dict, not None. Decoders do
        spec.decoder_options.get(name, {}) and expect a dict."""
        from rfcensus.decoders.base import DecoderRunSpec

        # Build a minimal spec; decoder_options should auto-init
        spec = DecoderRunSpec(
            lease=MagicMock(),
            freq_hz=915_000_000,
            sample_rate=2_400_000,
            duration_s=60.0,
            event_bus=MagicMock(),
            session_id=1,
        )
        assert spec.decoder_options == {}
        assert isinstance(spec.decoder_options, dict)

    def test_decoder_options_is_nested_dict(self):
        """Top-level key = decoder name, value = dict of options.

        This is the contract decoders rely on:
          opts = spec.decoder_options.get("rtlamr", {})
          msgtype = opts.get("msgtype", "all")
        """
        from rfcensus.decoders.base import DecoderRunSpec

        spec = DecoderRunSpec(
            lease=MagicMock(),
            freq_hz=915_000_000,
            sample_rate=2_400_000,
            duration_s=60.0,
            event_bus=MagicMock(),
            session_id=1,
            decoder_options={"rtlamr": {"msgtype": "scm,idm"}},
        )
        assert spec.decoder_options["rtlamr"]["msgtype"] == "scm,idm"


class TestBandConfigDecoderOptions:
    """BandConfig.decoder_options lets a TOML band override decoder
    behavior. Verifies schema + round-trip from builtin TOML."""

    def test_band_config_has_decoder_options_field(self):
        from rfcensus.config.schema import BandConfig

        assert "decoder_options" in BandConfig.model_fields, (
            "BandConfig must have a decoder_options field for per-band "
            "decoder-specific overrides (v0.5.33 added this to fix "
            "the rtlamr msgtype CPU regression)."
        )

    def test_decoder_options_defaults_to_empty(self):
        """Most bands don't override anything. Empty = use decoder
        defaults."""
        from rfcensus.config.schema import BandConfig

        # Build a minimal band; decoder_options should default to {}
        band = BandConfig(
            id="test_band",
            name="test",
            freq_low=915_000_000,
            freq_high=917_000_000,
        )
        assert band.decoder_options == {}

    def test_decoder_options_accepts_nested_dict(self):
        from rfcensus.config.schema import BandConfig

        band = BandConfig(
            id="test_band",
            name="test",
            freq_low=915_000_000,
            freq_high=917_000_000,
            decoder_options={
                "rtlamr": {"msgtype": "r900,r900bcd"},
            },
        )
        assert band.decoder_options == {
            "rtlamr": {"msgtype": "r900,r900bcd"},
        }


class TestBuiltinBandsHaveCorrectDecoderOptions:
    """The specific overrides the v0.5.33 fix relies on."""

    def _load_bands(self) -> dict[str, Any]:
        from rfcensus.config.loader import _load_builtin_bands
        return {b.id: b for b in _load_builtin_bands("US")}

    def test_915_ism_r900_overrides_msgtype(self):
        """The second-pass R900 band MUST request an r900-focused
        msgtype. If this fails, 915_ism_r900 would fall back to
        msgtype=all and we lose r900bcd + netidm coverage on the
        one pass where they're cheap to decode.
        """
        bands = self._load_bands()
        r900_band = bands["915_ism_r900"]

        assert "rtlamr" in r900_band.decoder_options, (
            "915_ism_r900 must declare decoder_options.rtlamr; "
            "without it the band silently falls back to msgtype=all "
            "and we lose the r900bcd/netidm coverage the second "
            "pass exists for."
        )

        msgtype = r900_band.decoder_options["rtlamr"].get("msgtype")
        assert msgtype is not None, (
            "915_ism_r900.decoder_options.rtlamr must include "
            "a msgtype key."
        )

        # Verify all four expected protocols are present
        types = set(t.strip() for t in msgtype.split(","))
        assert "r900" in types, "r900 protocol missing"
        assert "r900bcd" in types, "r900bcd protocol missing"
        assert "idm" in types, "idm protocol missing"
        assert "netidm" in types, "netidm protocol missing"

    def test_915_ism_does_NOT_override_msgtype(self):
        """The primary 915_ism band MUST NOT override msgtype — it
        should use the default 'all', which matches v0.5.27's
        working CPU budget. Setting msgtype here (especially one
        that includes r900bcd) is what caused the v0.5.31 drop
        regression."""
        bands = self._load_bands()
        primary = bands["915_ism"]

        # Either no decoder_options at all, or rtlamr key absent, or
        # rtlamr.msgtype absent — any of these mean "use default".
        rtlamr_opts = primary.decoder_options.get("rtlamr", {})
        assert "msgtype" not in rtlamr_opts, (
            f"915_ism must NOT override rtlamr.msgtype; found "
            f"{rtlamr_opts.get('msgtype')!r}. v0.5.31 set this to an "
            f"expanded list including r900bcd, which doubled r900's "
            f"per-block DSP work and pushed the CPU budget. The fix "
            f"is to leave the primary band on msgtype=all (via no "
            f"override) and isolate r900bcd on the 915_ism_r900 "
            f"second pass."
        )


class TestRtlamrDecoderReadsDecoderOptions:
    """The rtlamr decoder must source its msgtype from decoder_options
    when a band supplies one, defaulting to 'all' otherwise."""

    def _get_rtlamr_source(self) -> str:
        from rfcensus.decoders.builtin import rtlamr
        return inspect.getsource(rtlamr)

    def test_default_msgtype_is_all(self):
        """v0.5.33: revert to 'all' (scm, scm+, idm, r900) which
        matches v0.5.27's working behavior."""
        src = self._get_rtlamr_source()
        assert 'get("msgtype", "all")' in src, (
            "rtlamr decoder must default msgtype to 'all' when no "
            "band override is set. 'all' is rtlamr's internal alias "
            "for (scm, scm+, idm, r900) and matches v0.5.27's known-"
            "working CPU profile."
        )

    def test_reads_from_decoder_options(self):
        """The source must read from spec.decoder_options["rtlamr"]."""
        src = self._get_rtlamr_source()
        assert 'decoder_options.get("rtlamr"' in src, (
            "rtlamr decoder must read spec.decoder_options.get(\"rtlamr\", {})"
            " so bands can override msgtype. Without this, the 915_ism_r900 "
            "second-pass override has no effect."
        )

    def test_does_NOT_hardcode_expanded_msgtype(self):
        """Regression guard: v0.5.31's hardcoded expanded list must
        not come back. Any rtlamr call that bakes r900bcd into the
        default risks re-triggering the CPU regression."""
        src = self._get_rtlamr_source()
        # The literal string that caused v0.5.31's regression:
        assert '"-msgtype=scm,scm+,idm,netidm,r900,r900bcd"' not in src, (
            "rtlamr decoder must not hardcode the expanded msgtype list. "
            "Including r900bcd in the default list doubles r900's "
            "per-block DSP work (r900bcd.Parse spawns an inner goroutine "
            "that calls r900.Parse again, and r900.Parse does filter+"
            "quantize on every block regardless of preamble matches). "
            "This caused the v0.5.31 60x drop regression."
        )

    def test_msgtype_arg_uses_equals_syntax(self):
        """Go's flag.Parse requires '-flag=value' for string flags in
        mixed contexts. See test_v0521 for the fuller rationale."""
        src = self._get_rtlamr_source()
        assert 'f"-msgtype={msgtype}"' in src


class TestStrategyPopulatesDecoderOptions:
    """The strategy layer is what bridges BandConfig (config) and
    DecoderRunSpec (runtime). v0.5.33's fix only works if strategy
    actually copies band.decoder_options into the spec."""

    def test_strategy_copies_band_decoder_options_into_spec(self):
        """Grep-based test: strategy.py must pass decoder_options
        when building DecoderRunSpec."""
        from rfcensus.engine import strategy
        src = inspect.getsource(strategy)

        assert "decoder_options=dict(band.decoder_options)" in src, (
            "strategy.py must forward band.decoder_options into the "
            "DecoderRunSpec. dict() wraps the reference so later "
            "mutation of band doesn't leak into past run specs."
        )

    def test_decoder_run_spec_instantiation_has_decoder_options_kwarg(self):
        """Locate the DecoderRunSpec() call and verify decoder_options
        is among the keyword arguments. Belt-and-suspenders check in
        case the string form above gets reformatted."""
        from rfcensus.engine import strategy
        src = inspect.getsource(strategy)

        # Find the DecoderRunSpec call (there's currently only one).
        # Check that decoder_options appears in the same block as the
        # construction.
        idx = src.find("DecoderRunSpec(")
        assert idx >= 0, "strategy.py should instantiate DecoderRunSpec"

        # Grab the next ~400 chars — the entire keyword argument block.
        call_block = src[idx:idx + 500]
        assert "decoder_options=" in call_block, (
            "DecoderRunSpec(...) construction in strategy.py must "
            "include decoder_options= kwarg."
        )
