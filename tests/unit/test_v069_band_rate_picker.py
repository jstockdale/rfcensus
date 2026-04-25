"""v0.6.9 — band-level shared sample rate picker.

Determinism contract: given a band's full decoder cohort, the
shared-slot sample rate must be selected up front and identically
regardless of allocation order. Decoders with
requires_exact_sample_rate establish the rate; flexible decoders
follow.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rfcensus.config.schema import BandConfig
from rfcensus.decoders.base import DecoderBase, DecoderCapabilities
from rfcensus.engine.strategy import _band_shared_sample_rate


def _band(band_id="test", freq=915_000_000, bw=2_000_000, decoders=()):
    """Minimal BandConfig for picker tests."""
    return BandConfig(
        id=band_id,
        name=band_id,
        freq_low=freq - bw // 2,
        freq_high=freq + bw // 2,
        suggested_decoders=list(decoders),
    )


def _ctx_with_decoders(decoder_classes):
    """StrategyContext stub whose decoder_registry returns the given
    decoder classes. _pick_decoders calls names() / get() / instantiate()
    in a tight loop — we satisfy that contract without a full registry."""
    ctx = MagicMock()
    ctx.decoder_registry.names.return_value = [c.capabilities.name for c in decoder_classes]
    classes_by_name = {c.capabilities.name: c for c in decoder_classes}
    ctx.decoder_registry.get.side_effect = lambda n: classes_by_name.get(n)

    def instantiate(name, _config):
        cls = classes_by_name.get(name)
        if cls is None:
            return None
        # The picker uses .capabilities on the instance, which is a
        # class attribute — bare construction is fine.
        return cls()

    ctx.decoder_registry.instantiate.side_effect = instantiate
    ctx.config = MagicMock()
    return ctx


def _make_decoder(name, *, exact_rate=None, preferred=2_400_000, freq_low=900e6, freq_high=930e6):
    """Build a DecoderBase subclass with the given capabilities."""

    class _Decoder(DecoderBase):
        capabilities = DecoderCapabilities(
            name=name,
            protocols=[name],
            freq_ranges=((int(freq_low), int(freq_high)),),
            min_sample_rate=preferred if exact_rate else 1_024_000,
            preferred_sample_rate=exact_rate if exact_rate else preferred,
            requires_exact_sample_rate=bool(exact_rate),
            requires_exclusive_dongle=False,
            external_binary=name,
        )

        async def check_available(self):  # pragma: no cover - not used
            from rfcensus.decoders.base import DecoderAvailability
            return DecoderAvailability(name=self.name, available=True)

        async def run(self, spec):  # pragma: no cover - not used
            from rfcensus.decoders.base import DecoderResult
            return DecoderResult(name=self.name)

    _Decoder.__name__ = f"_{name.title()}Decoder"
    return _Decoder


class TestBandRatePicker:
    """Determinism: same band, same decoders → same rate every time,
    regardless of order or allocation timing."""

    def test_no_decoders_falls_back_to_default(self):
        """Empty band → 2.4M default. Used by lora_survey-only paths
        that would otherwise have no rate to ask for."""
        rate = _band_shared_sample_rate(
            _band(decoders=[]), _ctx_with_decoders([])
        )
        assert rate == 2_400_000

    def test_single_flexible_decoder_uses_its_preferred(self):
        Rtl433 = _make_decoder("rtl_433", preferred=2_400_000)
        rate = _band_shared_sample_rate(
            _band(decoders=["rtl_433"]),
            _ctx_with_decoders([Rtl433]),
        )
        assert rate == 2_400_000

    def test_single_exact_decoder_uses_its_rate(self):
        """rtlamr alone: slot rate is rtlamr's exact rate."""
        Rtlamr = _make_decoder("rtlamr", exact_rate=2_359_296)
        rate = _band_shared_sample_rate(
            _band(decoders=["rtlamr"]),
            _ctx_with_decoders([Rtlamr]),
        )
        assert rate == 2_359_296

    def test_exact_decoder_wins_over_flexible(self):
        """rtl_433 + rtlamr together: rtlamr's exact rate wins. THIS
        is the bug the user hit on metatron — without this guarantee,
        whichever allocated first established the slot rate, and the
        other decoder either failed loudly (v0.6.8 filter) or silently
        produced wrong-rate output."""
        Rtl433 = _make_decoder("rtl_433", preferred=2_400_000)
        Rtlamr = _make_decoder("rtlamr", exact_rate=2_359_296)
        rate = _band_shared_sample_rate(
            _band(decoders=["rtl_433", "rtlamr"]),
            _ctx_with_decoders([Rtl433, Rtlamr]),
        )
        assert rate == 2_359_296

    def test_order_independence(self):
        """The picker must give the same answer regardless of which
        order decoders appear in suggested_decoders or the registry —
        determinism is the whole point."""
        Rtl433 = _make_decoder("rtl_433", preferred=2_400_000)
        Rtlamr = _make_decoder("rtlamr", exact_rate=2_359_296)

        rate_a = _band_shared_sample_rate(
            _band(decoders=["rtl_433", "rtlamr"]),
            _ctx_with_decoders([Rtl433, Rtlamr]),
        )
        rate_b = _band_shared_sample_rate(
            _band(decoders=["rtlamr", "rtl_433"]),
            _ctx_with_decoders([Rtlamr, Rtl433]),
        )
        assert rate_a == rate_b == 2_359_296

    def test_multiple_flexible_picks_max(self):
        """Several flexible decoders → highest preferred rate wins
        (max bandwidth, and any decoder happy at low rate is happy
        at high rate since min_sample_rate is a floor not ceiling)."""
        D1 = _make_decoder("d1", preferred=1_024_000)
        D2 = _make_decoder("d2", preferred=2_400_000)
        D3 = _make_decoder("d3", preferred=1_500_000)
        rate = _band_shared_sample_rate(
            _band(decoders=["d1", "d2", "d3"]),
            _ctx_with_decoders([D1, D2, D3]),
        )
        assert rate == 2_400_000

    def test_two_exact_in_agreement_succeed(self):
        """Two exact-rate decoders that agree on the rate: fine."""
        D1 = _make_decoder("d1", exact_rate=2_048_000)
        D2 = _make_decoder("d2", exact_rate=2_048_000)
        rate = _band_shared_sample_rate(
            _band(decoders=["d1", "d2"]),
            _ctx_with_decoders([D1, D2]),
        )
        assert rate == 2_048_000

    def test_two_exact_in_conflict_picks_one_and_warns(self, caplog):
        """Two exact-rate decoders that disagree: this is an
        unsatisfiable band. We pick the LOWEST rate (deterministic,
        so the failure is reproducible) and log loudly so the operator
        sees WHY one of their decoders is producing zero output."""
        import logging
        D1 = _make_decoder("d1", exact_rate=2_048_000)
        D2 = _make_decoder("d2", exact_rate=2_359_296)

        with caplog.at_level(logging.WARNING):
            rate = _band_shared_sample_rate(
                _band(decoders=["d1", "d2"]),
                _ctx_with_decoders([D1, D2]),
            )

        # Lowest rate picked (deterministic).
        assert rate == 2_048_000
        # Warning logged so the operator knows.
        assert any(
            "conflicting exact-rate" in r.message
            for r in caplog.records
        ), f"expected conflict warning, got: {[r.message for r in caplog.records]}"


class TestRequiresExactSampleRateCapability:
    """The capability flag must be properly exposed on real decoders."""

    def test_rtlamr_declares_exact_rate(self):
        """rtlamr's demod hardcodes 2,359,296 — capability must reflect
        that or the band-level picker can't honor it."""
        from rfcensus.decoders.builtin.rtlamr import RtlamrDecoder
        assert RtlamrDecoder.capabilities.requires_exact_sample_rate is True
        assert RtlamrDecoder.capabilities.preferred_sample_rate == 2_359_296

    def test_rtl_433_does_not_require_exact(self):
        """rtl_433 is rate-flexible — must NOT declare exact-rate or
        the picker would lock the slot to rtl_433's rate even when
        rtlamr is also on the band."""
        from rfcensus.decoders.builtin.rtl_433 import Rtl433Decoder
        assert Rtl433Decoder.capabilities.requires_exact_sample_rate is False


class TestSharedSlotCompatExactRate:
    """The compat predicate gate against silently joining a wrong-rate
    slot. Belt-and-suspenders — band-level picker should prevent
    the situation, but if anything bypasses the picker the gate fails
    the allocation rather than silently doing wrong DSP."""

    def _slot(self, rate):
        """Minimal _SharedSlot for compat predicate testing."""
        from rfcensus.hardware.broker import _SharedSlot
        return _SharedSlot(
            process=MagicMock(),
            host="127.0.0.1", port=0,
            sample_rate=rate, center_freq_hz=915_000_000,
        )

    def test_exact_match_accepts(self):
        from rfcensus.hardware.broker import _shared_slot_compatible
        slot = self._slot(2_359_296)
        assert _shared_slot_compatible(
            slot, 915_000_000, 2_359_296, require_exact_rate=True
        )

    def test_exact_mismatch_rejects(self):
        """rtlamr asks for 2,359,296 but slot is at 2,400,000 →
        rejected. Without this gate the slot would silently accept
        (because 2.4M ≥ 2.359M) and rtlamr would process wrong-rate
        samples."""
        from rfcensus.hardware.broker import _shared_slot_compatible
        slot = self._slot(2_400_000)
        assert not _shared_slot_compatible(
            slot, 915_000_000, 2_359_296, require_exact_rate=True
        )

    def test_flexible_consumer_still_accepts_higher_slot(self):
        """Without require_exact_rate, the legacy "slot rate ≥ req"
        rule applies — a flexible decoder can join a higher-rate slot
        and downsample if needed."""
        from rfcensus.hardware.broker import _shared_slot_compatible
        slot = self._slot(2_400_000)
        assert _shared_slot_compatible(
            slot, 915_000_000, 2_048_000, require_exact_rate=False
        )
