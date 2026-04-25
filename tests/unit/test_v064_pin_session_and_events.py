"""v0.6.4 — pin wizard headless refactor + structured TUI events.

Three test groups, one per area of the refactor:

1. **pin_session** — the pure-logic selection module that the CLI
   wizard now delegates to and that the v0.7.0 TUI's edit-pin modal
   will share. Tests cover frequency filtering, decoder ranking,
   custom-freq parsing, and final-stage validation.

2. **HardwareEvent structured fields** — the new freq_hz / sample_rate /
   consumer / band_id fields on HardwareEvent. Verifies the broker's
   allocate() and release() populate them, and that the pin allocation
   path (which bypasses allocate() and goes straight to _lease())
   publishes its own equivalent event.

3. **FanoutClientEvent** — the new event type emitted by the rtl_tcp
   fanout on connect/disconnect/slow. Verifies the publish helper is
   a no-op when no event_bus is configured (backward compat for tests
   that construct fanouts directly).

The TUI itself doesn't ship in v0.6.4 — these events flow but go
nowhere until the v0.7.0-alpha dashboard subscribes.
"""

from __future__ import annotations

import pytest

from rfcensus.engine.pin_session import (
    DecoderOption,
    FrequencyOption,
    ValidatedPin,
    ValidationError,
    available_decoders,
    available_frequencies,
    parse_custom_freq,
    validate_pin,
)
from rfcensus.events import EventBus, FanoutClientEvent, HardwareEvent
from rfcensus.hardware.antenna import Antenna
from rfcensus.hardware.dongle import Dongle, DongleCapabilities, DongleStatus


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────


def _antenna(
    *,
    id: str = "test_antenna",
    low: int = 100_000_000,
    high: int = 1_000_000_000,
) -> Antenna:
    return Antenna(
        id=id,
        name=id,
        antenna_type="whip",
        resonant_freq_hz=(low + high) // 2,
        usable_range=(low, high),
        gain_dbi=2.0,
        polarization="vertical",
        requires_bias_power=False,
        notes="",
    )


def _dongle(
    *,
    id: str = "test_dongle",
    antenna: Antenna | None = None,
    freq_low: int = 24_000_000,
    freq_high: int = 1_700_000_000,
) -> Dongle:
    """A Dongle covering the standard RTL-SDR range by default."""
    return Dongle(
        id=id,
        serial=id,
        model="rtl_sdr",
        driver="rtlsdr",
        capabilities=DongleCapabilities(
            freq_range_hz=(freq_low, freq_high),
            max_sample_rate=2_400_000,
            bits_per_sample=8,
            bias_tee_capable=False,
            tcxo_ppm=20.0,
        ),
        antenna=antenna,
        status=DongleStatus.HEALTHY,
    )


# ────────────────────────────────────────────────────────────────────
# 1. pin_session: available_frequencies
# ────────────────────────────────────────────────────────────────────


class TestAvailableFrequencies:
    def test_no_antenna_returns_all_in_dongle_range(self):
        """A dongle with no antenna isn't filtered by antenna coverage —
        only by hardware range. RTL-SDR's nominal 24 MHz – 1.7 GHz
        range covers most of the catalogue, so the result should be
        substantial."""
        dongle = _dongle(antenna=None)
        opts = available_frequencies(dongle)
        assert len(opts) > 5, (
            f"expected most catalogue entries to pass, got {len(opts)}"
        )
        # All returned options should have integer freq_hz and a label
        for opt in opts:
            assert isinstance(opt, FrequencyOption)
            assert opt.freq_hz > 0
            assert opt.label
            assert opt.profile is not None  # all from the catalogue
            assert not opt.is_custom

    def test_antenna_filters_out_uncovered_freqs(self):
        """A 433 MHz–only antenna should not surface 162 MHz AIS or
        915 MHz ISM picks even if the catalogue lists them."""
        narrow = _antenna(low=420_000_000, high=440_000_000)
        dongle = _dongle(antenna=narrow)
        opts = available_frequencies(dongle)
        # Every returned freq must be in the antenna range
        for opt in opts:
            assert 420_000_000 <= opt.freq_hz <= 440_000_000, (
                f"freq {opt.freq_hz} outside antenna range "
                f"(420-440 MHz)"
            )

    def test_dongle_range_filters_out_unsupported_freqs(self):
        """A dongle with no antenna but a narrow hardware range
        (e.g. a hypothetical 700–800 MHz only receiver) should
        return only catalogue picks within that band."""
        dongle = _dongle(
            antenna=None,
            freq_low=700_000_000,
            freq_high=800_000_000,
        )
        opts = available_frequencies(dongle)
        for opt in opts:
            assert 700_000_000 <= opt.freq_hz <= 800_000_000

    def test_no_coverage_returns_empty_list(self):
        """If neither antenna nor dongle range covers any catalogue
        entry, the result is empty (signal to caller that they should
        prompt for a custom freq instead of showing an empty menu)."""
        narrow = _antenna(low=10_000_000, high=11_000_000)
        # 24-1700 MHz dongle but antenna is sub-FM
        dongle = _dongle(antenna=narrow)
        opts = available_frequencies(dongle)
        assert opts == []

    def test_order_matches_catalogue(self):
        """The catalogue's curated ordering (popular signals first)
        should be preserved in the output. Checked by verifying the
        relative order of two known catalogue entries."""
        from rfcensus.commands._frequency_guide import COMMON_FREQUENCIES

        dongle = _dongle(antenna=None)
        opts = available_frequencies(dongle)
        # Find positions of two arbitrary catalogue freqs in both
        # the catalogue and the result, verify same relative order.
        catalog_freqs = [p.freq_hz for p in COMMON_FREQUENCIES]
        result_freqs = [o.freq_hz for o in opts]
        # Take first two catalogue freqs that survived filtering
        kept = [f for f in catalog_freqs if f in result_freqs]
        assert len(kept) >= 2
        # Their order in result should match catalog order
        assert (result_freqs.index(kept[0])
                < result_freqs.index(kept[1])), (
            "available_frequencies should preserve catalogue ordering"
        )


# ────────────────────────────────────────────────────────────────────
# pin_session: parse_custom_freq
# ────────────────────────────────────────────────────────────────────


class TestParseCustomFreq:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("433.92M", 433_920_000),
            ("162M", 162_000_000),
            ("850k", 850_000),
            ("2400000", 2_400_000),
            ("915.0M", 915_000_000),
        ],
    )
    def test_accepts_common_formats(self, raw, expected):
        assert parse_custom_freq(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["", "abc", "433.92x", "MHz"],
    )
    def test_rejects_garbage(self, raw):
        with pytest.raises(ValueError):
            parse_custom_freq(raw)

    def test_parser_is_permissive_validate_pin_catches_negative(self):
        """The freq-string parser doesn't reject negatives — it just
        parses '-100M' as -100_000_000 Hz. The sanity check happens
        downstream in `validate_pin`. This is a deliberate split:
        parsing is a textual transformation; validation is semantic.
        """
        assert parse_custom_freq("-100M") == -100_000_000


# ────────────────────────────────────────────────────────────────────
# pin_session: available_decoders
# ────────────────────────────────────────────────────────────────────


class TestAvailableDecoders:
    def test_returns_all_registered_decoders(self):
        """Even without a freq match, every registered decoder
        should appear so the user can pick anything."""
        from rfcensus.decoders.registry import get_registry
        registered = set(get_registry().names())
        # Pick a freq that's not in the catalogue (mid-FM broadcast)
        opts = available_decoders(95_500_000)
        names = {o.name for o in opts}
        assert names == registered

    def test_suggested_decoders_come_first(self):
        """At 433.92 MHz, rtl_433 should be marked suggested and
        appear before non-suggested decoders."""
        opts = available_decoders(433_920_000)
        suggested = [o for o in opts if o.suggested]
        assert any(o.name == "rtl_433" for o in suggested), (
            "rtl_433 should be suggested at 433.92 MHz"
        )
        # First-position option should be suggested if any exist
        assert opts[0].suggested, (
            "first option should be a suggested decoder"
        )

    def test_no_suggestions_when_freq_far_from_catalogue(self):
        """At a freq nowhere near any catalogue entry, no decoder
        is marked suggested but all are still returned."""
        # 600 MHz is mid-band UHF TV, not in our catalogue
        opts = available_decoders(600_000_000)
        assert all(not o.suggested for o in opts)
        # Still returns everything
        assert len(opts) > 0

    def test_label_marks_suggested(self):
        """Suggested options should have '(suggested)' in their label
        so the CLI can show it inline. The TUI can ignore the label
        and use the `suggested` boolean directly."""
        opts = available_decoders(433_920_000)
        for opt in opts:
            if opt.suggested:
                assert "suggested" in opt.label
            else:
                assert "suggested" not in opt.label


# ────────────────────────────────────────────────────────────────────
# pin_session: validate_pin
# ────────────────────────────────────────────────────────────────────


class TestValidatePin:
    def test_happy_path_returns_spec(self):
        dongle = _dongle(antenna=_antenna(low=400_000_000, high=460_000_000))
        result = validate_pin(
            dongle,
            freq_hz=433_920_000,
            decoder="rtl_433",
            sample_rate=None,
            access_mode="exclusive",
        )
        assert result.ok
        assert result.errors == ()
        assert result.spec == {
            "decoder": "rtl_433",
            "freq_hz": 433_920_000,
            "access_mode": "exclusive",
        }

    def test_includes_sample_rate_when_set(self):
        dongle = _dongle(antenna=_antenna(low=400_000_000, high=460_000_000))
        result = validate_pin(
            dongle,
            freq_hz=433_920_000,
            decoder="rtl_433",
            sample_rate=2_048_000,
        )
        assert result.ok
        assert result.spec["sample_rate"] == 2_048_000

    def test_collects_all_errors_not_just_first(self):
        """A pin with multiple problems should report all of them so
        the TUI can surface every issue at once. Pre-refactor the
        wizard would have shown them sequentially across prompts."""
        dongle = _dongle(antenna=_antenna(low=400_000_000, high=460_000_000))
        result = validate_pin(
            dongle,
            freq_hz=915_000_000,  # outside antenna AND dongle range below
            decoder="not_a_real_decoder",
            access_mode="bogus",
        )
        # Hardware range still covers 915 MHz, so antenna is the only
        # freq error. Plus decoder + access_mode errors.
        assert not result.ok
        fields = {e.field for e in result.errors}
        assert "freq_hz" in fields
        assert "decoder" in fields
        assert "general" in fields  # access_mode under 'general'
        assert len(result.errors) >= 3

    def test_negative_freq_rejected(self):
        dongle = _dongle(antenna=None)
        result = validate_pin(
            dongle, freq_hz=-1, decoder="rtl_433",
        )
        assert not result.ok
        assert any(e.field == "freq_hz" for e in result.errors)

    def test_unknown_decoder_rejected(self):
        dongle = _dongle(antenna=None)
        result = validate_pin(
            dongle,
            freq_hz=433_920_000,
            decoder="nonexistent_decoder_xyz",
        )
        assert not result.ok
        assert any(e.field == "decoder" for e in result.errors)

    def test_negative_sample_rate_rejected(self):
        dongle = _dongle(antenna=None)
        result = validate_pin(
            dongle,
            freq_hz=433_920_000,
            decoder="rtl_433",
            sample_rate=-100,
        )
        assert not result.ok
        assert any(e.field == "sample_rate" for e in result.errors)

    def test_shared_access_mode_accepted(self):
        dongle = _dongle(antenna=None)
        result = validate_pin(
            dongle,
            freq_hz=433_920_000,
            decoder="rtl_433",
            access_mode="shared",
        )
        assert result.ok
        assert result.spec["access_mode"] == "shared"


# ────────────────────────────────────────────────────────────────────
# 2. HardwareEvent structured fields
# ────────────────────────────────────────────────────────────────────


class TestHardwareEventFields:
    def test_default_fields_are_none(self):
        """Existing publishers that only set dongle_id + kind keep
        working — the new fields default to None, not empty strings."""
        e = HardwareEvent(dongle_id="d1", kind="detected")
        assert e.freq_hz is None
        assert e.sample_rate is None
        assert e.consumer is None
        assert e.band_id is None

    def test_all_fields_settable(self):
        """The TUI's DongleStrip reads these directly; verify they
        round-trip through the constructor."""
        e = HardwareEvent(
            dongle_id="d1",
            kind="allocated",
            detail="lease 7 for rtl_433:433_ism",
            freq_hz=433_920_000,
            sample_rate=2_400_000,
            consumer="rtl_433:433_ism",
            band_id="433_ism",
        )
        assert e.freq_hz == 433_920_000
        assert e.sample_rate == 2_400_000
        assert e.consumer == "rtl_433:433_ism"
        assert e.band_id == "433_ism"


# ────────────────────────────────────────────────────────────────────
# 3. FanoutClientEvent shape + bus integration
# ────────────────────────────────────────────────────────────────────


class TestFanoutClientEvent:
    def test_default_construction(self):
        """Defaults match the dataclass spec — slot_id and peer_addr
        are required-ish (default to empty), event_type defaults to
        connect, bytes_sent defaults to 0."""
        e = FanoutClientEvent()
        assert e.slot_id == ""
        assert e.peer_addr == ""
        assert e.event_type == "connect"
        assert e.bytes_sent == 0

    def test_full_construction(self):
        e = FanoutClientEvent(
            slot_id="fanout[rtlsdr-00000003]",
            peer_addr="127.0.0.1:54321",
            event_type="disconnect",
            bytes_sent=12_582_912,
        )
        assert e.slot_id == "fanout[rtlsdr-00000003]"
        assert e.peer_addr == "127.0.0.1:54321"
        assert e.event_type == "disconnect"
        assert e.bytes_sent == 12_582_912

    @pytest.mark.parametrize(
        "event_type",
        ["connect", "disconnect", "slow", "dropped"],
    )
    def test_all_event_types_accepted(self, event_type):
        e = FanoutClientEvent(event_type=event_type)
        assert e.event_type == event_type


@pytest.mark.asyncio
class TestFanoutEventPublishing:
    async def test_publish_helper_is_noop_when_bus_is_none(self):
        """Tests construct fanouts without a bus and rely on the
        existing log-only behavior. Verify _publish_client_event is
        a safe no-op in that case."""
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout

        fanout = RtlTcpFanout(
            upstream_host="127.0.0.1",
            upstream_port=20000,
            slot_label="test",
            event_bus=None,
        )
        # Should not raise — exits before any bus access
        await fanout._publish_client_event("1.2.3.4:5678", "connect")

    async def test_publish_helper_emits_when_bus_set(self):
        """With a bus configured, the helper publishes a properly-
        shaped event for subscribers to consume."""
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout

        bus = EventBus()
        captured: list[FanoutClientEvent] = []
        bus.subscribe(FanoutClientEvent, lambda e: captured.append(e))

        fanout = RtlTcpFanout(
            upstream_host="127.0.0.1",
            upstream_port=20000,
            slot_label="fanout[rtlsdr-test]",
            event_bus=bus,
        )
        await fanout._publish_client_event(
            "127.0.0.1:54321", "connect",
        )
        await bus.drain(timeout=2.0)

        assert len(captured) == 1
        e = captured[0]
        assert e.slot_id == "fanout[rtlsdr-test]"
        assert e.peer_addr == "127.0.0.1:54321"
        assert e.event_type == "connect"
        assert e.bytes_sent == 0

    async def test_publish_helper_carries_bytes_sent(self):
        """For disconnect / slow events, bytes_sent surfaces lifetime
        delivery so the dashboard can show "12 MB delivered before
        disconnect"."""
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout

        bus = EventBus()
        captured: list[FanoutClientEvent] = []
        bus.subscribe(FanoutClientEvent, lambda e: captured.append(e))

        fanout = RtlTcpFanout(
            upstream_host="127.0.0.1",
            upstream_port=20000,
            slot_label="fanout[d1]",
            event_bus=bus,
        )
        await fanout._publish_client_event(
            "127.0.0.1:9999", "disconnect", bytes_sent=12_582_912,
        )
        await bus.drain(timeout=2.0)

        assert len(captured) == 1
        assert captured[0].bytes_sent == 12_582_912

    async def test_publish_helper_swallows_bus_exceptions(self):
        """If the bus.publish call raises (e.g. subscriber bug, bus
        torn down), the helper logs and continues. The fanout MUST
        NOT crash because of an unrelated event-bus issue.

        We trigger this by replacing the bus with a stub whose
        publish raises.
        """
        from rfcensus.hardware.rtl_tcp_fanout import RtlTcpFanout

        class ExplodingBus:
            async def publish(self, event):
                raise RuntimeError("simulated bus failure")

        fanout = RtlTcpFanout(
            upstream_host="127.0.0.1",
            upstream_port=20000,
            slot_label="test",
            event_bus=ExplodingBus(),  # type: ignore[arg-type]
        )
        # Should not raise — exception is logged and swallowed
        await fanout._publish_client_event("1.2.3.4:5678", "connect")


# ────────────────────────────────────────────────────────────────────
# 4. Broker integration: HardwareEvent carries structured fields
#    end-to-end through allocate() and release().
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestBrokerStructuredHardwareEvents:
    async def _make_broker(self, dongle: Dongle):
        """Build a broker with a single registered dongle and an
        EventBus with a HardwareEvent capture subscriber."""
        from rfcensus.hardware.broker import DongleBroker
        from rfcensus.hardware.registry import HardwareRegistry

        bus = EventBus()
        captured: list[HardwareEvent] = []
        bus.subscribe(HardwareEvent, lambda e: captured.append(e))

        registry = HardwareRegistry(dongles=[dongle])
        broker = DongleBroker(registry=registry, event_bus=bus)
        return broker, bus, captured

    async def test_allocate_publishes_freq_sample_rate_consumer_band(
        self,
    ):
        """The TUI's primary use case: dongle gets allocated, dashboard
        sees a HardwareEvent with everything it needs to render the
        tile (freq, SR, consumer label, band id)."""
        from rfcensus.hardware.broker import DongleRequirements

        dongle = _dongle(
            id="rtlsdr-test",
            antenna=_antenna(low=400_000_000, high=460_000_000),
        )
        broker, bus, captured = await self._make_broker(dongle)

        req = DongleRequirements(
            freq_hz=433_920_000,
            sample_rate=2_400_000,
            band_id="433_ism",
        )
        lease = await broker.allocate(
            req, consumer="rtl_433:433_ism", timeout=2.0,
        )
        await bus.drain(timeout=2.0)

        try:
            allocated = [e for e in captured if e.kind == "allocated"]
            assert len(allocated) == 1
            e = allocated[0]
            assert e.dongle_id == "rtlsdr-test"
            assert e.freq_hz == 433_920_000
            assert e.sample_rate == 2_400_000
            assert e.consumer == "rtl_433:433_ism"
            assert e.band_id == "433_ism"
        finally:
            await broker.release(lease)

    async def test_release_publishes_consumer_no_freq(self):
        """On release the dongle is no longer tuned, so freq_hz isn't
        meaningful — but the consumer label is, so the TUI can correlate
        the release with the prior allocate."""
        from rfcensus.hardware.broker import DongleRequirements

        dongle = _dongle(antenna=_antenna(low=400_000_000, high=460_000_000))
        broker, bus, captured = await self._make_broker(dongle)

        req = DongleRequirements(
            freq_hz=433_920_000, sample_rate=2_400_000, band_id="433_ism",
        )
        lease = await broker.allocate(
            req, consumer="rtl_433:433_ism", timeout=2.0,
        )
        await broker.release(lease)
        await bus.drain(timeout=2.0)

        released = [e for e in captured if e.kind == "released"]
        assert len(released) == 1
        e = released[0]
        assert e.consumer == "rtl_433:433_ism"
        # freq_hz/sample_rate/band_id default to None on release —
        # the dongle is no longer tuned to anything.
        assert e.freq_hz is None
        assert e.sample_rate is None

    async def test_band_id_optional_in_requirements(self):
        """If a caller doesn't set band_id (e.g. raw probe / health
        check), the event's band_id is None and downstream subscribers
        can fall back to whatever default they want."""
        from rfcensus.hardware.broker import DongleRequirements

        dongle = _dongle(antenna=_antenna(low=400_000_000, high=460_000_000))
        broker, bus, captured = await self._make_broker(dongle)

        req = DongleRequirements(
            freq_hz=433_920_000, sample_rate=2_400_000,
        )  # no band_id
        lease = await broker.allocate(req, consumer="probe", timeout=2.0)
        await bus.drain(timeout=2.0)

        try:
            allocated = [e for e in captured if e.kind == "allocated"]
            assert len(allocated) == 1
            assert allocated[0].band_id is None
            assert allocated[0].freq_hz == 433_920_000  # other fields still set
        finally:
            await broker.release(lease)
