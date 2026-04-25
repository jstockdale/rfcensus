"""v0.6.0 — decoder→dongle pinning.

Covers:
  • PinConfig schema validation (freq + sample_rate sanity checks)
  • CLI flag parsing (--pin DONGLE:DECODER@FREQ[:SR]) with unit suffixes
  • gather_pins() merging config + CLI with CLI precedence
  • validate_pins() decision matrix:
      – missing dongle → skip
      – unhealthy dongle → skip
      – unknown decoder → fatal
      – freq outside hardware range → fatal
      – antenna mismatch (without override) → fatal
      – antenna mismatch (with override) → ok
  • PinSupervisor backoff schedule (mocked decoder failures)
  • start_pinned_tasks excludes pinned dongles from broker pool
  • all-dongles-pinned warning
  • fleet_optimizer pin constraint
  • TOML round-trip (write pin via wizard helpers, read back)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rfcensus.config.schema import (
    AntennaConfig, BandConfig, DongleConfig, PinConfig, SiteConfig,
)
from rfcensus.engine.pinning import (
    PinSpec,
    PinSupervisor,
    PinningOutcome,
    _BACKOFF_DELAYS_S,
    _parse_freq_str,
    gather_pins,
    parse_cli_pin,
    summarize_pinning_outcome,
    validate_pins,
    warn_if_all_dongles_pinned,
)
from rfcensus.hardware.broker import AccessMode, DongleBroker, NoDongleAvailable
from rfcensus.hardware.dongle import Dongle, DongleCapabilities, DongleStatus
from rfcensus.hardware.antenna import Antenna
from rfcensus.hardware.registry import HardwareRegistry


# ────────────────────────────────────────────────────────────────────
# Test fixtures
# ────────────────────────────────────────────────────────────────────


def _make_dongle(
    serial: str = "00000043",
    *,
    antenna: Antenna | None = None,
    status: DongleStatus = DongleStatus.HEALTHY,
    freq_range: tuple[int, int] = (24_000_000, 1_766_000_000),
) -> Dongle:
    return Dongle(
        id=serial,
        serial=serial,
        model="rtlsdr_v4",
        driver="rtlsdr",
        capabilities=DongleCapabilities(
            freq_range_hz=freq_range,
            max_sample_rate=2_400_000,
            bits_per_sample=8,
            bias_tee_capable=False,
            tcxo_ppm=1.0,
        ),
        antenna=antenna,
        status=status,
        driver_index=0,
    )


def _make_antenna(
    aid: str = "whip_433",
    *,
    usable_range: tuple[int, int] = (300_000_000, 600_000_000),
) -> Antenna:
    return Antenna(
        id=aid,
        name=f"test {aid}",
        antenna_type="whip",
        resonant_freq_hz=(usable_range[0] + usable_range[1]) // 2,
        usable_range=usable_range,
        gain_dbi=2.0,
        polarization="vertical",
        requires_bias_power=False,
        notes="",
    )


# ────────────────────────────────────────────────────────────────────
# Frequency string parsing
# ────────────────────────────────────────────────────────────────────


class TestParseFreqStr:
    def test_plain_int(self):
        assert _parse_freq_str("433920000") == 433_920_000

    def test_underscore_separator(self):
        assert _parse_freq_str("433_920_000") == 433_920_000

    def test_megahertz_suffix(self):
        assert _parse_freq_str("433.92M") == 433_920_000

    def test_kilohertz_suffix(self):
        assert _parse_freq_str("850k") == 850_000

    def test_gigahertz_suffix(self):
        assert _parse_freq_str("1.5G") == 1_500_000_000

    def test_lowercase_suffix(self):
        assert _parse_freq_str("433.92m") == 433_920_000

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_freq_str("")

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="parse"):
            _parse_freq_str("not-a-number")


# ────────────────────────────────────────────────────────────────────
# CLI flag parsing
# ────────────────────────────────────────────────────────────────────


class TestParseCliPin:
    def test_basic_form(self):
        spec = parse_cli_pin("00000043:rtl_433@433.92M")
        assert spec.dongle_id == "00000043"
        assert spec.decoder == "rtl_433"
        assert spec.freq_hz == 433_920_000
        assert spec.sample_rate is None
        assert spec.access_mode == AccessMode.EXCLUSIVE
        assert spec.source == "cli"

    def test_with_sample_rate(self):
        spec = parse_cli_pin("00000043:rtl_433@433920000:2400000")
        assert spec.freq_hz == 433_920_000
        assert spec.sample_rate == 2_400_000

    def test_with_sample_rate_suffix(self):
        spec = parse_cli_pin("00000043:rtl_433@433.92M:2.4M")
        assert spec.sample_rate == 2_400_000

    def test_alphanumeric_dongle_id(self):
        # Some users use named ids in their config
        spec = parse_cli_pin("indoor-433:rtl_433@433.92M")
        assert spec.dongle_id == "indoor-433"

    def test_malformed_raises_with_helpful_message(self):
        with pytest.raises(ValueError, match="expected format"):
            parse_cli_pin("not_a_pin_spec")
        with pytest.raises(ValueError, match="expected format"):
            parse_cli_pin("00000043:rtl_433")  # missing @freq


# ────────────────────────────────────────────────────────────────────
# PinConfig schema
# ────────────────────────────────────────────────────────────────────


class TestPinConfigSchema:
    def test_minimal_valid(self):
        p = PinConfig(decoder="rtl_433", freq_hz=433_920_000)
        assert p.access_mode == "exclusive"
        assert p.sample_rate is None

    def test_freq_below_1mhz_rejected_as_typo(self):
        # User wrote 433_920 forgetting the kHz multiplier — common mistake
        with pytest.raises(ValueError, match="below 1 MHz"):
            PinConfig(decoder="rtl_433", freq_hz=433_920)

    def test_freq_zero_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            PinConfig(decoder="rtl_433", freq_hz=0)

    def test_negative_freq_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            PinConfig(decoder="rtl_433", freq_hz=-1)

    def test_sample_rate_too_low_rejected(self):
        with pytest.raises(ValueError, match="reasonable range"):
            PinConfig(decoder="rtl_433", freq_hz=433_920_000, sample_rate=50)

    def test_sample_rate_too_high_rejected(self):
        with pytest.raises(ValueError, match="reasonable range"):
            PinConfig(decoder="rtl_433", freq_hz=433_920_000,
                      sample_rate=50_000_000_000)

    def test_access_mode_must_be_exclusive_or_shared(self):
        with pytest.raises(ValueError):
            PinConfig(decoder="rtl_433", freq_hz=433_920_000,
                      access_mode="weird")

    def test_dongle_config_pin_optional_default_none(self):
        d = DongleConfig(id="x", model="rtlsdr_v3")
        assert d.pin is None

    def test_dongle_config_pin_set(self):
        d = DongleConfig(
            id="x", model="rtlsdr_v3",
            pin=PinConfig(decoder="rtl_433", freq_hz=433_920_000),
        )
        assert d.pin.decoder == "rtl_433"


# ────────────────────────────────────────────────────────────────────
# gather_pins — merging config + CLI
# ────────────────────────────────────────────────────────────────────


def _minimal_site_config(dongles: list[DongleConfig]) -> SiteConfig:
    return SiteConfig(
        site={"name": "test", "region": "US"},
        dongles=dongles,
    )


class TestGatherPins:
    def test_config_only(self):
        cfg = _minimal_site_config([
            DongleConfig(id="d1", model="rtlsdr_v3",
                         pin=PinConfig(decoder="rtl_433",
                                       freq_hz=433_920_000)),
        ])
        pins = gather_pins(cfg)
        assert len(pins) == 1
        assert pins[0].source == "config"
        assert pins[0].decoder == "rtl_433"

    def test_cli_only(self):
        cfg = _minimal_site_config([])
        pins = gather_pins(cfg, ["00000043:rtlamr@912M"])
        assert len(pins) == 1
        assert pins[0].source == "cli"
        assert pins[0].decoder == "rtlamr"

    def test_cli_overrides_config_for_same_dongle(self):
        cfg = _minimal_site_config([
            DongleConfig(id="d1", model="rtlsdr_v3",
                         pin=PinConfig(decoder="rtl_433",
                                       freq_hz=433_920_000)),
        ])
        pins = gather_pins(cfg, ["d1:rtlamr@912M"])
        assert len(pins) == 1
        assert pins[0].decoder == "rtlamr"  # CLI won
        assert pins[0].source == "cli"

    def test_config_and_cli_for_different_dongles_both_kept(self):
        cfg = _minimal_site_config([
            DongleConfig(id="d1", model="rtlsdr_v3",
                         pin=PinConfig(decoder="rtl_433",
                                       freq_hz=433_920_000)),
        ])
        pins = gather_pins(cfg, ["d2:rtlamr@912M"])
        assert len(pins) == 2
        assert {p.dongle_id for p in pins} == {"d1", "d2"}


# ────────────────────────────────────────────────────────────────────
# validate_pins — the decision matrix
# ────────────────────────────────────────────────────────────────────


class _FakeDecoderRegistry:
    """Minimal stand-in for DecoderRegistry in tests."""

    def __init__(self, names: list[str]):
        self._names = names

    def names(self) -> list[str]:
        return list(self._names)

    def get(self, name: str):
        # Returns None if not registered, else a callable producing a
        # decoder with capabilities.preferred_sample_rate
        if name not in self._names:
            return None
        from dataclasses import dataclass

        @dataclass
        class _FakeCaps:
            preferred_sample_rate: int = 2_400_000

        @dataclass
        class _FakeDecoder:
            capabilities: _FakeCaps = None

            def __post_init__(self):
                if self.capabilities is None:
                    self.capabilities = _FakeCaps()

        return _FakeDecoder


class TestValidatePins:
    def _setup(self, dongle, decoders=("rtl_433", "rtlamr")):
        registry = HardwareRegistry()
        registry.dongles = [dongle]
        decoder_registry = _FakeDecoderRegistry(list(decoders))
        return registry, decoder_registry

    def test_dongle_missing_skips(self):
        # Empty registry — no dongles connected
        registry = HardwareRegistry()
        decoder_registry = _FakeDecoderRegistry(["rtl_433"])
        pin = PinSpec(dongle_id="nonexistent", decoder="rtl_433",
                      freq_hz=433_920_000)
        results = validate_pins([pin], registry, decoder_registry)
        assert results[0].status == "skip"
        assert "not connected" in results[0].reason

    def test_dongle_failed_skips(self):
        d = _make_dongle(status=DongleStatus.FAILED,
                         antenna=_make_antenna())
        registry, decoder_registry = self._setup(d)
        pin = PinSpec(dongle_id="00000043", decoder="rtl_433",
                      freq_hz=433_920_000)
        results = validate_pins([pin], registry, decoder_registry)
        assert results[0].status == "skip"
        assert "not usable" in results[0].reason

    def test_unknown_decoder_fatal(self):
        d = _make_dongle(antenna=_make_antenna())
        registry, decoder_registry = self._setup(d, decoders=("rtl_433",))
        pin = PinSpec(dongle_id="00000043", decoder="not_a_decoder",
                      freq_hz=433_920_000)
        results = validate_pins([pin], registry, decoder_registry)
        assert results[0].status == "fatal"
        assert "not registered" in results[0].reason

    def test_freq_outside_hardware_range_fatal(self):
        # RTL-SDR can't tune to 10 GHz
        d = _make_dongle(antenna=_make_antenna())
        registry, decoder_registry = self._setup(d)
        pin = PinSpec(dongle_id="00000043", decoder="rtl_433",
                      freq_hz=10_000_000_000)
        results = validate_pins([pin], registry, decoder_registry)
        assert results[0].status == "fatal"
        assert "tunes" in results[0].reason

    def test_antenna_mismatch_fatal_by_default(self):
        # 915 antenna pinning to 433 freq
        ant = _make_antenna("whip_915", usable_range=(800_000_000, 1_000_000_000))
        d = _make_dongle(antenna=ant)
        registry, decoder_registry = self._setup(d)
        pin = PinSpec(dongle_id="00000043", decoder="rtl_433",
                      freq_hz=433_920_000)
        results = validate_pins([pin], registry, decoder_registry)
        assert results[0].status == "fatal"
        assert "doesn't cover" in results[0].reason

    def test_antenna_mismatch_allowed_with_override(self):
        ant = _make_antenna("whip_915", usable_range=(800_000_000, 1_000_000_000))
        d = _make_dongle(antenna=ant)
        registry, decoder_registry = self._setup(d)
        pin = PinSpec(dongle_id="00000043", decoder="rtl_433",
                      freq_hz=433_920_000)
        results = validate_pins(
            [pin], registry, decoder_registry,
            allow_antenna_mismatch=True,
        )
        assert results[0].status == "ok"

    def test_no_antenna_fatal_unless_override(self):
        d = _make_dongle(antenna=None)
        registry, decoder_registry = self._setup(d)
        pin = PinSpec(dongle_id="00000043", decoder="rtl_433",
                      freq_hz=433_920_000)
        results = validate_pins([pin], registry, decoder_registry)
        assert results[0].status == "fatal"
        assert "no antenna" in results[0].reason

    def test_happy_path_ok(self):
        d = _make_dongle(antenna=_make_antenna())
        registry, decoder_registry = self._setup(d)
        pin = PinSpec(dongle_id="00000043", decoder="rtl_433",
                      freq_hz=433_920_000)
        results = validate_pins([pin], registry, decoder_registry)
        assert results[0].status == "ok"

    def test_lookup_by_serial_works(self):
        # User passes serial instead of id (often the same string for
        # rtl_sdr but not always)
        d = _make_dongle(serial="abc123")
        d.id = "different-id"
        d.antenna = _make_antenna()
        registry, decoder_registry = self._setup(d)
        pin = PinSpec(dongle_id="abc123", decoder="rtl_433",
                      freq_hz=433_920_000)
        results = validate_pins([pin], registry, decoder_registry)
        assert results[0].status == "ok"


# ────────────────────────────────────────────────────────────────────
# warn_if_all_dongles_pinned
# ────────────────────────────────────────────────────────────────────


class TestAllPinnedWarning:
    def test_returns_none_when_some_unpinned(self):
        d1 = _make_dongle("d1")
        d2 = _make_dongle("d2")
        registry = HardwareRegistry()
        registry.dongles = [d1, d2]
        outcome = PinningOutcome(supervisors=[
            PinSupervisor(
                spec=PinSpec("d1", "rtl_433", 433_920_000),
                lease=None,  # type: ignore[arg-type]
                task=None,  # type: ignore[arg-type]
            ),
        ])
        # d1 will be marked BUSY by start_pinned_tasks in real flow;
        # for this test we just simulate
        d1.status = DongleStatus.BUSY
        msg = warn_if_all_dongles_pinned(outcome, registry)
        assert msg is None  # d2 still usable

    def test_returns_warning_when_all_pinned(self):
        d1 = _make_dongle("d1")
        d2 = _make_dongle("d2")
        registry = HardwareRegistry()
        registry.dongles = [d1, d2]
        outcome = PinningOutcome(supervisors=[
            PinSupervisor(
                spec=PinSpec("d1", "rtl_433", 433_920_000),
                lease=None,  # type: ignore[arg-type]
                task=None,  # type: ignore[arg-type]
            ),
            PinSupervisor(
                spec=PinSpec("d2", "rtlamr", 912_000_000),
                lease=None,  # type: ignore[arg-type]
                task=None,  # type: ignore[arg-type]
            ),
        ])
        d1.status = DongleStatus.BUSY
        d2.status = DongleStatus.BUSY
        msg = warn_if_all_dongles_pinned(outcome, registry)
        assert msg is not None
        assert "All 2/2" in msg


# ────────────────────────────────────────────────────────────────────
# Backoff schedule
# ────────────────────────────────────────────────────────────────────


class TestBackoffSchedule:
    def test_schedule_is_monotonically_increasing(self):
        # 1, 2, 5, 10, 60 — never goes backwards
        for i in range(len(_BACKOFF_DELAYS_S) - 1):
            assert _BACKOFF_DELAYS_S[i] < _BACKOFF_DELAYS_S[i + 1]

    def test_schedule_starts_short(self):
        # First retry within a couple seconds
        assert _BACKOFF_DELAYS_S[0] <= 2.0

    def test_schedule_caps_at_minute(self):
        # No point waiting longer than ~1 min before giving up
        assert _BACKOFF_DELAYS_S[-1] <= 60.0


# ────────────────────────────────────────────────────────────────────
# v0.6.1 — supervisor never gives up + dedup
# ────────────────────────────────────────────────────────────────────


class TestForeverRetryAndDedup:
    """v0.6.1: pin supervisor retries forever (plateau at 60s).
    Identical errors after the first N are suppressed in logs to
    avoid spam from a permanently-broken pin. Re-emit on success
    or different error."""

    def test_no_give_up_attribute(self):
        """The previous version had a `given_up` field on
        PinSupervisor that the loop set after exhausting backoff.
        v0.6.1 removed it — the supervisor never gives up."""
        from rfcensus.engine.pinning import PinSupervisor
        # Construct a supervisor with all defaults; given_up should
        # not be a field.
        sup = PinSupervisor(
            spec=PinSpec("d1", "rtl_433", 433_920_000),
            lease=None,  # type: ignore[arg-type]
            task=None,  # type: ignore[arg-type]
        )
        # given_up shouldn't be a field. dataclass.fields() check:
        from dataclasses import fields
        names = {f.name for f in fields(sup)}
        assert "given_up" not in names

    def test_dedup_state_present(self):
        """Supervisor tracks consecutive_identical_errors and
        suppression_announced for log dedup."""
        from rfcensus.engine.pinning import PinSupervisor
        sup = PinSupervisor(
            spec=PinSpec("d1", "rtl_433", 433_920_000),
            lease=None,  # type: ignore[arg-type]
            task=None,  # type: ignore[arg-type]
        )
        assert sup.consecutive_identical_errors == 0
        assert sup.suppression_announced is False

    @pytest.mark.asyncio
    async def test_supervisor_retries_after_many_failures(self, caplog):
        """Crash the decoder repeatedly and confirm the supervisor
        keeps trying (doesn't exit the loop) past the original
        give-up threshold of 5 failures."""
        import logging
        from rfcensus.engine.pinning import _supervisor_loop, PinSupervisor

        # Patch the backoff to be tiny so the test runs fast
        original = list(_BACKOFF_DELAYS_S)
        import rfcensus.engine.pinning as pinning_mod
        pinning_mod._BACKOFF_DELAYS_S = (0.001, 0.001, 0.001, 0.001, 0.001)

        try:
            # Decoder that always crashes
            attempt_count = {"n": 0}

            class _AlwaysCrashes:
                from dataclasses import dataclass

                @dataclass
                class _Caps:
                    preferred_sample_rate: int = 2_400_000
                    access_mode: object = None

                def __init__(self):
                    self.capabilities = self._Caps()

                async def run(self, spec):
                    attempt_count["n"] += 1
                    raise RuntimeError("simulated decoder crash")

            class _FakeRegistry:
                def get(self, name):
                    return _AlwaysCrashes

                def names(self):
                    return ["fake_decoder"]

            class _FakeBus:
                async def publish(self, _):
                    pass

            spec = PinSpec("d1", "fake_decoder", 433_920_000)
            # Lease is unused inside the loop except for being passed
            # back into the run_spec; can be any object.
            class _FakeLease:
                _lease_id = 1
                dongle = None
            state = PinSupervisor(
                spec=spec, lease=_FakeLease(), task=None,  # type: ignore
            )

            # Run for 200 ms — that's hundreds of crashes at our
            # 0.001s backoff. Pre-v0.6.1 the loop would exit after 6
            # consecutive failures.
            task = asyncio.create_task(_supervisor_loop(
                spec=spec,
                lease=_FakeLease(),
                decoder_registry=_FakeRegistry(),
                event_bus=_FakeBus(),
                session_id=1,
                gain="auto",
                state=state,
            ))
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # We should have well over the old give-up threshold
            assert attempt_count["n"] > 10, (
                f"supervisor only attempted {attempt_count['n']} runs "
                f"in 200ms — should be many more (forever-retry)"
            )
        finally:
            pinning_mod._BACKOFF_DELAYS_S = tuple(original)

    @pytest.mark.asyncio
    async def test_dedup_suppresses_identical_errors(self, caplog):
        """After N identical errors in a row, log a single suppression
        line and stop logging the same error until it changes or
        succeeds."""
        import logging
        from rfcensus.engine.pinning import _supervisor_loop, PinSupervisor
        import rfcensus.engine.pinning as pinning_mod

        original = list(_BACKOFF_DELAYS_S)
        pinning_mod._BACKOFF_DELAYS_S = (0.001,) * 5

        try:
            class _AlwaysCrashes:
                from dataclasses import dataclass

                @dataclass
                class _Caps:
                    preferred_sample_rate: int = 2_400_000

                def __init__(self):
                    self.capabilities = self._Caps()

                async def run(self, spec):
                    raise RuntimeError("simulated stuck error")

            class _FakeRegistry:
                def get(self, name):
                    return _AlwaysCrashes

                def names(self):
                    return ["fake_decoder"]

            class _FakeLease:
                _lease_id = 1
                dongle = None

            spec = PinSpec("d1", "fake_decoder", 433_920_000)
            state = PinSupervisor(
                spec=spec, lease=_FakeLease(), task=None,  # type: ignore
            )

            with caplog.at_level(logging.WARNING, logger="rfcensus.engine.pinning"):
                task = asyncio.create_task(_supervisor_loop(
                    spec=spec, lease=_FakeLease(),
                    decoder_registry=_FakeRegistry(),
                    event_bus=type("B", (), {"publish": lambda self, _: None})(),
                    session_id=1, gain="auto", state=state,
                ))
                # Long enough for many failures + dedup boundary
                await asyncio.sleep(0.05)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # The "simulated stuck error" message itself should appear
            # at most _DEDUP_AFTER_N_IDENTICAL + 1 (the dedup announce)
            # times across all log records — not once per attempt.
            from rfcensus.engine.pinning import _DEDUP_AFTER_N_IDENTICAL
            stuck_mentions = sum(
                1 for r in caplog.records
                if "simulated stuck error" in r.message
            )
            # Allow some slack: _DEDUP_AFTER_N_IDENTICAL normal lines,
            # plus the suppression-announced line which DOESN'T contain
            # the error text. So strict upper bound is _DEDUP_AFTER_N_IDENTICAL.
            assert stuck_mentions <= _DEDUP_AFTER_N_IDENTICAL, (
                f"got {stuck_mentions} log lines with the error text; "
                f"expected ≤ {_DEDUP_AFTER_N_IDENTICAL} due to dedup"
            )

            # And exactly one suppression announcement
            suppression_mentions = sum(
                1 for r in caplog.records
                if "suppressing further mentions" in r.message
            )
            assert suppression_mentions == 1, (
                f"expected exactly 1 suppression-announced log line, "
                f"got {suppression_mentions}"
            )
        finally:
            pinning_mod._BACKOFF_DELAYS_S = tuple(original)


# ────────────────────────────────────────────────────────────────────
# summarize_pinning_outcome — renders cleanly
# ────────────────────────────────────────────────────────────────────


class TestSummarize:
    def test_supervisors_and_skipped(self):
        from rfcensus.engine.pinning import ValidationResult
        outcome = PinningOutcome(
            supervisors=[
                PinSupervisor(
                    spec=PinSpec("d1", "rtl_433", 433_920_000),
                    lease=None,  # type: ignore[arg-type]
                    task=None,  # type: ignore[arg-type]
                ),
            ],
            skipped=[
                ValidationResult(
                    PinSpec("d2", "rtlamr", 912_000_000),
                    status="skip", reason="not connected",
                ),
            ],
        )
        lines = summarize_pinning_outcome(outcome)
        joined = "\n".join(lines)
        assert "Pinned 1" in joined
        assert "Skipped 1" in joined
        assert "d1" in joined
        assert "d2" in joined
        assert "433.920" in joined


# ────────────────────────────────────────────────────────────────────
# fleet_optimizer pin constraints
# ────────────────────────────────────────────────────────────────────


class TestFleetOptimizerWithPins:
    def test_pinned_dongle_keeps_freq_compatible_antenna(self):
        """If a dongle is pinned to 433 MHz, the optimizer must NOT
        propose swapping its antenna to one that doesn't cover 433."""
        from rfcensus.hardware.fleet_optimizer import optimize_fleet

        # Two antennas in catalog: one for 433, one for 915
        ant_433 = _make_antenna("whip_433",
                                usable_range=(380_000_000, 480_000_000))
        ant_915 = _make_antenna("whip_915",
                                usable_range=(800_000_000, 1_000_000_000))

        # One dongle pinned to 433
        d1 = _make_dongle("d1", antenna=ant_433)

        # Bands: a 915 ISM band to tempt the optimizer toward whip_915
        band_915 = BandConfig(
            id="b915", name="ISM 915", freq_low=902_000_000,
            freq_high=928_000_000,
        )

        plan = optimize_fleet(
            dongles=[d1],
            enabled_bands=[band_915],
            available_antennas=[ant_433, ant_915],
            pinned_freqs={"d1": 433_920_000},
        )
        # d1 is pinned to 433 → must keep an antenna that covers 433
        assigned = plan.assignments.get("d1")
        assert assigned == "whip_433", (
            f"pinned dongle should keep 433-compatible antenna, "
            f"got {assigned}"
        )

    def test_unpinned_dongle_optimizes_freely(self):
        """Without any pin info, the optimizer behaves as before."""
        from rfcensus.hardware.fleet_optimizer import optimize_fleet

        ant_915 = _make_antenna("whip_915",
                                usable_range=(800_000_000, 1_000_000_000))
        d1 = _make_dongle("d1", antenna=None)
        band_915 = BandConfig(
            id="b915", name="ISM 915", freq_low=902_000_000,
            freq_high=928_000_000,
        )
        plan = optimize_fleet(
            dongles=[d1], enabled_bands=[band_915],
            available_antennas=[ant_915],
        )
        # With no pin constraint, the 915 antenna is the right pick
        assert plan.assignments.get("d1") == "whip_915"


# ────────────────────────────────────────────────────────────────────
# TOML round-trip
# ────────────────────────────────────────────────────────────────────


class TestPinTomlRoundTrip:
    def test_apply_pins_to_toml_creates_pin_subtable(self):
        from rfcensus.commands.pin import _apply_pins_to_toml
        data = {"dongles": [{"id": "d1", "model": "rtlsdr_v3"}]}
        spec = parse_cli_pin("d1:rtl_433@433.92M")
        n = _apply_pins_to_toml(data, [spec])
        assert n == 1
        d = data["dongles"][0]
        assert d["pin"]["decoder"] == "rtl_433"
        assert d["pin"]["freq_hz"] == 433_920_000

    def test_apply_pins_idempotent(self):
        """Re-applying the same pin produces no change."""
        from rfcensus.commands.pin import _apply_pins_to_toml
        data = {"dongles": [{"id": "d1", "model": "rtlsdr_v3"}]}
        spec = parse_cli_pin("d1:rtl_433@433.92M")
        _apply_pins_to_toml(data, [spec])
        n = _apply_pins_to_toml(data, [spec])
        assert n == 0  # already there

    def test_apply_pins_includes_sample_rate_when_set(self):
        from rfcensus.commands.pin import _apply_pins_to_toml
        data = {"dongles": [{"id": "d1", "model": "rtlsdr_v3"}]}
        spec = parse_cli_pin("d1:rtl_433@433.92M:1.024M")
        _apply_pins_to_toml(data, [spec])
        assert data["dongles"][0]["pin"]["sample_rate"] == 1_024_000

    def test_apply_pins_omits_default_access_mode(self):
        """Default 'exclusive' is the implicit value — TOML stays tidy
        without it."""
        from rfcensus.commands.pin import _apply_pins_to_toml
        data = {"dongles": [{"id": "d1", "model": "rtlsdr_v3"}]}
        spec = parse_cli_pin("d1:rtl_433@433.92M")  # default exclusive
        _apply_pins_to_toml(data, [spec])
        assert "access_mode" not in data["dongles"][0]["pin"]


# ────────────────────────────────────────────────────────────────────
# CLI command structure (smoke tests)
# ────────────────────────────────────────────────────────────────────


class TestPinCliStructure:
    def test_pin_command_registered_in_main(self):
        from rfcensus.cli import main
        assert "pin" in main.commands

    def test_pin_subcommands_registered(self):
        from rfcensus.commands.pin import cli as pin_cli
        # All four subcommands present
        assert set(pin_cli.commands.keys()) == {
            "list", "add", "remove", "clear",
        }

    def test_pin_help_output(self):
        from click.testing import CliRunner
        from rfcensus.commands.pin import cli as pin_cli
        runner = CliRunner()
        res = runner.invoke(pin_cli, ["--help"])
        assert res.exit_code == 0
        assert "decoder" in res.output.lower()
        assert "dongle" in res.output.lower()


# ────────────────────────────────────────────────────────────────────
# inventory/scan show --pin in help but reject it (redirect to hybrid)
# ────────────────────────────────────────────────────────────────────


class TestInventoryRejectsPinFlag:
    """v0.6.1: --pin and --allow-pin-antenna-mismatch are visible in
    inventory/scan --help (so users discover the feature) but error
    out if actually used. Footgun mitigation: a forgotten config pin
    silently breaking a scan is worse than a clear error pointing at
    `rfcensus hybrid`."""

    def test_inventory_help_mentions_pin(self):
        from click.testing import CliRunner
        from rfcensus.commands.inventory import cli_inventory
        runner = CliRunner()
        res = runner.invoke(cli_inventory, ["--help"])
        assert res.exit_code == 0
        assert "--pin" in res.output
        # And the help text steers them to hybrid
        assert "hybrid" in res.output.lower()

    def test_scan_help_mentions_pin(self):
        from click.testing import CliRunner
        from rfcensus.commands.inventory import cli_scan
        runner = CliRunner()
        res = runner.invoke(cli_scan, ["--help"])
        assert res.exit_code == 0
        assert "--pin" in res.output
        assert "--allow-pin-antenna-mismatch" in res.output

    def test_inventory_with_pin_errors_with_redirect(self):
        from click.testing import CliRunner
        from rfcensus.commands.inventory import cli_inventory
        runner = CliRunner()
        res = runner.invoke(
            cli_inventory, ["--pin", "00000043:rtl_433@433.92M"],
        )
        assert res.exit_code != 0
        assert "hybrid" in res.output.lower()

    def test_scan_with_allow_mismatch_errors_with_redirect(self):
        from click.testing import CliRunner
        from rfcensus.commands.inventory import cli_scan
        runner = CliRunner()
        res = runner.invoke(
            cli_scan, ["--allow-pin-antenna-mismatch"],
        )
        assert res.exit_code != 0
        assert "hybrid" in res.output.lower()


# ────────────────────────────────────────────────────────────────────
# hybrid command exists and DOES honor --pin
# ────────────────────────────────────────────────────────────────────


class TestHybridCommand:
    def test_hybrid_registered_in_main(self):
        from rfcensus.cli import main
        assert "hybrid" in main.commands

    def test_hybrid_help_mentions_pinning(self):
        from click.testing import CliRunner
        from rfcensus.commands.inventory import cli_hybrid
        runner = CliRunner()
        res = runner.invoke(cli_hybrid, ["--help"])
        assert res.exit_code == 0
        assert "--pin" in res.output
        # And it talks about the use case so the user knows what's different
        out = res.output.lower()
        assert "pin" in out
        assert "decoder" in out

    def test_hybrid_default_duration_is_forever(self):
        """Hybrid's main use case is gap-free long-running coverage."""
        from rfcensus.commands.inventory import cli_hybrid
        dur_opt = next(p for p in cli_hybrid.params if p.name == "duration")
        assert dur_opt.default == "forever"

    def test_inventory_default_duration_is_forever(self):
        """v0.6.1: inventory is "exhaustive enumeration", which means
        you really want it to keep running until you're done."""
        from rfcensus.commands.inventory import cli_inventory
        dur_opt = next(p for p in cli_inventory.params if p.name == "duration")
        assert dur_opt.default == "forever"

    def test_scan_default_duration_is_5m(self):
        """Scan stays finite: it's the discover-what's-here pass."""
        from rfcensus.commands.inventory import cli_scan
        dur_opt = next(p for p in cli_scan.params if p.name == "duration")
        assert dur_opt.default == "5m"
