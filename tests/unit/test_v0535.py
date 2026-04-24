"""v0.5.35 regression tests: plan-time splitting of multi-exclusive
decoder conflicts across waves.

The failure mode this fixes
===========================

When a band's `suggested_decoders` contains multiple exclusive-access
decoders (direwolf + multimon on aprs_2m, for example) and the fleet's
antenna-suitable dongle pool for that band can't host them all in
parallel, the v0.5.34 behavior was:

  1. Scheduler places the band in a single wave, reserving the primary
     dongle.
  2. At runtime, strategy fans out one asyncio task per decoder.
  3. The first decoder to call broker.allocate() wins the exclusive
     lease; remaining exclusive decoders fail with NoDongleAvailable.
  4. The loser logs a warning ("no dongle for multimon@aprs_2m...")
     and silently does nothing.

From a user's perspective: aprs_2m decoded APRS fine (direwolf won),
but multimon-catchable signals (say if the band also has paging or
some other mode multimon handles) were never decoded and the user had
no clear way to notice.

v0.5.35 fix
===========

Scheduler.plan() now runs `_defer_surplus_exclusive_decoders` as a
post-packing pass. For each task:

  • Classify its band's matched decoders as exclusive vs shared.
  • If there are more exclusives than the wave can host (capacity =
    count of antenna-suitable dongles free in this wave), keep the
    first N (by suggested_decoders order) via `allowed_decoders` and
    defer the surplus to later waves with spare capacity.
  • "Spare capacity" = count of suitable dongles not already reserved
    by other tasks in the target wave. Ties broken by earliest wave.
  • If NO later wave has a suitable dongle free, append a new final
    wave for the deferred decoder.

Shared decoders always ride with the primary task: they cost nothing
in exclusive-dongle terms and the band's shared protocols only need
decoding once.

This test module asserts the mechanics.
"""

from __future__ import annotations

from rfcensus.config.schema import BandConfig, StrategyKind
from rfcensus.engine.scheduler import ScheduleTask


class TestScheduleTaskHasAllowedDecoders:
    """ScheduleTask carries the restriction used by the splitter."""

    def test_allowed_decoders_defaults_to_none(self):
        """None = 'run every decoder matched by band.suggested_decoders'.
        Non-None = 'only run exactly these.' The default preserves
        pre-v0.5.35 behavior for tasks built by code that doesn't
        know about the splitter."""
        task = ScheduleTask(
            band=BandConfig(
                id="x",
                name="x",
                freq_low=100_000_000,
                freq_high=101_000_000,
            ),
            suggested_dongle_id=None,
            suggested_antenna_id=None,
        )
        assert task.allowed_decoders is None

    def test_allowed_decoders_accepts_set(self):
        task = ScheduleTask(
            band=BandConfig(
                id="x",
                name="x",
                freq_low=100_000_000,
                freq_high=101_000_000,
            ),
            suggested_dongle_id=None,
            suggested_antenna_id=None,
            allowed_decoders={"direwolf", "multimon"},
        )
        assert task.allowed_decoders == {"direwolf", "multimon"}


class TestAprs2mIsNoLongerSplitInV0537:
    """v0.5.37 regression test.

    v0.5.35 shipped the splitter specifically because aprs_2m's two
    suggested decoders (direwolf + multimon) were both
    `requires_exclusive_dongle=True` and could not coexist on the
    fleet's single 2m-capable dongle. v0.5.37 moved both to shared
    access via fm_bridge + rtl_tcp, so they now coexist on one task
    and the splitter has nothing to do for this band.

    This test pins the new behavior so we don't accidentally revert
    to exclusive access without also updating v0.5.19 / v0.5.35 tests
    that depend on the wave count.
    """

    def _make_plan(self):
        """Same fleet + config as the old v0.5.35 aprs_2m splitter
        tests. Uses the real decoder registry (multimon/direwolf are
        shared in v0.5.37)."""
        from rfcensus.config.loader import load_config
        from rfcensus.decoders.registry import get_registry
        from rfcensus.engine.scheduler import Scheduler
        from rfcensus.events import EventBus
        from rfcensus.hardware.antenna import Antenna
        from rfcensus.hardware.broker import DongleBroker
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
        )
        from rfcensus.hardware.registry import HardwareRegistry

        def ant(aid, mhz, low, high):
            return Antenna(
                id=aid, name=aid, antenna_type="whip",
                resonant_freq_hz=int(mhz * 1e6),
                usable_range=(int(low * 1e6), int(high * 1e6)),
                gain_dbi=2.15, polarization="vertical",
                requires_bias_power=False, notes="",
            )

        def d(serial, a, model="rtlsdr_generic"):
            caps = DongleCapabilities(
                freq_range_hz=(24_000_000, 1_700_000_000),
                max_sample_rate=2_400_000, bits_per_sample=8,
                bias_tee_capable=False, tcxo_ppm=10.0,
                can_share_via_rtl_tcp=True,
            )
            dd = Dongle(
                id=f"rtlsdr-{serial}", serial=serial, model=model,
                driver="rtlsdr", capabilities=caps,
                status=DongleStatus.HEALTHY, driver_index=0,
            )
            dd.antenna = a
            return dd

        dongles = [
            d("00000003", ant("whip_915", 915, 594, 1235)),
            d("00000002", ant("whip_433", 433, 282, 586)),
            d("00000001", ant("whip_915", 915, 594, 1235), model="rtlsdr_v4"),
            d("00000043", ant("whip_315", 315, 204, 425)),
            d("07262454", ant("marine_vhf", 156.8, 137, 180),
              model="nesdr_smart_v5"),
        ]
        config = load_config()
        broker = DongleBroker(HardwareRegistry(dongles=dongles), EventBus())
        return Scheduler(
            config, broker, decoder_registry=get_registry(),
        ).plan(config.enabled_bands())

    def test_aprs_2m_has_single_unrestricted_task(self):
        """Single task, no allowed_decoders restriction — both
        direwolf and multimon ride together via shared rtl_tcp."""
        plan = self._make_plan()
        aprs_tasks = [t for t in plan.tasks if t.band.id == "aprs_2m"]
        assert len(aprs_tasks) == 1, (
            f"v0.5.37: aprs_2m should produce exactly 1 task now that "
            f"multimon and direwolf are both shared. Got "
            f"{len(aprs_tasks)} tasks. If this fails with >1 task, "
            f"likely someone reverted multimon.py or direwolf.py to "
            f"requires_exclusive_dongle=True — check capabilities."
        )
        task = aprs_tasks[0]
        assert task.allowed_decoders is None, (
            f"v0.5.37: aprs_2m task should allow all decoders (no "
            f"splitter restriction); got allowed_decoders="
            f"{task.allowed_decoders}"
        )


# ------------------------------------------------------------------
# Splitter-mechanics tests with synthetic exclusive decoders
# ------------------------------------------------------------------
#
# The splitter code in Scheduler._defer_surplus_exclusive_decoders is
# unchanged by v0.5.37 — it still exists, still runs, and still needs
# coverage. Since no real band triggers it after v0.5.37 (only rtl_ais
# remains exclusive and AIS is alone on its band), we exercise it with
# a custom DecoderRegistry that forces multimon + direwolf back into
# exclusive mode for test purposes only. This verifies the splitter
# algorithm end-to-end without relying on any particular real-world
# band configuration remaining in conflict.


def _build_exclusive_decoder_registry():
    """Build a fresh DecoderRegistry containing the full real decoder
    set BUT with multimon + direwolf overridden to be exclusive.

    Uses the real decoder classes for everything else so band → decoder
    matching works exactly like production. Only the two specific
    decoders we want to conflict are modified, via
    `dataclasses.replace()` on their frozen capabilities.
    """
    from dataclasses import replace

    from rfcensus.decoders.builtin.direwolf import DirewolfDecoder
    from rfcensus.decoders.builtin.multimon import MultimonDecoder
    from rfcensus.decoders.registry import DecoderRegistry, get_registry

    class _ExclusiveMultimon(MultimonDecoder):
        capabilities = replace(
            MultimonDecoder.capabilities, requires_exclusive_dongle=True
        )

    class _ExclusiveDirewolf(DirewolfDecoder):
        capabilities = replace(
            DirewolfDecoder.capabilities, requires_exclusive_dongle=True
        )

    # Clone the real registry's contents into a fresh one, swapping
    # out multimon and direwolf. We use the real class registrations
    # for every other decoder so bands match correctly.
    real = get_registry()
    test_reg = DecoderRegistry()
    for name in real.names():
        cls = real.get(name)
        if cls is None:
            continue
        if name == "multimon":
            test_reg.register(_ExclusiveMultimon)
        elif name == "direwolf":
            test_reg.register(_ExclusiveDirewolf)
        else:
            test_reg.register(cls)
    return test_reg


class TestSplitterMechanicsWithExclusiveDecoders:
    """Exercise the splitter algorithm by forcing multimon + direwolf
    back into exclusive access (via a custom registry). This keeps the
    splitter test coverage strong even though the real aprs_2m no
    longer triggers it in v0.5.37.
    """

    def _make_plan(self):
        from rfcensus.config.loader import load_config
        from rfcensus.engine.scheduler import Scheduler
        from rfcensus.events import EventBus
        from rfcensus.hardware.antenna import Antenna
        from rfcensus.hardware.broker import DongleBroker
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
        )
        from rfcensus.hardware.registry import HardwareRegistry

        def ant(aid, mhz, low, high):
            return Antenna(
                id=aid, name=aid, antenna_type="whip",
                resonant_freq_hz=int(mhz * 1e6),
                usable_range=(int(low * 1e6), int(high * 1e6)),
                gain_dbi=2.15, polarization="vertical",
                requires_bias_power=False, notes="",
            )

        def d(serial, a, model="rtlsdr_generic"):
            caps = DongleCapabilities(
                freq_range_hz=(24_000_000, 1_700_000_000),
                max_sample_rate=2_400_000, bits_per_sample=8,
                bias_tee_capable=False, tcxo_ppm=10.0,
                can_share_via_rtl_tcp=True,
            )
            dd = Dongle(
                id=f"rtlsdr-{serial}", serial=serial, model=model,
                driver="rtlsdr", capabilities=caps,
                status=DongleStatus.HEALTHY, driver_index=0,
            )
            dd.antenna = a
            return dd

        dongles = [
            d("00000003", ant("whip_915", 915, 594, 1235)),
            d("00000002", ant("whip_433", 433, 282, 586)),
            d("00000001", ant("whip_915", 915, 594, 1235), model="rtlsdr_v4"),
            d("00000043", ant("whip_315", 315, 204, 425)),
            d("07262454", ant("marine_vhf", 156.8, 137, 180),
              model="nesdr_smart_v5"),
        ]
        config = load_config()
        broker = DongleBroker(HardwareRegistry(dongles=dongles), EventBus())
        test_registry = _build_exclusive_decoder_registry()
        return Scheduler(
            config, broker, decoder_registry=test_registry,
        ).plan(config.enabled_bands())

    def test_aprs_2m_gets_split_into_two_tasks(self):
        plan = self._make_plan()
        aprs_tasks = [t for t in plan.tasks if t.band.id == "aprs_2m"]
        assert len(aprs_tasks) == 2, (
            f"with forced-exclusive multimon+direwolf, aprs_2m should "
            f"split into 2 tasks; got {len(aprs_tasks)}"
        )

    def test_primary_has_direwolf_retry_has_multimon(self):
        """Order follows suggested_decoders: direwolf first → kept
        in the primary slot; multimon → deferred."""
        plan = self._make_plan()
        aprs_tasks = [t for t in plan.tasks if t.band.id == "aprs_2m"]

        primary = next(
            (t for t in aprs_tasks if t.allowed_decoders == {"direwolf"}),
            None,
        )
        retry = next(
            (t for t in aprs_tasks if t.allowed_decoders == {"multimon"}),
            None,
        )
        assert primary is not None, (
            f"no primary aprs_2m task with allowed_decoders={{direwolf}}; "
            f"got: {[t.allowed_decoders for t in aprs_tasks]}"
        )
        assert retry is not None, (
            f"no retry aprs_2m task with allowed_decoders={{multimon}}; "
            f"got: {[t.allowed_decoders for t in aprs_tasks]}"
        )

    def test_retry_task_comes_in_later_wave_than_primary(self):
        plan = self._make_plan()
        primary_wave = None
        retry_wave = None
        for wave in plan.waves:
            for task in wave.tasks:
                if task.band.id != "aprs_2m":
                    continue
                if task.allowed_decoders == {"direwolf"}:
                    primary_wave = wave.index
                elif task.allowed_decoders == {"multimon"}:
                    retry_wave = wave.index
        assert primary_wave is not None
        assert retry_wave is not None
        assert retry_wave > primary_wave, (
            f"retry (wave {retry_wave}) must come AFTER primary "
            f"(wave {primary_wave}); otherwise splitter didn't solve "
            f"the conflict."
        )

    def test_retry_task_uses_same_dongle_as_primary(self):
        plan = self._make_plan()
        aprs_tasks = [t for t in plan.tasks if t.band.id == "aprs_2m"]
        dongles = {t.suggested_dongle_id for t in aprs_tasks}
        assert dongles == {"rtlsdr-07262454"}, (
            f"both aprs_2m tasks should target rtlsdr-07262454; "
            f"got {dongles}"
        )

    def test_retry_task_is_marked_as_deferred_in_notes(self):
        plan = self._make_plan()
        retry = next(
            (t for t in plan.tasks
             if t.band.id == "aprs_2m"
             and t.allowed_decoders == {"multimon"}),
            None,
        )
        assert retry is not None
        assert any(
            "plan retry" in note.lower() for note in retry.notes
        ), (
            f"retry task should have a 'plan retry' note; got "
            f"notes={retry.notes}"
        )


class TestSplitterIsNoopWhenNoConflict:
    """A band with zero or one exclusive decoder needs no splitting;
    the splitter must not touch it."""

    def test_single_exclusive_decoder_band_unchanged(self):
        """Most bands have exactly one exclusive decoder. They should
        pass through the splitter untouched."""
        from rfcensus.config.loader import load_config
        from rfcensus.decoders.registry import get_registry
        from rfcensus.engine.scheduler import Scheduler
        from rfcensus.events import EventBus
        from rfcensus.hardware.antenna import Antenna
        from rfcensus.hardware.broker import DongleBroker
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
        )
        from rfcensus.hardware.registry import HardwareRegistry

        # Minimal fleet — one dongle that can hit everything
        caps = DongleCapabilities(
            freq_range_hz=(24_000_000, 1_700_000_000),
            max_sample_rate=2_400_000,
            bits_per_sample=8,
            bias_tee_capable=False,
            tcxo_ppm=10.0,
            can_share_via_rtl_tcp=True,
        )
        ant = Antenna(
            id="omni",
            name="omni",
            antenna_type="discone",
            resonant_freq_hz=400_000_000,
            usable_range=(25_000_000, 1_500_000_000),
            gain_dbi=2.15,
            polarization="vertical",
            requires_bias_power=False,
            notes="",
        )
        dongle = Dongle(
            id="rtlsdr-001",
            serial="001",
            model="rtlsdr_generic",
            driver="rtlsdr",
            capabilities=caps,
            status=DongleStatus.HEALTHY,
            driver_index=0,
        )
        dongle.antenna = ant

        config = load_config()
        broker = DongleBroker(
            HardwareRegistry(dongles=[dongle]), EventBus()
        )
        plan = Scheduler(
            config, broker, decoder_registry=get_registry(),
        ).plan(config.enabled_bands())

        # For every band that is NOT aprs_2m, allowed_decoders should
        # still be None (no restriction). aprs_2m is the only known
        # split case today.
        for task in plan.tasks:
            if task.band.id == "aprs_2m":
                continue
            assert task.allowed_decoders is None, (
                f"unexpected allowed_decoders on non-conflicted task "
                f"{task.band.id}: {task.allowed_decoders}. Splitter "
                f"should only restrict tasks with a genuine "
                f"multi-exclusive conflict."
            )


class TestPickDecodersHonorsAllowedDecoders:
    """The strategy layer is what actually makes `allowed_decoders`
    effective. If `_pick_decoders` ignores it, the scheduler's split
    does nothing at runtime."""

    def test_pick_decoders_default_returns_all_matched(self):
        """Baseline: no allowed_decoders = same set as pre-v0.5.35."""
        import inspect

        from rfcensus.engine import strategy
        src = inspect.getsource(strategy._pick_decoders)
        # The signature must have a keyword-only allowed_decoders
        assert "allowed_decoders" in src, (
            "_pick_decoders must accept allowed_decoders kwarg so the "
            "v0.5.35 splitter can restrict decoder sets per task."
        )

    def test_pick_decoders_filters_when_restricted(self):
        """_pick_decoders must skip decoders not in allowed_decoders."""
        import inspect

        from rfcensus.engine import strategy
        src = inspect.getsource(strategy._pick_decoders)
        assert "allowed_decoders is not None and name not in allowed_decoders" in src, (
            "_pick_decoders must explicitly filter by allowed_decoders "
            "when the restriction is provided."
        )


class TestStrategyExecuteAcceptsAllowedDecoders:
    """session.py calls strategy.execute(band, ctx, allowed_decoders=...)
    for every task. All strategy subclasses must accept the kwarg."""

    def test_all_strategies_accept_allowed_decoders(self):
        import inspect

        from rfcensus.engine.strategy import (
            DecoderOnlyStrategy,
            DecoderPrimaryStrategy,
            ExplorationStrategy,
            PowerPrimaryStrategy,
        )

        for cls in [
            DecoderOnlyStrategy,
            DecoderPrimaryStrategy,
            PowerPrimaryStrategy,
            ExplorationStrategy,
        ]:
            sig = inspect.signature(cls.execute)
            assert "allowed_decoders" in sig.parameters, (
                f"{cls.__name__}.execute must accept allowed_decoders "
                f"keyword arg; got signature {sig}"
            )
            param = sig.parameters["allowed_decoders"]
            # Must be keyword-only so positional callers don't silently
            # pass bands as allowed_decoders or vice versa
            assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"{cls.__name__}.execute.allowed_decoders must be "
                f"keyword-only; got {param.kind}"
            )

    def test_session_passes_allowed_decoders_to_strategy(self):
        """session.py must thread task.allowed_decoders into the
        strategy.execute call, otherwise the plan-time split has no
        runtime effect."""
        import inspect

        from rfcensus.engine import session
        src = inspect.getsource(session)
        # Grep for the pattern; brittle but catches obvious regressions.
        assert "allowed_decoders=task.allowed_decoders" in src, (
            "session.py must pass task.allowed_decoders into "
            "strategy.execute. Without this, the v0.5.35 splitter's "
            "allowed_decoders restriction has no runtime effect — "
            "both decoders still race for the dongle."
        )


class TestNewWaveAppendedWhenNoCapacity:
    """When no later wave has spare capacity for the deferred decoder,
    the scheduler must append a new final wave rather than silently
    dropping the decoder.

    Uses the forced-exclusive multimon/direwolf registry (see
    `_build_exclusive_decoder_registry` above) since v0.5.37 moved
    those to shared access in production — the splitter mechanics
    still exist and still need coverage.
    """

    def test_aprs_2m_case_appends_new_wave(self):
        """In the user's real fleet, waves 0-4 all use the 2m dongle.
        With multimon + direwolf forced exclusive, the multimon retry
        has nowhere to slot in and must end up in a brand-new final
        wave rather than being dropped."""
        from rfcensus.config.loader import load_config
        from rfcensus.engine.scheduler import Scheduler
        from rfcensus.events import EventBus
        from rfcensus.hardware.antenna import Antenna
        from rfcensus.hardware.broker import DongleBroker
        from rfcensus.hardware.dongle import (
            Dongle, DongleCapabilities, DongleStatus,
        )
        from rfcensus.hardware.registry import HardwareRegistry

        def ant(aid, mhz, low, high):
            return Antenna(
                id=aid, name=aid, antenna_type="whip",
                resonant_freq_hz=int(mhz * 1e6),
                usable_range=(int(low * 1e6), int(high * 1e6)),
                gain_dbi=2.15, polarization="vertical",
                requires_bias_power=False, notes="",
            )

        def d(serial, a, model="rtlsdr_generic"):
            caps = DongleCapabilities(
                freq_range_hz=(24_000_000, 1_700_000_000),
                max_sample_rate=2_400_000, bits_per_sample=8,
                bias_tee_capable=False, tcxo_ppm=10.0,
                can_share_via_rtl_tcp=True,
            )
            dd = Dongle(
                id=f"rtlsdr-{serial}", serial=serial, model=model,
                driver="rtlsdr", capabilities=caps,
                status=DongleStatus.HEALTHY, driver_index=0,
            )
            dd.antenna = a
            return dd

        dongles = [
            d("00000003", ant("whip_915", 915, 594, 1235)),
            d("00000002", ant("whip_433", 433, 282, 586)),
            d("00000001", ant("whip_915", 915, 594, 1235), model="rtlsdr_v4"),
            d("00000043", ant("whip_315", 315, 204, 425)),
            d("07262454", ant("marine_vhf", 156.8, 137, 180),
              model="nesdr_smart_v5"),
        ]

        config = load_config()
        broker = DongleBroker(HardwareRegistry(dongles=dongles), EventBus())
        # Forced-exclusive registry so the aprs_2m conflict exists for
        # the splitter to resolve. In production v0.5.37 this conflict
        # doesn't arise.
        test_registry = _build_exclusive_decoder_registry()
        plan = Scheduler(
            config, broker, decoder_registry=test_registry,
        ).plan(config.enabled_bands())

        # The last wave should contain only the multimon retry task.
        last_wave = plan.waves[-1]
        assert len(last_wave.tasks) == 1, (
            f"last wave should be the appended multimon-retry wave "
            f"with 1 task; got {len(last_wave.tasks)} tasks: "
            f"{[t.band.id for t in last_wave.tasks]}"
        )
        t = last_wave.tasks[0]
        assert t.band.id == "aprs_2m"
        assert t.allowed_decoders == {"multimon"}


class TestSplitterRespectsSuggestedDecodersOrder:
    """When splitting, 'first listed = kept, later = deferred' must
    follow band.suggested_decoders ordering. Otherwise swapping the
    decoder list order in TOML would silently change runtime behavior
    in surprising ways."""

    def test_aprs_2m_lists_direwolf_first(self):
        """Sanity check: confirm the band config actually lists
        direwolf before multimon. If this fails the test below's
        assumption is wrong."""
        from rfcensus.config.loader import load_config
        config = load_config()
        aprs = next(
            (b for b in config.enabled_bands() if b.id == "aprs_2m"),
            None,
        )
        assert aprs is not None
        # Index of direwolf must be less than index of multimon
        decs = aprs.suggested_decoders
        assert decs.index("direwolf") < decs.index("multimon"), (
            f"test fixture assumption: aprs_2m.suggested_decoders "
            f"should list direwolf before multimon; got {decs}"
        )
