"""v0.6.5 TUI — Pilot-driven binding tests.

These tests use Textual's Pilot to drive the app with simulated key
presses, then assert on state. They're slower than the pure reducer
tests (each test starts and stops the app) but catch real
integration bugs in key wiring.

We test a handful of high-value flows:
  • `f` cycles filter mode
  • `t` toggles plan-tree visibility
  • Number keys 1-9 + 0 set focus
  • `?` opens help, Esc closes it
  • Quit confirmation flow

We deliberately don't test every binding — that becomes brittle
maintenance work on top of the reducer tests we already have.
"""

from __future__ import annotations

import pytest

from rfcensus.tui.app import TUIApp
from rfcensus.tui.state import TUIState


@pytest.mark.asyncio
class TestTUIPilot:

    async def test_app_starts_and_quits(self):
        """Smoke test: app starts, accepts quit signal, exits."""
        app = TUIApp(runner=None, no_color=True, site_name="test")
        async with app.run_test() as pilot:
            # App is up; no runner, no events, just exits cleanly
            await pilot.pause()
            app.exit()
            await pilot.pause()
        # If we got here without hanging, success

    async def test_f_cycles_filter_mode(self):
        app = TUIApp(runner=None, no_color=True, site_name="test")
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.state.filter_mode == "filtered"
            await pilot.press("f")
            assert app.state.filter_mode == "verbose"
            await pilot.press("f")
            assert app.state.filter_mode == "minimal"
            await pilot.press("f")
            assert app.state.filter_mode == "filtered"
            app.exit()

    async def test_t_toggles_plan_tree(self):
        app = TUIApp(runner=None, no_color=True, site_name="test")
        async with app.run_test() as pilot:
            await pilot.pause()
            initial = app.state.plan_tree_visible
            await pilot.press("t")
            assert app.state.plan_tree_visible != initial
            await pilot.press("t")
            assert app.state.plan_tree_visible == initial
            app.exit()

    async def test_help_opens_and_closes(self):
        app = TUIApp(runner=None, no_color=True, site_name="test")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("question_mark")
            await pilot.pause()
            # Help should be on the screen stack
            assert len(app.screen_stack) >= 2
            await pilot.press("escape")
            await pilot.pause()
            # Back to main
            assert len(app.screen_stack) == 1
            app.exit()

    async def test_focus_navigation_with_arrow_keys(self):
        """Add some dongles and verify ←/→ moves focus."""
        from rfcensus.events import HardwareEvent
        from rfcensus.tui.state import reduce
        app = TUIApp(runner=None, no_color=True, site_name="test")
        # Pre-populate state with dongles
        for did in ["d1", "d2", "d3"]:
            reduce(app.state, HardwareEvent(dongle_id=did, kind="detected"))

        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.state.focused_dongle_index == 0
            await pilot.press("right")
            assert app.state.focused_dongle_index == 1
            await pilot.press("right")
            assert app.state.focused_dongle_index == 2
            # Already at end — should not wrap
            await pilot.press("right")
            assert app.state.focused_dongle_index == 2
            await pilot.press("left")
            assert app.state.focused_dongle_index == 1
            app.exit()

    async def test_number_key_focuses_slot(self):
        """Pressing 2 should set focus to slot 2 (index 1)."""
        from rfcensus.events import HardwareEvent
        from rfcensus.tui.state import reduce
        app = TUIApp(runner=None, no_color=True, site_name="test")
        for did in ["d1", "d2", "d3"]:
            reduce(app.state, HardwareEvent(dongle_id=did, kind="detected"))

        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("2")
            await pilot.pause()
            # Slot 2 → index 1; opens detail screen
            assert app.state.focused_dongle_index == 1
            assert len(app.screen_stack) >= 2  # detail screen pushed
            # Esc back
            await pilot.press("escape")
            await pilot.pause()
            assert len(app.screen_stack) == 1
            app.exit()

    async def test_zero_key_focuses_slot_10(self):
        """0 = slot 10."""
        from rfcensus.events import HardwareEvent
        from rfcensus.tui.state import reduce
        app = TUIApp(runner=None, no_color=True, site_name="test")
        # Need 10 dongles
        for i in range(10):
            reduce(app.state,
                   HardwareEvent(dongle_id=f"d{i}", kind="detected"))

        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("0")
            await pilot.pause()
            assert app.state.focused_dongle_index == 9
            app.exit()
