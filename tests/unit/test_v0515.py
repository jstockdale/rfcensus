"""Tests for v0.5.15 — shared-slot frequency tracking.

rtl_tcp serves a SINGLE tuned frequency to all connected clients.
Previously the broker let decoders join a shared slot based only on
sample-rate compatibility, so e.g. a 915 MHz ISM decoder and a 929 MHz
POCSAG decoder could both be granted shared leases on the same dongle
and then fight over the tuning. This is wrong — they can only share
if both frequencies fall within the dongle's instantaneous bandwidth
(~2.4 MHz for the typical sample rate).

This module tests:
  • Incompatible-freq shared joiners are rejected
  • Compatible-freq joiners (e.g. multiple Meshtastic presets within
    a 2 MHz window) share the same rtl_tcp slot
  • rtl_tcp is started with -f <freq> to lock the initial tuning
  • The center_freq_hz is tracked on the slot dataclass
"""

from __future__ import annotations

import pytest


def _make_dongle(idx, antenna=None):
    from rfcensus.hardware.dongle import (
        Dongle, DongleCapabilities, DongleStatus,
    )
    caps = DongleCapabilities(
        freq_range_hz=(24_000_000, 1_700_000_000),
        max_sample_rate=2_400_000, bits_per_sample=8,
        bias_tee_capable=False, tcxo_ppm=10.0,
        can_share_via_rtl_tcp=True,
    )
    d = Dongle(
        id=f"rtl-{idx}", serial=f"S{idx}", model="rtlsdr_generic",
        driver="rtlsdr", capabilities=caps, status=DongleStatus.HEALTHY,
        driver_index=idx,
    )
    d.antenna = antenna
    return d


def _make_whip_915():
    from rfcensus.hardware.antenna import Antenna
    return Antenna(
        id="whip_915", name="whip_915", antenna_type="whip",
        resonant_freq_hz=915_000_000,
        usable_range=(594_000_000, 1_235_000_000),
        gain_dbi=2.15, polarization="vertical",
        requires_bias_power=False, notes="",
    )


# ──────────────────────────────────────────────────────────────────
# Shared slot compatibility predicate
# ──────────────────────────────────────────────────────────────────


class TestSharedSlotCompatible:
    def test_same_freq_compatible(self):
        from rfcensus.hardware.broker import _SharedSlot, _shared_slot_compatible
        slot = _SharedSlot(
            process=None, host="127.0.0.1", port=1234,
            sample_rate=2_400_000, center_freq_hz=915_000_000,
        )
        assert _shared_slot_compatible(slot, 915_000_000, 2_400_000) is True

    def test_close_freq_compatible(self):
        """Meshtastic preset channels within 2 MHz of each other should
        be able to share a single rtl_tcp slot."""
        from rfcensus.hardware.broker import _SharedSlot, _shared_slot_compatible
        slot = _SharedSlot(
            process=None, host="127.0.0.1", port=1234,
            sample_rate=2_400_000, center_freq_hz=915_000_000,
        )
        # ±500 kHz inside the 2.4 MHz window
        assert _shared_slot_compatible(slot, 915_500_000, 2_400_000) is True
        assert _shared_slot_compatible(slot, 914_500_000, 2_400_000) is True

    def test_distant_freq_incompatible(self):
        """915 MHz ISM and 929 MHz POCSAG can NOT share — 14 MHz apart,
        way outside the 2.4 MHz instantaneous bandwidth."""
        from rfcensus.hardware.broker import _SharedSlot, _shared_slot_compatible
        slot = _SharedSlot(
            process=None, host="127.0.0.1", port=1234,
            sample_rate=2_400_000, center_freq_hz=915_000_000,
        )
        assert _shared_slot_compatible(slot, 929_000_000, 2_400_000) is False

    def test_edge_freq_outside_80pct_window(self):
        """The usable window is 0.8 × sample_rate / 2 = ±960 kHz at 2.4 SR.
        Requests at the band edge (1.2 MHz from center) are outside."""
        from rfcensus.hardware.broker import _SharedSlot, _shared_slot_compatible
        slot = _SharedSlot(
            process=None, host="127.0.0.1", port=1234,
            sample_rate=2_400_000, center_freq_hz=915_000_000,
        )
        # 1.2 MHz away — theoretically in bandwidth but at filter rolloff
        assert _shared_slot_compatible(slot, 916_200_000, 2_400_000) is False

    def test_sample_rate_mismatch_incompatible(self):
        """If slot's SR < request SR, joining would give less resolution
        than the decoder asked for."""
        from rfcensus.hardware.broker import _SharedSlot, _shared_slot_compatible
        slot = _SharedSlot(
            process=None, host="127.0.0.1", port=1234,
            sample_rate=1_024_000, center_freq_hz=915_000_000,
        )
        assert _shared_slot_compatible(slot, 915_000_000, 2_400_000) is False


# ──────────────────────────────────────────────────────────────────
# Broker integration — find_candidates respects freq compat
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestBrokerFreqCompat:
    async def test_rejects_incompatible_joiner(self):
        """First consumer allocates shared at 915 MHz. Second consumer
        wants shared at 929 MHz. Must NOT be allowed to join the slot."""
        from rfcensus.events import EventBus
        from rfcensus.hardware.broker import (
            AccessMode, DongleBroker, DongleRequirements, NoDongleAvailable,
        )
        from rfcensus.hardware.registry import HardwareRegistry

        dongle = _make_dongle(0, antenna=_make_whip_915())
        broker = DongleBroker(HardwareRegistry(dongles=[dongle]), EventBus())

        # Manually inject a pre-existing shared slot at 915 MHz so we
        # don't have to actually start rtl_tcp in a unit test
        from rfcensus.hardware.broker import _SharedSlot
        broker._shared_slots[dongle.id] = _SharedSlot(
            process=None, host="127.0.0.1", port=1234,
            sample_rate=2_400_000, center_freq_hz=915_000_000,
            lease_ids={999},
        )

        # Now request shared at 929 MHz — should fail even though the
        # dongle antenna covers it and the slot has capacity
        req = DongleRequirements(
            freq_hz=929_000_000,
            sample_rate=2_400_000,
            access_mode=AccessMode.SHARED,
            require_suitable_antenna=True,
        )
        with pytest.raises(NoDongleAvailable):
            await broker.allocate(req, consumer="pocsag_test", timeout=0.3)

    async def test_allows_compatible_joiner(self):
        """Decoder at 915.5 MHz should join an existing 915 MHz slot."""
        from rfcensus.events import EventBus
        from rfcensus.hardware.broker import (
            AccessMode, DongleBroker, DongleRequirements,
        )
        from rfcensus.hardware.registry import HardwareRegistry

        dongle = _make_dongle(0, antenna=_make_whip_915())
        broker = DongleBroker(HardwareRegistry(dongles=[dongle]), EventBus())

        from rfcensus.hardware.broker import _SharedSlot
        broker._shared_slots[dongle.id] = _SharedSlot(
            process=None, host="127.0.0.1", port=1234,
            sample_rate=2_400_000, center_freq_hz=915_000_000,
            lease_ids={999},
        )

        req = DongleRequirements(
            freq_hz=915_500_000,  # 500 kHz from slot center, well within window
            sample_rate=2_400_000,
            access_mode=AccessMode.SHARED,
            require_suitable_antenna=True,
        )
        lease = await broker.allocate(req, consumer="close_test", timeout=1.0)
        # Lease should point to the existing slot's host:port, not a new one
        assert lease.rtl_tcp_port == 1234
        await broker.release(lease)


# ──────────────────────────────────────────────────────────────────
# Slot creation records center_freq_hz and passes -f to rtl_tcp
# ──────────────────────────────────────────────────────────────────


class TestSlotCreationFreqTracking:
    def test_slot_dataclass_has_center_freq_hz(self):
        """Regression: this field was added in v0.5.15. Without it
        the slot can't enforce frequency compatibility."""
        import dataclasses
        from rfcensus.hardware.broker import _SharedSlot
        fields = {f.name for f in dataclasses.fields(_SharedSlot)}
        assert "center_freq_hz" in fields

    def test_start_shared_slot_accepts_center_freq_kwarg(self):
        """Regression: _start_shared_slot signature changed in v0.5.15."""
        import inspect
        from rfcensus.hardware.broker import DongleBroker
        sig = inspect.signature(DongleBroker._start_shared_slot)
        assert "center_freq_hz" in sig.parameters, (
            f"_start_shared_slot should accept center_freq_hz; "
            f"got params: {list(sig.parameters)}"
        )
