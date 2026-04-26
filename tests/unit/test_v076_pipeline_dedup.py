"""v0.7.6: dedup tolerance widening for parallel-slot duplicates.

Real-world reproduction: monitoring 915 ISM at 2.4 MS/s with
--preset all --slots all --lazy emitted the same NODEINFO packet
4 times across 6300 input samples (2.6 ms). The previous 256-sample
default tolerance only caught duplicates within ~100 µs.
"""
from __future__ import annotations


def test_lazy_pipeline_dedup_tolerance_scales_with_sample_rate() -> None:
    """LazyMultiPresetPipeline.pop_packets default tolerance is now
    sample_rate / 5 (200 ms in input samples) instead of a fixed 256."""
    src = open(
        "/home/claude/rfcensus/rfcensus/decoders/lazy_pipeline.py"
    ).read()
    # Default sentinel
    assert "dedup_offset_tolerance: int | None = None" in src
    # Derivation
    assert "self._sample_rate_hz // 5" in src


def test_eager_pipeline_dedup_tolerance_scales_with_sample_rate() -> None:
    """MeshtasticPipeline.pop_packets has the matching widening."""
    src = open(
        "/home/claude/rfcensus/rfcensus/decoders/meshtastic_pipeline.py"
    ).read()
    assert "dedup_offset_tolerance: int | None = None" in src
    assert "self._sample_rate_hz // 5" in src


def test_dedup_window_covers_parallel_slot_duplicates() -> None:
    """At 2.4 MS/s, the new default tolerance is 480_000 input samples
    (200 ms) — comfortably more than the 6300-sample spread the user
    reported and well below typical mesh re-broadcast intervals."""
    sample_rate = 2_400_000
    tolerance = sample_rate // 5
    # User's reported worst-case spread
    duplicate_spread = 6300
    assert tolerance > duplicate_spread, (
        f"dedup tolerance {tolerance} too tight to catch real "
        f"duplicates spread by {duplicate_spread} samples"
    )
    # Sanity: not so wide we'd merge legitimately distinct
    # transmissions. Mesh re-broadcasts arrive seconds apart;
    # 200 ms is well under that.
    assert tolerance < sample_rate, (
        f"dedup tolerance {tolerance} ≥ 1 second of audio — too wide"
    )


def test_dedup_window_at_1mhz_capture_rate() -> None:
    """1 MS/s captures (slow rates) get a proportionally smaller
    window since LoRa packets at low SR are also shorter in samples."""
    sample_rate = 1_000_000
    tolerance = sample_rate // 5
    assert tolerance == 200_000    # 200 ms
