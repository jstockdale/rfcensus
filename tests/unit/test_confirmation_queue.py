"""v0.5.41 tests: ConfirmationQueue + BatchedConfirmationTask.

Pure-logic tests for the queue itself — no event bus, no dongles. The
queue's job is:

  • Accept ConfirmationTasks; dedup by (rounded_freq, bw).
  • Greedy-cluster tasks so each cluster fits within
    cluster_coverage_hz (2 MHz default).
  • Expose pending/in-flight/completed counts so the session can
    decide whether to prompt for an extra wave.
"""

from __future__ import annotations

import pytest

from rfcensus.engine.confirmation_queue import (
    BatchedConfirmationTask,
    ConfirmationQueue,
    ConfirmationTask,
    DEFAULT_CLUSTER_COVERAGE_HZ,
)


def _task(
    detection_id: int,
    freq_mhz: float,
    bandwidth_khz: int = 250,
    technology: str = "lora",
) -> ConfirmationTask:
    return ConfirmationTask(
        detection_id=detection_id,
        freq_hz=int(freq_mhz * 1_000_000),
        bandwidth_hz=bandwidth_khz * 1000,
        technology=technology,
        detector_name="lora",
    )


# ------------------------------------------------------------
# Dedup semantics
# ------------------------------------------------------------


class TestDedup:
    @pytest.mark.asyncio
    async def test_identical_submits_dedupe(self):
        """Two tasks at the exact same freq/bw collapse to one queue entry."""
        q = ConfirmationQueue()
        r1 = await q.submit(_task(1, 906.875, 250))
        r2 = await q.submit(_task(2, 906.875, 250))
        assert r1 is True
        assert r2 is False  # deduped
        assert q.pending_count() == 1

    @pytest.mark.asyncio
    async def test_small_freq_wobble_dedupes(self):
        """Transmitters detected at slightly-drifting center freqs
        (e.g., 906.875 vs 906.880 MHz due to bin alignment) should
        dedupe — same bucket."""
        q = ConfirmationQueue()
        await q.submit(_task(1, 906.875, 250))
        res = await q.submit(_task(2, 906.880, 250))
        assert res is False  # within 50 kHz bucket → same key

    @pytest.mark.asyncio
    async def test_different_bandwidths_are_distinct(self):
        """Same freq, different bandwidth = distinct transmitters."""
        q = ConfirmationQueue()
        await q.submit(_task(1, 906.875, 125))
        res = await q.submit(_task(2, 906.875, 250))
        assert res is True  # distinct
        assert q.pending_count() == 2

    @pytest.mark.asyncio
    async def test_large_freq_gap_is_distinct(self):
        """Two clearly-different freqs submit as distinct tasks."""
        q = ConfirmationQueue()
        await q.submit(_task(1, 906.875, 250))
        await q.submit(_task(2, 912.500, 250))
        assert q.pending_count() == 2


# ------------------------------------------------------------
# Clustering
# ------------------------------------------------------------


class TestClustering:
    def test_empty_queue_yields_no_clusters(self):
        q = ConfirmationQueue()
        assert q.cluster_for_capture([]) == []

    def test_single_task_is_one_cluster(self):
        task = _task(1, 906.875, 250)
        q = ConfirmationQueue()
        clusters = q.cluster_for_capture([task])
        assert len(clusters) == 1
        assert clusters[0].size == 1
        assert clusters[0].center_freq_hz == 906_875_000

    def test_nearby_tasks_cluster_together(self):
        """Two tasks 100 kHz apart: cleanly fit in one 2 MHz window."""
        tasks = [_task(1, 912.502, 250), _task(2, 912.603, 250)]
        q = ConfirmationQueue()
        clusters = q.cluster_for_capture(tasks)
        assert len(clusters) == 1
        cluster = clusters[0]
        assert cluster.size == 2
        # Center between the two
        assert 912_500_000 <= cluster.center_freq_hz <= 912_610_000

    def test_distant_tasks_form_separate_clusters(self):
        """Tasks > 2 MHz apart need separate captures."""
        tasks = [
            _task(1, 903.000, 250),
            _task(2, 910.000, 250),
            _task(3, 925.000, 250),
        ]
        q = ConfirmationQueue()
        clusters = q.cluster_for_capture(tasks)
        assert len(clusters) == 3

    def test_cluster_span_respects_budget(self):
        """No cluster may exceed cluster_coverage_hz."""
        tasks = [_task(i, 910.0 + i * 0.300, 250) for i in range(10)]
        q = ConfirmationQueue()
        clusters = q.cluster_for_capture(tasks)
        for c in clusters:
            assert c.span_hz <= DEFAULT_CLUSTER_COVERAGE_HZ, (
                f"cluster span {c.span_hz/1e6:.3f} MHz exceeds budget "
                f"{DEFAULT_CLUSTER_COVERAGE_HZ/1e6:.1f} MHz"
            )

    def test_field_scenario_two_lorawan(self):
        """The exact scenario from the 15:51 scan: 912.502 MHz and
        912.603 MHz, both LoRaWAN 250 kHz. Should cluster to one
        capture."""
        tasks = [_task(1, 912.502, 250), _task(2, 912.603, 500)]
        q = ConfirmationQueue()
        clusters = q.cluster_for_capture(tasks)
        assert len(clusters) == 1, (
            "the actual 15:51 detections should have batched into "
            "one IQ capture, not two"
        )

    def test_greedy_packing_is_stable(self):
        """Cluster output is deterministic regardless of input order."""
        tasks = [
            _task(1, 906.0, 250),
            _task(2, 910.0, 250),
            _task(3, 906.5, 250),
            _task(4, 911.0, 250),
        ]
        q = ConfirmationQueue()
        import random
        result_a = q.cluster_for_capture(list(tasks))
        random.seed(42)
        random.shuffle(tasks)
        result_b = q.cluster_for_capture(list(tasks))
        # Same cluster count regardless of input order
        assert len(result_a) == len(result_b)

    def test_in_range_filters_by_dongle_coverage(self):
        """A 915 MHz dongle shouldn't see 433 MHz clusters."""
        q = ConfirmationQueue()
        import asyncio
        asyncio.run(q.submit(_task(1, 906.875, 250)))
        asyncio.run(q.submit(_task(2, 433.920, 125)))

        clusters_915 = q.clusters_in_range(
            freq_low=902_000_000, freq_high=928_000_000
        )
        clusters_433 = q.clusters_in_range(
            freq_low=430_000_000, freq_high=435_000_000
        )
        assert len(clusters_915) == 1
        assert len(clusters_433) == 1
        # And they're different clusters
        assert clusters_915[0].center_freq_hz > 900_000_000
        assert clusters_433[0].center_freq_hz < 500_000_000


# ------------------------------------------------------------
# BatchedConfirmationTask properties
# ------------------------------------------------------------


class TestBatchedTask:
    def test_center_is_midpoint(self):
        cluster = BatchedConfirmationTask(
            tasks=[_task(1, 912.5, 250), _task(2, 913.5, 250)]
        )
        assert cluster.center_freq_hz == 913_000_000

    def test_empty_cluster_center_raises(self):
        cluster = BatchedConfirmationTask(tasks=[])
        with pytest.raises(ValueError):
            _ = cluster.center_freq_hz

    def test_span_includes_bandwidths(self):
        cluster = BatchedConfirmationTask(
            tasks=[_task(1, 912.0, 250), _task(2, 912.5, 250)]
        )
        # 500 kHz center distance + 250 kHz bandwidth = 750 kHz span
        assert cluster.span_hz == 750_000

    def test_describe_is_informative(self):
        cluster = BatchedConfirmationTask(
            tasks=[_task(1, 912.502, 250), _task(2, 912.603, 250)]
        )
        desc = cluster.describe()
        assert "912" in desc
        assert "2" in desc  # "2 task(s)"


# ------------------------------------------------------------
# Lifecycle transitions
# ------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_submit_increments_pending(self):
        q = ConfirmationQueue()
        await q.submit(_task(1, 906.875, 250))
        await q.submit(_task(2, 912.500, 250))
        assert q.pending_count() == 2
        assert q.outstanding_count() == 2
        assert q.completed_count() == 0

    @pytest.mark.asyncio
    async def test_mark_scheduled_moves_out_of_pending(self):
        q = ConfirmationQueue()
        await q.submit(_task(1, 906.875, 250))
        clusters = q.cluster_for_capture()
        assert len(clusters) == 1
        await q.mark_scheduled(clusters[0])
        assert q.pending_count() == 0
        assert q.outstanding_count() == 1

    @pytest.mark.asyncio
    async def test_mark_completed_with_full_success(self):
        q = ConfirmationQueue()
        await q.submit(_task(1, 906.875, 250))
        await q.submit(_task(2, 912.500, 250))  # separate cluster
        clusters = q.cluster_for_capture()
        for c in clusters:
            await q.mark_scheduled(c)
        # Confirm all
        for c in clusters:
            confirmed = {t.detection_id for t in c.tasks}
            await q.mark_completed(c, confirmed)
        assert q.completed_count() == 2
        assert q.abandoned_count() == 0
        assert q.outstanding_count() == 0
        assert q.has_work() is False

    @pytest.mark.asyncio
    async def test_mark_completed_with_timeout(self):
        """Tasks not in confirmed_ids are counted as abandoned."""
        q = ConfirmationQueue()
        await q.submit(_task(1, 906.875, 250))
        await q.submit(_task(2, 906.880, 250))  # dedup w/ 1
        await q.submit(_task(3, 907.000, 250))  # distinct
        # Only 2 distinct tasks after dedup
        assert q.pending_count() == 2
        clusters = q.cluster_for_capture()
        for c in clusters:
            await q.mark_scheduled(c)
            # Confirm none
            await q.mark_completed(c, confirmed_ids=set())
        assert q.completed_count() == 0
        assert q.abandoned_count() == 2
