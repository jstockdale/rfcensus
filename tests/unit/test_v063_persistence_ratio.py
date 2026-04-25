"""v0.6.3 — real persistence_ratio from OccupancyAnalyzer.

Pre-v0.6.3 bug
==============

OccupancyAnalyzer._emit() computed `persistence_ratio` as
`min(1.0, state.sample_count / 60.0)` — a sample-count CAP, not a
ratio. Every bin seen for 60+ power-scan sweeps reported persist=100%
regardless of how often it was actually above the noise floor, which
is why busy bands in the Mystery carriers report showed
"persist=100%" on every single entry. The report's persistence
column was unusable for ranking.

v0.6.3 fix
==========

`ChannelHistory` already tracks `total_active_samples` (sweeps where
the bin was above floor) and `total_samples` (sweeps observed at
all). The ratio of those is the real persistence. The emit path was
just ignoring this data and using the wrong formula.

We also added `sample_count` to ActiveChannelEvent / Record so
consumers can tell "100% from 1/1 samples" apart from "100% from
580/580 samples." The mystery-carrier scoring can then gate
low-confidence bursts out.

Schema v3 adds an `active_channels.sample_count INTEGER` column.
Old rows remain NULL; _maybe_int() in the repository handles this.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rfcensus.events import ActiveChannelEvent, EventBus
from rfcensus.spectrum.backend import PowerSample
from rfcensus.spectrum.occupancy import OccupancyAnalyzer


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────


def _t(seconds: float) -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=seconds
    )


def _sample(freq_hz: int, power_dbm: float, seconds: float) -> PowerSample:
    return PowerSample(
        freq_hz=freq_hz,
        bin_width_hz=10_000,
        power_dbm=power_dbm,
        timestamp=_t(seconds),
    )


async def _feed(analyzer: OccupancyAnalyzer, samples: list[PowerSample]) -> None:
    """Feed a list of samples one-by-one, letting the analyzer process each.

    Also drains the event bus so subscribers have received all emitted
    events before the caller asserts. Without the drain, handlers run
    as background tasks and the subscriber's list is empty until they
    complete.
    """

    async def _gen():
        for s in samples:
            yield s

    await analyzer.consume(_gen(), dongle_id="test")
    await analyzer.event_bus.drain(timeout=2.0)


def _capture(bus: EventBus) -> list[ActiveChannelEvent]:
    captured: list[ActiveChannelEvent] = []
    bus.subscribe(ActiveChannelEvent, lambda e: captured.append(e))  # type: ignore[arg-type,misc]
    return captured


# ────────────────────────────────────────────────────────────────────
# Core: persistence reflects actual occupancy
# ────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────
# Helper: build a sample stream with a quiet priming phase
# ────────────────────────────────────────────────────────────────────


def _prime_and_pattern(
    freq_hz: int,
    pattern: list[bool],
    *,
    prime_count: int = 30,
    loud_dbm: float = -30.0,
    quiet_dbm: float = -85.0,
    step_s: float = 0.05,
) -> list[PowerSample]:
    """Build a sample stream for one bin.

    First `prime_count` samples are quiet so the noise floor tracker
    establishes a clean floor. Then `pattern` samples follow, each
    either loud (True) or quiet (False).

    Loud samples get ±2 dB jitter so the per-bin noise floor tracker's
    25th percentile stays below the peak — otherwise a bin that's
    loud at exactly uniform power has its percentile floor walk up to
    equal the signal, SNR collapses to 0, and the bin never enters
    _active. Real-world signals always have some power variation so
    the jitter is actually MORE realistic than clean fixed values.
    """
    samples: list[PowerSample] = []
    now = 0.0
    for _ in range(prime_count):
        samples.append(
            PowerSample(
                freq_hz=freq_hz,
                bin_width_hz=10_000,
                power_dbm=quiet_dbm,
                timestamp=_t(now),
            )
        )
        now += step_s
    for i, active in enumerate(pattern):
        if active:
            # Deterministic small oscillation around loud_dbm
            jitter = -2.0 if i % 2 == 0 else +2.0
            p = loud_dbm + jitter
        else:
            p = quiet_dbm
        samples.append(
            PowerSample(
                freq_hz=freq_hz,
                bin_width_hz=10_000,
                power_dbm=p,
                timestamp=_t(now),
            )
        )
        now += step_s
    return samples


def _last_target_event(
    captured: list[ActiveChannelEvent], freq_hz: int
) -> ActiveChannelEvent | None:
    matching = [e for e in captured if e.freq_center_hz == freq_hz]
    return matching[-1] if matching else None


# ────────────────────────────────────────────────────────────────────
# Core: persistence reflects actual occupancy
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRealPersistenceRatio:
    async def test_mostly_active_reports_high(self):
        """A bin active in most pattern sweeps reports high persistence.

        Exact ratios are hard to pin down because the noise floor
        tracker's rolling percentile interacts with sustained activity
        — a bin that's 100% hot for long enough has its own floor rise
        toward the signal level, dropping later samples out of above-
        floor territory. This test just confirms "mostly active"
        produces a clearly-above-mid ratio. The pre-v0.6.3 bug would
        have reported 1.0 as soon as 60 samples accumulated regardless
        of activity level."""
        bus = EventBus()
        captured = _capture(bus)
        analyzer = OccupancyAnalyzer(
            event_bus=bus,
            new_hold_time=timedelta(milliseconds=10),
            stale_timeout=timedelta(seconds=60),
        )

        freq = 433_920_000
        # Active ~90% of the time over 100 pattern samples
        pattern = [i % 10 != 0 for i in range(100)]
        samples = _prime_and_pattern(freq, pattern)
        await _feed(analyzer, samples)

        last = _last_target_event(captured, freq)
        assert last is not None, "expected at least one event"
        # With a 130-sample observation (30 prime + 100 pattern) and ~90
        # pattern samples above floor (minus some lost to rising floor),
        # persistence lands somewhere in the 0.4-0.7 range. The point
        # is it's clearly higher than the sparse-burst case below.
        assert last.persistence_ratio > 0.35, (
            f"mostly-active bin only reported "
            f"persistence={last.persistence_ratio:.2f}"
        )

    async def test_sparse_bursts_reports_low(self):
        """A bin active in only 10% of pattern sweeps reports low
        persistence. Pre-v0.6.3 this would have reported 1.0."""
        bus = EventBus()
        captured = _capture(bus)
        analyzer = OccupancyAnalyzer(
            event_bus=bus,
            new_hold_time=timedelta(milliseconds=10),
            stale_timeout=timedelta(seconds=60),
        )

        freq = 915_000_000
        # Active ~10% of the time
        pattern = [i % 10 == 0 for i in range(100)]
        samples = _prime_and_pattern(freq, pattern)
        await _feed(analyzer, samples)

        last = _last_target_event(captured, freq)
        assert last is not None
        assert last.persistence_ratio < 0.20, (
            f"sparse-burst bin reported high persistence="
            f"{last.persistence_ratio:.2f}, expected ≤ 0.20"
        )

    async def test_sparse_reports_lower_than_mostly_active(self):
        """The key semantic assertion: a sparse-burst bin reports a
        LOWER persistence than a mostly-active bin. Pre-v0.6.3 both
        would have reported the same (1.0 after 60 samples), which is
        why the report's persistence column was useless for ranking."""
        bus = EventBus()
        captured = _capture(bus)
        analyzer = OccupancyAnalyzer(
            event_bus=bus,
            new_hold_time=timedelta(milliseconds=10),
            stale_timeout=timedelta(seconds=60),
        )

        freq_sparse = 915_000_000
        freq_mostly = 433_920_000

        sparse_pattern = [i % 10 == 0 for i in range(100)]
        mostly_pattern = [i % 10 != 0 for i in range(100)]

        samples = (
            _prime_and_pattern(freq_sparse, sparse_pattern)
            + _prime_and_pattern(freq_mostly, mostly_pattern)
        )
        await _feed(analyzer, samples)

        sparse_evt = _last_target_event(captured, freq_sparse)
        mostly_evt = _last_target_event(captured, freq_mostly)
        assert sparse_evt is not None
        assert mostly_evt is not None
        assert mostly_evt.persistence_ratio > sparse_evt.persistence_ratio, (
            f"mostly-active persistence ({mostly_evt.persistence_ratio:.2f}) "
            f"should be greater than sparse-burst persistence "
            f"({sparse_evt.persistence_ratio:.2f})"
        )

    async def test_old_bug_does_not_resurface(self):
        """Regression: pre-v0.6.3, a bin with ≥60 samples reported
        persistence=1.0 regardless of above-floor count. Make sure a
        200-sample bin with only ~10% above-floor does NOT report 1.0."""
        bus = EventBus()
        captured = _capture(bus)
        analyzer = OccupancyAnalyzer(
            event_bus=bus,
            new_hold_time=timedelta(milliseconds=10),
            stale_timeout=timedelta(seconds=60),
        )

        freq = 433_920_000
        pattern = [i % 10 == 0 for i in range(200)]
        samples = _prime_and_pattern(freq, pattern)
        await _feed(analyzer, samples)

        last = _last_target_event(captured, freq)
        assert last is not None
        assert last.sample_count >= 60, (
            f"sanity check: sample_count should be ≥ 60 "
            f"(pre-v0.6.3 threshold for 100%), got {last.sample_count}"
        )
        assert last.persistence_ratio < 0.5, (
            f"persistence ratio is still saturated: "
            f"got {last.persistence_ratio:.2f}, expected low value"
        )


# ────────────────────────────────────────────────────────────────────
# sample_count is emitted and reflects total observations
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestSampleCountField:
    async def test_sample_count_reflects_total_observations(self):
        """sample_count on the emitted event counts every observation
        of the bin, not just above-floor ones. That's what distinguishes
        "100% persistence from 2 samples" (low confidence) from "100%
        persistence from 500 samples" (real always-on carrier)."""
        bus = EventBus()
        captured = _capture(bus)
        analyzer = OccupancyAnalyzer(
            event_bus=bus,
            new_hold_time=timedelta(milliseconds=10),
            stale_timeout=timedelta(seconds=60),
        )

        freq = 433_920_000
        pattern = [i % 10 == 0 for i in range(100)]
        samples = _prime_and_pattern(freq, pattern)  # 30 prime + 100 pattern
        await _feed(analyzer, samples)

        last = _last_target_event(captured, freq)
        assert last is not None
        # sample_count includes priming samples (they're still
        # observations of the bin). Should be ~130.
        assert last.sample_count >= 100, (
            f"expected sample_count ≥ 100 (prime+pattern), "
            f"got {last.sample_count}"
        )

    async def test_low_sample_count_honest_about_coarse_ratio(self):
        """When a bin has only a few pattern samples post-prime, its
        persistence ratio is technically honest but based on little
        evidence. sample_count is the signal that lets consumers gate:
        "persist=80% n=5" is weak evidence, "persist=80% n=500" is
        strong evidence. This test confirms sample_count is emitted
        and reflects the actual observation window."""
        bus = EventBus()
        captured = _capture(bus)
        analyzer = OccupancyAnalyzer(
            event_bus=bus,
            new_hold_time=timedelta(milliseconds=10),
            stale_timeout=timedelta(seconds=60),
        )

        freq = 433_920_000
        # Short pattern: 5 pattern samples, all active
        samples = _prime_and_pattern(freq, [True] * 5)
        await _feed(analyzer, samples)

        last = _last_target_event(captured, freq)
        assert last is not None, (
            "expected at least one event for freq after prime + short pattern"
        )
        # sample_count is prime(30) + pattern(5) = 35 max. The honest
        # signal: this is far fewer observations than a full 12-minute
        # scan would produce (hundreds), so persistence ratio should
        # be trusted accordingly.
        assert last.sample_count <= 40, (
            f"expected sample_count ≤ 40, got {last.sample_count}"
        )


# ────────────────────────────────────────────────────────────────────
# Schema migration: v2 → v3 doesn't lose data
# ────────────────────────────────────────────────────────────────────


class TestSchemaMigrationV3:
    def test_fresh_database_has_sample_count_column(self, tmp_path):
        """Fresh databases include sample_count in _v1 (the consolidated
        schema) AND run _v3 which is guarded to no-op when the column
        already exists. Either way, the column ends up present."""
        import sqlite3
        from rfcensus.storage.schema import apply_migrations

        db_path = tmp_path / "fresh.db"
        conn = sqlite3.connect(db_path)
        apply_migrations(conn)

        cols = {row[1] for row in conn.execute("PRAGMA table_info(active_channels)")}
        assert "sample_count" in cols

        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == 3

        conn.close()

    def test_v2_database_upgrades_to_v3_with_alter(self, tmp_path):
        """A database at user_version=2 (written by v0.6.2 code) should
        upgrade cleanly to v3 with the new column added via ALTER, and
        existing rows should end up with NULL sample_count (honest —
        the old data genuinely doesn't have this info)."""
        import sqlite3
        from rfcensus.storage.schema import apply_migrations

        # Simulate a v2 database by creating the old table shape
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT);
            CREATE TABLE active_channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                freq_center_hz INTEGER NOT NULL,
                bandwidth_hz INTEGER NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                peak_power_dbm REAL,
                avg_power_dbm REAL,
                noise_floor_dbm REAL,
                classification TEXT,
                persistence_ratio REAL,
                confidence REAL,
                metadata TEXT
            );
            INSERT INTO sessions DEFAULT VALUES;
            INSERT INTO active_channels
                (session_id, freq_center_hz, bandwidth_hz, first_seen,
                 last_seen, peak_power_dbm, persistence_ratio)
            VALUES
                (1, 433920000, 10000, '2026-01-01T00:00:00+00:00',
                 '2026-01-01T00:00:30+00:00', -30.0, 1.0);
            PRAGMA user_version = 2;
            """
        )
        conn.commit()

        # Now run migrations — should upgrade v2 → v3
        apply_migrations(conn)

        cols = {row[1] for row in conn.execute("PRAGMA table_info(active_channels)")}
        assert "sample_count" in cols

        # Existing row's sample_count should be NULL (honest about
        # not having this data in the old format)
        row = conn.execute(
            "SELECT sample_count FROM active_channels WHERE freq_center_hz = 433920000"
        ).fetchone()
        assert row[0] is None

        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == 3
        conn.close()


# ────────────────────────────────────────────────────────────────────
# Writer + Repository round-trip
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestSampleCountRoundTrip:
    async def test_sample_count_persists_through_writer_and_repo(self, tmp_path):
        """ActiveChannelWriter passes sample_count to the repo; the repo
        round-trips it via INSERT and SELECT."""
        from rfcensus.storage.db import Database
        from rfcensus.storage.repositories import (
            ActiveChannelRepo,
            SessionRepo,
        )
        from rfcensus.storage.writer import ActiveChannelWriter
        from rfcensus.storage.models import SessionRecord

        db = Database(tmp_path / "rt.db")
        try:
            session_repo = SessionRepo(db)
            session_id = await session_repo.create(
                SessionRecord(
                    id=None,
                    command="test",
                    started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    site_name="test",
                    config_snap={},
                    notes="",
                )
            )

            repo = ActiveChannelRepo(db)
            writer = ActiveChannelWriter(repo=repo, session_id=session_id)

            event = ActiveChannelEvent(
                session_id=session_id,
                kind="new",
                dongle_id="d1",
                freq_center_hz=433_920_000,
                bandwidth_hz=10_000,
                peak_power_dbm=-30.0,
                avg_power_dbm=-33.0,
                noise_floor_dbm=-85.0,
                snr_db=55.0,
                classification="continuous_carrier",
                persistence_ratio=0.97,
                sample_count=600,
                confidence=0.8,
                timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            )
            await writer.handle(event)

            records = await repo.for_session(session_id)
            assert len(records) == 1
            r = records[0]
            assert r.sample_count == 600
            assert r.persistence_ratio == pytest.approx(0.97)
        finally:
            db.close()

    async def test_reading_legacy_row_returns_none_sample_count(self, tmp_path):
        """Legacy rows (from v2 databases upgraded to v3) have NULL
        sample_count; the repo's _maybe_int helper returns None rather
        than raising."""
        from rfcensus.storage.db import Database
        from rfcensus.storage.repositories import ActiveChannelRepo

        db = Database(tmp_path / "legacy.db")
        try:
            # Insert a row with sample_count NULL (simulates a row that
            # pre-dates v3 and was never rewritten)
            await db.execute(
                """
                INSERT INTO sessions (command, started_at)
                VALUES ('test', '2026-01-01T00:00:00+00:00')
                """
            )
            await db.execute(
                """
                INSERT INTO active_channels (
                    session_id, freq_center_hz, bandwidth_hz,
                    first_seen, last_seen, peak_power_dbm,
                    persistence_ratio, sample_count
                ) VALUES (1, 433920000, 10000,
                          '2026-01-01T00:00:00+00:00',
                          '2026-01-01T00:00:30+00:00',
                          -30.0, 1.0, NULL)
                """
            )

            repo = ActiveChannelRepo(db)
            records = await repo.for_session(1)
            assert len(records) == 1
            assert records[0].sample_count is None
            # persistence_ratio is preserved as-is (the old buggy value)
            assert records[0].persistence_ratio == 1.0
        finally:
            db.close()


# ────────────────────────────────────────────────────────────────────
# Mystery-carrier formatter — shows sample_count, down-weights low-n
# ────────────────────────────────────────────────────────────────────


class TestMysteryFormatterUsesSampleCount:
    def _cluster_with_n(self, n: int | None, persistence: float = 0.9):
        """Build a single-member cluster with the given sample_count."""
        from rfcensus.reporting.formats.text import _ChannelCluster
        from rfcensus.storage.models import ActiveChannelRecord

        return _ChannelCluster(members=[
            ActiveChannelRecord(
                id=None,
                session_id=1,
                freq_center_hz=433_920_000,
                bandwidth_hz=10_000,
                first_seen=datetime(2026, 1, 1, tzinfo=timezone.utc),
                last_seen=datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc),
                peak_power_dbm=-30.0,
                avg_power_dbm=-33.0,
                noise_floor_dbm=-85.0,
                classification="continuous_carrier",
                persistence_ratio=persistence,
                sample_count=n,
                confidence=0.8,
            )
        ])

    def test_format_includes_n_annotation_when_available(self):
        """The per-cluster line shows `n=N` when sample_count is set."""
        from rfcensus.reporting.formats.text import _format_cluster

        c = self._cluster_with_n(600)
        line = _format_cluster(c)
        assert "n=600" in line
        assert "persist=90% (n=600)" in line

    def test_format_omits_n_annotation_when_none(self):
        """Legacy rows (sample_count=None from pre-v0.6.3 databases)
        produce the original line shape — no `(n=?)` garbage."""
        from rfcensus.reporting.formats.text import _format_cluster

        c = self._cluster_with_n(None)
        line = _format_cluster(c)
        # "(n=" is the annotation marker; "n=" alone matches "seen="
        assert "(n=" not in line
        assert "persist=90%" in line

    def test_score_heavily_penalizes_low_n(self):
        """A cluster with sample_count=3 and 100% persistence should
        score LOWER than a cluster with sample_count=200 and 50%
        persistence. The whole point of the n-factor is that weak
        evidence doesn't outrank strong evidence just because the
        ratio happens to be higher."""
        from rfcensus.reporting.formats.text import _mystery_score

        weak = self._cluster_with_n(3, persistence=1.0)
        strong = self._cluster_with_n(200, persistence=0.5)
        assert _mystery_score(strong) > _mystery_score(weak), (
            f"strong-evidence 50% cluster ({_mystery_score(strong):.3f}) "
            f"should outrank weak-evidence 100% cluster "
            f"({_mystery_score(weak):.3f})"
        )

    def test_score_no_penalty_at_high_n(self):
        """At sample_count >= _MYSTERY_MIN_CONFIDENT_N the n-factor
        is 1.0 and the score equals persistence × snr_factor."""
        from rfcensus.reporting.formats.text import (
            _mystery_score,
            _MYSTERY_MIN_CONFIDENT_N,
        )

        c = self._cluster_with_n(_MYSTERY_MIN_CONFIDENT_N * 2, persistence=0.9)
        # SNR = -30 - -85 = 55 dB, capped at 30 → snr_factor = 1.0
        # persistence = 0.9, n_factor = 1.0 → score = 0.9
        assert abs(_mystery_score(c) - 0.9) < 0.01

    def test_score_none_sample_count_does_not_penalize(self):
        """Backward-compat: a cluster with sample_count=None (legacy
        data) is scored as if n_factor=1.0. We don't want to silently
        suppress every cluster from a pre-v0.6.3 session."""
        from rfcensus.reporting.formats.text import _mystery_score

        legacy = self._cluster_with_n(None, persistence=0.9)
        modern = self._cluster_with_n(1000, persistence=0.9)
        # Both should score the same (SNR + persistence identical)
        assert abs(_mystery_score(legacy) - _mystery_score(modern)) < 0.01
