"""Unit tests for utility modules."""

from __future__ import annotations

import pytest

from rfcensus.utils.hashing import format_id, generate_salt, hash_id
from rfcensus.utils.rate_limiter import RateLimiter


class TestHashing:
    def test_generate_salt_produces_hex_string(self):
        salt = generate_salt()
        assert len(salt) == 64  # 32 bytes = 64 hex chars
        assert all(c in "0123456789abcdef" for c in salt)

    def test_different_salts_differ(self):
        assert generate_salt() != generate_salt()

    def test_same_id_same_salt_same_hash(self, salt):
        assert hash_id("meter-12345", salt) == hash_id("meter-12345", salt)

    def test_different_id_different_hash(self, salt):
        assert hash_id("meter-12345", salt) != hash_id("meter-67890", salt)

    def test_different_salt_different_hash(self):
        s1 = generate_salt()
        s2 = generate_salt()
        assert hash_id("same-id", s1) != hash_id("same-id", s2)

    def test_hash_length_default(self, salt):
        assert len(hash_id("x", salt)) == 12

    def test_hash_length_custom(self, salt):
        assert len(hash_id("x", salt, length=20)) == 20

    def test_empty_salt_raises(self):
        with pytest.raises(ValueError):
            hash_id("x", "")

    def test_format_id_hashed_by_default(self, salt):
        out = format_id("real-id", salt)
        assert out.startswith("hash:")
        assert "real-id" not in out

    def test_format_id_raw_when_opted_in(self, salt):
        assert format_id("real-id", salt, include_raw=True) == "real-id"


class TestRateLimiter:
    def test_starts_full(self):
        lim = RateLimiter(rate=10.0, burst=5)
        # Can consume up to burst immediately
        for _ in range(5):
            assert lim.try_acquire()
        # Next one fails
        assert not lim.try_acquire()

    def test_refills_over_time(self, monkeypatch):
        import rfcensus.utils.rate_limiter as rl_mod

        fake_time = [1000.0]
        monkeypatch.setattr(rl_mod.time, "monotonic", lambda: fake_time[0])

        # Create limiter AFTER patching so _last_refill starts at fake_time
        lim = RateLimiter(rate=10.0, burst=5)
        lim._last_refill = fake_time[0]  # align internal clock
        for _ in range(5):
            assert lim.try_acquire()
        assert not lim.try_acquire()

        # After 0.5s of elapsed, 5 tokens refill at rate=10
        fake_time[0] += 0.5
        count = 0
        for _ in range(10):
            if lim.try_acquire():
                count += 1
        assert 4 <= count <= 6

    def test_cannot_exceed_burst(self, monkeypatch):
        import rfcensus.utils.rate_limiter as rl_mod

        fake_time = [1000.0]
        monkeypatch.setattr(rl_mod.time, "monotonic", lambda: fake_time[0])

        lim = RateLimiter(rate=10.0, burst=5)
        lim._last_refill = fake_time[0]
        # Wait 10 seconds without consuming
        fake_time[0] += 10.0
        # Should still only have burst tokens
        count = 0
        for _ in range(100):
            if lim.try_acquire():
                count += 1
        assert count == 5
