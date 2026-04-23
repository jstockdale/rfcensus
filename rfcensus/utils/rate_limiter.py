"""Token-bucket rate limiter.

Used to prevent runaway decoders from flooding the event bus or database.
If a decoder is producing frames faster than a plausible real-world rate,
that's a strong signal it's decoding noise and we should throttle it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    """Classic token bucket.

    `rate` tokens are added per second, up to `burst` maximum. Each call
    to `try_acquire()` consumes one token if available.
    """

    rate: float  # tokens per second
    burst: int  # max tokens held
    _tokens: float = field(default=0.0, init=False)
    _last_refill: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.burst)

    def try_acquire(self, n: int = 1) -> bool:
        """Attempt to consume `n` tokens. Returns True if successful."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_refill = now
        if self._tokens >= n:
            self._tokens -= n
            return True
        return False

    def available(self) -> float:
        """Return current token count without modifying state."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        return min(self.burst, self._tokens + elapsed * self.rate)
