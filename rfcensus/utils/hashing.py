"""Privacy-preserving hashing for device identifiers.

Device IDs from decoded protocols (meter IDs, TPMS IDs, etc.) are highly
identifying. An ERT meter ID correlates directly to a utility customer
record; a TPMS ID travels with a vehicle. We hash these by default in
reports and exports unless the user opts in.

The hash is salted per-site. Two installations of rfcensus produce
different hashes for the same raw ID, preventing correlation across
sites. Within a single site, the same raw ID always hashes the same way
so that time-series analysis works.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Final

_SALT_BYTES: Final[int] = 32
_HASH_LENGTH: Final[int] = 12  # Characters shown in reports


def generate_salt() -> str:
    """Generate a new per-site salt. Call once at `init` time."""
    return secrets.token_hex(_SALT_BYTES)


def hash_id(raw_id: str, salt: str, length: int = _HASH_LENGTH) -> str:
    """Hash a raw device ID with the given salt.

    Returns a hex string of `length` characters. Always the same output
    for the same (raw_id, salt) pair.
    """
    if not salt:
        raise ValueError("salt must be non-empty")
    mac = hmac.new(salt.encode("utf-8"), raw_id.encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()[:length]


def format_id(raw_id: str, salt: str, include_raw: bool = False) -> str:
    """Format an ID for display, hashed or raw.

    If `include_raw` is false (default), returns "hash:abc123def456".
    If true, returns the raw ID as-is.
    """
    if include_raw:
        return raw_id
    return f"hash:{hash_id(raw_id, salt)}"
