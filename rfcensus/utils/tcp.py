"""TCP readiness polling.

When we spawn rtl_tcp (or any other network-serving subprocess), the
process takes some time between fork and actually accepting connections
on its listening port. If a client connects in that window, it sees
connection-refused and exits — which for decoders gets misclassified
as hardware loss by the early-exit detector.

The canonical fix is not a blind sleep (timing varies with system load)
but active polling: try to open a TCP connection on the target port,
retry until it succeeds or we give up. This module provides that.
"""

from __future__ import annotations

import asyncio


async def wait_for_tcp_ready(
    host: str, port: int, timeout_s: float = 5.0
) -> bool:
    """Poll until a TCP connection to host:port succeeds.

    Returns True if a connection succeeded within timeout_s, False
    otherwise. Safe to call multiple times on the same port — this
    is just a liveness probe, it doesn't hold the connection.
    """
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=0.5,
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return True
        except (OSError, asyncio.TimeoutError):
            await asyncio.sleep(0.15)
    return False
