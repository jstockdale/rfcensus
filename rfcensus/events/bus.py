"""Simple asyncio pub-sub event bus.

Publishers call `bus.publish(event)`. Subscribers register handlers via
`bus.subscribe(EventType, handler)` and receive events of that type (or
any subtype).

Handlers run as asyncio tasks; a slow handler will not block publishers.
If a handler raises, the exception is logged and the bus continues
delivering to other subscribers.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from rfcensus.events.events import Event
from rfcensus.utils.logging import get_logger

log = get_logger(__name__)

E = TypeVar("E", bound=Event)
Handler = Callable[[E], Awaitable[None]] | Callable[[E], None]


@dataclass
class Subscription:
    """Handle returned by `subscribe()`. Use it to unsubscribe later."""

    event_type: type
    handler: Handler[Any]
    _active: bool = field(default=True, init=False)

    def cancel(self) -> None:
        self._active = False


class EventBus:
    """Pub-sub bus keyed by event type.

    The bus uses `isinstance` checks so subscribing to a base class
    (`Event`) delivers all events. Subscribing to a specific subclass
    delivers only that type.
    """

    def __init__(self) -> None:
        self._subscriptions: list[Subscription] = []
        self._background_tasks: set[asyncio.Task[None]] = set()

    def subscribe(self, event_type: type[E], handler: Handler[E]) -> Subscription:
        sub = Subscription(event_type=event_type, handler=handler)  # type: ignore[arg-type]
        self._subscriptions.append(sub)
        return sub

    async def publish(self, event: Event) -> None:
        """Publish an event. Handlers run concurrently as background tasks.

        This returns when dispatch is complete (all handler tasks scheduled),
        not when all handlers have finished. For back-pressure guarantees,
        handlers should implement their own queuing.
        """
        for sub in list(self._subscriptions):
            if not sub._active:
                continue
            if not isinstance(event, sub.event_type):
                continue
            self._dispatch(sub, event)

    def _dispatch(self, sub: Subscription, event: Event) -> None:
        handler = sub.handler

        async def _run() -> None:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                log.exception("event handler %s raised", handler)

        task = asyncio.create_task(_run())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait for all in-flight handler tasks to complete."""
        if not self._background_tasks:
            return
        pending = list(self._background_tasks)
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True), timeout=timeout
            )
        except TimeoutError:
            log.warning("event bus drain timed out with %d tasks pending", len(pending))

    def stats(self) -> dict[str, int]:
        return {
            "subscriptions": sum(1 for s in self._subscriptions if s._active),
            "in_flight": len(self._background_tasks),
        }
