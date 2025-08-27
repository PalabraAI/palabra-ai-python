import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import (
    Any,
    Generic,
    NamedTuple,
    TypeVar,
)

from palabra_ai.task.base import TaskEvent
from palabra_ai.util.logger import debug

T = TypeVar("T")


class Subscription(NamedTuple):
    id_: str
    q: asyncio.Queue[T | None]
    foq: "FanoutQueue[T]" = None


class FanoutQueue(Generic[T]):
    def __init__(self):
        self.subscribers: dict[str, Subscription] = {}
        self._closed = False

    def _get_id(self, subscriber: Any) -> str:
        if isinstance(subscriber, str):
            return subscriber
        elif isinstance(subscriber, object):
            # Include queue instance ID to avoid collisions between different queues
            queue_id = id(self)
            subscriber_id = id(subscriber)
            if name := getattr(subscriber, "name", None):
                return f"{name}_{subscriber_id}_{queue_id}"
            return f"{type(subscriber)}_{subscriber_id}_{queue_id}"
        else:
            raise TypeError(
                f"Subscriber must be a string or an object, got: {type(subscriber)}"
            )

    def is_subscribed(self, subscriber: Any) -> bool:
        """Check if subscriber is currently subscribed"""
        subscriber_id = self._get_id(subscriber)
        return subscriber_id in self.subscribers

    def subscribe(self, subscriber: Any, maxsize: int = 0) -> Subscription:
        if self._closed:
            raise RuntimeError("FanoutQueue is closed")
        subscriber_id = self._get_id(subscriber)
        if subscriber_id in self.subscribers:
            raise ValueError(f"Subscriber {subscriber} is already subscribed")
        subscription = Subscription(
            id_=subscriber_id, q=asyncio.Queue(maxsize), foq=self
        )
        self.subscribers[subscriber_id] = subscription
        return subscription

    def unsubscribe(self, subscriber: Any) -> None:
        subscriber_id = self._get_id(subscriber)
        subscription = self.subscribers.pop(subscriber_id, None)
        if subscription is None:
            return

        # Always send None to signal termination
        subscription.q.put_nowait(None)

    def publish(self, message: T | None) -> None:
        """Publish message to all subscribers. Can be None."""
        if self._closed:
            raise RuntimeError("FanoutQueue is closed")

        for subscription in self.subscribers.values():
            try:
                subscription.q.put_nowait(message)
            except asyncio.QueueFull:
                debug(f"Queue full for subscriber {subscription.id_}, skipping message")

    def close(self) -> None:
        """Close the FanoutQueue and unsubscribe all subscribers"""
        if self._closed:
            return

        self._closed = True
        debug("Closing FanoutQueue")

        # Copy list to avoid modification during iteration
        subscriber_ids = list(self.subscribers.keys())
        for subscriber_id in subscriber_ids:
            self.unsubscribe(subscriber_id)

        debug(f"Closed FanoutQueue, unsubscribed {len(subscriber_ids)} subscribers")

    @asynccontextmanager
    async def receiver(
        self, subscriber: Any, stopper: TaskEvent, timeout: float | None = None
    ) -> AsyncIterator[AsyncGenerator[T, None]]:
        """Context manager for subscribing and receiving messages

        Args:
            subscriber: Subscriber object
            stopper: TaskEvent to signal when to stop receiving messages
            timeout: Optional timeout for waiting on messages (prevents hanging)
        """
        subscriber_id = self._get_id(subscriber)

        async def message_generator(
            subscription: Subscription,
        ) -> AsyncGenerator[T, None]:
            """Inner generator for messages"""
            while not stopper:
                try:
                    if timeout is not None:
                        # Use timeout to prevent hanging
                        msg: T | None = await asyncio.wait_for(
                            subscription.q.get(), timeout=timeout
                        )
                    else:
                        msg = await subscription.q.get()

                    # If None received, just exit
                    if msg is None:
                        break

                    yield msg

                except TimeoutError:
                    # Timeout reached, check if we should continue
                    if self._closed or not self.is_subscribed(subscriber_id) or stopper:
                        debug(
                            f"Subscriber {subscriber_id} stopping due to timeout and closed/unsubscribed state"
                        )
                        break
                    # Otherwise continue waiting

        debug(f"Starting subscriber {subscriber_id} for queue {type(self).__name__}")

        # Subscribe
        _ = self.subscribe(subscriber, maxsize=0)
        subscription = self.subscribers[subscriber_id]
        generator = message_generator(subscription)

        try:
            yield generator
        finally:
            debug(f"Cleaning up subscriber {subscriber_id}")

            # CORRECT ORDER:
            # 1. First unsubscribe (sends None to queue)
            self.unsubscribe(subscriber_id)

            # 2. Then close generator
            await generator.aclose()

            debug(f"Cleanup done for subscriber {subscriber_id}")
