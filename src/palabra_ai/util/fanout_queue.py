import asyncio
from typing import Any, NamedTuple


class Subscription(NamedTuple):
    queue: asyncio.Queue


class FanoutQueue:
    def __init__(self):
        self.subscribers: dict[str, Subscription] = {}
        self._lock = asyncio.Lock()

    def _get_id(self, subscriber: Any) -> str:
        if not isinstance(subscriber, str):
            return str(id(subscriber))
        return subscriber

    def subscribe(self, subscriber: Any, maxsize: int = 0) -> asyncio.Queue:
        subscriber_id = self._get_id(subscriber)
        if subscriber_id not in self.subscribers:
            queue = asyncio.Queue(maxsize)
            self.subscribers[subscriber_id] = Subscription(queue)
            return queue
        return self.subscribers[subscriber_id].queue

    def unsubscribe(self, subscriber: Any) -> None:
        subscriber_id = self._get_id(subscriber)
        self.subscribers.pop(subscriber_id, None)

    def publish(self, message: Any) -> None:
        for subscription in self.subscribers.values():
            subscription.queue.put_nowait(message)
