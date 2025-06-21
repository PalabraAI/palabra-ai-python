import asyncio
import pytest
from palabra_ai.util.fanout_queue import FanoutQueue


class TestFanoutQueue:
    def test_subscribe_operations(self):
        foq = FanoutQueue()
        subscriber = "test_subscriber"

        # First subscription
        q1 = foq.subscribe(subscriber)
        assert len(foq.subscribers) == 1

        # Second subscription - same queue
        q2 = foq.subscribe(subscriber)
        assert q1 is q2
        assert len(foq.subscribers) == 1

    def test_unsubscribe_operations(self):
        foq = FanoutQueue()
        subscriber = "test_subscriber"

        # Subscribe first
        foq.subscribe(subscriber)
        assert len(foq.subscribers) == 1

        # Unsubscribe existing
        foq.unsubscribe(subscriber)
        assert len(foq.subscribers) == 0

        # Unsubscribe non-existing - should not error
        foq.unsubscribe("non_existing")
        assert len(foq.subscribers) == 0

    def test_publish_with_none(self):
        foq = FanoutQueue()
        subscriber = "test"
        q = foq.subscribe(subscriber)

        # Publish None should put None in queue
        foq.publish(None)
        assert q.get_nowait() is None
