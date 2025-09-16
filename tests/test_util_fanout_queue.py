import asyncio
from unittest.mock import MagicMock, patch
import pytest
from palabra_ai.util.fanout_queue import FanoutQueue, Subscription
from palabra_ai.task.base import TaskEvent

class TestFanoutQueue:
    """Test FanoutQueue class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.queue = FanoutQueue()

    def test_init(self):
        """Test initialization"""
        assert self.queue.subscribers == {}
        assert self.queue._closed is False

    def test_get_id_string(self):
        """Test _get_id with string subscriber"""
        subscriber_id = self.queue._get_id("test_subscriber")
        assert subscriber_id == "test_subscriber"

    def test_get_id_object_with_name(self):
        """Test _get_id with object that has name attribute"""
        obj = MagicMock()
        obj.name = "test_object"
        subscriber_id = self.queue._get_id(obj)
        # Should include name, object id, and queue id
        assert subscriber_id.startswith("test_object_")
        assert str(id(obj)) in subscriber_id
        assert str(id(self.queue)) in subscriber_id

    def test_get_id_object_without_name(self):
        """Test _get_id with object without name attribute"""
        obj = MagicMock()
        del obj.name  # Remove name attribute
        subscriber_id = self.queue._get_id(obj)
        # Should include type name, object id, and queue id
        assert "MagicMock" in subscriber_id
        assert str(id(obj)) in subscriber_id
        assert str(id(self.queue)) in subscriber_id

    def test_get_id_integer(self):
        """Test _get_id with integer (which is an object in Python)"""
        # In Python, integers are objects, so this should work
        subscriber_id = self.queue._get_id(123)
        # Should contain type name, object id, and queue id
        assert "<class 'int'>_" in subscriber_id
        assert str(id(123)) in subscriber_id
        assert str(id(self.queue)) in subscriber_id

    def test_is_subscribed(self):
        """Test is_subscribed method"""
        assert not self.queue.is_subscribed("test")

        # Subscribe and check
        self.queue.subscribe("test")
        assert self.queue.is_subscribed("test")

    def test_subscribe_success(self):
        """Test successful subscription"""
        subscription = self.queue.subscribe("test_sub", maxsize=10)

        assert isinstance(subscription, Subscription)
        assert subscription.id_ == "test_sub"
        assert subscription.foq == self.queue
        assert subscription.q.maxsize == 10
        assert "test_sub" in self.queue.subscribers

    def test_subscribe_already_subscribed(self):
        """Test subscribing when already subscribed"""
        self.queue.subscribe("test_sub")

        with pytest.raises(ValueError) as exc_info:
            self.queue.subscribe("test_sub")
        assert "is already subscribed" in str(exc_info.value)

    def test_subscribe_when_closed(self):
        """Test subscribing to closed queue"""
        self.queue.close()

        with pytest.raises(RuntimeError) as exc_info:
            self.queue.subscribe("test")
        assert "FanoutQueue is closed" in str(exc_info.value)

    def test_unsubscribe_existing(self):
        """Test unsubscribing existing subscriber"""
        subscription = self.queue.subscribe("test_sub")

        # Unsubscribe
        self.queue.unsubscribe("test_sub")

        # Check it's removed
        assert "test_sub" not in self.queue.subscribers
        # Check None was sent to queue
        assert subscription.q.get_nowait() is None

    def test_unsubscribe_non_existing(self):
        """Test unsubscribing non-existing subscriber"""
        # Should not raise any error
        self.queue.unsubscribe("non_existing")

    def test_publish_to_subscribers(self):
        """Test publishing message to subscribers"""
        sub1 = self.queue.subscribe("sub1")
        sub2 = self.queue.subscribe("sub2")

        # Publish message
        self.queue.publish("test_message")

        # Check both received it
        assert sub1.q.get_nowait() == "test_message"
        assert sub2.q.get_nowait() == "test_message"

    def test_publish_none(self):
        """Test publishing None message"""
        sub = self.queue.subscribe("sub")

        # Publish None
        self.queue.publish(None)

        # Check it was received
        assert sub.q.get_nowait() is None

    def test_publish_when_closed(self):
        """Test publishing to closed queue"""
        self.queue.close()

        with pytest.raises(RuntimeError) as exc_info:
            self.queue.publish("test")
        assert "FanoutQueue is closed" in str(exc_info.value)

    @patch('palabra_ai.util.fanout_queue.debug')
    def test_publish_queue_full(self, mock_debug):
        """Test publishing when subscriber queue is full"""
        # Subscribe with maxsize=1
        sub = self.queue.subscribe("sub", maxsize=1)

        # Fill the queue
        self.queue.publish("msg1")

        # Try to publish another (should skip and log debug)
        self.queue.publish("msg2")

        # Check debug was called
        mock_debug.assert_called_once()
        assert "Queue full" in mock_debug.call_args[0][0]

        # Check only first message in queue
        assert sub.q.get_nowait() == "msg1"
        assert sub.q.empty()

    def test_close(self):
        """Test close method"""
        # Subscribe some subscribers
        sub1 = self.queue.subscribe("sub1")
        sub2 = self.queue.subscribe("sub2")

        # Close queue
        self.queue.close()

        # Check it's closed
        assert self.queue._closed is True

        # Check all subscribers were unsubscribed
        assert len(self.queue.subscribers) == 0

        # Check None was sent to all queues
        assert sub1.q.get_nowait() is None
        assert sub2.q.get_nowait() is None

    def test_close_already_closed(self):
        """Test closing already closed queue"""
        self.queue.close()

        # Should not raise error
        self.queue.close()

    @pytest.mark.asyncio
    async def test_receiver_context_manager(self):
        """Test receiver context manager"""
        stopper = TaskEvent()
        messages_received = []

        # Use receiver
        async with self.queue.receiver("test_subscriber", stopper, timeout=0.1) as receiver:
            # Publish some messages
            self.queue.publish("msg1")
            self.queue.publish("msg2")

            # Receive messages
            async for msg in receiver:
                messages_received.append(msg)
                if len(messages_received) >= 2:
                    break

        assert messages_received == ["msg1", "msg2"]
        # Check subscriber was cleaned up
        assert "test_subscriber" not in self.queue.subscribers

    @pytest.mark.asyncio
    async def test_receiver_with_none_message(self):
        """Test receiver stops on None message"""
        stopper = TaskEvent()
        messages_received = []

        async with self.queue.receiver("test_subscriber", stopper, timeout=0.1) as receiver:
            # Publish message then None
            self.queue.publish("msg1")
            self.queue.publish(None)

            # Receive messages
            async for msg in receiver:
                messages_received.append(msg)

        # Should only receive msg1, not None
        assert messages_received == ["msg1"]

    @pytest.mark.asyncio
    async def test_receiver_with_stopper(self):
        """Test receiver stops when stopper is set"""
        stopper = TaskEvent()
        messages_received = []

        async with self.queue.receiver("test_subscriber", stopper, timeout=0.1) as receiver:
            # Publish message
            self.queue.publish("msg1")

            # Start receiving
            async for msg in receiver:
                messages_received.append(msg)
                # Set stopper after first message
                +stopper

        assert messages_received == ["msg1"]


    @pytest.mark.asyncio
    async def test_receiver_cleanup_on_exception(self):
        """Test receiver cleanup on exception"""
        stopper = TaskEvent()

        with pytest.raises(RuntimeError):
            async with self.queue.receiver("test_subscriber", stopper, timeout=0.1) as receiver:
                # Publish message
                self.queue.publish("msg1")

                # Raise exception during iteration
                async for msg in receiver:
                    raise RuntimeError("Test error")

        # Check subscriber was cleaned up
        assert "test_subscriber" not in self.queue.subscribers

    def test_queue_collision_prevention(self):
        """Test that multiple queues with same subscriber don't collide"""
        queue1 = FanoutQueue()
        queue2 = FanoutQueue()

        # Same object subscribed to both queues
        obj = MagicMock()
        obj.name = "test_obj"

        # Subscribe to both queues
        sub1 = queue1.subscribe(obj)
        sub2 = queue2.subscribe(obj)

        # Should create different subscriber IDs due to queue ID inclusion
        assert sub1.id_ != sub2.id_
        assert str(id(queue1)) in sub1.id_
        assert str(id(queue2)) in sub2.id_

        # Publish to each queue
        queue1.publish("msg1")
        queue2.publish("msg2")

        # Each should receive only its own message
        assert sub1.q.get_nowait() == "msg1"
        assert sub2.q.get_nowait() == "msg2"
