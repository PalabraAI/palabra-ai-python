from collections import deque
from threading import RLock
from typing import Generic, TypeVar

T = TypeVar("T")


class CappedSet(Generic[T]):
    """Thread-safe set with maximum capacity and FIFO eviction strategy.

    When capacity is reached, the oldest item is removed to make space for new items.
    """

    __slots__ = ("_capacity", "_data", "_order", "_lock")

    def __init__(self, capacity: int):
        """Initialize CappedSet with given capacity.

        Args:
            capacity: Maximum number of items the set can hold

        Raises:
            ValueError: If capacity is not positive
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self._capacity = capacity
        self._data: set[T] = set()
        self._order: deque[T] = deque()
        self._lock = RLock()

    def add(self, item: T) -> None:
        """Add an item to the set.

        If the set is at capacity and the item is not already present,
        the oldest item is removed (FIFO).
        """
        with self._lock:
            if item in self._data:
                return

            if len(self._data) >= self._capacity:
                # FIFO: remove oldest item
                oldest_item = self._order.popleft()
                self._data.remove(oldest_item)

            self._data.add(item)
            self._order.append(item)

    def __contains__(self, item: object) -> bool:
        """Test for membership using 'in' operator."""
        with self._lock:
            return item in self._data

    def __len__(self) -> int:
        """Return the number of items in the set."""
        with self._lock:
            return len(self._data)

    def __repr__(self) -> str:
        """Return string representation of the set."""
        with self._lock:
            return f"CappedSet({list(self._order)})"

    @property
    def capacity(self) -> int:
        """Get the maximum capacity of the set."""
        return self._capacity
