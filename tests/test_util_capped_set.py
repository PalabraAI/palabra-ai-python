import pytest
from threading import Thread
import time
from palabra_ai.util.capped_set import CappedSet


def test_capped_set_creation():
    """Test CappedSet creation with valid capacity"""
    cs = CappedSet(5)
    assert cs.capacity == 5
    assert len(cs) == 0


def test_capped_set_invalid_capacity():
    """Test CappedSet creation with invalid capacity"""
    with pytest.raises(ValueError) as exc_info:
        CappedSet(0)
    assert "Capacity must be positive" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        CappedSet(-5)
    assert "Capacity must be positive" in str(exc_info.value)


def test_capped_set_add():
    """Test adding items to CappedSet"""
    cs = CappedSet(3)

    cs.add("a")
    assert len(cs) == 1
    assert "a" in cs

    cs.add("b")
    assert len(cs) == 2
    assert "b" in cs

    cs.add("c")
    assert len(cs) == 3
    assert "c" in cs


def test_capped_set_add_duplicate():
    """Test adding duplicate items"""
    cs = CappedSet(3)

    cs.add("a")
    cs.add("a")  # Duplicate
    assert len(cs) == 1


def test_capped_set_fifo_eviction():
    """Test FIFO eviction when capacity is reached"""
    cs = CappedSet(3)

    cs.add("a")
    cs.add("b")
    cs.add("c")
    assert len(cs) == 3

    # Add fourth item, should evict "a"
    cs.add("d")
    assert len(cs) == 3
    assert "a" not in cs  # Oldest was removed
    assert "b" in cs
    assert "c" in cs
    assert "d" in cs


def test_capped_set_contains():
    """Test membership testing"""
    cs = CappedSet(5)

    cs.add("test")
    assert "test" in cs
    assert "missing" not in cs

    # Test with different types
    cs.add(123)
    assert 123 in cs
    assert "123" not in cs


def test_capped_set_repr():
    """Test string representation"""
    cs = CappedSet(3)

    cs.add("x")
    cs.add("y")
    cs.add("z")

    repr_str = repr(cs)
    assert repr_str == "CappedSet(['x', 'y', 'z'])"


def test_capped_set_thread_safety():
    """Test thread safety of CappedSet"""
    cs = CappedSet(100)
    errors = []

    def add_items(start, count):
        try:
            for i in range(start, start + count):
                cs.add(i)
                # Also test contains
                assert i in cs
        except Exception as e:
            errors.append(e)

    # Create multiple threads
    threads = []
    for i in range(5):
        t = Thread(target=add_items, args=(i * 20, 20))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Check no errors occurred
    assert len(errors) == 0
    # Should have 100 items (0-99)
    assert len(cs) == 100


def test_capped_set_capacity_property():
    """Test capacity property is read-only"""
    cs = CappedSet(10)
    assert cs.capacity == 10

    # Try to change capacity (should not be possible)
    with pytest.raises(AttributeError):
        cs.capacity = 20


def test_capped_set_mixed_types():
    """Test CappedSet with mixed types"""
    cs = CappedSet[object](5)

    cs.add("string")
    cs.add(123)
    cs.add(45.6)
    cs.add((1, 2))
    cs.add(None)

    assert len(cs) == 5
    assert "string" in cs
    assert 123 in cs
    assert 45.6 in cs
    assert (1, 2) in cs
    assert None in cs
