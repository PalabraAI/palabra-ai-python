"""Tests for TaskEvent class."""
import asyncio
import pytest

from palabra_ai.base.task_event import TaskEvent


@pytest.mark.asyncio
async def test_pos_neg_operators():
    e = TaskEvent()

    # Test __pos__
    assert not e
    result = +e
    assert e
    assert result is e

    # Test __neg__
    result = -e
    assert not e
    assert result is e


@pytest.mark.asyncio
async def test_bool_returns_is_set():
    e = TaskEvent()
    assert not e
    e.set()
    assert e
    e.clear()
    assert not e


@pytest.mark.asyncio
async def test_await_functionality():
    e = TaskEvent()
    results = []

    # Test waiting
    async def waiter():
        results.append("waiting")
        await e
        results.append("done")

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0.01)
    assert results == ["waiting"]

    +e
    await task
    assert results == ["waiting", "done"]

    # Test immediate return when set
    e.set()
    start = asyncio.get_event_loop().time()
    await e
    elapsed = asyncio.get_event_loop().time() - start
    assert elapsed < 0.001


def test_if_statement():
    e = TaskEvent()

    if e:
        assert False, "Should not enter"
    else:
        assert True

    +e

    if e:
        assert True
    else:
        assert False, "Should enter if block"

    -e

    if e:
        assert False, "Should not enter"
    else:
        assert True, "Should enter else block"


def test_repr():
    e = TaskEvent()
    assert repr(e) == "TaskEvent(False)"
    +e
    assert repr(e) == "TaskEvent(True)"
