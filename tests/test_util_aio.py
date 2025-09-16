import asyncio
import pytest
from palabra_ai.util.aio import boot, shutdown, any_event, all_events, warn_if_cancel
from palabra_ai.constant import BOOT_TIMEOUT, SHUTDOWN_TIMEOUT

async def quick_task():
    """Quick async task that completes immediately"""
    await asyncio.sleep(0.01)
    return "done"

async def slow_task():
    """Slow async task that will timeout"""
    await asyncio.sleep(100)
    return "never"

@pytest.mark.asyncio
async def test_boot_success():
    """Test boot with task that completes in time"""
    result = await boot(quick_task())
    assert result == "done"

@pytest.mark.asyncio
async def test_shutdown_success():
    """Test shutdown with task that completes in time"""
    result = await shutdown(quick_task())
    assert result == "done"

@pytest.mark.asyncio
async def test_any_event():
    """Test any_event waits for first event"""
    event1 = asyncio.Event()
    event2 = asyncio.Event()

    async def set_event1():
        await asyncio.sleep(0.01)
        event1.set()

    # Start task to set event1
    asyncio.create_task(set_event1())

    # Should complete when event1 is set
    await any_event(event1, event2)
    assert event1.is_set()
    assert not event2.is_set()

@pytest.mark.asyncio
async def test_all_events():
    """Test all_events waits for all events"""
    event1 = asyncio.Event()
    event2 = asyncio.Event()

    async def set_events():
        await asyncio.sleep(0.01)
        event1.set()
        await asyncio.sleep(0.01)
        event2.set()

    # Start task to set both events
    asyncio.create_task(set_events())

    # Should complete when both events are set
    await all_events(event1, event2)
    assert event1.is_set()
    assert event2.is_set()

@pytest.mark.asyncio
async def test_warn_if_cancel_success():
    """Test warn_if_cancel with successful task"""
    result = await warn_if_cancel(quick_task(), "Should not see this warning")
    assert result == "done"

@pytest.mark.asyncio
async def test_warn_if_cancel_cancelled():
    """Test warn_if_cancel with cancelled task"""
    async def cancellable_task():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await warn_if_cancel(cancellable_task(), "Task was cancelled")
