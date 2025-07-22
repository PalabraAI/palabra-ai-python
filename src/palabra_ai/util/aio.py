from __future__ import annotations

import asyncio
from asyncio import ALL_COMPLETED, FIRST_COMPLETED, create_task, wait, wait_for

from palabra_ai.constant import BOOT_TIMEOUT, SHUTDOWN_TIMEOUT
from palabra_ai.util.logger import warning


async def boot(fn):
    return await wait_for(fn, timeout=BOOT_TIMEOUT)


async def shutdown(fn):
    return await wait_for(fn, timeout=SHUTDOWN_TIMEOUT)


async def any_event(*events):
    await wait([create_task(e.wait()) for e in events], return_when=FIRST_COMPLETED)


async def all_events(*events):
    await wait([create_task(e.wait()) for e in events], return_when=ALL_COMPLETED)


async def warn_if_cancel(coro, warning_msg: str):
    """Handle cancellation with logging."""
    try:
        return await coro
    except asyncio.CancelledError:
        warning(warning_msg)
        raise
