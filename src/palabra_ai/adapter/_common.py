from __future__ import annotations

import asyncio

from palabra_ai.util.logger import warning


async def warn_if_cancel(coro, warning_msg: str):
    """Handle cancellation with logging."""
    try:
        return await coro
    except asyncio.CancelledError:
        warning(warning_msg)
        raise
