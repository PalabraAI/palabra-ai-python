from __future__ import annotations

import abc
import asyncio
from dataclasses import KW_ONLY, dataclass, field
from typing import Optional

from palabra_ai.base.audio_frame import AudioFrame

from palabra_ai.base.task import Task
from palabra_ai.base.task_event import TaskEvent
from palabra_ai.constant import CHUNK_SIZE


@dataclass
class Reader(Task):
    """Abstract PCM audio reader process."""

    _: KW_ONLY
    sender: Optional["palabra_ai.task.sender.SenderSourceAudio"] = None  # noqa
    q: asyncio.Queue[bytes] = field(default_factory=asyncio.Queue)
    chunk_size: int = CHUNK_SIZE
    eof: TaskEvent = field(default_factory=TaskEvent, init=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eof.set_owner(f"{self.__class__.__name__}.eof")

    @abc.abstractmethod
    async def read(self, size: int = CHUNK_SIZE) -> bytes | None:
        """Read PCM16 data. Must handle CancelledError."""
        ...


@dataclass
class Writer(Task):
    """Abstract PCM audio writer process."""

    _: KW_ONLY
    q: asyncio.Queue[AudioFrame] = field(default_factory=asyncio.Queue)

    async def _exit(self):
        return await self.exit()
