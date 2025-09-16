from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field

from palabra_ai.config import Config
from palabra_ai.constant import SLEEP_INTERVAL_DEFAULT
from palabra_ai.message import Message, TranscriptionMessage
from palabra_ai.task.base import Task
from palabra_ai.task.io.base import Io

# from palabra_ai.task.realtime import Realtime
from palabra_ai.util.logger import debug, error


@dataclass
class Transcription(Task):
    """Processes transcriptions and calls configured callbacks."""

    cfg: Config
    io: Io
    _: KW_ONLY
    suppress_callback_errors: bool = True
    _out_q: asyncio.Queue | None = field(default=None, init=False)
    _callbacks: dict[str, Callable] = field(default_factory=dict, init=False)

    def __post_init__(self):
        # Collect callbacks by language
        if self.cfg.source.on_transcription:
            self._callbacks[self.cfg.source.lang.code] = (
                self.cfg.source.on_transcription
            )

        for target in self.cfg.targets:
            if target.on_transcription:
                self._callbacks[target.lang.code] = target.on_transcription

    async def boot(self):
        self._out_q = self.io.out_msg_foq.subscribe(self, maxsize=0).q
        await self.io.ready
        debug(
            f"Transcription processor started for languages: {list(self._callbacks.keys())}"
        )

    async def do(self):
        while not self.stopper:
            try:
                msg = await asyncio.wait_for(
                    self._out_q.get(), timeout=SLEEP_INTERVAL_DEFAULT
                )
                if msg is None:
                    debug("Received None from WebRTC queue, stopping...")
                    break
            except TimeoutError:
                continue
            self._out_q.task_done()
            # Process message
            await self._process_message(msg)

    async def exit(self):
        self.io.out_msg_foq.unsubscribe(self)

    async def _process_message(self, msg: Message):
        """Process a single message and call appropriate callbacks."""
        try:
            if not isinstance(msg, TranscriptionMessage):
                return

            callback = self._callbacks.get(msg.language.code)
            if not callback:
                return

            # Call the callback
            await self._call_callback(callback, msg)

        except Exception as e:
            error(f"Error processing transcription message: {e}")

    async def _call_callback(self, callback: Callable, data: TranscriptionMessage):
        """Call a callback, handling both sync and async callbacks."""
        try:
            if asyncio.iscoroutinefunction(callback):
                self.sub_tg.create_task(callback(data), name="Transcription:callback")
                # await callback(data)
            else:
                # Run sync callback in executor to not block
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, callback, data)

        except Exception as e:
            if self.suppress_callback_errors:
                error(f"Error in transcription callback: {e}")
            else:
                raise
