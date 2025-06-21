from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from functools import partial

from palabra_ai.base.message import Message, TranscriptionMessage
from palabra_ai.base.task import Task
from palabra_ai.config import Config
from palabra_ai.constant import SLEEP_INTERVAL_DEFAULT
from palabra_ai.task.realtime import Realtime
from palabra_ai.util.capped_set import CappedSet
from palabra_ai.util.logger import debug, error


@dataclass
class Transcription(Task):
    """Processes transcriptions and calls configured callbacks."""

    cfg: Config
    rt: Realtime
    _: KW_ONLY
    suppress_callback_errors: bool = True
    _webrtc_queue: asyncio.Queue | None = field(default=None, init=False)
    _callbacks: dict[str, Callable] = field(default_factory=dict, init=False)
    _dedup: CappedSet[str] = field(default_factory=partial(CappedSet, 100), init=False)

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
        self._webrtc_queue = self.rt.out_foq.subscribe(self, maxsize=0)
        await self.rt.ready
        debug(
            f"Transcription processor started for languages: {list(self._callbacks.keys())}"
        )

    async def do(self):
        while not self.stopper:
            try:
                rt_msg = await asyncio.wait_for(
                    self._webrtc_queue.get(), timeout=SLEEP_INTERVAL_DEFAULT
                )
                if rt_msg is None:
                    debug("Received None from WebRTC queue, stopping...")
                    break
            except TimeoutError:
                continue
            self._webrtc_queue.task_done()
            # Process message
            await self._process_message(rt_msg.msg)

    async def exit(self):
        self.rt.out_foq.unsubscribe(self)

    async def _process_message(self, msg: Message):
        """Process a single message and call appropriate callbacks."""
        try:
            if not isinstance(msg, TranscriptionMessage):
                return

            _dedup = msg.dedup
            if _dedup in self._dedup:
                return
            self._dedup.add(_dedup)

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
