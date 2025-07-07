from __future__ import annotations

import asyncio
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any

from palabra_ai.base.adapter import Reader
from palabra_ai.base.task import Task
from palabra_ai.config import (
    Config,
)
from palabra_ai.constant import SAFE_PUBLICATION_END_DELAY, TRACK_CLOSE_TIMEOUT
from palabra_ai.adapter.realtime_webrtc import AudioTrackSettings
from palabra_ai.task.realtime import Realtime
from palabra_ai.util.logger import debug, error, warning

if TYPE_CHECKING:
    pass


BYTES_PER_SAMPLE = 2  # PCM16 = 2 bytes per sample


@dataclass
class SenderSourceAudio(Task):
    cfg: Config
    rt: Realtime
    reader: Reader
    translation_settings: dict[str, Any]
    track_settings: AudioTrackSettings
    _: KW_ONLY
    bytes_sent: int = field(default=0, init=False)

    async def boot(self):
        await self.rt.ready

        await self.rt.set_translation_settings(self.translation_settings)

    async def do(self):
        while not self.stopper and not self.eof:
            chunk = await self.reader.read()

            if chunk is None:
                debug(f"T{self.name}: Audio EOF reached")
                +self.eof  # noqa
                break

            if not chunk:
                continue

            self.bytes_sent += len(chunk)
            await self.rt.adapter.publish_audio(chunk)

    async def _exit(self):
        try:
            debug(f"{self.name}._exit()/proto exit() called, waiting for exit...")
            return await asyncio.wait_for(
                self.exit(), timeout=SAFE_PUBLICATION_END_DELAY
            )
        except TimeoutError:
            error(
                f"{self.name}.exit()/proto timed out after {SAFE_PUBLICATION_END_DELAY}s"
            )
            # Cancel all subtasks
            await self.cancel_all_subtasks()
            warning(f"{self.name}.exit()/proto all subtasks cancelled")

    async def exit(self):
        try:
            debug(
                f"T{self.name}: WAITING FOR {SAFE_PUBLICATION_END_DELAY=} seconds"
            )
            await asyncio.sleep(SAFE_PUBLICATION_END_DELAY)
        except asyncio.CancelledError:
            debug(f"T{self.name}: Cancelled during publication end delay")
        debug(f"T{self.name}: Sender exited ({self.bytes_sent} bytes sent)")
        +self.eof  # noqa
