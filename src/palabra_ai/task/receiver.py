from __future__ import annotations

import asyncio
from dataclasses import KW_ONLY, dataclass, field
from typing import Any

from palabra_ai.base.adapter import Writer
from palabra_ai.base.task import Task
from palabra_ai.config import (
    Config,
)
from palabra_ai.constant import (
    SHUTDOWN_TIMEOUT,
    SLEEP_INTERVAL_DEFAULT,
    TRACK_RETRY_DELAY,
    TRACK_RETRY_MAX_ATTEMPTS,
)
from palabra_ai.lang import Language
from palabra_ai.task.realtime import Realtime
from palabra_ai.util.logger import debug, error, info


@dataclass
class ReceiverTranslatedAudio(Task):
    cfg: Config
    writer: Writer
    rt: Realtime
    target_lang: Language
    _: KW_ONLY
    _track: Any = field(default=None, init=False)

    async def boot(self):
        await self.rt.ready
        await self.writer.ready
        await self.setup_translation()

    async def do(self):
        while not self.stopper:
            await asyncio.sleep(SLEEP_INTERVAL_DEFAULT)

    async def setup_translation(self):
        """Get translation track with retries."""
        debug(f"Getting translation track for {self.target_lang!r}...")
        for i in range(TRACK_RETRY_MAX_ATTEMPTS):
            if self.stopper:
                debug("ReceiverTranslatedAudio stopped before getting track")
                return

            try:
                debug(
                    f"Attempt {i + 1}/{TRACK_RETRY_MAX_ATTEMPTS} to get translation tracks..."
                )
                tracks = await self.rt.c.get_translation_tracks(
                    langs=[self.target_lang.code]  # TODO: know more about this
                )
                debug(f"Got tracks response: {list(tracks.keys())}")

                if self.target_lang.code in tracks:
                    self._track = tracks[self.target_lang.code]
                    debug(
                        f"Found track for {self.target_lang!r}, starting listening..."
                    )
                    self._track.start_listening(self.writer.q)
                    info(f"✔️ Now receiving audio for {self.target_lang!r}")
                    return

                debug(f"Track for {self.target_lang!r} not found yet")
            except Exception as e:
                error(f"Error getting tracks: {e}")

            await asyncio.sleep(TRACK_RETRY_DELAY)

        raise TimeoutError(
            f"Track for {self.target_lang!r} not available after {TRACK_RETRY_MAX_ATTEMPTS}s"
        )

    async def exit(self):
        debug("Cleaning up ReceiverTranslatedAudio...")
        if self._track:
            try:
                await asyncio.wait_for(
                    self._track.stop_listening(), timeout=SHUTDOWN_TIMEOUT
                )
            except TimeoutError:
                error(f"Timeout while stopping track for {self.target_lang!r}")
            self._track = None
        +self.eof  # noqa
        self.writer.q.put_nowait(None)
