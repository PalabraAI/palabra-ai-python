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
        # For WebSocket mode, just subscribe to audio without waiting for tracks
        from palabra_ai.config import Mode
        
        if self.rt.cfg.mode == Mode.WEBSOCKET:
            debug(f"WebSocket mode: subscribing to audio for {self.target_lang!r}")
            from palabra_ai.base.audio_frame import AudioFrame
            
            async def audio_callback(audio_data: bytes):
                # Convert bytes to AudioFrame
                frame = AudioFrame.from_bytes(audio_data, sample_rate=24000, num_channels=1)
                await self.writer.q.put(frame)
            
            await self.rt.adapter.subscribe_to_audio(
                self.target_lang.code, audio_callback
            )
            info(f"✔️ Now receiving audio for {self.target_lang!r} via WebSocket")
            return
            
        debug(f"Getting translation track for {self.target_lang!r}...")
        for i in range(TRACK_RETRY_MAX_ATTEMPTS):
            if self.stopper:
                debug("ReceiverTranslatedAudio stopped before getting track")
                return

            try:
                debug(
                    f"Attempt {i + 1}/{TRACK_RETRY_MAX_ATTEMPTS} to get translation tracks..."
                )
                # For mixed adapter, use get_translation_tracks method
                if hasattr(self.rt.adapter, 'get_translation_tracks'):
                    tracks = await self.rt.adapter.get_translation_tracks(
                        langs=[self.target_lang.code]
                    )
                else:
                    # For other adapters, just create a simple dict
                    tracks = {self.target_lang.code: None}
                debug(f"Got tracks response: {list(tracks.keys())}")

                if self.target_lang.code in tracks:
                    self._track = tracks[self.target_lang.code]
                    
                    if self._track is not None:
                        # Mixed adapter with RemoteAudioTrack
                        debug(
                            f"Found track for {self.target_lang!r}, starting listening..."
                        )
                        self._track.start_listening(self.writer.q)
                    else:
                        # Other adapters - subscribe to audio
                        debug(
                            f"Subscribing to audio for {self.target_lang!r}..."
                        )
                        from palabra_ai.base.audio_frame import AudioFrame
                        
                        async def audio_callback(audio_data: bytes):
                            # Convert bytes to AudioFrame
                            frame = AudioFrame.from_bytes(audio_data, sample_rate=48000, num_channels=1)
                            await self.writer.q.put(frame)
                        
                        await self.rt.adapter.subscribe_to_audio(
                            self.target_lang.code, audio_callback
                        )
                    
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
        if self._track and hasattr(self._track, 'stop_listening'):
            try:
                await asyncio.wait_for(
                    self._track.stop_listening(), timeout=SHUTDOWN_TIMEOUT
                )
            except TimeoutError:
                error(f"Timeout while stopping track for {self.target_lang!r}")
            self._track = None
        +self.eof  # noqa
        self.writer.q.put_nowait(None)
