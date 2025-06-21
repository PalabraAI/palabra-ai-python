import asyncio
import time
import typing as tp
from functools import partial

from livekit import rtc

from ..util.logger import debug, error
from .webrtc import (
    _PALABRA_TRANSLATOR_PARTICIPANT_IDENTITY_PREFIX,
    _PALABRA_TRANSLATOR_TRACK_NAME_PREFIX,
    AudioPublication,
    AudioTrackSettings,
    RoomClient,
)
from .ws import WebSocketClient


class RemoteAudioTrack:
    def __init__(
        self,
        tg: asyncio.TaskGroup,
        lang: str,
        participant: rtc.RemoteParticipant,
        publication: rtc.RemoteTrackPublication,
    ):
        self.tg = tg
        self.lang = lang
        self.participant = participant
        self.publication = publication
        self._listen_task = None

    def start_listening(self, q: asyncio.Queue[rtc.AudioFrame]):
        if not self._listen_task:
            self._listen_task = self.tg.create_task(self.listen(q), name="Rt:listen")

    async def listen(self, q: asyncio.Queue[rtc.AudioFrame]):
        stream = rtc.AudioStream(self.publication.track)
        try:
            async for frame in stream:
                frame: rtc.AudioFrameEvent
                try:
                    await q.put(frame.frame)
                    await asyncio.sleep(0)
                except asyncio.CancelledError:
                    debug(f"RemoteAudioTrack {self.lang} listen cancelled during put")
                    raise
        except asyncio.CancelledError:
            debug(f"Cancelled listening to {self.lang} track")
            raise
        finally:
            q.put_nowait(None)
            debug(f"Closing {self.lang} stream")
            try:
                await stream.aclose()
            except asyncio.CancelledError:
                debug(f"RemoteAudioTrack {self.lang} stream close cancelled")
            except Exception as e:
                error(f"Error closing {self.lang} stream: {e}")
            debug(f"Closed {self.lang} stream")
            self._listen_task = None

    async def stop_listening(self):
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass


class PalabraRTClient:
    def __init__(
        self,
        tg: asyncio.TaskGroup,
        jwt_token: str,
        control_url: str,
        stream_url: str,
    ):
        self.tg = tg
        self._jwt_token = jwt_token
        self._control_url = control_url
        self._stream_url = stream_url
        self.wsc = WebSocketClient(tg=tg, uri=self._control_url, token=self._jwt_token)
        self.room = RoomClient()

    async def connect(self):
        try:
            self.wsc.connect()
            await self.room.connect(url=self._stream_url, token=self._jwt_token)
        except asyncio.CancelledError:
            debug("PalabraRTClient connect cancelled")
            raise

    async def new_translated_publication(
        self,
        translation_settings: dict[str, tp.Any],
        track_settings: AudioTrackSettings | None = None,
    ) -> AudioPublication:
        track_settings = track_settings or AudioTrackSettings()
        try:
            await self.wsc.send(
                {"message_type": "set_task", "data": translation_settings}
            )
            return await self.room.new_publication(track_settings)
        except asyncio.CancelledError:
            debug("PalabraRTClient new_translated_publication cancelled")
            raise

    async def get_translation_settings(
        self, timeout: int | None = None
    ) -> dict[str, tp.Any]:
        start = time.perf_counter()
        while True:
            try:
                debug("PalabraRTClient get_translation_settings sending request")
                await self.wsc.send({"message_type": "get_task", "data": {}})
            except asyncio.CancelledError:
                debug("PalabraRTClient get_translation_settings send cancelled")
                raise

            if timeout and time.perf_counter() - start > timeout:
                raise TimeoutError("Timeout waiting for translation cfg")

            try:
                message = await self.wsc.receive(1)
            except asyncio.CancelledError:
                debug("PalabraRTClient get_translation_settings receive cancelled")
                raise

            if message is None:
                try:
                    await asyncio.sleep(0)
                except asyncio.CancelledError:
                    debug("PalabraRTClient get_translation_settings sleep cancelled")
                    raise
                continue

            if message["message_type"] == "current_task":
                self.wsc.mark_received()
                return message["data"]

            await asyncio.sleep(0)

    async def get_translation_languages(self, timeout: int | None = None) -> list[str]:
        _get_trans_settings = self.get_translation_settings
        if timeout:
            _get_trans_settings = partial(_get_trans_settings, timeout=timeout)
        try:
            translation_settings = await _get_trans_settings()
        except asyncio.CancelledError:
            debug("PalabraRTClient get_translation_languages cancelled")
            raise
        return [
            translation["target_language"]
            for translation in translation_settings["pipeline"]["translations"]
        ]

    async def get_translation_tracks(
        self, langs: list[str] | None = None
    ) -> dict[str, RemoteAudioTrack]:
        response = {}
        try:
            langs = langs or await self.get_translation_languages()
            participant = await self.room.wait_for_participant_join(
                _PALABRA_TRANSLATOR_PARTICIPANT_IDENTITY_PREFIX
            )
            for lang in langs:
                publication = await self.room.wait_for_track_publish(
                    participant, _PALABRA_TRANSLATOR_TRACK_NAME_PREFIX + lang
                )
                response[lang] = RemoteAudioTrack(
                    self.tg, lang, participant, publication
                )
        except asyncio.CancelledError:
            debug("PalabraRTClient get_translation_tracks cancelled")
            raise
        return response

    async def close(self):
        try:
            if self.room:
                await self.room.close()
            if self.wsc:
                await self.wsc.close()
        except asyncio.CancelledError:
            debug("PalabraRTClient close cancelled, forcing close")
            try:
                if self.room:
                    await asyncio.wait_for(self.room.close(), timeout=1.0)
                if self.wsc:
                    await asyncio.wait_for(self.wsc.close(), timeout=1.0)
            except (TimeoutError, asyncio.CancelledError):
                error("PalabraRTClient force close failed")
        except Exception as e:
            error(f"Error closing PalabraRTClient: {e}")
