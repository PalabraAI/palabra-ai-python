import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from palabra_ai.internal.realtime import RemoteAudioTrack, PalabraRTClient


class TestRemoteAudioTrack:
    @pytest.mark.asyncio
    async def test_stop_listening_no_task(self):
        async with asyncio.TaskGroup() as tg:
            track = RemoteAudioTrack(tg, "en", MagicMock(), MagicMock())
            await track.stop_listening()  # Should not raise


class TestPalabraRTClient:
    @pytest.mark.asyncio
    async def test_connect_cancelled(self):
        async with asyncio.TaskGroup() as tg:
            client = PalabraRTClient(tg, "token", "wss://control", "wss://stream")

            client.wsc = MagicMock()
            client.wsc.connect = MagicMock(side_effect=asyncio.CancelledError)

            with pytest.raises(asyncio.CancelledError):
                await client.connect()

    @pytest.mark.asyncio
    async def test_new_translated_publication_cancelled(self):
        async with asyncio.TaskGroup() as tg:
            client = PalabraRTClient(tg, "token", "wss://control", "wss://stream")

            client.wsc = MagicMock()
            client.wsc.send = AsyncMock(side_effect=asyncio.CancelledError)

            with pytest.raises(asyncio.CancelledError):
                await client.new_translated_publication({})

    @pytest.mark.asyncio
    async def test_get_translation_settings_timeout(self):
        async with asyncio.TaskGroup() as tg:
            client = PalabraRTClient(tg, "token", "wss://control", "wss://stream")

            client.wsc = MagicMock()
            client.wsc.send = AsyncMock()
            client.wsc.receive = AsyncMock(return_value=None)

            with pytest.raises(TimeoutError):
                await client.get_translation_settings(timeout=0.1)

    @pytest.mark.asyncio
    async def test_get_translation_languages_cancelled(self):
        async with asyncio.TaskGroup() as tg:
            client = PalabraRTClient(tg, "token", "wss://control", "wss://stream")

            client.get_translation_settings = AsyncMock(side_effect=asyncio.CancelledError)

            with pytest.raises(asyncio.CancelledError):
                await client.get_translation_languages()

    @pytest.mark.asyncio
    async def test_close_error(self):
        async with asyncio.TaskGroup() as tg:
            client = PalabraRTClient(tg, "token", "wss://control", "wss://stream")

            client.room = MagicMock()
            client.room.close = AsyncMock(side_effect=Exception("Test error"))

            client.wsc = MagicMock()
            client.wsc.close = AsyncMock()

            # Should handle error gracefully
            await client.close()
