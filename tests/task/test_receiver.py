import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from palabra_ai.task.receiver import ReceiverTranslatedAudio
from palabra_ai.config import Config
from palabra_ai.lang import Language
from palabra_ai.adapter.dummy import DummyWriter


class TestTaskReceiver:
    @pytest.mark.asyncio
    async def test_boot(self):
        cfg = MagicMock(spec=Config)
        writer = MagicMock()
        rt = MagicMock()

        async def writer_ready(): pass
        async def rt_ready(): pass

        writer.ready = writer_ready()
        rt.ready = rt_ready()
        rt.c = MagicMock()
        rt.c.get_translation_tracks = AsyncMock(return_value={
            "es": MagicMock(start_listening=MagicMock())
        })

        receiver = ReceiverTranslatedAudio(cfg, writer, rt, Language("es"))
        receiver.setup_translation = AsyncMock()

        await receiver.boot()
        receiver.setup_translation.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_translation_success(self):
        cfg = MagicMock(spec=Config)
        writer = DummyWriter()
        rt = MagicMock()

        track = MagicMock()
        track.start_listening = MagicMock()

        rt.c = MagicMock()
        rt.c.get_translation_tracks = AsyncMock(return_value={"es": track})

        receiver = ReceiverTranslatedAudio(cfg, writer, rt, Language("es"))
        await receiver.setup_translation()

        assert receiver._track == track
        track.start_listening.assert_called_once_with(writer.q)

    @pytest.mark.asyncio
    async def test_setup_translation_timeout(self):
        cfg = MagicMock(spec=Config)
        writer = DummyWriter()
        rt = MagicMock()

        rt.c = MagicMock()
        rt.c.get_translation_tracks = AsyncMock(return_value={})

        receiver = ReceiverTranslatedAudio(cfg, writer, rt, Language("es"))

        import palabra_ai.task.receiver
        original_max = palabra_ai.task.receiver.TRACK_RETRY_MAX_ATTEMPTS
        original_delay = palabra_ai.task.receiver.TRACK_RETRY_DELAY

        palabra_ai.task.receiver.TRACK_RETRY_MAX_ATTEMPTS = 2
        palabra_ai.task.receiver.TRACK_RETRY_DELAY = 0.01

        try:
            with pytest.raises(TimeoutError):
                await receiver.setup_translation()
        finally:
            palabra_ai.task.receiver.TRACK_RETRY_MAX_ATTEMPTS = original_max
            palabra_ai.task.receiver.TRACK_RETRY_DELAY = original_delay

    @pytest.mark.asyncio
    async def test_setup_translation_with_stopper(self):
        cfg = MagicMock(spec=Config)
        writer = DummyWriter()
        rt = MagicMock()

        receiver = ReceiverTranslatedAudio(cfg, writer, rt, Language("es"))
        receiver.stopper.set()

        await receiver.setup_translation()
        assert receiver._track is None

    @pytest.mark.asyncio
    async def test_exit_with_track(self):
        cfg = MagicMock(spec=Config)
        writer = DummyWriter()
        rt = MagicMock()

        receiver = ReceiverTranslatedAudio(cfg, writer, rt, Language("es"))

        track = MagicMock()
        track.stop_listening = AsyncMock()
        receiver._track = track

        await receiver.exit()

        track.stop_listening.assert_called_once()
        assert receiver.eof.is_set()
        assert writer.q.get_nowait() is None

    @pytest.mark.asyncio
    async def test_exit_track_timeout(self):
        cfg = MagicMock(spec=Config)
        writer = DummyWriter()
        rt = MagicMock()

        receiver = ReceiverTranslatedAudio(cfg, writer, rt, Language("es"))

        track = MagicMock()
        async def hanging_stop():
            await asyncio.sleep(10)
        track.stop_listening = hanging_stop
        receiver._track = track

        with patch('palabra_ai.constant.SHUTDOWN_TIMEOUT', 0.1):
            await receiver.exit()

        assert receiver.eof.is_set()
        assert receiver._track is None

    @pytest.mark.asyncio
    async def test_do_method(self):
        cfg = MagicMock(spec=Config)
        writer = DummyWriter()
        rt = MagicMock()

        receiver = ReceiverTranslatedAudio(cfg, writer, rt, Language("es"))

        async def stop_soon():
            await asyncio.sleep(0.1)
            receiver.stopper.set()

        await asyncio.gather(
            receiver.do(),
            stop_soon(),
            return_exceptions=True
        )
