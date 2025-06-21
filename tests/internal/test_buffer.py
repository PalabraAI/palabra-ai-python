import asyncio
import io
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from palabra_ai.internal.buffer import AudioBufferWriter
from tests.conftest import BaseTaskTest


class TestAudioBufferWriter(BaseTaskTest):
    @pytest.mark.asyncio
    async def test_start_stop(self):
        async with asyncio.TaskGroup() as tg:
            writer = AudioBufferWriter(tg)
            await writer.start()
            assert writer._task is not None
            assert not writer._task.done()

            # Start again - should warn
            await writer.start()

            await writer.stop()

    @pytest.mark.asyncio
    async def test_task_dies_immediately(self):
        with pytest.raises(ExceptionGroup) as exc_info:
            async with asyncio.TaskGroup() as tg:
                writer = AudioBufferWriter(tg)

                async def failing_write():
                    raise Exception("Test error")

                writer._write = failing_write
                await writer.start()
                await asyncio.sleep(0.2)

        assert len(exc_info.value.exceptions) == 1
        assert isinstance(exc_info.value.exceptions[0], Exception)

    @pytest.mark.asyncio
    async def test_write_frames(self, mock_audio_frame):
        async with asyncio.TaskGroup() as tg:
            writer = AudioBufferWriter(tg)
            await writer.start()

            await writer.queue.put(mock_audio_frame)

            # Wait for processing
            timeout = 1.0
            start_time = asyncio.get_event_loop().time()
            while writer._frames_processed == 0:
                await asyncio.sleep(0.01)
                if asyncio.get_event_loop().time() - start_time > timeout:
                    break

            assert writer.buffer.tell() > 0
            assert writer._frames_processed == 1

            await writer.stop()


    @pytest.mark.asyncio
    async def test_drop_empty_frames(self, mock_audio_frame):
        async with asyncio.TaskGroup() as tg:
            writer = AudioBufferWriter(tg, drop_empty_frames=True)
            await writer.start()

            # Empty frame
            empty_frame = MagicMock()
            empty_frame.data.tobytes.return_value = b"\x00" * 100

            await writer.queue.put(empty_frame)
            await asyncio.sleep(0.2)

            assert writer.buffer.tell() == 0

            await writer.stop()

    @pytest.mark.asyncio
    async def test_stop_cancelled(self):
        async with asyncio.TaskGroup() as tg:
            writer = AudioBufferWriter(tg)
            await writer.start()

            writer._task.cancel()
            await writer.stop()
            assert writer._task is None

    @pytest.mark.asyncio
    async def test_write_cancelled(self):
        writer = None
        task_ref = None

        try:
            async with asyncio.TaskGroup() as tg:
                writer = AudioBufferWriter(tg)
                await writer.start()
                task_ref = writer._task

                writer._task.cancel()
                await asyncio.sleep(0.1)
        except* asyncio.CancelledError:
            pass

        assert task_ref and task_ref.cancelled()

    def test_to_wav_bytes(self, mock_audio_frame):
        buffer = io.BytesIO()
        writer = AudioBufferWriter(MagicMock(), buffer=buffer)

        # Without frames
        assert writer.to_wav_bytes() == b""

        # With frame
        writer._frame_sample = mock_audio_frame
        writer.buffer.write(b"test_data")

        result = writer.to_wav_bytes()
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_write_to_disk(self):
        writer = AudioBufferWriter(MagicMock())
        writer._frame_sample = MagicMock(num_channels=1, sample_rate=48000)

        with patch("palabra_ai.internal.buffer.aiofile.async_open") as mock_open:
            mock_file = AsyncMock()
            mock_open.return_value.__aenter__.return_value = mock_file

            await writer.write_to_disk("test.wav")
            mock_file.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_to_disk_cancelled(self):
        writer = AudioBufferWriter(MagicMock())

        with patch("palabra_ai.internal.buffer.aiofile.async_open") as mock_open:
            mock_file = AsyncMock()
            mock_file.write.side_effect = asyncio.CancelledError()
            mock_open.return_value.__aenter__.return_value = mock_file

            with pytest.raises(asyncio.CancelledError):
                await writer.write_to_disk("test.wav")
