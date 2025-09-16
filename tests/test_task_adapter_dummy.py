import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, create_autospec
from palabra_ai.task.adapter.dummy import DummyReader, DummyWriter
from palabra_ai.task.base import TaskEvent
from palabra_ai.audio import AudioFrame
import numpy as np


class TestDummyReader:
    """Test DummyReader class"""

    def test_init_default(self):
        """Test initialization with defaults"""
        reader = DummyReader()
        assert reader.return_data == b""
        assert reader.eof_after_reads is None

    def test_init_with_data(self):
        """Test initialization with custom data"""
        data = b"test data"
        reader = DummyReader(return_data=data, eof_after_reads=5)
        assert reader.return_data == data
        assert reader.eof_after_reads == 5

    @pytest.mark.asyncio
    async def test_boot(self):
        """Test boot method (no-op)"""
        reader = DummyReader()
        await reader.boot()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_read_unlimited(self):
        """Test read with unlimited reads (eof_after_reads=None)"""
        data = b"test data"
        reader = DummyReader(return_data=data)

        # Should always return the same data
        for _ in range(10):
            result = await reader.read()
            assert result == data

        # EOF should not be set
        assert not reader.eof.is_set()

    @pytest.mark.asyncio
    async def test_read_with_limit(self):
        """Test read with limited reads"""
        data = b"test data"
        reader = DummyReader(return_data=data, eof_after_reads=3)

        # First 3 reads should return data
        for i in range(3):
            result = await reader.read()
            assert result == data
            assert reader.eof_after_reads == 2 - i

        # 4th read should return None and set EOF
        result = await reader.read()
        assert result is None
        assert reader.eof.is_set()

    @pytest.mark.asyncio
    async def test_read_with_zero_limit(self):
        """Test read with eof_after_reads=0"""
        reader = DummyReader(return_data=b"data", eof_after_reads=0)

        result = await reader.read()
        assert result is None
        assert reader.eof.is_set()

    @pytest.mark.asyncio
    async def test_do_with_stopper(self):
        """Test do method exits when stopper is set"""
        reader = DummyReader()
        reader.stopper = TaskEvent()

        # Set stopper after a short delay
        async def set_stopper():
            await asyncio.sleep(0.01)
            +reader.stopper

        task = asyncio.create_task(set_stopper())

        # Should exit when stopper is set
        await reader.do()
        await task

    @pytest.mark.asyncio
    async def test_do_with_eof(self):
        """Test do method exits when EOF is set"""
        reader = DummyReader()
        reader.eof = TaskEvent()
        reader.stopper = TaskEvent()

        # Set EOF after a short delay
        async def set_eof():
            await asyncio.sleep(0.01)
            +reader.eof

        task = asyncio.create_task(set_eof())

        # Should exit when EOF is set
        await reader.do()
        await task

    @pytest.mark.asyncio
    async def test_exit_with_eof_not_set(self):
        """Test exit method when EOF is not set"""
        reader = DummyReader()
        reader.eof = TaskEvent()

        with patch('palabra_ai.task.adapter.dummy.debug') as mock_debug:
            await reader.exit()

            mock_debug.assert_called_once()
            assert reader.eof.is_set()

    @pytest.mark.asyncio
    async def test_exit_with_eof_already_set(self):
        """Test exit method when EOF is already set"""
        reader = DummyReader()
        reader.eof = TaskEvent()
        +reader.eof  # Set EOF

        with patch('palabra_ai.task.adapter.dummy.debug') as mock_debug:
            await reader.exit()

            mock_debug.assert_called_once()
            assert reader.eof.is_set()


class TestDummyWriter:
    """Test DummyWriter class"""

    def test_init(self):
        """Test initialization"""
        writer = DummyWriter()
        assert writer.frames_processed == 0
        assert writer._q_reader is None

    @pytest.mark.asyncio
    async def test_boot(self):
        """Test boot method creates q_reader task"""
        writer = DummyWriter()
        writer.sub_tg = MagicMock()
        mock_task = MagicMock()
        writer.sub_tg.create_task.return_value = mock_task

        # Mock the q_reader method to avoid creating actual coroutine
        mock_coro = asyncio.Future()
        mock_coro.set_result(None)
        writer.q_reader = MagicMock(return_value=mock_coro)

        await writer.boot()

        writer.sub_tg.create_task.assert_called_once()
        assert writer._q_reader == mock_task

    @pytest.mark.asyncio
    async def test_write(self):
        """Test write method increments counter"""
        writer = DummyWriter()
        frame = AudioFrame(np.array([1, 2, 3, 4], dtype=np.int16), 16000, 1, 4)

        await writer.write(frame)
        assert writer.frames_processed == 1

        await writer.write(frame)
        assert writer.frames_processed == 2

    @pytest.mark.asyncio
    async def test_q_reader_normal_operation(self):
        """Test q_reader processes frames from queue"""
        writer = DummyWriter()
        writer.q = asyncio.Queue()
        writer.stopper = TaskEvent()
        writer.eof = TaskEvent()

        # Add frames to queue
        frame1 = AudioFrame(np.array([1, 2], dtype=np.int16), 16000, 1, 2)
        frame2 = AudioFrame(np.array([3, 4], dtype=np.int16), 16000, 1, 2)

        await writer.q.put(frame1)
        await writer.q.put(frame2)
        await writer.q.put(None)  # Signal EOF

        # Run q_reader
        await writer.q_reader()

        assert writer.frames_processed == 2
        assert writer.eof.is_set()
        assert writer.stopper.is_set()

    @pytest.mark.asyncio
    async def test_q_reader_cancelled(self):
        """Test q_reader handles cancellation"""
        writer = DummyWriter()
        writer.q = MagicMock()
        writer.q.get = AsyncMock(side_effect=asyncio.CancelledError())
        writer.stopper = TaskEvent()
        writer.eof = TaskEvent()

        with patch('palabra_ai.task.adapter.dummy.debug') as mock_debug:
            await writer.q_reader()

            mock_debug.assert_called_once()
            assert writer.eof.is_set()
            assert writer.stopper.is_set()

    @pytest.mark.asyncio
    async def test_q_reader_exception(self):
        """Test q_reader handles exceptions"""
        writer = DummyWriter()
        writer.q = MagicMock()
        writer.q.get = AsyncMock(side_effect=RuntimeError("Test error"))
        writer.stopper = TaskEvent()
        writer.eof = TaskEvent()

        with patch('palabra_ai.task.adapter.dummy.error') as mock_error:
            await writer.q_reader()

            mock_error.assert_called_once()
            assert "Test error" in str(mock_error.call_args[0][0])
            assert writer.eof.is_set()
            assert writer.stopper.is_set()

    @pytest.mark.asyncio
    async def test_do(self):
        """Test do method waits for EOF or stopper"""
        writer = DummyWriter()
        writer.eof = TaskEvent()
        writer.stopper = TaskEvent()

        # Set EOF after a short delay
        async def set_eof():
            await asyncio.sleep(0.01)
            +writer.eof

        task = asyncio.create_task(set_eof())

        # Should exit when EOF is set
        await writer.do()
        await task

    @pytest.mark.asyncio
    async def test_exit_with_running_task(self):
        """Test exit cancels running q_reader task"""
        writer = DummyWriter()
        mock_task = MagicMock()
        mock_task.done.return_value = False
        writer._q_reader = mock_task

        with patch('palabra_ai.task.adapter.dummy.info') as mock_info:
            await writer.exit()

            mock_task.cancel.assert_called_once()
            mock_info.assert_called_once_with("DummyWriter processed 0 frames")

    @pytest.mark.asyncio
    async def test_exit_with_completed_task(self):
        """Test exit when q_reader task is already done"""
        writer = DummyWriter()
        mock_task = MagicMock()
        mock_task.done.return_value = True
        writer._q_reader = mock_task
        writer.frames_processed = 10

        with patch('palabra_ai.task.adapter.dummy.info') as mock_info:
            await writer.exit()

            mock_task.cancel.assert_not_called()
            mock_info.assert_called_once_with("DummyWriter processed 10 frames")

    @pytest.mark.asyncio
    async def test_exit_no_task(self):
        """Test exit when no q_reader task exists"""
        writer = DummyWriter()
        writer._q_reader = None

        with patch('palabra_ai.task.adapter.dummy.info') as mock_info:
            await writer.exit()

            mock_info.assert_called_once_with("DummyWriter processed 0 frames")
