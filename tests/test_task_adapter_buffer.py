import asyncio
import io
import os
import signal
import subprocess
import threading
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from palabra_ai.task.adapter.buffer import BufferReader, BufferWriter, RunAsPipe
from palabra_ai.task.base import TaskEvent
from palabra_ai.audio import AudioBuffer


class TestBufferReader:
    """Test BufferReader class"""

    def test_init_with_bytesio(self):
        """Test initialization with BytesIO buffer"""
        buffer = io.BytesIO(b"test data")
        reader = BufferReader(buffer=buffer)
        assert reader.buffer == buffer
        assert reader._position == 0
        assert reader._buffer_size == 9

    def test_init_with_runas_pipe(self):
        """Test initialization with RunAsPipe"""
        mock_pipe = MagicMock(spec=RunAsPipe)
        mock_pipe.tell.return_value = 0
        mock_pipe.seek = MagicMock()

        reader = BufferReader(buffer=mock_pipe)
        assert reader.buffer == mock_pipe
        mock_pipe.seek.assert_called()

    @pytest.mark.asyncio
    async def test_boot(self):
        """Test boot method"""
        buffer = io.BytesIO(b"test data")
        reader = BufferReader(buffer=buffer)

        with patch('palabra_ai.task.adapter.buffer.debug') as mock_debug:
            await reader.boot()
            mock_debug.assert_called_once()
            assert "contains 9 bytes" in str(mock_debug.call_args[0][0])

    @pytest.mark.asyncio
    async def test_exit_without_eof(self):
        """Test exit when EOF not reached"""
        buffer = io.BytesIO(b"test data")
        reader = BufferReader(buffer=buffer)
        reader.eof = TaskEvent()

        with patch('palabra_ai.task.adapter.buffer.debug') as mock_debug:
            with patch('palabra_ai.task.adapter.buffer.warning') as mock_warning:
                await reader.exit()

                mock_debug.assert_called_once()
                mock_warning.assert_called_once()
                assert "stopped without reaching EOF" in str(mock_warning.call_args[0][0])

    @pytest.mark.asyncio
    async def test_exit_with_eof(self):
        """Test exit when EOF reached"""
        buffer = io.BytesIO(b"test data")
        reader = BufferReader(buffer=buffer)
        reader.eof = TaskEvent()
        +reader.eof

        with patch('palabra_ai.task.adapter.buffer.debug') as mock_debug:
            with patch('palabra_ai.task.adapter.buffer.warning') as mock_warning:
                await reader.exit()

                mock_debug.assert_called_once()
                mock_warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_read_with_data(self):
        """Test reading data from buffer"""
        buffer = io.BytesIO(b"test data here")
        reader = BufferReader(buffer=buffer)
        reader.ready = TaskEvent()
        +reader.ready

        # Read first chunk
        data = await reader.read(5)
        assert data == b"test "
        assert reader._position == 5

        # Read second chunk
        data = await reader.read(4)
        assert data == b"data"
        assert reader._position == 9

    @pytest.mark.asyncio
    async def test_read_at_eof(self):
        """Test reading at EOF"""
        buffer = io.BytesIO(b"test")
        reader = BufferReader(buffer=buffer)
        reader.ready = TaskEvent()
        +reader.ready
        reader._position = 4  # At end

        with patch('palabra_ai.task.adapter.buffer.debug') as mock_debug:
            data = await reader.read(5)
            assert data is None
            assert reader.eof.is_set()
            mock_debug.assert_called_once()
            assert "EOF reached" in str(mock_debug.call_args[0][0])


class TestBufferWriter:
    """Test BufferWriter class"""

    def test_init(self):
        """Test initialization"""
        buffer = io.BytesIO()
        writer = BufferWriter(buffer=buffer)
        assert writer.buffer == buffer

    @pytest.mark.asyncio
    async def test_boot(self):
        """Test boot method"""
        buffer = io.BytesIO()
        writer = BufferWriter(buffer=buffer)
        writer.ab = MagicMock(spec=AudioBuffer)
        writer.ab.replace_buffer = MagicMock()

        # Mock super().boot()
        with patch('palabra_ai.task.adapter.buffer.super') as mock_super:
            mock_super_obj = MagicMock()
            mock_super.return_value = mock_super_obj
            mock_super_obj.boot = AsyncMock()

            await writer.boot()

            mock_super_obj.boot.assert_called_once()
            writer.ab.replace_buffer.assert_called_once_with(buffer)

    @pytest.mark.asyncio
    async def test_exit(self):
        """Test exit method"""
        buffer = io.BytesIO()
        writer = BufferWriter(buffer=buffer)
        writer.ab = MagicMock()
        writer.ab.to_wav_bytes = MagicMock(return_value=b"WAV data")

        with patch('palabra_ai.task.adapter.buffer.debug'):
            await writer.exit()
            writer.ab.to_wav_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_exit_no_timeout_on_slow_save(self):
        """Test that BufferWriter doesn't timeout on slow saves (via UnlimitedExitMixin)"""
        from palabra_ai.constant import SHUTDOWN_TIMEOUT

        buffer = io.BytesIO()
        writer = BufferWriter(buffer=buffer)
        writer.ab = MagicMock()

        # Simulate slow save operation (longer than SHUTDOWN_TIMEOUT=5s)
        slow_duration = SHUTDOWN_TIMEOUT + 2

        def slow_save():
            import time
            time.sleep(slow_duration)
            return b"WAV after delay"

        writer.ab.to_wav_bytes = slow_save

        with patch('palabra_ai.task.adapter.buffer.debug'):
            # Should complete without timeout
            start_time = asyncio.get_event_loop().time()
            await writer.exit()
            elapsed = asyncio.get_event_loop().time() - start_time

            # Verify it waited for the slow operation
            assert elapsed >= slow_duration
            assert elapsed < slow_duration + 1


class TestRunAsPipe:
    """Test RunAsPipe class"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        # Store original values
        original_active = RunAsPipe._active_processes.copy()
        original_registered = RunAsPipe._cleanup_registered

        # Clear for test
        RunAsPipe._active_processes.clear()
        RunAsPipe._cleanup_registered = False

        yield

        # Restore
        RunAsPipe._active_processes = original_active
        RunAsPipe._cleanup_registered = original_registered

    def test_init_first_instance(self):
        """Test initialization of first RunAsPipe instance"""
        with patch('subprocess.Popen') as mock_popen:
            with patch('atexit.register') as mock_atexit:
                with patch('signal.signal') as mock_signal:
                    mock_proc = MagicMock()
                    mock_proc.stdout.read.return_value = b""
                    mock_proc.poll.return_value = 0
                    mock_popen.return_value = mock_proc

                    pipe = RunAsPipe(['echo', 'test'])

                    assert pipe.cmd == ['echo', 'test']
                    assert pipe.process == mock_proc
                    assert pipe._pos == 0
                    assert not pipe._closed

                    # Verify cleanup registration
                    mock_atexit.assert_called_once()
                    assert mock_signal.call_count == 2  # SIGINT and SIGTERM
                    assert RunAsPipe._cleanup_registered is True

    def test_init_second_instance(self):
        """Test initialization of second RunAsPipe instance"""
        # Set as already registered
        RunAsPipe._cleanup_registered = True

        with patch('subprocess.Popen') as mock_popen:
            with patch('atexit.register') as mock_atexit:
                with patch('signal.signal') as mock_signal:
                    mock_proc = MagicMock()
                    mock_proc.stdout.read.return_value = b""
                    mock_proc.poll.return_value = 0
                    mock_popen.return_value = mock_proc

                    pipe = RunAsPipe(['echo', 'test'])

                    # Should not register again
                    mock_atexit.assert_not_called()
                    mock_signal.assert_not_called()

    def test_start_process(self):
        """Test _start_process method"""
        with patch('subprocess.Popen') as mock_popen:
            with patch('threading.Thread') as mock_thread:
                mock_proc = MagicMock()
                mock_popen.return_value = mock_proc
                mock_thread_instance = MagicMock()
                mock_thread.return_value = mock_thread_instance

                pipe = RunAsPipe(['ls'])

                # Verify process started
                mock_popen.assert_called_once()
                assert pipe.process in RunAsPipe._active_processes

                # Verify reader thread started
                mock_thread.assert_called_once()
                mock_thread_instance.start.assert_called_once()

    def test_read_pipe_thread(self):
        """Test _read_pipe background thread"""
        pipe = RunAsPipe.__new__(RunAsPipe)
        pipe._closed = False
        pipe._buffer = bytearray()
        pipe._lock = threading.Lock()

        mock_proc = MagicMock()
        mock_proc.stdout.read.side_effect = [b"data1", b"data2", b""]
        mock_proc.poll.side_effect = [None, None, 0]
        pipe.process = mock_proc

        # Run the reader thread
        pipe._read_pipe()

        # Verify data was read
        assert bytes(pipe._buffer) == b"data1data2"

    def test_read_pipe_thread_exception(self):
        """Test _read_pipe handles exceptions"""
        pipe = RunAsPipe.__new__(RunAsPipe)
        pipe._closed = False
        pipe._buffer = bytearray()
        pipe._lock = threading.Lock()

        mock_proc = MagicMock()
        mock_proc.stdout.read.side_effect = RuntimeError("Read error")
        mock_proc.poll.return_value = None
        pipe.process = mock_proc

        # Should not raise
        pipe._read_pipe()

    def test_read_all(self):
        """Test read with size=-1"""
        pipe = RunAsPipe.__new__(RunAsPipe)
        pipe._buffer = bytearray(b"hello world")
        pipe._pos = 6
        pipe._lock = threading.Lock()

        data = pipe.read(-1)
        assert data == b"world"
        assert pipe._pos == 11

    def test_read_size(self):
        """Test read with specific size"""
        pipe = RunAsPipe.__new__(RunAsPipe)
        pipe._buffer = bytearray(b"hello world")
        pipe._pos = 0
        pipe._lock = threading.Lock()

        data = pipe.read(5)
        assert data == b"hello"
        assert pipe._pos == 5

    def test_seek_set(self):
        """Test seek with SEEK_SET"""
        pipe = RunAsPipe.__new__(RunAsPipe)
        pipe._buffer = bytearray(b"hello world")
        pipe._pos = 5
        pipe._lock = threading.Lock()

        pos = pipe.seek(8, 0)  # SEEK_SET
        assert pos == 8
        assert pipe._pos == 8

    def test_seek_cur(self):
        """Test seek with SEEK_CUR"""
        pipe = RunAsPipe.__new__(RunAsPipe)
        pipe._buffer = bytearray(b"hello world")
        pipe._pos = 5
        pipe._lock = threading.Lock()

        pos = pipe.seek(3, 1)  # SEEK_CUR
        assert pos == 8
        assert pipe._pos == 8

    def test_seek_end(self):
        """Test seek with SEEK_END"""
        pipe = RunAsPipe.__new__(RunAsPipe)
        pipe._buffer = bytearray(b"hello world")
        pipe._pos = 5
        pipe._lock = threading.Lock()

        pos = pipe.seek(-3, 2)  # SEEK_END
        assert pos == 8  # 11 - 3
        assert pipe._pos == 8

    def test_tell(self):
        """Test tell method"""
        pipe = RunAsPipe.__new__(RunAsPipe)
        pipe._pos = 42

        assert pipe.tell() == 42

    def test_cleanup(self):
        """Test _cleanup method"""
        pipe = RunAsPipe.__new__(RunAsPipe)
        pipe._closed = False

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        pipe.process = mock_proc

        RunAsPipe._active_processes = [mock_proc]

        pipe._cleanup()

        assert pipe._closed is True
        assert mock_proc not in RunAsPipe._active_processes
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=2)

    def test_cleanup_already_closed(self):
        """Test _cleanup when already closed"""
        pipe = RunAsPipe.__new__(RunAsPipe)
        pipe._closed = True

        mock_proc = MagicMock()
        pipe.process = mock_proc

        pipe._cleanup()

        # Should not do anything
        mock_proc.terminate.assert_not_called()

    def test_cleanup_timeout(self):
        """Test _cleanup with timeout"""
        pipe = RunAsPipe.__new__(RunAsPipe)
        pipe._closed = False

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        # First wait() throws timeout, second wait() succeeds
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired('cmd', 2), None]
        pipe.process = mock_proc

        RunAsPipe._active_processes = [mock_proc]

        pipe._cleanup()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert mock_proc.wait.call_count == 2

    def test_cleanup_all(self):
        """Test _cleanup_all static method"""
        mock_proc1 = MagicMock()
        mock_proc1.poll.return_value = None
        mock_proc2 = MagicMock()
        mock_proc2.poll.return_value = 0  # Already terminated

        RunAsPipe._active_processes = [mock_proc1, mock_proc2]

        RunAsPipe._cleanup_all()

        mock_proc1.terminate.assert_called_once()
        mock_proc2.terminate.assert_not_called()
        assert len(RunAsPipe._active_processes) == 0

    def test_signal_handler(self):
        """Test _signal_handler static method"""
        with patch('os.kill') as mock_kill:
            with patch('signal.signal') as mock_signal:
                # Mock some processes
                RunAsPipe._active_processes = [MagicMock()]

                RunAsPipe._signal_handler(signal.SIGINT, None)

                # Should cleanup and re-raise signal
                assert len(RunAsPipe._active_processes) == 0
                mock_signal.assert_called_once_with(signal.SIGINT, signal.SIG_DFL)
                mock_kill.assert_called_once()

    def test_del(self):
        """Test __del__ method"""
        pipe = RunAsPipe.__new__(RunAsPipe)
        pipe._closed = False
        pipe.process = MagicMock()
        pipe._cleanup = MagicMock()

        pipe.__del__()

        pipe._cleanup.assert_called_once()
