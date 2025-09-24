import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
from palabra_ai.task.adapter.file import FileReader, FileWriter
from palabra_ai.task.base import TaskEvent
from palabra_ai.audio import AudioFrame, AudioBuffer
import tempfile
import os


class TestFileReader:
    """Test FileReader class"""

    def test_init_with_existing_file(self, tmp_path):
        """Test initialization with existing file"""
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"dummy data")

        reader = FileReader(path=test_file)
        assert reader.path == test_file
        assert reader._position == 0
        assert reader._preprocessed == False
        assert reader._buffer is not None

    def test_init_with_string_path(self, tmp_path):
        """Test initialization with string path"""
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"dummy data")

        reader = FileReader(path=str(test_file))
        assert reader.path == test_file

    def test_init_with_nonexistent_file(self):
        """Test initialization with non-existent file"""
        with pytest.raises(FileNotFoundError) as exc_info:
            FileReader(path="/nonexistent/file.wav")

        assert "File not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_boot_success(self, tmp_path):
        """Test boot does nothing - preprocessing happens in do_preprocess"""
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"dummy audio data")

        reader = FileReader(path=test_file)

        with patch('palabra_ai.task.adapter.file.debug') as mock_debug:
            await reader.boot()
            mock_debug.assert_called_once_with("FileReader boot: audio ready for reading")

    def test_do_preprocess_with_preprocessing(self, tmp_path):
        """Test do_preprocess with preprocessing=True"""
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"dummy audio data")

        reader = FileReader(path=test_file, preprocess=True)
        reader.cfg = MagicMock()
        reader.cfg.mode.input_sample_rate = 16000

        with patch.object(reader, '_preprocess_audio') as mock_preprocess:
            with patch('palabra_ai.task.adapter.file.debug') as mock_debug:
                reader.do_preprocess()

                mock_preprocess.assert_called_once()
                assert mock_debug.call_count >= 2

    def test_do_preprocess_streaming_mode(self, tmp_path):
        """Test do_preprocess with preprocessing=False (streaming)"""
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"dummy audio data")

        reader = FileReader(path=test_file, preprocess=False)
        reader.cfg = MagicMock()
        reader.cfg.mode.input_sample_rate = 16000
        reader.cfg.mode.mode_type = "ws"

        with patch('palabra_ai.task.adapter.file.simple_setup_streaming_audio') as mock_setup:
                # Setup mocks for setup_streaming_audio return values
                mock_container = MagicMock()
                mock_resampler = MagicMock()
                mock_metadata = {
                    "original_rate": 8000,
                    "final_rate": 16000,
                    "resampled": True,
                    "duration": 1.0,
                }
                mock_setup.return_value = (mock_container, mock_resampler, 16000, mock_metadata)
                mock_container.decode.return_value = iter([])

                reader.do_preprocess()

                assert reader._container is not None
                assert reader._target_rate == 16000
                mock_setup.assert_called_once_with(test_file, target_rate=16000, timeout=10.0)
    @pytest.mark.asyncio
    async def test_read_preprocessed_mode(self, tmp_path):
        """Test read in preprocessed mode"""
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"dummy")

        reader = FileReader(path=test_file)
        reader._preprocessed = True
        reader._buffer.append(b"test ")
        reader._buffer.append(b"audio data")
        reader.ready = TaskEvent()
        +reader.ready

        # Read first chunk
        chunk = await reader.read(5)
        assert chunk == b"test "
        assert reader._position == 5

        # Read second chunk
        chunk = await reader.read(5)
        assert chunk == b"audio"
        assert reader._position == 10

    @pytest.mark.asyncio
    async def test_read_streaming_mode(self, tmp_path):
        """Test read in streaming mode"""
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"dummy")

        reader = FileReader(path=test_file)
        reader._preprocessed = False
        reader._buffer.append(b"stream data")
        reader.ready = TaskEvent()
        +reader.ready

        with patch.object(reader, '_ensure_buffer_has_data', new_callable=AsyncMock) as mock_ensure:
            chunk = await reader.read(6)
            assert chunk == b"stream"
            assert reader._position == 6
            mock_ensure.assert_called_once_with(6)

    @pytest.mark.asyncio
    async def test_read_at_eof(self, tmp_path):
        """Test read at EOF"""
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"dummy")

        reader = FileReader(path=test_file)
        reader._preprocessed = True
        # Empty buffer = EOF
        reader.ready = TaskEvent()
        +reader.ready
        reader.eof = TaskEvent()

        with patch('palabra_ai.task.adapter.file.debug') as mock_debug:
            chunk = await reader.read(5)
            assert chunk is None
            assert reader.eof.is_set()
            mock_debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_exit_with_stats(self, tmp_path):
        """Test exit logs processing stats"""
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"dummy")

        reader = FileReader(path=test_file)
        reader._position = 16000 * 2 * 5  # 5 seconds at 16kHz, 16-bit
        reader._target_rate = 16000
        reader.duration = 10.0  # 10 seconds total
        reader._container = MagicMock()

        with patch('palabra_ai.task.adapter.file.debug') as mock_debug:
            await reader.exit()

            # Should log processing stats
            debug_calls = [str(call) for call in mock_debug.call_args_list]
            assert any("processed" in call and "5.0s" in call for call in debug_calls)
            assert any("50.0%" in call for call in debug_calls)


class TestFileWriter:
    """Test FileWriter class"""

    def test_init_with_path(self, tmp_path):
        """Test initialization with path"""
        output_file = tmp_path / "output" / "test.wav"

        writer = FileWriter(path=output_file)
        assert writer.path == output_file
        assert writer.delete_on_error is False
        # Parent directory should be created
        assert output_file.parent.exists()

    def test_init_with_delete_on_error(self, tmp_path):
        """Test initialization with delete_on_error flag"""
        output_file = tmp_path / "test.wav"

        writer = FileWriter(path=output_file, delete_on_error=True)
        assert writer.delete_on_error is True

    @pytest.mark.asyncio
    async def test_exit_success(self, tmp_path):
        """Test successful exit with data"""
        output_file = tmp_path / "test.wav"

        writer = FileWriter(path=output_file)
        writer.ab = MagicMock()
        writer.ab.to_wav_bytes.return_value = b"WAV file data"

        with patch('palabra_ai.task.adapter.file.write_to_disk', new_callable=AsyncMock) as mock_write:
            with patch('palabra_ai.task.adapter.file.debug') as mock_debug:
                result = await writer.exit()

                assert result == b"WAV file data"
                writer.ab.to_wav_bytes.assert_called_once()
                mock_write.assert_called_once_with(output_file, b"WAV file data")
                assert mock_debug.call_count >= 3

    @pytest.mark.asyncio
    async def test_exit_no_data(self, tmp_path):
        """Test exit with no data"""
        output_file = tmp_path / "test.wav"

        writer = FileWriter(path=output_file)
        writer.ab = MagicMock()
        writer.ab.to_wav_bytes.return_value = b""

        with patch('palabra_ai.task.adapter.file.warning') as mock_warning:
            with patch('palabra_ai.task.adapter.file.debug'):
                result = await writer.exit()

                assert result == b""
                mock_warning.assert_called_once_with("No WAV data generated")

    @pytest.mark.asyncio
    async def test_exit_cancelled(self, tmp_path):
        """Test exit when cancelled"""
        output_file = tmp_path / "test.wav"
        output_file.write_bytes(b"existing data")

        writer = FileWriter(path=output_file, delete_on_error=True)
        writer.ab = MagicMock()
        writer.ab.to_wav_bytes.side_effect = asyncio.CancelledError()

        with patch('palabra_ai.task.adapter.file.warning') as mock_warning:
            with pytest.raises(asyncio.CancelledError):
                await writer.exit()

            mock_warning.assert_called_once()
            # File should be deleted
            assert not output_file.exists()

    @pytest.mark.asyncio
    async def test_exit_error(self, tmp_path):
        """Test exit with error"""
        output_file = tmp_path / "test.wav"
        output_file.write_bytes(b"existing data")

        writer = FileWriter(path=output_file, delete_on_error=True)
        writer.ab = MagicMock()
        writer.ab.to_wav_bytes.side_effect = RuntimeError("WAV conversion failed")

        with patch('palabra_ai.task.adapter.file.error') as mock_error:
            with pytest.raises(RuntimeError):
                await writer.exit()

            mock_error.assert_called()
            # File should be deleted
            assert not output_file.exists()

    def test_delete_on_error_success(self, tmp_path):
        """Test _delete_on_error when file exists"""
        output_file = tmp_path / "test.wav"
        output_file.write_bytes(b"data")

        writer = FileWriter(path=output_file, delete_on_error=True)

        writer._delete_on_error()

        assert not output_file.exists()

    def test_delete_on_error_no_file(self, tmp_path):
        """Test _delete_on_error when file doesn't exist"""
        output_file = tmp_path / "nonexistent.wav"

        writer = FileWriter(path=output_file, delete_on_error=True)

        # Should not raise error
        writer._delete_on_error()

    def test_delete_on_error_disabled(self, tmp_path):
        """Test _delete_on_error when delete_on_error is False"""
        output_file = tmp_path / "test.wav"
        output_file.write_bytes(b"data")

        writer = FileWriter(path=output_file, delete_on_error=False)

        writer._delete_on_error()

        # File should still exist
        assert output_file.exists()

    def test_delete_on_error_permission_error(self, tmp_path):
        """Test _delete_on_error with permission error"""
        output_file = tmp_path / "test.wav"
        output_file.write_bytes(b"data")

        writer = FileWriter(path=output_file, delete_on_error=True)

        with patch.object(Path, 'unlink', side_effect=PermissionError("No permission")):
            with patch('palabra_ai.task.adapter.file.error') as mock_error:
                with pytest.raises(PermissionError):
                    writer._delete_on_error()

                mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_exit_no_timeout_on_slow_save(self, tmp_path):
        """Test that FileWriter doesn't timeout on slow saves (via UnlimitedExitMixin)"""
        from palabra_ai.constant import SHUTDOWN_TIMEOUT
        import time

        output_file = tmp_path / "test.wav"
        writer = FileWriter(path=output_file)
        writer.ab = MagicMock()

        # Simulate slow save operation (longer than SHUTDOWN_TIMEOUT=5s)
        slow_duration = SHUTDOWN_TIMEOUT + 2

        def slow_save():
            time.sleep(slow_duration)
            return b"WAV data after delay"

        writer.ab.to_wav_bytes = slow_save

        with patch('palabra_ai.task.adapter.file.write_to_disk', new_callable=AsyncMock):
            # Should complete without timeout
            start_time = asyncio.get_event_loop().time()
            result = await writer.exit()
            elapsed = asyncio.get_event_loop().time() - start_time

            assert result == b"WAV data after delay"
            assert elapsed >= slow_duration  # Verify it actually waited
            assert elapsed < slow_duration + 1  # But not much more

    @pytest.mark.asyncio
    async def test_exit_calls_unlimited_exit_mixin(self, tmp_path):
        """Test that FileWriter uses UnlimitedExitMixin._exit (no timeout)"""
        from palabra_ai.task.adapter.base import UnlimitedExitMixin

        output_file = tmp_path / "test.wav"
        writer = FileWriter(path=output_file)

        # Verify FileWriter has UnlimitedExitMixin in MRO
        assert UnlimitedExitMixin in FileWriter.__mro__
        # Verify _exit comes from UnlimitedExitMixin, not Writer
        assert writer._exit.__qualname__.startswith("UnlimitedExitMixin")
