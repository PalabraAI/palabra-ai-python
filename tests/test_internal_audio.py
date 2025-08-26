import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import numpy as np
from palabra_ai.internal.audio import (
    write_to_disk, 
    read_from_disk,
    resample_pcm,
    convert_any_to_pcm16,
    pull_until_blocked
)


@pytest.mark.asyncio
async def test_write_to_disk():
    """Test write_to_disk function"""
    mock_file = AsyncMock()
    mock_file.write.return_value = 10
    
    mock_async_open = AsyncMock()
    mock_async_open.__aenter__.return_value = mock_file
    
    with patch('palabra_ai.internal.audio.async_open', return_value=mock_async_open) as mock_open:
        result = await write_to_disk("test.txt", b"test data")
        assert result == 10
        mock_open.assert_called_once_with("test.txt", "wb")
        mock_file.write.assert_called_once_with(b"test data")


@pytest.mark.asyncio
async def test_write_to_disk_cancelled():
    """Test write_to_disk when cancelled"""
    mock_file = AsyncMock()
    mock_file.write.side_effect = asyncio.CancelledError()
    
    mock_async_open = AsyncMock()
    mock_async_open.__aenter__.return_value = mock_file
    
    with patch('palabra_ai.internal.audio.async_open', return_value=mock_async_open):
        with pytest.raises(asyncio.CancelledError):
            await write_to_disk("test.txt", b"test data")


@pytest.mark.asyncio
async def test_read_from_disk():
    """Test read_from_disk function"""
    mock_file = AsyncMock()
    mock_file.read.return_value = b"test data"
    
    mock_async_open = AsyncMock()
    mock_async_open.__aenter__.return_value = mock_file
    
    with patch('palabra_ai.internal.audio.async_open', return_value=mock_async_open) as mock_open:
        result = await read_from_disk("test.txt")
        assert result == b"test data"
        mock_open.assert_called_once_with("test.txt", "rb")
        mock_file.read.assert_called_once()


@pytest.mark.asyncio
async def test_read_from_disk_cancelled():
    """Test read_from_disk when cancelled"""
    mock_file = AsyncMock()
    mock_file.read.side_effect = asyncio.CancelledError()
    
    mock_async_open = AsyncMock()
    mock_async_open.__aenter__.return_value = mock_file
    
    with patch('palabra_ai.internal.audio.async_open', return_value=mock_async_open):
        with pytest.raises(asyncio.CancelledError):
            await read_from_disk("test.txt")


def test_resample_pcm_mono_to_mono_same_rate():
    """Test resample_pcm with mono to mono, same sample rate"""
    # Create test audio data
    audio_data = np.array([100, 200, 300, 400], dtype=np.int16).tobytes()
    
    result = resample_pcm(audio_data, 16000, 16000, 1, 1)
    
    # Should be unchanged when rates are same
    result_array = np.frombuffer(result, dtype=np.int16)
    assert np.allclose(result_array, [100, 200, 300, 400], atol=1)


def test_resample_pcm_stereo_to_mono():
    """Test resample_pcm with stereo to mono conversion"""
    # Create interleaved stereo data (L, R, L, R)
    audio_data = np.array([100, 200, 300, 400], dtype=np.int16).tobytes()
    
    result = resample_pcm(audio_data, 16000, 16000, 2, 1)
    
    # Should average channels: (100+200)/2=150, (300+400)/2=350
    result_array = np.frombuffer(result, dtype=np.int16)
    assert len(result_array) == 2
    assert np.allclose(result_array, [150, 350], atol=1)


@patch('palabra_ai.internal.audio.librosa.resample')
def test_resample_pcm_different_rates(mock_resample):
    """Test resample_pcm with different sample rates"""
    # Mock librosa resample
    mock_resample.return_value = np.array([0.1, 0.2], dtype=np.float32)
    
    audio_data = np.array([1000, 2000], dtype=np.int16).tobytes()
    
    result = resample_pcm(audio_data, 16000, 8000, 1, 1)
    
    # Check librosa was called
    mock_resample.assert_called_once()
    assert mock_resample.call_args[1]['orig_sr'] == 16000
    assert mock_resample.call_args[1]['target_sr'] == 8000
    
    # Check result
    result_array = np.frombuffer(result, dtype=np.int16)
    assert len(result_array) == 2


@patch('palabra_ai.internal.audio.open_audio_container')
@patch('palabra_ai.internal.audio.create_pcm_output_container')
@patch('palabra_ai.internal.audio.process_audio_frames')
@patch('palabra_ai.internal.audio.flush_filters_and_encoder')
@patch('palabra_ai.internal.audio.time.perf_counter')
def test_convert_any_to_pcm16_simple(mock_time, mock_flush, mock_process, 
                                     mock_create_output, mock_open_container):
    """Test convert_any_to_pcm16 without normalization"""
    mock_time.side_effect = [0.0, 0.1]  # Start and end time
    
    # Mock components
    mock_input_container = MagicMock()
    mock_output_container = MagicMock()
    mock_output_stream = MagicMock()
    mock_output_buffer = MagicMock()
    
    # Setup mocks
    mock_open_container.return_value = (mock_input_container, MagicMock())
    mock_create_output.return_value = (mock_output_container, mock_output_stream)
    mock_process.return_value = 100  # Mock dts value
    mock_output_buffer.read.return_value = b"converted audio"
    
    with patch('palabra_ai.internal.audio.BytesIO', return_value=mock_output_buffer):
        with patch('palabra_ai.internal.audio.av.AudioResampler'):
            result = convert_any_to_pcm16(b"input audio", 16000, "mono", normalize=False)
            
            assert result == b"converted audio"
            mock_open_container.assert_called_once()
            mock_create_output.assert_called_once()
            mock_process.assert_called_once()
            mock_flush.assert_called_once()


def test_pull_until_blocked():
    """Test pull_until_blocked function"""
    mock_graph = MagicMock()
    mock_frames = [MagicMock(), MagicMock()]
    
    # Create a custom exception that behaves like AvBlockingIOError
    class MockBlockingError(Exception):
        pass
    
    # Patch where AvBlockingIOError is used in the except clause
    with patch('palabra_ai.internal.audio.AvBlockingIOError', MockBlockingError):
        # Mock pull to return frames then raise our mock error
        mock_graph.pull.side_effect = [mock_frames[0], mock_frames[1], MockBlockingError("Blocked")]
        
        result = pull_until_blocked(mock_graph)
        
        assert len(result) == 2
        assert result == mock_frames
        assert mock_graph.pull.call_count == 3


def test_pull_until_blocked_ffmpeg_error():
    """Test pull_until_blocked with FFmpeg error"""
    mock_graph = MagicMock()
    
    # Create a custom FFmpeg error
    class MockFFmpegError(Exception):
        pass
    
    # Patch where FFmpegError is used
    with patch('palabra_ai.internal.audio.FFmpegError', MockFFmpegError):
        test_error = MockFFmpegError("Test error")
        mock_graph.pull.side_effect = test_error
        
        with pytest.raises(MockFFmpegError):
            pull_until_blocked(mock_graph)