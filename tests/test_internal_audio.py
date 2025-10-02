import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import numpy as np
from palabra_ai.internal.audio import (
    write_to_disk,
    read_from_disk,
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


# Add tests for uncovered functions
@patch('palabra_ai.internal.audio.av.open')
def test_open_audio_container(mock_av_open):
    """Test open_audio_container function"""
    from palabra_ai.internal.audio import open_audio_container

    # Mock container with audio stream
    mock_container = MagicMock()
    mock_audio_stream = MagicMock()
    mock_audio_stream.type = "audio"
    mock_container.streams = [mock_audio_stream]
    mock_av_open.return_value = mock_container

    container, audio_stream = open_audio_container("test.wav")

    assert container == mock_container
    assert audio_stream == mock_audio_stream
    mock_av_open.assert_called_once_with("test.wav", timeout=None, metadata_errors="ignore")


@patch('palabra_ai.internal.audio.av.open')
def test_open_audio_container_no_audio_streams(mock_av_open):
    """Test open_audio_container with no audio streams"""
    from palabra_ai.internal.audio import open_audio_container

    # Mock container with no audio streams
    mock_container = MagicMock()
    mock_video_stream = MagicMock()
    mock_video_stream.type = "video"
    mock_container.streams = [mock_video_stream]
    mock_av_open.return_value = mock_container

    with pytest.raises(ValueError, match="No audio streams found"):
        open_audio_container("test.mp4")

    mock_container.close.assert_called_once()


def test_get_audio_stream_info():
    """Test get_audio_stream_info function"""
    from palabra_ai.internal.audio import get_audio_stream_info

    # Mock audio stream
    mock_stream = MagicMock()
    mock_stream.duration = 1000
    mock_stream.time_base = 0.001  # 1ms
    mock_stream.codec.name = "pcm_s16le"
    mock_stream.sample_rate = 16000
    mock_stream.channels = 1

    info = get_audio_stream_info(mock_stream)

    assert info["duration"] == 1.0  # 1000 * 0.001
    assert info["codec"] == "pcm_s16le"
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1


def test_get_audio_stream_info_no_duration():
    """Test get_audio_stream_info with no duration"""
    from palabra_ai.internal.audio import get_audio_stream_info

    # Mock audio stream with no duration
    mock_stream = MagicMock()
    mock_stream.duration = None
    mock_stream.codec.name = "mp3"
    mock_stream.sample_rate = 44100
    mock_stream.channels = 2

    info = get_audio_stream_info(mock_stream)

    assert info["duration"] == 0
    assert info["codec"] == "mp3"
    assert info["sample_rate"] == 44100
    assert info["channels"] == 2


@patch('palabra_ai.internal.audio.FilterGraph')
def test_create_normalization_filter_graph(mock_filter_graph_class):
    """Test create_normalization_filter_graph function"""
    from palabra_ai.internal.audio import create_normalization_filter_graph

    # Mock filter graph and components
    mock_graph = MagicMock()
    mock_buffer = MagicMock()
    mock_loudnorm = MagicMock()
    mock_speechnorm = MagicMock()
    mock_sink = MagicMock()

    mock_filter_graph_class.return_value = mock_graph
    mock_graph.add_abuffer.return_value = mock_buffer

    # Set up add method to return different objects in sequence
    add_calls = [mock_loudnorm, mock_speechnorm, mock_sink]
    mock_graph.add.side_effect = add_calls

    result = create_normalization_filter_graph("s16", 16000, "mono", 0.001)

    assert len(result) == 3
    assert result[0] == mock_graph
    assert result[1] == mock_buffer
    assert result[2] == mock_sink
    mock_graph.configure.assert_called_once()


@patch('palabra_ai.internal.audio.av.open')
def test_create_pcm_output_container(mock_av_open):
    """Test create_pcm_output_container function"""
    from palabra_ai.internal.audio import create_pcm_output_container
    from fractions import Fraction

    # Mock container and stream
    mock_container = MagicMock()
    mock_stream = MagicMock()
    mock_container.add_stream.return_value = mock_stream
    mock_av_open.return_value = mock_container

    mock_buffer = MagicMock()

    container, stream = create_pcm_output_container(mock_buffer, 16000, "mono")

    assert container == mock_container
    assert stream == mock_stream
    mock_av_open.assert_called_once_with(mock_buffer, mode="w", format="s16le")
    mock_container.add_stream.assert_called_once_with("pcm_s16le", rate=16000)
    assert stream.layout == "mono"
    assert stream.time_base == Fraction(1, 16000)


def test_process_audio_frames():
    """Test process_audio_frames function"""
    from palabra_ai.internal.audio import process_audio_frames

    # Mock components
    mock_container = MagicMock()
    mock_stream = MagicMock()
    mock_resampler = MagicMock()

    # Mock frame and packet
    mock_frame = MagicMock()
    mock_frame.samples = 1024
    mock_resampled_frame = MagicMock()
    mock_resampled_frame.samples = 1024
    mock_packet = MagicMock()

    # Setup decode and resample
    mock_container.decode.return_value = [mock_frame]
    mock_resampler.resample.return_value = [mock_resampled_frame]
    mock_stream.encode.return_value = [mock_packet]
    mock_stream.container = MagicMock()

    dts = process_audio_frames(mock_container, mock_stream, mock_resampler)

    assert dts == 1024
    mock_resampler.resample.assert_called_once_with(mock_frame)
    mock_stream.encode.assert_called_once_with(mock_resampled_frame)
    mock_stream.container.mux.assert_called_once_with(mock_packet)


def test_flush_filters_and_encoder():
    """Test flush_filters_and_encoder function"""
    from palabra_ai.internal.audio import flush_filters_and_encoder

    # Mock components
    mock_buffer = MagicMock()
    mock_sink = MagicMock()
    mock_stream = MagicMock()
    mock_stream.container = MagicMock()

    # Mock frame and packet
    mock_frame = MagicMock()
    mock_frame.samples = 512
    mock_packet = MagicMock()

    # Setup encoder
    mock_stream.encode.return_value = [mock_packet]

    # Create custom AvEOFError class
    class MockAvEOFError(Exception):
        pass

    with patch('palabra_ai.internal.audio.AvEOFError', MockAvEOFError):
        with patch('palabra_ai.internal.audio.AvBlockingIOError', Exception):
            # Mock filter flush - first return frame, then EOF
            mock_sink.pull.side_effect = [mock_frame, MockAvEOFError()]

            dts = flush_filters_and_encoder(mock_buffer, mock_sink, mock_stream, 0)

            assert dts == 512
            mock_buffer.push.assert_called_with(None)
            mock_stream.encode.assert_called()
            mock_stream.container.mux.assert_called()


@patch('palabra_ai.internal.audio.open_audio_container')
@patch('palabra_ai.internal.audio.create_pcm_output_container')
@patch('palabra_ai.internal.audio.create_normalization_filter_graph')
def test_convert_any_to_pcm16_with_normalization(mock_create_filter, mock_create_output, mock_open_container):
    """Test convert_any_to_pcm16 with normalization enabled"""
    from palabra_ai.internal.audio import convert_any_to_pcm16

    # Mock all components
    mock_input_container = MagicMock()
    mock_audio_stream = MagicMock()
    mock_audio_stream.format.name = "s16"
    mock_audio_stream.rate = 16000
    mock_audio_stream.layout = "mono"
    mock_audio_stream.time_base = 0.001

    mock_output_container = MagicMock()
    mock_output_stream = MagicMock()

    mock_filter_graph = MagicMock()
    mock_filter_buffer = MagicMock()
    mock_filter_sink = MagicMock()

    # Setup returns
    mock_open_container.return_value = (mock_input_container, mock_audio_stream)
    mock_create_output.return_value = (mock_output_container, mock_output_stream)
    mock_create_filter.return_value = (mock_filter_graph, mock_filter_buffer, mock_filter_sink)

    mock_output_buffer = MagicMock()
    mock_output_buffer.read.return_value = b"normalized audio"

    with patch('palabra_ai.internal.audio.BytesIO', return_value=mock_output_buffer):
        with patch('palabra_ai.internal.audio.av.AudioResampler'):
            with patch('palabra_ai.internal.audio.process_audio_frames', return_value=100):
                with patch('palabra_ai.internal.audio.flush_filters_and_encoder'):
                    result = convert_any_to_pcm16(b"input", 16000, "mono", normalize=True)

                    assert result == b"normalized audio"
                    mock_create_filter.assert_called_once()


def test_convert_any_to_pcm16_ffmpeg_error():
    """Test convert_any_to_pcm16 with FFmpeg error"""
    from palabra_ai.internal.audio import convert_any_to_pcm16

    # Create custom FFmpegError
    class MockFFmpegError(Exception):
        pass

    with patch('palabra_ai.internal.audio.FFmpegError', MockFFmpegError):
        with patch('palabra_ai.internal.audio.open_audio_container', side_effect=MockFFmpegError("Test error")):
            with pytest.raises(MockFFmpegError):
                convert_any_to_pcm16(b"input", 16000)


class TestPullUntilBlocked:
    """Test pull_until_blocked function"""

    def test_pull_success(self):
        """Test successful frame pulling"""
        mock_graph = MagicMock()
        mock_frame1 = MagicMock()
        mock_frame2 = MagicMock()

        # Mock to return two frames then block
        from av.error import BlockingIOError as AvBlockingIOError
        mock_graph.pull.side_effect = [mock_frame1, mock_frame2, AvBlockingIOError("test", "test", "test")]

        result = pull_until_blocked(mock_graph)

        assert len(result) == 2
        assert result[0] == mock_frame1
        assert result[1] == mock_frame2

    def test_pull_ffmpeg_error(self):
        """Test FFmpeg error propagation"""
        from av.error import FFmpegError

        mock_graph = MagicMock()
        mock_graph.pull.side_effect = FFmpegError("Test error", "test")

        with pytest.raises(FFmpegError):
            pull_until_blocked(mock_graph)

    def test_pull_immediate_block(self):
        """Test immediate blocking"""
        from av.error import BlockingIOError as AvBlockingIOError

        mock_graph = MagicMock()
        mock_graph.pull.side_effect = AvBlockingIOError("test", "test", "test")

        result = pull_until_blocked(mock_graph)

        assert result == []
