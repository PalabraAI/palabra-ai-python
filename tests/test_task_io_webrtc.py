"""Tests for palabra_ai.task.io.webrtc module"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from palabra_ai.task.io.webrtc import WebrtcIo
from palabra_ai.enum import Channel
from palabra_ai.task.base import TaskEvent


@pytest.fixture
def mock_config():
    """Create mock config"""
    config = MagicMock()
    config.mode = MagicMock()
    config.mode.sample_rate = 8000
    config.mode.num_channels = 1
    config.targets = [MagicMock()]
    config.targets[0].lang = MagicMock()
    config.targets[0].lang.code = "en"
    return config


@pytest.fixture
def mock_credentials():
    """Create mock credentials"""
    creds = MagicMock()
    creds.webrtc_url = "ws://test.com"
    creds.jwt_token = "test_token"
    return creds


@pytest.fixture
def mock_reader():
    """Create mock reader"""
    reader = MagicMock()
    reader.ready = TaskEvent()
    reader.ready.set()
    return reader


@pytest.fixture
def mock_writer():
    """Create mock writer"""
    writer = MagicMock()
    writer.q = asyncio.Queue()
    return writer


class TestWebrtcIo:
    """Test WebrtcIo class"""

    def test_init_with_mock_room(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test initialization with proper room mocking"""
        with patch('palabra_ai.task.io.webrtc.rtc.Room') as mock_room_class:
            mock_room_instance = MagicMock()
            mock_room_class.return_value = mock_room_instance

            io = WebrtcIo(
                cfg=mock_config,
                credentials=mock_credentials,
                reader=mock_reader,
                writer=mock_writer
            )

            assert io.cfg == mock_config
            assert io.credentials == mock_credentials
            assert io.reader == mock_reader
            assert io.writer == mock_writer
            assert io.room == mock_room_instance
            assert io.channel == Channel.WEBRTC

    def test_channel_property(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test channel property"""
        with patch('palabra_ai.task.io.webrtc.rtc.Room'):
            io = WebrtcIo(
                cfg=mock_config,
                credentials=mock_credentials,
                reader=mock_reader,
                writer=mock_writer
            )

            assert io.channel == Channel.WEBRTC

    @pytest.mark.asyncio
    async def test_send_frame(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test send_frame method"""
        with patch('palabra_ai.task.io.webrtc.rtc.Room') as mock_room_class:
            mock_room_instance = MagicMock()
            mock_room_class.return_value = mock_room_instance

            io = WebrtcIo(
                cfg=mock_config,
                credentials=mock_credentials,
                reader=mock_reader,
                writer=mock_writer
            )

            # Mock audio source
            mock_audio_source = AsyncMock()
            io.in_audio_source = mock_audio_source

            # Mock frame
            mock_frame = MagicMock()
            mock_rtc_frame = MagicMock()
            mock_frame.to_rtc.return_value = mock_rtc_frame

            await io.send_frame(mock_frame)

            mock_frame.to_rtc.assert_called_once()
            mock_audio_source.capture_frame.assert_called_once_with(mock_rtc_frame)

    def test_name_property(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test name property exists and can be set"""
        with patch('palabra_ai.task.io.webrtc.rtc.Room'):
            io = WebrtcIo(
                cfg=mock_config,
                credentials=mock_credentials,
                reader=mock_reader,
                writer=mock_writer
            )

            # Test that name can be set and retrieved (may have [T] prefix)
            io.name = "test_webrtc_io"
            assert "test_webrtc_io" in io.name

    @pytest.mark.asyncio
    async def test_send_frame_with_raw_parameter(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test send_frame accepts raw parameter (called from base.push())"""
        import numpy as np

        mock_config.mode.input_samples_per_channel = 160
        mock_config.mode.for_input_audio_frame = (8000, 1, 160)
        mock_config.benchmark = True

        with patch('palabra_ai.task.io.webrtc.rtc.Room'):
            io = WebrtcIo(
                cfg=mock_config,
                credentials=mock_credentials,
                reader=mock_reader,
                writer=mock_writer
            )

            # Mock audio source
            mock_audio_source = AsyncMock()
            io.in_audio_source = mock_audio_source

            # Create audio bytes (160 samples * 2 bytes)
            audio_bytes = np.random.randint(-32768, 32767, 160, dtype=np.int16).tobytes()

            # This should call send_frame(audio_frame, raw) internally
            await io.push(audio_bytes)

            # Verify send_frame was called (it will be called via push)
            assert mock_audio_source.capture_frame.call_count > 0

    @pytest.mark.asyncio
    async def test_send_frame_ignores_raw_uses_rtc(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test send_frame ignores raw parameter and always uses frame.to_rtc()"""
        with patch('palabra_ai.task.io.webrtc.rtc.Room'):
            io = WebrtcIo(
                cfg=mock_config,
                credentials=mock_credentials,
                reader=mock_reader,
                writer=mock_writer
            )

            # Mock audio source
            mock_audio_source = AsyncMock()
            io.in_audio_source = mock_audio_source

            # Mock frame
            mock_frame = MagicMock()
            mock_rtc_frame = MagicMock()
            mock_frame.to_rtc = MagicMock(return_value=mock_rtc_frame)
            mock_frame.to_ws = MagicMock(return_value=b"ws_format_should_be_ignored")

            # Call with raw parameter (should be ignored)
            raw_bytes = b"this_should_be_ignored"
            await io.send_frame(mock_frame, raw=raw_bytes)

            # to_rtc() should be called (WebRTC format)
            mock_frame.to_rtc.assert_called_once()

            # to_ws() should NOT be called (raw is ignored)
            mock_frame.to_ws.assert_not_called()

            # capture_frame should be called with RTC frame (not raw)
            mock_audio_source.capture_frame.assert_called_once_with(mock_rtc_frame)

    @pytest.mark.asyncio
    async def test_push_benchmark_mode_converts_to_rtc(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test push() in benchmark mode uses frame.to_rtc() for WebRTC (not to_ws())"""
        import numpy as np

        mock_config.mode.input_samples_per_channel = 160
        mock_config.mode.for_input_audio_frame = (8000, 1, 160)
        mock_config.benchmark = True

        with patch('palabra_ai.task.io.webrtc.rtc.Room'):
            io = WebrtcIo(
                cfg=mock_config,
                credentials=mock_credentials,
                reader=mock_reader,
                writer=mock_writer
            )

            # Mock audio source
            mock_audio_source = AsyncMock()
            io.in_audio_source = mock_audio_source
            io._idx = iter(range(100))
            io._in_audio_num = iter(range(100))
            io.bench_audio_foq = MagicMock()

            # Create audio bytes
            audio_bytes = np.random.randint(-32768, 32767, 160, dtype=np.int16).tobytes()

            with patch('palabra_ai.task.io.base.IoEvent'):
                with patch('asyncio.to_thread', new_callable=AsyncMock, return_value=-20.0):
                    with patch.object(io, 'new_input_frame') as mock_new_frame:
                        mock_frame = MagicMock()
                        mock_rtc_frame = MagicMock()
                        mock_frame.to_rtc = MagicMock(return_value=mock_rtc_frame)
                        mock_frame.to_ws = MagicMock(return_value=b"ws_data")
                        mock_frame.data = bytearray(320)
                        mock_frame.duration = 0.02
                        mock_new_frame.return_value = mock_frame

                        await io.push(audio_bytes)

            # to_rtc() should be called (WebRTC uses RTC format)
            mock_frame.to_rtc.assert_called()

            # capture_frame should be called with RTC frame
            assert mock_audio_source.capture_frame.call_count > 0
            mock_audio_source.capture_frame.assert_called_with(mock_rtc_frame)
