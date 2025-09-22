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
