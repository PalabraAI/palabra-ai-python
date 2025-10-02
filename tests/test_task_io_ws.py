"""Tests for palabra_ai.task.io.ws module"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
import numpy as np

from palabra_ai.task.io.ws import WsIo
from palabra_ai.audio import AudioFrame
from palabra_ai.enum import Channel, Direction, Kind
from palabra_ai.message import Dbg, Message, EosMessage, TranscriptionMessage
from palabra_ai.config import Config
from palabra_ai.task.base import TaskEvent


@pytest.fixture
def mock_config():
    """Create mock config"""
    config = MagicMock()
    config.mode = MagicMock()
    config.mode.sample_rate = 16000
    config.mode.num_channels = 1
    config.mode.samples_per_channel = 160
    config.mode.chunk_duration_ms = 10
    config.mode.for_input_audio_frame = (16000, 1, 160)
    config.benchmark = False
    return config


@pytest.fixture
def mock_credentials():
    """Create mock credentials"""
    creds = MagicMock()
    creds.ws_url = "wss://test.example.com"
    creds.jwt_token = "test_token"
    return creds


@pytest.fixture
def mock_reader():
    """Create mock reader"""
    reader = MagicMock()
    return reader


@pytest.fixture
def mock_writer():
    """Create mock writer"""
    writer = MagicMock()
    writer.q = asyncio.Queue()
    return writer


class TestWsIo:
    """Test WsIo class"""

    def test_init(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test initialization"""
        # Create mock reader
        mock_reader = MagicMock()

        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)

        assert ws_io.cfg == mock_config
        assert ws_io.credentials == mock_credentials
        assert ws_io.reader == mock_reader
        assert ws_io.writer == mock_writer
        assert ws_io.ws is None
        assert ws_io._ws_cm is None

    def test_dsn_property(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test dsn property"""
        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)

        expected = "wss://test.example.com?token=test_token"
        assert ws_io.dsn == expected

    def test_channel_property(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test channel property"""
        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)

        assert ws_io.channel == Channel.WS

    @pytest.mark.asyncio
    async def test_send_message(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test send_message method"""
        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)

        # Mock WebSocket connection
        ws_io.ws = AsyncMock()

        msg_data = b"test_message"
        await ws_io.send_message(msg_data)

        ws_io.ws.send.assert_called_once_with(msg_data)

    @pytest.mark.asyncio
    async def test_send_frame(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test send_frame method"""
        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)

        # Mock WebSocket connection
        ws_io.ws = AsyncMock()

        # Mock frame
        frame = MagicMock(spec=AudioFrame)
        frame.to_ws.return_value = b"frame_data"

        with patch('palabra_ai.task.io.ws.debug') as mock_debug:
            await ws_io.send_frame(frame)

            frame.to_ws.assert_called_once()
            ws_io.ws.send.assert_called_once_with(b"frame_data")
            mock_debug.assert_called_once()
            assert str(frame) in str(mock_debug.call_args[0][0])

    def test_new_frame(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test new_input_frame method"""
        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)

        with patch('palabra_ai.audio.AudioFrame.create') as mock_create:
            mock_create.return_value = MagicMock(spec=AudioFrame)

            frame = ws_io.new_input_frame()

            mock_create.assert_called_once_with(16000, 1, 160)
            assert frame == mock_create.return_value

    @pytest.mark.asyncio
    async def test_ws_receiver_audio_frame(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test ws_receiver processing audio frame"""
        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)
        ws_io.stopper = TaskEvent()
        ws_io.eof = TaskEvent()
        ws_io._idx = iter(range(100))
        ws_io._out_audio_num = iter(range(100))

        # Mock WebSocket that yields one audio frame
        ws_io.ws = AsyncMock()
        raw_audio = b"audio_frame_data"
        ws_io.ws.__aiter__.return_value = [raw_audio]

        # Mock AudioFrame.from_ws to return a frame
        mock_frame = MagicMock(spec=AudioFrame)

        with patch('palabra_ai.audio.AudioFrame.from_ws', return_value=mock_frame):
            with patch('palabra_ai.task.io.ws.debug') as mock_debug:
                with patch('palabra_ai.task.io.ws.trace'):
                    try:
                        await ws_io.ws_receiver()
                    except EOFError:
                        pass  # Expected when no more messages

                    # Check frame was added to writer queue
                    assert ws_io.writer.q.qsize() == 2  # frame + None
                    item = await ws_io.writer.q.get()
                    assert item == mock_frame

    @pytest.mark.asyncio
    async def test_ws_receiver_message(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test ws_receiver processing regular message"""
        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)
        ws_io.stopper = TaskEvent()
        ws_io.eof = TaskEvent()
        ws_io._idx = iter(range(100))
        ws_io._out_audio_num = iter(range(100))
        ws_io.out_msg_foq = MagicMock()

        # Mock WebSocket that yields one message
        ws_io.ws = AsyncMock()
        raw_msg = b"message_data"
        ws_io.ws.__aiter__.return_value = [raw_msg]

        # Mock AudioFrame.from_ws to return None (not audio)
        # Mock Message.decode to return a message
        mock_msg = MagicMock(spec=TranscriptionMessage)

        with patch('palabra_ai.audio.AudioFrame.from_ws', return_value=None):
            with patch('palabra_ai.message.Message.decode', return_value=mock_msg):
                with patch('palabra_ai.task.io.ws.IoEvent'):
                    with patch('palabra_ai.task.io.ws.debug'):
                        with patch('palabra_ai.task.io.ws.trace'):
                            try:
                                await ws_io.ws_receiver()
                            except EOFError:
                                pass  # Expected when no more messages

                            # Check message was published
                            ws_io.out_msg_foq.publish.assert_any_call(mock_msg)
                            assert mock_msg._dbg is not None

    @pytest.mark.asyncio
    async def test_ws_receiver_eos_message(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test ws_receiver handling EOS message"""
        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)
        ws_io.stopper = TaskEvent()
        ws_io.eof = TaskEvent()
        ws_io._idx = iter(range(100))
        ws_io._out_audio_num = iter(range(100))
        ws_io.out_msg_foq = MagicMock()

        # Mock WebSocket that yields an EOS message
        ws_io.ws = AsyncMock()
        raw_msg = b"eos_message"
        ws_io.ws.__aiter__.return_value = [raw_msg]

        # Mock AudioFrame.from_ws to return None
        # Mock Message.decode to return EosMessage
        mock_eos = MagicMock(spec=EosMessage)

        with patch('palabra_ai.audio.AudioFrame.from_ws', return_value=None):
            with patch('palabra_ai.message.Message.decode', return_value=mock_eos):
                with patch('palabra_ai.task.io.ws.IoEvent'):
                    with patch('palabra_ai.task.io.ws.debug') as mock_debug:
                        with patch('palabra_ai.task.io.ws.trace'):
                            await ws_io.ws_receiver()

                            # Check EOF was set
                            assert ws_io.eof.is_set()
                            mock_debug.assert_any_call(f"EOF!!! End of stream received: {mock_eos}")

    @pytest.mark.asyncio
    async def test_ws_receiver_benchmark_mode(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test ws_receiver in benchmark mode"""
        mock_config.benchmark = True

        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)
        ws_io.stopper = TaskEvent()
        ws_io.eof = TaskEvent()
        ws_io._idx = iter(range(100))
        ws_io._out_audio_num = iter(range(100))
        ws_io.bench_audio_foq = MagicMock()
        ws_io.calc_rms_db = MagicMock(return_value=-20.0)

        # Mock WebSocket that yields one audio frame
        ws_io.ws = AsyncMock()
        raw_audio = b"audio_frame_data"
        ws_io.ws.__aiter__.return_value = [raw_audio]

        # Mock AudioFrame.from_ws to return a frame
        mock_frame = MagicMock(spec=AudioFrame)

        with patch('palabra_ai.audio.AudioFrame.from_ws', return_value=mock_frame):
            with patch('palabra_ai.task.io.ws.IoEvent'):
                with patch('palabra_ai.task.io.ws.debug'):
                    with patch('palabra_ai.task.io.ws.trace'):
                        with patch('asyncio.to_thread', new_callable=AsyncMock, return_value=-20.0):
                            try:
                                await ws_io.ws_receiver()
                            except EOFError:
                                pass

                            # Check benchmark data was added
                            assert mock_frame._dbg is not None
                            ws_io.bench_audio_foq.publish.assert_called_once_with(mock_frame)

    @pytest.mark.asyncio
    async def test_boot(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test boot method"""
        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)
        ws_io.sub_tg = MagicMock()
        ws_io.sub_tg.create_task = MagicMock()
        ws_io.set_task = AsyncMock()

        # Mock WebSocket context manager
        mock_ws_cm = AsyncMock()
        mock_ws = AsyncMock()
        mock_ws_cm.__aenter__.return_value = mock_ws

        with patch('palabra_ai.task.io.ws.ws_connect', return_value=mock_ws_cm) as mock_connect:
            await ws_io.boot()

            # Check connection was established
            mock_connect.assert_called_once_with(ws_io.dsn)
            assert ws_io.ws == mock_ws
            assert ws_io._ws_cm == mock_ws_cm

            # Check ping was called
            mock_ws.ping.assert_called_once()

            # Check tasks were created
            assert ws_io.sub_tg.create_task.call_count == 2

            # Check set_task was called
            ws_io.set_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_exit(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test exit method"""
        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)

        # Mock WebSocket context manager
        mock_ws_cm = AsyncMock()
        mock_ws = AsyncMock()
        ws_io._ws_cm = mock_ws_cm
        ws_io.ws = mock_ws

        await ws_io.exit()

        # Check cleanup
        mock_ws_cm.__aexit__.assert_called_once_with(None, None, None)
        assert ws_io.ws is None

    @pytest.mark.asyncio
    async def test_exit_no_connection(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test exit method with no connection"""
        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)

        # No WebSocket connection
        ws_io._ws_cm = None
        ws_io.ws = None

        await ws_io.exit()

        # Should complete without error
        assert ws_io.ws is None

    def test_eos_received_field_default(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test WsIo eos_received field defaults to False"""
        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)

        assert hasattr(ws_io, 'eos_received')
        assert ws_io.eos_received is False

    @pytest.mark.asyncio
    async def test_ws_receiver_eos_sets_flag(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test ws_receiver sets eos_received flag when EOS message received"""
        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)
        ws_io.stopper = TaskEvent()
        ws_io.eof = TaskEvent()
        ws_io._idx = iter(range(100))
        ws_io._out_audio_num = iter(range(100))
        ws_io.out_msg_foq = MagicMock()

        # Verify eos_received starts as False
        assert ws_io.eos_received is False

        # Mock WebSocket that yields an EOS message
        ws_io.ws = AsyncMock()
        raw_msg = b"eos_message"
        ws_io.ws.__aiter__.return_value = [raw_msg]

        # Mock AudioFrame.from_ws to return None
        # Mock Message.decode to return EosMessage
        mock_eos = MagicMock(spec=EosMessage)

        with patch('palabra_ai.audio.AudioFrame.from_ws', return_value=None):
            with patch('palabra_ai.message.Message.decode', return_value=mock_eos):
                with patch('palabra_ai.task.io.ws.IoEvent'):
                    with patch('palabra_ai.task.io.ws.debug'):
                        with patch('palabra_ai.task.io.ws.trace'):
                            await ws_io.ws_receiver()

                            # Check eos_received was set to True
                            assert ws_io.eos_received is True
                            # Check EOF was also set
                            assert ws_io.eof.is_set()

    @pytest.mark.asyncio
    async def test_ws_receiver_eof_exception_sets_flag(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test ws_receiver sets eos_received flag in EOFError handler"""
        ws_io = WsIo(cfg=mock_config, credentials=mock_credentials, reader=mock_reader, writer=mock_writer)
        ws_io.stopper = TaskEvent()
        ws_io.eof = TaskEvent()
        ws_io._idx = iter(range(100))
        ws_io._out_audio_num = iter(range(100))
        ws_io.out_msg_foq = MagicMock()

        # Verify eos_received starts as False
        assert ws_io.eos_received is False

        # Mock WebSocket that raises EOFError directly
        ws_io.ws = AsyncMock()
        ws_io.ws.__aiter__.side_effect = EOFError("Connection closed")

        with patch('palabra_ai.task.io.ws.debug'):
            await ws_io.ws_receiver()

            # Check eos_received was set to True in exception handler
            assert ws_io.eos_received is True
            # Check EOF was also set
            assert ws_io.eof.is_set()