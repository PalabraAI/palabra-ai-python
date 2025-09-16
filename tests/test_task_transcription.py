"""Tests for palabra_ai.task.transcription module"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from dataclasses import dataclass

from palabra_ai.task.transcription import Transcription
from palabra_ai.config import Config
from palabra_ai.message import TranscriptionMessage, Message
from palabra_ai.task.base import TaskEvent
from palabra_ai.util.fanout_queue import FanoutQueue, Subscription


@pytest.fixture
def mock_config():
    """Create mock config with callbacks"""
    config = MagicMock()

    # Source language config
    config.source = MagicMock()
    config.source.lang = MagicMock()
    config.source.lang.code = "en"
    config.source.on_transcription = MagicMock()

    # Target language configs
    target1 = MagicMock()
    target1.lang = MagicMock()
    target1.lang.code = "es"
    target1.on_transcription = MagicMock()

    target2 = MagicMock()
    target2.lang = MagicMock()
    target2.lang.code = "fr"
    target2.on_transcription = None  # No callback

    config.targets = [target1, target2]

    return config


@pytest.fixture
def mock_io():
    """Create mock IO"""
    io = MagicMock()
    io.out_msg_foq = MagicMock(spec=FanoutQueue)
    io.ready = TaskEvent()
    io.ready.set()

    # Create mock subscription
    sub = MagicMock(spec=Subscription)
    sub.q = asyncio.Queue()
    io.out_msg_foq.subscribe.return_value = sub

    return io


class TestTranscription:
    """Test Transcription task"""

    def test_init(self, mock_config, mock_io):
        """Test initialization and callback collection"""
        trans = Transcription(cfg=mock_config, io=mock_io)

        assert trans.cfg == mock_config
        assert trans.io == mock_io
        assert trans.suppress_callback_errors is True
        assert trans._out_q is None

        # Check callbacks were collected
        assert "en" in trans._callbacks
        assert "es" in trans._callbacks
        assert "fr" not in trans._callbacks  # No callback set
        assert trans._callbacks["en"] == mock_config.source.on_transcription
        assert trans._callbacks["es"] == mock_config.targets[0].on_transcription

    def test_init_no_callbacks(self):
        """Test initialization with no callbacks"""
        config = MagicMock()
        config.source = MagicMock()
        config.source.on_transcription = None
        config.targets = []

        io = MagicMock()
        trans = Transcription(cfg=config, io=io)

        assert trans._callbacks == {}

    @pytest.mark.asyncio
    async def test_boot(self, mock_config, mock_io):
        """Test boot method"""
        trans = Transcription(cfg=mock_config, io=mock_io)

        with patch('palabra_ai.task.transcription.debug') as mock_debug:
            await trans.boot()

            # Check subscription
            mock_io.out_msg_foq.subscribe.assert_called_once_with(trans, maxsize=0)
            assert trans._out_q is not None

            # Check debug message
            mock_debug.assert_called_once()
            assert "['en', 'es']" in str(mock_debug.call_args[0][0])

    @pytest.mark.asyncio
    async def test_do_with_transcription_message(self, mock_config, mock_io):
        """Test do method processing TranscriptionMessage"""
        trans = Transcription(cfg=mock_config, io=mock_io)
        trans.stopper = TaskEvent()

        # Set up queue
        await trans.boot()

        # Create TranscriptionMessage
        msg = MagicMock(spec=TranscriptionMessage)
        msg.language = MagicMock()
        msg.language.code = "en"

        # Add message to queue
        await trans._out_q.put(msg)
        await trans._out_q.put(None)  # Signal stop

        with patch.object(trans, '_process_message', new_callable=AsyncMock) as mock_process:
            await trans.do()

            mock_process.assert_called_once_with(msg)

    @pytest.mark.asyncio
    async def test_do_timeout(self, mock_config, mock_io):
        """Test do method with timeout"""
        trans = Transcription(cfg=mock_config, io=mock_io)
        trans.stopper = TaskEvent()

        await trans.boot()

        # Set stopper after short delay
        async def set_stopper():
            await asyncio.sleep(0.1)
            +trans.stopper

        asyncio.create_task(set_stopper())

        with patch('palabra_ai.task.transcription.debug'):
            await trans.do()
            # Should complete without error

    @pytest.mark.asyncio
    async def test_exit(self, mock_config, mock_io):
        """Test exit method"""
        trans = Transcription(cfg=mock_config, io=mock_io)

        await trans.exit()

        mock_io.out_msg_foq.unsubscribe.assert_called_once_with(trans)

    @pytest.mark.asyncio
    async def test_process_message_transcription(self, mock_config, mock_io):
        """Test _process_message with TranscriptionMessage"""
        trans = Transcription(cfg=mock_config, io=mock_io)

        msg = MagicMock(spec=TranscriptionMessage)
        msg.language = MagicMock()
        msg.language.code = "en"

        with patch.object(trans, '_call_callback', new_callable=AsyncMock) as mock_call:
            await trans._process_message(msg)

            mock_call.assert_called_once_with(
                mock_config.source.on_transcription,
                msg
            )

    @pytest.mark.asyncio
    async def test_process_message_no_callback(self, mock_config, mock_io):
        """Test _process_message with no matching callback"""
        trans = Transcription(cfg=mock_config, io=mock_io)

        msg = MagicMock(spec=TranscriptionMessage)
        msg.language = MagicMock()
        msg.language.code = "de"  # No callback for German

        with patch.object(trans, '_call_callback', new_callable=AsyncMock) as mock_call:
            await trans._process_message(msg)

            mock_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_message_not_transcription(self, mock_config, mock_io):
        """Test _process_message with non-TranscriptionMessage"""
        trans = Transcription(cfg=mock_config, io=mock_io)

        msg = MagicMock(spec=Message)  # Not a TranscriptionMessage

        with patch.object(trans, '_call_callback', new_callable=AsyncMock) as mock_call:
            await trans._process_message(msg)

            mock_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_message_error(self, mock_config, mock_io):
        """Test _process_message error handling"""
        trans = Transcription(cfg=mock_config, io=mock_io)

        msg = MagicMock(spec=TranscriptionMessage)
        msg.language = MagicMock()
        msg.language.code = "en"

        with patch.object(trans, '_call_callback', side_effect=Exception("Test error")):
            with patch('palabra_ai.task.transcription.error') as mock_error:
                await trans._process_message(msg)

                mock_error.assert_called_once()
                assert "Test error" in str(mock_error.call_args[0][0])

    @pytest.mark.asyncio
    async def test_call_callback_async(self, mock_config, mock_io):
        """Test _call_callback with async callback"""
        trans = Transcription(cfg=mock_config, io=mock_io)
        trans.sub_tg = MagicMock()
        trans.sub_tg.create_task = MagicMock()

        async_callback = AsyncMock()
        data = MagicMock(spec=TranscriptionMessage)

        await trans._call_callback(async_callback, data)

        # Should create task for async callback
        trans.sub_tg.create_task.assert_called_once()
        call_args = trans.sub_tg.create_task.call_args
        assert call_args[1]['name'] == 'Transcription:callback'

    @pytest.mark.asyncio
    async def test_call_callback_sync(self, mock_config, mock_io):
        """Test _call_callback with sync callback"""
        trans = Transcription(cfg=mock_config, io=mock_io)

        sync_callback = MagicMock()
        data = MagicMock(spec=TranscriptionMessage)

        with patch('asyncio.get_event_loop') as mock_loop:
            loop = MagicMock()
            mock_loop.return_value = loop
            loop.run_in_executor = AsyncMock()

            await trans._call_callback(sync_callback, data)

            loop.run_in_executor.assert_called_once_with(None, sync_callback, data)

    @pytest.mark.asyncio
    async def test_call_callback_error_suppressed(self, mock_config, mock_io):
        """Test _call_callback error suppression"""
        trans = Transcription(cfg=mock_config, io=mock_io)
        trans.suppress_callback_errors = True

        callback = MagicMock(side_effect=Exception("Callback error"))
        data = MagicMock(spec=TranscriptionMessage)

        with patch('asyncio.get_event_loop') as mock_loop:
            loop = MagicMock()
            mock_loop.return_value = loop
            loop.run_in_executor = AsyncMock(side_effect=Exception("Callback error"))

            with patch('palabra_ai.task.transcription.error') as mock_error:
                await trans._call_callback(callback, data)

                mock_error.assert_called_once()
                assert "Callback error" in str(mock_error.call_args[0][0])

    @pytest.mark.asyncio
    async def test_call_callback_error_raised(self, mock_config, mock_io):
        """Test _call_callback error raised when not suppressed"""
        trans = Transcription(cfg=mock_config, io=mock_io)
        trans.suppress_callback_errors = False

        callback = MagicMock(side_effect=Exception("Callback error"))
        data = MagicMock(spec=TranscriptionMessage)

        with patch('asyncio.get_event_loop') as mock_loop:
            loop = MagicMock()
            mock_loop.return_value = loop
            loop.run_in_executor = AsyncMock(side_effect=Exception("Callback error"))

            with pytest.raises(Exception, match="Callback error"):
                await trans._call_callback(callback, data)
