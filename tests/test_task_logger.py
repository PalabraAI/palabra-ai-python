import asyncio
import json
import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, call, mock_open
from dataclasses import asdict

from palabra_ai.task.logger import Logger
from palabra_ai.task.base import TaskEvent
from palabra_ai.message import Dbg
from palabra_ai.config import Config
from palabra_ai.util.fanout_queue import FanoutQueue, Subscription
from palabra_ai.enum import Channel, Direction


class TestLogger:
    """Test Logger class"""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock config"""
        from io import StringIO
        config = MagicMock(spec=Config)
        config.log_file = tmp_path / "test.log"
        config.trace_file = tmp_path / "trace.json"
        config.debug = True
        config.benchmark = False  # Add benchmark attribute
        # Add internal_logs attribute
        config.internal_logs = StringIO("internal log 1\ninternal log 2\n")
        # Add to_dict method that returns a serializable dict
        config.to_dict.return_value = {
            "log_file": str(tmp_path / "test.log"),
            "trace_file": str(tmp_path / "trace.json"),
            "debug": True,
            "benchmark": False
        }
        return config

    @pytest.fixture
    def mock_io(self):
        """Create mock IO"""
        io = MagicMock()
        io.in_msg_foq = MagicMock(spec=FanoutQueue)
        io.out_msg_foq = MagicMock(spec=FanoutQueue)

        # Create mock subscriptions
        in_sub = MagicMock(spec=Subscription)
        in_sub.q = asyncio.Queue()
        out_sub = MagicMock(spec=Subscription)
        out_sub.q = asyncio.Queue()

        io.in_msg_foq.subscribe.return_value = in_sub
        io.out_msg_foq.subscribe.return_value = out_sub

        return io

    def test_init(self, mock_config, mock_io):
        """Test initialization"""
        logger = Logger(cfg=mock_config, io=mock_io)

        assert logger.cfg == mock_config
        assert logger.io == mock_io
        assert logger._messages == []
        assert isinstance(logger._start_ts, float)
        assert logger._io_in_sub is not None
        assert logger._io_out_sub is not None

        # Verify subscriptions
        mock_io.in_msg_foq.subscribe.assert_called_once_with(logger, maxsize=0)
        mock_io.out_msg_foq.subscribe.assert_called_once_with(logger, maxsize=0)

    @pytest.mark.asyncio
    async def test_boot(self, mock_config, mock_io):
        """Test boot method"""
        logger = Logger(cfg=mock_config, io=mock_io)
        logger.sub_tg = MagicMock()
        mock_in_task = MagicMock()
        mock_out_task = MagicMock()
        logger.sub_tg.create_task.side_effect = [mock_in_task, mock_out_task]

        with patch('palabra_ai.task.logger.debug') as mock_debug:
            await logger.boot()

            # Verify tasks created
            assert logger.sub_tg.create_task.call_count == 2
            assert logger._in_task == mock_in_task
            assert logger._out_task == mock_out_task

            # Verify debug message
            mock_debug.assert_called_once()
            assert "Logger started" in str(mock_debug.call_args[0][0])

    @pytest.mark.asyncio
    async def test_do(self, mock_config, mock_io):
        """Test do method"""
        logger = Logger(cfg=mock_config, io=mock_io)
        logger.stopper = TaskEvent()

        # Set stopper after short delay
        async def set_stopper():
            await asyncio.sleep(0.01)
            +logger.stopper

        with patch('palabra_ai.task.logger.debug') as mock_debug:
            asyncio.create_task(set_stopper())
            await logger.do()

            mock_debug.assert_called_once()
            assert "task stopped" in str(mock_debug.call_args[0][0])

    @pytest.mark.asyncio
    async def test_consume_with_message(self, mock_config, mock_io):
        """Test _consume with valid message"""
        logger = Logger(cfg=mock_config, io=mock_io)
        logger.stopper = TaskEvent()

        # Create mock message
        mock_msg = MagicMock()
        # Set _dbg as a Dbg instance (as expected by the logger)
        from palabra_ai.enum import Kind
        dbg = Dbg(kind=Kind.MESSAGE, ch=Channel.WS, dir=Direction.IN)
        dbg.ts = 1234.5  # Set specific timestamp
        mock_msg._dbg = dbg
        mock_msg.model_dump.return_value = {"type": "test_message"}

        # Create queue and add message
        q = asyncio.Queue()
        await q.put(mock_msg)
        await q.put(None)  # Signal stop

        with patch('palabra_ai.task.logger.debug') as mock_debug:
            await logger._consume(q)

            # Verify message was processed
            assert len(logger._messages) == 1
            assert logger._messages[0]["msg"]["type"] == "test_message"
            mock_debug.assert_called()

    @pytest.mark.asyncio
    async def test_consume_timeout(self, mock_config, mock_io):
        """Test _consume with timeout"""
        logger = Logger(cfg=mock_config, io=mock_io)
        logger.stopper = TaskEvent()

        q = asyncio.Queue()

        # Set stopper after short delay
        async def set_stopper():
            await asyncio.sleep(0.1)
            +logger.stopper

        asyncio.create_task(set_stopper())
        await logger._consume(q)

        # Should complete without error
        assert len(logger._messages) == 0

    @pytest.mark.asyncio
    async def test_consume_cancelled(self, mock_config, mock_io):
        """Test _consume when cancelled"""
        logger = Logger(cfg=mock_config, io=mock_io)
        logger.stopper = TaskEvent()

        q = AsyncMock()
        q.get = AsyncMock(side_effect=asyncio.CancelledError())

        with patch('palabra_ai.task.logger.debug') as mock_debug:
            await logger._consume(q)

            mock_debug.assert_called_once()
            assert "cancelled" in str(mock_debug.call_args[0][0])

    @pytest.mark.asyncio
    async def test_consume_no_dbg_attribute(self, mock_config, mock_io):
        """Test _consume with message without _dbg attribute"""
        logger = Logger(cfg=mock_config, io=mock_io)
        logger.stopper = TaskEvent()

        # Create mock message without _dbg
        mock_msg = MagicMock()
        del mock_msg._dbg  # Remove _dbg attribute
        mock_msg.model_dump.return_value = {"type": "test_message"}

        # Create queue and add message
        q = asyncio.Queue()
        await q.put(mock_msg)
        await q.put(None)  # Signal stop

        await logger._consume(q)

        # Verify message was processed with empty Dbg
        assert len(logger._messages) == 1
        assert logger._messages[0]["msg"]["type"] == "test_message"

    @pytest.mark.asyncio
    async def test_exit_success(self, mock_config, mock_io, tmp_path):
        """Test successful exit"""
        logger = Logger(cfg=mock_config, io=mock_io)
        logger._messages = [{"msg": {"type": "test1"}}, {"msg": {"type": "test2"}}]

        # Create mock tasks that are actual asyncio.Task objects
        async def dummy_task():
            await asyncio.sleep(0.001)  # Very short sleep

        logger._in_task = asyncio.create_task(dummy_task())
        logger._out_task = asyncio.create_task(dummy_task())

        # Create log file
        log_file = tmp_path / "test.log"
        log_file.write_text("Log line 1\nLog line 2\n")

        with patch('palabra_ai.task.logger.debug') as mock_debug:
            with patch('palabra_ai.task.logger.get_system_info') as mock_sysinfo:
                mock_sysinfo.return_value = {"os": "test", "version": "1.0"}

                # Mock the file write operation
                with patch('builtins.open', mock_open()) as mock_file:
                    result = await logger.exit()

                    # Tasks should be done or cancelled
                    assert logger._in_task.done()
                    assert logger._out_task.done()

                    # Verify trace file write was attempted with correct path
                    mock_file.assert_any_call(mock_config.trace_file, "wb")

                # Verify unsubscribe
                mock_io.in_msg_foq.unsubscribe.assert_called_once_with(logger)
                mock_io.out_msg_foq.unsubscribe.assert_called_once_with(logger)

                assert mock_debug.call_count >= 3

    @pytest.mark.asyncio
    async def test_exit_log_file_error(self, mock_config, mock_io, tmp_path):
        """Test exit when log file can't be read"""
        logger = Logger(cfg=mock_config, io=mock_io)
        logger._messages = []

        # Create mock tasks that are actual asyncio.Task objects
        async def dummy_task():
            await asyncio.sleep(0.001)  # Very short sleep

        logger._in_task = asyncio.create_task(dummy_task())
        logger._out_task = asyncio.create_task(dummy_task())

        # Make log file unreadable
        mock_config.log_file = "/nonexistent/file.log"

        with patch('palabra_ai.task.logger.debug'):
            with patch('palabra_ai.task.logger.get_system_info') as mock_sysinfo:
                mock_sysinfo.return_value = {"os": "test"}

                # Mock the file write operation
                with patch('builtins.open', mock_open()) as mock_file:
                    result = await logger.exit()

                    # Verify trace file write was attempted with correct path
                    mock_file.assert_any_call(mock_config.trace_file, "wb")

    @pytest.mark.asyncio
    async def test_exit_sysinfo_error(self, mock_config, mock_io, tmp_path):
        """Test exit when sysinfo fails"""
        logger = Logger(cfg=mock_config, io=mock_io)
        logger._messages = []

        # Create mock tasks that are actual asyncio.Task objects
        async def dummy_task():
            await asyncio.sleep(0.001)  # Very short sleep

        logger._in_task = asyncio.create_task(dummy_task())
        logger._out_task = asyncio.create_task(dummy_task())

        # Create log file
        log_file = tmp_path / "test.log"
        log_file.write_text("Log line\n")

        with patch('palabra_ai.task.logger.debug'):
            with patch('palabra_ai.task.logger.get_system_info') as mock_sysinfo:
                mock_sysinfo.side_effect = RuntimeError("Sysinfo error")

                # Mock the file write operation
                with patch('builtins.open', mock_open()) as mock_file:
                    result = await logger.exit()

                    # Verify trace file write was attempted with correct path
                    mock_file.assert_any_call(mock_config.trace_file, "wb")

    @pytest.mark.asyncio
    async def test_exit_with_version(self, mock_config, mock_io):
        """Test exit includes version info"""
        logger = Logger(cfg=mock_config, io=mock_io)
        logger._messages = []

        # Create mock tasks that are actual asyncio.Task objects
        async def dummy_task():
            await asyncio.sleep(0.001)  # Very short sleep

        logger._in_task = asyncio.create_task(dummy_task())
        logger._out_task = asyncio.create_task(dummy_task())

        with patch('palabra_ai.task.logger.debug'):
            with patch('palabra_ai.task.logger.get_system_info') as mock_sysinfo:
                with patch('palabra_ai.__version__', '1.2.3'):
                    mock_sysinfo.return_value = {}

                    # Mock the file write operation
                    with patch('builtins.open', mock_open()) as mock_file:
                        await logger.exit()

                        # Verify trace file write was attempted
                        mock_file.assert_called_with(mock_config.trace_file, "wb")

    @pytest.mark.asyncio
    async def test_exit_no_tasks(self, mock_config, mock_io):
        """Test exit when no tasks were created"""
        logger = Logger(cfg=mock_config, io=mock_io)
        logger._messages = []
        logger._in_task = None
        logger._out_task = None

        with patch('palabra_ai.task.logger.debug'):
            with patch('palabra_ai.task.logger.get_system_info') as mock_sysinfo:
                mock_sysinfo.return_value = {}

                # Mock the file write operation
                with patch('builtins.open', mock_open()) as mock_file:
                    await logger.exit()

                    # Should complete without error
                    mock_file.assert_called_with(mock_config.trace_file, "wb")

    @pytest.mark.asyncio
    async def test_underscore_exit(self, mock_config, mock_io):
        """Test _exit method calls exit"""
        logger = Logger(cfg=mock_config, io=mock_io)
        logger.exit = AsyncMock(return_value="test_result")

        result = await logger._exit()

        logger.exit.assert_called_once()
        assert result == "test_result"
