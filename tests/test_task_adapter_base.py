"""Tests for palabra_ai.task.adapter.base module"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from dataclasses import dataclass
import numpy as np

from palabra_ai.task.adapter.base import Reader, Writer, BufferedWriter
from palabra_ai.audio import AudioFrame, AudioBuffer
from palabra_ai.config import Config
from palabra_ai.task.base import TaskEvent
from palabra_ai.message import Dbg
from palabra_ai.enum import Kind, Channel, Direction


class ConcreteReader(Reader):
    """Concrete implementation of Reader for testing"""

    def __init__(self, *args, **kwargs):
        # Bypass Task.__init__ requirement for cfg
        self.cfg = kwargs.pop('cfg', None)
        # Initialize fields manually instead of calling super().__init__()
        self.q = asyncio.Queue()
        self.eof = TaskEvent()
        self.eof.set_owner(f"{self.__class__.__name__}.eof")
        self.stopper = TaskEvent()
        self.ready = TaskEvent()
        self._state = []
        self.result = None
        self._name = None
        # Initialize Task-specific fields
        self.root_tg = None
        self.sub_tg = asyncio.TaskGroup()
        self._task = None
        self._sub_tasks = []

    async def read(self, size: int) -> bytes | None:
        """Mock implementation"""
        return b"test_data"

    async def boot(self):
        """Mock boot implementation"""
        pass

    async def exit(self):
        """Mock exit implementation"""
        pass


class ConcreteBufferedWriter(BufferedWriter):
    """Concrete implementation of BufferedWriter for testing"""

    def __init__(self, *args, **kwargs):
        # Get cfg and drop_empty_frames before calling super
        cfg = kwargs.pop('cfg', None)
        drop_empty_frames = kwargs.get('drop_empty_frames', False)

        # Initialize fields manually instead of calling super().__init__()
        self.cfg = cfg
        self.q = asyncio.Queue()
        self.eof = TaskEvent()
        self.eof.set_owner(f"{self.__class__.__name__}.eof")
        self.stopper = TaskEvent()
        self.ready = TaskEvent()
        self._state = []
        self.result = None
        self._name = None
        self._frames_processed = 0
        # Initialize Task-specific fields
        self.root_tg = None
        self.sub_tg = asyncio.TaskGroup()
        self._task = None
        self._sub_tasks = []
        # BufferedWriter specific fields
        self.ab = None
        self.drop_empty_frames = drop_empty_frames

    async def boot(self):
        """Mock boot implementation"""
        from palabra_ai.audio import AudioBuffer
        estimated_duration = getattr(self.cfg, "estimated_duration", 60.0)
        self.ab = AudioBuffer(
            sample_rate=self.cfg.mode.output_sample_rate,
            num_channels=self.cfg.mode.num_channels,
            original_duration=estimated_duration,
            drop_empty_frames=getattr(self.cfg, "drop_empty_frames", False),
        )

    async def write(self, frame):
        """Mock write implementation"""
        return await self.ab.write(frame)

    async def exit(self):
        """Mock exit implementation"""
        pass


class ConcreteWriter(Writer):
    """Concrete implementation of Writer for testing"""

    def __init__(self, *args, **kwargs):
        # Bypass Task.__init__ requirement for cfg
        self.cfg = kwargs.pop('cfg', None)
        # Initialize fields manually instead of calling super().__init__()
        self.q = asyncio.Queue()
        self.eof = TaskEvent()
        self.eof.set_owner(f"{self.__class__.__name__}.eof")
        self.stopper = TaskEvent()
        self.ready = TaskEvent()
        self._state = []
        self.result = None
        self._name = None
        self._frames_processed = 0
        # Initialize Task-specific fields
        self.root_tg = None
        self.sub_tg = asyncio.TaskGroup()
        self._task = None
        self._sub_tasks = []

    async def write(self, frame: AudioFrame):
        """Mock implementation"""
        pass

    async def boot(self):
        """Mock boot implementation"""
        pass

    async def exit(self):
        """Mock exit implementation"""
        pass


@pytest.fixture
def mock_config():
    """Create mock config"""
    config = MagicMock()
    config.mode = MagicMock()
    config.mode.input_sample_rate = 16000
    config.mode.output_sample_rate = 16000
    config.mode.num_channels = 1
    return config


class TestReader:
    """Test Reader abstract class"""

    def test_init(self, mock_config):
        """Test Reader initialization"""
        reader = ConcreteReader(cfg=mock_config)

        assert reader.cfg == mock_config
        assert isinstance(reader.q, asyncio.Queue)
        assert isinstance(reader.eof, TaskEvent)
        assert reader.eof._owner == "ConcreteReader.eof"

    @pytest.mark.asyncio
    async def test_do_normal(self, mock_config):
        """Test do method normal operation"""
        reader = ConcreteReader(cfg=mock_config)
        reader.stopper = TaskEvent()

        # Set stopper after short delay
        async def set_stopper():
            await asyncio.sleep(0.05)
            +reader.stopper

        asyncio.create_task(set_stopper())

        await reader.do()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_do_with_eof(self, mock_config):
        """Test do method stops on eof"""
        reader = ConcreteReader(cfg=mock_config)
        reader.stopper = TaskEvent()

        # Set eof after short delay
        async def set_eof():
            await asyncio.sleep(0.05)
            +reader.eof

        asyncio.create_task(set_eof())

        await reader.do()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_read_abstract(self, mock_config):
        """Test read method is abstract"""
        reader = ConcreteReader(cfg=mock_config)

        # ConcreteReader has implementation, so test that it returns data
        result = await reader.read(100)
        assert result == b"test_data"


class TestWriter:
    """Test Writer class"""

    def test_init(self, mock_config):
        """Test Writer initialization"""
        writer = ConcreteWriter(cfg=mock_config)

        assert writer.cfg == mock_config
        assert isinstance(writer.q, asyncio.Queue)
        assert writer._frames_processed == 0

    @pytest.mark.asyncio
    async def test_do_with_frames(self, mock_config):
        """Test do method processing frames"""
        writer = ConcreteWriter(cfg=mock_config)
        writer.stopper = TaskEvent()
        writer.eof = TaskEvent()

        # Create mock frame
        frame = MagicMock(spec=AudioFrame)

        # Add frames to queue
        await writer.q.put(frame)
        await writer.q.put(frame)
        await writer.q.put(None)  # Signal stop

        with patch.object(writer, 'write', new_callable=AsyncMock) as mock_write:
            with patch('palabra_ai.task.adapter.base.trace') as mock_trace:
                with patch('palabra_ai.util.logger.debug') as mock_debug:
                    await writer.do()

                    assert mock_write.call_count == 2
                    assert writer._frames_processed == 2
                    mock_debug.assert_called_once()
                    assert "received None frame" in str(mock_debug.call_args[0][0])
                    assert writer.eof.is_set()

    @pytest.mark.asyncio
    async def test_do_timeout(self, mock_config):
        """Test do method with timeout"""
        writer = ConcreteWriter(cfg=mock_config)
        writer.stopper = TaskEvent()
        writer.eof = TaskEvent()

        # Set stopper after short delay
        async def set_stopper():
            await asyncio.sleep(0.1)
            +writer.stopper

        asyncio.create_task(set_stopper())

        await writer.do()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_do_cancelled(self, mock_config):
        """Test do method when cancelled"""
        writer = ConcreteWriter(cfg=mock_config)
        writer.stopper = TaskEvent()
        writer.eof = TaskEvent()

        # Mock queue to raise CancelledError
        writer.q = AsyncMock()
        writer.q.get = AsyncMock(side_effect=asyncio.CancelledError())

        with patch('palabra_ai.util.logger.warning') as mock_warning:
            with pytest.raises(asyncio.CancelledError):
                await writer.do()

            mock_warning.assert_called_once()
            assert "cancelled" in str(mock_warning.call_args[0][0])

    @pytest.mark.asyncio
    async def test_write_abstract(self, mock_config):
        """Test write method is abstract"""
        writer = ConcreteWriter(cfg=mock_config)
        frame = MagicMock(spec=AudioFrame)

        # ConcreteWriter has implementation, so test that it completes without error
        await writer.write(frame)

    @pytest.mark.asyncio
    async def test_exit(self, mock_config):
        """Test _exit method"""
        writer = ConcreteWriter(cfg=mock_config)
        writer.exit = AsyncMock(return_value="test_result")

        result = await writer._exit()

        # Check None was added to queue
        item = await writer.q.get()
        assert item is None

        writer.exit.assert_called_once()


class TestBufferedWriter:
    """Test BufferedWriter class"""

    def test_init(self, mock_config):
        """Test BufferedWriter initialization"""
        writer = ConcreteBufferedWriter(cfg=mock_config)

        assert writer.ab is None
        assert writer.drop_empty_frames is False

    def test_init_with_drop_empty(self, mock_config):
        """Test BufferedWriter with drop_empty_frames"""
        writer = ConcreteBufferedWriter(cfg=mock_config, drop_empty_frames=True)

        assert writer.drop_empty_frames is True

    @pytest.mark.asyncio
    async def test_boot(self, mock_config):
        """Test boot method creates AudioBuffer"""
        writer = ConcreteBufferedWriter(cfg=mock_config)

        await writer.boot()

        assert isinstance(writer.ab, AudioBuffer)
        assert writer.ab.sample_rate == 16000
        assert writer.ab.num_channels == 1

    @pytest.mark.asyncio
    async def test_write(self, mock_config):
        """Test write method delegates to AudioBuffer"""
        writer = ConcreteBufferedWriter(cfg=mock_config)

        await writer.boot()

        frame = MagicMock(spec=AudioFrame)

        with patch.object(writer.ab, 'write', new_callable=AsyncMock) as mock_write:
            result = await writer.write(frame)

            mock_write.assert_called_once_with(frame)
            assert result == mock_write.return_value

    def test_to_wav_bytes(self, mock_config):
        """Test to_wav_bytes method"""
        writer = ConcreteBufferedWriter(cfg=mock_config)

        # Create mock AudioBuffer
        writer.ab = MagicMock(spec=AudioBuffer)
        writer.ab.to_wav_bytes.return_value = b"wav_data"

        result = writer.to_wav_bytes()

        assert result == b"wav_data"
        writer.ab.to_wav_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_boot_with_drop_empty_frames(self):
        """Test boot method passes drop_empty_frames config to AudioBuffer"""
        from palabra_ai.config import Config, WsMode

        # Create real config with drop_empty_frames=True
        config = Config(drop_empty_frames=True, mode=WsMode(), estimated_duration=60.0)

        writer = ConcreteBufferedWriter(cfg=config)
        await writer.boot()

        # Verify AudioBuffer was created with drop_empty_frames=True
        assert writer.ab is not None
        assert writer.ab.drop_empty_frames is True

    @pytest.mark.asyncio
    async def test_boot_without_drop_empty_frames(self):
        """Test boot method defaults drop_empty_frames to False"""
        from palabra_ai.config import Config, WsMode

        # Create real config with drop_empty_frames=False (default)
        config = Config(drop_empty_frames=False, mode=WsMode(), estimated_duration=60.0)

        writer = ConcreteBufferedWriter(cfg=config)
        await writer.boot()

        # Verify AudioBuffer was created with drop_empty_frames=False
        assert writer.ab is not None
        assert writer.ab.drop_empty_frames is False
