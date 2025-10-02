import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
from palabra_ai.task.manager import Manager
from palabra_ai.exc import ConfigurationError
from palabra_ai.config import Config, WsMode, WebrtcMode
from palabra_ai.internal.rest import SessionCredentials
from palabra_ai.task.adapter.base import Reader, Writer
from palabra_ai.task.adapter.dummy import DummyWriter
from palabra_ai.task.base import TaskEvent
from palabra_ai.constant import BOOT_TIMEOUT, SAFE_PUBLICATION_END_DELAY

class MockReader(Reader):
    """Mock reader for testing"""
    def __init__(self):
        # Skip parent init to avoid cfg requirement
        self.cfg = None
        self.ready = asyncio.Event()
        self.stopper = False
        self.eof = False
        self._task = None
        self.name = "MockReader"

    async def boot(self):
        pass

    async def do(self):
        pass

    async def exit(self):
        pass

    async def read(self):
        pass

class MockWriter(Writer):
    """Mock writer for testing"""
    def __init__(self):
        # Skip parent init to avoid cfg requirement
        self.cfg = None
        self.ready = asyncio.Event()
        self.stopper = False
        self.eof = False
        self._task = None
        self.name = "MockWriter"

    async def boot(self):
        pass

    async def do(self):
        pass

    async def exit(self):
        pass

    async def write(self, audio_frame):
        pass

class TestManager:
    """Test Manager class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.mock_config = MagicMock()
        self.mock_credentials = MagicMock()

        # Setup config mocks
        self.mock_config.targets = [MagicMock()]
        self.mock_config.source = MagicMock()
        self.mock_config.source.reader = MockReader()
        self.mock_config.targets[0].writer = MockWriter()
        self.mock_config.targets[0].on_transcription = None
        self.mock_config.mode = WsMode()
        self.mock_config.log_file = None
        self.mock_config.to_dict.return_value = {}

    def test_init_success(self):
        """Test successful manager initialization"""
        manager = Manager(cfg=self.mock_config, credentials=self.mock_credentials)

        assert manager.cfg == self.mock_config
        assert manager.credentials == self.mock_credentials
        assert isinstance(manager.reader, MockReader)
        assert isinstance(manager.writer, MockWriter)
        assert manager.io_class is not None
        assert manager.tasks is not None

    def test_init_multiple_targets_error(self):
        """Test error when multiple targets provided"""
        self.mock_config.targets = [MagicMock(), MagicMock()]

        with pytest.raises(ConfigurationError) as exc_info:
            Manager(cfg=self.mock_config, credentials=self.mock_credentials)

        assert "Only single target language supported" in str(exc_info.value)

    def test_init_invalid_reader_error(self):
        """Test error when reader is not a Reader instance"""
        self.mock_config.source.reader = "not a reader"

        with pytest.raises(ConfigurationError) as exc_info:
            Manager(cfg=self.mock_config, credentials=self.mock_credentials)

        assert "cfg.source.reader should be an instance of Reader" in str(exc_info.value)

    def test_init_no_writer_or_callback_error(self):
        """Test error when neither writer nor on_transcription provided"""
        self.mock_config.targets[0].writer = None
        self.mock_config.targets[0].on_transcription = None

        with pytest.raises(ConfigurationError) as exc_info:
            Manager(cfg=self.mock_config, credentials=self.mock_credentials)

        assert "You should use at least [writer] or [on_transcription]" in str(exc_info.value)

    def test_init_dummy_writer_when_no_writer(self):
        """Test DummyWriter is used when no writer provided but callback exists"""
        self.mock_config.targets[0].writer = None
        self.mock_config.targets[0].on_transcription = lambda x: x

        manager = Manager(cfg=self.mock_config, credentials=self.mock_credentials)

        assert isinstance(manager.writer, DummyWriter)

    def test_init_unsupported_io_mode(self):
        """Test error when unsupported IO mode"""
        # Create a custom mode class that's not WebrtcMode or WsMode
        from palabra_ai.config import IoMode

        class UnsupportedMode(IoMode):
            pass

        self.mock_config.mode = UnsupportedMode(
            name="unsupported",
            input_sample_rate=16000,
            output_sample_rate=16000,
            num_channels=1,
            input_chunk_duration_ms=100
        )
        with pytest.raises(ConfigurationError) as exc_info:
            Manager(cfg=self.mock_config, credentials=self.mock_credentials)

        assert "Unsupported IO mode" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_start_system(self):
        """Test start_system method"""
        manager = Manager(cfg=self.mock_config, credentials=self.mock_credentials)

        # Create proper async mocks
        async def mock_call(tg):
            pass

        # Mock all components with ready events
        manager.stat = MagicMock()
        manager.stat.__call__ = mock_call
        manager.stat.ready = TaskEvent()
        +manager.stat.ready  # Set it
        manager.stat.run_banner = MagicMock(return_value=MagicMock())

        manager.io.__call__ = mock_call
        manager.io.ready = TaskEvent()
        +manager.io.ready  # Set it

        # Make writer callable
        manager.writer = MagicMock()
        manager.writer.__call__ = mock_call
        manager.writer.ready = TaskEvent()
        +manager.writer.ready  # Set it

        manager.transcription.__call__ = mock_call
        manager.transcription.ready = TaskEvent()
        +manager.transcription.ready  # Set it

        # Make reader callable
        manager.reader = MagicMock()
        manager.reader.__call__ = mock_call
        manager.reader.ready = TaskEvent()
        +manager.reader.ready  # Set it

        manager.root_tg = MagicMock()
        manager.sub_tg = MagicMock()

        # Mock logger properly
        manager.logger = MagicMock()
        manager.logger.ready = TaskEvent()
        +manager.logger.ready  # Set it

        await manager.start_system()

        # Verify components were initialized
        assert manager.stat.run_banner.called

    @pytest.mark.asyncio
    async def test_boot_success(self):
        """Test successful boot"""
        manager = Manager(cfg=self.mock_config, credentials=self.mock_credentials)

        # Mock start_system to complete quickly
        manager.start_system = AsyncMock()

        await manager.boot()

        manager.start_system.assert_called_once()

    @pytest.mark.asyncio
    async def test_boot_timeout(self):
        """Test boot timeout error"""
        manager = Manager(cfg=self.mock_config, credentials=self.mock_credentials)

        # Mock start_system to hang indefinitely
        async def hanging_start():
            await asyncio.sleep(100)  # Much longer than timeout

        manager.start_system = hanging_start

        # Temporarily patch BOOT_TIMEOUT to a smaller value for testing
        with patch('palabra_ai.task.manager.BOOT_TIMEOUT', 0.1):
            with pytest.raises(ConfigurationError) as exc_info:
                await manager.boot()

            assert "Timeout 0.1s while starting tasks" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_do_normal_exit(self):
        """Test normal exit from do method when EOF received"""
        manager = Manager(cfg=self.mock_config, credentials=self.mock_credentials)

        # Mock stopper
        manager.stopper = False
        manager.tasks = [MagicMock(eof=False, stopper=False)]
        manager.graceful_exit = AsyncMock()

        # Simulate EOF after first sleep
        async def side_effect():
            await asyncio.sleep(0.01)
            manager.tasks[0].eof = True

        with patch('asyncio.sleep', side_effect=side_effect):
            await manager.do()

        manager.graceful_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_task(self):
        """Test shutdown_task method"""
        manager = Manager(cfg=self.mock_config, credentials=self.mock_credentials)

        # Create mock task
        mock_task = MagicMock()
        mock_task.name = "test_task"
        mock_task.stopper = TaskEvent()

        # Create a real coroutine that completes quickly
        async def quick_task():
            await asyncio.sleep(0.01)

        mock_task._task = asyncio.create_task(quick_task())

        await manager.shutdown_task(mock_task, timeout=1)

        assert mock_task.stopper.is_set()


    @pytest.mark.asyncio
    async def test_graceful_exit(self):
        """Test graceful_exit method"""
        manager = Manager(cfg=self.mock_config, credentials=self.mock_credentials)

        # Mock components
        manager.reader = MagicMock()
        manager.transcription = MagicMock()
        manager.io = MagicMock()
        manager.writer = MagicMock()
        manager.writer.stopper = TaskEvent()
        manager.stopper = TaskEvent()

        manager.shutdown_task = AsyncMock()

        await manager.graceful_exit()

        # Verify shutdown_task was called for each component
        assert manager.shutdown_task.call_count >= 3
        assert manager.writer.stopper.is_set()
        assert manager.stopper.is_set()

    @pytest.mark.asyncio
    async def test_writer_mercy(self):
        """Test writer_mercy method"""
        manager = Manager(cfg=self.mock_config, credentials=self.mock_credentials)

        # Mock writer
        manager.writer = MagicMock()
        manager.writer.stopper = TaskEvent()

        # Create a completed task
        async def completed_task():
            pass

        task = asyncio.create_task(completed_task())
        await task  # Ensure it's done
        manager.writer._task = task

        await manager.writer_mercy()

        assert manager.writer.stopper.is_set()

    @pytest.mark.asyncio
    async def test_exit(self):
        """Test exit method"""
        manager = Manager(cfg=self.mock_config, credentials=self.mock_credentials)

        # Mock components
        manager.writer_mercy = AsyncMock()
        manager.cancel_all_subtasks = AsyncMock()
        manager._show_banner_loop = MagicMock()
        manager.stat = MagicMock()
        manager.stat.stopper = TaskEvent()
        manager.stopper = TaskEvent()
        manager.tasks = []

        await manager.exit()

        manager.writer_mercy.assert_called_once()
        manager.cancel_all_subtasks.assert_called_once()
        manager._show_banner_loop.cancel.assert_called_once()
        assert manager.stopper.is_set()
        assert manager.stat.stopper.is_set()
