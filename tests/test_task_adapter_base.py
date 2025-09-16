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
        self.ab = AudioBuffer(
            sample_rate=self.cfg.mode.sample_rate,
            num_channels=self.cfg.mode.num_channels,
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
    config.mode.sample_rate = 16000
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


class TestWriterPauseFrame:
    """Test write_pause_frame functionality"""

    @pytest.fixture
    def writer_with_start_time(self, mock_config):
        """Create writer with start_perf_ts set"""
        writer = ConcreteWriter(cfg=mock_config)
        writer.start_perf_ts = 1000.0
        writer._last_sentence_trans_id = None
        writer._last_sentence_end_perf_ts = None
        return writer

    def create_frame_with_timing(self, perf_ts, trans_id="trans1",
                                 last_chunk=False, sample_rate=16000,
                                 samples_per_channel=160):
        """Helper to create frame with timing"""
        data = np.zeros(samples_per_channel, dtype=np.int16)
        frame = AudioFrame(
            data=data,
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=samples_per_channel,
            transcription_id=trans_id,
            last_chunk=last_chunk,
            perf_ts=perf_ts
        )
        return frame

    @pytest.mark.asyncio
    async def test_first_chunk_creates_pause(self, writer_with_start_time):
        """Test that first chunk creates pause from start_perf_ts"""
        writer = writer_with_start_time

        # Create frame that arrives 0.5 seconds after start
        frame = self.create_frame_with_timing(
            perf_ts=1000.5,  # 0.5 seconds after start
            trans_id="trans1",
            last_chunk=False
        )

        with patch.object(writer, 'write', new_callable=AsyncMock) as mock_write:
            await writer.write_pause_frame(frame)

            # Should create pause of 0.5 seconds
            mock_write.assert_called_once()
            pause_frame = mock_write.call_args[0][0]

            # Check pause frame properties
            assert pause_frame.sample_rate == 16000
            assert pause_frame.num_channels == 1
            assert pause_frame.samples_per_channel == 8000  # 0.5 * 16000
            assert np.all(pause_frame.data == 0)  # All zeros (silence)

    @pytest.mark.asyncio
    async def test_middle_chunk_no_pause(self, writer_with_start_time):
        """Test that middle chunk of sentence creates no pause"""
        writer = writer_with_start_time
        writer._last_sentence_trans_id = "trans1"  # Same sentence

        frame = self.create_frame_with_timing(
            perf_ts=1001.0,
            trans_id="trans1",  # Same transcription ID
            last_chunk=False
        )

        with patch.object(writer, 'write', new_callable=AsyncMock) as mock_write:
            await writer.write_pause_frame(frame)

            # Should NOT call write (no pause needed)
            mock_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_last_chunk_updates_state(self, writer_with_start_time):
        """Test that last chunk updates sentence state"""
        writer = writer_with_start_time
        writer._last_sentence_trans_id = "trans1"

        frame = self.create_frame_with_timing(
            perf_ts=1001.0,
            trans_id="trans1",
            last_chunk=True,  # Last chunk of sentence
            samples_per_channel=160  # 0.01 seconds at 16kHz
        )

        with patch.object(writer, 'write', new_callable=AsyncMock) as mock_write:
            await writer.write_pause_frame(frame)

            # Should not create pause (same sentence)
            mock_write.assert_not_called()

            # Should update state
            assert writer._last_sentence_trans_id == "trans1"
            assert writer._last_sentence_end_perf_ts == 1001.01  # 1001.0 + 0.01

    @pytest.mark.asyncio
    async def test_new_sentence_creates_pause(self, writer_with_start_time):
        """Test that new sentence creates pause from previous sentence end"""
        writer = writer_with_start_time
        writer._last_sentence_trans_id = "trans1"
        writer._last_sentence_end_perf_ts = 1001.0

        # New sentence starts 0.3 seconds after previous ended
        frame = self.create_frame_with_timing(
            perf_ts=1001.3,
            trans_id="trans2",  # Different transcription ID
            last_chunk=False
        )

        with patch.object(writer, 'write', new_callable=AsyncMock) as mock_write:
            await writer.write_pause_frame(frame)

            # Should create pause of 0.3 seconds
            mock_write.assert_called_once()
            pause_frame = mock_write.call_args[0][0]

            assert pause_frame.samples_per_channel == int((1001.3 - 1001.0) * 16000)  # Actual computed value

    @pytest.mark.asyncio
    async def test_single_chunk_sentence(self, writer_with_start_time):
        """Test chunk that is both first and last of sentence"""
        writer = writer_with_start_time

        frame = self.create_frame_with_timing(
            perf_ts=1000.5,
            trans_id="trans1",
            last_chunk=True,  # Both first and last
            samples_per_channel=160
        )

        with patch.object(writer, 'write', new_callable=AsyncMock) as mock_write:
            await writer.write_pause_frame(frame)

            # Should create pause AND update state
            mock_write.assert_called_once()
            assert writer._last_sentence_trans_id == "trans1"
            assert writer._last_sentence_end_perf_ts == 1000.51  # 1000.5 + 0.01

    @pytest.mark.asyncio
    async def test_no_start_time_safe_exit(self, mock_config):
        """Test safe exit when start_perf_ts not set"""
        writer = ConcreteWriter(cfg=mock_config)
        # start_perf_ts is None

        frame = self.create_frame_with_timing(perf_ts=1000.0)

        with patch.object(writer, 'write', new_callable=AsyncMock) as mock_write:
            await writer.write_pause_frame(frame)

            # Should exit safely without calling write
            mock_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_negative_pause_duration(self, writer_with_start_time):
        """Test handling of negative pause duration"""
        writer = writer_with_start_time
        writer._last_sentence_end_perf_ts = 1001.0

        # Frame arrives BEFORE expected (negative duration)
        frame = self.create_frame_with_timing(
            perf_ts=1000.5,  # Before last sentence end
            trans_id="trans2"
        )

        with patch.object(writer, 'write', new_callable=AsyncMock) as mock_write:
            await writer.write_pause_frame(frame)

            # Should not create pause for negative duration
            mock_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_very_short_pause(self, writer_with_start_time):
        """Test very short pause handling"""
        writer = writer_with_start_time

        # Very short pause (1ms)
        frame = self.create_frame_with_timing(
            perf_ts=1000.001,  # 1ms after start
            trans_id="trans1"
        )

        with patch.object(writer, 'write', new_callable=AsyncMock) as mock_write:
            await writer.write_pause_frame(frame)

            # Should create very small pause
            mock_write.assert_called_once()
            pause_frame = mock_write.call_args[0][0]
            assert pause_frame.samples_per_channel == int((1000.001 - 1000.0) * 16000)  # Actual computed value

    @pytest.mark.asyncio
    async def test_very_long_pause(self, writer_with_start_time):
        """Test very long pause handling"""
        writer = writer_with_start_time

        # Very long pause (10 seconds)
        frame = self.create_frame_with_timing(
            perf_ts=1010.0,  # 10 seconds after start
            trans_id="trans1"
        )

        with patch.object(writer, 'write', new_callable=AsyncMock) as mock_write:
            await writer.write_pause_frame(frame)

            # Should create long pause
            mock_write.assert_called_once()
            pause_frame = mock_write.call_args[0][0]
            assert pause_frame.samples_per_channel == 160000  # 10.0 * 16000

    @pytest.mark.asyncio
    async def test_zero_duration_no_pause(self, writer_with_start_time):
        """Test that zero duration creates no pause"""
        writer = writer_with_start_time
        writer._last_sentence_end_perf_ts = 1001.0

        # Frame arrives at exact same time as last sentence end
        frame = self.create_frame_with_timing(
            perf_ts=1001.0,  # Exact same time
            trans_id="trans2"
        )

        with patch.object(writer, 'write', new_callable=AsyncMock) as mock_write:
            await writer.write_pause_frame(frame)

            # Should not create pause for zero duration
            mock_write.assert_not_called()

    def test_create_silence_method(self):
        """Test AudioFrame.create_silence method"""
        # Create donor frame
        donor_data = np.ones(160, dtype=np.int16)
        donor_frame = AudioFrame(
            data=donor_data,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=160
        )

        # Create 0.5 second silence
        pause_start = 1000.0
        silence_frame = AudioFrame.create_silence(donor_frame, 0.5, pause_start)

        assert silence_frame is not None
        assert silence_frame.sample_rate == 16000
        assert silence_frame.num_channels == 1
        assert silence_frame.samples_per_channel == 8000  # 0.5 * 16000
        assert np.all(silence_frame.data == 0)  # All zeros
        assert len(silence_frame.data) == 8000  # Total samples
        assert silence_frame.perf_ts == pause_start  # Correct timing

    def test_create_silence_invalid_duration(self):
        """Test create_silence with invalid duration"""
        donor_data = np.ones(160, dtype=np.int16)
        donor_frame = AudioFrame(
            data=donor_data,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=160
        )

        # Test zero duration
        assert AudioFrame.create_silence(donor_frame, 0.0, 1000.0) is None

        # Test negative duration
        assert AudioFrame.create_silence(donor_frame, -0.1, 1000.0) is None
