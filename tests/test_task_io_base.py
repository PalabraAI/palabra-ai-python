import asyncio
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from palabra_ai.task.io.base import Io
from palabra_ai.audio import AudioFrame
from palabra_ai.enum import Channel, Direction
from palabra_ai.message import (
    Message, EndTaskMessage, SetTaskMessage, GetTaskMessage, CurrentTaskMessage, ErrorMessage
)
from palabra_ai.constant import BYTES_PER_SAMPLE, SLEEP_INTERVAL_LONG
from palabra_ai.util.fanout_queue import FanoutQueue

class ConcreteIo(Io):
    """Concrete implementation of Io for testing"""

    @property
    def channel(self) -> Channel:
        return Channel.WS

    async def send_frame(self, frame) -> None:
        pass

    async def send_message(self, msg_data: bytes) -> None:
        pass

    async def boot(self):
        pass

    async def do(self):
        await super().do()

    async def exit(self):
        pass

class TestIo:
    """Test Io abstract base class"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = MagicMock()
        config.mode = MagicMock()
        config.mode.input_chunk_bytes = 320
        config.mode.input_chunk_duration_ms = 20
        config.mode.input_samples_per_channel = 160
        config.mode.for_input_audio_frame = (8000, 1, 160)
        config.to_dict = MagicMock(return_value={"test": "config"})
        return config

    @pytest.fixture
    def mock_credentials(self):
        """Create mock credentials"""
        return MagicMock()

    @pytest.fixture
    def mock_reader(self):
        """Create mock reader"""
        reader = MagicMock()
        from palabra_ai.task.base import TaskEvent
        reader.ready = TaskEvent()
        reader.ready.set()
        reader.read = AsyncMock()
        return reader

    @pytest.fixture
    def mock_writer(self):
        """Create mock writer"""
        writer = MagicMock()
        writer.q = asyncio.Queue()
        return writer

    def test_init(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test Io initialization"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        assert io.cfg == mock_config
        assert io.credentials == mock_credentials
        assert io.reader == mock_reader
        assert io.writer == mock_writer
        assert isinstance(io.in_msg_foq, FanoutQueue)
        assert isinstance(io.out_msg_foq, FanoutQueue)

    def test_channel_property(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test channel property"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        assert io.channel == Channel.WS

    @pytest.mark.asyncio
    async def test_push_in_msg(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test push_in_msg method"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        msg = EndTaskMessage()

        with patch('palabra_ai.task.io.base.debug') as mock_debug:
            await io.push_in_msg(msg)

            # Check debug info was set
            assert msg._dbg is not None
            assert msg._dbg.ch == Channel.WS
            assert msg._dbg.dir == Direction.IN

            # Check message was published
            mock_debug.assert_called_once()
            assert "Pushing message" in str(mock_debug.call_args)

    def test_new_frame(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test new_input_frame method"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        frame = io.new_input_frame()

        assert isinstance(frame, AudioFrame)
        assert frame.sample_rate == 8000
        assert frame.num_channels == 1
        assert frame.samples_per_channel == 160

    @pytest.mark.asyncio
    async def test_push(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test push method"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        io.send_frame = AsyncMock()

        # Create audio data (320 bytes = 160 samples * 2 bytes per sample)
        audio_bytes = np.random.randint(-32768, 32767, 160, dtype=np.int16).tobytes()

        await io.push(audio_bytes)

        # Should send one frame
        io.send_frame.assert_called_once()
        frame = io.send_frame.call_args[0][0]
        assert isinstance(frame, AudioFrame)

    @pytest.mark.asyncio
    async def test_push_with_padding(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test push method with audio that needs padding"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        io.send_frame = AsyncMock()

        # Create partial audio data (100 bytes < 320 bytes)
        audio_bytes = np.random.randint(-32768, 32767, 50, dtype=np.int16).tobytes()

        await io.push(audio_bytes)

        # Should still send one frame with padding
        io.send_frame.assert_called_once()

    @pytest.mark.asyncio
    async def test_push_multiple_frames(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test push method with multiple frames worth of audio"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        io.send_frame = AsyncMock()

        # Create audio data for 2 frames (640 bytes = 320 samples * 2 bytes)
        audio_bytes = np.random.randint(-32768, 32767, 320, dtype=np.int16).tobytes()

        await io.push(audio_bytes)

        # Should send two frames
        assert io.send_frame.call_count == 2

    @pytest.mark.asyncio
    async def test_exit(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test _exit method"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        await io._exit()

        # Should put None in writer queue
        assert mock_writer.q.qsize() == 1
        assert await mock_writer.q.get() is None

    @pytest.mark.asyncio
    async def test_set_task_success(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test set_task method with successful response"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        io.push_in_msg = AsyncMock()

        # Mock subscription to return CurrentTaskMessage
        async def mock_receiver():
            yield CurrentTaskMessage(timestamp=0.0, data={"task": "test"})

        with patch.object(io.out_msg_foq, 'receiver') as mock_receiver_ctx:
            mock_receiver_ctx.return_value.__aenter__.return_value = mock_receiver()

            with patch('palabra_ai.task.io.base.debug') as mock_debug:
                await io.set_task()

                # Check messages were sent
                assert io.push_in_msg.call_count >= 2  # SetTaskMessage and GetTaskMessage

                # Check debug messages
                assert any("Setting task configuration" in str(call) for call in mock_debug.call_args_list)
                assert any("Received current task" in str(call) for call in mock_debug.call_args_list)

    @pytest.mark.asyncio
    async def test_set_task_not_found_error_retry(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test set_task handles NOT_FOUND errors and retries"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        io.push_in_msg = AsyncMock()

        # Mock subscription to return NOT_FOUND error then success
        async def mock_receiver():
            # First return NOT_FOUND error
            error_msg = ErrorMessage(
                message_type="error",
                timestamp=0.0,
                raw={"data": {"code": "NOT_FOUND", "desc": "No active task found"}},
                data={"data": {"code": "NOT_FOUND", "desc": "No active task found"}}
            )
            yield error_msg

            # Then return success
            yield CurrentTaskMessage(timestamp=0.0, data={"task": "test"})

        with patch.object(io.out_msg_foq, 'receiver') as mock_receiver_ctx:
            mock_receiver_ctx.return_value.__aenter__.return_value = mock_receiver()

            with patch('palabra_ai.task.io.base.debug') as mock_debug:
                await io.set_task()

                # Verify NOT_FOUND was logged but didn't cause immediate failure
                debug_calls = [str(call) for call in mock_debug.call_args_list]
                assert any("Got NOT_FOUND error, will retry" in call for call in debug_calls)
                assert any("set_task() SUCCESS" in call for call in debug_calls)

    @pytest.mark.asyncio
    async def test_set_task_other_error_immediate_failure(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test set_task raises immediately for non-NOT_FOUND errors"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        io.push_in_msg = AsyncMock()

        # Mock subscription to return other error
        async def mock_receiver():
            error_msg = MagicMock(spec=ErrorMessage)
            error_msg.data = {"data": {"code": "SERVER_ERROR", "desc": "Internal server error"}}
            error_msg.raise_ = MagicMock(side_effect=RuntimeError("Server error"))
            yield error_msg

        with patch.object(io.out_msg_foq, 'receiver') as mock_receiver_ctx:
            mock_receiver_ctx.return_value.__aenter__.return_value = mock_receiver()

            with pytest.raises(RuntimeError, match="Server error"):
                await io.set_task()

    @pytest.mark.asyncio
    async def test_set_task_debug_logging(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test set_task produces expected debug messages"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        io.push_in_msg = AsyncMock()

        # Mock subscription to return success immediately
        async def mock_receiver():
            yield CurrentTaskMessage(timestamp=0.0, data={"task": "test"})

        with patch.object(io.out_msg_foq, 'receiver') as mock_receiver_ctx:
            mock_receiver_ctx.return_value.__aenter__.return_value = mock_receiver()

            with patch('palabra_ai.task.io.base.debug') as mock_debug:
                await io.set_task()

                # Check for new debug messages
                debug_calls = [str(call) for call in mock_debug.call_args_list]
                assert any("set_task() STARTED" in call for call in debug_calls)
                assert any("set_task() creating receiver" in call for call in debug_calls)
                assert any("set_task() receiver created" in call for call in debug_calls)

    @pytest.mark.asyncio
    async def test_set_task_timeout(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test set_task method with timeout"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        io.push_in_msg = AsyncMock()

        # Mock subscription to return wrong message type
        async def mock_receiver():
            # Return non-CurrentTaskMessage until timeout
            while True:
                yield EndTaskMessage()
                await asyncio.sleep(0.01)

        with patch.object(io.out_msg_foq, 'receiver') as mock_receiver_ctx:
            mock_receiver_ctx.return_value.__aenter__.return_value = mock_receiver()

            with patch('palabra_ai.task.io.base.BOOT_TIMEOUT', 0.1):  # Short timeout for test
                with patch('palabra_ai.task.io.base.debug') as mock_debug:
                    with pytest.raises(TimeoutError, match="Timeout waiting for task configuration"):
                        await io.set_task()

                    # Check timeout message was logged
                    assert any("Timeout waiting for task configuration" in str(call)
                              for call in mock_debug.call_args_list)

    def test_calc_rms_db_static_method(self):
        """Test calc_rms_db static method"""
        # Create test audio frame with known values
        audio_data = np.array([16384, -16384, 0, 32767], dtype=np.int16)
        audio_frame = MagicMock()
        audio_frame.data = audio_data.tobytes()

        rms_db = Io.calc_rms_db(audio_frame)

        # Should return a reasonable dB value
        assert isinstance(rms_db, float)
        assert rms_db > -50  # Should not be too quiet
        assert rms_db < 10   # Should not be too loud

    def test_calc_rms_db_silent_audio(self):
        """Test calc_rms_db with silent audio"""
        # Create silent audio frame
        audio_data = np.zeros(1024, dtype=np.int16)
        audio_frame = MagicMock()
        audio_frame.data = audio_data.tobytes()

        rms_db = Io.calc_rms_db(audio_frame)

        # Silent audio should return -infinity
        assert rms_db == -np.inf

    @pytest.mark.asyncio
    async def test_in_msg_sender(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test in_msg_sender method"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        io.send_message = AsyncMock()

        # Create a message to send
        test_msg = EndTaskMessage()

        # Start the sender task
        sender_task = asyncio.create_task(io.in_msg_sender())

        # Give it time to start
        await asyncio.sleep(0.01)

        # Publish a message
        io.in_msg_foq.publish(test_msg)

        # Give it time to process
        await asyncio.sleep(0.01)

        # Stop the sender by publishing None
        io.in_msg_foq.publish(None)

        # Wait for completion
        await asyncio.wait_for(sender_task, timeout=1.0)

        # Check that send_message was called
        assert io.send_message.call_count >= 1

    @pytest.mark.asyncio
    async def test_timing_initialization(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test that global timing is initialized on first chunk"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        # Ensure timing not initialized
        assert io.global_start_perf_ts is None
        assert io._frames_sent == 0
        assert io._total_duration_sent == 0.0

        # Call init_global_start_ts
        io.init_global_start_ts()

        # Check timing was initialized
        assert io.global_start_perf_ts is not None
        assert isinstance(io.global_start_perf_ts, float)
        assert io.global_start_perf_ts > 0

        # Check writer was updated
        assert mock_writer.start_perf_ts == io.global_start_perf_ts

    @pytest.mark.asyncio
    async def test_single_chunk_timing_precision(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test precise timing for single chunk sending"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        io.push = AsyncMock()
        chunk = b"test_chunk"

        # Initialize timing slightly in the past to force a wait
        io.global_start_perf_ts = time.perf_counter() - 0.001  # 1ms ago
        io._frames_sent = 0
        io._total_duration_sent = 0.0

        # Send single chunk with timing
        with patch('asyncio.sleep') as mock_sleep:
            await io._send_single_chunk(chunk)

            # Should have called sleep with appropriate timing
            if mock_sleep.call_count > 0:
                sleep_time = mock_sleep.call_args[0][0]
                assert 0 <= sleep_time <= 0.02  # Should be between 0 and chunk_duration

        # Check chunk was sent
        io.push.assert_called_once_with(chunk)

        # Check metrics updated
        assert io._frames_sent == 1
        assert io._total_duration_sent == 0.02  # 20ms

    @pytest.mark.asyncio
    async def test_burst_mode_activation(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test that burst mode activates when behind schedule"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        # Initialize timing from past to simulate being behind
        io.global_start_perf_ts = time.perf_counter() - 1.0  # 1 second ago
        io._frames_sent = 0
        io._total_duration_sent = 0.0  # Should have sent 50 chunks by now (1000ms / 20ms)

        # Check that we're behind schedule
        assert io._is_behind_schedule() is True

        # Calculate timing metrics
        target_time, current_time, time_behind = io._calculate_timing_metrics()
        assert time_behind > 0.02  # Should be more than one chunk behind

    @pytest.mark.asyncio
    async def test_burst_mode_chunk_sending(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test burst mode sends multiple chunks at once"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        io._send_chunk_immediately = AsyncMock()
        io._read_next_chunk = AsyncMock(return_value=b"next_chunk")

        # Initialize timing to be significantly behind (500ms behind = 25 chunks)
        io.global_start_perf_ts = time.perf_counter() - 0.5
        io._frames_sent = 0
        io._total_duration_sent = 0.0

        with patch('palabra_ai.task.io.base.debug') as mock_debug:
            await io._send_burst_chunks(b"initial_chunk")

            # Should send up to MAX_BURST (20) chunks
            assert io._send_chunk_immediately.call_count == 20

            # Check debug message
            debug_calls = [str(call) for call in mock_debug.call_args_list]
            assert any("BURST:" in call for call in debug_calls)
            assert any("25 chunks" in call for call in debug_calls)  # Should mention being 25 chunks behind
            assert any("sending 20 chunks" in call for call in debug_calls)  # But only sending MAX_BURST

    @pytest.mark.asyncio
    async def test_burst_mode_limited_by_max(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test burst mode respects MAX_BURST limit"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        io._send_chunk_immediately = AsyncMock()
        io._read_next_chunk = AsyncMock(return_value=b"next_chunk")

        # Initialize timing to be very far behind (2 seconds = 100 chunks)
        io.global_start_perf_ts = time.perf_counter() - 2.0
        io._frames_sent = 0
        io._total_duration_sent = 0.0

        await io._send_burst_chunks(b"initial_chunk")

        # Should still only send MAX_BURST (20) chunks
        assert io._send_chunk_immediately.call_count == 20

    @pytest.mark.asyncio
    async def test_timing_metrics_calculation(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test accurate calculation of timing metrics"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        # Set known timing state
        start_time = time.perf_counter()
        io.global_start_perf_ts = start_time
        io._total_duration_sent = 0.1  # 100ms sent

        target_time, current_time, time_behind = io._calculate_timing_metrics()

        # Target time should be start + duration sent
        assert target_time == start_time + 0.1

        # Current time should be approximately now
        assert abs(current_time - time.perf_counter()) < 0.001

        # Time behind can be negative (if we're ahead) or positive (if behind)
        assert isinstance(time_behind, float)

    @pytest.mark.asyncio
    async def test_do_method_integration(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test the main do() loop with timing"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        # Mock reader to return chunks then EOF
        chunks = [b"chunk1", b"chunk2", b"chunk3", None]
        mock_reader.read.side_effect = chunks

        io.push = AsyncMock()
        io.push_in_msg = AsyncMock()

        with patch('asyncio.sleep') as mock_sleep:
            await io.do()

            # Should have processed 3 chunks
            assert io.push.call_count == 3

            # Should have updated metrics
            assert io._frames_sent == 3
            assert io._total_duration_sent == 0.06  # 3 * 20ms

            # Should have sent EOF message
            io.push_in_msg.assert_called_once()
            assert isinstance(io.push_in_msg.call_args[0][0], EndTaskMessage)

    @pytest.mark.asyncio
    async def test_no_drift_over_time(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test that timing doesn't drift over many chunks"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        io.push = AsyncMock()

        # Initialize timing
        start_time = time.perf_counter()
        io.global_start_perf_ts = start_time
        io._frames_sent = 0
        io._total_duration_sent = 0.0

        # Simulate sending 100 chunks
        for i in range(100):
            await io._send_chunk_immediately(b"chunk")

        # Check that total duration matches expected (within floating point precision)
        expected_duration = 100 * 0.02  # 100 chunks * 20ms
        assert abs(io._total_duration_sent - expected_duration) < 1e-10
        assert io._frames_sent == 100

        # The target time for next chunk should be exactly start + total duration
        target_time, _, _ = io._calculate_timing_metrics()
        assert abs(target_time - (start_time + expected_duration)) < 1e-10

    @pytest.mark.asyncio
    async def test_handles_eof_correctly(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test EOF handling in the main loop"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        # Mock reader to return None (EOF)
        mock_reader.read.return_value = None

        io.push_in_msg = AsyncMock()

        await io.do()

        # Should handle EOF
        assert io.eof.is_set()

        # Should send EndTaskMessage
        io.push_in_msg.assert_called_once()
        assert isinstance(io.push_in_msg.call_args[0][0], EndTaskMessage)

    @pytest.mark.asyncio
    async def test_empty_chunk_handling(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test that empty chunks are skipped"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        # Mock reader to return empty chunk, real chunk, then EOF
        mock_reader.read.side_effect = [b"", b"real_chunk", None]

        io.push = AsyncMock()
        io.push_in_msg = AsyncMock()

        await io.do()

        # Should only process the real chunk
        assert io.push.call_count == 1
        io.push.assert_called_with(b"real_chunk")

    def test_eos_received_field_default(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test Io eos_received field defaults to False"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        assert hasattr(io, 'eos_received')
        assert io.eos_received is False

    def test_eos_received_field_can_be_set(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test Io eos_received field can be set to True"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )

        io.eos_received = True
        assert io.eos_received is True