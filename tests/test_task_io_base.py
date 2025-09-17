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
        config.mode.chunk_bytes = 320
        config.mode.chunk_duration_ms = 20
        config.mode.samples_per_channel = 160
        config.mode.for_audio_frame = (8000, 1, 160)
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
        """Test new_frame method"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )
        
        frame = io.new_frame()
        
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
    async def test_wait_after_push(self, mock_config, mock_credentials, mock_reader, mock_writer):
        """Test wait_after_push method"""
        io = ConcreteIo(
            cfg=mock_config,
            credentials=mock_credentials,
            reader=mock_reader,
            writer=mock_writer
        )
        
        with patch('asyncio.sleep') as mock_sleep:
            # Simulate 5ms processing time
            await io.wait_after_push(0.005)
            
            # Should sleep for chunk_duration - delta = 20ms - 5ms = 15ms = 0.015s
            mock_sleep.assert_called_once_with(0.015)

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