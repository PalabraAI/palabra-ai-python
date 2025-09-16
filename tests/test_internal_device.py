import asyncio
import queue
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
from palabra_ai.internal.device import (
    batch,
    InputSoundDevice,
    OutputSoundDevice,
    SoundDeviceManager
)
from palabra_ai.constant import SAMPLE_RATE_DEFAULT, SAMPLE_RATE_HALF, AUDIO_CHUNK_SECONDS


def test_batch():
    """Test batch function"""
    # Test basic batching
    result = list(batch([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]

    # Test batch size 1
    result = list(batch([1, 2, 3], 1))
    assert result == [[1], [2], [3]]

    # Test batch size larger than sequence
    result = list(batch([1, 2], 5))
    assert result == [[1, 2]]

    # Test empty sequence
    result = list(batch([], 2))
    assert result == []


class TestInputSoundDevice:
    """Test InputSoundDevice class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.mock_tg = MagicMock()
        self.mock_manager = MagicMock()
        self.device = InputSoundDevice(self.mock_tg, "test_device", self.mock_manager)

    def test_init(self):
        """Test initialization"""
        assert self.device.name == "test_device"
        assert self.device.manager == self.mock_manager
        assert self.device.reading_device is False
        assert self.device.stream_latency == -1
        assert isinstance(self.device.buffer, queue.Queue)

    def test_get_read_delay_ms(self):
        """Test get_read_delay_ms method"""
        self.device.stream_latency = 0.05
        self.device.audio_chunk_seconds = 0.01

        delay = self.device.get_read_delay_ms()
        # 0.05 + 0.01 + 0.01 = 0.07 seconds = 70ms
        assert delay == 70

    @pytest.mark.asyncio
    async def test_start_reading(self):
        """Test start_reading method"""
        mock_callback = AsyncMock()

        # Mock device info
        self.mock_manager.get_device_info.return_value = {
            "input_devices": {
                "test_device": {"index": 0}
            }
        }

        # Mock threading
        with patch('palabra_ai.internal.device.threading.Thread') as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            # Start reading in a task
            task = asyncio.create_task(
                self.device.start_reading(mock_callback, SAMPLE_RATE_DEFAULT, 2, AUDIO_CHUNK_SECONDS)
            )

            # Simulate stream becoming ready
            await asyncio.sleep(0.01)
            self.device.stream_latency = 0.1

            # Wait for start_reading to complete
            await task

            # Verify
            assert self.device.sample_rate == SAMPLE_RATE_DEFAULT
            assert self.device.channels == 2
            assert self.device.audio_chunk_seconds == AUDIO_CHUNK_SECONDS
            assert self.device.async_callback_fn == mock_callback
            mock_thread_instance.start.assert_called_once()

    def test_stop_reading(self):
        """Test stop_reading method"""
        # Setup mocks
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        self.device.device_reading_thread = mock_thread

        mock_task = MagicMock()
        mock_task.done.return_value = False
        self.device.callback_task = mock_task

        with patch('asyncio.get_event_loop') as mock_loop:
            # Stop reading
            self.device.stop_reading(timeout=5)

            # Verify
            assert self.device.reading_device is False
            assert self.device.stream_latency == -1
            mock_thread.join.assert_called_once_with(timeout=5)
            mock_task.cancel.assert_called_once()

    def test_push_to_buffer(self):
        """Test _push_to_buffer method"""
        audio_bytes = b"test audio"
        self.device._push_to_buffer(audio_bytes)

        # Verify data was added to buffer
        assert not self.device.buffer.empty()
        assert self.device.buffer.get() == audio_bytes

    @pytest.mark.asyncio
    async def test_run_callback_worker(self):
        """Test _run_callback_worker method"""
        mock_callback = AsyncMock()
        self.device.async_callback_fn = mock_callback

        # Add data to buffer
        self.device.buffer.put(b"test1")
        self.device.buffer.put(b"test2")

        # Run worker in a task
        task = asyncio.create_task(self.device._run_callback_worker())

        # Let it process
        await asyncio.sleep(0.2)

        # Cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify callbacks were called
        assert mock_callback.call_count >= 2
        mock_callback.assert_any_call(b"test1")
        mock_callback.assert_any_call(b"test2")


class TestOutputSoundDevice:
    """Test OutputSoundDevice class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.mock_manager = MagicMock()
        self.device = OutputSoundDevice("test_output", self.mock_manager)

    def test_init(self):
        """Test initialization"""
        assert self.device.name == "test_output"
        assert self.device.manager == self.mock_manager
        assert self.device.writing_device is False
        assert self.device.block_size > 0

    @patch('palabra_ai.internal.device.threading.Thread')
    def test_start_writing(self, mock_thread):
        """Test start_writing method"""
        # Mock device info
        self.mock_manager.get_device_info.return_value = {
            "output_devices": {
                "test_output": {"index": 1}
            }
        }

        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        # Start writing
        self.device.start_writing(channels=1, sample_rate=SAMPLE_RATE_HALF)

        # Verify
        assert self.device.device_ix == 1
        assert self.device.channels == 1
        assert self.device.sample_rate == SAMPLE_RATE_HALF
        assert isinstance(self.device.write_buffer, queue.Queue)
        mock_thread_instance.start.assert_called_once()

    def test_stop_writing(self):
        """Test stop_writing method"""
        # Setup mocks
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        self.device.device_writing_thread = mock_thread
        self.device.write_buffer = queue.Queue()

        # Stop writing
        self.device.stop_writing(timeout=5)

        # Verify
        assert self.device.writing_device is False
        assert self.device.write_buffer is None
        mock_thread.join.assert_called_once_with(timeout=5)

    def test_add_audio_data(self):
        """Test add_audio_data method"""
        # Setup mock stream
        mock_stream = MagicMock()
        mock_stream.samplesize = 2
        self.device.stream = mock_stream
        self.device.channels = 1

        # Add audio data
        audio_data = b"1234567890"
        self.device.add_audio_data(audio_data)

        # Verify write was called
        assert mock_stream.write.called


class TestSoundDeviceManager:
    """Test SoundDeviceManager class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.manager = SoundDeviceManager()

    def test_init(self):
        """Test initialization"""
        assert isinstance(self.manager.input_device_map, dict)
        assert isinstance(self.manager.output_device_map, dict)
        assert len(self.manager.input_device_map) == 0
        assert len(self.manager.output_device_map) == 0

    @patch('palabra_ai.internal.device.sd')
    def test_get_device_info(self, mock_sd):
        """Test get_device_info method"""
        # Mock sounddevice queries
        mock_sd.query_devices.return_value = [
            {"name": "Device1", "max_input_channels": 2, "max_output_channels": 0},
            {"name": "Device2", "max_input_channels": 0, "max_output_channels": 2},
            {"name": "Device3", "max_input_channels": 2, "max_output_channels": 2}
        ]
        mock_sd.query_hostapis.return_value = [
            {"name": "API1", "devices": [0, 1, 2]}
        ]

        # Get device info
        info = self.manager.get_device_info()

        # Verify
        assert "input_devices" in info
        assert "output_devices" in info
        assert len(info["input_devices"]) == 2  # Device1 and Device3
        assert len(info["output_devices"]) == 2  # Device2 and Device3

    @pytest.mark.asyncio
    async def test_start_input_device(self):
        """Test start_input_device method"""
        mock_callback = AsyncMock()
        self.manager.tg = MagicMock()

        with patch.object(InputSoundDevice, 'start_reading', new_callable=AsyncMock) as mock_start:
            # Start device
            device = await self.manager.start_input_device(
                "test_input", mock_callback, SAMPLE_RATE_DEFAULT, 2, AUDIO_CHUNK_SECONDS
            )

            # Verify
            assert device.name == "test_input"
            assert "test_input" in self.manager.input_device_map
            mock_start.assert_called_once_with(
                mock_callback, SAMPLE_RATE_DEFAULT, 2, AUDIO_CHUNK_SECONDS
            )

    def test_start_output_device(self):
        """Test start_output_device method"""
        with patch.object(OutputSoundDevice, 'start_writing') as mock_start:
            # Start device
            device = self.manager.start_output_device("test_output", 1, SAMPLE_RATE_HALF)

            # Verify
            assert device.name == "test_output"
            assert "test_output" in self.manager.output_device_map
            mock_start.assert_called_once_with(1, SAMPLE_RATE_HALF)

    def test_stop_input_device(self):
        """Test stop_input_device method"""
        # Add mock device
        mock_device = MagicMock()
        self.manager.input_device_map["test_input"] = mock_device

        # Stop device
        self.manager.stop_input_device("test_input", timeout=10)

        # Verify
        mock_device.stop_reading.assert_called_once_with(timeout=10)

    def test_stop_output_device(self):
        """Test stop_output_device method"""
        # Add mock device
        mock_device = MagicMock()
        self.manager.output_device_map["test_output"] = mock_device

        # Stop device
        self.manager.stop_output_device("test_output", timeout=10)

        # Verify
        mock_device.stop_writing.assert_called_once_with(timeout=10)

    def test_stop_all(self):
        """Test stop_all method"""
        # Add mock devices
        mock_input = MagicMock()
        mock_output = MagicMock()
        self.manager.input_device_map["input1"] = mock_input
        self.manager.output_device_map["output1"] = mock_output

        # Stop all
        self.manager.stop_all(timeout=15)

        # Verify
        mock_input.stop_reading.assert_called_once_with(timeout=15)
        mock_output.stop_writing.assert_called_once_with(timeout=15)
