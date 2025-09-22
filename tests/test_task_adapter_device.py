import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, create_autospec, PropertyMock
import numpy as np
from palabra_ai.task.adapter.device import DeviceReader, DeviceWriter, Device
from palabra_ai.task.base import TaskEvent
from palabra_ai.audio import AudioFrame
from palabra_ai.internal.device import SoundDeviceManager


class TestDeviceReader:
    """Test DeviceReader class"""

    @pytest.fixture
    def mock_device(self):
        """Create a mock device"""
        return Device(
            name="Test Microphone",
            id="test_mic_123",
            channels=1,
            sample_rate=16000,
            is_default=True
        )

    @pytest.fixture
    def mock_sdm(self):
        """Create a mock SoundDeviceManager"""
        mock = MagicMock(spec=SoundDeviceManager)
        mock.start_input_device = AsyncMock()
        mock.stop_device = AsyncMock()
        return mock

    def test_init_with_device_object(self, mock_device):
        """Test initialization with Device object"""
        reader = DeviceReader(device=mock_device)
        assert reader.device == mock_device
        assert isinstance(reader.sdm, SoundDeviceManager)
        assert reader.tg is None

    def test_init_with_device_string(self):
        """Test initialization with device string"""
        reader = DeviceReader(device="test_device_id")
        assert reader.device == "test_device_id"
        assert isinstance(reader.sdm, SoundDeviceManager)

    @pytest.mark.asyncio
    async def test_boot_success(self, mock_device, mock_sdm):
        """Test successful boot"""
        reader = DeviceReader(device=mock_device)
        reader.sdm = mock_sdm
        reader.sub_tg = MagicMock()
        reader.cfg = MagicMock()
        reader.cfg.mode.input_sample_rate = 16000
        reader.cfg.mode.channels = 1
        reader.q = asyncio.Queue()
        reader.ready = TaskEvent()

        # Mock signal handling
        with patch('palabra_ai.task.adapter.device.signal.signal'):
            with patch('palabra_ai.task.adapter.device.debug') as mock_debug:
                await reader.boot()

                # Verify sdm.tg was set
                assert reader.sdm.tg == reader.sub_tg

                # Verify start_input_device was called with correct parameters
                mock_sdm.start_input_device.assert_called_once()
                call_args = mock_sdm.start_input_device.call_args
                assert call_args[0][0] == mock_device.name  # device_name
                assert call_args[1]['channels'] == 1
                assert call_args[1]['sample_rate'] == 16000
                assert 'async_callback_fn' in call_args[1]

                assert mock_debug.call_count >= 1

    @pytest.mark.asyncio
    async def test_boot_with_string_device(self, mock_sdm):
        """Test boot with string device ID"""
        reader = DeviceReader(device="test_device_id")
        reader.sdm = mock_sdm
        reader.sub_tg = MagicMock()
        reader.cfg = MagicMock()
        reader.cfg.mode.input_sample_rate = 16000
        reader.cfg.mode.channels = 1
        reader.q = asyncio.Queue()
        reader.ready = TaskEvent()

        with patch('palabra_ai.task.adapter.device.signal.signal'):
            await reader.boot()

            # Verify start_input_device was called with string device
            mock_sdm.start_input_device.assert_called_once()
            call_args = mock_sdm.start_input_device.call_args
            assert call_args[0][0] == "test_device_id"

    @pytest.mark.asyncio
    async def test_audio_callback(self, mock_device):
        """Test audio callback puts data in queue"""
        reader = DeviceReader(device=mock_device)
        reader.q = asyncio.Queue()

        test_data = b"test audio data"
        await reader._audio_callback(test_data)

        # Verify data was put in queue
        assert reader.q.qsize() == 1
        result = await reader.q.get()
        assert result == test_data

    @pytest.mark.asyncio
    async def test_read_from_queue(self, mock_device):
        """Test reading from queue"""
        reader = DeviceReader(device=mock_device)
        reader.q = asyncio.Queue()
        reader.ready = TaskEvent()
        +reader.ready

        # Put data in queue
        test_data = b"test audio data"
        await reader.q.put(test_data)

        # Read should get data from queue
        result = await reader.read()
        assert result == test_data

    @pytest.mark.asyncio
    async def test_read_timeout(self, mock_device):
        """Test read with timeout"""
        reader = DeviceReader(device=mock_device)
        reader.q = asyncio.Queue()
        reader.ready = TaskEvent()
        +reader.ready

        # Empty queue should timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(reader.read(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_exit_success(self, mock_device, mock_sdm):
        """Test successful exit"""
        reader = DeviceReader(device=mock_device)
        reader.sdm = mock_sdm

        mock_sdm.stop_input_device = MagicMock()

        await reader.exit()

        mock_sdm.stop_input_device.assert_called_once_with(mock_device.name)

    @pytest.mark.asyncio
    async def test_exit_with_string_device(self, mock_sdm):
        """Test exit with string device"""
        reader = DeviceReader(device="test_device_id")
        reader.sdm = mock_sdm

        mock_sdm.stop_input_device = MagicMock()

        await reader.exit()

        mock_sdm.stop_input_device.assert_called_once_with("test_device_id")

    @pytest.mark.asyncio
    async def test_exit_with_error(self, mock_device, mock_sdm):
        """Test exit with error during stop"""
        reader = DeviceReader(device=mock_device)
        reader.sdm = mock_sdm

        mock_sdm.stop_input_device = MagicMock(side_effect=RuntimeError("Stop failed"))

        with patch('palabra_ai.task.adapter.device.error') as mock_error:
            await reader.exit()

            mock_error.assert_called_once()
            assert "Error stopping input device" in str(mock_error.call_args[0][0])


class TestDeviceWriter:
    """Test DeviceWriter class"""

    @pytest.fixture
    def mock_device(self):
        """Create a mock device"""
        return Device(
            name="Test Speaker",
            id="test_spk_123",
            channels=1,
            sample_rate=16000,
            is_default=True
        )

    @pytest.fixture
    def mock_sdm(self):
        """Create a mock SoundDeviceManager"""
        mock = MagicMock(spec=SoundDeviceManager)
        mock.start_output_device = MagicMock()
        mock.stop_output_device = MagicMock()
        mock.write = MagicMock()
        return mock

    def test_init_with_device_object(self, mock_device):
        """Test initialization with Device object"""
        writer = DeviceWriter(device=mock_device)
        assert writer.device == mock_device
        assert isinstance(writer._sdm, SoundDeviceManager)
        assert writer._output_device is None
        assert writer._loop is None

    def test_init_with_device_string(self):
        """Test initialization with device string"""
        writer = DeviceWriter(device="test_device_id")
        assert writer.device == "test_device_id"
        assert isinstance(writer._sdm, SoundDeviceManager)

    @pytest.mark.asyncio
    async def test_boot_success(self, mock_device, mock_sdm):
        """Test successful boot"""
        writer = DeviceWriter(device=mock_device)
        writer._sdm = mock_sdm
        writer.sub_tg = MagicMock()
        writer.cfg = MagicMock()
        writer.cfg.mode.output_sample_rate = 16000
        writer.cfg.mode.num_channels = 1

        mock_output_device = MagicMock()
        mock_sdm.start_output_device.return_value = mock_output_device

        with patch('palabra_ai.task.adapter.device.asyncio.get_running_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            # Mock super().boot()
            with patch('palabra_ai.task.adapter.device.super') as mock_super:
                mock_super_obj = MagicMock()
                mock_super.return_value = mock_super_obj
                mock_super_obj.boot = AsyncMock()

                await writer.boot()

                # Verify sdm.tg was set
                assert writer._sdm.tg == writer.sub_tg

                # Verify start_output_device was called
                mock_sdm.start_output_device.assert_called_once_with(
                    mock_device.name,
                    channels=1,
                    sample_rate=16000
                )

                # Verify output device was stored
                assert writer._output_device == mock_output_device
                assert writer._loop == mock_loop

    @pytest.mark.asyncio
    async def test_boot_with_string_device(self, mock_sdm):
        """Test boot with string device ID"""
        writer = DeviceWriter(device="test_device_id")
        writer._sdm = mock_sdm
        writer.sub_tg = MagicMock()
        writer.cfg = MagicMock()
        writer.cfg.mode.output_sample_rate = 16000
        writer.cfg.mode.num_channels = 1

        mock_output_device = MagicMock()
        mock_sdm.start_output_device.return_value = mock_output_device

        with patch('palabra_ai.task.adapter.device.asyncio.get_running_loop'):
            # Mock super().boot()
            with patch('palabra_ai.task.adapter.device.super') as mock_super:
                mock_super_obj = MagicMock()
                mock_super.return_value = mock_super_obj
                mock_super_obj.boot = AsyncMock()

                await writer.boot()

            # Verify start_output_device was called with string device
            mock_sdm.start_output_device.assert_called_once_with(
                "test_device_id",
                channels=1,
                sample_rate=16000
            )

    @pytest.mark.asyncio
    async def test_write_audio_frame(self, mock_device, mock_sdm):
        """Test writing an audio frame"""
        writer = DeviceWriter(device=mock_device)
        writer._sdm = mock_sdm
        writer._output_device = MagicMock()
        writer._loop = asyncio.get_running_loop()
        writer._executor = MagicMock()

        frame = AudioFrame(np.array([1, 2, 3, 4], dtype=np.int16), 16000, 1, 4)

        # Mock run_in_executor to run the function immediately
        async def mock_run_in_executor(executor, func):
            func()

        writer._loop.run_in_executor = mock_run_in_executor

        # Mock output device
        writer._output_device.add_audio_data = MagicMock()

        await writer.write(frame)

        # Verify add_audio_data was called
        writer._output_device.add_audio_data.assert_called_once()
        call_args = writer._output_device.add_audio_data.call_args
        assert isinstance(call_args[0][0], bytes)
        assert len(call_args[0][0]) == 8  # 4 samples * 2 bytes/sample

    @pytest.mark.asyncio
    async def test_stop_device(self, mock_device, mock_sdm):
        """Test stopping device"""
        writer = DeviceWriter(device=mock_device)
        writer._sdm = mock_sdm
        writer._output_device = MagicMock()

        await writer._stop_device()

        mock_sdm.stop_output_device.assert_called_once_with(mock_device.name)
        # Note: _stop_device doesn't set _output_device to None

    @pytest.mark.asyncio
    async def test_stop_device_no_device(self, mock_device, mock_sdm):
        """Test stopping when no device exists"""
        writer = DeviceWriter(device=mock_device)
        writer._sdm = mock_sdm
        writer._output_device = None

        await writer._stop_device()

        mock_sdm.stop_output_device.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_device_error(self, mock_device, mock_sdm):
        """Test stop device with error"""
        writer = DeviceWriter(device=mock_device)
        writer._sdm = mock_sdm
        writer._output_device = MagicMock()

        mock_sdm.stop_output_device.side_effect = RuntimeError("Close failed")

        with patch('palabra_ai.task.adapter.device.error') as mock_error:
            await writer._stop_device()

            mock_error.assert_called_once()
            # Note: _stop_device doesn't set _output_device to None even on error

    @pytest.mark.asyncio
    async def test_exit_success(self, mock_device, mock_sdm):
        """Test successful exit"""
        writer = DeviceWriter(device=mock_device)
        writer._sdm = mock_sdm
        writer._output_device = MagicMock()
        writer._executor = MagicMock()

        # Mock super().exit()
        with patch('palabra_ai.task.adapter.device.super') as mock_super:
            mock_super_obj = MagicMock()
            mock_super.return_value = mock_super_obj
            mock_super_obj.exit = AsyncMock()

            await writer.exit()

            mock_sdm.stop_output_device.assert_called_once_with(mock_device.name)
            writer._executor.shutdown.assert_called_once_with(wait=False)

    @pytest.mark.asyncio
    async def test_exit_no_device(self, mock_device):
        """Test exit when no device exists"""
        writer = DeviceWriter(device=mock_device)
        writer._output_device = None
        writer._executor = MagicMock()

        # Mock super().exit()
        with patch('palabra_ai.task.adapter.device.super') as mock_super:
            mock_super_obj = MagicMock()
            mock_super.return_value = mock_super_obj
            mock_super_obj.exit = AsyncMock()

            await writer.exit()

            writer._executor.shutdown.assert_called_once_with(wait=False)
