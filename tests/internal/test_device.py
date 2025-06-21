import asyncio
import threading
import time
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from palabra_ai.internal.device import (
    SoundDeviceManager, InputSoundDevice, OutputSoundDevice, batch
)


def test_batch():
    result = list(batch([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]


class TestInputSoundDevice:
    def test_get_read_delay_ms(self):
        device = InputSoundDevice(MagicMock(), "test", MagicMock())
        device.stream_latency = 0.01
        device.audio_chunk_seconds = 0.5

        delay = device.get_read_delay_ms()
        assert delay == 520  # (0.01 + 0.5 + 0.01) * 1000

    @pytest.mark.asyncio
    async def test_start_reading(self, mock_sounddevice):
        mock_tg = MagicMock()
        mock_task = MagicMock()
        mock_tg.create_task.return_value = mock_task

        manager = MagicMock()
        manager.get_device_info.return_value = {
            "input_devices": {"test": {"index": 0}}
        }

        device = InputSoundDevice(mock_tg, "test", manager)

        with patch("palabra_ai.internal.device.threading.Thread") as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            callback = AsyncMock()

            # Start and immediately set latency
            start_task = asyncio.create_task(device.start_reading(callback))
            await asyncio.sleep(0.01)
            device.stream_latency = 0.1

            await start_task

            assert device.sample_rate == 48000
            assert device.channels == 2
            assert device.async_callback_fn == callback
            mock_thread_instance.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_reading_cancelled(self, mock_sounddevice):
        mock_tg = MagicMock()
        mock_task = MagicMock()
        mock_tg.create_task.return_value = mock_task

        manager = MagicMock()
        manager.get_device_info.return_value = {
            "input_devices": {"test": {"index": 0}}
        }

        device = InputSoundDevice(mock_tg, "test", manager)

        with patch("palabra_ai.internal.device.threading.Thread"):
            task = asyncio.create_task(device.start_reading(AsyncMock()))
            await asyncio.sleep(0.01)
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

    def test_read_from_device(self, mock_sounddevice):
        manager = MagicMock()
        manager.get_device_info.return_value = {
            "input_devices": {"test": {"index": 0}}
        }

        device = InputSoundDevice(MagicMock(), "test", manager)
        device.channels = 2
        device.sample_rate = 48000
        device.audio_chunk_seconds = 0.5

        mock_stream = MagicMock()
        mock_stream.latency = 0.1
        mock_stream.active = True
        mock_sounddevice.RawInputStream.return_value.__enter__.return_value = mock_stream

        # Run briefly
        def stop_reading():
            time.sleep(0.1)
            device.reading_device = False
            mock_stream.active = False

        stop_thread = threading.Thread(target=stop_reading)
        stop_thread.start()

        device._read_from_device_to_buffer()
        stop_thread.join()

        assert device.stream_latency == 0.1

    def test_read_from_device_errors(self, mock_sounddevice):
        manager = MagicMock()
        manager.get_device_info.return_value = {
            "input_devices": {"test": {"index": 0}}
        }

        device = InputSoundDevice(MagicMock(), "test", manager)

        # Exception case
        mock_sounddevice.RawInputStream.side_effect = Exception("Test error")
        device._read_from_device_to_buffer()
        assert not device.reading_device

        # Inactive stream
        mock_sounddevice.RawInputStream.side_effect = None
        mock_stream = MagicMock()
        mock_stream.latency = 0.1
        mock_stream.active = False
        mock_sounddevice.RawInputStream.return_value.__enter__.return_value = mock_stream

        device._read_from_device_to_buffer()
        assert not device.reading_device

    def test_push_to_buffer(self):
        device = InputSoundDevice(MagicMock(), "test", MagicMock())
        device.reading_device = True

        device._push_to_buffer(b"audio")
        assert device.buffer.get_nowait() == b"audio"


class TestOutputSoundDevice:
    def test_start_stop_writing(self, mock_sounddevice):
        manager = MagicMock()
        manager.get_device_info.return_value = {
            "output_devices": {"test": {"index": 0}}
        }

        device = OutputSoundDevice("test", manager)

        with patch("palabra_ai.internal.device.threading.Thread") as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            device.start_writing()

            assert device.channels == 1
            assert device.sample_rate == 24000
            mock_thread_instance.start.assert_called_once()

            device.stop_writing(timeout=1)
            mock_thread_instance.join.assert_called_once()

    def test_add_audio_data(self, mock_sounddevice):
        device = OutputSoundDevice("test", MagicMock())
        device.writing_device = True
        device.channels = 1

        mock_stream = MagicMock()
        mock_stream.samplesize = 2
        device.stream = mock_stream

        device.add_audio_data(b"\x00" * 4096)

        # 4096 bytes / (1024 * 1 * 2) = 2 calls
        assert mock_stream.write.call_count == 2

    def test_write_device(self, mock_sounddevice):
        device = OutputSoundDevice("test", MagicMock())
        device.device_ix = 0
        device.channels = 1
        device.sample_rate = 24000

        mock_stream = MagicMock()
        mock_stream.latency = 0.1
        mock_sounddevice.RawOutputStream.return_value.__enter__.return_value = mock_stream

        def stop_writing():
            time.sleep(0.02)
            device.writing_device = False

        stop_thread = threading.Thread(target=stop_writing)
        stop_thread.start()

        device._write_device()
        stop_thread.join()

        assert device.stream_latency == 0.1

    def test_write_device_exception(self, mock_sounddevice):
        device = OutputSoundDevice("test", MagicMock())
        device.device_ix = 0
        device.channels = 1
        device.sample_rate = 24000

        mock_sounddevice.RawOutputStream.side_effect = Exception("Test error")

        device._write_device()
        assert not device.writing_device


class TestSoundDeviceManager:
    def test_get_device_info(self, mock_sounddevice):
        sdm = SoundDeviceManager()
        info = sdm.get_device_info()

        assert "input_devices" in info
        assert "output_devices" in info
        assert len(info["input_devices"]) > 0

        # With reload
        mock_sounddevice._terminate = MagicMock()
        mock_sounddevice._initialize = MagicMock()

        info = sdm.get_device_info(reload_sd=True)

        mock_sounddevice._terminate.assert_called_once()
        mock_sounddevice._initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_input_device(self, mock_sounddevice):
        async with asyncio.TaskGroup() as tg:
            sdm = SoundDeviceManager()
            sdm.tg = tg

            with patch("palabra_ai.internal.device.InputSoundDevice") as mock_class:
                mock_device = MagicMock()
                mock_device.name = "test (test)"
                mock_device.start_reading = AsyncMock()
                mock_device.stop_reading = MagicMock()
                mock_class.return_value = mock_device

                callback = AsyncMock()
                device = await sdm.start_input_device("test (test)", callback)

                assert device is not None
                assert device.name == "test (test)"
                mock_device.start_reading.assert_called_once()

                sdm.stop_input_device("test (test)")

    @pytest.mark.asyncio
    async def test_start_existing_input_device(self, mock_sounddevice):
        async with asyncio.TaskGroup() as tg:
            sdm = SoundDeviceManager()
            sdm.tg = tg

            existing = MagicMock()
            existing.start_reading = AsyncMock()
            sdm.input_device_map["test"] = existing

            device = await sdm.start_input_device("test", AsyncMock())
            assert device is existing

    @pytest.mark.asyncio
    async def test_start_input_device_cancelled(self, mock_sounddevice):
        async with asyncio.TaskGroup() as tg:
            sdm = SoundDeviceManager()
            sdm.tg = tg

            with patch("palabra_ai.internal.device.InputSoundDevice") as mock_class:
                mock_device = MagicMock()
                mock_device.start_reading = AsyncMock(side_effect=asyncio.CancelledError)
                mock_class.return_value = mock_device

                with pytest.raises(asyncio.CancelledError):
                    await sdm.start_input_device("test", AsyncMock())

    def test_start_output_device(self, mock_sounddevice):
        sdm = SoundDeviceManager()

        with patch("palabra_ai.internal.device.OutputSoundDevice") as mock_class:
            mock_device = MagicMock()
            mock_device.name = "test (test)"
            mock_device.start_writing = MagicMock()
            mock_device.stop_writing = MagicMock()
            mock_class.return_value = mock_device

            device = sdm.start_output_device("test (test)")

            assert device is not None
            assert device.name == "test (test)"
            mock_device.start_writing.assert_called_once()

            sdm.stop_output_device("test (test)")

    def test_stop_all(self):
        sdm = SoundDeviceManager()

        input_device = MagicMock()
        output_device = MagicMock()

        sdm.input_device_map["input"] = input_device
        sdm.output_device_map["output"] = output_device

        sdm.stop_all()

        input_device.stop_reading.assert_called_once_with(timeout=5)
        output_device.stop_writing.assert_called_once_with(timeout=5)
