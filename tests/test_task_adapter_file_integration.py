"""Integration tests for FileReader with real audio files and ffmpeg"""

import asyncio
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import wave

from palabra_ai.config import Config, WsMode
from palabra_ai.task.adapter.file import FileReader
from palabra_ai.task.base import TaskEvent


def create_synthetic_wav(
    file_path: Path,
    duration_s: float,
    sample_rate: int = 16000,
    frequency: int = 440,
    amplitude: float = 0.3,
):
    """
    Create a synthetic WAV file with a sine wave tone.

    Args:
        file_path: Path to save the file
        duration_s: Duration in seconds
        sample_rate: Sampling rate (Hz)
        frequency: Tone frequency (Hz)
        amplitude: Amplitude (0.0-1.0)
    """
    # Generate sine wave
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, False)
    audio = amplitude * np.sin(2 * np.pi * frequency * t)

    # Convert to int16
    audio_int16 = (audio * np.iinfo(np.int16).max).astype(np.int16)

    # Write WAV file
    with wave.open(str(file_path), "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes (int16)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


class TestFileReaderIntegration:
    """Integration tests with real audio files and ffmpeg"""

    @pytest.mark.asyncio
    async def test_filereader_streaming_adds_eos_padding_with_real_audio(
        self, tmp_path
    ):
        """
        Test FileReader streaming mode adds EOS padding with real audio file.

        Creates a real WAV file (1 second), reads through FileReader
        with preprocess=False, verifies padding is added.
        """
        # 1. Create synthetic WAV file
        audio_file = tmp_path / "test_audio.wav"
        create_synthetic_wav(
            audio_file, duration_s=1.0, sample_rate=16000, frequency=440  # A4 note
        )

        # 2. Configure with eos_silence_s=2.0
        config = Config(
            source="en", targets=["es"], mode=WsMode(), eos_silence_s=2.0  # 2 seconds
        )

        # 3. Create FileReader with preprocess=False (streaming)
        reader = FileReader(path=audio_file, preprocess=False)
        reader.cfg = config
        reader.do_preprocess()
        await reader.boot()

        # 4. Read all data and count bytes
        total_bytes = 0
        padding_bytes = 0
        padding_started = False
        max_iterations = 1000  # Safety limit to prevent infinite loop on CI

        for _ in range(max_iterations):
            chunk = await reader.read(1024)
            if chunk is None:
                break

            if reader._padding_started and not padding_started:
                padding_started = True
                padding_bytes = 0

            if padding_started:
                padding_bytes += len(chunk)

            total_bytes += len(chunk)
        else:
            pytest.fail(f"Read loop exceeded {max_iterations} iterations without EOF")

        # 5. Verify
        assert reader.eof.is_set()
        assert padding_started, "Padding should have started"

        # Expected padding: 2s * 16000 Hz * 1 channel * 2 bytes = 64000 bytes
        expected_padding = int(2.0 * 16000 * 1 * 2)
        assert (
            abs(padding_bytes - expected_padding) < 100
        ), f"Expected ~{expected_padding} padding bytes, got {padding_bytes}"

        # Original audio: 1s * 16000 * 1 * 2 = 32000 bytes
        # Total: 32000 + 64000 = 96000 bytes
        expected_total = 32000 + 64000
        assert (
            abs(total_bytes - expected_total) < 200
        ), f"Expected ~{expected_total} total bytes, got {total_bytes}"

    @pytest.mark.asyncio
    async def test_filereader_preprocessed_mode_has_padding_in_buffer(self, tmp_path):
        """
        Test that preprocessed mode already has padding in buffer.

        Reader padding should NOT activate in preprocessed mode because
        padding is already added during preprocessing.
        """
        audio_file = tmp_path / "test_audio.wav"
        create_synthetic_wav(audio_file, duration_s=1.0, sample_rate=16000)

        config = Config(
            source="en", targets=["es"], mode=WsMode(), eos_silence_s=2.0
        )

        # preprocess=True (default)
        reader = FileReader(path=audio_file, preprocess=True)
        reader.cfg = config
        reader.do_preprocess()
        await reader.boot()

        # Read all
        total_bytes = 0
        max_iterations = 1000  # Safety limit
        for _ in range(max_iterations):
            chunk = await reader.read(1024)
            if chunk is None:
                break
            total_bytes += len(chunk)
        else:
            pytest.fail(f"Read loop exceeded {max_iterations} iterations without EOF")

        # Verify
        assert reader.eof.is_set()
        assert (
            not reader._padding_started
        ), "Reader padding should NOT activate in preprocessed mode"

        # Padding already in buffer from preprocess: 1s audio + 2s padding = 3s total
        expected_total = int(3.0 * 16000 * 1 * 2)
        assert abs(total_bytes - expected_total) < 200

    @pytest.mark.asyncio
    async def test_filereader_streaming_logs_success_on_padding_start(self, tmp_path):
        """
        Test that success() is called when padding starts.
        """
        audio_file = tmp_path / "test_audio.wav"
        create_synthetic_wav(audio_file, duration_s=0.5, sample_rate=16000)

        config = Config(
            source="en", targets=["es"], mode=WsMode(), eos_silence_s=1.0
        )

        reader = FileReader(path=audio_file, preprocess=False)
        reader.cfg = config
        reader.do_preprocess()
        await reader.boot()

        with patch("palabra_ai.task.adapter.base.success") as mock_success:
            # Read until padding
            max_iterations = 1000  # Safety limit
            for _ in range(max_iterations):
                chunk = await reader.read(1024)
                if chunk is None:
                    break
            else:
                pytest.fail(f"Read loop exceeded {max_iterations} iterations without EOF")

            # Verify success() was called
            mock_success.assert_called_once()
            call_args = str(mock_success.call_args)
            assert "Starting EOS padding" in call_args
            assert "1.0s" in call_args or "32000 bytes" in call_args

    @pytest.mark.asyncio
    async def test_filereader_concurrent_reads_no_corruption(self, tmp_path):
        """
        Test that concurrent read() calls don't corrupt state (CRITICAL).

        This tests the fix for vulnerability #13 - concurrent reads could
        corrupt _position and _padding_remaining due to race conditions.
        The asyncio.Lock should prevent this.
        """
        audio_file = tmp_path / "test_audio.wav"
        create_synthetic_wav(audio_file, duration_s=1.0, sample_rate=16000)

        config = Config(
            source="en", targets=["es"], mode=WsMode(), eos_silence_s=2.0
        )

        reader = FileReader(path=audio_file, preprocess=False)
        reader.cfg = config
        reader.do_preprocess()
        await reader.boot()

        # Simulate concurrent reads from multiple coroutines
        async def read_worker(worker_id):
            bytes_read = 0
            max_iterations = 1000  # Safety limit
            for _ in range(max_iterations):
                chunk = await reader.read(512)
                if chunk is None:
                    break
                bytes_read += len(chunk)
            else:
                raise TimeoutError(f"Worker {worker_id} exceeded {max_iterations} iterations")
            return worker_id, bytes_read

        # Run 3 concurrent workers
        results = await asyncio.gather(
            read_worker(1),
            read_worker(2),
            read_worker(3),
            return_exceptions=True
        )

        # Verify no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception), f"Worker failed: {result}"

        # Verify total bytes read is consistent
        # Original: 1s * 16000Hz * 2 bytes = 32000
        # Padding: 2s * 16000Hz * 2 bytes = 64000
        # Total: 96000 bytes
        # With 3 concurrent readers, total should still be 96000
        # (lock ensures only one reader progresses at a time)
        total_bytes = sum(worker_bytes for _, worker_bytes in results)
        expected_total = 32000 + 64000

        # Due to concurrent access, only one worker will read all data
        # Others will get None immediately due to EOF
        assert total_bytes == expected_total, \
            f"Expected {expected_total} total bytes, got {total_bytes}"

        # Verify EOF was set
        assert reader.eof.is_set()

    @pytest.mark.asyncio
    async def test_filereader_resampler_exception_triggers_padding(self, tmp_path):
        """
        Test that resampler exceptions trigger padding (CRITICAL).

        This tests the fix for vulnerability #3 - if resampler.resample()
        throws an exception (corrupt frame), padding should still be triggered.
        """
        audio_file = tmp_path / "test_audio.wav"
        create_synthetic_wav(audio_file, duration_s=0.5, sample_rate=16000)

        config = Config(
            source="en", targets=["es"], mode=WsMode(), eos_silence_s=1.0
        )

        reader = FileReader(path=audio_file, preprocess=False)
        reader.cfg = config
        reader.do_preprocess()
        await reader.boot()

        # Mock resampler to throw exception on 2nd call (simulate corrupt frame)
        original_resampler = reader._resampler
        call_count = [0]

        class MockResampler:
            def resample(self, frame):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise ValueError("Simulated corrupt frame")
                return original_resampler.resample(frame)

        reader._resampler = MockResampler()

        # Read until EOF
        total_bytes = 0
        padding_started = False
        max_iterations = 1000  # Safety limit
        for _ in range(max_iterations):
            chunk = await reader.read(1024)
            if chunk is None:
                break
            if reader._padding_started and not padding_started:
                padding_started = True
            total_bytes += len(chunk)
        else:
            pytest.fail(f"Read loop exceeded {max_iterations} iterations without EOF")

        # CRITICAL: Verify padding was triggered despite exception
        assert padding_started, "Padding should have started even after resampler exception"
        assert reader._source_exhausted, "_source_exhausted should be True after exception"
        assert reader.eof.is_set(), "EOF should be set"

        # Verify padding was added (even though some data was lost to exception)
        # At minimum, should have padding bytes
        min_expected = int(1.0 * 16000 * 2)  # 1s padding
        assert total_bytes >= min_expected, \
            f"Expected at least {min_expected} bytes of padding, got {total_bytes}"
