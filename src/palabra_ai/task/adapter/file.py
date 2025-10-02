from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterator
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import av
import numpy as np
from tqdm import tqdm

from palabra_ai.constant import (
    AUDIO_CHUNK_DURATION_SECONDS,
    BYTES_PER_SAMPLE,
    DECODE_TIMEOUT,
    MAX_FRAMES_PER_READ,
)
from palabra_ai.internal.audio import (
    simple_preprocess_audio_file,
    simple_setup_streaming_audio,
    write_to_disk,
)
from palabra_ai.task.adapter.base import BufferedWriter, Reader
from palabra_ai.util.aio import warn_if_cancel
from palabra_ai.util.logger import debug, error, warning

if TYPE_CHECKING:
    pass


@dataclass
class FileReader(Reader):
    """Read PCM audio from file with streaming support."""

    path: Path | str
    _: KW_ONLY
    preprocess: bool = True

    # Streaming fields
    _container: av.Container | None = None
    _resampler: av.AudioResampler | None = None
    _iterator: Iterator[av.AudioFrame] | None = None
    _buffer: deque = None
    _position: int = 0

    _target_rate: int = 0
    _preprocessed: bool = False

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        self._buffer = deque()

    def _preprocess_audio(self):
        """Preprocess audio with configurable pipeline."""
        # Setup progress bar
        progress = tqdm(
            desc=f"Preprocessing {self.path.name}",
            unit="frames",
            unit_scale=True,
        )

        def progress_callback(samples):
            progress.update(samples)

        try:
            # New simple pipeline
            debug(f"Using simple audio processing pipeline for {self.path}")
            normalize = getattr(self.cfg.preprocessing, "normalize_audio", False)
            preprocessed_data, metadata = simple_preprocess_audio_file(
                self.path,
                target_rate=self.cfg.mode.input_sample_rate,
                normalize=normalize,
                progress_callback=progress_callback,
                eof_silence_duration_s=self.cfg.mode.eof_silence_duration_s,
            )
            # Simple mode uses config as-is
            debug(
                f"Simple mode: using config sample rate {self.cfg.mode.input_sample_rate}Hz"
            )

            self.duration = metadata["duration"]
            self._target_rate = metadata["final_rate"]

            # Split into chunks and store in buffer
            chunk_size = (
                self._target_rate * BYTES_PER_SAMPLE * AUDIO_CHUNK_DURATION_SECONDS
            )
            for i in range(0, len(preprocessed_data), chunk_size):
                self._buffer.append(preprocessed_data[i : i + chunk_size])

            self._preprocessed = True
            debug(
                f"Preprocessing complete: {len(preprocessed_data)} bytes, {len(self._buffer)} chunks"
            )

        finally:
            progress.close()

    def do_preprocess(self):
        """Preprocess audio if needed, or open for streaming."""
        if self.preprocess:
            debug(f"Starting preprocessing for {self.path}...")
            self._preprocess_audio()
            debug(f"Preprocessing complete for {self.path}")
        else:
            debug(f"Opening {self.path} for streaming...")

            # New simple streaming setup
            debug(f"Using simple streaming setup for {self.path}")
            self._container, self._resampler, self._target_rate, metadata = (
                simple_setup_streaming_audio(
                    self.path,
                    target_rate=self.cfg.mode.input_sample_rate,
                    timeout=DECODE_TIMEOUT,
                )
            )
            # Simple mode uses config as-is
            debug(
                f"Simple streaming: using config sample rate {self.cfg.mode.input_sample_rate}Hz"
            )

            self.duration = metadata["duration"]

            # Create iterator but don't start reading yet
            self._iterator = self._container.decode(audio=0)
            debug(f"Streaming ready for {self.path}")

    async def boot(self):
        # Nothing to do - preprocess() already handled everything
        debug("FileReader boot: audio ready for reading")
        +self.ready  # noqa

    async def exit(self):
        seconds_processed = self._position / (self._target_rate * BYTES_PER_SAMPLE)
        progress_pct = (
            (seconds_processed / self.duration) * 100 if self.duration > 0 else 0
        )
        debug(f"{self.name} processed {seconds_processed:.1f}s ({progress_pct:.1f}%)")

        if self._container:
            self._container.close()
            self._container = None

    async def read(self, size: int) -> bytes | None:
        await self.ready

        if not self._preprocessed:
            # Fill buffer if needed (streaming mode)
            await self._ensure_buffer_has_data(size)

        # Extract from buffer (same logic for both preprocessed and streaming)
        if not self._buffer:
            debug(f"EOF at position {self._position}")
            +self.eof  # noqa
            return None

        result = bytearray()
        while self._buffer and len(result) < size:
            chunk = self._buffer.popleft()
            if len(result) + len(chunk) <= size:
                result.extend(chunk)
            else:
                # Split chunk
                needed = size - len(result)
                result.extend(chunk[:needed])
                self._buffer.appendleft(chunk[needed:])
                break

        if result:
            self._position += len(result)
            return bytes(result)
        else:
            +self.eof  # noqa
            return None

    async def _ensure_buffer_has_data(self, needed_size: int):
        """Ensure buffer has enough data for read request"""
        current_size = sum(len(chunk) for chunk in self._buffer)

        if current_size >= needed_size:
            return  # Already enough data

        # Read a few frames to fill buffer
        chunk_bytes = self.cfg.mode.input_chunk_bytes
        frames_to_read = max(1, (needed_size - current_size) // chunk_bytes + 1)

        for _ in range(
            min(frames_to_read, MAX_FRAMES_PER_READ)
        ):  # Limit to avoid blocking
            try:
                frame = await asyncio.to_thread(next, self._iterator)

                # Resample frame to target format
                for resampled in self._resampler.resample(frame):
                    array = resampled.to_ndarray()

                    # Convert to mono if needed
                    if array.ndim > 1:
                        array = array.mean(axis=0)

                    # Convert to int16
                    if array.dtype != np.int16:
                        array = (array * np.iinfo(np.int16).max).astype(np.int16)

                    chunk_bytes = array.tobytes()
                    self._buffer.append(chunk_bytes)

                # Check if we have enough now
                current_size = sum(len(chunk) for chunk in self._buffer)
                if current_size >= needed_size:
                    break

            except StopIteration:
                self._iterator = None
                break
            except Exception as e:
                debug(f"Error reading frame: {e}")
                break


@dataclass
class FileWriter(BufferedWriter):
    """Write PCM audio to file."""

    path: Path | str
    _: KW_ONLY
    delete_on_error: bool = False

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def exit(self):
        """Write the buffered WAV data to file"""
        debug("Finalizing FileWriter...")

        wav_data = b""
        try:
            wav_data = await asyncio.to_thread(self.ab.to_wav_bytes)
            if wav_data:
                debug(f"Generated {len(wav_data)} bytes of WAV data")
                await warn_if_cancel(
                    write_to_disk(self.path, wav_data),
                    "FileWriter write_to_disk cancelled",
                )
                debug(f"Saved {len(wav_data)} bytes to {self.path}")
            else:
                warning("No WAV data generated")
        except asyncio.CancelledError:
            warning("FileWriter finalize cancelled during WAV processing")
            self._delete_on_error()
            raise
        except Exception as e:
            error(f"Error converting to WAV: {e}")
            self._delete_on_error()
            raise

        return wav_data

    def _delete_on_error(self):
        if self.delete_on_error and self.path.exists():
            try:
                self.path.unlink()
            except Exception as clear_e:
                error(f"Failed to remove file on error: {clear_e}")
                raise
