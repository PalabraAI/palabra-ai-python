from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterator
from dataclasses import KW_ONLY, dataclass, field
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

    # Streaming state field (for streaming mode only)
    _source_exhausted: bool = field(default=False, init=False, repr=False)

    # Concurrency protection
    _read_lock: asyncio.Lock = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        self._buffer = deque()
        self._read_lock = asyncio.Lock()

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
                eos_silence_s=self.cfg.eos_silence_s,
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

        # CRITICAL: Check if padding is in progress during shutdown
        if self._padding_started and self._padding_remaining > 0:
            remaining_seconds = self._padding_remaining / (
                self.cfg.mode.input_sample_rate * BYTES_PER_SAMPLE
            )
            warning(
                f"{self.name}: exit() called while EOS padding in progress! "
                f"Remaining: {self._padding_remaining} bytes ({remaining_seconds:.2f}s). "
                f"Output audio may be truncated. This indicates premature shutdown."
            )

        if self._container:
            try:
                self._container.close()
                self._container = None
            except Exception as e:
                # Don't fail shutdown on container close errors
                error(f"{self.name}: error closing container during exit: {e}")

    async def read(self, size: int) -> bytes | None:
        await self.ready

        # Protect against concurrent reads (CRITICAL: prevents race conditions)
        async with self._read_lock:
            # If in padding mode (streaming only), deliver padding chunks
            if self._padding_started:
                return self._deliver_padding(size)

            if not self._preprocessed:
                # Fill buffer if needed (streaming mode)
                await self._ensure_buffer_has_data(size)

            # Extract from buffer (same logic for both preprocessed and streaming)
            if not self._buffer:
                # Buffer exhausted - check if we need padding (streaming mode only)
                if not self._preprocessed and self._source_exhausted:
                    return await self._start_padding(
                        f"streaming source exhausted at position {self._position}", size
                    )

                # No padding needed (preprocessed mode) or no padding configured
                debug(f"{self.name}: EOF at position {self._position}")
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
                # Buffer became empty during extraction
                if not self._preprocessed and self._source_exhausted:
                    return await self._start_padding(
                        f"streaming buffer empty at position {self._position}", size
                    )

                +self.eof  # noqa
                return None

    async def _ensure_buffer_has_data(self, needed_size: int):
        """Ensure buffer has enough data for read request"""
        current_size = sum(len(chunk) for chunk in self._buffer)

        if current_size >= needed_size:
            return  # Already enough data

        # Check if iterator already exhausted (prevents next(None) TypeError)
        if self._iterator is None:
            return  # Source already exhausted, can't read more

        # Read a few frames to fill buffer
        chunk_bytes = self.cfg.mode.input_chunk_bytes
        frames_to_read = max(1, (needed_size - current_size) // chunk_bytes + 1)

        for _ in range(
            min(frames_to_read, MAX_FRAMES_PER_READ)
        ):  # Limit to avoid blocking
            try:
                # Double-check iterator not None (race condition protection)
                if self._iterator is None:
                    break

                # Add timeout to prevent Python 3.11 hang (asyncio.to_thread bug with StopIteration)
                frame = await asyncio.wait_for(
                    asyncio.to_thread(next, self._iterator), timeout=DECODE_TIMEOUT
                )

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
                # Direct StopIteration (shouldn't happen with asyncio.to_thread but kept for safety)
                self._iterator = None
                self._source_exhausted = True
                debug(f"{self.name}: streaming source exhausted (StopIteration)")
                break
            except RuntimeError as e:
                # asyncio.to_thread wraps StopIteration in RuntimeError because
                # StopIteration cannot be raised into a Future in async contexts
                # Check __cause__ chain for StopIteration (more reliable than string matching)
                is_stop_iteration = (
                    isinstance(getattr(e, "__cause__", None), StopIteration)
                    or isinstance(getattr(e, "__context__", None), StopIteration)
                    or "StopIteration" in str(e)  # Fallback for edge cases
                )

                if is_stop_iteration:
                    self._iterator = None
                    self._source_exhausted = True
                    debug(
                        f"{self.name}: streaming source exhausted (RuntimeError wrapping StopIteration)"
                    )
                    break
                else:
                    error(f"{self.name}: unexpected RuntimeError reading frame: {e}")
                    # CRITICAL: Ensure padding is triggered even on unexpected errors
                    self._iterator = None
                    self._source_exhausted = True
                    break
            except TimeoutError:
                # Python 3.11 bug: asyncio.to_thread can hang on exhausted iterators
                # Timeout means iterator is likely exhausted or stuck
                error(
                    f"{self.name}: timeout waiting for next frame (iterator likely exhausted)"
                )
                self._iterator = None
                self._source_exhausted = True
                break
            except Exception as e:
                # CRITICAL: Handle resampler/conversion exceptions (corrupt frames, etc)
                # Must set _source_exhausted to guarantee padding is triggered
                error(f"{self.name}: error reading/processing frame: {e}")
                self._iterator = None
                self._source_exhausted = True
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
