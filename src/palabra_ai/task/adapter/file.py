from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterator
from dataclasses import KW_ONLY, dataclass
from io import BytesIO
from pathlib import Path

import av
import numpy as np
from tqdm import tqdm

from palabra_ai.constant import BYTES_PER_SAMPLE, DECODE_TIMEOUT, MAX_FRAMES_PER_READ
from palabra_ai.internal.audio import (
    create_audio_resampler,
    create_normalization_filter_graph,
    create_pcm_output_container,
    flush_filters_and_encoder,
    get_audio_stream_info,
    open_audio_container,
    process_audio_frames,
    write_to_disk,
)
from palabra_ai.task.adapter.base import BufferedWriter, Reader
from palabra_ai.util.aio import warn_if_cancel
from palabra_ai.util.logger import debug, error, warning


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
    _total_duration: float = 0.0
    _target_rate: int = 0
    _preprocessed: bool = False

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        self._buffer = deque()

    def _preprocess_audio(self):
        """Preprocess audio with loudnorm and speechnorm filters."""
        debug(f"Preprocessing audio file {self.path}...")

        # Open input container and get info
        input_container, audio_stream = open_audio_container(str(self.path))
        audio_info = get_audio_stream_info(audio_stream)
        self._total_duration = audio_info["duration"]

        debug(
            f"Audio: {audio_info['codec']}, {audio_info['sample_rate']}Hz, {audio_info['channels']}ch"
        )
        debug(f"Duration: {self._total_duration:.1f}s")

        # Get target configuration
        target_rate = self.cfg.mode.sample_rate

        # Create output container and filter graph
        output_buffer = BytesIO()
        output_container, output_stream = create_pcm_output_container(
            output_buffer, target_rate, "mono"
        )
        _, filter_buffer, filter_sink = create_normalization_filter_graph(
            output_stream.format.name,
            output_stream.rate,
            output_stream.layout,
            output_stream.time_base,
        )

        # Create resampler
        resampler = create_audio_resampler(target_rate)

        # Setup progress bar
        total_frames = (
            int(self._total_duration * audio_info["sample_rate"])
            if self._total_duration > 0
            else None
        )
        progress = tqdm(
            total=total_frames,
            desc=f"Preprocessing {self.path.name}",
            unit="frames",
            unit_scale=True,
        )

        # Process frames with progress callback
        def progress_callback(samples):
            progress.update(samples)

        try:
            dts = process_audio_frames(
                input_container,
                output_stream,
                resampler,
                filter_buffer,
                filter_sink,
                progress_callback,
            )
            flush_filters_and_encoder(filter_buffer, filter_sink, output_stream, dts)
        finally:
            progress.close()
            output_container.close()
            input_container.close()

        # Store preprocessed data in buffer
        output_buffer.seek(0)
        preprocessed_data = output_buffer.read()

        # Split into chunks and store in buffer
        chunk_size = target_rate * BYTES_PER_SAMPLE * 2  # 2 seconds per chunk
        for i in range(0, len(preprocessed_data), chunk_size):
            self._buffer.append(preprocessed_data[i : i + chunk_size])

        self._preprocessed = True
        self._target_rate = target_rate
        debug(
            f"Preprocessing complete: {len(preprocessed_data)} bytes, {len(self._buffer)} chunks"
        )

    def do_preprocess(self):
        """Preprocess audio if needed, or open for streaming."""
        if self.preprocess:
            debug(f"Starting preprocessing for {self.path}...")
            self._preprocess_audio()
            debug(f"Preprocessing complete for {self.path}")
        else:
            debug(f"Opening {self.path} for streaming...")

            # Open container for streaming mode
            self._container = av.open(
                str(self.path), timeout=DECODE_TIMEOUT, metadata_errors="ignore"
            )

            # Find audio stream
            audio_streams = [s for s in self._container.streams if s.type == "audio"]
            if not audio_streams:
                raise ValueError(f"No audio streams found in {self.path}")

            audio_stream = audio_streams[0]
            self._total_duration = (
                float(audio_stream.duration * audio_stream.time_base)
                if audio_stream.duration
                else 0
            )

            debug(
                f"Audio: {audio_stream.codec.name}, {audio_stream.sample_rate}Hz, {audio_stream.channels}ch"
            )
            debug(f"Duration: {self._total_duration:.1f}s")

            # Setup target rate
            self._target_rate = self.cfg.mode.sample_rate

            # Create resampler
            self._resampler = create_audio_resampler(self._target_rate)
            debug(f"Resampler: {self._target_rate}Hz mono")

            # Enable threading for faster decode
            audio_stream.codec_context.thread_type = av.codec.context.ThreadType.FRAME

            # Create iterator but don't start reading yet
            self._iterator = self._container.decode(audio=0)
            debug(f"Streaming ready for {self.path}")

    async def boot(self):
        # Nothing to do - preprocess() already handled everything
        debug("FileReader boot: audio ready for reading")

    async def exit(self):
        seconds_processed = self._position / (self._target_rate * BYTES_PER_SAMPLE)
        progress_pct = (
            (seconds_processed / self._total_duration) * 100
            if self._total_duration > 0
            else 0
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
        chunk_bytes = self.cfg.mode.chunk_bytes
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
            error(f"Error converting to WAV: {e}", exc_info=True)
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
