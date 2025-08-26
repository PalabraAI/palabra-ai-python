from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import KW_ONLY, dataclass
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Iterator

import av
import numpy as np
from tqdm import tqdm

from palabra_ai.constant import BYTES_PER_SAMPLE, DECODE_TIMEOUT, MAX_FRAMES_PER_READ
from palabra_ai.internal.audio import write_to_disk
from palabra_ai.task.adapter.base import BufferedWriter, Reader
from palabra_ai.util.aio import warn_if_cancel
from palabra_ai.util.logger import debug, error, warning
from palabra_ai.util.logger import success


@dataclass
class FileReader(Reader):
    """Read PCM audio from file with streaming support."""

    path: Path | str
    _: KW_ONLY
    preprocess: bool = False

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
        
        if self.preprocess:
            success(f"Preprocessing {self.path}...")
            self._preprocess_audio()
            success(f"Preprocessing {self.path} done!")

    def _preprocess_audio(self):
        """Preprocess audio with loudnorm and speechnorm filters."""
        debug(f"Preprocessing audio file {self.path}...")
        
        # Open input container
        input_container = av.open(str(self.path), metadata_errors="ignore")
        audio_streams = [s for s in input_container.streams if s.type == 'audio']
        if not audio_streams:
            raise ValueError(f"No audio streams found in {self.path}")
        
        audio_stream = audio_streams[0]
        self._total_duration = float(audio_stream.duration * audio_stream.time_base) if audio_stream.duration else 0
        
        # Get target configuration from cfg if available, otherwise use default
        target_rate = self.cfg.mode.sample_rate

        debug(f"Audio: {audio_stream.codec.name}, {audio_stream.sample_rate}Hz, {audio_stream.channels}ch")
        debug(f"Duration: {self._total_duration:.1f}s")
        
        # Create in-memory output container
        output_buffer = BytesIO()
        output_container = av.open(output_buffer, mode="w", format="s16le")
        output_stream = output_container.add_stream("pcm_s16le", rate=target_rate)
        output_stream.layout = "mono"
        output_stream.time_base = Fraction(1, target_rate)
        
        # Create filter graph
        filter_graph = av.filter.Graph()
        filter_graph_buffer = filter_graph.add_abuffer(
            format=output_stream.format.name,
            sample_rate=output_stream.rate,
            layout=output_stream.layout,
            time_base=output_stream.time_base,
        )
        loudnorm_filter = filter_graph.add("loudnorm", "I=-23:LRA=5:TP=-1")
        speechnorm_filter = filter_graph.add("speechnorm", "e=50:r=0.0005:l=1")
        filter_graph_sink = filter_graph.add("abuffersink")
        
        # Connect filters
        filter_graph_buffer.link_to(loudnorm_filter)
        loudnorm_filter.link_to(speechnorm_filter)
        speechnorm_filter.link_to(filter_graph_sink)
        filter_graph.configure()
        
        # Create resampler
        resampler = av.AudioResampler(
            format=av.AudioFormat("s16"), 
            layout="mono", 
            rate=target_rate
        )
        
        # Process with progress bar
        total_frames = int(self._total_duration * audio_stream.sample_rate) if self._total_duration > 0 else None
        progress = tqdm(
            total=total_frames,
            desc=f"Preprocessing {self.path.name}",
            unit="frames",
            unit_scale=True
        )
        
        dts = 0
        processed_frames = 0
        
        try:
            # Process all frames
            for frame in input_container.decode(audio=0):
                if frame is not None:
                    # Resample frame
                    for resampled_frame in resampler.resample(frame):
                        # Push to filter graph
                        filter_graph_buffer.push(resampled_frame)
                        
                        # Pull processed frames
                        while True:
                            try:
                                filtered_frame = filter_graph_sink.pull()
                                filtered_frame.pts = dts
                                dts += filtered_frame.samples
                                
                                # Encode to output
                                for packet in output_stream.encode(filtered_frame):
                                    output_container.mux(packet)
                                
                                processed_frames += filtered_frame.samples
                                progress.update(filtered_frame.samples)
                                
                            except av.error.BlockingIOError:
                                break
            
            # Flush filters
            filter_graph_buffer.push(None)
            while True:
                try:
                    filtered_frame = filter_graph_sink.pull()
                    filtered_frame.pts = dts
                    dts += filtered_frame.samples
                    
                    for packet in output_stream.encode(filtered_frame):
                        output_container.mux(packet)
                        
                    processed_frames += filtered_frame.samples
                    progress.update(filtered_frame.samples)
                    
                except (av.error.BlockingIOError, av.error.EOFError):
                    break
            
            # Flush encoder
            for packet in output_stream.encode(None):
                output_container.mux(packet)
                
        finally:
            progress.close()

    async def boot(self):
        if self._preprocessed:
            # Already preprocessed, just set target rate from config
            self._target_rate = self.cfg.mode.sample_rate
            debug(f"Using preprocessed audio, target rate: {self._target_rate}Hz")
            return
            
        debug(f"Opening audio file {self.path} for streaming...")
        
        # Open container with timeout to avoid hanging
        self._container = await asyncio.to_thread(
            lambda: av.open(str(self.path), timeout=DECODE_TIMEOUT, metadata_errors="ignore")
        )
        
        # Find audio stream
        audio_streams = [s for s in self._container.streams if s.type == 'audio']
        if not audio_streams:
            raise ValueError(f"No audio streams found in {self.path}")
            
        audio_stream = audio_streams[0]
        self._total_duration = float(audio_stream.duration * audio_stream.time_base) if audio_stream.duration else 0
        
        debug(f"Audio: {audio_stream.codec.name}, {audio_stream.sample_rate}Hz, {audio_stream.channels}ch")
        debug(f"Duration: {self._total_duration:.1f}s")
        
        # Setup resampler for target format
        self._target_rate = self.cfg.mode.sample_rate
        
        self._resampler = av.AudioResampler(
            format=av.AudioFormat("s16"),
            layout="mono", 
            rate=self._target_rate
        )
        debug(f"Resampler: {self._target_rate}Hz mono")
        
        # Enable threading for faster decode
        audio_stream.codec_context.thread_type = av.codec.context.ThreadType.FRAME
        
        # Create iterator but don't start reading yet
        self._iterator = self._container.decode(audio=0)
        debug(f"Streaming ready for {self.path}")

    async def exit(self):
        seconds_processed = (self._position / (self._target_rate * BYTES_PER_SAMPLE))
        progress_pct = (seconds_processed / self._total_duration) * 100 if self._total_duration > 0 else 0
        debug(f"{self.name} processed {seconds_processed:.1f}s ({progress_pct:.1f}%)")
        
        if self._container:
            self._container.close()
            self._container = None

    async def read(self, size: int) -> bytes | None:
        await self.ready
        
        if self._preprocessed:
            # Read from preprocessed buffer
            if not self._buffer:
                debug(f"EOF at position {self._position}")
                +self.eof  # noqa
                return None
            
            # Extract requested amount from buffer
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
        else:
            # Fill buffer if needed (streaming mode)
            await self._ensure_buffer_has_data(size)
            
            if not self._buffer:
                debug(f"EOF at position {self._position}")
                +self.eof  # noqa
                return None
            
            # Extract requested amount from buffer
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
        
        for _ in range(min(frames_to_read, MAX_FRAMES_PER_READ)):  # Limit to avoid blocking
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