from __future__ import annotations

import asyncio
from dataclasses import KW_ONLY, dataclass
from pathlib import Path

from palabra_ai.adapter._common import warn_if_cancel
from palabra_ai.base.adapter import Reader, Writer
from palabra_ai.constant import SLEEP_INTERVAL_DEFAULT
from palabra_ai.internal.audio import (
    convert_any_to_pcm16,
    read_from_disk,
    write_to_disk,
)
from palabra_ai.internal.buffer import AudioBufferWriter
from palabra_ai.adapter.realtime_webrtc import AudioTrackSettings
from palabra_ai.util.logger import debug, error, warning


@dataclass
class FileReader(Reader):
    """Read PCM audio from file."""

    path: Path | str
    _: KW_ONLY

    _pcm_data: bytes | None = None
    _position: int = 0
    _track_settings: AudioTrackSettings | None = None

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

    def set_track_settings(self, track_settings: AudioTrackSettings) -> None:
        self._track_settings = track_settings

    async def boot(self):
        if not self._track_settings:
            self._track_settings = AudioTrackSettings()

        debug(f"Loading and converting audio file {self.path}...")
        raw_data = await warn_if_cancel(
            read_from_disk(self.path), "FileReader read_from_disk cancelled"
        )
        debug(f"Loaded {len(raw_data)} bytes from {self.path}")

        debug("Converting audio to PCM16 format...")
        try:
            self._pcm_data = await asyncio.to_thread(
                convert_any_to_pcm16,
                raw_data,
                sample_rate=self._track_settings.sample_rate,
            )
            debug(f"Converted to {len(self._pcm_data)} bytes")
        except Exception as e:
            error(f"Failed to convert audio: {e}")
            raise

    async def do(self):
        while not self.stopper and not self.eof:
            await asyncio.sleep(SLEEP_INTERVAL_DEFAULT)

    async def exit(self):
        debug(f"{self.name} exiting, position: {self._position}, eof: {self.eof}")
        if not self.eof:
            debug(f"{self.name} stopped without reaching EOF")
        else:
            debug(f"{self.name} reached EOF at position {self._position}")

    async def read(self, size: int | None = None) -> bytes | None:
        await self.ready
        size = size or self.chunk_size

        if self._position >= len(self._pcm_data):
            debug(f"EOF reached at position {self._position}")
            +self.eof  # noqa
            return None

        chunk = self._pcm_data[self._position : self._position + size]
        self._position += len(chunk)

        return chunk if chunk else None


@dataclass
class FileWriter(Writer):
    """Write PCM audio to file."""

    path: Path | str
    delete_on_error: bool = False
    _: KW_ONLY

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer_writer = None

    async def boot(self):
        self._buffer_writer = AudioBufferWriter(self.sub_tg, queue=self.q)
        await self._buffer_writer.start()

    async def do(self):
        debug(f"{self.name}.do() begin")
        while not self.stopper and not self.eof:
            await asyncio.sleep(SLEEP_INTERVAL_DEFAULT)
        debug(f"{self.name}.do() end")

    async def exit(self) -> bytes:
        debug("Finalizing FileWriter...")

        # Wait for the buffer writer task to complete (it will finish when EOF marker is processed)
        if self._buffer_writer._task and not self._buffer_writer._task.done():
            try:
                debug("Waiting for AudioBufferWriter task to complete...")
                await self._buffer_writer._task
                debug("AudioBufferWriter task completed")
            except asyncio.CancelledError:
                debug("AudioBufferWriter task was cancelled")
            except Exception as e:
                debug(f"AudioBufferWriter task failed: {e}")

        debug("All frames processed, generating WAV...")

        wav_data = b""
        try:
            wav_data = await asyncio.to_thread(self._buffer_writer.to_wav_bytes)
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
            raise
        except Exception as e:
            error(f"Error converting to WAV: {e}", exc_info=True)

        if self.delete_on_error and self.path.exists():
            try:
                self.path.unlink()
                debug(f"Removed partial file {self.path}")
            except asyncio.CancelledError:
                warning("FileWriter cancel interrupted")
                raise
            except Exception as e:
                error(f"Failed to remove partial file: {e}")
        else:
            debug(
                f"Keeping partial file {self.path} (delete_on_error={self.delete_on_error})"
            )
        return wav_data
