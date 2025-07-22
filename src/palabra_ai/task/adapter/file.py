from __future__ import annotations

import asyncio
from dataclasses import KW_ONLY, dataclass
from pathlib import Path

from palabra_ai.internal.audio import (
    convert_any_to_pcm16,
    read_from_disk,
    write_to_disk,
)
from palabra_ai.task.adapter.base import BufferedWriter, Reader
from palabra_ai.util.aio import warn_if_cancel

# from palabra_ai.internal.webrtc import AudioTrackSettings
from palabra_ai.util.logger import debug, error, warning


@dataclass
class FileReader(Reader):
    """Read PCM audio from file."""

    path: Path | str
    _: KW_ONLY

    _pcm_data: bytes | None = None
    _position: int = 0

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

    async def boot(self):
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
                sample_rate=self.cfg.mode.sample_rate,
            )
            debug(f"Converted to {len(self._pcm_data)} bytes")
        except Exception as e:
            error(f"Failed to convert audio: {e}")
            raise

    async def exit(self):
        debug(f"{self.name} exiting, position: {self._position}, eof: {self.eof}")
        if not self.eof:
            debug(f"{self.name} stopped without reaching EOF")
        else:
            debug(f"{self.name} reached EOF at position {self._position}")

    async def read(self, size: int) -> bytes | None:
        await self.ready

        if self._position >= len(self._pcm_data):
            debug(f"EOF reached at position {self._position}")
            +self.eof  # noqa
            return None

        chunk = self._pcm_data[self._position : self._position + size]
        self._position += len(chunk)

        return chunk if chunk else None


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
