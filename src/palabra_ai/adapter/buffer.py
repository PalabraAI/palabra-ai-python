from __future__ import annotations

import asyncio
import atexit
import io
import os
import signal
import subprocess
import threading
from dataclasses import KW_ONLY, dataclass

from palabra_ai.base.adapter import Reader, Writer
from palabra_ai.constant import SLEEP_INTERVAL_DEFAULT
from palabra_ai.internal.buffer import AudioBufferWriter
from palabra_ai.util.logger import debug, error, warning


@dataclass
class BufferReader(Reader):
    """Read PCM audio from io.BytesIO buffer."""

    buffer: io.BytesIO | RunAsPipe
    _: KW_ONLY

    def __post_init__(self):
        self._position = 0
        current_pos = self.buffer.tell()
        self.buffer.seek(0, io.SEEK_END)
        self._buffer_size = self.buffer.tell()
        self.buffer.seek(current_pos)

    async def boot(self):
        debug(f"{self.name} contains {self._buffer_size} bytes")

    async def do(self):
        while not self.stopper and not self.eof:
            await asyncio.sleep(SLEEP_INTERVAL_DEFAULT)

    async def exit(self):
        debug(f"{self.name} exiting")
        if not self.eof:
            warning(f"{self.name} stopped without reaching EOF")

    async def read(self, size: int | None = None) -> bytes | None:
        await self.ready
        size = size or self.chunk_size

        self.buffer.seek(self._position)
        chunk = self.buffer.read(size)

        if not chunk:
            +self.eof  # noqa
            debug(f"EOF reached at position {self._position}")
            return None

        self._position = self.buffer.tell()
        return chunk


@dataclass
class BufferWriter(Writer):
    """Write PCM audio to io.BytesIO buffer."""

    buffer: io.BytesIO
    _: KW_ONLY

    def __post_init__(self):
        self._buffer_writer = AudioBufferWriter(self.sub_tg, queue=self.q)
        self._started = False

    async def boot(self):
        await self._buffer_writer.start()
        self._transfer_task = self.sub_tg.create_task(
            self._transfer_audio(), name="Buffer:transfer"
        )

    async def do(self):
        while not self.stopper and not self.eof:
            await asyncio.sleep(SLEEP_INTERVAL_DEFAULT)

    async def exit(self):
        try:
            await self._transfer_task
        except asyncio.CancelledError:
            pass
        debug("Finalizing BufferWriter...")

        wav_data = self._buffer_writer.to_wav_bytes()
        if wav_data:
            self.buffer.seek(0)
            self.buffer.truncate()
            self.buffer.write(wav_data)
            self.buffer.seek(0)
            debug(f"Generated {len(wav_data)} bytes of WAV data in buffer")
        else:
            warning("No WAV data generated")

        return wav_data

    async def _transfer_audio(self):
        try:
            while True:
                try:
                    audio_frame = await self._buffer_writer.queue.get()
                    if audio_frame is None:
                        +self.eof  # noqa
                        return

                    audio_bytes = audio_frame.data.tobytes()
                    self.buffer.write(audio_bytes)

                except asyncio.CancelledError:
                    debug("BufferWriter transfer cancelled")
                    raise
                except Exception as e:
                    error(f"Transfer error: {e}")
        except asyncio.CancelledError:
            debug("BufferWriter transfer loop cancelled")
            raise


class RunAsPipe:
    """Universal pipe wrapper for subprocesses with automatic cleanup"""

    _active_processes = []
    _cleanup_registered = False

    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None
        self._buffer = bytearray()
        self._pos = 0
        self._reader_thread = None
        self._lock = threading.Lock()
        self._closed = False

        # Register cleanup only once
        if not RunAsPipe._cleanup_registered:
            RunAsPipe._cleanup_registered = True
            atexit.register(RunAsPipe._cleanup_all)
            signal.signal(signal.SIGINT, RunAsPipe._signal_handler)
            signal.signal(signal.SIGTERM, RunAsPipe._signal_handler)

        # Start process immediately
        self._start_process()

    def _start_process(self):
        """Start subprocess and reader thread"""
        self.process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
        RunAsPipe._active_processes.append(self.process)

        # Start background reader thread as daemon
        self._reader_thread = threading.Thread(target=self._read_pipe, daemon=True)
        self._reader_thread.start()

    def _read_pipe(self):
        """Background thread to read from pipe"""
        try:
            while not self._closed and self.process.poll() is None:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    break
                with self._lock:
                    self._buffer.extend(chunk)
        except Exception:
            pass

    def read(self, size=-1):
        """Read from buffer (compatible with io.BytesIO)"""
        with self._lock:
            if size == -1:
                data = bytes(self._buffer[self._pos :])
                self._pos = len(self._buffer)
            else:
                data = bytes(self._buffer[self._pos : self._pos + size])
                self._pos += len(data)
            return data

    def seek(self, pos, whence=0):
        """Seek in buffer"""
        with self._lock:
            if whence == 0:  # SEEK_SET
                self._pos = min(pos, len(self._buffer))
            elif whence == 1:  # SEEK_CUR
                self._pos = min(self._pos + pos, len(self._buffer))
            elif whence == 2:  # SEEK_END
                self._pos = len(self._buffer) + pos
            return self._pos

    def tell(self):
        """Current position"""
        return self._pos

    def __del__(self):
        """Cleanup on garbage collection"""
        self._cleanup()

    def _cleanup(self):
        """Clean up process"""
        if self._closed:
            return

        self._closed = True
        if self.process and self.process in RunAsPipe._active_processes:
            RunAsPipe._active_processes.remove(self.process)

            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

    @staticmethod
    def _cleanup_all():
        """Clean up all processes"""
        for proc in list(RunAsPipe._active_processes):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
        RunAsPipe._active_processes.clear()

    @staticmethod
    def _signal_handler(signum, frame):
        """Handle Ctrl-C"""
        RunAsPipe._cleanup_all()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
