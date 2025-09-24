from __future__ import annotations

import abc
import asyncio
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING

from palabra_ai.audio import AudioBuffer, AudioFrame
from palabra_ai.constant import SLEEP_INTERVAL_DEFAULT, SLEEP_INTERVAL_LONG
from palabra_ai.task.base import Task, TaskEvent
from palabra_ai.util.logger import debug, trace

if TYPE_CHECKING:
    from palabra_ai.audio import AudioFrame
    from palabra_ai.config import Config


@dataclass
class Reader(Task):
    """Abstract PCM audio reader process."""

    _: KW_ONLY
    cfg: Config = field(default=None, init=False, repr=False)
    # sender: Optional["palabra_ai.task.sender.SenderSourceAudio"] = None  # noqa
    q: asyncio.Queue[AudioFrame] = field(default_factory=asyncio.Queue)
    # chunk_size: int = CHUNK_SIZE
    eof: TaskEvent = field(default_factory=TaskEvent, init=False)
    duration: float | None = field(default=None, init=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eof.set_owner(f"{self.__class__.__name__}.eof")

    async def do(self):
        while not self.stopper and not self.eof:
            await asyncio.sleep(SLEEP_INTERVAL_DEFAULT)

    def do_preprocess(self):
        """Override in subclasses that need preprocessing."""
        debug(f"{self.__class__.__name__}: no preprocessing needed")

    @abc.abstractmethod
    async def read(self, size: int) -> bytes | None:
        """Read PCM16 data. Must handle CancelledError."""
        ...


@dataclass
class Writer(Task):
    _: KW_ONLY
    cfg: Config = field(default=None, init=False, repr=False)
    q: asyncio.Queue[AudioFrame | None] = field(default_factory=asyncio.Queue)
    _frames_processed: int = field(default=0, init=False)
    __start_perf_ts: float | None = field(default=None, init=False, repr=False)

    @property
    def start_perf_ts(self):
        return self.__start_perf_ts

    @start_perf_ts.setter
    def start_perf_ts(self, value):
        if self.__start_perf_ts:
            return
        self.__start_perf_ts = value

    async def do(self):
        from palabra_ai.util.logger import debug, warning

        while not self.stopper and not self.eof:
            try:
                frame: AudioFrame | None = await asyncio.wait_for(
                    self.q.get(), timeout=SLEEP_INTERVAL_LONG
                )

                if frame is None:
                    debug(f"{self.name}: received None frame, stopping")
                    +self.eof  # noqa
                    break

                trace(f"{self.name}: processing frame {self._frames_processed}")

                self._frames_processed += 1
                await self.write(frame)
                self.q.task_done()

            except TimeoutError:
                continue
            except asyncio.CancelledError:
                warning(f"{self.name}: queue processing cancelled")
                raise

    @abc.abstractmethod
    async def write(self, frame: AudioFrame):
        """Write a single frame. Override in subclasses."""
        ...

    async def _exit(self):
        self.q.put_nowait(None)
        return await super()._exit()


class UnlimitedExitMixin:
    """Mixin to allow unlimited time for exit operations (file/buffer saving)."""

    async def _exit(self):
        """Override to allow unlimited time for saving operations"""
        self.q.put_nowait(None)
        debug(f"{self.name}._exit() allowing unlimited time for save operation...")
        return await self.exit()


@dataclass
class BufferedWriter(UnlimitedExitMixin, Writer):
    """Writer that buffers audio frames before writing."""

    _: KW_ONLY
    ab: AudioBuffer | None = field(default=None, init=False)

    async def boot(self):
        # Create buffer with estimated duration from config
        self.ab = AudioBuffer(
            sample_rate=self.cfg.mode.output_sample_rate,
            num_channels=self.cfg.mode.num_channels,
            original_duration=self.cfg.estimated_duration,
            drop_empty_frames=getattr(self.cfg, "drop_empty_frames", False),
        )

    async def write(self, frame: AudioFrame):
        # Set global start time on first frame
        if self.ab.global_start_time is None and self.start_perf_ts is not None:
            self.ab.set_start_time(self.start_perf_ts)

        return await self.ab.write(frame)

    def to_wav_bytes(self) -> bytes:
        return self.ab.to_wav_bytes()
