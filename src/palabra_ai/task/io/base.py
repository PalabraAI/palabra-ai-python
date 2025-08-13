import abc
import time
from asyncio import get_running_loop, sleep
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from itertools import count
from typing import Iterator
from typing import TYPE_CHECKING

import numpy as np

from palabra_ai.audio import AudioFrame
from palabra_ai.constant import BOOT_TIMEOUT, BYTES_PER_SAMPLE, SLEEP_INTERVAL_LONG
from palabra_ai.enum import Channel, Direction
from palabra_ai.enum import Kind
from palabra_ai.message import (
    CurrentTaskMessage,
    Dbg,
    EndTaskMessage,
    GetTaskMessage,
    SetTaskMessage,
)
from palabra_ai.task.base import Task
from palabra_ai.util.fanout_queue import FanoutQueue
from palabra_ai.util.logger import debug
from palabra_ai.util.orjson import to_json

if TYPE_CHECKING:
    from palabra_ai.internal.rest import SessionCredentials
    from palabra_ai.message import Message
    from palabra_ai.task.adapter import Reader, Writer


@dataclass
class Io(Task):
    credentials: "SessionCredentials"
    reader: "Reader"
    writer: "Writer"
    _: KW_ONLY
    in_msg_foq: FanoutQueue["Message"] = field(default_factory=FanoutQueue, init=False)
    out_msg_foq: FanoutQueue["Message"] = field(default_factory=FanoutQueue, init=False)
    bench_audio_foq: FanoutQueue[AudioFrame] = field(default_factory=FanoutQueue, init=False)
    _buffer_callback: Callable | None = field(default=None, init=False)
    _idx: Iterator[int] = field(default_factory=count, init=False)
    _in_msg_num: Iterator[int] = field(default_factory=count, init=False)
    _out_msg_num: Iterator[int] = field(default_factory=count, init=False)
    _in_audio_num: Iterator[int] = field(default_factory=count, init=False)
    _out_audio_num: Iterator[int] = field(default_factory=count, init=False)

    @property
    @abc.abstractmethod
    def channel(self) -> Channel:
        """Return the channel type for this IO."""
        ...

    @abc.abstractmethod
    async def send_frame(self, frame) -> None:
        """Send an audio frame through the transport."""
        ...

    @abc.abstractmethod
    async def send_message(self, msg_data: bytes) -> None:
        """Send a message through the transport."""
        ...

    async def push_in_msg(self, msg: "Message") -> None:
        """Push an incoming message with debug tracking."""
        _dbg = Dbg(Kind.MESSAGE, self.channel, Direction.IN, num=next(self._in_msg_num), idx=next(self._idx))
        msg._dbg = _dbg
        debug(f"Pushing message: {msg!r}")
        self.in_msg_foq.publish(msg)

    async def in_msg_sender(self):
        """Send messages from the input queue through the transport."""
        async with self.in_msg_foq.receiver(self, self.stopper) as msgs:
            async for msg in msgs:
                if msg is None or self.stopper:
                    debug("stopping in_msg_sender due to None or stopper")
                    return
                raw = to_json(msg)
                debug(f"<- {raw[0:30]}")
                await self.send_message(raw)

    async def do(self):
        """Main processing loop - read audio chunks and push them."""
        await self.reader.ready
        while not self.stopper and not self.eof:
            chunk = await self.reader.read(self.cfg.mode.chunk_bytes)

            if chunk is None:
                debug(f"T{self.name}: Audio EOF reached")
                +self.eof  # noqa
                await self.push_in_msg(EndTaskMessage())
                break

            if not chunk:
                continue
            start_time = time.time()
            await self.push(chunk)
            stop_time = time.time()
            await self.wait_after_push(stop_time - start_time)

    async def wait_after_push(self, delta: float):
        """Hook for subclasses to add post-chunk processing."""
        await sleep(self.cfg.mode.chunk_duration_ms / 1000 - delta)

    def new_frame(self) -> "AudioFrame":
        return AudioFrame.create(*self.cfg.mode.for_audio_frame)

    async def push(self, audio_bytes: bytes) -> None:
        """Process and send audio chunks."""
        samples_per_channel = self.cfg.mode.samples_per_channel
        total_samples = len(audio_bytes) // BYTES_PER_SAMPLE
        audio_frame = self.new_frame()
        audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)

        for i in range(0, total_samples, samples_per_channel):
            if get_running_loop().is_closed():
                break
            frame_chunk = audio_bytes[
                i * BYTES_PER_SAMPLE : (i + samples_per_channel) * BYTES_PER_SAMPLE
            ]

            if len(frame_chunk) < samples_per_channel * BYTES_PER_SAMPLE:
                padded_chunk = np.zeros(samples_per_channel, dtype=np.int16)
                frame_chunk = np.frombuffer(frame_chunk, dtype=np.int16)
                padded_chunk[: len(frame_chunk)] = frame_chunk
            else:
                padded_chunk = np.frombuffer(frame_chunk, dtype=np.int16)

            np.copyto(audio_data, padded_chunk)

            if self.cfg.benchmark:
                _dbg = Dbg(Kind.AUDIO, self.channel, Direction.IN, idx=next(self._idx), num=next(self._in_audio_num))
                audio_frame._dbg = _dbg
                self.bench_audio_foq.publish(audio_frame)

            await self.send_frame(audio_frame)

    async def _exit(self):
        await self.writer.q.put(None)
        return await super()._exit()

    async def set_task(self):
        debug("Setting task configuration...")
        await sleep(SLEEP_INTERVAL_LONG)
        async with self.out_msg_foq.receiver(self, self.stopper) as msgs_out:
            await self.push_in_msg(SetTaskMessage.from_config(self.cfg))
            start_ts = time.time()
            await sleep(SLEEP_INTERVAL_LONG)
            while start_ts + BOOT_TIMEOUT > time.time():
                await self.push_in_msg(GetTaskMessage())
                msg = await anext(msgs_out)
                if isinstance(msg, CurrentTaskMessage):
                    debug(f"Received current task: {msg.data}")
                    return
                debug(f"Received unexpected message: {msg}")
                await sleep(SLEEP_INTERVAL_LONG)
        debug("Timeout waiting for task configuration")
        raise TimeoutError("Timeout waiting for task configuration")
