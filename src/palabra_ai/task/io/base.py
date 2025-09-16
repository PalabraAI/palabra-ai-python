import abc
import asyncio as aio
import time
from collections.abc import Callable, Iterator
from dataclasses import KW_ONLY, dataclass, field
from itertools import count
from typing import TYPE_CHECKING

import numpy as np

from palabra_ai.audio import AudioFrame
from palabra_ai.constant import BOOT_TIMEOUT, BYTES_PER_SAMPLE, SLEEP_INTERVAL_LONG
from palabra_ai.enum import Channel, Direction, Kind
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
    bench_audio_foq: FanoutQueue[AudioFrame] = field(
        default_factory=FanoutQueue, init=False
    )
    _buffer_callback: Callable | None = field(default=None, init=False)
    _idx: Iterator[int] = field(default_factory=count, init=False)
    _in_msg_num: Iterator[int] = field(default_factory=count, init=False)
    _out_msg_num: Iterator[int] = field(default_factory=count, init=False)
    _in_audio_num: Iterator[int] = field(default_factory=count, init=False)
    _out_audio_num: Iterator[int] = field(default_factory=count, init=False)
    _start_time: float | None = field(default=None, init=False)
    _sent_count: int = field(default=0, init=False)
    _pending: list = field(default_factory=list, init=False)

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

    @staticmethod
    def calc_rms_db(audio_frame: AudioFrame) -> float:
        audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)) / 32768.0
        return float(20 * np.log10(rms) if rms > 0 else -np.inf)

    async def push_in_msg(self, msg: "Message") -> None:
        """Push an incoming message with debug tracking."""
        _dbg = Dbg(
            Kind.MESSAGE,
            self.channel,
            Direction.IN,
            num=next(self._in_msg_num),
            idx=next(self._idx),
        )
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
        MAX_BURST = 20

        while not self.stopper and not self.eof:
            chunk = await self.reader.read(self.cfg.mode.chunk_bytes)

            if chunk is None:
                debug(f"T{self.name}: Audio EOF reached")
                +self.eof  # noqa
                await self.push_in_msg(EndTaskMessage())
                break

            if not chunk:
                continue

            # Initialize on first chunk
            if self._start_time is None:
                self._start_time = time.perf_counter()

            # Check if we're behind schedule
            now = time.perf_counter()
            elapsed_ms = (now - self._start_time) * 1000
            should_have_sent = int(elapsed_ms / self.cfg.mode.chunk_duration_ms)
            behind = should_have_sent - self._sent_count

            # If behind by more than 1 chunk - enable burst mode
            if behind > 1:
                self._pending.append(chunk)
                burst_count = min(len(self._pending), min(behind - 1, MAX_BURST))

                if burst_count > 0:
                    debug(
                        f"BURST: Behind by {behind}, sending {burst_count} extra chunks"
                    )
                    for _ in range(burst_count):
                        await self.push(self._pending.pop(0))
                        self._sent_count += 1

            # Send current chunk (or from queue if any)
            if self._pending:
                chunk_to_send = self._pending.pop(0)
            else:
                chunk_to_send = chunk

            start_time = time.perf_counter()
            await self.push(chunk_to_send)
            stop_time = time.perf_counter()
            self._sent_count += 1

            # Wait until next chunk time
            await self.wait_after_push(stop_time - start_time)

    async def wait_after_push(self, delta: float):
        """Hook for subclasses to add post-chunk processing."""
        await aio.sleep(self.cfg.mode.chunk_duration_ms / 1000 - delta)

    def new_frame(self) -> "AudioFrame":
        return AudioFrame.create(*self.cfg.mode.for_audio_frame)

    async def push(self, audio_bytes: bytes) -> None:
        """Process and send audio chunks."""
        samples_per_channel = self.cfg.mode.samples_per_channel
        total_samples = len(audio_bytes) // BYTES_PER_SAMPLE
        audio_frame = self.new_frame()
        audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)

        for i in range(0, total_samples, samples_per_channel):
            if aio.get_running_loop().is_closed():
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
                _dbg = Dbg(
                    Kind.AUDIO,
                    self.channel,
                    Direction.IN,
                    idx=next(self._idx),
                    num=next(self._in_audio_num),
                    chunk_duration_ms=self.cfg.mode.chunk_duration_ms,
                )
                _dbg.rms = await aio.to_thread(self.calc_rms_db, audio_frame)
                audio_frame._dbg = _dbg
                self.bench_audio_foq.publish(audio_frame)

            await self.send_frame(audio_frame)

    async def _exit(self):
        await self.writer.q.put(None)
        return await super()._exit()

    async def set_task(self):
        debug(f"set_task() STARTED for {self.name} id={id(self)}")
        debug("Setting task configuration...")
        await aio.sleep(SLEEP_INTERVAL_LONG)
        debug(f"set_task() creating receiver for {self.name} id={id(self)}")
        async with self.out_msg_foq.receiver(self, self.stopper) as msgs_out:
            debug(f"set_task() receiver created for {self.name}")
            await self.push_in_msg(SetTaskMessage.from_config(self.cfg))
            start_time = time.perf_counter()
            await aio.sleep(SLEEP_INTERVAL_LONG)
            while start_time + BOOT_TIMEOUT > time.perf_counter():
                await self.push_in_msg(GetTaskMessage())
                msg = await anext(msgs_out)
                if isinstance(msg, CurrentTaskMessage):
                    debug(f"set_task() SUCCESS: Received current task: {msg.data}")
                    return
                # Handle error messages from server
                from palabra_ai.message import ErrorMessage

                if isinstance(msg, ErrorMessage):
                    debug(f"Received error from server: {msg.data}")
                    # Don't immediately fail on NOT_FOUND - it may be temporary
                    if msg.data.get("data", {}).get("code") == "NOT_FOUND":
                        debug("Got NOT_FOUND error, will retry...")
                    else:
                        # For other errors, raise immediately
                        msg.raise_()
                debug(f"Received unexpected message: {msg}")
                await aio.sleep(SLEEP_INTERVAL_LONG)
        debug("Timeout waiting for task configuration")
        raise TimeoutError("Timeout waiting for task configuration")
