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
    IoEvent,
    SetTaskMessage,
)
from palabra_ai.model import IoData
from palabra_ai.task.base import Task
from palabra_ai.util.fanout_queue import FanoutQueue
from palabra_ai.util.logger import debug
from palabra_ai.util.orjson import to_json
from palabra_ai.util.timing import get_perf_ts, get_utc_ts

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
    _frames_sent: int = field(default=0, init=False)
    _total_duration_sent: float = field(default=0.0, init=False)
    global_start_perf_ts: float | None = field(default=None, init=False)
    global_start_utc_ts: float | None = field(default=None, init=False)
    eos_received: bool = field(default=False, init=False)
    io_events: list[IoEvent] = field(default_factory=list, init=False)

    @property
    def io_data(self) -> IoData:
        return IoData(
            start_perf_ts=self.global_start_perf_ts,
            start_utc_ts=self.global_start_utc_ts,
            in_sr=self.cfg.mode.input_sample_rate,
            out_sr=self.cfg.mode.output_sample_rate,
            mode=self.cfg.mode.name,
            channels=self.cfg.mode.num_channels,
            events=self.io_events,
            count_events=len(self.io_events),
        )

    @property
    @abc.abstractmethod
    def channel(self) -> Channel:
        """Return the channel type for this IO."""
        ...

    @abc.abstractmethod
    async def send_frame(self, frame: AudioFrame, raw: bytes | None = None) -> None:
        """Send an audio frame through the transport."""
        ...

    def init_global_start_ts(self):
        if self.global_start_perf_ts is None:
            self.global_start_perf_ts = get_perf_ts()
            self.global_start_utc_ts = get_utc_ts()
            if self.writer:
                self.writer.start_perf_ts = self.global_start_perf_ts

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
        msg._dbg = Dbg(
            Kind.MESSAGE,
            self.channel,
            Direction.IN,
            num=next(self._in_msg_num),
            idx=next(self._idx),
        )
        msg._dbg.calc_dawn_ts(self.global_start_perf_ts)
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
                debug(f"<- {raw[0:30]} / {msg.dbg_delta=}")
                await self.send_message(raw)
                self.io_events.append(IoEvent(msg._dbg, raw))

    async def do(self):
        """Main processing loop - read audio chunks and push them."""
        await self.reader.ready

        while not self.stopper and not self.eof:
            chunk = await self._read_next_chunk()
            if chunk is None:
                await self._handle_eof()
                break

            if not chunk:
                continue

            if self._is_behind_schedule():
                await self._send_burst_chunks(chunk)
            else:
                await self._send_single_chunk(chunk)

    async def _read_next_chunk(self) -> bytes | None:
        """Read the next audio chunk from reader."""
        return await self.reader.read(self.cfg.mode.input_chunk_bytes)

    async def _handle_eof(self):
        """Handle end of file condition."""
        debug(f"T{self.name}: Audio EOF reached")
        +self.eof  # noqa
        await self.push_in_msg(EndTaskMessage())

    def _is_behind_schedule(self) -> bool:
        """Check if we're behind the expected schedule."""
        if not self.global_start_perf_ts:
            return False
        chunk_duration_s = self.cfg.mode.input_chunk_duration_ms / 1000
        target_time = self.global_start_perf_ts + self._total_duration_sent
        current_time = time.perf_counter()
        time_behind = current_time - target_time
        return time_behind > chunk_duration_s

    def _calculate_timing_metrics(self) -> tuple[float, float, float]:
        """Calculate current timing metrics.

        Returns:
            tuple of (target_time, current_time, time_behind)
        """
        target_time = self.global_start_perf_ts + self._total_duration_sent
        current_time = time.perf_counter()
        time_behind = current_time - target_time
        return target_time, current_time, time_behind

    async def _send_burst_chunks(self, initial_chunk: bytes):
        """Send multiple chunks in burst mode when behind schedule."""
        MAX_BURST = 20
        chunk_duration_s = self.cfg.mode.input_chunk_duration_ms / 1000

        target_time, current_time, time_behind = self._calculate_timing_metrics()
        chunks_behind = int(time_behind / chunk_duration_s)
        burst_count = min(chunks_behind, MAX_BURST)

        debug(
            f"BURST: Behind by {time_behind:.3f}s ({chunks_behind} chunks), sending {burst_count} chunks"
        )

        chunk = initial_chunk
        for i in range(burst_count):
            await self._send_chunk_immediately(chunk)

            # Read next chunk for burst if not last iteration
            if i < burst_count - 1:
                next_chunk = await self._read_next_chunk()
                if next_chunk is None:
                    break
                chunk = next_chunk

    async def _send_single_chunk(self, chunk: bytes):
        """Send a single chunk with precise timing."""
        # First chunk - send immediately without timing
        if self.global_start_perf_ts is None:
            await self._send_chunk_immediately(chunk)
            return

        chunk_duration_s = self.cfg.mode.input_chunk_duration_ms / 1000
        target_time, current_time, _ = self._calculate_timing_metrics()

        # Calculate precise wait time
        wait_time = max(0.0, min(chunk_duration_s, target_time - current_time))

        if wait_time > 0:
            await aio.sleep(wait_time)

        await self._send_chunk_immediately(chunk)

    async def _send_chunk_immediately(self, chunk: bytes):
        """Send a chunk and update timing metrics."""
        chunk_duration_s = self.cfg.mode.input_chunk_duration_ms / 1000
        await self.push(chunk)
        self._frames_sent += 1
        self._total_duration_sent += chunk_duration_s

    def new_input_frame(self) -> "AudioFrame":
        return AudioFrame.create(*self.cfg.mode.for_input_audio_frame)

    async def push(self, audio_bytes: bytes) -> None:
        """Process and send audio chunks."""
        samples_per_channel = self.cfg.mode.input_samples_per_channel
        total_samples = len(audio_bytes) // BYTES_PER_SAMPLE
        audio_frame = self.new_input_frame()
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

            raw = None

            if self.cfg.benchmark:
                rms_db = await aio.to_thread(self.calc_rms_db, audio_frame)
                audio_frame._dbg = Dbg(
                    Kind.AUDIO,
                    self.channel,
                    Direction.IN,
                    idx=next(self._idx),
                    num=next(self._in_audio_num),
                    dur_s=audio_frame.duration,
                    rms_db=rms_db,
                )
                audio_frame._dbg.calc_dawn_ts(self.global_start_perf_ts)
                self.bench_audio_foq.publish(audio_frame)
                raw = audio_frame.to_ws()
                self.io_events.append(IoEvent(audio_frame._dbg, raw))

            await self.send_frame(audio_frame, raw)

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
