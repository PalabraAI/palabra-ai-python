import asyncio
import io
import wave

import aiofile
from palabra_ai.base.audio_frame import AudioFrame

from palabra_ai.config import DEEPEST_DEBUG
from palabra_ai.util.logger import debug, error, warning


class AudioBufferWriter:
    def __init__(
        self,
        tg: asyncio.TaskGroup,
        queue: asyncio.Queue[AudioFrame] | None = None,
        buffer: io.BytesIO | None = None,
        drop_empty_frames: bool = False,
    ):
        self.buffer = buffer or io.BytesIO()
        self.queue = queue or asyncio.Queue()
        self.tg = tg
        self.drop_empty_frames = drop_empty_frames
        self._stop_event = asyncio.Event()
        self._task = None
        self._frame_sample: AudioFrame | None = None
        self._frames_processed = 0

    async def start(self):
        debug(f"AudioBufferWriter.start() called, _task={self._task}")
        if self._task is None:
            debug("Creating _write task...")
            self._task = self.tg.create_task(self._write(), name="Buffer:write")
            debug(f"AudioBufferWriter.start() created task: {self._task}")
            await asyncio.sleep(0.1)
            if self._task.done():
                error(f"Task died immediately! Exception: {self._task.exception()}")
        else:
            warning(
                f"AudioBufferWriter.start() called but task already exists: {self._task}"
            )

    async def stop(self):
        if self._task is not None:
            self._stop_event.set()
            try:
                await self._task
            except asyncio.CancelledError:
                warning("AudioBufferWriter stop cancelled")
            self._task = None

    async def _write(self):
        debug(f"AudioBufferWriter._write() STARTED, queue ID: {id(self.queue)}")

        try:
            while True:
                if DEEPEST_DEBUG:
                    debug(
                        f"_write loop iteration, stop_event={self._stop_event.is_set()}, frames_processed={self._frames_processed}"
                    )

                if self._stop_event.is_set():
                    debug("_write: stop_event is set, breaking")
                    break

                try:
                    if DEEPEST_DEBUG:
                        debug(
                            f"_write: waiting for frame from queue (size={self.queue.qsize()})"
                        )
                    frame: AudioFrame | None = await asyncio.wait_for(
                        self.queue.get(), timeout=0.1
                    )
                    if DEEPEST_DEBUG:
                        debug(f"_write: GOT FRAME! {frame}")
                    self._frames_processed += 1
                except TimeoutError:
                    if DEEPEST_DEBUG:
                        debug("_write: timeout waiting for frame")
                    continue
                except asyncio.CancelledError:
                    warning("_write: queue.get cancelled")
                    raise

                if frame is None:
                    debug("_write: got None frame, breaking")
                    self.queue.task_done()
                    break

                frame_bytes = frame.data.tobytes()

                if self.drop_empty_frames:
                    if all(byte == 0 for byte in frame_bytes):
                        if DEEPEST_DEBUG:
                            debug("_write: dropping empty frame")
                        self.queue.task_done()
                        continue

                self.buffer.write(frame_bytes)
                self.queue.task_done()
                if DEEPEST_DEBUG:
                    debug(
                        f"_write: wrote {len(frame_bytes)} bytes to buffer, total processed: {self._frames_processed}"
                    )

                if self._frame_sample is None:
                    self._frame_sample = frame
                    if DEEPEST_DEBUG:
                        debug(
                            f"_write: saved first frame sample: channels={frame.num_channels}, rate={frame.sample_rate}"
                        )

                await asyncio.sleep(0)

        except asyncio.CancelledError:
            warning(f"_write: cancelled after {self._frames_processed} frames")
            raise
        finally:
            debug(f"_write: EXITED loop, processed {self._frames_processed} frames")

    def to_wav_bytes(self) -> bytes:
        if self._frame_sample is None:
            error("No frame sample available for WAV conversion")
            return b""

        with io.BytesIO() as wav_file:
            with wave.open(wav_file, "wb") as wav:
                wav.setnchannels(self._frame_sample.num_channels)
                wav.setframerate(self._frame_sample.sample_rate)
                wav.setsampwidth(2)
                wav.writeframes(self.buffer.getvalue())
            return wav_file.getvalue()

    async def write_to_disk(self, file_path: str) -> int:
        try:
            async with aiofile.async_open(file_path, "wb") as f:
                return await f.write(await asyncio.to_thread(self.to_wav_bytes))
        except asyncio.CancelledError:
            warning("AudioBufferWriter write_to_disk cancelled")
            raise
