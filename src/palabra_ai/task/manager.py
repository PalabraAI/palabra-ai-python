from __future__ import annotations

import asyncio
from dataclasses import KW_ONLY, dataclass, field

from palabra_ai.config import Config
from palabra_ai.constant import (
    BOOT_TIMEOUT,
    SHUTDOWN_TIMEOUT,
    SINGLE_TARGET_SUPPORTED_COUNT,
    SLEEP_INTERVAL_DEFAULT,
)
from palabra_ai.exc import ConfigurationError
from palabra_ai.internal.rest import SessionCredentials
from palabra_ai.task.adapter.base import Reader, Writer
from palabra_ai.task.adapter.dummy import DummyWriter
from palabra_ai.task.base import Task

# from palabra_ai.internal.webrtc import AudioTrackSettings
from palabra_ai.task.io.base import Io
from palabra_ai.task.logger import Logger
from palabra_ai.task.stat import Stat
from palabra_ai.task.transcription import Transcription
from palabra_ai.util.logger import debug, exception, success, warning


@dataclass
class Manager(Task):
    """Manages the translation process and monitors progress."""

    cfg: Config
    credentials: SessionCredentials
    _: KW_ONLY
    reader: Reader = field(init=False)
    writer: Writer = field(init=False)
    # track_settings: AudioTrackSettings = field(default_factory=AudioTrackSettings)
    # rt: Realtime = field(init=False)
    io_class: type[Io] | None = field(default=None, init=False)
    io: Io = field(init=False)
    # sender: SenderSourceAudio = field(init=False)
    # receiver: ReceiverTranslatedAudio = field(init=False)
    logger: Logger | None = field(default=None, init=False)
    # rtmon: IoMon = field(init=False)
    transcription: Transcription = field(init=False)
    stat: Stat = field(init=False)

    tasks: list[Task] = field(default_factory=list, init=False)

    _debug_mode: bool = field(default=True, init=False)
    _transcriptions_shown: set = field(default_factory=set, init=False)
    _show_banner_loop: asyncio.Task | None = field(default=None, init=False)
    _graceful_completion: bool = field(default=False, init=False)

    def __post_init__(self):
        self.stat = Stat(manager=self, cfg=self.cfg)

        if len(self.cfg.targets) != SINGLE_TARGET_SUPPORTED_COUNT:
            raise ConfigurationError(
                f"Only single target language supported, got {len(self.cfg.targets)}"
            )

        self.reader = self.cfg.source.reader
        target = self.cfg.targets[0]
        self.writer = target.writer

        if not isinstance(self.reader, Reader):
            raise ConfigurationError(
                f"cfg.source.reader should be an instance of Reader, got {type(self.reader)}"
            )

        if not any(
            [isinstance(self.writer, Writer), callable(target.on_transcription)]
        ):
            raise ConfigurationError(
                f"You should use at least [writer] or [on_transcription] for TargetLang: "
                f"{self.cfg.targets[0]}, got neither or mistyped them, "
                f"writer={type(self.writer)}, on_transcription={type(target.on_transcription)}"
            )

        if not self.writer:
            debug(f"🔧 {self.name} using DummyWriter for target {target.lang}")
            self.writer = DummyWriter()

        self.reader.cfg = self.cfg
        self.writer.cfg = self.cfg

        # Preprocess audio if needed (before boot/start_system to avoid timeouts)
        self.reader.do_preprocess()

        # Set estimated duration from reader
        if hasattr(self.reader, "duration") and self.reader.duration:
            self.cfg.estimated_duration = self.reader.duration
            debug(
                f"🔧 {self.name} set estimated_duration={self.cfg.estimated_duration:.2f}s"
            )
        else:
            # Fallback if Reader didn't set duration
            self.cfg.estimated_duration = 60.0
            debug(
                f"🔧 {self.name} using default estimated_duration={self.cfg.estimated_duration}s"
            )

        # if hasattr(self.writer, "set_track_settings"):
        #     self.writer.set_track_settings(self.track_settings)
        # if hasattr(self.reader, "set_track_settings"):
        #     self.reader.set_track_settings(self.track_settings)

        if not self.io_class:
            self.io_class = self.cfg.mode.get_io_class()
        self.io = self.io_class(
            cfg=self.cfg,
            credentials=self.credentials,
            reader=self.reader,
            writer=self.writer,
        )

        # self.rt = Realtime(self.cfg, self.io)
        self.logger = Logger(self.cfg, self.io)
        #
        self.transcription = Transcription(self.cfg, self.io)
        #
        # self.receiver = ReceiverTranslatedAudio(
        #     self.cfg,
        #     self.writer,
        #     self.rt,
        #     target.lang,
        # )

        # self.sender = SenderSourceAudio(
        #     self.cfg,
        #     self.rt,
        #     self.reader,
        #     self.cfg.to_dict(),
        #     self.track_settings,
        # )

        # self.rtmon = IoMon(self.cfg, self.rt)

        self.tasks.extend(
            [
                t
                for t in [
                    self.reader,
                    # self.sender,
                    # self.rt,
                    self.io,
                    # self.receiver,
                    self.writer,
                    # self.rtmon,
                    self.transcription,
                    self.logger,
                    self,
                    self.stat,
                ]
                if isinstance(t, Task)
            ]
        )

    async def start_system(self):
        self.logger(self.root_tg)
        await self.logger.ready

        self.stat(self.root_tg)
        await self.stat.ready
        self._show_banner_loop = self.stat.run_banner()

        debug(f"🔧 {self.name} run listening...")
        # self.rtmon(self.sub_tg)
        self.io(self.sub_tg)
        # self.rt(self.sub_tg)
        self.transcription(self.sub_tg)
        self.writer(self.sub_tg)
        # self.receiver(self.sub_tg)
        # self.sender(self.sub_tg)
        # await self.rt.ready
        await self.io.ready
        # await self.rtmon.ready
        await self.writer.ready
        # await self.receiver.ready
        # await self.sender.ready
        await self.transcription.ready
        debug(f"🔧 {self.name} listening ready!")

        debug(f"🔧 {self.name} run reader...")
        self.reader(self.sub_tg)
        await self.reader.ready
        debug(f"🔧 {self.name} reader ready!")

    async def boot(self):
        debug(f"🔧 {self.name}.boot()...")

        try:
            await asyncio.wait_for(self.start_system(), timeout=BOOT_TIMEOUT)
        except TimeoutError as e:
            raise ConfigurationError(
                f"Timeout {BOOT_TIMEOUT}s while starting tasks. "
                f"Check your configuration and network connection."
            ) from e

    async def do(self):
        success("🚀🚀🚀 Starting translation process 🚀🚀🚀")
        while not self.stopper:
            try:
                await asyncio.sleep(SLEEP_INTERVAL_DEFAULT)
            except asyncio.CancelledError:
                warning("☠️ Translation process cancelled, breaking!")
                break
            except Exception as e:
                warning(f"☠️ {self.name}.do() error: {e}, breaking!")
                break
            if any(t.eof for t in self.tasks) or any(t.stopper for t in self.tasks):
                debug(f"🔚 {self.name}.do() received EOF or stopper, exiting...")
                success("🏁 Done! ⏻ Shutting down...")
                self._graceful_completion = True
                break
        +self.stopper  # noqa
        await self.graceful_exit()

    async def exit(self):
        debug(f"🔧 {self.name}.exit() begin")
        try:
            await self.writer_mercy()
        except asyncio.CancelledError:
            debug(f"🔧 {self.name}.exit() writer shutdown cancelled")
        except Exception as e:
            debug(f"🔧 {self.name}.exit() writer shutdown error: {e}")
        finally:
            debug(f"🔧 {self.name}.exit() exiting...")
            +self.stopper  # noqa
            +self.stat.stopper  # noqa
            +self.logger.stopper  # noqa
            debug(f"🔧 {self.name}.exit() tasks: {[t.name for t in self.tasks]}")
            # DON'T use _abort() - it's internal!
            # Cancel all subtasks properly
            try:
                await self.cancel_all_subtasks()
            except asyncio.CancelledError:
                debug(f"🔧 {self.name}.exit() cancelled while cancelling subtasks")
            self._show_banner_loop.cancel()

    async def shutdown_task(self, task, timeout=SHUTDOWN_TIMEOUT):
        +task.stopper  # noqa
        debug(f"🔧 {self.name}.shutdown_task() shutting down task: {task.name}...")
        try:
            await asyncio.wait_for(task._task, timeout=timeout)
        except TimeoutError:
            # TimeoutError during graceful shutdown is expected for some tasks (e.g. WebSocket)
            # Only log as warning, not error
            warning(f"{task.name} shutdown timeout ({timeout}s) - cancelling task")
            task._task.cancel()
            try:
                await task._task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            debug(f"🔧 {self.name}.shutdown_task() {task.name} shutdown cancelled!")
        except Exception as e:
            exception(f"Cancelling {task.name} due to shutdown error: {e}")
            task._task.cancel()
            try:
                await task._task
            except asyncio.CancelledError:
                pass
        finally:
            debug(f"🔧 {self.name}.shutdown_task() {task.name} end.")

    async def graceful_exit(self):
        debug(f"🔧 {self.name}.graceful_exit() starting...")
        try:
            await asyncio.gather(
                self.shutdown_task(self.reader),
                # self.shutdown_task(self.sender),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            debug(
                f"🔧 {self.name}.graceful_exit() reader and sender shutdown cancelled"
            )

        debug(f"🔧 {self.name}.graceful_exit() gathering... ")
        try:
            await asyncio.gather(
                self.shutdown_task(self.transcription),
                self.shutdown_task(self.io),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            debug(
                f"🔧 {self.name}.graceful_exit() receiver, rtmon, transcription and rt shutdown cancelled"
            )
        finally:
            +self.writer.stopper  # noqa
            +self.stopper  # noqa
        debug(f"🔧 {self.name}.graceful_exit() gathered!")

    async def writer_mercy(self):
        +self.writer.stopper  # noqa
        debug(f"🔧 {self.name}.writer_mercy() waiting for writer to finish...")
        max_attempts = 3
        attempt = 0
        while not self.writer._task.done() and attempt < max_attempts:
            try:
                debug(
                    f"🔧 {self.name}.writer_mercy() waiting for writer task to finish (attempt {attempt + 1}/{max_attempts})..."
                )
                await asyncio.wait_for(self.writer._task, timeout=SHUTDOWN_TIMEOUT)
            except TimeoutError:
                debug(f"🔧 {self.name}.writer_mercy() writer shutdown timeout!")
                attempt += 1
                if attempt >= max_attempts:
                    exception(
                        f"Cancelling writer {self.writer.name} due to max shutdown attempts ({max_attempts}) reached"
                    )
                    self.writer._task.cancel()
                    try:
                        await self.writer._task
                    except asyncio.CancelledError:
                        pass
            except asyncio.CancelledError:
                debug(f"🔧 {self.name}.writer_mercy() cancelled")
                raise
        debug(f"🔧 {self.name}.writer_mercy() writer finished!")
