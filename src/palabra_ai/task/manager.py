from __future__ import annotations

import asyncio
from dataclasses import KW_ONLY, dataclass, field

from palabra_ai.adapter.dummy import DummyWriter
from palabra_ai.base.adapter import Reader, Writer
from palabra_ai.base.task import Task
from palabra_ai.config import (
    Config,
)
from palabra_ai.constant import (
    BOOT_TIMEOUT,
    SAFE_PUBLICATION_END_DELAY,
    SHUTDOWN_TIMEOUT,
    SINGLE_TARGET_SUPPORTED_COUNT,
    SLEEP_INTERVAL_DEFAULT,
)
from palabra_ai.exc import ConfigurationError
from palabra_ai.internal.rest import SessionCredentials
from palabra_ai.adapter.realtime_webrtc import AudioTrackSettings
from palabra_ai.task.logger import Logger
from palabra_ai.task.monitor import RtMonitor
from palabra_ai.task.realtime import Realtime
from palabra_ai.task.receiver import ReceiverTranslatedAudio
from palabra_ai.task.sender import SenderSourceAudio
from palabra_ai.task.stat import Stat
from palabra_ai.task.transcription import Transcription
from palabra_ai.util.logger import debug, warning


@dataclass
class Manager(Task):
    """Manages the translation process and monitors progress."""

    cfg: Config
    credentials: SessionCredentials
    _: KW_ONLY
    reader: Reader = field(init=False)
    writer: Writer = field(init=False)
    track_settings: AudioTrackSettings = field(default_factory=AudioTrackSettings)
    rt: Realtime = field(init=False)
    sender: SenderSourceAudio = field(init=False)
    receiver: ReceiverTranslatedAudio = field(init=False)
    logger: Logger | None = field(default=None, init=False)
    rtmon: RtMonitor = field(init=False)
    transcription: Transcription = field(init=False)
    stat: Stat = field(init=False)

    tasks: list[Task] = field(default_factory=list, init=False)

    _debug_mode: bool = field(default=True, init=False)
    _transcriptions_shown: set = field(default_factory=set, init=False)
    _show_banner_loop: asyncio.Task | None = field(default=None, init=False)

    def __post_init__(self):
        self.stat = Stat(self)

        if len(self.cfg.targets) != SINGLE_TARGET_SUPPORTED_COUNT:
            raise ConfigurationError(
                f"Only single target language supported, got {len(self.cfg.targets)}"
            )

        self.reader = reader = self.cfg.source.reader
        target = self.cfg.targets[0]
        self.writer = writer = target.writer

        if not isinstance(reader, Reader):
            raise ConfigurationError(
                f"cfg.source.reader should be an instance of Reader, got {type(reader)}"
            )

        if not any([isinstance(writer, Writer), callable(target.on_transcription)]):
            raise ConfigurationError(
                f"You should use at least [writer] or [on_transcription] for TargetLang: "
                f"{self.cfg.targets[0]}, got neither or mistyped them, "
                f"writer={type(writer)}, on_transcription={type(target.on_transcription)}"
            )

        if not self.writer:
            debug(f"🔧 {self.name} using DummyWriter for target {target.lang}")
            self.writer = DummyWriter()

        if hasattr(self.writer, "set_track_settings"):
            self.writer.set_track_settings(self.track_settings)
        if hasattr(self.reader, "set_track_settings"):
            self.reader.set_track_settings(self.track_settings)

        self.rt = Realtime(self.cfg, self.credentials)
        if self.cfg.log_file:
            self.logger = Logger(self.cfg, self.rt)

        self.transcription = Transcription(self.cfg, self.rt)

        self.receiver = ReceiverTranslatedAudio(
            self.cfg,
            self.writer,
            self.rt,
            target.lang,
        )

        self.sender = SenderSourceAudio(
            self.cfg,
            self.rt,
            self.reader,
            self.cfg.to_dict(),
            self.track_settings,
        )

        self.rtmon = RtMonitor(self.cfg, self.rt)

        self.tasks.extend(
            [
                t
                for t in [
                    self.reader,
                    self.sender,
                    self.rt,
                    self.receiver,
                    self.writer,
                    self.rtmon,
                    self.transcription,
                    self.logger,
                    self,
                    self.stat,
                ]
                if isinstance(t, Task)
            ]
        )

    async def start_system(self):
        if self.logger:
            self.logger(self.root_tg)
            await self.logger.ready

        self.stat(self.root_tg)
        await self.stat.ready
        self._show_banner_loop = self.stat.run_banner()

        debug(f"🔧 {self.name} run listening...")
        self.rtmon(self.sub_tg)
        self.rt(self.sub_tg)
        self.transcription(self.sub_tg)
        self.writer(self.sub_tg)
        self.receiver(self.sub_tg)
        self.sender(self.sub_tg)
        await self.rt.ready
        await self.rtmon.ready
        await self.writer.ready
        await self.receiver.ready
        await self.sender.ready
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
        warning("🚀🚀🚀 Starting translation process 🚀🚀🚀")
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
                try:
                    await asyncio.sleep(SAFE_PUBLICATION_END_DELAY)
                except asyncio.CancelledError:
                    debug(f"🔚 {self.name}.do() sleep cancelled, exiting...")
                debug(f"🔚 {self.name}.do() received EOF or stopper, exiting...")
                warning("🏁 Done! ⏻ Shutting down...")
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
            if self.logger:
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
            debug(f"🔧 {self.name}.shutdown_task() {task.name} shutdown timeout!")
            task._task.cancel()
            try:
                await task._task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            debug(f"🔧 {self.name}.shutdown_task() {task.name} shutdown cancelled!")
        except Exception as e:
            debug(f"🔧 {self.name}.shutdown_task() {task.name} shutdown error: {e}")
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
                self.shutdown_task(self.sender),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            debug(
                f"🔧 {self.name}.graceful_exit() reader and sender shutdown cancelled"
            )

        debug(
            f"🔧 {self.name}.graceful_exit() waiting {SAFE_PUBLICATION_END_DELAY=}..."
        )
        try:
            await asyncio.sleep(SAFE_PUBLICATION_END_DELAY)
        except asyncio.CancelledError:
            debug(f"🔧 {self.name}.graceful_exit() sleep cancelled")
        debug(f"🔧 {self.name}.graceful_exit() {SAFE_PUBLICATION_END_DELAY=} waited!")
        debug(f"🔧 {self.name}.graceful_exit() gathering... ")
        try:
            await asyncio.gather(
                self.shutdown_task(self.receiver),
                self.shutdown_task(self.rtmon),
                self.shutdown_task(self.transcription),
                self.shutdown_task(self.rt),
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
                    debug(
                        f"🔧 {self.name}.writer_mercy() max attempts reached, cancelling writer!"
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
