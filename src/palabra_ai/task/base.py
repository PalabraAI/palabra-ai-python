import abc
import asyncio
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING

from palabra_ai.constant import SHUTDOWN_TIMEOUT
from palabra_ai.util.emoji import Emoji
from palabra_ai.util.logger import debug, error, warning

if TYPE_CHECKING:
    from palabra_ai.config import Config


class TaskEvent(asyncio.Event):
    _owner: str = ""

    def __init__(self, *args, **kwargs):
        # self._log = logger
        super().__init__(*args, **kwargs)

    def set_owner(self, owner: str):
        self._owner = owner

    def log(self):
        status = "[+] " if self.is_set() else "[-] "
        debug(f"{status}{self._owner}")

    def __pos__(self):
        self.set()
        self.log()
        return self

    def __neg__(self):
        self.clear()
        self.log()
        return self

    def __bool__(self):
        return self.is_set()

    def __await__(self):
        if self.is_set():
            return self._immediate_return().__await__()
        return self.wait().__await__()

    async def _immediate_return(self):
        return

    def __repr__(self):
        return f"TaskEvent({self.is_set()})"


@dataclass
class Task(abc.ABC):
    _: KW_ONLY
    cfg: "Config"
    root_tg: asyncio.TaskGroup = field(default=None, init=False, repr=False)
    sub_tg: asyncio.TaskGroup = field(
        default_factory=asyncio.TaskGroup, init=False, repr=False
    )
    _task: asyncio.Task = field(default=None, init=False, repr=False)
    _sub_tasks: list[asyncio.Task] = field(default_factory=list, init=False, repr=False)
    _name: str | None = field(default=None, init=False, repr=False)
    ready: TaskEvent = field(default_factory=TaskEvent, init=False)
    eof: TaskEvent = field(default_factory=TaskEvent, init=False)
    stopper: TaskEvent = field(default_factory=TaskEvent)
    _state: list[str] = field(default_factory=list, init=False, repr=False)
    result: any = field(default=None, init=False, repr=False)

    def __call__(self, tg: asyncio.TaskGroup) -> "Task":
        self.root_tg = tg
        self.ready.set_owner(f"{self.name}.ready")
        self.eof.set_owner(f"{self.name}.eof")
        self.stopper.set_owner(f"{self.name}.stopper")
        self._task = tg.create_task(self.run(), name=self.name)
        return self

    async def run(self):
        self._state.append("🚀")
        try:
            async with self.sub_tg:
                try:
                    debug(f"{self.name}.run() starting...")
                    self._state.append("🌀")
                    await self._boot()
                    self._state.append("🟢")
                    +self.ready  # noqa
                    debug(f"{self.name}.run() ready, doing...")
                    self._state.append("💫")
                    await self.do()
                    self._state.append("🎉")
                    debug(f"{self.name}.run() done, exiting...")
                    +self.stopper  # noqa
                except asyncio.CancelledError:
                    self._state.append("🚫")
                    debug(f"{self.name}.run() cancelled, exiting...")
                    raise
                except Exception as e:
                    import traceback

                    self._state.append("💥")
                    error(f"{self.name}.run() failed with error: {e}, exiting...")
                    error(f"{self.name} full traceback:\n{traceback.format_exc()}")
                    self.sub_tg._abort()
                    self.root_tg._abort()
                    raise
        finally:
            +self.stopper  # noqa
            self._state.append("👋")
            debug(f"{self.name}.run() trying to exit...")
            result = await self._exit()
            self._state.append("🟠")
            debug(f"{self.name}.run() exited successfully!")
        return result

    async def _boot(self):
        return await self.boot()

    @abc.abstractmethod
    async def boot(self):
        raise NotImplementedError()

    @abc.abstractmethod
    async def do(self):
        raise NotImplementedError()

    @abc.abstractmethod
    async def exit(self):
        raise NotImplementedError()

    async def _exit(self):
        try:
            debug(f"{self.name}._exit()/proto exit() called, waiting for exit...")
            return await asyncio.wait_for(self.exit(), timeout=SHUTDOWN_TIMEOUT)
        except TimeoutError:
            error(f"{self.name}.exit()/proto timed out after {SHUTDOWN_TIMEOUT}s")
            # Cancel all subtasks
            await self.cancel_all_subtasks()
            warning(f"{self.name}.exit()/proto all subtasks cancelled")

    async def cancel_all_subtasks(self):
        """Cancel all tasks in sub_tg"""
        # Get all tasks from the sub_tg
        # TODO: more reliable way to get tasks
        all_tasks = [
            t
            for t in asyncio.all_tasks()
            if t.get_name() and t.get_name().startswith(self.name)
        ]
        for task in all_tasks:
            if task != self._task and not task.done():
                debug(f"Cancelling subtask: {task.get_name()}")
                task.cancel()

        # Wait for cancellation with timeout
        if all_tasks:
            done, pending = await asyncio.wait(all_tasks, timeout=1.0)
            for task in pending:
                debug(f"Force cancelling hung task: {task.get_name()}")
                task.cancel()

    @property
    def name(self) -> str:
        return f"[T]{self._name or self.__class__.__name__}"

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def task(self) -> asyncio.Task:
        if not self._task:
            raise RuntimeError(f"{self.name} task not set. Call the process first")
        return self._task

    def __str__(self):
        from palabra_ai.config import DEEP_DEBUG

        ready = Emoji.bool(self.ready)
        stopper = Emoji.bool(self.stopper)
        eof = Emoji.bool(self.eof)
        states = "".join(self._state) if self._state else "⭕"
        if DEEP_DEBUG:
            return f"{self.name:>28}(ready={ready}, stopper={stopper}, eof={eof}, states={states})"
        else:
            return f"{self.name:>28}🎬{ready} 🪦{stopper} 🏁{eof} {states}"
