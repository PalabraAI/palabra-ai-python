import asyncio
import time
from dataclasses import KW_ONLY, dataclass

from palabra_ai.config import (
    DEEP_DEBUG,
)
from palabra_ai.constant import (
    SLEEP_INTERVAL_DEFAULT,
    SLEEP_INTERVAL_LONG,
    SLEEP_INTERVAL_MEDIUM,
)
from palabra_ai.debug.hang_coroutines import diagnose_hanging_tasks
from palabra_ai.task.base import Task
from palabra_ai.util.logger import debug, info


@dataclass
class Stat(Task):
    manager: "palabra_ai.manager.Manager"
    _: KW_ONLY

    async def boot(self):
        pass

    async def do(self):
        show_every = 30 if DEEP_DEBUG else 150
        i = 0
        last_state = ""
        while not self.stopper:
            new_state = self.stat_palabra_tasks
            if new_state != last_state or i % show_every == 0:
                debug(self.stat)
                last_state = new_state
            i += 1
            try:
                await asyncio.sleep(SLEEP_INTERVAL_DEFAULT)
            except asyncio.CancelledError:
                debug("Stat.do() tried to cancel, but we don't approve of that!")
        try:
            await asyncio.sleep(SLEEP_INTERVAL_LONG)
        except asyncio.CancelledError:
            pass
        debug(self.stat)

    async def exit(self):
        debug(self.stat)

        moment = time.perf_counter()

        while time.perf_counter() - moment < SLEEP_INTERVAL_LONG:
            try:
                await asyncio.sleep(SLEEP_INTERVAL_MEDIUM)
            except asyncio.CancelledError:
                pass
            debug(self.stat)

    @property
    def stat_palabra_tasks(self):
        return "\n".join(
            (
                "\nPalabra tasks:",
                "\n".join([str(t) for t in self.manager.tasks]),
            )
        )

    @property
    def stat_asyncio_tasks(self):
        return "\n".join(
            (
                "\nAsyncio tasks:\n",
                " | ".join(sorted([t.get_name() for t in asyncio.all_tasks()])),
            )
        )

    @property
    def stat(self):
        deep = diagnose_hanging_tasks() if self.manager.cfg.deep_debug else ""
        return f"{deep}\n{self.stat_palabra_tasks}\n{self.stat_asyncio_tasks}"

    @property
    def _banner(self):
        return "".join(t._state[-1] if t._state else "⭕" for t in self.manager.tasks)

    def show_banner(self):
        info(self._banner)

    async def banner(self):
        last_banner = ""
        while True:
            new_banner = self._banner
            if new_banner != last_banner:
                info(new_banner)
                last_banner = new_banner
            try:
                await asyncio.sleep(SLEEP_INTERVAL_MEDIUM)
            except asyncio.CancelledError:
                debug("Stat.banner() cancelled")
                break

    def run_banner(self):
        return self.sub_tg.create_task(self.banner(), name="Stat:info_banner")
