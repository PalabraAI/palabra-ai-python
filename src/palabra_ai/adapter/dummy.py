from __future__ import annotations

import asyncio
from dataclasses import KW_ONLY, dataclass, field

from palabra_ai.base.adapter import Reader, Writer
from palabra_ai.constant import SLEEP_INTERVAL_DEFAULT
from palabra_ai.util.logger import debug, error


@dataclass
class DummyReader(Reader):
    return_data: bytes = field(default=b"", repr=False)
    eof_after_reads: int | None = field(default=None, repr=False)
    _: KW_ONLY

    async def boot(self):
        pass

    async def read(self, size: int = None) -> bytes | None:
        if self.eof_after_reads is None:
            return self.return_data
        if self.eof_after_reads <= 0:
            +self.eof  # noqa
            return None
        self.eof_after_reads -= 1
        return self.return_data

    async def do(self):
        while not self.stopper and not self.eof:
            await asyncio.sleep(SLEEP_INTERVAL_DEFAULT)

    async def exit(self):
        debug(f"{self.name}.exit() called")
        if not self.eof.is_set():
            +self.eof  # noqa


@dataclass
class DummyWriter(Writer):
    _: KW_ONLY
    _q_reader: asyncio.Task = field(default=None, init=False, repr=False)

    async def q_reader(self):
        while not self.stopper and not self.eof:
            try:
                _ = await self.q.get()
                if _ is None:
                    +self.eof  # noqa
                    +self.stopper  # noqa
                    break
            except asyncio.CancelledError:
                debug(f"{self.name} q_reader cancelled")
                +self.eof  # noqa
                +self.stopper  # noqa
                break
            except Exception as e:
                error(f"{self.name} q_reader error: {e}")
                +self.eof  # noqa
                +self.stopper  # noqa
                break

    async def boot(self):
        self._q_reader = self.sub_tg.create_task(
            self.q_reader(), name=f"{self.name}.q_reader"
        )

    async def do(self):
        debug(f"{self.name}.do() begin")
        while not self.stopper and not self.eof:
            await asyncio.sleep(SLEEP_INTERVAL_DEFAULT)
        debug(f"{self.name}.do() end")

    async def exit(self):
        if self._q_reader and not self._q_reader.done():
            self._q_reader.cancel()
