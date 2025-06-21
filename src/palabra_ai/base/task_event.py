import asyncio

import loguru


class TaskEvent(asyncio.Event):
    _owner: str = ""

    def __init__(self, *args, **kwargs):
        self._log = loguru.logger
        super().__init__(*args, **kwargs)

    def set_owner(self, owner: str):
        self._owner = owner

    def log(self):
        status = "[+] " if self.is_set() else "[-] "
        self._log.debug(f"{status}{self._owner}")

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
