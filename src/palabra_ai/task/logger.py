from __future__ import annotations

import asyncio
import time
from dataclasses import KW_ONLY, asdict, dataclass, field

import palabra_ai
from palabra_ai.config import (
    Config,
)
from palabra_ai.constant import (
    QUEUE_READ_TIMEOUT,
    SHUTDOWN_TIMEOUT,
    SLEEP_INTERVAL_DEFAULT,
)
from palabra_ai.message import Dbg
from palabra_ai.task.base import Task
from palabra_ai.task.io.base import Io

# from palabra_ai.task.realtime import Realtime
from palabra_ai.util.fanout_queue import Subscription
from palabra_ai.util.logger import debug
from palabra_ai.util.orjson import to_json
from palabra_ai.util.sysinfo import get_system_info


@dataclass
class Logger(Task):
    """Logs all WebSocket and WebRTC messages to files."""

    cfg: Config
    io: Io
    _: KW_ONLY
    _messages: list[dict] = field(default_factory=list, init=False)
    _start_ts: float = field(default_factory=time.time, init=False)
    _io_in_sub: Subscription | None = field(default=None, init=False)
    _io_out_sub: Subscription | None = field(default=None, init=False)
    _in_task: asyncio.Task | None = field(default=None, init=False)
    _out_task: asyncio.Task | None = field(default=None, init=False)

    def __post_init__(self):
        self._io_in_sub = self.io.in_msg_foq.subscribe(self, maxsize=0)
        self._io_out_sub = self.io.out_msg_foq.subscribe(self, maxsize=0)

    async def boot(self):
        self._in_task = self.sub_tg.create_task(
            self._consume(self._io_in_sub.q), name="Logger:io_in"
        )
        self._out_task = self.sub_tg.create_task(
            self._consume(self._io_out_sub.q), name="Logger:io_out"
        )
        debug(f"Logger started, writing to {self.cfg.log_file}")

    async def do(self):
        # Wait for stopper
        while not self.stopper:
            await asyncio.sleep(SLEEP_INTERVAL_DEFAULT)
        debug(f"{self.name} task stopped, exiting...")

    async def exit(self):
        debug("Finalizing Logger...")
        if self._in_task:
            self._in_task.cancel()
        if self._out_task:
            self._out_task.cancel()

        logs = []
        try:
            with open(self.cfg.log_file) as f:
                logs = f.readlines()
        except BaseException as e:
            logs = ["Can't collect logs", str(e)]

        try:
            sysinfo = get_system_info()
        except BaseException as e:
            sysinfo = {"error": str(e)}

        json_data = {
            "version": getattr(palabra_ai, "__version__", "n/a"),
            "sysinfo": sysinfo,
            "messages": self._messages,
            "start_ts": self._start_ts,
            "cfg": self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else {},
            "log_file": str(self.cfg.log_file),
            "trace_file": str(self.cfg.trace_file),
            "debug": self.cfg.debug,
            "logs": logs,
        }

        with open(self.cfg.trace_file, "wb") as f:
            f.write(to_json(json_data))

        debug(f"Saved {len(self._messages)} messages to {self.cfg.trace_file}")

        self.io.in_msg_foq.unsubscribe(self)
        self.io.out_msg_foq.unsubscribe(self)

        debug(f"{self.name} tasks cancelled, waiting for completion...")
        if self._in_task and self._out_task:
            try:
                await asyncio.gather(
                    asyncio.wait_for(self._in_task, timeout=SHUTDOWN_TIMEOUT),
                    asyncio.wait_for(self._out_task, timeout=SHUTDOWN_TIMEOUT),
                    return_exceptions=True,  # This will return CancelledError instead of raising it
                )
            except Exception:
                pass  # Tasks may be cancelled, which is expected
        debug(f"{self.name} tasks completed")

    async def _exit(self):
        return await self.exit()

    async def _consume(self, q: asyncio.Queue):
        """Process WebSocket messages."""
        while not self.stopper:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=QUEUE_READ_TIMEOUT)
                if msg is None:
                    debug(f"Received None from {q}, stopping consumer")
                    break

                dbg_msg = asdict(getattr(msg, "_dbg", Dbg.empty()))
                if not isinstance(dbg_msg, dict):
                    dbg_msg["msg"] = msg.model_dump()
                else:
                    dbg_msg["msg"] = msg
                self._messages.append(dbg_msg)
                q.task_done()
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                debug(f"Consumer for {q} cancelled")
                break
