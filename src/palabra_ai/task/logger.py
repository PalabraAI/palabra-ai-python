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
from palabra_ai.model import LogData
from palabra_ai.task.base import Task
from palabra_ai.task.io.base import Io

# from palabra_ai.task.realtime import Realtime
from palabra_ai.util.fanout_queue import Subscription
from palabra_ai.util.logger import debug, error
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
    _io_audio_in_sub: Subscription | None = field(default=None, init=False)
    _io_audio_out_sub: Subscription | None = field(default=None, init=False)
    _io_out_sub: Subscription | None = field(default=None, init=False)
    _in_task: asyncio.Task | None = field(default=None, init=False)
    _out_task: asyncio.Task | None = field(default=None, init=False)
    _audio_inout_task: asyncio.Task | None = field(default=None, init=False)

    def __post_init__(self):
        self._io_in_sub = self.io.in_msg_foq.subscribe(self, maxsize=0)
        self._io_out_sub = self.io.out_msg_foq.subscribe(self, maxsize=0)
        if self.cfg.benchmark:
            self._io_audio_in_sub = self.io.bench_audio_foq.subscribe(self, maxsize=0)

    async def boot(self):
        self._in_task = self.sub_tg.create_task(
            self._consume(self._io_in_sub.q), name="Logger:io_in"
        )
        self._out_task = self.sub_tg.create_task(
            self._consume(self._io_out_sub.q), name="Logger:io_out"
        )
        if self.cfg.benchmark:
            self._audio_inout_task = self.sub_tg.create_task(
                self._consume(self._io_audio_in_sub.q), name="Logger:io_audio_inout"
            )
        debug(f"Logger started, writing to {self.cfg.log_file}")

    async def do(self):
        # Wait for stopper
        while not self.stopper:
            await asyncio.sleep(SLEEP_INTERVAL_DEFAULT)
        debug(f"{self.name} task stopped, exiting...")

    async def cancel_subtasks(self):
        debug("Cancelling Logger subtasks...")
        +self.stopper  # noqa
        tasks_to_wait = []
        if self._in_task and self._out_task:
            tasks_to_wait.extend([self._in_task, self._out_task])
        if self.cfg.benchmark and self._audio_inout_task:
            tasks_to_wait.append(self._audio_inout_task)
        for t in tasks_to_wait:
            t.cancel()
        debug(f"Waiting for {len(tasks_to_wait)} tasks to complete...")
        try:
            await asyncio.gather(
                *(asyncio.wait_for(t, timeout=SHUTDOWN_TIMEOUT) for t in tasks_to_wait),
                return_exceptions=True,  # This will return CancelledError instead of raising it
            )
            debug("All Logger subtasks cancelled successfully")
        except Exception:
            debug("Some Logger subtasks were cancelled or failed")

    async def _exit(self):
        debug(f"{self.name}._exit()")
        return await self.exit()

    async def exit(self) -> LogData:
        debug("Finalizing Logger...")

        # First create LogData BEFORE cancelling tasks
        try:
            self.cfg.internal_logs.seek(0)
            logs = self.cfg.internal_logs.readlines()
            debug(f"Collected {len(logs)} internal log lines")

            try:
                sysinfo = get_system_info()
            except BaseException as e:
                sysinfo = {"error": str(e)}

            log_data = LogData(
                version=getattr(palabra_ai, "__version__", "n/a"),
                sysinfo=sysinfo,
                messages=self._messages.copy(),  # Copy to avoid losing data
                start_ts=self._start_ts,
                cfg=self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else {},
                log_file=str(self.cfg.log_file),
                trace_file=str(self.cfg.trace_file),
                debug=self.cfg.debug,
                logs=logs,
            )

            # CRITICAL: Save result immediately
            self.result = log_data
            debug(
                f"Logger: Saved LogData with {len(self._messages)} messages to self.result"
            )

            # Save to file if needed
            if self.cfg.trace_file:
                try:
                    with open(self.cfg.trace_file, "wb") as f:
                        f.write(to_json(log_data))
                    debug(f"Saved trace to {self.cfg.trace_file}")
                except Exception as e:
                    error(f"Failed to save trace file: {e}")

        except Exception as e:
            error(f"Failed to create LogData: {e}")
            # Create minimal LogData with what we have
            log_data = LogData(
                version="error",
                sysinfo={"error": str(e)},
                messages=self._messages.copy() if self._messages else [],
                start_ts=self._start_ts,
                cfg={},
                log_file="",
                trace_file="",
                debug=False,
                logs=[],
            )
            self.result = log_data

        # Now cancel tasks
        try:
            cancel_task = asyncio.create_task(self.cancel_subtasks())
            await asyncio.wait_for(cancel_task, timeout=2.0)
        except TimeoutError:
            debug("Logger subtasks cancellation timeout")
        except Exception as e:
            debug(f"Error cancelling logger subtasks: {e}")

        # Unsubscribe from queues
        try:
            self.io.in_msg_foq.unsubscribe(self)
            self.io.out_msg_foq.unsubscribe(self)
            if self.cfg.benchmark:
                self.io.bench_audio_foq.unsubscribe(self)
            debug("Unsubscribed from IO queues")
        except Exception as e:
            debug(f"Error unsubscribing: {e}")

        debug(
            f"Logger.exit() completed, returning LogData with {len(log_data.messages)} messages"
        )
        return log_data

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

                # Convert enums to strings for benchmark compatibility
                if "kind" in dbg_msg and dbg_msg["kind"] is not None:
                    dbg_msg["kind"] = dbg_msg["kind"].value
                if "ch" in dbg_msg and dbg_msg["ch"] is not None:
                    dbg_msg["ch"] = dbg_msg["ch"].value
                if "dir" in dbg_msg and dbg_msg["dir"] is not None:
                    dbg_msg["dir"] = dbg_msg["dir"].value

                if hasattr(msg, "model_dump"):
                    dbg_msg["msg"] = msg.model_dump()
                elif hasattr(msg, "to_bench"):
                    dbg_msg["msg"] = msg.to_bench()
                else:
                    raise TypeError(
                        f"Message {msg} does not have model_dump() or to_bench() method"
                    )
                self._messages.append(dbg_msg)
                debug(f"Consumed message from {q}: {dbg_msg}")
                q.task_done()
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                debug(f"Consumer for {q} cancelled")
                break
