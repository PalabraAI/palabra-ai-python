from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import KW_ONLY, dataclass, field
from functools import partial

from palabra_ai.base.message import Message
from palabra_ai.base.task import Task
from palabra_ai.config import (
    Config,
)
from palabra_ai.constant import EMPTY_MESSAGE_THRESHOLD, MONITOR_TIMEOUT
from palabra_ai.task.realtime import Realtime
from palabra_ai.util.capped_set import CappedSet
from palabra_ai.util.logger import debug, info


@dataclass
class RtMonitor(Task):
    cfg: Config
    rt: Realtime
    _: KW_ONLY
    q: asyncio.Queue = field(init=False)
    msg_history: deque[Message] = field(
        default_factory=lambda: deque(maxlen=EMPTY_MESSAGE_THRESHOLD)
    )
    msg_counter: Counter[Message.Type] = field(default_factory=Counter)
    _dedup: CappedSet[str] = field(default_factory=partial(CappedSet, 100), init=False)

    def __post_init__(self):
        self.q = self.rt.out_foq.subscribe(self, maxsize=0)

    @property
    def silence(self) -> bool:
        return len(self.msg_history) >= EMPTY_MESSAGE_THRESHOLD and all(
            msg.type_ not in Message.IN_PROCESS_TYPES for msg in self.msg_history
        )

    async def boot(self):
        await self.rt.ready

    async def do(self):
        while not self.stopper:
            try:
                rt_msg = await asyncio.wait_for(self.q.get(), MONITOR_TIMEOUT)
                if rt_msg is None:
                    debug(f"{self.name} received None, stopping...")
                    break
            except TimeoutError:
                continue

            msg = rt_msg.msg
            debug(f"📨 {self.name} received: {msg}...")
            self.msg_history.append(msg)
            self.msg_counter[msg.type_] += 1
            self.q.task_done()
            match msg.type_:
                case type_ if type_ in Message.IN_PROCESS_TYPES:
                    _dedup = msg.dedup
                    if _dedup not in self._dedup:
                        info(repr(msg))
                        self._dedup.add(_dedup)

    async def exit(self):
        self.rt.out_foq.unsubscribe(self)
