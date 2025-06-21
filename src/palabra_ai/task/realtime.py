from __future__ import annotations

import asyncio
import time
from dataclasses import KW_ONLY, dataclass, field
from typing import Any

from palabra_ai.base.enum import Channel, Direction
from palabra_ai.base.task import Task
from palabra_ai.config import Config
from palabra_ai.constant import SHUTDOWN_TIMEOUT, SLEEP_INTERVAL_LONG
from palabra_ai.internal.realtime import PalabraRTClient
from palabra_ai.util.fanout_queue import FanoutQueue
from palabra_ai.util.logger import debug


@dataclass
class RtMsg:
    ch: Channel  # "ws" or "webrtc"
    dir: Direction
    msg: Any
    ts: float = field(default_factory=time.time)


@dataclass
class Realtime(Task):
    cfg: Config
    credentials: Any
    _: KW_ONLY
    c: PalabraRTClient | None = field(default=None, init=False)
    in_foq: FanoutQueue = field(default_factory=FanoutQueue, init=False)
    out_foq: FanoutQueue = field(default_factory=FanoutQueue, init=False)
    ws_in_foq: FanoutQueue = field(default_factory=FanoutQueue, init=False)
    ws_out_foq: FanoutQueue = field(default_factory=FanoutQueue, init=False)
    webrtc_out_foq: FanoutQueue = field(default_factory=FanoutQueue, init=False)

    async def _reroute(
        self,
        ch: Channel,
        dir: Direction,
        from_q: asyncio.Queue,
        to_qs: list[FanoutQueue],
    ):
        while not self.stopper:
            try:
                msg = await asyncio.wait_for(from_q.get(), timeout=SLEEP_INTERVAL_LONG)
                for to_q in to_qs:
                    to_q.publish(RtMsg(ch, dir, msg))
                from_q.task_done()
                if msg is None:
                    debug(f"Received None in {ch} {dir}, stopping reroute...")
                    break
            except TimeoutError:
                continue

    def _reroute_ws_in(self):
        ws_in_q = self.c.wsc.ws_raw_in_foq.subscribe(self, maxsize=0)
        self.sub_tg.create_task(
            self._reroute(
                Channel.WS, Direction.IN, ws_in_q, [self.in_foq, self.ws_in_foq]
            ),
            name="Rt:reroute_ws_in",
        )

    def _reroute_ws_out(self):
        ws_out_q = self.c.wsc.ws_out_foq.subscribe(self, maxsize=0)
        self.sub_tg.create_task(
            self._reroute(
                Channel.WS, Direction.OUT, ws_out_q, [self.out_foq, self.ws_out_foq]
            ),
            name="Rt:reroute_ws_out",
        )

    def _reroute_webrtc_out(self):
        webrtc_out_q = self.c.room.out_foq.subscribe(self, maxsize=0)
        self.sub_tg.create_task(
            self._reroute(
                Channel.WEBRTC,
                Direction.OUT,
                webrtc_out_q,
                [self.out_foq, self.webrtc_out_foq],
            ),
            name="Rt:reroute_webrtc_out",
        )

    async def boot(self):
        self.c = PalabraRTClient(
            self.sub_tg,
            self.credentials.publisher[0],
            self.credentials.control_url,
            self.credentials.stream_url,
        )
        self._reroute_ws_in()
        self._reroute_ws_out()
        self._reroute_webrtc_out()
        await self.c.connect()

    async def do(self):
        while not self.stopper:
            await asyncio.sleep(SLEEP_INTERVAL_LONG)

    async def exit(self):
        self.in_foq.publish(None)
        self.out_foq.publish(None)
        self.ws_in_foq.publish(None)
        self.ws_out_foq.publish(None)
        self.webrtc_out_foq.publish(None)
        await asyncio.wait_for(self.c.close(), timeout=SHUTDOWN_TIMEOUT)
