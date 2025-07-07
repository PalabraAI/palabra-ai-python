from __future__ import annotations

import asyncio
import time
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Optional

from palabra_ai.base.enum import Channel, Direction
from palabra_ai.base.task import Task
from palabra_ai.config import Config
from palabra_ai.constant import SHUTDOWN_TIMEOUT, SLEEP_INTERVAL_LONG
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
    adapter: Any = field(default=None, init=False)  # RealtimeAdapter instance
    in_foq: FanoutQueue = field(default_factory=FanoutQueue, init=False)
    out_foq: FanoutQueue = field(default_factory=FanoutQueue, init=False)


    async def boot(self):
        # Import adapters based on config mode
        from palabra_ai.config import Mode
        
        if self.cfg.mode == Mode.WEBSOCKET:
            from palabra_ai.adapter.realtime_websocket import WebSocketRealtimeAdapter
            adapter_class = WebSocketRealtimeAdapter
        elif self.cfg.mode == Mode.WEBRTC:
            from palabra_ai.adapter.realtime_webrtc import WebRTCRealtimeAdapter
            adapter_class = WebRTCRealtimeAdapter
        else:  # Mode.MIXED is default
            from palabra_ai.adapter.realtime_mixed import MixedRealtimeAdapter
            adapter_class = MixedRealtimeAdapter
        
        # Create adapter instance
        self.adapter = adapter_class(
            jwt_token=self.credentials.publisher[0],
            control_url=self.credentials.control_url,
            stream_url=self.credentials.stream_url,
            tg=self.sub_tg,
            in_foq=self.in_foq,
            out_foq=self.out_foq,
        )
        
        # Boot the adapter
        await self.adapter.boot()

    async def do(self):
        while not self.stopper:
            await asyncio.sleep(SLEEP_INTERVAL_LONG)
    
    async def send_message(self, message: dict[str, Any]) -> None:
        """Send message through the adapter"""
        await self.adapter.send_message(message)
    
    async def set_translation_settings(self, settings: dict[str, Any]) -> None:
        """Set translation settings through the adapter"""
        await self.adapter.set_translation_settings(settings)
    
    async def get_translation_settings(self, timeout: Optional[int] = None) -> dict[str, Any]:
        """Get translation settings through the adapter"""
        return await self.adapter.get_translation_settings(timeout)

    async def exit(self):
        self.in_foq.publish(None)
        self.out_foq.publish(None)
        
        await asyncio.wait_for(self.adapter.exit(), timeout=SHUTDOWN_TIMEOUT)
