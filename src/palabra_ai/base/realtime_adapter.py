from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from palabra_ai.base.task import Task
from palabra_ai.util.fanout_queue import FanoutQueue


@dataclass
class RealtimeAdapter(Task, ABC):
    """Base class for realtime adapters (WebSocket, WebRTC, or Mixed)"""
    
    jwt_token: str
    control_url: str
    stream_url: str
    tg: asyncio.TaskGroup
    
    # Output queues for messages
    in_foq: FanoutQueue
    out_foq: FanoutQueue
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the realtime service"""
        pass
    
    @abstractmethod
    async def send_message(self, message: dict[str, Any]) -> None:
        """Send a message through the adapter"""
        pass
    
    @abstractmethod
    async def set_translation_settings(self, settings: dict[str, Any]) -> None:
        """Set translation settings"""
        pass
    
    @abstractmethod
    async def get_translation_settings(self, timeout: Optional[int] = None) -> dict[str, Any]:
        """Get current translation settings"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close all connections"""
        pass
    
    @abstractmethod
    async def publish_audio(self, audio_data: bytes) -> None:
        """Publish audio data"""
        pass
    
    @abstractmethod
    async def subscribe_to_audio(self, language: str, callback: callable) -> None:
        """Subscribe to translated audio for a specific language"""
        pass