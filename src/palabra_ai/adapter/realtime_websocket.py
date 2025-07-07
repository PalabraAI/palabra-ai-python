from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from palabra_ai.base.enum import Channel, Direction
from palabra_ai.base.message import Message
from palabra_ai.base.realtime_adapter import RealtimeAdapter
from palabra_ai.constant import WS_TIMEOUT
from palabra_ai.task.realtime import RtMsg
from palabra_ai.util.fanout_queue import FanoutQueue
from palabra_ai.util.logger import debug, error


@dataclass
class WebSocketRealtimeAdapter(RealtimeAdapter):
    """WebSocket-only adapter for realtime communication"""
    
    _websocket: Any = field(default=None, init=False)
    _keep_running: bool = field(default=True, init=False)
    _uri: str = field(default="", init=False)
    _task: asyncio.Task | None = field(default=None, init=False)
    ws_raw_in_foq: FanoutQueue = field(default_factory=FanoutQueue, init=False)
    ws_out_foq: FanoutQueue = field(default_factory=FanoutQueue, init=False)
    _raw_in_q: asyncio.Queue = field(default=None, init=False)
    _audio_callbacks: dict[str, list[callable]] = field(default_factory=dict, init=False)
    
    async def boot(self):
        """Initialize the adapter"""
        self._uri = f"{self.control_url}?token={self.jwt_token}"
        self._raw_in_q = self.ws_raw_in_foq.subscribe(self)
        await self.connect()
    
    async def connect(self) -> None:
        """Connect to WebSocket"""
        self._task = self.tg.create_task(self._join(), name="WSAdapter:join")
        
        # Set up message routing
        self._setup_message_routing()
        debug("WebSocketRealtimeAdapter connected")
    
    async def _join(self):
        """WebSocket connection loop with reconnection"""
        while self._keep_running:
            try:
                async with ws_connect(self._uri) as websocket:
                    self._websocket = websocket

                    receive_task = self.tg.create_task(
                        self._receive_message(), name="WSAdapter:receive"
                    )
                    send_task = self.tg.create_task(
                        self._send_message(), name="WSAdapter:send"
                    )

                    done, pending = await asyncio.wait(
                        [receive_task, send_task],
                        return_when=asyncio.FIRST_EXCEPTION,
                    )

                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            debug("Task cancelled")
                            self._keep_running = False
            except asyncio.CancelledError:
                debug("WebSocketAdapter join cancelled")
                self._keep_running = False
                raise
            except ConnectionClosed as e:
                if not self._keep_running:
                    debug(f"Connection closed during shutdown: {e}")
                else:
                    error(f"Connection closed with error: {e}")
            except Exception as e:
                error(f"Connection error: {e}")
            finally:
                if self._keep_running:
                    debug(f"Reconnecting to {self._uri}")
                    try:
                        await asyncio.sleep(1)
                    except asyncio.CancelledError:
                        debug("WebSocketAdapter reconnect sleep cancelled")
                        self._keep_running = False
                        raise
                else:
                    debug("WebSocket adapter shutting down gracefully")
                    break

    def _setup_message_routing(self):
        """Set up routing for incoming/outgoing messages"""
        # Route outgoing WebSocket messages
        ws_out_q = self.ws_out_foq.subscribe(self, maxsize=0)
        self.tg.create_task(
            self._route_ws_out(ws_out_q),
            name="WSAdapter:route_out"
        )
    
    async def _send_message(self):
        """Send messages through WebSocket"""
        while self._keep_running and self._websocket:
            try:
                try:
                    message = await asyncio.wait_for(
                        self._raw_in_q.get(), timeout=WS_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    continue
                await self._websocket.send(json.dumps(message))
                debug(f"Sent message: {message}")
                self._raw_in_q.task_done()
            except asyncio.CancelledError:
                debug("WebSocketAdapter _send_message cancelled")
                raise
            except ConnectionClosed as e:
                if self._keep_running:
                    error(f"Unable to send message: {e}")
                break
            except Exception as e:
                error(f"Error in _send_message: {e}")
                break

    async def _receive_message(self):
        """Receive messages from WebSocket"""
        while self._keep_running and self._websocket:
            try:
                async for raw_msg in self._websocket:
                    debug(f"Received message: {raw_msg}")
                    
                    # First, handle raw audio data messages
                    try:
                        data = json.loads(raw_msg) if isinstance(raw_msg, str) else json.loads(raw_msg.decode())
                        if data.get("message_type") == "output_audio_data":
                            await self._handle_audio_output(data)
                    except:
                        pass
                    
                    # Then decode as Message object
                    msg = Message.decode(raw_msg)
                    self.ws_out_foq.publish(msg)
                    
                    # Publish to general input queue
                    self.in_foq.publish(RtMsg(Channel.WS, Direction.IN, msg))
            except asyncio.CancelledError:
                debug("WebSocketAdapter _receive_message cancelled")
                raise
            except ConnectionClosed as e:
                if self._keep_running:
                    error(f"Unable to receive message: {e}")
                break
            except Exception as e:
                error(f"Error in _receive_message: {e}")
                break
    
    async def _route_ws_out(self, from_q: asyncio.Queue):
        """Route outgoing WebSocket messages"""
        while not self.stopper:
            try:
                msg = await asyncio.wait_for(from_q.get(), timeout=1.0)
                if msg is None:
                    debug("Received None in WS OUT, stopping route...")
                    break
                
                self.out_foq.publish(RtMsg(Channel.WS, Direction.OUT, msg))
                from_q.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                error(f"Error routing WS OUT: {e}")
    
    async def _handle_audio_output(self, msg: dict[str, Any]):
        """Handle audio output messages"""
        try:
            # If data is a string, parse it as JSON
            data = msg.get("data", {})
            if isinstance(data, str):
                data = json.loads(data)
            
            language = data.get("language", "unknown")
            audio_data = base64.b64decode(data.get("data", ""))
            
            # Call registered callbacks for this language
            if language in self._audio_callbacks:
                for callback in self._audio_callbacks[language]:
                    try:
                        await callback(audio_data)
                    except Exception as e:
                        error(f"Error in audio callback: {e}")
        except Exception as e:
            error(f"Error handling audio output: {e}")
    
    async def send_message(self, message: dict[str, Any]) -> None:
        """Send a message through WebSocket"""
        if not self._keep_running:
            debug("WebSocketAdapter send called after shutdown")
            return
        try:
            self.ws_raw_in_foq.publish(message)
        except asyncio.CancelledError:
            debug("WebSocketAdapter send cancelled")
            raise
    
    async def set_translation_settings(self, settings: dict[str, Any]) -> None:
        """Set translation settings"""
        # Update settings to use WebSocket for input/output
        if "input_stream" in settings:
            settings["input_stream"]["source"] = {
                "type": "ws",
                "format": "pcm_s16le",
                "sample_rate": 24000,
                "channels": 1
            }
        if "output_stream" in settings:
            settings["output_stream"]["target"] = {
                "type": "ws",
                "format": "pcm_s16le"
            }
        
        await self.send_message({
            "message_type": "set_task",
            "data": settings
        })
    
    async def get_translation_settings(self, timeout: Optional[int] = None) -> dict[str, Any]:
        """Get current translation settings"""
        start = time.perf_counter()
        subscriber_id = "WSAdapter.get_settings"
        
        try:
            out_q = self.ws_out_foq.subscribe(subscriber_id, 5)
            
            while True:
                await self.send_message({"message_type": "get_task", "data": {}})
                
                if timeout and time.perf_counter() - start > timeout:
                    raise TimeoutError("Timeout waiting for translation settings")
                
                try:
                    message = await asyncio.wait_for(out_q.get(), timeout=1.0)
                    if message and message.get("message_type") == "current_task":
                        out_q.task_done()
                        return message["data"]
                except asyncio.TimeoutError:
                    continue
                
                await asyncio.sleep(0.1)
        finally:
            self.ws_out_foq.unsubscribe(subscriber_id)
    
    async def publish_audio(self, audio_data: bytes) -> None:
        """Publish audio data through WebSocket"""
        message = {
            "message_type": "input_audio_data",
            "data": {
                "data": base64.b64encode(audio_data).decode("utf-8")
            }
        }
        await self.send_message(message)
    
    async def subscribe_to_audio(self, language: str, callback: callable) -> None:
        """Subscribe to translated audio for a specific language"""
        if language not in self._audio_callbacks:
            self._audio_callbacks[language] = []
        self._audio_callbacks[language].append(callback)
    
    async def do(self):
        """Main loop"""
        while not self.stopper:
            await asyncio.sleep(1.0)
    
    async def exit(self):
        """Clean up and close connections"""
        self.in_foq.publish(None)
        self.out_foq.publish(None)
        await self.close()
    
    async def close(self) -> None:
        """Close WebSocket connection"""
        if not self._keep_running:
            return

        self._keep_running = False

        try:
            await self.send_message({"message_type": "end_task", "data": {"force": True}})
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            debug("WebSocketAdapter close cancelled during send/wait")

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._websocket:
            try:
                await self._websocket.close()
            except asyncio.CancelledError:
                debug("WebSocketAdapter websocket close cancelled")
            except Exception as e:
                error(f"Error closing websocket: {e}")