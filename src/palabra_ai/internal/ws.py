import asyncio
import json
import typing as tp

import websockets

from palabra_ai.base.message import Message
from palabra_ai.constant import WS_TIMEOUT
from palabra_ai.util.fanout_queue import FanoutQueue
from palabra_ai.util.logger import debug, error


class WebSocketClient:
    def __init__(
        self,
        tg: asyncio.TaskGroup,
        uri: str,
        token: str,
        in_q_size: int = 1024,
        out_q_size: int = 1024,
    ):
        self.tg = tg
        self._uri = f"{uri}?token={token}"
        self._websocket = None
        self._keep_running = True
        self.ws_raw_in_foq = FanoutQueue()
        self._raw_in_q = self.ws_raw_in_foq.subscribe(self, in_q_size)
        self.ws_out_foq = FanoutQueue()
        # self._out_q = self.ws_out_foq.subscribe(self, out_q_size)
        self.ws_raw_out_foq = FanoutQueue()
        self._raw_out_q = self.ws_raw_out_foq.subscribe(self, out_q_size)
        self._task = None

    def connect(self):
        self._task = self.tg.create_task(self.join(), name="Ws:join")

    async def join(self):
        while self._keep_running:
            try:
                async with websockets.connect(self._uri) as websocket:
                    self._websocket = websocket

                    receive_task = self.tg.create_task(
                        self._receive_message(), name="Ws:receive"
                    )
                    send_task = self.tg.create_task(
                        self._send_message(), name="Ws:send"
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
                debug("WebSocketClient join cancelled")
                self._keep_running = False
                raise
            except websockets.exceptions.WebSocketException as e:
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
                        debug("WebSocketClient reconnect sleep cancelled")
                        self._keep_running = False
                        raise
                else:
                    debug("WebSocket client shutting down gracefully")
                    break

    async def _send_message(self):
        while self._keep_running and self._websocket and self._websocket.open:
            try:
                try:
                    message = await asyncio.wait_for(
                        self._raw_in_q.get(), timeout=WS_TIMEOUT
                    )
                except TimeoutError:
                    continue
                await self._websocket.send(json.dumps(message))
                debug(f"Sent message: {message}")
                self._raw_in_q.task_done()
            except asyncio.CancelledError:
                debug("WebSocketClient _send_message cancelled")
                raise
            except websockets.exceptions.WebSocketException as e:
                if self._keep_running:
                    error(f"Unable to send message: {e}")
                break
            except Exception as e:
                error(f"Error in _send_message: {e}")
                break

    def _decode_raw_msg(self, raw_msg: str) -> dict[str, tp.Any]:
        try:
            data = json.loads(raw_msg)
            if (
                isinstance(data, dict)
                and "data" in data
                and isinstance(data["data"], str)
            ):
                data["data"] = json.loads(data["data"])
            return data
        except json.JSONDecodeError as e:
            debug(f"Failed to decode raw message: {e}")

    async def _receive_message(self):
        while self._keep_running and self._websocket and self._websocket.open:
            try:
                raw_msg = await self._websocket.recv()
                debug(f"Received message: {raw_msg}")
                msg = Message.decode(raw_msg)
                self.ws_out_foq.publish(msg)
                self.ws_raw_out_foq.publish(self._decode_raw_msg(raw_msg))
            except asyncio.CancelledError:
                debug("WebSocketClient _receive_message cancelled")
                raise
            except websockets.exceptions.WebSocketException as e:
                if self._keep_running:
                    error(f"Unable to receive message: {e}")
                break
            except Exception as e:
                error(f"Error in _receive_message: {e}")
                break

    async def send(self, message: dict[str, tp.Any]) -> None:
        if not self._keep_running:
            debug("WebSocketClient send called after shutdown")
            return
        try:
            self.ws_raw_in_foq.publish(message)
        except asyncio.CancelledError:
            debug("WebSocketClient send cancelled")
            raise

    async def receive(
        self, timeout: int | float | None = None
    ) -> dict[str, tp.Any] | None:
        if timeout is None:
            try:
                return await self._raw_out_q.get()
            except asyncio.CancelledError:
                debug("WebSocketClient receive cancelled")
                raise
        try:
            return await asyncio.wait_for(self._raw_out_q.get(), timeout=timeout)
        except TimeoutError:
            return None
        except asyncio.CancelledError:
            debug("WebSocketClient receive with timeout cancelled")
            raise

    def mark_received(self):
        self._raw_out_q.task_done()

    async def close(self, wait_sec: int = 3) -> None:
        if not self._keep_running:
            return

        self._keep_running = False

        try:
            await self.send({"message_type": "end_task", "data": {"force": True}})
            await asyncio.sleep(wait_sec)
        except asyncio.CancelledError:
            debug("WebSocketClient close cancelled during send/wait")

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
                debug("WebSocketClient websocket close cancelled")
                # Don't retry on cancel
            except Exception as e:
                error(f"Error closing websocket: {e}")
