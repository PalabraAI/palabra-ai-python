from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import websockets

from .events import Event, Raw, ServerError, parse_event
from .exc import SessionError, TaskError

if TYPE_CHECKING:
    from .client import Palabra, Session

MAX_TEXT_LEN = 256  # server limit per text message


@dataclass(frozen=True)
class TtsChunk(Event):
    """One audio_chunk message. audio is decoded; format is what you asked
    for in Palabra.tts() (pcm by default: s16le, mono)."""

    audio: bytes
    generation_id: str
    last_chunk: bool
    audio_len: float | None = None  # seconds; None on the final marker
    delta_ms: int | None = None  # generation latency metric


def _parse_tts_event(msg: dict[str, Any]) -> Event:
    if msg.get("message_type") == "audio_chunk":
        data = msg.get("data") or {}
        return TtsChunk(
            type="audio_chunk",
            audio=base64.b64decode(data.get("audio", "") or ""),
            generation_id=data.get("generation_id", ""),
            last_chunk=bool(data.get("last_chunk", False)),
            audio_len=data.get("audio_len"),
            delta_ms=data.get("chunk_generation_delta"),
        )
    return parse_event(msg)  # error / anything else


class TtsSession:
    """One Realtime TTS session over one WebSocket connection.

    Created via Palabra.tts(). On enter: create a REST session if needed,
    connect to ws_tts_url and send init. The session persists until close;
    settings can be overridden per send_text() call.
    """

    def __init__(
        self,
        palabra: Palabra,
        init: dict[str, Any],
        *,
        session: Session | None = None,
        ws_url: str | None = None,
        token: str | None = None,
    ):
        self._palabra = palabra
        self._init = init
        self._session = session
        self._own_session = session is None and ws_url is None
        self._direct = (ws_url, token) if ws_url else None
        self._ws: websockets.ClientConnection | None = None
        self._events: asyncio.Queue[Event | None] = asyncio.Queue()
        self._recv_task: asyncio.Task | None = None
        self._recv_error: BaseException | None = None

    async def __aenter__(self) -> TtsSession:
        if self._direct:
            url = f"{self._direct[0]}?token={self._direct[1]}"
        else:
            if self._session is None:
                self._session = await self._palabra.create_session()
            if not self._session.ws_tts_url:
                raise SessionError("Session has no ws_tts_url (Realtime TTS API not available?)")
            url = f"{self._session.ws_tts_url}?token={self._session.publisher}"
        try:
            self._ws = await websockets.connect(url, ping_interval=10, ping_timeout=30, max_size=None)
        except Exception as e:
            if self._own_session and self._session is not None:
                await self._palabra.delete_session(self._session.id)
            raise SessionError(f"TTS WebSocket connection failed: {e}") from e
        self._recv_task = asyncio.create_task(self._receive_loop())
        await self._send({"type": "init", **self._init})
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._recv_task
            self._recv_task = None
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
        if self._own_session and self._session is not None:
            await self._palabra.delete_session(self._session.id)

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._events.put(_parse_tts_event(msg))
        except websockets.ConnectionClosed:
            pass
        except Exception as e:  # surfaced by __anext__/receive, not swallowed
            self._recv_error = e
        finally:
            await self._events.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self) -> Event:
        ev = await self._events.get()
        if ev is None:
            if self._recv_error is not None:
                raise SessionError(f"Receive loop failed: {self._recv_error}") from self._recv_error
            raise StopAsyncIteration
        return ev

    async def receive(self, timeout: float | None = None) -> Event | None:
        ev = await (self._events.get() if timeout is None else asyncio.wait_for(self._events.get(), timeout))
        if ev is None and self._recv_error is not None:
            raise SessionError(f"Receive loop failed: {self._recv_error}") from self._recv_error
        return ev

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise SessionError("TTS session is not connected")
        await self._ws.send(json.dumps(payload))

    async def send_text(
        self,
        text: str,
        *,
        eos: bool = False,
        generation_id: str | None = None,
        **voice_overrides: Any,
    ) -> None:
        """One raw text message (max 256 chars, the server limit).

        voice_overrides (voice_id/speed/deaccent_strength) apply to this and
        subsequent messages. Rate limit: 50 text messages per second.

        Raises ValueError if text is empty or longer than 256 characters --
        splitting is the caller's responsibility (mark the last piece of a
        sentence with eos=True).
        """
        if not text:
            raise ValueError("Empty text")
        if len(text) > MAX_TEXT_LEN:
            raise ValueError(f"Text is {len(text)} chars; the server limit is {MAX_TEXT_LEN} per message")
        payload: dict[str, Any] = {"type": "text", "text": text, "is_eos": eos}
        if generation_id is not None:
            payload["generation_id"] = generation_id
        if voice_overrides:
            payload["voice_options"] = voice_overrides
        await self._send(payload)

    async def synthesize(self, text: str, *, timeout: float = 60.0, **voice_overrides: Any) -> bytes:
        """One sentence (max 256 chars) -> audio bytes.

        Sends the text with eos=True and collects chunks until last_chunk.
        Chunks of other generations queued in between are dropped; raises
        TaskError on a server error event.
        """
        gen = uuid.uuid4().hex[:8]
        await self.send_text(text, eos=True, generation_id=gen, **voice_overrides)
        out = bytearray()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            ev = await self.receive(timeout=deadline - loop.time())
            if ev is None:
                raise SessionError("TTS connection closed mid-synthesis")
            if isinstance(ev, ServerError):
                raise TaskError(ev.code, ev.desc)
            if isinstance(ev, TtsChunk) and ev.generation_id == gen:
                out.extend(ev.audio)
                if ev.last_chunk:
                    return bytes(out)
            elif isinstance(ev, Raw):
                continue

    async def cancel(self) -> None:
        """Stop the current synthesis; the session stays open."""
        await self._send({"type": "cancel"})
