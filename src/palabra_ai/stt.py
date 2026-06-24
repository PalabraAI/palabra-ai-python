from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import websockets

from .audio import chunks, load_pcm, paced_chunks
from .events import Event, parse_event
from .exc import AuthError, SessionError

if TYPE_CHECKING:
    from .client import Palabra, Session

ASR_STREAM_PATH = "/asr/v1/speech-to-text/stream"
DEFAULT_SAMPLE_RATE = 16000  # ASR recommended/default input rate (translation/TTS use 24k)

_STT_TRANSCRIPT_TYPES = frozenset({"transcription", "translated_transcription"})


@dataclass(frozen=True)
class SttTranscript(Event):
    """One transcription / translated_transcription message.

    segment.text (``text`` here) is authoritative: with the filler filter
    enabled the recognizer's tail may be rewritten mid-segment, so render the
    whole segment on each message rather than appending ``delta``.
    """

    text: str
    language: str
    transcription_id: str
    is_eos: bool
    is_translation: bool
    start_time: float | None = None
    end_time: float | None = None
    delta: str | None = None

    def __str__(self) -> str:
        prefix = "" if self.is_eos else "~ "
        return f"{prefix}[{self.language}] {self.text}"


def _parse_stt_event(msg: dict[str, Any]) -> Event:
    mtype = msg.get("message_type", "")
    if mtype in _STT_TRANSCRIPT_TYPES:
        seg = msg.get("segment") or {}
        delta = msg.get("delta") or {}
        return SttTranscript(
            type=mtype,
            text=seg.get("text", ""),
            language=msg.get("language", ""),
            transcription_id=msg.get("transcription_id", ""),
            is_eos=bool(msg.get("is_eos", False)),
            is_translation=mtype == "translated_transcription",
            start_time=seg.get("start_time"),
            end_time=seg.get("end_time"),
            delta=delta.get("text") if delta else None,
        )
    return parse_event(msg)  # anything unexpected


def _asr_ws_url(api_url: str) -> str:
    """Derive the ASR WebSocket endpoint from the REST api_url (same host)."""
    if api_url.startswith("https://"):
        base = "wss://" + api_url[len("https://") :]
    elif api_url.startswith("http://"):
        base = "ws://" + api_url[len("http://") :]
    else:
        base = api_url
    return base.rstrip("/") + ASR_STREAM_PATH


class SttSession:
    """One Realtime ASR session over one WebSocket connection.

    Created via Palabra.stt(). On enter: create a REST session if needed,
    connect to the ASR endpoint with all settings as query parameters, and
    start receiving. Iterating yields events; iteration stops when the server
    closes the connection.
    """

    def __init__(
        self,
        palabra: Palabra,
        params: dict[str, str],
        *,
        session: Session | None = None,
        ws_url: str | None = None,
        token: str | None = None,
    ):
        self._palabra = palabra
        self._params = params  # query parameters, excluding the token
        self._session = session
        self._own_session = session is None and ws_url is None
        self._direct = (ws_url, token) if ws_url else None
        self._in_rate = int(params.get("sample_rate") or DEFAULT_SAMPLE_RATE)
        self._ws: websockets.ClientConnection | None = None
        self._events: asyncio.Queue[Event | None] = asyncio.Queue()
        self._recv_task: asyncio.Task | None = None
        self._recv_error: BaseException | None = None

    async def __aenter__(self) -> SttSession:
        if self._direct:
            base, token = self._direct
        else:
            if self._session is None:
                self._session = await self._palabra.create_session()
            base = _asr_ws_url(self._palabra.api_url)
            token = self._session.publisher
        url = f"{base}?{urlencode({'token': token, **self._params}, safe=',')}"
        try:
            self._ws = await websockets.connect(url, ping_interval=10, ping_timeout=30, max_size=None)
        except Exception as e:
            if self._own_session and self._session is not None:
                await self._palabra.delete_session(self._session.id)

            # Auth/routing failures surface as the HTTP status of the failed upgrade.
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 401:
                raise AuthError("STT connection rejected (401): invalid or missing token") from e
            if status == 409:
                raise SessionError(
                    "STT connection rejected (409): a session is already active for this identity"
                ) from e
            raise SessionError(f"STT WebSocket connection failed: {e}") from e
        self._recv_task = asyncio.create_task(self._receive_loop())
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
                except (json.JSONDecodeError, TypeError):
                    continue
                await self._events.put(_parse_stt_event(msg))
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
        """Next event, or None once the stream has ended."""
        ev = await (self._events.get() if timeout is None else asyncio.wait_for(self._events.get(), timeout))
        if ev is None and self._recv_error is not None:
            raise SessionError(f"Receive loop failed: {self._recv_error}") from self._recv_error
        return ev

    async def send_audio(self, chunk: bytes) -> None:
        """One raw audio frame in the configured format, as a binary WS frame.

        Pace chunks to real time yourself; ~320 ms per chunk is recommended.
        """
        if self._ws is None:
            raise SessionError("STT session is not connected")
        await self._ws.send(chunk)

    async def send_pcm(self, pcm: bytes, *, realtime: bool = True) -> None:
        """Send a whole PCM buffer in 320 ms chunks (mono, at the configured rate).

        With realtime=True chunks are paced to real time (recommended for the
        live service); with realtime=False they are sent back-to-back.
        """
        if realtime:
            async for chunk in paced_chunks(pcm, sample_rate=self._in_rate, channels=1):
                await self.send_audio(chunk)
        else:
            for chunk in chunks(pcm, sample_rate=self._in_rate, channels=1):
                await self.send_audio(chunk)

    async def send_file(self, path: str | Path) -> None:
        pcm = load_pcm(path, sample_rate=self._in_rate, channels=1)
        await self.send_pcm(pcm)
