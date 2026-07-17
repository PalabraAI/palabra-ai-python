from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import websockets

from .audio import OUTPUT_SAMPLE_RATE, chunks, load_pcm, paced_chunks, write_wav
from .events import Audio, Event, ServerError, TaskInfo, Transcript, parse_event
from .exc import AuthError, NotReadyError, SessionError, TaskError
from .task import build_task

if TYPE_CHECKING:
    from .stt import SttSession
    from .tts import TtsSession

DEFAULT_API_URL = "https://api.palabra.ai"
READY_TIMEOUT = 30.0
GET_TASK_INTERVAL = 2.1  # server allows 1 get_task per 2s
SESSION_RETRIES = 3  # create_session attempts on network errors / 5xx
RETRY_BACKOFF = 0.5  # seconds; doubles per attempt (0.5, 1.0)
S2S_SESSION_INTENT = "api"


@dataclass(frozen=True)
class Session:
    """What POST /session-storage/session returns."""

    id: str
    publisher: str = field(repr=False)  # the token; kept out of repr/logs
    ws_url: str
    webrtc_url: str = ""
    webrtc_room_name: str = ""
    subscriber: tuple[str, ...] = ()


class Palabra:
    """Entry point: credentials, session management, stream factory.

    Credentials are only needed for the REST API. With a ready ws_url +
    publisher token (issued by your backend, for example) pass them to
    translation()/tts() directly and skip REST entirely.
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        *,
        api_url: str = DEFAULT_API_URL,
    ):
        self.client_id = client_id or os.getenv("PALABRA_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("PALABRA_CLIENT_SECRET")
        self.api_url = api_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        if not self.client_id or not self.client_secret:
            raise AuthError(
                "Missing credentials: pass client_id/client_secret or set "
                "PALABRA_CLIENT_ID / PALABRA_CLIENT_SECRET "
                "(or pass ws_url= and token= to translation() to skip the REST API)"
            )
        return {"ClientID": self.client_id, "ClientSecret": self.client_secret}

    async def create_session(self, *, intent: str | None = None) -> Session:
        """Create a streaming session via REST.

        intent is the session kind sent as data.intent; session-storage routes
        and bills by it. Each product hardcodes its own value (s2s "api",
        tts "tts_api", stt "stt") — not a user choice. Omitted when None.

        Transient failures (network errors, 5xx) are retried up to
        SESSION_RETRIES times with backoff; 4xx fails immediately.
        """
        data: dict[str, Any] = {}
        if intent is not None:
            data["intent"] = intent
        last_error: Exception | None = None
        for attempt in range(SESSION_RETRIES):
            if attempt:
                await asyncio.sleep(RETRY_BACKOFF * 2 ** (attempt - 1))
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        f"{self.api_url}/session-storage/session",
                        headers=self._headers(),
                        json={"data": data},
                    )
            except httpx.TransportError as e:
                last_error = e
                continue
            if resp.status_code >= 500:
                last_error = SessionError(f"Session creation failed ({resp.status_code}): {resp.text[:500]}")
                continue
            if resp.status_code in (401, 403, 404):
                raise AuthError(f"Session creation failed ({resp.status_code}): check your credentials")
            if resp.status_code >= 400:
                raise SessionError(f"Session creation failed ({resp.status_code}): {resp.text[:500]}")
            data = resp.json().get("data") or {}
            return Session(
                id=data.get("id", ""),
                publisher=data.get("publisher", ""),
                ws_url=data.get("ws_url", ""),
                webrtc_url=data.get("webrtc_url", ""),
                webrtc_room_name=data.get("webrtc_room_name", ""),
                subscriber=tuple(data.get("subscriber") or ()),
            )
        raise SessionError(
            f"Session creation failed after {SESSION_RETRIES} attempts: {last_error}"
        ) from last_error

    async def delete_session(self, session_id: str) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            with contextlib.suppress(httpx.HTTPError):
                await client.delete(
                    f"{self.api_url}/session-storage/sessions/{session_id}",
                    headers=self._headers(),
                )

    def translation(
        self,
        source: str | None = None,
        targets: str | Sequence[str] | Mapping[str, dict] | None = None,
        *,
        task: dict[str, Any] | None = None,
        session: Session | None = None,
        ws_url: str | None = None,
        token: str | None = None,
        **task_options: Any,
    ) -> TranslationSession:
        """Open a translation stream, use as `async with`.

        Settings: either source/targets (+ build_task options), or a complete
        task= dict.

        Connection: by default a session is created via REST and deleted on
        exit. ws_url= + token= connect directly with an existing publisher
        token (no REST, nothing deleted on exit). session= reuses a Session
        from create_session() whose lifecycle you own.
        """
        if task is None:
            if source is None or targets is None:
                raise ValueError("Pass source and targets, or a complete task=...")
            task = build_task(source, targets, **task_options)
        elif task_options or source or targets:
            raise ValueError("task= is mutually exclusive with source/targets/options")

        if (ws_url is None) != (token is None):
            raise ValueError("ws_url and token must be passed together")
        if ws_url is not None:
            if session is not None:
                raise ValueError("session= is mutually exclusive with ws_url/token")
            session = Session(id="", publisher=token, ws_url=ws_url)
        return TranslationSession(self, task, session=session)

    def tts(
        self,
        language: str,
        *,
        voice_id: str = "default_low",
        speed: float | None = None,
        deaccent_strength: float = 1.0,
        model: str = "auto",
        format: str = "pcm",
        sample_rate: int = 24000,
        session: Session | None = None,
        ws_url: str | None = None,
        token: str | None = None,
    ) -> TtsSession:
        """Open a standalone Realtime TTS session (use as `async with`).

        This is the dedicated text-to-speech API (no translation pipeline) --
        a different product from translation(). To speak text inside a translation
        session, use TranslationSession.speak().

        format: pcm (s16le, the client default) | mp3 | wav.
        Connection options work like in translation(): default REST session,
        or ws_url=/token= to connect directly. Like the ASR endpoint, the TTS
        endpoint is fixed (TTS_STREAM_URL), not taken from the session response.
        """
        from .tts import TtsSession

        # deaccent_strength is sent explicitly: the intended default is 1.0,
        # while the current server-side model default is 0.0
        voice_options: dict[str, Any] = {"voice_id": voice_id, "deaccent_strength": deaccent_strength}
        if speed is not None:
            voice_options["speed"] = speed
        init = {
            "language": language,
            "model": model,
            "voice_options": voice_options,
            "output": {"format": format, "sample_rate": sample_rate},
        }
        if (ws_url is None) != (token is None):
            raise ValueError("ws_url and token must be passed together")
        if ws_url is not None and session is not None:
            raise ValueError("session= is mutually exclusive with ws_url/token")
        return TtsSession(self, init, session=session, ws_url=ws_url, token=token)

    def stt(
        self,
        language: str | None = None,
        *,
        format: str = "pcm_s16le",
        sample_rate: int | None = None,
        translate_languages: str | Sequence[str] | None = None,
        enable_filler_filter: bool | None = None,
        session: Session | None = None,
        ws_url: str | None = None,
        token: str | None = None,
    ) -> SttSession:
        """Open a standalone Realtime Speech-to-Text (ASR) session (use as `async with`).

        Settings are sent as URL query parameters:
        - language: source language code; defaults to auto-detect when omitted.
        - format: audio format (pcm_s16le | pcm_f32le/be | pcm_s32le/be | mulaw |
          alaw | webm | mp3 | aac | ogg | flac | wav); see the docs.
        - sample_rate: required for raw PCM formats other than 16 kHz pcm_s16le;
          omitted when None (the server assumes 16000).
        - translate_languages: target language(s) for translated_transcription
          (a comma string or a sequence of codes).
        - enable_filler_filter: server default is True for every language but ja.

        Connection options work like in translation(): default REST session,
        session=, or ws_url=/token= to connect directly. Like the TTS endpoint,
        the ASR endpoint is fixed (derived from api_url), not taken from the
        session response.
        """
        from .stt import SttSession

        params: dict[str, str] = {"format": format}
        if sample_rate is not None:
            params["sample_rate"] = str(sample_rate)
        if language is not None:
            params["language"] = language
        if translate_languages:
            if not isinstance(translate_languages, str):
                translate_languages = ",".join(translate_languages)
            params["translate_languages"] = translate_languages
        if enable_filler_filter is not None:
            params["enable_filler_filter"] = "true" if enable_filler_filter else "false"

        if (ws_url is None) != (token is None):
            raise ValueError("ws_url and token must be passed together")
        if ws_url is not None and session is not None:
            raise ValueError("session= is mutually exclusive with ws_url/token")
        return SttSession(self, params, session=session, ws_url=ws_url, token=token)

    async def atranslate_file(
        self,
        path: str | Path,
        *,
        source: str,
        targets: str | Sequence[str],
        output: str | Path | None = None,
        on_transcript: Callable[[Transcript], None] | None = None,
        **task_options: Any,
    ) -> dict[str, bytes]:
        """Translate an audio file, returns {language: pcm_s16le_24k_mono}.

        Takes about as long as the audio itself (real-time pacing). With
        output= one WAV per target is written; "{lang}" in the name is a
        placeholder, added automatically for multiple targets.
        """
        pcm = load_pcm(path, sample_rate=24000, channels=1)
        results: dict[str, bytearray] = {}

        async with self.translation(source, targets, **task_options) as s:
            feeder = asyncio.create_task(s.send_pcm(pcm, eos_timeout=4))
            try:
                async for ev in s:
                    if isinstance(ev, Audio):
                        results.setdefault(ev.language, bytearray()).extend(ev.pcm)
                    elif isinstance(ev, Transcript) and on_transcript:
                        on_transcript(ev)
            finally:
                feeder.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await feeder

        out = {lang: bytes(buf) for lang, buf in results.items()}
        if output is not None:
            output = str(output)
            if "{lang}" not in output and len(out) > 1:
                p = Path(output)
                output = str(p.with_name(f"{p.stem}_{{lang}}{p.suffix}"))
            for lang, audio in out.items():
                write_wav(output.format(lang=lang), audio)
        return out

    def translate_file(self, path: str | Path, **kwargs: Any) -> dict[str, bytes]:
        return asyncio.run(self.atranslate_file(path, **kwargs))


class TranslationSession:
    """One live translation session over one WebSocket connection.

    Created via Palabra.translation(). On enter: create a REST session if needed,
    connect, send set_task and wait for the pipeline to confirm it. Iterating
    yields events; iteration stops when the server closes the connection
    (after end() or on a fatal error).
    """

    def __init__(self, palabra: Palabra, task: dict[str, Any], *, session: Session | None = None):
        self._palabra = palabra
        self._task = task
        self._session = session
        self._own_session = session is None
        self._ws: websockets.ClientConnection | None = None
        self._events: asyncio.Queue[Event | None] = asyncio.Queue()
        self._recv_task: asyncio.Task | None = None
        self._recv_error: BaseException | None = None
        self._setup_error: ServerError | None = None
        self._ready = asyncio.Event()
        self._ended = False
        self._zlib = task.get("output_stream", {}).get("target", {}).get("format") == "zlib_pcm_s16le"
        src = task.get("input_stream", {}).get("source", {})
        self._in_rate = src.get("sample_rate", OUTPUT_SAMPLE_RATE)
        self._in_channels = src.get("channels", 1)

    @property
    def session(self) -> Session:
        if self._session is None:
            raise SessionError("Session is not created yet (use 'async with')")
        return self._session

    @property
    def task(self) -> dict[str, Any]:
        return self._task

    async def __aenter__(self) -> TranslationSession:
        if self._session is None:
            self._session = await self._palabra.create_session(intent=S2S_SESSION_INTENT)
        url = f"{self._session.ws_url}?token={self._session.publisher}"
        try:
            self._ws = await websockets.connect(url, ping_interval=10, ping_timeout=30, max_size=None)
        except Exception as e:
            if self._own_session:
                await self._palabra.delete_session(self._session.id)
            raise SessionError(f"WebSocket connection failed: {e}") from e
        self._recv_task = asyncio.create_task(self._receive_loop())
        try:
            await self._send("set_task", self._task)
            await self._wait_ready()
        except BaseException:
            await self.close()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if exc_type is None and not self._ended and self._ws is not None:
            with contextlib.suppress(Exception):
                await self.end()
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

    async def _wait_ready(self) -> None:
        # The server doesn't acknowledge set_task. get_task returns NOT_FOUND
        # until the pipeline has started, then current_task -- so poll it.
        # A non-NOT_FOUND error before readiness (e.g. VALIDATION_ERROR)
        # means the task was rejected -- fail fast with the real reason.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + READY_TIMEOUT
        while True:
            if self._setup_error is not None:
                raise TaskError(self._setup_error.code, self._setup_error.desc)
            if self._ready.is_set():
                return
            await self._send("get_task", {})
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._ready.wait(), timeout=GET_TASK_INTERVAL)
            if self._setup_error is not None:
                raise TaskError(self._setup_error.code, self._setup_error.desc)
            if self._recv_task is not None and self._recv_task.done():
                raise SessionError("Connection closed before the task was confirmed")
            if loop.time() > deadline:
                raise NotReadyError(f"Pipeline did not confirm the task within {READY_TIMEOUT:.0f}s")

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                    if isinstance(msg.get("data"), str):  # server may double-encode data
                        msg["data"] = json.loads(msg["data"])
                except (json.JSONDecodeError, AttributeError):
                    continue
                event = parse_event(msg, zlib_output=self._zlib)
                if isinstance(event, TaskInfo):
                    self._ready.set()
                if isinstance(event, ServerError) and not self._ready.is_set():
                    if event.code == "NOT_FOUND":
                        continue  # expected while polling readiness
                    # set_task was rejected: hand the error to _wait_ready
                    self._setup_error = event
                    self._ready.set()
                    continue
                await self._events.put(event)
        except websockets.ConnectionClosed:
            pass
        except Exception as e:  # surfaced by __anext__/receive, not swallowed
            self._recv_error = e
        finally:
            await self._events.put(None)  # end-of-stream sentinel

    def __aiter__(self) -> AsyncIterator[Event]:
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

    async def _send(self, message_type: str, data: dict[str, Any]) -> None:
        if self._ws is None:
            raise SessionError("Stream is not connected")
        await self._ws.send(json.dumps({"message_type": message_type, "data": data}))

    async def send_audio(self, chunk: bytes) -> None:
        """One raw chunk matching the task's declared format.

        Pace chunks to real time yourself; ~320 ms per chunk is optimal.
        """
        await self._send("input_audio_data", {"data": base64.b64encode(chunk).decode()})

    async def send_pcm(self, pcm: bytes, *, realtime: bool = True, eos_timeout: int | None = None) -> None:
        """Send a whole PCM buffer in 320 ms chunks, paced to real time.

        With eos_timeout, end() is called afterwards so the stream finishes
        once the tail is processed.
        """
        if realtime:
            async for chunk in paced_chunks(pcm, sample_rate=self._in_rate, channels=self._in_channels):
                await self.send_audio(chunk)
        else:
            for chunk in chunks(pcm, sample_rate=self._in_rate, channels=self._in_channels):
                await self.send_audio(chunk)
        if eos_timeout is not None:
            await self.end(eos_timeout=eos_timeout)

    async def send_file(self, path: str | Path, *, eos_timeout: int | None = 4) -> None:
        pcm = load_pcm(path, sample_rate=self._in_rate, channels=self._in_channels)
        await self.send_pcm(pcm, eos_timeout=eos_timeout)

    async def set_task(self, task: dict[str, Any]) -> None:
        """Update settings on the fly (also resumes after pause())."""
        self._task = task
        await self._send("set_task", task)

    async def get_task(self) -> None:
        """Ask for the current task; the reply arrives as a TaskInfo event."""
        await self._send("get_task", {})

    async def pause(self) -> None:
        """Pause processing (billing stops too)."""
        await self._send("pause_task", {})

    async def resume(self) -> None:
        await self._send("set_task", self._task)

    async def flush(self, languages: Sequence[str] | None = None, *, pause: bool = False) -> None:
        """Drop the phrase being processed (e.g. the speaker got interrupted)."""
        await self._send(
            "flush_task",
            {"languages": list(languages) if languages else ["global"], "pause_task": pause},
        )

    async def speak(self, text: str, language: str, *, translate: bool = False) -> None:
        """Inject text into the translation pipeline (tts_task). Rate limit: 60/min.

        For standalone text-to-speech without a translation pipeline, use the
        separate Realtime TTS API: Palabra.tts().

        translate=False: speak as-is, language must be one of the task's
        target languages. translate=True: language is the text's language,
        the text is translated to all targets first.
        """
        await self._send("tts_task", {"text": text, "language": language, "translate_text": translate})

    async def end(self, *, eos_timeout: int | None = None) -> None:
        """Finish the task; the server closes the connection afterwards.

        With eos_timeout (1-30 s) the server waits for that much silence
        after the last speech, emits an StreamEnd event and only then closes --
        use it to get the tail of the translation delivered.
        """
        self._ended = True
        data: dict[str, Any] = {}
        if eos_timeout is not None:
            data["eos_timeout"] = eos_timeout
        await self._send("end_task", data)

    def raise_on_error(self, event: Event) -> Event:
        """Turn a ServerError event into a TaskError exception."""
        if isinstance(event, ServerError):
            raise TaskError(event.code, event.desc)
        return event
