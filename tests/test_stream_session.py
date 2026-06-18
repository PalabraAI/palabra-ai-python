"""End-to-end test against a local fake server emulating the real protocol.

The fake server reproduces the documented behaviour:

- no acknowledgement of ``set_task``; ``get_task`` returns a NOT_FOUND
  ``error`` until the pipeline has "started", then ``current_task``;
- ``input_audio_data`` produces transcriptions and flat ``output_audio_data``;
- ``end_task`` with ``eos_timeout`` emits ``eos`` and closes the connection.
"""

import asyncio
import base64
import json

import pytest
import websockets

from palabra_ai import Audio, Palabra, Session, StreamEnd, Transcript

pytestmark = pytest.mark.asyncio


class FakeServer:
    def __init__(self):
        self.received_audio_chunks = 0
        self.task = None
        self.started = False
        self.server = None
        self.port = None

    async def __aenter__(self):
        self.server = await websockets.serve(self.handler, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()

    async def handler(self, ws):
        async for raw in ws:
            msg = json.loads(raw)
            mtype, data = msg["message_type"], msg.get("data", {})
            if mtype == "set_task":
                self.task = data
                asyncio.get_event_loop().call_later(0.3, self._start)
            elif mtype == "get_task":
                if not self.started:
                    await ws.send(
                        json.dumps(
                            {
                                "message_type": "error",
                                "data": {"code": "NOT_FOUND", "desc": "No active task found"},
                            }
                        )
                    )
                else:
                    await ws.send(
                        json.dumps(
                            {
                                "message_type": "current_task",
                                "data": {"task_status": "running", **self.task},
                            }
                        )
                    )
            elif mtype == "input_audio_data":
                self.received_audio_chunks += 1
                lang = self.task["pipeline"]["translations"][0]["target_language"]

                def tr(mt, language, text):
                    return json.dumps(
                        {
                            "message_type": mt,
                            "data": {
                                "transcription": {
                                    "transcription_id": "t1",
                                    "language": language,
                                    "text": text,
                                }
                            },
                        }
                    )

                await ws.send(tr("partial_transcription", "en", "one two"))
                await ws.send(tr("translated_transcription", lang, "uno dos"))
                # flat output_audio_data -- as produced by nats_ws_writer.py
                await ws.send(
                    json.dumps(
                        {
                            "message_type": "output_audio_data",
                            "data": {
                                "transcription_id": "t1",
                                "translation_part_id": "0",
                                "language": lang,
                                "last_chunk": True,
                                "data": base64.b64encode(b"\x01\x00" * 100).decode(),
                            },
                        }
                    )
                )
            elif mtype == "end_task":
                if data.get("eos_timeout") is not None:
                    await ws.send(json.dumps({"message_type": "eos", "data": {}}))
                await ws.close()
                return

    def _start(self):
        self.started = True


@pytest.fixture
def palabra(monkeypatch):
    p = Palabra(client_id="test", client_secret="test")
    return p


async def test_full_stream_flow(palabra):
    async with FakeServer() as srv:
        session = Session(id="s1", publisher="tok", ws_url=f"ws://127.0.0.1:{srv.port}")

        events = []
        async with palabra.translation("en", "es", session=session) as s:
            chunk = b"\x00\x00" * int(24000 * 0.32)
            await s.send_audio(chunk)
            await s.end(eos_timeout=2)
            async for ev in s:
                events.append(ev)

        assert srv.received_audio_chunks == 1
        transcripts = [e for e in events if isinstance(e, Transcript)]
        audio = [e for e in events if isinstance(e, Audio)]
        assert any(t.is_eos and t.is_translation and t.text == "uno dos" for t in transcripts)
        assert any(not t.is_eos and t.text == "one two" for t in transcripts)
        assert audio and audio[0].pcm == b"\x01\x00" * 100 and audio[0].last_chunk
        assert any(isinstance(e, StreamEnd) for e in events)


async def test_direct_ws_url_token_without_credentials(monkeypatch):
    """ws_url+token bypass: no credentials, no REST calls at all."""
    monkeypatch.delenv("PALABRA_CLIENT_ID", raising=False)
    monkeypatch.delenv("PALABRA_CLIENT_SECRET", raising=False)
    palabra = Palabra()  # no credentials -- fine for direct connection

    async def boom(*a, **kw):
        raise AssertionError("REST API must not be used with ws_url/token")

    monkeypatch.setattr(palabra, "create_session", boom)
    monkeypatch.setattr(palabra, "delete_session", boom)

    async with FakeServer() as srv:
        async with palabra.translation("en", "es", ws_url=f"ws://127.0.0.1:{srv.port}", token="tok") as s:
            await s.send_audio(b"\x00\x00" * int(24000 * 0.32))
            await s.end(eos_timeout=1)
            async for _ in s:
                pass
        assert srv.received_audio_chunks == 1


async def test_credentials_required_only_for_rest(monkeypatch):
    monkeypatch.delenv("PALABRA_CLIENT_ID", raising=False)
    monkeypatch.delenv("PALABRA_CLIENT_SECRET", raising=False)
    from palabra_ai import AuthError

    palabra = Palabra()  # does not raise
    with pytest.raises(AuthError):
        await palabra.create_session()


async def test_stream_argument_validation(palabra):
    with pytest.raises(ValueError):
        palabra.translation("en", "es", ws_url="ws://x")  # token missing
    with pytest.raises(ValueError):
        palabra.translation("en", "es", token="tok")  # ws_url missing
    with pytest.raises(ValueError):
        palabra.translation(
            "en",
            "es",
            ws_url="ws://x",
            token="tok",
            session=Session(id="s", publisher="p", ws_url="ws://y"),
        )


async def test_send_pcm_paces_realtime(palabra):
    async with FakeServer() as srv:
        session = Session(id="s1", publisher="tok", ws_url=f"ws://127.0.0.1:{srv.port}")
        async with palabra.translation("en", "es", session=session) as s:
            pcm = b"\x00\x00" * int(24000 * 0.96)  # 960 ms -> 3 chunks of 320 ms
            loop = asyncio.get_event_loop()
            t0 = loop.time()
            await s.send_pcm(pcm)
            elapsed = loop.time() - t0
            assert elapsed >= 0.6  # paced (first chunk immediate, then 2 x 320 ms)
            await s.end(eos_timeout=1)
            async for _ in s:
                pass
        assert srv.received_audio_chunks == 3


async def test_rejected_set_task_fails_fast_with_task_error(palabra):
    """An ``error`` other than NOT_FOUND before readiness must raise TaskError
    immediately (with the server's reason), not NotReadyError after 30 s."""
    from palabra_ai import TaskError

    async def handler(ws):
        async for raw in ws:
            if json.loads(raw)["message_type"] == "set_task":
                await ws.send(
                    json.dumps(
                        {
                            "message_type": "error",
                            "data": {"code": "VALIDATION_ERROR", "desc": "bad sample_rate"},
                        }
                    )
                )

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        session = Session(id="s1", publisher="tok", ws_url=f"ws://127.0.0.1:{port}")
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        with pytest.raises(TaskError) as ei:
            async with palabra.translation("en", "es", session=session):
                pass
        assert ei.value.code == "VALIDATION_ERROR"
        assert loop.time() - t0 < 5  # no 30 s readiness timeout
    finally:
        server.close()
        await server.wait_closed()


async def test_receive_loop_failure_surfaces_as_session_error(palabra):
    """A crash in the receive loop (e.g. corrupt zlib output) must raise
    SessionError on iteration, not end the stream silently."""
    import zlib

    from palabra_ai import SessionError

    async with FakeServer() as srv:
        session = Session(id="s1", publisher="tok", ws_url=f"ws://127.0.0.1:{srv.port}")
        # zlib output declared, but FakeServer sends plain pcm -> decompress fails
        async with palabra.translation("en", "es", output_format="zlib_pcm_s16le", session=session) as s:
            await s.send_audio(b"\x00\x00" * int(24000 * 0.32))
            with pytest.raises(SessionError) as ei:
                async for _ in s:
                    pass
            assert isinstance(ei.value.__cause__, zlib.error)


class _FakeHttpResponse:
    def __init__(self, status, data=None):
        self.status_code = status
        self.text = "server error"
        self._data = data or {}

    def json(self):
        return {"data": self._data}


def _fake_http_client(responses, calls):
    """httpx.AsyncClient stand-in: pops one canned response per post()."""

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def post(self, *a, **kw):
            calls.append(1)
            resp = responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return resp

    return FakeClient


async def test_create_session_retries_transient_then_succeeds(palabra, monkeypatch):
    import httpx

    calls = []
    responses = [
        httpx.ConnectError("boom"),
        _FakeHttpResponse(503),
        _FakeHttpResponse(200, {"id": "s1", "publisher": "tok", "ws_url": "ws://x"}),
    ]
    monkeypatch.setattr("palabra_ai.client.httpx.AsyncClient", _fake_http_client(responses, calls))
    monkeypatch.setattr("palabra_ai.client.RETRY_BACKOFF", 0)

    session = await palabra.create_session()
    assert session.id == "s1"
    assert len(calls) == 3


async def test_create_session_does_not_retry_4xx(palabra, monkeypatch):
    from palabra_ai import AuthError, SessionError

    calls = []
    monkeypatch.setattr(
        "palabra_ai.client.httpx.AsyncClient", _fake_http_client([_FakeHttpResponse(401)], calls)
    )
    with pytest.raises(AuthError):
        await palabra.create_session()
    assert len(calls) == 1  # no retries on auth errors

    calls.clear()
    monkeypatch.setattr(
        "palabra_ai.client.httpx.AsyncClient", _fake_http_client([_FakeHttpResponse(422)], calls)
    )
    with pytest.raises(SessionError):
        await palabra.create_session()
    assert len(calls) == 1  # no retries on client errors


async def test_create_session_exhausts_retries(palabra, monkeypatch):
    from palabra_ai import SessionError
    from palabra_ai.client import SESSION_RETRIES

    calls = []
    responses = [_FakeHttpResponse(503)] * SESSION_RETRIES
    monkeypatch.setattr("palabra_ai.client.httpx.AsyncClient", _fake_http_client(responses, calls))
    monkeypatch.setattr("palabra_ai.client.RETRY_BACKOFF", 0)

    with pytest.raises(SessionError, match="after 3 attempts"):
        await palabra.create_session()
    assert len(calls) == SESSION_RETRIES
