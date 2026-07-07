"""Realtime Speech-to-Text (ASR) API client tests.

The session logic runs against a fake server reproducing the documented
protocol: settings arrive as URL query parameters, audio comes in as raw
binary frames, and the server replies with top-level JSON `transcription`
(and `translated_transcription`) messages.
"""

import asyncio
import json
from urllib.parse import parse_qs, urlsplit

import pytest
import websockets

from palabra_ai import Palabra, Session, SttTranscript

pytestmark = pytest.mark.asyncio


class FakeAsrServer:
    def __init__(self, close_after=1):
        self.close_after = close_after  # close the connection after N audio frames
        self.query = None
        self.audio_frames = 0
        self.server = None
        self.port = None

    async def __aenter__(self):
        self.server = await websockets.serve(self.handler, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()

    @property
    def api_url(self):  # what Palabra(api_url=...) should point at
        return f"http://127.0.0.1:{self.port}"

    async def handler(self, ws):
        self.query = parse_qs(urlsplit(ws.request.path).query)
        async for raw in ws:
            if not isinstance(raw, bytes):
                continue  # ASR input is raw binary; ignore anything else
            self.audio_frames += 1
            tid = f"seg-{self.audio_frames}"
            # partial, then final transcription of the same segment
            await ws.send(
                json.dumps(
                    {
                        "message_type": "transcription",
                        "transcription_id": tid,
                        "language": "en",
                        "is_eos": False,
                        "segment": {"text": "hello world", "start_time": 0.3, "end_time": 1.8},
                        "delta": {"text": "world", "start_time": 1.2, "end_time": 1.8},
                    }
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "message_type": "transcription",
                        "transcription_id": tid,
                        "language": "en",
                        "is_eos": True,
                        "segment": {"text": "Hello world.", "start_time": 0.3, "end_time": 1.9},
                    }
                )
            )
            targets = self.query.get("translate_languages", [""])[0]
            for lang in targets.split(",") if targets else []:
                await ws.send(
                    json.dumps(
                        {
                            "message_type": "translated_transcription",
                            "transcription_id": tid,
                            "language": lang,
                            "is_eos": True,
                            "segment": {"text": f"<{lang}> hola mundo"},
                        }
                    )
                )
            if self.audio_frames >= self.close_after:
                await ws.close()
                return


@pytest.fixture
def palabra():
    return Palabra(client_id="test", client_secret="test")


def _session(srv):
    return Session(id="s1", publisher="tok", ws_url="")


async def test_connect_query_and_transcription(palabra):
    async with FakeAsrServer() as srv:
        palabra.api_url = srv.api_url  # ASR endpoint is derived from api_url
        events = []
        async with palabra.stt(language="en", session=_session(srv)) as stt:
            await stt.send_audio(b"\x00\x00" * 100)
            async for ev in stt:
                events.append(ev)

        assert srv.audio_frames == 1
        # settings went out as query parameters, including the session token
        assert srv.query["token"] == ["tok"]
        assert srv.query["format"] == ["pcm_s16le"]
        assert srv.query["language"] == ["en"]

        transcripts = [e for e in events if isinstance(e, SttTranscript)]
        assert any(not t.is_eos and t.text == "hello world" and t.delta == "world" for t in transcripts)
        final = next(t for t in transcripts if t.is_eos)
        assert final.text == "Hello world." and final.transcription_id == "seg-1"
        assert final.end_time == 1.9 and not final.is_translation


async def test_translate_languages_query_and_messages(palabra):
    async with FakeAsrServer() as srv:
        palabra.api_url = srv.api_url
        events = []
        async with palabra.stt(language="en", translate_languages=["es", "de"], session=_session(srv)) as stt:
            await stt.send_audio(b"\x00\x00" * 100)
            async for ev in stt:
                events.append(ev)

        # sequence joined into a comma string (kept literal in the URL)
        assert srv.query["translate_languages"] == ["es,de"]
        translations = [e for e in events if isinstance(e, SttTranscript) and e.is_translation]
        assert {t.language for t in translations} == {"es", "de"}
        assert all(t.is_eos for t in translations)


async def test_enable_filler_filter_query(palabra):
    async with FakeAsrServer() as srv:
        palabra.api_url = srv.api_url
        async with palabra.stt(
            language="ja", enable_filler_filter=False, sample_rate=48000, session=_session(srv)
        ) as stt:
            await stt.send_audio(b"\x00\x00" * 100)
            async for _ in stt:
                pass
        assert srv.query["enable_filler_filter"] == ["false"]
        assert srv.query["sample_rate"] == ["48000"]


async def test_send_pcm_paces_realtime(palabra):
    async with FakeAsrServer(close_after=2) as srv:
        palabra.api_url = srv.api_url
        async with palabra.stt(language="en", session=_session(srv)) as stt:
            # 640 ms at 16 kHz mono -> 2 chunks of 320 ms
            pcm = b"\x00\x00" * int(16000 * 0.64)
            loop = asyncio.get_event_loop()
            t0 = loop.time()
            await stt.send_pcm(pcm)
            assert loop.time() - t0 >= 0.3  # paced (first chunk immediate, then 1 x 320 ms)
            async for _ in stt:
                pass
        assert srv.audio_frames == 2


async def test_direct_ws_url_token_bypasses_rest(monkeypatch):
    """ws_url+token: no credentials, no REST calls at all."""
    monkeypatch.delenv("PALABRA_CLIENT_ID", raising=False)
    monkeypatch.delenv("PALABRA_CLIENT_SECRET", raising=False)
    palabra = Palabra()

    async def boom(*a, **kw):
        raise AssertionError("REST API must not be used with ws_url/token")

    monkeypatch.setattr(palabra, "create_session", boom)
    monkeypatch.setattr(palabra, "delete_session", boom)

    async with FakeAsrServer() as srv:
        base = f"ws://127.0.0.1:{srv.port}/asr/v1/speech-to-text/stream"
        async with palabra.stt(language="en", ws_url=base, token="tok") as stt:
            await stt.send_audio(b"\x00\x00" * 100)
            async for _ in stt:
                pass
        assert srv.audio_frames == 1
        assert srv.query["token"] == ["tok"]


async def test_stt_factory_validation(palabra):
    with pytest.raises(ValueError):
        palabra.stt("en", ws_url="ws://x")  # token missing
    with pytest.raises(ValueError):
        palabra.stt("en", token="t")  # ws_url missing
    with pytest.raises(ValueError):
        palabra.stt(
            "en",
            ws_url="ws://x",
            token="t",
            session=Session(id="s", publisher="p", ws_url=""),
        )


async def test_receive_loop_failure_surfaces_as_session_error(palabra):
    """A crash in the receive loop must raise SessionError on iteration."""
    import palabra_ai.stt as stt_mod
    from palabra_ai import SessionError

    def boom(_msg):
        raise RuntimeError("parse exploded")

    async with FakeAsrServer() as srv:
        palabra.api_url = srv.api_url
        orig = stt_mod._parse_stt_event
        stt_mod._parse_stt_event = boom
        try:
            async with palabra.stt(language="en", session=_session(srv)) as stt:
                await stt.send_audio(b"\x00\x00" * 100)
                with pytest.raises(SessionError) as ei:
                    async for _ in stt:
                        pass
                assert isinstance(ei.value.__cause__, RuntimeError)
        finally:
            stt_mod._parse_stt_event = orig
