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

from palabra_ai import Palabra, Region, SttTranscript

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
    return Palabra(api_key="test-key")


def _fake_region(monkeypatch, srv):
    """Point the 'eu' region's STT endpoint at the fake server."""
    monkeypatch.setattr(
        "palabra_ai.client.REGIONS",
        {"eu": Region(stt=f"ws://127.0.0.1:{srv.port}/asr/v1/speech-to-text/stream")},
    )


async def test_connect_query_and_transcription(palabra, monkeypatch):
    async with FakeAsrServer() as srv:
        _fake_region(monkeypatch, srv)
        events = []
        async with palabra.stt(language="en") as stt:
            await stt.send_audio(b"\x00\x00" * 100)
            async for ev in stt:
                events.append(ev)

        assert srv.audio_frames == 1
        # settings went out as query parameters, including the API Key as token
        assert srv.query["token"] == ["test-key"]
        assert srv.query["format"] == ["pcm_s16le"]
        assert srv.query["language"] == ["en"]

        transcripts = [e for e in events if isinstance(e, SttTranscript)]
        assert any(not t.is_eos and t.text == "hello world" and t.delta == "world" for t in transcripts)
        final = next(t for t in transcripts if t.is_eos)
        assert final.text == "Hello world." and final.transcription_id == "seg-1"
        assert final.end_time == 1.9 and not final.is_translation


async def test_translate_languages_query_and_messages(palabra, monkeypatch):
    async with FakeAsrServer() as srv:
        _fake_region(monkeypatch, srv)
        events = []
        async with palabra.stt(language="en", translate_languages=["es", "de"]) as stt:
            await stt.send_audio(b"\x00\x00" * 100)
            async for ev in stt:
                events.append(ev)

        # sequence joined into a comma string (kept literal in the URL)
        assert srv.query["translate_languages"] == ["es,de"]
        translations = [e for e in events if isinstance(e, SttTranscript) and e.is_translation]
        assert {t.language for t in translations} == {"es", "de"}
        assert all(t.is_eos for t in translations)


async def test_enable_filler_filter_query(palabra, monkeypatch):
    async with FakeAsrServer() as srv:
        _fake_region(monkeypatch, srv)
        async with palabra.stt(language="ja", enable_filler_filter=False, sample_rate=48000) as stt:
            await stt.send_audio(b"\x00\x00" * 100)
            async for _ in stt:
                pass
        assert srv.query["enable_filler_filter"] == ["false"]
        assert srv.query["sample_rate"] == ["48000"]


async def test_send_pcm_paces_realtime(palabra, monkeypatch):
    async with FakeAsrServer(close_after=2) as srv:
        _fake_region(monkeypatch, srv)
        async with palabra.stt(language="en") as stt:
            # 640 ms at 16 kHz mono -> 2 chunks of 320 ms
            pcm = b"\x00\x00" * int(16000 * 0.64)
            loop = asyncio.get_event_loop()
            t0 = loop.time()
            await stt.send_pcm(pcm)
            assert loop.time() - t0 >= 0.3  # paced (first chunk immediate, then 1 x 320 ms)
            async for _ in stt:
                pass
        assert srv.audio_frames == 2


async def test_stt_not_available_in_region():
    # the us region has no STT endpoint yet
    palabra = Palabra(api_key="k", region="us")
    with pytest.raises(ValueError, match="not available in region 'us'"):
        palabra.stt("en")


async def test_receive_loop_failure_surfaces_as_session_error(palabra, monkeypatch):
    """A crash in the receive loop must raise SessionError on iteration."""
    import palabra_ai.stt as stt_mod
    from palabra_ai import SessionError

    def boom(_msg):
        raise RuntimeError("parse exploded")

    async with FakeAsrServer() as srv:
        _fake_region(monkeypatch, srv)
        orig = stt_mod._parse_stt_event
        stt_mod._parse_stt_event = boom
        try:
            async with palabra.stt(language="en") as stt:
                await stt.send_audio(b"\x00\x00" * 100)
                with pytest.raises(SessionError) as ei:
                    async for _ in stt:
                        pass
                assert isinstance(ei.value.__cause__, RuntimeError)
        finally:
            stt_mod._parse_stt_event = orig
