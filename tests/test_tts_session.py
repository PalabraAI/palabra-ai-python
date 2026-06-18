"""Realtime TTS API client tests.

The session logic runs against a fake server reproducing the documented
protocol (flat `type` messages in, `message_type`+`data` envelopes out).
"""

import asyncio
import base64
import json

import pytest
import websockets

from palabra_ai import Palabra, Session, TtsChunk
from palabra_ai.tts import MAX_TEXT_LEN

pytestmark = pytest.mark.asyncio


class FakeTtsServer:
    def __init__(self):
        self.init = None
        self.texts = []
        self.cancelled = 0
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
            mtype = msg.pop("type")
            if mtype == "init":
                self.init = msg
            elif mtype == "text":
                self.texts.append(msg)
                if msg.get("is_eos"):
                    gen = msg.get("generation_id") or "srv-gen"
                    for last in (False, True):
                        await ws.send(
                            json.dumps(
                                {
                                    "message_type": "audio_chunk",
                                    "data": {
                                        "audio": "" if last else base64.b64encode(b"\x01\x00" * 50).decode(),
                                        "size": 0 if last else 100,
                                        "generation_id": gen,
                                        "last_chunk": last,
                                        "chunk_generation_delta": None if last else 42,
                                        "audio_len": None if last else 0.1,
                                    },
                                }
                            )
                        )
            elif mtype == "cancel":
                self.cancelled += 1


@pytest.fixture
def palabra():
    return Palabra(client_id="test", client_secret="test")


def tts_session(palabra, srv):
    session = Session(id="s1", publisher="tok", ws_url="", ws_tts_url=f"ws://127.0.0.1:{srv.port}")
    return palabra.tts("en", voice_id="default_low", speed=0.6, session=session)


async def test_init_and_synthesize(palabra):
    async with FakeTtsServer() as srv:
        async with tts_session(palabra, srv) as tts:
            pcm = await tts.synthesize("Curious minds think alike.", timeout=5)
        assert pcm == b"\x01\x00" * 50
        assert srv.init["voice_options"]["voice_id"] == "default_low"
        assert srv.init["voice_options"]["speed"] == 0.6
        # client always sends the intended default explicitly
        # (server-side model default is 0.0, which is wrong)
        assert srv.init["voice_options"]["deaccent_strength"] == 1.0
        assert srv.init["output"]["format"] == "pcm"  # client default differs from server's mp3
        assert len(srv.texts) == 1 and srv.texts[0]["is_eos"]


async def test_send_text_streaming_pieces(palabra):
    async with FakeTtsServer() as srv:
        async with tts_session(palabra, srv) as tts:
            await tts.send_text("The sun was setting,", generation_id="gen-0001")
            await tts.send_text(" casting long shadows.", eos=True, generation_id="gen-0001")
            async for ev in tts:
                if isinstance(ev, TtsChunk) and ev.last_chunk:
                    assert ev.generation_id == "gen-0001"
                    break
        assert [t["is_eos"] for t in srv.texts] == [False, True]


async def test_send_text_rejects_oversized_and_empty(palabra):
    async with FakeTtsServer() as srv:
        async with tts_session(palabra, srv) as tts:
            with pytest.raises(ValueError):
                await tts.send_text("a" * (MAX_TEXT_LEN + 1))
            with pytest.raises(ValueError):
                await tts.send_text("")
            with pytest.raises(ValueError):
                await tts.synthesize("b" * 300, timeout=1)
        assert srv.texts == []  # nothing reached the server


async def test_cancel(palabra):
    async with FakeTtsServer() as srv:
        async with tts_session(palabra, srv) as tts:
            await tts.send_text("abc")
            await tts.cancel()
            await asyncio.sleep(0.1)
        assert srv.cancelled == 1


async def test_tts_factory_validation(palabra):
    with pytest.raises(ValueError):
        palabra.tts("en", ws_url="ws://x")  # token missing
    with pytest.raises(ValueError):
        palabra.tts("en", ws_url="ws://x", token="t", session=Session(id="s", publisher="p", ws_url=""))
