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

from palabra_ai import Audio, Palabra, Region, StreamEnd, Transcript

pytestmark = pytest.mark.asyncio


class FakeServer:
    def __init__(self):
        self.received_audio_chunks = 0
        self.task = None
        self.started = False
        self.server = None
        self.port = None
        self.path = None  # request path of the last connection (incl. query)

    async def __aenter__(self):
        self.server = await websockets.serve(self.handler, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()

    async def handler(self, ws):
        self.path = ws.request.path
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
def palabra():
    return Palabra(api_key="test-key")


def _fake_region(monkeypatch, srv):
    """Point the 'eu' region's translation endpoint at the fake server,
    keeping the {random_hash} placeholder the real URL carries."""
    monkeypatch.setattr(
        "palabra_ai.client.REGIONS",
        {
            "eu": Region(
                translation=f"ws://127.0.0.1:{srv.port}"
                "/streaming-api/{random_hash}/v1/speech-to-speech/stream"
            )
        },
    )


async def test_full_stream_flow(palabra, monkeypatch):
    async with FakeServer() as srv:
        _fake_region(monkeypatch, srv)

        events = []
        async with palabra.translation("en", "es") as s:
            chunk = b"\x00\x00" * int(24000 * 0.32)
            await s.send_audio(chunk)
            await s.end(eos_timeout=2)
            async for ev in s:
                events.append(ev)

        assert srv.received_audio_chunks == 1
        # the API Key went out as the token query parameter, and the
        # {random_hash} placeholder was replaced with a real value
        assert "token=test-key" in srv.path
        assert "{random_hash}" not in srv.path
        hash_part = srv.path.split("/streaming-api/")[1].split("/")[0]
        assert len(hash_part) == 32  # uuid4().hex

        transcripts = [e for e in events if isinstance(e, Transcript)]
        audio = [e for e in events if isinstance(e, Audio)]
        assert any(t.is_eos and t.is_translation and t.text == "uno dos" for t in transcripts)
        assert any(not t.is_eos and t.text == "one two" for t in transcripts)
        assert audio and audio[0].pcm == b"\x01\x00" * 100 and audio[0].last_chunk
        assert any(isinstance(e, StreamEnd) for e in events)


async def test_missing_api_key_raises_auth_error(monkeypatch):
    monkeypatch.delenv("PALABRA_API_KEY", raising=False)
    from palabra_ai import AuthError

    palabra = Palabra()  # does not raise
    with pytest.raises(AuthError, match="PALABRA_API_KEY"):
        palabra.translation("en", "es")


async def test_api_key_and_region_from_env(monkeypatch):
    monkeypatch.setenv("PALABRA_API_KEY", "env-key")
    monkeypatch.setenv("PALABRA_REGION", "us")
    palabra = Palabra()
    assert palabra.api_key == "env-key"
    assert palabra.region == "us"


async def test_unknown_region_rejected():
    with pytest.raises(ValueError, match="available regions"):
        Palabra(api_key="k", region="mars")


async def test_product_not_available_in_region():
    # the us region has no translation endpoint yet
    palabra = Palabra(api_key="k", region="us")
    with pytest.raises(ValueError, match="not available in region 'us'"):
        palabra.translation("en", "es")


async def test_send_pcm_paces_realtime(palabra, monkeypatch):
    async with FakeServer() as srv:
        _fake_region(monkeypatch, srv)
        async with palabra.translation("en", "es") as s:
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


async def test_rejected_set_task_fails_fast_with_task_error(palabra, monkeypatch):
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
    monkeypatch.setattr(
        "palabra_ai.client.REGIONS", {"eu": Region(translation=f"ws://127.0.0.1:{port}")}
    )
    try:
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        with pytest.raises(TaskError) as ei:
            async with palabra.translation("en", "es"):
                pass
        assert ei.value.code == "VALIDATION_ERROR"
        assert loop.time() - t0 < 5  # no 30 s readiness timeout
    finally:
        server.close()
        await server.wait_closed()


async def test_receive_loop_failure_surfaces_as_session_error(palabra, monkeypatch):
    """A crash in the receive loop (e.g. corrupt zlib output) must raise
    SessionError on iteration, not end the stream silently."""
    import zlib

    from palabra_ai import SessionError

    async with FakeServer() as srv:
        _fake_region(monkeypatch, srv)
        # zlib output declared, but FakeServer sends plain pcm -> decompress fails
        async with palabra.translation("en", "es", output_format="zlib_pcm_s16le") as s:
            await s.send_audio(b"\x00\x00" * int(24000 * 0.32))
            with pytest.raises(SessionError) as ei:
                async for _ in s:
                    pass
            assert isinstance(ei.value.__cause__, zlib.error)
