"""Voices & Glossaries management API tests.

The REST flows run against a fake HTTP server reproducing the documented
protocol: JSON requests/responses wrapped in "data", multipart uploads
(field name "glossary" for glossary CSV, S3-style pre-signed POST for
voice samples), token pagination and RFC 9457 error payloads.
"""

import email
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from palabra_ai import ApiError, Glossary, Palabra, Voice, build_task
from palabra_ai.management import GlossariesAPI, VoicesAPI

pytestmark = pytest.mark.asyncio

VOICE_ID = "10545719-5dfb-4164-9b39-cc70ed2ff97d"
GLOSSARY_ID = "62edf6b3-458a-4bed-ab6e-e0b257bb4471"


def parse_multipart(headers, body: bytes) -> dict[str, tuple[str | None, bytes]]:
    """{field_name: (filename, value_bytes)} of a multipart/form-data body."""
    msg = email.message_from_bytes(f"Content-Type: {headers['Content-Type']}\r\n\r\n".encode() + body)
    return {
        part.get_param("name", header="content-disposition"): (
            part.get_filename(),
            part.get_payload(decode=True) or b"",
        )
        for part in msg.get_payload()
    }


class FakeApiServer:
    """Records every request; responses are scripted per (method, path)."""

    def __init__(self):
        self.requests = []  # (method, path, query, headers, body)
        self.responses = {}  # (method, path) -> scripted response | list of them
        self.server = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _handle(self, method):
                path, _, query = self.path.partition("?")
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                headers = {k.title(): v for k, v in self.headers.items()}  # httpx sends lowercase
                outer.requests.append((method, path, query, headers, body))
                scripted = outer.responses.get((method, path))
                if isinstance(scripted, list):
                    scripted = scripted.pop(0)
                if scripted is None:
                    scripted = {"status": 404, "body": b'{"ok": false, "errors": []}'}
                self.send_response(scripted.get("status", 200))
                self.send_header("Content-Type", scripted.get("content_type", "application/json"))
                self.end_headers()
                self.wfile.write(scripted.get("body", b"{}"))

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def do_DELETE(self):
                self._handle("DELETE")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def respond(self, method, path, data=None, *, status=200, raw=None, many=None):
        if many is not None:
            self.responses[(method, path)] = [
                {"status": 200, "body": json.dumps({"data": d}).encode()} for d in many
            ]
        else:
            body = raw if raw is not None else json.dumps({"data": data}).encode()
            self.responses[(method, path)] = {"status": status, "body": body}


def glossary_payload(**over):
    payload = {
        "glossary_id": GLOSSARY_ID,
        "user_id": "u",
        "name": "ML terms",
        "is_enabled": True,
        "glossary_type": "translation",
        "source_lang": "en",
        "target_lang": "es",
        "utc_created_at": "2025-06-23T11:30:44",
    }
    payload.update(over)
    return payload


def voice_payload(**over):
    payload = {
        "voice_id": VOICE_ID,
        "user_id": "u",
        "name": "My voice",
        "voice_type": "instantly_cloned",
        "processing_status": "created",
        "lang_code": "en",
    }
    payload.update(over)
    return payload


# ---------------------------------------------------------------- glossaries


async def test_glossary_create_translation():
    with FakeApiServer() as srv:
        srv.respond("POST", "/saas/glossary", glossary_payload())
        srv.respond("POST", f"/saas/glossary/{GLOSSARY_ID}/upload", glossary_payload())

        api = GlossariesAPI("plbr_key", api_url=srv.url)
        glossary = await api.create(
            {"neural network": "red neuronal", "pipeline, the": "canalización"},
            name="ML terms",
            kind="translation",
            source_language="en",
            target_language="es",
        )

        assert glossary == Glossary(
            id=GLOSSARY_ID,
            name="ML terms",
            kind="translation",
            source_language="en",
            target_language="es",
            enabled=True,
            raw=glossary_payload(),
        )

        create, upload = srv.requests
        assert create[3]["Authorization"] == "Bearer plbr_key"
        assert json.loads(create[4]) == {
            "data": {
                "name": "ML terms",
                "is_enabled": True,
                "glossary_type": "translation",
                "source_lang": "en",
                "target_lang": "es",
            }
        }
        # CSV goes as a multipart field named "glossary", comma-quoted per csv rules
        fields = parse_multipart(upload[3], upload[4])
        filename, csv_bytes = fields["glossary"]
        assert filename == "ML terms.csv"
        assert csv_bytes == b'neural network,red neuronal\n"pipeline, the",canalizaci\xc3\xb3n\n'


@pytest.mark.parametrize(
    ("kind", "api_type", "entries"),
    [("hotwords", "asr_hot", ["Palabra", "WebRTC"]), ("verification", "asr", {"a b": "AB"})],
)
async def test_glossary_create_same_language_kinds(kind, api_type, entries):
    with FakeApiServer() as srv:
        srv.respond("POST", "/saas/glossary", glossary_payload(glossary_type=api_type, target_lang="en"))
        srv.respond(
            "POST",
            f"/saas/glossary/{GLOSSARY_ID}/upload",
            glossary_payload(glossary_type=api_type, target_lang="en"),
        )

        api = GlossariesAPI("k", api_url=srv.url)
        glossary = await api.create(entries, name="G", kind=kind, source_language="en")

        # target_lang must equal source_lang for these kinds; hidden on the model
        sent = json.loads(srv.requests[0][4])["data"]
        assert sent["glossary_type"] == api_type
        assert sent["target_lang"] == "en"
        assert glossary.kind == kind
        assert glossary.target_language is None


async def test_glossary_create_validates_input():
    api = GlossariesAPI("k", api_url="http://127.0.0.1:1")
    with pytest.raises(ValueError, match="kind"):
        await api.create(["x"], name="G", kind="wrong", source_language="en")
    with pytest.raises(ValueError, match="target_language is required"):
        await api.create({"a": "b"}, name="G", kind="translation", source_language="en")
    with pytest.raises(ValueError, match="mapping"):
        await api.create(["term"], name="G", kind="verification", source_language="en")
    with pytest.raises(ValueError, match="list of terms"):
        await api.create({"a": "b"}, name="G", kind="hotwords", source_language="en")
    with pytest.raises(ValueError, match="empty"):
        await api.create([], name="G", kind="hotwords", source_language="en")


async def test_glossary_list_paginated_and_update_fills_missing_fields():
    with FakeApiServer() as srv:
        srv.respond(
            "GET",
            "/saas/glossaries",
            many=[
                {"items": [glossary_payload()], "next": "tok1", "page_size": 1, "count": 1},
                {"items": [glossary_payload(glossary_id="g2")], "next": None, "page_size": 1, "count": 1},
            ],
        )
        srv.respond("GET", f"/saas/glossary/{GLOSSARY_ID}", glossary_payload())
        srv.respond("POST", f"/saas/glossary/{GLOSSARY_ID}", glossary_payload(is_enabled=False))

        api = GlossariesAPI("k", api_url=srv.url)
        glossaries = await api.list()
        assert [g.id for g in glossaries] == [GLOSSARY_ID, "g2"]
        assert "token=tok1" in srv.requests[1][2]

        updated = await api.update(GLOSSARY_ID, enabled=False)
        assert updated.enabled is False
        # name was not passed -> the client fetched it and sent both fields
        assert json.loads(srv.requests[-1][4]) == {"data": {"name": "ML terms", "is_enabled": False}}


async def test_glossary_api_error_carries_details():
    with FakeApiServer() as srv:
        srv.respond(
            "POST",
            "/saas/glossary",
            status=422,
            raw=json.dumps(
                {"ok": False, "errors": [{"status": 422, "detail": "Source and target must differ"}]}
            ).encode(),
        )
        api = GlossariesAPI("k", api_url=srv.url)
        with pytest.raises(ApiError, match="Source and target must differ") as e:
            await api.create(
                {"a": "b"}, name="G", kind="translation", source_language="en", target_language="es"
            )
        assert e.value.status == 422


# -------------------------------------------------------------------- voices


async def test_voice_create_uploads_sample_and_waits(tmp_path, monkeypatch):
    monkeypatch.setattr("palabra_ai.management.POLL_INTERVAL", 0)
    sample = tmp_path / "voice.wav"
    sample.write_bytes(b"RIFFfake-wav")

    with FakeApiServer() as srv:
        srv.respond(
            "POST",
            "/saas/voice/clone",
            voice_payload(
                samples=[{"item_id": "0", "url": f"{srv.url}/s3-upload", "form_data": {"key": "k1"}}]
            ),
        )
        srv.respond("POST", "/s3-upload", raw=b"", status=204)
        srv.respond(
            "GET",
            f"/saas/voice/m/{VOICE_ID}",
            many=[voice_payload(processing_status="pending"), voice_payload(processing_status="ready")],
        )

        api = VoicesAPI("plbr_key", api_url=srv.url)
        voice = await api.create(sample, name="My voice", language="en", description="mine")

        assert voice.id == VOICE_ID and voice.ready

        clone, upload = srv.requests[0], srv.requests[1]
        assert json.loads(clone[4]) == {
            "data": {
                "name": "My voice",
                "samples": [
                    {
                        "filename": "voice.wav",
                        "mime_type": "audio/wav",
                        "lang_code": "en",
                        "denoise": False,
                    }
                ],
                "description": "mine",
            }
        }
        # S3 pre-signed POST: form_data fields + the file, no Authorization header
        assert "Authorization" not in upload[3]
        fields = parse_multipart(upload[3], upload[4])
        assert fields["key"] == (None, b"k1")
        assert fields["file"] == ("voice.wav", b"RIFFfake-wav")


async def test_voice_create_fails_on_failed_processing(monkeypatch):
    monkeypatch.setattr("palabra_ai.management.POLL_INTERVAL", 0)
    with FakeApiServer() as srv:
        srv.respond(
            "POST",
            "/saas/voice/clone",
            voice_payload(samples=[{"url": f"{srv.url}/s3-upload", "form_data": {}}]),
        )
        srv.respond("POST", "/s3-upload", raw=b"", status=204)
        srv.respond(
            "GET",
            f"/saas/voice/m/{VOICE_ID}",
            voice_payload(
                processing_status="failed",
                processing_result={"errors": [{"detail": "sample too short"}]},
            ),
        )
        api = VoicesAPI("k", api_url=srv.url)
        with pytest.raises(ApiError, match="sample too short"):
            await api.create(("voice.wav", b"RIFF"), name="V", language="en")


async def test_voice_create_rejects_unknown_extension():
    api = VoicesAPI("k", api_url="http://127.0.0.1:1")
    with pytest.raises(ValueError, match="Unsupported sample type"):
        await api.create(("voice.txt", b"x"), name="V", language="en")


async def test_voice_list_builtin_delete():
    with FakeApiServer() as srv:
        srv.respond("GET", "/saas/voice", {"items": [voice_payload(processing_status="ready")], "next": None})
        srv.respond(
            "GET",
            "/saas/voice/builtin",
            {"items": [voice_payload(voice_id="default_low", name="Default Low")], "next": None},
        )
        srv.respond("DELETE", f"/saas/voice/m/{VOICE_ID}", {"voice_id": VOICE_ID})

        api = VoicesAPI("k", api_url=srv.url)
        voices = await api.list()
        assert [v.id for v in voices] == [VOICE_ID]
        assert isinstance(voices[0], Voice)
        assert (await api.builtin())[0].id == "default_low"
        await api.delete(VOICE_ID)
        assert srv.requests[-1][0] == "DELETE"


# ----------------------------------------------------- client & task wiring


async def test_client_properties():
    palabra = Palabra(api_key="plbr_key")
    assert isinstance(palabra.voices, VoicesAPI)
    assert isinstance(palabra.glossaries, GlossariesAPI)
    assert palabra.voices.api_key == "plbr_key"


async def test_build_task_glossary_options():
    glossary = Glossary(id="g1", name="G", kind="translation", source_language="en")
    task = build_task(
        "en",
        {"es": {}, "fr": {"allowed_translation_glossary_ids": ["override"]}},
        hotwords_glossaries=["h1", "h2"],
        verification_glossaries=True,
        translation_glossaries=[glossary],
    )

    transcription = task["pipeline"]["transcription"]
    assert transcription["allow_hotwords_glossaries"] is True
    assert transcription["allowed_hotwords_glossary_ids"] == ["h1", "h2"]
    assert transcription["verification"] == {"allow_verification_glossaries": True}

    es, fr = task["pipeline"]["translations"]
    assert es["allow_translation_glossaries"] is True
    assert es["allowed_translation_glossary_ids"] == ["g1"]  # Glossary object -> id
    assert fr["allowed_translation_glossary_ids"] == ["override"]  # per-target override wins


async def test_build_task_glossaries_disabled_and_default():
    task = build_task("en", "es", hotwords_glossaries=False)
    transcription = task["pipeline"]["transcription"]
    assert transcription["allow_hotwords_glossaries"] is False
    assert "allowed_hotwords_glossary_ids" not in transcription
    assert "verification" not in transcription

    default = build_task("en", "es")["pipeline"]["transcription"]
    assert "allow_hotwords_glossaries" not in default  # server defaults untouched
