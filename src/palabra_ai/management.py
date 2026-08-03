from __future__ import annotations

import asyncio
import csv
import io
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .exc import ApiError

API_URL = "https://api.palabra.ai"
TIMEOUT = 60.0
POLL_INTERVAL = 2.0  # seconds between voice processing-status polls

# public glossary kind -> API glossary_type value
GLOSSARY_TYPES = {"hotwords": "asr_hot", "verification": "asr", "translation": "translation"}
KIND_BY_API_TYPE = {v: k for k, v in GLOSSARY_TYPES.items()}

MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/x-m4a",
    ".webm": "video/webm",
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
}


@dataclass(frozen=True)
class Voice:
    """One custom (cloned) or built-in voice.

    Use `id` as speech_generation.voice_id (S2S Translation API) or as
    voice_id= (Realtime TTS API) once `status` is "ready".
    """

    id: str
    name: str
    status: str  # created | pending | ready | failed
    language: str | None = None
    description: str | None = None
    raw: dict[str, Any] | None = None  # full API payload

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Voice:
        return cls(
            id=str(data.get("voice_id", "")),
            name=data.get("name", ""),
            status=data.get("processing_status", ""),
            language=data.get("lang_code"),
            description=data.get("description"),
            raw=data,
        )


@dataclass(frozen=True)
class Glossary:
    """One glossary. `kind` is hotwords | verification | translation."""

    id: str
    name: str
    kind: str
    source_language: str
    target_language: str | None = None  # translation glossaries only
    enabled: bool = True
    raw: dict[str, Any] | None = None  # full API payload

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Glossary:
        kind = KIND_BY_API_TYPE.get(data.get("glossary_type", ""), data.get("glossary_type", ""))
        return cls(
            id=str(data.get("glossary_id", "")),
            name=data.get("name", ""),
            kind=kind,
            source_language=data.get("source_lang", ""),
            target_language=data.get("target_lang") if kind == "translation" else None,
            enabled=bool(data.get("is_enabled", True)),
            raw=data,
        )


def problems(payload: Any) -> str:
    """Joined detail/title strings of an RFC 9457 errors list."""
    errors = payload.get("errors") if isinstance(payload, dict) else None
    return "; ".join(e.get("detail") or e.get("title") or "" for e in errors or []).strip("; ")


def check(resp: httpx.Response) -> None:
    """Turn a non-2xx response into an ApiError with the server's details."""
    if resp.is_success:
        return
    try:
        detail = problems(resp.json())
    except ValueError:
        detail = resp.text[:300]
    url = str(resp.request.url).split("?")[0]
    raise ApiError(
        f"{resp.request.method} {url} failed: HTTP {resp.status_code}" + (f" - {detail}" if detail else ""),
        status=resp.status_code,
    )


def read_sample(sample: str | Path | tuple[str, bytes]) -> tuple[str, bytes]:
    if isinstance(sample, tuple):
        return sample
    path = Path(sample)
    return path.name, path.read_bytes()


def entries_to_csv(entries: Mapping[str, str] | Sequence[str] | str | Path, kind: str) -> bytes:
    if isinstance(entries, (str, Path)):
        return Path(entries).read_bytes()
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    if isinstance(entries, Mapping):
        if kind == "hotwords":
            raise ValueError("hotwords entries must be a list of terms, not a mapping")
        for source, target in entries.items():
            writer.writerow([source, target])
    else:
        if kind != "hotwords":
            raise ValueError(f"{kind} entries must be a {{source: target}} mapping, not a list")
        for term in entries:
            writer.writerow([term])
    if not buf.getvalue():
        raise ValueError("entries must not be empty")
    return buf.getvalue().encode()


class ManagementAPI:
    def __init__(self, api_key: str, api_url: str | None = None):
        self.api_key = api_key
        self.api_url = (api_url or os.getenv("PALABRA_API_URL") or API_URL).rstrip("/")

    async def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        files: Mapping[str, Any] | None = None,
        query: dict[str, str] | None = None,
        raw: bool = False,
    ) -> Any:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            try:
                resp = await client.request(
                    method,
                    self.api_url + path,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"data": payload} if payload is not None else None,
                    files=files,
                    params=query,
                )
            except httpx.HTTPError as e:
                raise ApiError(f"{method} {path} failed: {e}") from e
        check(resp)
        if raw:
            return resp.content
        data = resp.json()
        return data.get("data", data)  # requests and responses are wrapped in "data"

    async def pages(self, path: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        query = None
        while True:
            page = await self.request("GET", path, query=query)
            items += page.get("items") or []
            if not page.get("next"):
                return items
            query = {"token": page["next"]}


class VoicesAPI(ManagementAPI):
    """Custom voices for the S2S Translation and TTS APIs (Palabra.voices).

    A voice is cloned from one audio sample: create() submits the metadata,
    uploads the file, and (by default) waits until processing finishes.
    The resulting Voice.id is then usable as a voice_id in both APIs.
    """

    async def create(
        self,
        sample: str | Path | tuple[str, bytes],
        *,
        name: str,
        language: str,
        description: str | None = None,
        denoise: bool = False,
        labels: dict[str, Any] | None = None,
        wait: bool = True,
        timeout: float = 300.0,
    ) -> Voice:
        """Clone a voice from an audio sample; returns the new Voice.

        sample: path to the file, or a (filename, bytes) pair. A clean WAV
        recording of 5-20 seconds with a single speaker and no background
        noise works best (mp3/flac/m4a/webm/mp4 are accepted too).
        language: language spoken in the sample (e.g. "en").
        denoise: enable if the sample has background noise.
        labels: optional {"gender", "age_group", "mood"} metadata.
        wait: poll until processing finishes (raises ApiError if it fails);
        with wait=False the voice is returned in "created" status -- check
        it later via get().
        """
        filename, data = read_sample(sample)
        mime = MIME_TYPES.get(Path(filename).suffix.lower())
        if mime is None:
            raise ValueError(f"Unsupported sample type {filename!r}; accepted: {', '.join(MIME_TYPES)}")

        payload: dict[str, Any] = {
            "name": name,
            "samples": [{"filename": filename, "mime_type": mime, "lang_code": language, "denoise": denoise}],
        }
        if description is not None:
            payload["description"] = description
        if labels is not None:
            payload["labels"] = labels

        created = await self.request("POST", "/saas/voice/clone", payload=payload)

        # pre-signed S3 POST: the policy in form_data authorizes it, no API Key
        upload = created["samples"][0]
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            try:
                resp = await client.post(
                    upload["url"], data=upload["form_data"], files={"file": (filename, data, mime)}
                )
            except httpx.HTTPError as e:
                raise ApiError(f"Sample upload failed: {e}") from e
        check(resp)

        voice = Voice.from_api(created)
        if wait:
            voice = await self.wait_ready(voice.id, timeout=timeout)
        return voice

    async def wait_ready(self, voice_id: str, *, timeout: float = 300.0) -> Voice:
        """Poll a voice until processing finishes; returns the ready Voice.

        Raises ApiError if processing failed or the timeout expired.
        """
        deadline = time.monotonic() + timeout
        while True:
            voice = await self.get(voice_id)
            if voice.status == "ready":
                return voice
            if voice.status == "failed":
                detail = problems((voice.raw or {}).get("processing_result"))
                raise ApiError(f"Voice {voice_id} processing failed" + (f": {detail}" if detail else ""))
            if time.monotonic() >= deadline:
                raise ApiError(f"Voice {voice_id} not ready after {timeout:.0f}s (status: {voice.status})")
            await asyncio.sleep(POLL_INTERVAL)

    async def get(self, voice_id: str) -> Voice:
        return Voice.from_api(await self.request("GET", f"/saas/voice/m/{voice_id}"))

    async def list(self) -> list[Voice]:
        """All voices created by you (every page is fetched)."""
        return [Voice.from_api(v) for v in await self.pages("/saas/voice")]

    async def builtin(self) -> list[Voice]:
        """Built-in voices available to every account."""
        return [Voice.from_api(v) for v in await self.pages("/saas/voice/builtin")]

    async def update(
        self,
        voice_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        labels: dict[str, Any] | None = None,
    ) -> Voice:
        payload = {"name": name, "description": description, "labels": labels}
        payload = {k: v for k, v in payload.items() if v is not None}
        return Voice.from_api(await self.request("POST", f"/saas/voice/m/{voice_id}", payload=payload))

    async def delete(self, voice_id: str) -> None:
        await self.request("DELETE", f"/saas/voice/m/{voice_id}")


class GlossariesAPI(ManagementAPI):
    """Glossaries for the S2S Translation API (Palabra.glossaries).

    Three kinds:
    - "hotwords": tricky words/terms the ASR should recognize correctly;
      entries is a list of terms.
    - "verification": replace one recognized text with another after ASR;
      entries is a {recognized: replacement} mapping (same language).
    - "translation": fix the translation of a term for one source->target
      language pair; entries is a {source_term: target_term} mapping.

    Unless a session lists explicit glossary ids, ALL your enabled
    glossaries of a kind matching the session's language apply (the
    source language for hotwords/verification, the source->target pair
    for translation). See build_task's hotwords/verification/
    translation_glossaries options.
    """

    async def create(
        self,
        entries: Mapping[str, str] | Sequence[str] | str | Path,
        *,
        name: str,
        kind: str,
        source_language: str,
        target_language: str | None = None,
        enabled: bool = True,
    ) -> Glossary:
        """Create a glossary and upload its entries; returns the Glossary.

        entries: a mapping (translation/verification), a list of terms
        (hotwords), or a path to a ready CSV file (rows of "source, target",
        or one term per row for hotwords).
        kind: "hotwords" | "verification" | "translation".
        target_language: required for kind="translation" only.
        """
        kind = KIND_BY_API_TYPE.get(kind, kind)  # raw API value accepted too
        if kind not in GLOSSARY_TYPES:
            raise ValueError(f"Unknown glossary kind {kind!r}; expected one of: {', '.join(GLOSSARY_TYPES)}")
        if kind == "translation":
            if target_language is None:
                raise ValueError("target_language is required for translation glossaries")
        else:
            if target_language is not None and target_language != source_language:
                raise ValueError(f"target_language does not apply to {kind} glossaries")
            target_language = source_language  # the API requires them equal for these kinds

        csv_bytes = entries_to_csv(entries, kind)

        created = await self.request(
            "POST",
            "/saas/glossary",
            payload={
                "name": name,
                "is_enabled": enabled,
                "glossary_type": GLOSSARY_TYPES[kind],
                "source_lang": source_language,
                "target_lang": target_language,
            },
        )
        uploaded = await self.request(
            "POST",
            f"/saas/glossary/{created['glossary_id']}/upload",
            files={"glossary": (f"{name}.csv", csv_bytes, "text/csv")},
        )
        return Glossary.from_api(uploaded)

    async def get(self, glossary_id: str) -> Glossary:
        return Glossary.from_api(await self.request("GET", f"/saas/glossary/{glossary_id}"))

    async def list(self) -> list[Glossary]:
        """All your glossaries (every page is fetched)."""
        return [Glossary.from_api(g) for g in await self.pages("/saas/glossaries")]

    async def download(self, glossary_id: str) -> bytes:
        """The glossary's CSV file as uploaded."""
        return await self.request("GET", f"/saas/glossary/{glossary_id}/upload", raw=True)

    async def update(
        self, glossary_id: str, *, name: str | None = None, enabled: bool | None = None
    ) -> Glossary:
        """Rename and/or enable/disable a glossary (entries are immutable)."""
        if name is None or enabled is None:  # the API requires both fields
            current = await self.get(glossary_id)
            name = current.name if name is None else name
            enabled = current.enabled if enabled is None else enabled
        return Glossary.from_api(
            await self.request(
                "POST", f"/saas/glossary/{glossary_id}", payload={"name": name, "is_enabled": enabled}
            )
        )

    async def delete(self, glossary_id: str) -> None:
        await self.request("DELETE", f"/saas/glossary/{glossary_id}")
