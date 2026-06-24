from __future__ import annotations

import base64
import zlib
from dataclasses import dataclass, field
from typing import Any

_TRANSCRIPT_TYPES = frozenset(
    {
        "transcription",
        "partial_transcription",
        "partial_translated_transcription",
        "validated_transcription",
        "translated_transcription",
    }
)


@dataclass(frozen=True)
class Event:
    type: str  # raw message_type


@dataclass(frozen=True)
class Transcript(Event):
    text: str
    language: str
    id: str = ""
    data: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_eos(self) -> bool:
        return self.type in ("validated_transcription", "translated_transcription")

    @property
    def is_translation(self) -> bool:
        return self.type in ("translated_transcription", "partial_translated_transcription")

    def __str__(self) -> str:
        prefix = "" if self.is_eos else "~ "
        return f"{prefix}[{self.language}] {self.text}"


@dataclass(frozen=True)
class Audio(Event):
    """A TTS chunk: PCM s16le, 24 kHz, mono (zlib output is decompressed here)."""

    pcm: bytes
    language: str
    last_chunk: bool
    id: str | None = None
    part_id: str | None = None


@dataclass(frozen=True)
class TaskInfo(Event):
    status: str  # running | paused | unknown
    task: dict[str, Any]


@dataclass(frozen=True)
class StreamEnd(Event):
    """Sent after end_task with eos_timeout, right before the server closes."""


@dataclass(frozen=True)
class ServerError(Event):
    code: str
    desc: str


@dataclass(frozen=True)
class ServerWarning(Event):
    """AUDIO_STREAM_TOO_FAST / AUDIO_STREAM_TOO_SLOW / AUDIO_STREAM_STALLED."""

    code: str
    message: str


@dataclass(frozen=True)
class Raw(Event):
    """Everything else: pipeline_timings, tts_buffer_stats, ..."""

    data: Any


def parse_event(message: dict[str, Any], *, zlib_output: bool = False) -> Event:
    mtype = message.get("message_type", "")
    data = message.get("data") or {}

    if mtype in _TRANSCRIPT_TYPES:
        tr = data.get("transcription") or {}
        return Transcript(
            type=mtype,
            text=tr.get("text", ""),
            language=tr.get("language", ""),
            id=tr.get("transcription_id", ""),
            data=tr,
        )

    if mtype == "output_audio_data":
        pcm = base64.b64decode(data.get("data", "") or "")
        if zlib_output and pcm:
            pcm = zlib.decompress(pcm)
        return Audio(
            type=mtype,
            pcm=pcm,
            language=data.get("language", ""),
            last_chunk=bool(data.get("last_chunk", False)),
            id=data.get("transcription_id"),
            part_id=data.get("translation_part_id"),
        )

    if mtype == "current_task":
        return TaskInfo(type=mtype, status=data.get("task_status", "unknown"), task=data)

    if mtype == "eos":
        return StreamEnd(type=mtype)

    if mtype == "error":
        return ServerError(type=mtype, code=data.get("code", "UNKNOWN_ERROR"), desc=data.get("desc", ""))

    if mtype == "warning":
        return ServerWarning(type=mtype, code=data.get("code", ""), message=data.get("message", ""))

    return Raw(type=mtype, data=data)
