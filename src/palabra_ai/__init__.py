"""Palabra AI - simple client for real-time speech-to-speech translation.

from palabra_ai import Palabra

async with Palabra().translation(source="en", targets=["es"]) as session:
    await session.send_audio(chunk)   # PCM s16le 24kHz mono, ~320 ms
    async for event in session:       # Transcript / Audio / ...
        ...
"""

from .audio import CHUNK_MS, OUTPUT_SAMPLE_RATE, load_pcm, read_wav, write_wav
from .client import Palabra, Session, TranslationSession
from .events import (
    Audio,
    Event,
    Raw,
    ServerError,
    ServerWarning,
    StreamEnd,
    TaskInfo,
    Transcript,
)
from .exc import AuthError, NotReadyError, PalabraError, SessionError, TaskError
from .task import build_task
from .tts import TtsChunk, TtsSession

__version__ = "1.0.0"

__all__ = [
    "CHUNK_MS",
    "OUTPUT_SAMPLE_RATE",
    "Audio",
    "AuthError",
    # events
    "Event",
    "NotReadyError",
    "Palabra",
    # errors
    "PalabraError",
    "Raw",
    "ServerError",
    "ServerWarning",
    "Session",
    "SessionError",
    "StreamEnd",
    "TaskError",
    "TaskInfo",
    "Transcript",
    "TranslationSession",
    "TtsChunk",
    "TtsSession",
    "build_task",
    # audio helpers
    "load_pcm",
    "read_wav",
    "write_wav",
]
