"""Palabra AI - simple client for real-time speech-to-speech translation.

from palabra_ai import Palabra

# reads $PALABRA_API_KEY (create one at https://platform.palabra.ai/api-keys)
# and $PALABRA_REGION (default "eu")
async with Palabra().translation(source="en", targets=["es"]) as session:
    await session.send_audio(chunk)   # PCM s16le 24kHz mono, ~320 ms
    async for event in session:       # Transcript / Audio / ...
        ...
"""

from .audio import CHUNK_MS, OUTPUT_SAMPLE_RATE, load_pcm, read_wav, write_wav
from .client import REGIONS, Palabra, Region, TranslationSession
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
from .stt import SttSession, SttTranscript
from .task import build_task
from .tts import TtsChunk, TtsSession

__version__ = "2.0.0"

__all__ = [
    "CHUNK_MS",
    "OUTPUT_SAMPLE_RATE",
    "REGIONS",
    "Audio",
    "AuthError",
    # events
    "Event",
    "NotReadyError",
    "Palabra",
    # errors
    "PalabraError",
    "Raw",
    "Region",
    "ServerError",
    "ServerWarning",
    "SessionError",
    "StreamEnd",
    "SttSession",
    "SttTranscript",
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
