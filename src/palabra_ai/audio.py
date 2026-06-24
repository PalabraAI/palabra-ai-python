from __future__ import annotations

import asyncio
import time
import wave
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

OUTPUT_SAMPLE_RATE = 24000  # fixed by the server
CHUNK_MS = 320  # recommended input chunk length


def read_wav(path: str | Path) -> tuple[bytes, int, int]:
    """Returns (pcm, sample_rate, channels). 16-bit PCM only."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM WAV, got {w.getsampwidth() * 8}-bit")
        return w.readframes(w.getnframes()), w.getframerate(), w.getnchannels()


def write_wav(path: str | Path, pcm: bytes, sample_rate: int = OUTPUT_SAMPLE_RATE, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def load_pcm(path: str | Path, sample_rate: int = OUTPUT_SAMPLE_RATE, channels: int = 1) -> bytes:
    """Load any audio file as PCM s16le at the given rate.

    WAV at the right rate/channels goes through the stdlib; everything else
    (mp3, ogg, resampling) needs the optional av package.
    """
    path = Path(path)
    if path.suffix.lower() == ".wav":
        pcm, sr, ch = read_wav(path)
        if sr == sample_rate and ch == channels:
            return pcm
    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError as e:
        raise ImportError(f"reading/resampling {path.name} requires the av package: uv add av") from e

    layout = "mono" if channels == 1 else "stereo"
    resampler = AudioResampler(format="s16", layout=layout, rate=sample_rate)
    out = bytearray()
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.audio[0]):
            for rframe in resampler.resample(frame):
                out += bytes(rframe.planes[0])[: rframe.samples * 2 * channels]
        for rframe in resampler.resample(None):  # flush
            out += bytes(rframe.planes[0])[: rframe.samples * 2 * channels]
    return bytes(out)


def chunks(
    pcm: bytes,
    *,
    chunk_ms: int = CHUNK_MS,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
    channels: int = 1,
) -> Iterator[bytes]:
    step = int(sample_rate * chunk_ms / 1000) * 2 * channels
    for i in range(0, len(pcm), step):
        yield pcm[i : i + step]


async def paced_chunks(
    pcm: bytes,
    *,
    chunk_ms: int = CHUNK_MS,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
    channels: int = 1,
) -> AsyncIterator[bytes]:
    """Same as chunks(), but paced to real time (the server requires it)."""
    started = time.monotonic()
    sent_ms = 0
    for chunk in chunks(pcm, chunk_ms=chunk_ms, sample_rate=sample_rate, channels=channels):
        yield chunk
        sent_ms += chunk_ms
        delay = started + sent_ms / 1000 - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
