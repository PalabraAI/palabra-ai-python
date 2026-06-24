import asyncio
import math
import struct

from palabra_ai import Audio, Palabra, Transcript

RATE = 24000
CHUNK_MS = 320
CHUNK_SAMPLES = RATE * CHUNK_MS // 1000

audio_buffer: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=50)


async def feed_source():
    # Stand-in for a real source: 5 seconds of a 440 Hz tone.
    # A real app pushes chunks here and None when the source ends.
    for i in range(5000 // CHUNK_MS):
        chunk = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * (i * CHUNK_SAMPLES + n) / RATE)))
            for n in range(CHUNK_SAMPLES)
        )
        await audio_buffer.put(chunk)
        await asyncio.sleep(CHUNK_MS / 1000)  # real-time pacing
    await audio_buffer.put(None)


async def main():
    palabra = Palabra() # set your credentials here or vie ENV

    async with palabra.translation(source="en", targets=["es"]) as session:

        async def pump():
            while (chunk := await audio_buffer.get()) is not None:
                await session.send_audio(chunk)
            await session.end(eos_timeout=4)  # deliver the tail, then close

        source = asyncio.create_task(feed_source())
        pumper = asyncio.create_task(pump())

        async for event in session:
            if isinstance(event, Transcript):
                print(event)  # "~ [lang] partial" / "[lang] final"
            elif isinstance(event, Audio):
                pass  # event.pcm -> your playback/output

        await asyncio.gather(source, pumper)


asyncio.run(main())
