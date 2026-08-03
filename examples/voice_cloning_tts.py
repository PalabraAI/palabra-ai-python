import asyncio

from palabra_ai import Palabra, write_wav


async def main():
    palabra = Palabra()  # set PALABRA_API_KEY via ENV, or pass api_key=

    # one call: submits metadata, uploads voice.wav, waits until processing ends
    voice = await palabra.voices.create(
        "voice.wav",
        name="My voice",
        language="en",
    )
    print(f"voice {voice.id} is {voice.status}")

    # use the cloned voice in the meantime TTS API
    async with palabra.tts(language="en", voice_id=voice.id) as tts:
        pcm = await tts.synthesize("This is my cloned voice speaking.")
        write_wav("cloned_voice_out.wav", pcm)

    # your voices are listed (and deleted) via the same API
    for v in await palabra.voices.list():
        print(v.id, v.name, v.status)


if __name__ == '__main__':
    asyncio.run(main())
