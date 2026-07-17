import asyncio

from palabra_ai import Palabra, write_wav


async def main():
    palabra = Palabra()  # set your credentials here or vie ENV

    async with palabra.tts(language="en", voice_id="default_low") as tts:
        # one-shot: send text, collect all chunks
        pcm = await tts.synthesize("Curious minds think alike.")
        write_wav("tts_out.wav", pcm)

        # streaming, e.g. from an LLM: chunks arrive as they are synthesized
        await tts.send_text("The sun was setting over the mountains,")
        await tts.send_text(" casting long golden shadows across the valley below.", eos=True)
        async for chunk in tts:
            ...  # chunk.audio -> your playback
            if chunk.last_chunk:
                break


asyncio.run(main())
