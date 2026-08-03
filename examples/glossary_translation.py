import asyncio

from palabra_ai import Audio, Palabra, Transcript


async def main():
    palabra = Palabra()  # set PALABRA_API_KEY via ENV, or pass api_key=

    # one call: creates the glossary and uploads its entries
    glossary = await palabra.glossaries.create(
        {"neural network": "red neuronal", "pipeline": "canalización"},
        name="ML terms",
        kind="translation",
        source_language="en",
        target_language="es",
    )
    print(f"glossary {glossary.id} created")

    # an enabled glossary applies to matching sessions automatically;
    # translation_glossaries=[...] pins the exact set for this session
    # (True = ALL your translation glossaries for the en->es pair, False = none)
    async with palabra.translation(
        source="en",
        targets=["es"],
        translation_glossaries=[glossary],
    ) as session:
        feeder = asyncio.create_task(session.send_file("speech_en.wav"))
        async for event in session:
            if isinstance(event, Transcript):
                print(event)
            elif isinstance(event, Audio):
                ...  # event.pcm -> your playback
        await feeder


if __name__ == '__main__':
    asyncio.run(main())
