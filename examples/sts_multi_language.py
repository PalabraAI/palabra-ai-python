from palabra_ai import Palabra

palabra = Palabra() # set your credentials here or vie ENV

results = palabra.translate_file(
    "presentation.mp3",  # mp3 needs: pip install palabra-ai[audio]
    source="en",
    targets={
        "es": {"speech_generation": {"voice_id": "default_high"}},
        "fr": {},
        "de": {},
    },
    output="presentation_{lang}.wav",
)
print({lang: f"{len(pcm) / 48000:.1f}s" for lang, pcm in results.items()})
