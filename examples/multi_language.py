#!/usr/bin/env python3
"""One source -> several target languages, with a custom voice for Spanish."""

from palabra_ai import Palabra

palabra = Palabra()

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
