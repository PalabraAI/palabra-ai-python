#!/usr/bin/env python3
"""Translate an audio file: EN speech -> ES speech + transcripts.

Note: input is paced to real time (server requirement), so this takes
about as long as the audio itself. Meant for tests and one-off jobs.
"""

from palabra_ai import Palabra

palabra = Palabra()  # PALABRA_CLIENT_ID / PALABRA_CLIENT_SECRET from env

palabra.translate_file(
    "speech_en.wav",
    source="en",
    targets="es",
    output="speech_es.wav",
    on_transcript=print,
)
