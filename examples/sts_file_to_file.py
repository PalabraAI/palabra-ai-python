from palabra_ai import Palabra

palabra = Palabra()  # set PALABRA_API_KEY / PALABRA_REGION via ENV, or pass api_key=/region=

palabra.translate_file(
    "speech_en.wav",
    source="en",
    targets="es",
    output="speech_es.wav",
    on_transcript=print,
)
