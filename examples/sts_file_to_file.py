from palabra_ai import Palabra

palabra = Palabra()  # set your credentials here or vie ENV

palabra.translate_file(
    "speech_en.wav",
    source="en",
    targets="es",
    output="speech_es.wav",
    on_transcript=print,
)
