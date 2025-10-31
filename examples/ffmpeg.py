import io

from palabra_ai import (
    AR,
    EN,
    BufferReader,
    BufferWriter,
    Config,
    PalabraAI,
    RunAsPipe,
    SourceLang,
    TargetLang,
)

if __name__ == "__main__":
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        "speech/ar.mp3",
        "-f",
        "s16le",  # 16-bit PCM
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",  # 48kHz
        "-ac",
        "1",  # mono
        "-",  # output to stdout
    ]

    pipe_buffer = RunAsPipe(ffmpeg_cmd)
    es_buffer = io.BytesIO()

    palabra = PalabraAI()
    reader = BufferReader(pipe_buffer)
    writer = BufferWriter(es_buffer)

    async def p(*args):
        print(f"{args}")

    cfg = Config(
        SourceLang(AR, reader, on_transcription=print),
        [TargetLang(EN, writer, on_transcription=print)],
    )
    palabra.run(cfg)

    print(
        f"Translated audio written to buffer with size: {es_buffer.getbuffer().nbytes} bytes"
    )
    with open("./ar2en_out.wav", "wb") as f:
        f.write(es_buffer.getbuffer())
