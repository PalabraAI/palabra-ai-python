import wat

from palabra_ai import (PalabraAI, Config, SourceLang, TargetLang,
                        FileReader, FileWriter, EN, ES)
from palabra_ai.config import WebrtcMode
from palabra_ai.util.orjson import to_json


def on_transcription(msg):
    print(f"[{msg!r}]", flush=True)

if __name__ == "__main__":
    palabra = PalabraAI()
    reader = FileReader("./speech/es.mp3")
    writer = FileWriter("./es2en_out.wav")
    cfg = Config(
        SourceLang(ES, reader, on_transcription=on_transcription),
        [TargetLang(EN, writer, on_transcription=on_transcription)],
        # mode=WebrtcMode()
        silent=True,
        # log_file="es2en.log",
        benchmark=True,
    )
    result = palabra.run(cfg)
    wat / result
    print(f"Result: {result}")
    with open("lagmeter.json", "wb") as f:
        f.write(to_json(dict(messages=result.log_data.messages)))
    # breakpoint()
