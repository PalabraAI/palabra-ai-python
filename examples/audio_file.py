from palabra_ai import (
    EN,
    ES,
    Config,
    FileReader,
    FileWriter,
    PalabraAI,
    SourceLang,
    TargetLang,
)

if __name__ == "__main__":
    palabra = PalabraAI()
    reader = FileReader("./speech/es.mp3")
    writer = FileWriter("./sep23_es2en_out.wav")
    cfg = Config(SourceLang(ES, reader), [TargetLang(EN, writer)], debug=True)
    palabra.run(cfg)
