from palabra_ai import (PalabraAI, Config, SourceLang, TargetLang,
                        FileReader, FileWriter, EN, ES)

if __name__ == "__main__":
    palabra = PalabraAI()
    reader = FileReader("./speech/es.mp3")
    writer = FileWriter("./es2en_out.wav")
    cfg = Config(SourceLang(ES, reader), [TargetLang(EN, writer)], debug=True)
    palabra.run(cfg)
