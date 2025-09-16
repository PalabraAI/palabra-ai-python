from palabra_ai import EN, ES, Config, DeviceManager, PalabraAI, SourceLang, TargetLang

if __name__ == "__main__":
    palabra = PalabraAI()
    dm = DeviceManager()
    mic, speaker = dm.select_devices_interactive()
    cfg = Config(SourceLang(EN, mic), [TargetLang(ES, speaker)])
    palabra.run(cfg)
