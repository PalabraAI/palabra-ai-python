# <a href="https://palabra.ai"><img src="https://avatars.githubusercontent.com/u/199107821?s=32" alt="Palabra AI" align="center"></a> Palabra AI Python SDK

[![Tests](https://github.com/PalabraAI/palabra-ai-python/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/PalabraAI/palabra-ai-python/actions/workflows/test.yml)
[![Release](https://github.com/PalabraAI/palabra-ai-python/actions/workflows/release.yml/badge.svg)](https://github.com/PalabraAI/palabra-ai-python/actions/workflows/release.yml)
[![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/PalabraAI/palabra-ai-python)
[![PyPI version](https://img.shields.io/pypi/v/palabra-ai.svg?color=blue)](https://pypi.org/project/palabra-ai/)
[![Downloads](https://pepy.tech/badge/palabra-ai)](https://pepy.tech/projects/palabra-ai)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue?logo=docker)](https://github.com/PalabraAI/palabra-ai-python/pkgs/container/palabra-ai-python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[//]: # ([![codecov]&#40;https://codecov.io/gh/PalabraAI/palabra-ai-python/graph/badge.svg?token=HRQAJ5VFY7&#41;]&#40;https://codecov.io/gh/PalabraAI/palabra-ai-python&#41;)

🌍 **Python SDK for Palabra AI's real-time speech-to-speech translation API**
🚀 Break down language barriers and enable seamless communication across 25+ languages

## Overview 📋

🎯 **The Palabra AI Python SDK provides a high-level API for integrating real-time speech-to-speech translation into your Python applications.**

✨ **What can Palabra.ai do?**
- ⚡ Real-time speech-to-speech translation with near-zero latency
- 🎙️ Auto voice cloning - speak any language in YOUR voice
- 🔄 Two-way simultaneous translation for live discussions
- 🚀 Developer API/SDK for building your own apps
- 🎯 Works everywhere - Zoom, streams, events, any platform
- 🔒 Zero data storage - your conversations stay private

🔧 **This SDK focuses on making real-time translation simple and accessible:**
- 🛡️ Uses WebRTC and WebSockets under the hood
- ⚡ Abstracts away all complexity
- 🎮 Simple configuration with source/target languages
- 🎤 Supports multiple input/output adapters (microphones, speakers, files, buffers)

📊 **How it works:**
1. 🎤 Configure input/output adapters
2. 🔄 SDK handles the entire pipeline
3. 🎯 Automatic transcription, translation, and synthesis
4. 🔊 Real-time audio stream ready for playback

💡 **All with just a few lines of code!**

## Installation 📦

### From PyPI 📦
```bash
pip install palabra-ai
```

### macOS SSL Certificate Setup 🔒

If you encounter SSL certificate errors on macOS like:
```
SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate
```

**Option 1: Install Python certificates** (recommended)
```zsh
/Applications/Python\ $(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")/Install\ Certificates.command
```

**Option 2: Use system certificates**
```bash
pip install pip-system-certs
```

This will configure Python to use your system's certificate store.

## Quick Start 🚀

### Real-time microphone translation 🎤

```python
from palabra_ai import (PalabraAI, Config, SourceLang, TargetLang,
                        EN, ES, DeviceManager)

palabra = PalabraAI()
dm = DeviceManager()
mic, speaker = dm.select_devices_interactive()
cfg = Config(SourceLang(EN, mic), [TargetLang(ES, speaker)])
palabra.run(cfg)
```

⚙️ **Set your API credentials as environment variables:**
```bash
export PALABRA_CLIENT_ID=your_client_id
export PALABRA_CLIENT_SECRET=your_client_secret
```

## Examples 💡

### File-to-file translation 📁

```python
from palabra_ai import (PalabraAI, Config, SourceLang, TargetLang,
                        FileReader, FileWriter, EN, ES)

palabra = PalabraAI()
reader = FileReader("./speech/es.mp3")
writer = FileWriter("./es2en_out.wav")
cfg = Config(SourceLang(ES, reader), [TargetLang(EN, writer)])
palabra.run(cfg)
```

### Multiple target languages 🌐

```python
from palabra_ai import (PalabraAI, Config, SourceLang, TargetLang,
                        FileReader, FileWriter, EN, ES, FR, DE)

palabra = PalabraAI()
config = Config(
    source=SourceLang(EN, FileReader("presentation.mp3")),
    targets=[
        TargetLang(ES, FileWriter("spanish.wav")),
        TargetLang(FR, FileWriter("french.wav")),
        TargetLang(DE, FileWriter("german.wav"))
    ]
)
palabra.run(config)
```

### Customizable output 📝

📋 **Add a transcription of the source and translated speech.**
⚙️ **Configure output to provide:**
- 🔊 Audio only
- 📝 Transcriptions only
- 🎯 Both audio and transcriptions

```python
from palabra_ai import (
    PalabraAI,
    Config,
    SourceLang,
    TargetLang,
    FileReader,
    EN,
    ES,
)
from palabra_ai.base.message import TranscriptionMessage


async def print_translation_async(msg: TranscriptionMessage):
    print(repr(msg))


def print_translation(msg: TranscriptionMessage):
    print(str(msg))


palabra = PalabraAI()
cfg = Config(
    source=SourceLang(
        EN,
        FileReader("speech/en.mp3"),
        print_translation  # Callback for source transcriptions
    ),
    targets=[
        TargetLang(
            ES,
            # You can use only transcription without audio writer if you want
            # FileWriter("./test_output.wav"),  # Optional: audio output
            on_transcription=print_translation_async  # Callback for translated transcriptions
        )
    ],
    silent=True,  # Set to True to disable verbose logging to console
)
palabra.run(cfg)
```

#### Transcription output options: 📊

1️⃣ **Audio only** (default):
```python
TargetLang(ES, FileWriter("output.wav"))
```

2️⃣ **Transcription only**:
```python
TargetLang(ES, on_transcription=your_callback_function)
```

3️⃣ **Audio and transcription**:
```python
TargetLang(ES, FileWriter("output.wav"), on_transcription=your_callback_function)
```

💡 **The transcription callbacks receive `TranscriptionMessage` objects containing the transcribed text and metadata.**
🔄 **Callbacks can be either synchronous or asynchronous functions.**

### Integrate with FFmpeg (streaming) 🎬

```python
import io
from palabra_ai import (PalabraAI, Config, SourceLang, TargetLang,
                        BufferReader, BufferWriter, AR, EN, RunAsPipe)

ffmpeg_cmd = [
    'ffmpeg',
    '-i', 'speech/ar.mp3',
    '-f', 's16le',      # 16-bit PCM
    '-acodec', 'pcm_s16le',
    '-ar', '48000',     # 48kHz
    '-ac', '1',         # mono
    '-'                 # output to stdout
]

pipe_buffer = RunAsPipe(ffmpeg_cmd)
es_buffer = io.BytesIO()

palabra = PalabraAI()
reader = BufferReader(pipe_buffer)
writer = BufferWriter(es_buffer)
cfg = Config(SourceLang(AR, reader), [TargetLang(EN, writer)])
palabra.run(cfg)

print(f"Translated audio written to buffer with size: {es_buffer.getbuffer().nbytes} bytes")
with open("./ar2en_out.wav", "wb") as f:
    f.write(es_buffer.getbuffer())
```

### Using buffers 💾

```python
import io
from palabra_ai import (PalabraAI, Config, SourceLang, TargetLang,
                        BufferReader, BufferWriter, AR, EN)
from palabra_ai.internal.audio import convert_any_to_pcm16

en_buffer, es_buffer = io.BytesIO(), io.BytesIO()
with open("speech/ar.mp3", "rb") as f:
    en_buffer.write(convert_any_to_pcm16(f.read()))
palabra = PalabraAI()
reader = BufferReader(en_buffer)
writer = BufferWriter(es_buffer)
cfg = Config(SourceLang(AR, reader), [TargetLang(EN, writer)])
palabra.run(cfg)
print(f"Translated audio written to buffer with size: {es_buffer.getbuffer().nbytes} bytes")
with open("./ar2en_out.wav", "wb") as f:
    f.write(es_buffer.getbuffer())
```

### Using default audio devices 🔊

```python
from palabra_ai import PalabraAI, Config, SourceLang, TargetLang, DeviceManager, EN, ES

dm = DeviceManager()
reader, writer = dm.get_default_readers_writers()

if reader and writer:
    palabra = PalabraAI()
    config = Config(
        source=SourceLang(EN, reader),
        targets=[TargetLang(ES, writer)]
    )
    palabra.run(config)
```

### Async Translation ⚡

```python
import asyncio
from palabra_ai import PalabraAI, Config, SourceLang, TargetLang, FileReader, FileWriter, EN, ES

async def translate():
    palabra = PalabraAI()
    config = Config(
        source=SourceLang(EN, FileReader("input.mp3")),
        targets=[TargetLang(ES, FileWriter("output.wav"))]
    )
    result = await palabra.arun(config)
    # Result contains: result.ok, result.exc, result.log_data

if __name__ == "__main__":
    asyncio.run(translate())
```

### Synchronous Translation 🔄

```python
from palabra_ai import PalabraAI, Config, SourceLang, TargetLang, FileReader, FileWriter, EN, ES

# Synchronous execution (blocks until complete)
palabra = PalabraAI()
config = Config(
    source=SourceLang(EN, FileReader("input.mp3")),
    targets=[TargetLang(ES, FileWriter("output.wav"))]
)
result = palabra.run(config)
# Result contains: result.ok, result.exc, result.log_data
```

### Signal Handling 🛡️

```python
# Enable Ctrl+C signal handlers (disabled by default)
result = palabra.run(config, signal_handlers=True)

# Default behavior (signal handlers disabled)
result = palabra.run(config)  # signal_handlers=False by default
```

### Result Handling 📊

Both `run()` and `arun()` return a `RunResult` object with status information:

```python
result = palabra.run(config)
# or: result = await palabra.arun(config)

if result.ok:
    print("✅ Translation completed successfully!")
    if result.log_data:
        print(f"📊 Processing stats: {result.log_data}")
    if result.eos:
        print("🔚 End of stream signal received")
else:
    print(f"❌ Translation failed: {result.exc}")
```

## I/O Adapters & Mixing 🔌

### Available adapters 🛠️

🎯 **The Palabra AI SDK provides flexible I/O adapters that can combined to:**

- 📁 **FileReader/FileWriter**: Read from and write to audio files
- 🎤 **DeviceReader/DeviceWriter**: Use microphones and speakers
- 💾 **BufferReader/BufferWriter**: Work with in-memory buffers
- 🔧 **RunAsPipe**: Run command and represent as pipe (e.g., FFmpeg stdout)

### Mixing examples 🎨

🔄 **Combine any input adapter with any output adapter:**

#### 🎤➡️📁 Microphone to file - record translations
```python
config = Config(
    source=SourceLang(EN, mic),
    targets=[TargetLang(ES, FileWriter("recording_es.wav"))]
)
```

#### 📁➡️🔊 File to speaker - play translations
```python
config = Config(
    source=SourceLang(EN, FileReader("presentation.mp3")),
    targets=[TargetLang(ES, speaker)]
)
```

#### 🎤➡️🔊📁 Microphone to multiple outputs
```python
config = Config(
    source=SourceLang(EN, mic),
    targets=[
        TargetLang(ES, speaker),  # Play Spanish through speaker
        TargetLang(ES, FileWriter("spanish.wav")),  # Save Spanish to file
        TargetLang(FR, FileWriter("french.wav"))    # Save French to file
    ]
)
```

#### 💾➡️💾 Buffer to buffer - for integration
```python
input_buffer = io.BytesIO(audio_data)
output_buffer = io.BytesIO()

config = Config(
    source=SourceLang(EN, BufferReader(input_buffer)),
    targets=[TargetLang(ES, BufferWriter(output_buffer))]
)
```

#### 🔧➡️🔊 FFmpeg pipe to speaker
```python
pipe = RunAsPipe(ffmpeg_process.stdout)
config = Config(
    source=SourceLang(EN, BufferReader(pipe)),
    targets=[TargetLang(ES, speaker)]
)
```

## Features ✨

### Real-time translation ⚡
🎯 Translate audio streams in real-time with minimal latency
💬 Perfect for live conversations, conferences, and meetings

### Voice cloning 🗣️
🎭 Preserve the original speaker's voice characteristics in translations
⚙️ Enable voice cloning in the configuration

### Device management 🎮
🎤 Easy device selection with interactive prompts or programmatic access:

```python
dm = DeviceManager()

# Interactive selection
mic, speaker = dm.select_devices_interactive()

# Get devices by name
mic = dm.get_mic_by_name("Blue Yeti")
speaker = dm.get_speaker_by_name("MacBook Pro Speakers")

# List all devices
input_devices = dm.get_input_devices()
output_devices = dm.get_output_devices()
```

## Audio Configuration 🎵

### Sample Rates by Protocol

The SDK automatically handles audio sample rates based on the connection protocol:

#### WebSocket (WS) Mode
- **Input (to API)**: Always 16kHz mono PCM
- **Output (from API)**: Always 24kHz mono PCM

#### WebRTC Mode
- **Input (to API)**: 48kHz mono PCM
- **Output (from API)**: 48kHz mono PCM

The SDK automatically resamples audio to match these requirements regardless of your input/output device capabilities.

## Supported languages 🌍

### Speech recognition languages 🎤 (Source)
🇸🇦 Arabic (AR), 🌐 Bashkir (BA), 🇧🇾 Belarusian (BE), 🇧🇬 Bulgarian (BG), 🇧🇩 Bengali (BN), 🌐 Catalan (CA), 🇨🇿 Czech (CS), 🏴 Welsh (CY), 🇩🇰 Danish (DA), 🇩🇪 German (DE), 🇬🇷 Greek (EL), 🇬🇧 English (EN), 🌐 Esperanto (EO), 🇪🇸 Spanish (ES), 🇪🇪 Estonian (ET), 🌐 Basque (EU), 🇮🇷 Persian (FA), 🇫🇮 Finnish (FI), 🇫🇷 French (FR), 🇮🇪 Irish (GA), 🌐 Galician (GL), 🇮🇱 Hebrew (HE), 🇮🇳 Hindi (HI), 🇭🇷 Croatian (HR), 🇭🇺 Hungarian (HU), 🌐 Interlingua (IA), 🇮🇩 Indonesian (ID), 🇮🇹 Italian (IT), 🇯🇵 Japanese (JA), 🇰🇷 Korean (KO), 🇱🇹 Lithuanian (LT), 🇱🇻 Latvian (LV), 🇲🇳 Mongolian (MN), 🇮🇳 Marathi (MR), 🇲🇾 Malay (MS), 🇲🇹 Maltese (MT), 🇳🇱 Dutch (NL), 🇳🇴 Norwegian (NO), 🇵🇱 Polish (PL), 🇵🇹 Portuguese (PT), 🇷🇴 Romanian (RO), 🇷🇺 Russian (RU), 🇸🇰 Slovak (SK), 🇸🇮 Slovenian (SL), 🇸🇪 Swedish (SV), 🇰🇪 Swahili (SW), 🇮🇳 Tamil (TA), 🇹🇭 Thai (TH), 🇹🇷 Turkish (TR), 🌐 Uyghur (UG), 🇺🇦 Ukrainian (UK), 🇵🇰 Urdu (UR), 🇻🇳 Vietnamese (VI), 🇨🇳 Chinese (ZH)

### Translation languages 🔄 (Target)
🇸🇦 Arabic (AR), 🇦🇿 Azerbaijani (AZ), 🇧🇾 Belarusian (BE), 🇧🇬 Bulgarian (BG), 🇧🇦 Bosnian (BS), 🌐 Catalan (CA), 🇨🇿 Czech (CS), 🏴 Welsh (CY), 🇩🇰 Danish (DA), 🇩🇪 German (DE), 🇬🇷 Greek (EL), 🇬🇧 English (EN), 🇦🇺 English Australian (EN_AU), 🇨🇦 English Canadian (EN_CA), 🇬🇧 English UK (EN_GB), 🇺🇸 English US (EN_US), 🇪🇸 Spanish (ES), 🇲🇽 Spanish Mexican (ES_MX), 🇪🇪 Estonian (ET), 🇫🇮 Finnish (FI), 🇵🇭 Filipino (FIL), 🇫🇷 French (FR), 🇨🇦 French Canadian (FR_CA), 🌐 Galician (GL), 🇮🇱 Hebrew (HE), 🇮🇳 Hindi (HI), 🇭🇷 Croatian (HR), 🇭🇺 Hungarian (HU), 🇮🇩 Indonesian (ID), 🇮🇸 Icelandic (IS), 🇮🇹 Italian (IT), 🇯🇵 Japanese (JA), 🇰🇿 Kazakh (KK), 🇰🇷 Korean (KO), 🇱🇹 Lithuanian (LT), 🇱🇻 Latvian (LV), 🇲🇰 Macedonian (MK), 🇲🇾 Malay (MS), 🇳🇱 Dutch (NL), 🇳🇴 Norwegian (NO), 🇵🇱 Polish (PL), 🇵🇹 Portuguese (PT), 🇧🇷 Portuguese Brazilian (PT_BR), 🇷🇴 Romanian (RO), 🇷🇺 Russian (RU), 🇸🇰 Slovak (SK), 🇸🇮 Slovenian (SL), 🇷🇸 Serbian (SR), 🇸🇪 Swedish (SV), 🇰🇪 Swahili (SW), 🇮🇳 Tamil (TA), 🇹🇷 Turkish (TR), 🇺🇦 Ukrainian (UK), 🇵🇰 Urdu (UR), 🇻🇳 Vietnamese (VI), 🇨🇳 Chinese (ZH), 🇨🇳 Chinese Simplified (ZH_HANS), 🇹🇼 Chinese Traditional (ZH_HANT)

### Available language constants 📚

```python
from palabra_ai import (
    # English variants - 1.5+ billion speakers (including L2)
    EN, EN_AU, EN_CA, EN_GB, EN_US,

    # Chinese variants - 1.3+ billion speakers
    ZH, ZH_HANS, ZH_HANT,  # ZH_HANS and ZH_HANT for translation only

    # Hindi & Indian languages - 800+ million speakers
    HI, BN, MR, TA, UR,

    # Spanish variants - 500+ million speakers
    ES, ES_MX,

    # Arabic variants - 400+ million speakers
    AR, AR_AE, AR_SA,

    # French variants - 280+ million speakers
    FR, FR_CA,

    # Portuguese variants - 260+ million speakers
    PT, PT_BR,

    # Russian & Slavic languages - 350+ million speakers
    RU, UK, PL, CS, SK, BG, HR, SR, SL, MK, BE,

    # Japanese & Korean - 200+ million speakers combined
    JA, KO,

    # Southeast Asian languages - 400+ million speakers
    ID, VI, MS, FIL, TH,

    # Germanic languages - 150+ million speakers
    DE, NL, SV, NO, DA, IS,

    # Romance languages (other) - 100+ million speakers
    IT, RO, CA, GL,

    # Turkic & Central Asian languages - 200+ million speakers
    TR, AZ, KK, UG,

    # Baltic languages - 10+ million speakers
    LT, LV, ET,

    # Other European languages - 50+ million speakers
    EL, HU, FI, EU, CY, MT,

    # Middle Eastern languages - 50+ million speakers
    HE, FA,

    # African languages - 100+ million speakers
    SW,

    # Asian languages (other) - 50+ million speakers
    MN, BA,

    # Constructed languages
    EO, IA,

    # Other languages
    GA, BS
)
```

**Note**: Source languages (for speech recognition) and target languages (for translation) have different support. The SDK automatically validates language compatibility when creating `SourceLang` and `TargetLang` objects.

## Development status 🛠️

### Current status ✅
- ✅ Core SDK functionality
- ✅ GitHub Actions CI/CD
- ✅ Docker packaging
- ✅ Python 3.11, 3.12, 3.13 support
- ✅ PyPI publication
- ✅ Documentation site (coming soon)
- ⏳ Code coverage reporting (setup required)

### Current dev roadmap 🗺️
- ⏳ TODO: global timeout support for long-running tasks
- ⏳ TODO: support for multiple source languages in a single run
- ⏳ TODO: fine cancelling on cancel_all_tasks()
- ⏳ TODO: error handling improvements

### Build status 🏗️
- 🧪 **Tests**: Running on Python 3.11, 3.12, 3.13
- 📦 **Release**: Automated releases with Docker images
- 📊 **Coverage**: Tests implemented, reporting setup needed

## Requirements 📋

- 🐍 Python 3.11+
- 🔑 Palabra AI API credentials (get them at [palabra.ai](https://palabra.ai))

## Support 🤝

- 📚 Documentation: [https://docs.palabra.ai](https://docs.palabra.ai)
- 🐛 Issues: [GitHub Issues](https://github.com/PalabraAI/palabra-ai-python/issues)
- 📧 Email: info@palabra.ai

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

© Palabra.ai, 2025 | 🌍 Breaking down language barriers with AI 🚀
