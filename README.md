# Palabra AI Python Client

Simple Python client for [Palabra AI](https://palabra.ai) **real-time** streaming APIs: speech-to-speech translation, speech-to-text, and low-latency text-to-speech.

```bash
uv add palabra-ai          # or: pip install palabra-ai
```

Full API documentation: [docs.palabra.ai](https://docs.palabra.ai).

Palabra has three separate streaming APIs, and the client mirrors that at the top level:

| Product                                                                            | Entry point                | What it is                                                                           |
|------------------------------------------------------------------------------------|----------------------------|--------------------------------------------------------------------------------------|
| [**Realtime Speech-to-Speech Translation API**](#speech-to-speech-translation-api) | `palabra.translation(...)` | full pipeline translation API                                                        |
| [**Realtime Speech-to-Text API**](#speech-to-text-api)                             | `palabra.stt(...)`         | transcription API: stream audio in, incremental text (and optional translations) out |
| [**Realtime TTS API**](#realtime-tts-api)                                          | `palabra.tts(...)`         | speech synthesis API: stream text in (e.g. from an LLM), audio out                   |

Authentication, regions, [errors](#errors) and [reconnection](#reconnection) are shared by all three.

On top of the streaming APIs there are two management APIs (plain REST, same API Key): [**Voices**](#voices-custom-voices) (`palabra.voices`) for cloning custom voices, and [**Glossaries**](#glossaries) (`palabra.glossaries`) for custom terminology in the S2S Translation API.

## Authentication

Everything is driven by two values — an **API Key** and a **region**; all endpoints are derived from them automatically. Create your API Key on the [Palabra API Keys page](https://platform.palabra.ai/api-keys) and set it via the environment or the constructor:

```bash
export PALABRA_API_KEY=...
export PALABRA_REGION=eu   # optional, defaults to "eu"
```

```python
from palabra_ai import Palabra

palabra = Palabra()                            # reads the env vars
palabra = Palabra(api_key="...", region="eu")  # or explicit
```

The API Key authorizes the WebSocket connection directly: a streaming session is created server-side when you connect and cleaned up when the connection ends — there is nothing to manage.

## Regions

Availability per region (more regions and endpoints are being added over time):

| Region | Speech-to-Speech Translation | Speech-to-Text | TTS |
|--------|------------------------------|----------------|-----|
| `eu`   | ✓                            | ✓              | ✓   |
| `us`   | —                            | —              | ✓   |

Opening a stream for a product that is not available in the configured region raises `ValueError` with the list of regions where it is.

---

# Speech-to-Speech Translation API

You continuously push audio chunks; Palabra streams back transcripts, translations, and synthesized speech.

## Quick start

Audio comes from *your* source — a microphone, a VoIP call leg, a telephony bridge — anything that hands you PCM chunks. Push them into the session and consume events:

```python
import asyncio
import time
from palabra_ai import Palabra, Transcript, Audio

CHUNK_MS = 320  # ~320 ms of PCM (s16le, 24 kHz, mono)

async def main():
    palabra = Palabra()

    async with palabra.translation(source="en", targets=["es"]) as session:

        async def send_audio():
            next_send = time.monotonic()
            while chunk := await audio_buffer.get():  # your audio source
                await session.send_audio(chunk)
                next_send += CHUNK_MS / 1000
                await asyncio.sleep(max(0, next_send - time.monotonic()))
            await session.end()

        sender = asyncio.create_task(send_audio())

        async for event in session:
            match event:
                case Transcript():
                    print(event)
                case Audio():
                    play(event.pcm)

        await sender

asyncio.run(main())
```

`async with palabra.translation(...)` does everything for you: connects the WebSocket (your API Key authorizes it; a streaming session is created server-side automatically), sends the translation task, waits until the pipeline actually confirms it, and cleans up on exit.

Two rules for the input stream:

1. Chunks must match the format declared in the task (default: PCM s16le, 24 kHz, mono; ~320 ms per chunk is optimal).
2. Push at **real-time rate** — faster/slower pacing triggers `ServerWarning` (`AUDIO_STREAM_TOO_FAST/TOO_SLOW/STALLED`) and degrades quality. If your source is a live device or call, pacing comes for free.

## WebRTC (browser / client-side apps)

This client uses WebSocket transport, which is the recommended option for **server-side** applications. For browser and mobile apps Palabra recommends the **WebRTC transport with a JavaScript client**: follow the [WebRTC Quick Start](https://docs.palabra.ai/docs/quick-start/webrtc), or start from the official [TypeScript example](https://github.com/PalabraAI/typescript-speech-to-speech-translation-example). WebRTC handles microphone capture, pacing, jitter, etc. in the browser natively.

## Events

Iterating the session yields typed events:

| Event           | Fields                                       | Meaning                                                 |
|-----------------|----------------------------------------------|---------------------------------------------------------|
| `Transcript`    | `text, language, id, is_eos, is_translation` | partial/validated transcription & translation           |
| `Audio`         | `pcm, language, last_chunk, id`              | TTS chunk (PCM s16le 24 kHz mono)                       |
| `TaskInfo`      | `status, task`                               | response to `get_task`                                  |
| `TaskStatus`    | `command, event, status, transcription_id`    | command acknowledgement and resulting pipeline state   |
| `StreamEnd`     | —                                            | end-of-stream confirmation after `end()`, right before close |
| `ServerError`   | `code, desc`                                 | server-side error                                       |
| `ServerWarning` | `code, message`                              | `AUDIO_STREAM_TOO_FAST / TOO_SLOW / STALLED`            |
| `Raw`           | `type, data`                                 | anything else                                           |

## Session control

Session entry waits for `TaskStatus(command="set_task", event="started", status="running")`
(or `updated` / `resumed` when attaching to an existing task). Older servers without
acknowledgements remain supported through a delayed `get_task` fallback.

Control methods send commands; their confirmations arrive through the event iterator or
`receive()`. `set_task` reports `started`, `updated`, or `resumed`; `pause_task` reports
`paused`; `interrupt_task` reports `interrupted`; `tts_task` reports `queued`; and
`end_task` reports `ending`. Failed commands produce `ServerError` instead of a success
acknowledgement. These control replies do not depend on `message_types`.

```python
from palabra_ai import TaskStatus

async for event in session:
    if isinstance(event, TaskStatus):
        print(event.command, event.event, event.status)
    # Handle Transcript, Audio, ServerError, etc. here too.
```

`queued` means the text was accepted; its `transcription_id` matches subsequent
`Audio.id` / `Transcript.id` values. `ending` means shutdown has begun: continue reading
audio and transcripts until `StreamEnd`. Neither event confirms completed output.
Older servers emit no `TaskStatus` events.

```python
await session.send_audio(chunk)                      # one raw chunk (pace it yourself)

await session.speak("Hola!", "es")                   # speak text into the stream (note: you must have this language as one of the target languages)
await session.speak("Hi all!", "en", translate=True) # translate to all targets first

await session.interrupt()                            # instantly stop the speech of the current phrase (interruption)
await session.pause();  await session.resume()       # pause and resume your session (stops billing)
await session.set_task(new_task)                     # change settings on the fly
await session.end()                                  # graceful finish: processes the tail, emits StreamEnd
```

`session.speak(text, lang)` (the `tts_task` command) speaks **through the translation pipeline** and is unrelated to the standalone [Realtime TTS API](#realtime-tts-api).

## Settings

Common options are keyword arguments of `translation(...)`; anything beyond that — build the task dict yourself:

```python
from palabra_ai import build_task, Palabra

# common options inline
session = Palabra().translation(
    source="auto",
    targets=["es", "fr"],
    translate_partials=True,
    silence_threshold=0.8
)

# or full control, including per-target overrides and any server option
task = build_task(
    "en",
    {"es": {"speech_generation": {"voice_id": "default_high"}}, "fr": {}},
    input_sample_rate=48000,
)
task["pipeline"]["transcription"]["silence_threshold"] = 0.75

async with Palabra().translation(task=task) as session:
    ...
```

The client does **not validate settings** — invalid options are rejected by the server (`TaskError` is raised on `async with`, with the server's reason). The full list of options, their constraints, and tuning advice live in the docs: see [Recommended Settings](https://docs.palabra.ai/docs/streaming_api/recommended_settings).

## Offline files (utility)

For testing and batch jobs there are file helpers — but keep in mind this is a **real-time** service: the input is paced to real time, so translating a file takes roughly as long as the audio itself. For UX experiments and pipeline debugging it's convenient; for bulk offline processing it's the wrong tool.

```python
palabra.translate_file(
    "speech_en.wav",
    source="en",
    targets="es",
    output="speech_es.wav",
    on_transcript=print
)
# mp3/ogg/resampling need: uv add "palabra-ai[audio]"
```

Related helpers: `session.send_file(path)`, `session.send_pcm(pcm)` (chunking + real-time pacing built in), `load_pcm` / `read_wav` / `write_wav`.

---

# Speech-to-Text API

Standalone transcription API (with optional translation).
You push raw audio frames in and receive incremental transcriptions back;
set `translate_languages` to also get a translation of each finalized segment.

All settings are sent as query parameters, audio goes out as raw binary frames.

## Quick start

```python
import asyncio
from palabra_ai import Palabra, SttTranscript

async def main():
    palabra = Palabra()

    async with palabra.stt(language="en") as stt:

        async def feed():
            # any audio source: PCM s16le, 16 kHz, mono, ~320 ms per chunk, real-time paced
            while chunk := await audio_buffer.get():
                await stt.send_audio(chunk)

        feeder = asyncio.create_task(feed())

        async for event in stt:
            if isinstance(event, SttTranscript):
                print(event) # "~ [en] partial" / "[en] final"

        await feeder

asyncio.run(main())
```

Iteration ends when the server closes the connection (or raises `SessionError` if the receive loop crashed).

## Events

Iterating the session yields `SttTranscript` events (anything unrecognized comes through as `Raw`):

| Field                     | Meaning                                                                                  |
|---------------------------|------------------------------------------------------------------------------------------|
| `text`                    | `segment.text` — the full segment text so far                                            |
| `language`                | source language of the segment; the **target** language on a translation                 |
| `transcription_id`        | stable per segment; shared by all messages (incl. the translation) of one segment        |
| `is_eos`                  | `False` while the segment is still being updated; `True` once committed/final            |
| `is_translation`          | `True` for `translated_transcription` (emitted once per target after each final segment) |
| `start_time` / `end_time` | segment timing in seconds from session start                                             |
| `delta`                   | text appended since the previous transcript                                              |

With the filler filter enabled the segment tail may be rewritten midsegment,
so render `text` whole on each message rather than appending `delta`.

## Settings

All settings are keyword arguments of `stt(...)` and become URL query parameters:

```python
async with Palabra().stt(
    language="en",                       # source; defaults to auto-detect
    format="pcm_s16le",                  # see the audio-formats table in the docs
    sample_rate=16000,                   # required for raw PCM other than 16 kHz pcm_s16le
    translate_languages=["es", "de"],    # also emit translated_transcription per target
    enable_filler_filter=True,           # server default: True for every language but ja
) as stt:
    ...
```

---

# Realtime TTS API

Standalone synthesis, no translation pipeline. Designed for incremental text (LLM token streams): send pieces as they come, audio chunks come back with minimal latency.

Two methods. `send_text()` -- incremental streaming, e.g. straight from an LLM token stream; mark the end of each sentence with `eos=True` and consume `TtsChunk` events as they arrive:

```python
async with palabra.tts(language="en", voice_id="default_low") as tts:
    await tts.send_text("The sun was setting over the mountains,")
    await tts.send_text(" casting long golden shadows.", eos=True)

    async for chunk in tts: # TtsChunk: audio, generation_id, last_chunk, audio_len
        play(chunk.audio)
        if chunk.last_chunk:
            break

    await tts.cancel() # stop current synthesis, session stays open
```

Each `send_text()` message is limited to 256 characters (the server limit); longer text raises `ValueError` -- splitting is up to you.

`synthesize()` -- one sentence in, audio bytes out:

```python
async with palabra.tts(language="en", voice_id="default_low") as tts:
    pcm = await tts.synthesize("Curious minds think alike.")   # bytes (pcm s16le by default)
```

## Options & limits

All `palabra.tts(...)` options (languages, voices, `speed`, output formats, sample rates), rate limits, and constraints are described in the [Realtime TTS API docs](https://docs.palabra.ai/docs/streaming_api/realtime_tts). Per-message voice overrides can be passed as keyword arguments of `send_text()`/`synthesize()`.

TTS is currently available in the `eu` and `us` [regions](#regions).

---

# Voices (custom voices)

`palabra.voices` clones a voice from a short audio sample and manages your voices. It's a management REST API (`https://api.palabra.ai`), authorized with the same API Key via the `Authorization: Bearer` header — nothing extra to configure. All methods are async.

Recommended sample: a clean **WAV recording of 5–20 seconds**, one speaker, no background noise (mp3/flac/m4a/webm/mp4 are accepted too, max 10 MB).

```python
palabra = Palabra()

voice = await palabra.voices.create("voice.wav", name="My voice", language="en")
print(voice.id, voice.status)   # ready
```

`create()` does the whole flow: submits the metadata, uploads the file to the returned pre-signed URL, and polls until processing finishes (`wait=False` returns immediately; check later with `get()` / `wait_ready()`). Options: `description=`, `denoise=True` for noisy samples, `labels={"gender": ..., "age_group": ..., "mood": ...}`.

Management:

```python
await palabra.voices.list()                    # your voices
await palabra.voices.builtin()                 # built-in voices (default_low, default_high, ...)
await palabra.voices.get(voice_id)
await palabra.voices.update(voice_id, name="Renamed")
await palabra.voices.delete(voice_id)
```

## Using a voice

A ready `voice.id` works everywhere a `voice_id` is accepted:

**Realtime TTS API** — pass it directly:

```python
async with palabra.tts(language="en", voice_id=voice.id) as tts:
    pcm = await tts.synthesize("This is my cloned voice.")
```

**S2S Translation API** — the voice ID lives in each target's `speech_generation.voice_id`:

```python
# per-session default for every target
session = palabra.translation(source="en", targets=["es"], voice_id=voice.id)

# or per target
session = palabra.translation(
    source="en",
    targets={"es": {"speech_generation": {"voice_id": voice.id}}, "fr": {}},
)
```

In a raw task dict the same setting is `pipeline.translations[N].speech_generation.voice_id`.

See [`examples/voice_cloning_tts.py`](examples/voice_cloning_tts.py).

---

# Glossaries

`palabra.glossaries` manages custom terminology for the **S2S Translation API** (the other streaming APIs don't use glossaries yet). Same management REST API and auth as voices; all methods are async.

Three kinds:

| Kind           | What it does                                                                        | Entries                      |
|----------------|-------------------------------------------------------------------------------------|------------------------------|
| `hotwords`     | tricky words/terms the ASR should pay attention to, so they're recognized correctly | list of terms                |
| `verification` | after recognition, replace one text with another (same language)                    | `{recognized: replacement}`  |
| `translation`  | pin the translation of a term for one source → target language pair                 | `{source_term: target_term}` |

```python
palabra = Palabra()

# translation: en -> es term pairs
glossary = await palabra.glossaries.create(
    {"neural network": "red neuronal", "pipeline": "canalización"},
    name="ML terms", kind="translation", source_language="en", target_language="es",
)

# hotwords: terms the ASR must get right
await palabra.glossaries.create(["Palabra", "WebRTC", "Kubernetes"],
                                name="Tech terms", kind="hotwords", source_language="en")

# verification: post-ASR text replacement
await palabra.glossaries.create({"palabra a i": "Palabra AI"},
                                name="ASR fixes", kind="verification", source_language="en")
```

`create()` uploads the entries in one call (a path to a ready CSV file is accepted instead of a dict/list). Management: `list()`, `get(id)`, `download(id)` (the CSV), `update(id, name=..., enabled=...)`, `delete(id)`.

## Using glossaries in a translation session

An **enabled** glossary applies to every matching session automatically — no settings needed. To control this per session, each kind has a pair of task settings: `allow_*_glossaries` and `allowed_*_glossary_ids`. When the kind is allowed and no explicit ids are set (the default), **ALL** your enabled glossaries of that kind matching the session's language apply — the source language for `hotwords`/`verification`, the source → target language pair for `translation`. Setting `allowed_*_glossary_ids` restricts that to an exact set.

The `translation(...)`/`build_task(...)` shortcuts cover all three kinds — `True` = all matching glossaries of the kind, `False` = none, a list of ids/`Glossary` objects = only those:

```python
session = palabra.translation(
    source="en", targets=["es"],
    hotwords_glossaries=["<glossary_id>"],       # pipeline.transcription.*
    verification_glossaries=True,                # ALL your verification glossaries for "en"
    translation_glossaries=[glossary],           # pipeline.translations[N].*  (every target)
)
```

In a raw task dict the same settings live at:

- hotwords — `pipeline.transcription.allow_hotwords_glossaries` / `allowed_hotwords_glossary_ids`
- verification — `pipeline.transcription.verification.allow_verification_glossaries` / `allowed_verification_glossary_ids`
- translation — `pipeline.translations[N].allow_translation_glossaries` / `allowed_translation_glossary_ids` (per target)

See [`examples/glossary_translation.py`](examples/glossary_translation.py).

---

# Common reference

## Errors

Shared by all APIs:

- `AuthError` — missing/invalid API Key.
- `SessionError` — WebSocket connection problems (including a crashed receive loop — the original exception is attached as `__cause__`).
- `NotReadyError` — the pipeline didn't confirm `set_task` in time (translation only).
- `ApiError` — a management REST request (voices/glossaries) failed; `status` carries the HTTP code when there is one.
- `TaskError` — the server rejected `set_task` (raised immediately on `async with`, with the server's `code`/`desc`), or raised by `session.raise_on_error(event)` for server `error` messages; by default in-stream errors are delivered as `ServerError` events so a long-running stream survives recoverable errors.
- `ValueError` — the requested product is not available in the configured [region](#regions), or an unknown region was set.

## Reconnection

There is **no automatic WebSocket reconnect**, by design: a session is tied to one connection and to server-side pipeline state, so a transparent resume would silently lose the audio in flight and the transcription context. When the connection drops, iteration simply ends (or `SessionError` is raised if the receive loop crashed).

If your application needs resilience, build the retry loop on top — you control what state to restore:

```python
while True:
    try:
        async with palabra.translation(source="en", targets=["es"]) as session:
            ...  # feed audio, consume events
        break  # finished normally
    except (SessionError, NotReadyError):
        await asyncio.sleep(1)  # reconnect with your own backoff policy
```

## Examples

| File                                                                   | What it shows                                                                                |
|------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [`examples/realtime_tts.py`](examples/realtime_tts.py)                 | Realtime TTS API streaming-in generation example                                             |
| [`examples/realtime_stt.py`](examples/realtime_stt.py)                 | Realtime Speech-to-Text API example live microphone example (`uv add "palabra-ai[devices]"`) |
| [`examples/sts_buffer_streaming.py`](examples/sts_buffer_streaming.py) | Realtime Speech-to-Speech API feeding chunks + async event loop                              |
| [`examples/sts_mic_to_speakers.py`](examples/sts_mic_to_speakers.py)   | Realtime Speech-to-Speech API  live microphone translation (`uv add "palabra-ai[devices]"`)  |
| [`examples/sts_multi_language.py`](examples/sts_multi_language.py)     | Realtime Speech-to-Speech API  several targets, per-target voices                            |
| [`examples/sts_file_to_file.py`](examples/sts_file_to_file.py)         | Realtime Speech-to-Speech API  offline file translation (see the caveat above)               |
| [`examples/voice_cloning_tts.py`](examples/voice_cloning_tts.py)       | Voices API: clone a voice from voice.wav and speak with it via the Realtime TTS API          |
| [`examples/glossary_translation.py`](examples/glossary_translation.py) | Glossaries API: create a translation glossary and use it in a Speech-to-Speech session       |

## Development

```bash
uv sync --dev   # editable install + pytest/ruff
make check      # ruff check + tests + format check
```

## Migrating from 1.x

| 1.x | 2.0 |
|---|---|
| `Palabra(client_id=..., client_secret=...)` / `PALABRA_CLIENT_ID`, `PALABRA_CLIENT_SECRET` | `Palabra(api_key=..., region=...)` / `PALABRA_API_KEY`, `PALABRA_REGION` — create the key at [platform.palabra.dev/api-keys](https://platform.palabra.ai/api-keys) |
| REST session management: `create_session()` / `delete_session()` / `session=` / `Session` | removed — the API Key authorizes the WebSocket directly, sessions are managed server-side |
| `Palabra(api_url=...)` | removed — endpoints are derived from `region` (see [Regions](#regions)) |
| `ws_url=` + `token=` direct mode | removed |

## Migrating from 0.x

| 0.x (<= 0.6.x) | 1.0 |
|---|---|
| `PalabraAI()` + `Config(SourceLang(EN, reader), [TargetLang(ES, writer)])` + `palabra.run(cfg)` | `Palabra().translation(source="en", targets="es")` + `send_audio` / events |
| `FileReader` / `FileWriter` / `BufferReader` / adapters | plain `bytes`: feed any source via `send_audio`; file utilities for tests |
| `DeviceManager` | use `sounddevice` directly (see `mic_to_speakers.py`) |
| `on_transcription=` callbacks | `async for event in session` |
| WebRTC transport | not included; see the WebRTC note above |
