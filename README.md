# Palabra AI Python Client

Simple Python client for [Palabra AI](https://palabra.ai) **real-time** streaming APIs: speech-to-speech translation and low-latency text-to-speech.

```bash
uv add palabra-ai          # or: pip install palabra-ai
```

Full API documentation: [docs.palabra.ai](https://docs.palabra.ai).

Palabra has two separate streaming products, and the client mirrors that at the top level:

| Product | Entry point | What it is |
|---|---|---|
| [**Speech-to-Speech Translation API**](#speech-to-speech-translation-api) | `palabra.translation(...)` | full pipeline: ASR -> translation -> TTS |
| [**Realtime TTS API**](#realtime-tts-api) | `palabra.tts(...)` | synthesis only: stream text in (e.g. from an LLM), audio out |

Authentication, connection options, [errors](#errors) and [reconnection](#reconnection) are shared between the two.

## Authentication

Credentials come from the constructor or from the environment:

```bash
export PALABRA_CLIENT_ID=...
export PALABRA_CLIENT_SECRET=...
```

```python
from palabra_ai import Palabra

palabra = Palabra()                                      # reads the env vars
palabra = Palabra(client_id="...", client_secret="...")  # or explicit
```

Credentials are only used for the REST API (session creation/deletion). They are **not required** for the direct-connection mode below.

## Connection options

Both `translation()` and `tts()` accept the same three connection modes:

1. **Default** — a session is created via REST on `async with` and deleted on exit. Nothing to manage.
2. **`session=`** — manually create `Session` with `await palabra.create_session()`; its lifecycle is yours (the client won't delete it).
3. **`ws_url=` + `token=`** — debug option: connect directly with a direct `ws_url` and already issued `publisher` token. For TTS the endpoint URL is the `ws_tts_url` field of the session. Here is an example:

```python
palabra = Palabra()  # credentials not required in this mode
async with palabra.translation(
    source="en",
    targets=["es"],
    ws_url=ws_url,
    token=publisher_token
) as session:
    ...
```

---

# Speech-to-Speech Translation API

You continuously push audio chunks; Palabra streams back transcripts, translations, and synthesized speech.

## Quick start

Audio comes from *your* source — a microphone, a VoIP call leg, a telephony bridge — anything that hands you PCM chunks. Push them into the session and consume events:

```python
import asyncio
from palabra_ai import Palabra, Transcript, Audio

async def main():
    palabra = Palabra()

    async with palabra.translation(source="en", targets=["es"]) as session:

        async def feed():
            # any audio source
            # chunks: PCM s16le, 24 kHz, mono, ~320 ms each, real-time paced
            while chunk := await my_audio_buffer.get():
                await session.send_audio(chunk)
            await session.end(eos_timeout=4)  # let the tail finish, then the server closes

        feeder = asyncio.create_task(feed())

        async for event in session:
            match event:
                case Transcript():
                    print(event)              # "~ [lang] partial" / "[lang] final"
                case Audio():
                    play(event.pcm)           # s16le, 24 kHz, mono

        await feeder

asyncio.run(main())
```

`async with palabra.translation(...)` does everything for you: creates a session via REST, connects the WebSocket, sends translation task, waits until the pipeline actually confirms the task, and cleans up on exit.

Two rules for the input stream:

1. Chunks must match the format declared in the task (default: PCM s16le, 24 kHz, mono; ~320 ms per chunk is optimal).
2. Push at **real-time rate** — faster/slower pacing triggers `ServerWarning` (`AUDIO_STREAM_TOO_FAST/TOO_SLOW/STALLED`) and degrades quality. If your source is a live device or call, pacing comes for free.

## WebRTC (browser / client-side apps)

This client uses WebSocket transport, which is the recommended option for **server-side** applications. For browser and mobile apps Palabra recommends the **WebRTC transport with a JavaScript client**: follow the [WebRTC Quick Start](https://docs.palabra.ai/docs/quick-start/webrtc), or start from the official [TypeScript example](https://github.com/PalabraAI/typescript-speech-to-speech-translation-example). WebRTC handles microphone capture, pacing, jitter, etc. in the browser natively.

## Events

Iterating the session yields typed events:

| Event | Fields | Meaning                                                 |
|---|---|---------------------------------------------------------|
| `Transcript` | `text, language, id, is_eos, is_translation` | partial/validated transcription & translation           |
| `Audio` | `pcm, language, last_chunk, id` | TTS chunk (PCM s16le 24 kHz mono)                       |
| `TaskInfo` | `status, task` | response to `get_task`                                  |
| `StreamEnd` | — | end-of-stream confirmation after `end(eos_timeout=...)` |
| `ServerError` | `code, desc` | server-side error                                       |
| `ServerWarning` | `code, message` | `AUDIO_STREAM_TOO_FAST / TOO_SLOW / STALLED`            |
| `Raw` | `type, data` | anything else                                           |

## Session control

```python
await session.send_audio(chunk)                      # one raw chunk (pace it yourself)

await session.speak("Hola!", "es")                   # speak text into the stream (note: you must have this language as one of the target languages)
await session.speak("Hi all!", "en", translate=True) # translate to all targets first

await session.flush()                                # drop the current transcription and audio (interruption)
await session.pause();  await session.resume()       # pause and resume your session (stops billing)
await session.set_task(new_task)                     # change settings on the fly
await session.end(eos_timeout=4)                     # graceful finish: waits for the tail, emits StreamEnd
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

# Realtime TTS API

Standalone synthesis, no translation pipeline. Designed for incremental text (LLM token streams): send pieces as they come, audio chunks come back with minimal latency.

Two methods. `send_text()` -- incremental streaming, e.g. straight from an LLM token stream; mark the end of each sentence with `eos=True` and consume `TtsChunk` events as they arrive:

```python
async with palabra.tts(language="en", voice_id="default_low") as tts:
    await tts.send_text("The sun was setting over the mountains,")
    await tts.send_text(" casting long golden shadows.", eos=True)

    async for chunk in tts:            # TtsChunk: audio, generation_id, last_chunk, audio_len
        play(chunk.audio)
        if chunk.last_chunk:
            break

    await tts.cancel()                 # stop current synthesis, session stays open
```

Each `send_text()` message is limited to 256 characters (the server limit); longer text raises `ValueError` -- splitting is up to you.

`synthesize()` -- one sentence in, audio bytes out:

```python
async with palabra.tts(language="en", voice_id="default_low") as tts:
    pcm = await tts.synthesize("Curious minds think alike.")   # bytes (pcm s16le by default)
```

## Options & limits

All `palabra.tts(...)` options (languages, voices, `speed`, output formats, sample rates), rate limits and constraints are described in the [Realtime TTS API docs](https://docs.palabra.ai/docs/streaming_api/realtime_tts). Per-message voice overrides can be passed as keyword arguments of `send_text()`/`synthesize()`.

[Connection options](#connection-options) are the same as in `translation()`, including `ws_url=`/`token=` (the TTS endpoint is the `ws_tts_url` field of the session).

---

# Common reference

## Errors

Shared by both APIs:

- `AuthError` — missing/invalid credentials.
- `SessionError` — REST/WebSocket connection problems (including a crashed receive loop — the original exception is attached as `__cause__`).
- `NotReadyError` — the pipeline didn't confirm `set_task` in time (translation only).
- `TaskError` — the server rejected `set_task` (raised immediately on `async with`, with the server's `code`/`desc`), or raised by `session.raise_on_error(event)` for server `error` messages; by default in-stream errors are delivered as `ServerError` events so a long-running stream survives recoverable errors.

REST session creation retries transient failures (network errors, 5xx) a few times with backoff; 4xx fails immediately.

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

| File | What it shows |
|---|---|
| [`examples/streaming.py`](examples/streaming.py) | feeding chunks + async event loop |
| [`examples/mic_to_speakers.py`](examples/mic_to_speakers.py) | live microphone translation (`uv add "palabra-ai[devices]"`) |
| [`examples/realtime_tts.py`](examples/realtime_tts.py) | standalone Realtime TTS API |
| [`examples/multi_language.py`](examples/multi_language.py) | several targets, per-target voices |
| [`examples/file_to_file.py`](examples/file_to_file.py) | offline file translation (see the caveat above) |

## Development

```bash
uv sync --dev   # editable install + pytest/ruff
make check      # ruff check + tests + format check
```

## Migrating from 0.x

| 0.x (<= 0.6.x) | 1.0 |
|---|---|
| `PalabraAI()` + `Config(SourceLang(EN, reader), [TargetLang(ES, writer)])` + `palabra.run(cfg)` | `Palabra().translation(source="en", targets="es")` + `send_audio` / events |
| `FileReader` / `FileWriter` / `BufferReader` / adapters | plain `bytes`: feed any source via `send_audio`; file utilities for tests |
| `DeviceManager` | use `sounddevice` directly (see `mic_to_speakers.py`) |
| `on_transcription=` callbacks | `async for event in session` |
| WebRTC transport | not included; see the WebRTC note above |
