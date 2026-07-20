import asyncio
import contextlib
import queue

import sounddevice as sd

from palabra_ai import Palabra, SttTranscript

RATE = 16000  # ASR recommended input rate
CHUNK = int(RATE * 0.32)  # 320 ms
CHUNK_BYTES = CHUNK * 2  # int16 mono = 2 bytes per sample


async def main():
    mic_q: queue.Queue[bytes] = queue.Queue(maxsize=100)

    def on_mic(indata, frames, t, status):
        try:
            mic_q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    palabra = Palabra()  # set PALABRA_API_KEY / PALABRA_REGION via ENV, or pass api_key=/region=

    async with palabra.stt(language="ru", translate_languages=["es", "en"]) as stt:

        async def feed():
            pending = b""
            while True:
                try:
                    pending += mic_q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.005)
                    continue

                while len(pending) >= CHUNK_BYTES:
                    await stt.send_audio(pending[:CHUNK_BYTES])
                    pending = pending[CHUNK_BYTES:]

        with sd.RawInputStream(
            samplerate=RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK,
            callback=on_mic,
        ):
            print("Speak! Ctrl+C to stop")
            feeder = asyncio.create_task(feed())

            try:
                async for event in stt:
                    if isinstance(event, SttTranscript):
                        end = "\n" if getattr(event, "is_eos", False) else ""
                        print(f"\r\033[K{event}", end=end, flush=True)
            finally:
                feeder.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await feeder


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDone")
