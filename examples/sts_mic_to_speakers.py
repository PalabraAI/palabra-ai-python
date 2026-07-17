import asyncio
import queue

import numpy as np
import sounddevice as sd

from palabra_ai import Audio, Palabra, Transcript

RATE = 24000
CHUNK = int(RATE * 0.32)  # 320 ms


async def main():
    mic_q: queue.Queue[bytes] = queue.Queue(maxsize=100)
    spk_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)

    def on_mic(indata, frames, t, status):
        try:
            mic_q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    buffer = np.zeros(0, dtype=np.int16)

    def on_spk(outdata, frames, t, status):
        nonlocal buffer
        while len(buffer) < frames:
            try:
                buffer = np.concatenate([buffer, spk_q.get_nowait()])
            except queue.Empty:
                break
        if len(buffer) >= frames:
            outdata[:] = buffer[:frames].reshape(-1, 1)
            buffer = buffer[frames:]
        else:
            outdata.fill(0)

    palabra = Palabra()  # set your credentials here or vie ENV
    async with palabra.translation(source="en", targets=["es"]) as session:

        async def feed():
            pending = b""
            while True:
                try:
                    pending += mic_q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.005)
                    continue
                while len(pending) >= CHUNK * 2:
                    await session.send_audio(pending[: CHUNK * 2])
                    pending = pending[CHUNK * 2 :]

        with (
            sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", callback=on_mic),
            sd.OutputStream(samplerate=RATE, channels=1, dtype="int16", callback=on_spk),
        ):
            print("Speak! Ctrl+C to stop")
            feeder = asyncio.create_task(feed())
            try:
                async for event in session:
                    if isinstance(event, Transcript):
                        end = "\n" if event.is_eos else ""
                        print(f"\r\033[K{event}", end=end, flush=True)
                    elif isinstance(event, Audio):
                        try:
                            spk_q.put_nowait(np.frombuffer(event.pcm, dtype=np.int16))
                        except queue.Full:
                            pass
            finally:
                feeder.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDone")
