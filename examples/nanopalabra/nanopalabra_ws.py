#!/usr/bin/env python3

# WebSocket-only version of nanopalabra - minimal Palabra client
# Audio chunks: 320ms @ 24000Hz, PCM 16-bit mono

# We recommend managing dependencies using https://astral.sh/uv/
# Create a virtual environment and activate it, then sync dependencies:
# $> `uv venv && . .venv/bin/activate && uv sync`
# Alternatively, install dependencies directly:
# $> `uv pip install httpx livekit numpy sounddevice websockets`


import asyncio
import base64
import json
import os
import queue
import signal
import threading
import time

import httpx
import numpy as np
import sounddevice as sd
import websockets


### Step 1: Create a Session ###
async def create_session(client_id: str, client_secret: str) -> dict:
    url = "https://api.palabra.ai/session-storage/session"
    headers = {"ClientId": client_id, "ClientSecret": client_secret}
    payload = {"data": {"subscriber_count": 0, "publisher_can_subscribe": True}}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


### Step 2: Connect to WebSocket ###
class SimpleWebSocket:
    def __init__(self, url: str, token: str):
        self.url = f"{url}?token={token}"
        self.ws = None
        self.audio_queue = queue.Queue(maxsize=100)

    async def connect(self):
        self.ws = await websockets.connect(self.url, ping_interval=10, ping_timeout=30)
        print("🔌 WebSocket connected")
        asyncio.create_task(self._receive_loop())

    async def _receive_loop(self):
        while self.ws:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=60)
                data = json.loads(msg)
                if isinstance(data.get("data"), str):
                    data["data"] = json.loads(data["data"])

                msg_type = data.get("message_type")
                if msg_type == "current_task":
                    print("📝 Task confirmed")
                elif msg_type == "output_audio_data":
                    # Handle TTS audio
                    transcription_data = data.get("data", {})
                    audio_b64 = transcription_data.get("data", "")

                    if audio_b64:
                        try:
                            audio_data = base64.b64decode(audio_b64)
                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                            try:
                                self.audio_queue.put_nowait(audio_array)
                            except queue.Full:
                                pass
                        except Exception as e:
                            print(f"Audio decode error: {e}")
                elif "transcription" in msg_type:
                    transcription = data.get("data", {}).get("transcription", {})
                    text = transcription.get("text", "")
                    lang = transcription.get("language", "")
                    if text:
                        part = msg_type == "partial_transcription"
                        print(
                            f"\r\033[K{'💬' if part else '✅'} [{lang}] {text}",
                            end="" if part else "\n",
                            flush=True,
                        )
            except websockets.exceptions.ConnectionClosed:
                print("📡 WebSocket connection closed")
                break
            except Exception as e:
                print(f"❌ WebSocket error: {e}")
                break

    async def send(self, message: dict):
        if self.ws:
            await self.ws.send(json.dumps(message))


### Step 3: Publish Audio ###
async def publish_audio(ws: SimpleWebSocket):
    sample_rate = 24000
    chunk_duration = 0.32  # 320ms
    chunk_samples = int(sample_rate * chunk_duration)

    audio_queue = queue.Queue(maxsize=100)
    stop_event = threading.Event()

    def input_callback(indata, frames, time_info, status):
        try:
            audio_queue.put_nowait(np.frombuffer(indata, dtype=np.int16).copy())
        except queue.Full:
            pass

    # Start microphone in background
    def recording_thread():
        with sd.RawInputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            callback=input_callback,
            blocksize=int(sample_rate * 0.02),
        ):
            print("🎤 Mic started. Please say something!..")
            while not stop_event.is_set():
                time.sleep(0.01)

    threading.Thread(target=recording_thread, daemon=True).start()

    # Send audio chunks
    buffer = np.array([], dtype=np.int16)
    while True:
        try:
            audio_data = audio_queue.get(timeout=0.1)
            buffer = np.concatenate([buffer, audio_data])

            while len(buffer) >= chunk_samples:
                chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]

                # Send via WebSocket
                message = {
                    "message_type": "input_audio_data",
                    "data": {"data": base64.b64encode(chunk.tobytes()).decode("utf-8")},
                }
                await ws.send(message)

                # CRITICAL: Must pace audio to real-time rate
                await asyncio.sleep(0.32)

        except queue.Empty:
            await asyncio.sleep(0.001)


### Step 4: Play Audio ###
def play_audio(ws: SimpleWebSocket):
    def run_playback():
        sample_rate = 24000
        buffer = np.array([], dtype=np.int16)

        def audio_callback(outdata, frames, time_info, status):
            nonlocal buffer
            # Fill buffer if needed
            while len(buffer) < frames:
                try:
                    buffer = np.concatenate([buffer, ws.audio_queue.get_nowait()])
                except queue.Empty:
                    break

            # Provide audio frames
            if len(buffer) >= frames:
                outdata[:] = buffer[:frames].reshape(-1, 1)
                buffer = buffer[frames:]
            else:
                outdata.fill(0)

        with sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            callback=audio_callback,
            blocksize=int(sample_rate * 0.02),
        ):
            print("🔊 Playing: es")
            while True:
                time.sleep(0.1)

    threading.Thread(target=run_playback, daemon=True).start()


### Step 5: Settings ###
MINIMAL_SETTINGS = {
    "input_stream": {
        "content_type": "audio",
        "source": {
            "type": "ws",
            "format": "pcm_s16le",
            "sample_rate": 24000,
            "channels": 1,
        },
    },
    "output_stream": {
        "content_type": "audio",
        "target": {
            "type": "ws",
            "format": "pcm_s16le",
            "sample_rate": 24000,
            "channels": 1,
        },
    },
    "pipeline": {
        "preprocessing": {},
        "transcription": {"source_language": "en"},
        "translations": [
            {
                "target_language": "es",
                "speech_generation": {},
            }
        ],
    },
}


### RUNNER ###
async def main():
    signal.signal(signal.SIGINT, lambda s, f: os._exit(0))
    print("🚀 Palabra WebSocket Minimal Client")

    # Create session
    session = await create_session(
        os.getenv("PALABRA_CLIENT_ID"), os.getenv("PALABRA_CLIENT_SECRET")
    )
    ws_url = session["data"]["ws_url"]
    publisher = session["data"]["publisher"]

    # Connect WebSocket
    ws = SimpleWebSocket(ws_url, publisher)
    await ws.connect()

    # Send settings and wait
    await ws.send({"message_type": "set_task", "data": MINIMAL_SETTINGS})
    langs = [t["target_language"] for t in MINIMAL_SETTINGS["pipeline"]["translations"]]
    print(f"⚙️ Settings sent: {langs}")

    # Wait for settings to process
    await asyncio.sleep(3)

    # Start playback
    play_audio(ws)

    # Publish microphone
    asyncio.create_task(publish_audio(ws))

    print("\n🎧 Listening... Press Ctrl+C to stop\n")

    try:
        await asyncio.Event().wait()
    except:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutdown complete")
        os._exit(0)
