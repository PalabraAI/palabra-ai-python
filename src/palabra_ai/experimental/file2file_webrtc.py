#!/usr/bin/env python3

# File-to-file WebRTC translation - reads from WAV file, outputs to WAV file
# Uses LiveKit's DataChannel for all control messages

# We recommend managing dependencies using https://astral.sh/uv/
# Create a virtual environment and activate it, then sync dependencies:
# $> `uv venv && . .venv/bin/activate && uv sync`
# Alternatively, install dependencies directly:
# $> `uv pip install httpx livekit numpy`

import asyncio
import json
import os
import signal
import time
import wave
from datetime import datetime

import httpx
import numpy as np
from livekit import rtc


### Step 1: Create a Session ###
async def create_session(client_id: str, client_secret: str) -> dict:
    url = "https://api.palabra.ai/session-storage/session"
    headers = {"ClientId": client_id, "ClientSecret": client_secret}
    payload = {"data": {"subscriber_count": 0, "publisher_can_subscribe": True}}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


### Step 2: Connect to the Translation Room ###
async def connect_translation_room(webrtc_url: str, publisher: str) -> rtc.Room:
    room = rtc.Room()
    await room.connect(webrtc_url, publisher, rtc.RoomOptions(auto_subscribe=True))
    print("💫 Connected to room")
    return room


### Step 3: Publish Audio Track from File ###
async def publish_audio_file(room: rtc.Room, input_file: str):
    sample_rate = 48000
    
    # Create audio source
    audio_source = rtc.AudioSource(sample_rate=sample_rate, num_channels=1)

    # Create and publish track
    track = rtc.LocalAudioTrack.create_audio_track("file_audio", audio_source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(dtx=False, red=False)
    )
    print("🎵 Audio track published")

    # Send file audio and return duration
    duration = await send_file_audio(audio_source, input_file)
    return duration


async def send_file_audio(audio_source: rtc.AudioSource, input_file: str):
    sample_rate = 48000
    frame_samples = 480  # 10ms at 48kHz
    frame = rtc.AudioFrame.create(sample_rate, 1, frame_samples)
    
    try:
        # Open input WAV file
        wav_reader = wave.open(input_file, 'rb')
        print(f"📖 Reading from: {input_file}")
        print(f"🎵 Format: {wav_reader.getnchannels()}ch, {wav_reader.getframerate()}Hz, {wav_reader.getnframes()} frames")
        
        # Calculate original duration
        original_duration = wav_reader.getnframes() / wav_reader.getframerate()
        print(f"⏱️ Original duration: {original_duration:.1f}s")
        
        # Read all audio data
        audio_data = wav_reader.readframes(wav_reader.getnframes())
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        original_rate = wav_reader.getframerate()
        wav_reader.close()
        
        # Resample if needed (simple decimation/interpolation)
        if original_rate != sample_rate:
            print(f"🔄 Resampling from {original_rate}Hz to {sample_rate}Hz")
            ratio = sample_rate / original_rate
            new_length = int(len(audio_array) * ratio)
            indices = np.linspace(0, len(audio_array) - 1, new_length)
            audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array).astype(np.int16)
        
    except Exception as e:
        print(f"❌ Error reading audio file: {e}")
        return None

    print("🎤 Sending file audio...")
    total_frames = len(audio_array) // frame_samples
    frame_duration = frame_samples / sample_rate  # 10ms
    batch_size = 100  # Send in batches for stability
    
    for i in range(0, len(audio_array), frame_samples):
        try:
            chunk = audio_array[i:i + frame_samples]
            
            # Pad last chunk if needed
            if len(chunk) < frame_samples:
                chunk = np.pad(chunk, (0, frame_samples - len(chunk)), mode='constant')
            
            # Copy to frame buffer
            np.copyto(np.frombuffer(frame.data, dtype=np.int16), chunk)
            await audio_source.capture_frame(frame)
            
            frame_num = i // frame_samples + 1
            print(f"\r🎵 Sent frame {frame_num}/{total_frames}", end="", flush=True)

            # CRITICAL: Must pace audio to real-time rate
            await asyncio.sleep(frame_duration)
            
            # Small pause every batch for stability
            if frame_num % batch_size == 0:
                await asyncio.sleep(0.01)
                
        except Exception as e:
            frame_num = i // frame_samples + 1
            print(f"\n❌ Error sending frame {frame_num}: {e}")
            # Continue trying to send remaining frames
            continue
    
    print(f"\n✅ Finished sending {total_frames} frames")
    return original_duration


### Step 4: Handle Translated Audio Track ###
class AudioFileReceiver:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.wav_writer = None
        self.total_frames = 0
        self.last_frame_time = None
        self.receiving_complete = False

    def start(self):
        self.wav_writer = wave.open(self.output_file, 'wb')
        self.wav_writer.setnchannels(1)
        self.wav_writer.setsampwidth(2)  # 16-bit
        self.wav_writer.setframerate(48000)
        print(f"📝 Output file: {self.output_file}")

    def write_audio(self, audio_data: bytes):
        if self.wav_writer:
            self.wav_writer.writeframes(audio_data)
            self.total_frames += len(audio_data) // 2  # 16-bit samples
            self.last_frame_time = time.time()
    
    def close(self):
        if self.wav_writer:
            self.wav_writer.close()
            duration = self.total_frames / 48000
            print(f"💾 Saved {duration:.1f}s audio to {self.output_file}")
            self.receiving_complete = True


audio_receiver = None


def on_track_subscribed(track, publication, participant):
    global audio_receiver
    if track.kind == rtc.TrackKind.KIND_AUDIO and "translation_" in publication.name:
        lang = publication.name.split("translation_")[-1]
        print(f"🔊 Receiving translation: {lang}")
        play_track(track, lang)


def play_track(track: rtc.Track, lang: str):
    global audio_receiver
    
    async def receive_audio():
        audio_stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1)
        frame_count = 0
        
        async for event in audio_stream:
            frame_data = np.frombuffer(event.frame.data, dtype=np.int16)
            audio_receiver.write_audio(frame_data.tobytes())
            
            frame_count += 1
            if frame_count % 100 == 0:  # Every 1 second at 10ms frames
                print(f"\r🔊 Receiving audio... {frame_count} frames", end="", flush=True)

    asyncio.create_task(receive_audio())


### Step 5: Start the Translation ###
async def start_translation(room: rtc.Room, translation_settings: dict):
    # Create the set_task message
    payload = {"message_type": "set_task", "data": translation_settings}

    # Send through data channel
    message_bytes = json.dumps(payload).encode("utf-8")
    await room.local_participant.publish_data(message_bytes, reliable=True)

    langs = [
        t["target_language"] for t in translation_settings["pipeline"]["translations"]
    ]
    print(f"⚙️ Settings sent: {langs}")


### SUPPORTING CODE ###
# Minimal translation settings
MINIMAL_SETTINGS = {
    "input_stream": {"content_type": "audio", "source": {"type": "webrtc"}},
    "output_stream": {"content_type": "audio", "target": {"type": "webrtc"}},
    "pipeline": {
        "transcription": {"source_language": "en"},
        "translations": [{"target_language": "es"}],
    },
}


# Wait for translator helper
async def wait_for_translator(room: rtc.Room, timeout: int = 10):
    start = time.time()
    while time.time() - start < timeout:
        for participant in room.remote_participants.values():
            if participant.identity.startswith("palabra_translator"):
                print(f"🌐 Translator joined: {participant.identity}")
                return participant
        await asyncio.sleep(0.1)
    raise TimeoutError("No translator joined")


# Data handler for transcriptions
def on_data_received(packet):
    try:
        data = json.loads(packet.data.decode())
        msg_type = data.get("message_type")
        if "transcription" in msg_type:
            text = data["data"]["transcription"]["text"]
            lang = data["data"]["transcription"]["language"]
            part = msg_type == "partial_transcription"
            print(
                f"\r\033[K{'💬' if part else '✅'} [{lang}] {text}",
                end="" if part else "\n",
                flush=True,
            )
    except:
        pass


### RUNNER ###
async def main():
    global audio_receiver
    
    signal.signal(signal.SIGINT, lambda s, f: os._exit(0))
    print("🚀 Palabra File2File WebRTC Client")

    # Input and output files
    input_file = "examples/speech/nbc_short.wav"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"nbc_short_out_webrtc_{timestamp}.wav"

    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return

    # Initialize audio receiver
    audio_receiver = AudioFileReceiver(output_file)
    audio_receiver.start()

    try:
        # Create session
        session = await create_session(
            os.getenv("PALABRA_CLIENT_ID"), os.getenv("PALABRA_CLIENT_SECRET")
        )
        webrtc_url = session["data"]["webrtc_url"]

        # Connect to room
        room = rtc.Room()
        room.on("track_subscribed", on_track_subscribed)
        room.on("data_received", on_data_received)
        await room.connect(
            webrtc_url, session["data"]["publisher"], rtc.RoomOptions(auto_subscribe=True)
        )

        # Wait for translator
        try:
            await wait_for_translator(room)
        except TimeoutError:
            print("⚠️ No translator, continuing...")

        # Send settings through DataChannel
        await start_translation(room, MINIMAL_SETTINGS)

        # Wait for settings to process
        await asyncio.sleep(3)

        # Publish audio file AFTER settings
        audio_duration = await publish_audio_file(room, input_file)
        if audio_duration is None:
            return
        
        # Calculate proper wait time based on audio duration
        # Need to wait for: sending time + translation processing time
        # Sending takes real-time duration, translation adds 1.5-2x 
        sending_time = audio_duration
        translation_time = audio_duration * 2
        max_wait_time = sending_time + translation_time + 30  # Add 30s buffer
        print(f"⏳ Waiting for translation to complete (max {max_wait_time:.0f}s)...")
        print(f"   Sending: {sending_time:.0f}s, Translation: {translation_time:.0f}s")
        
        # Wait for translation completion or timeout
        start_wait = time.time()
        while time.time() - start_wait < max_wait_time:
            elapsed = time.time() - start_wait
            if elapsed % 30 < 2:  # Print progress every 30 seconds
                print(f"   Progress: {elapsed:.0f}s / {max_wait_time:.0f}s")
            await asyncio.sleep(2)
        
        print("⏰ Translation complete")
        
        # Give a final grace period for any remaining frames
        print("⏳ Final grace period...")
        await asyncio.sleep(5)
        
        print("✅ Translation complete!")

    finally:
        audio_receiver.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutdown complete")
        os._exit(0)