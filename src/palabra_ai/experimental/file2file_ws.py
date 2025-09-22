#!/usr/bin/env python3

# File-to-file WebSocket translation - reads from WAV file, outputs to WAV file
# Audio chunks: 320ms @ 24000Hz, PCM 16-bit mono
#
# Usage: python file2file_ws.py <source_lang> <target_lang> <input_file> [--output-dir DIR]
# Example: python file2file_ws.py en es ./nbc.wav
#
# We recommend managing dependencies using https://astral.sh/uv/
# Create a virtual environment and activate it, then sync dependencies:
# $> `uv venv && . .venv/bin/activate && uv sync`
# Alternatively, install dependencies directly:
# $> `uv pip install httpx numpy websockets`

import argparse
import asyncio
import base64
import json
import os
import signal
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import av
import httpx
import numpy as np
import websockets
from loguru import logger


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
class FileWebSocket:
    def __init__(
        self, url: str, token: str, output_file: str, original_duration: float
    ):
        self.url = f"{url}?token={token}"
        self.ws = None
        self.output_file = output_file
        self.original_duration = original_duration
        self.sample_rate = 24000

        # Create empty audio array for entire duration + 60 seconds buffer
        total_samples = int((original_duration + 60) * self.sample_rate)
        self.audio_array = np.zeros(total_samples, dtype=np.int16)

        self.translation_complete = False
        self.receiving_audio = False
        self.global_start_time = None  # Set when first chunk is sent
        self.ws_connected = True
        self.ws_closed = False
        self.last_audio_end_position = (
            0  # Global tracking of last occupied audio position
        )
        self.chunk_stats = {}  # Track chunk counts for each transcription_id
        self.total_drift_seconds = 0.0  # Track accumulated drift
        self.max_drift_seconds = 0.0  # Track maximum drift
        self.transcription_drifts = {}  # Track drift per transcription
        self.total_chunks_received = 0  # Total audio chunks received
        self.transcription_count = 0  # Total transcriptions processed
        self.verbose_logging = True  # Always use full verbose logging

    async def connect(self):
        self.ws = await websockets.connect(
            self.url,
            ping_interval=30,
            ping_timeout=60,
            close_timeout=60,
            max_size=2**23,  # 8MB max message size
        )
        logger.info("🔌 WebSocket connected")
        logger.info(f"📝 Output file: {self.output_file}")

        asyncio.create_task(self._receive_loop())

    async def reconnect(self):
        """Reconnect WebSocket with retry logic"""
        if self.ws:
            try:
                await self.ws.close()
            except:
                pass

        for attempt in range(3):
            try:
                await asyncio.sleep(2**attempt)  # Exponential backoff
                self.ws = await websockets.connect(
                    self.url,
                    ping_interval=30,
                    ping_timeout=60,
                    close_timeout=60,
                    max_size=2**23,
                )
                logger.info(f"🔄 Reconnected (attempt {attempt + 1})")
                asyncio.create_task(self._receive_loop())
                return True
            except Exception as e:
                logger.error(f"❌ Reconnection attempt {attempt + 1} failed: {e}")

        return False

    async def _receive_loop(self):
        while self.ws:
            try:
                msg = await self.ws.recv()
                data = json.loads(msg)
                if isinstance(data.get("data"), str):
                    data["data"] = json.loads(data["data"])

                msg_type = data.get("message_type")
                if msg_type == "current_task":
                    logger.info("📝 Task confirmed")
                elif msg_type == "output_audio_data":
                    # Handle TTS audio
                    transcription_data = data.get("data", {})
                    audio_b64 = transcription_data.get("data", "")
                    transcription_id = transcription_data.get("transcription_id", "")
                    last_chunk = transcription_data.get("last_chunk", False)

                    if audio_b64:
                        try:
                            audio_data = base64.b64decode(audio_b64)
                            chunk_time = time.perf_counter()

                            if self.global_start_time is not None:
                                audio_samples = np.frombuffer(
                                    audio_data, dtype=np.int16
                                )
                                chunk_duration_s = len(audio_samples) / self.sample_rate

                                # Calculate ideal position based on arrival time
                                elapsed_time = chunk_time - self.global_start_time
                                ideal_position = int(elapsed_time * self.sample_rate)

                                # Find earliest available position without overlaps
                                actual_position = max(
                                    ideal_position, self.last_audio_end_position
                                )
                                end_position = actual_position + len(audio_samples)

                                # Calculate drift from ideal position
                                drift_samples = actual_position - ideal_position
                                drift_seconds = drift_samples / self.sample_rate

                                # Track drift statistics
                                if drift_seconds > 0:
                                    self.total_drift_seconds += drift_seconds
                                    self.max_drift_seconds = max(
                                        self.max_drift_seconds, drift_seconds
                                    )

                                    # Track per-transcription drift
                                    if (
                                        transcription_id
                                        not in self.transcription_drifts
                                    ):
                                        self.transcription_drifts[transcription_id] = []
                                    self.transcription_drifts[transcription_id].append(
                                        drift_seconds
                                    )

                                # Update chunk stats
                                if transcription_id not in self.chunk_stats:
                                    self.chunk_stats[transcription_id] = 0
                                self.chunk_stats[transcription_id] += 1
                                self.total_chunks_received += 1

                                if end_position <= len(self.audio_array):
                                    # Place chunk in audio array
                                    self.audio_array[actual_position:end_position] = (
                                        audio_samples
                                    )

                                    # Update global end position
                                    self.last_audio_end_position = end_position

                                    # Always log all chunk placement info
                                    actual_time = actual_position / self.sample_rate
                                    ideal_time = ideal_position / self.sample_rate
                                    chunk_num = self.chunk_stats[transcription_id]

                                    if drift_seconds > 0:
                                        logger.info(
                                            f"🎵 Chunk #{chunk_num}: pos={actual_position} ({actual_time:.3f}s), drift=+{drift_seconds:.3f}s from ideal {ideal_time:.3f}s (id: {transcription_id[:8]}{'*' if last_chunk else ''})"
                                        )
                                    else:
                                        logger.info(
                                            f"🎵 Chunk #{chunk_num}: pos={actual_position} ({actual_time:.3f}s) (id: {transcription_id[:8]}{'*' if last_chunk else ''})"
                                        )

                                    # Additional progress info every 10 chunks
                                    if self.total_chunks_received % 10 == 0:
                                        logger.info(
                                            f"📊 Progress: {self.total_chunks_received} chunks, {len(self.chunk_stats)} transcriptions, pos={actual_time:.1f}s, max_drift={self.max_drift_seconds:.1f}s"
                                        )
                                else:
                                    logger.error(
                                        f"❌ Chunk exceeds array bounds: pos={actual_position} > {len(self.audio_array)} (id: {transcription_id[:8]})"
                                    )
                            else:
                                logger.warning(
                                    f"⚠️ Global start time not set, cannot place chunk (id: {transcription_id[:8]})"
                                )

                            self.receiving_audio = True

                        except Exception as e:
                            logger.error(f"Audio decode error: {e}")
                elif "eos" in msg_type.lower():
                    logger.info("🎯 End of stream received")
                    self.translation_complete = True
                elif "transcription" in msg_type:
                    transcription = data.get("data", {}).get("transcription", {})
                    text = transcription.get("text", "")
                    lang = transcription.get("language", "")
                    if text:
                        part = msg_type == "partial_transcription"
                        if not part:  # Count only final transcriptions
                            self.transcription_count += 1

                        emoji = "💬" if part else "✅"

                        # Always log all transcriptions
                        logger.info(f"{emoji} [{lang}] {text}")
            except websockets.exceptions.ConnectionClosed:
                logger.info("📡 WebSocket connection closed")
                self.ws_connected = False
                self.ws_closed = True
                break
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
                self.ws_connected = False
                self.ws_closed = True
                break

        logger.info("🔚 WebSocket receive loop ended")

    async def send(self, message: dict):
        if not self.ws:
            return False

        for attempt in range(3):
            try:
                await self.ws.send(json.dumps(message))
                return True
            except websockets.exceptions.ConnectionClosed:
                logger.warning(
                    f"❌ Connection lost during send, attempting reconnection..."
                )
                if await self.reconnect():
                    logger.info("✅ Reconnected, retrying send...")
                    continue
                else:
                    logger.error("❌ Failed to reconnect")
                    return False
            except Exception as e:
                logger.error(f"❌ Send error (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(1)
                else:
                    return False

        return False

    def _save_audio_array(self):
        """Save the audio array to WAV file"""
        logger.info(f"💾 Saving audio array to {self.output_file}...")

        # Find the last non-zero sample to trim trailing silence
        last_sample = len(self.audio_array) - 1
        while last_sample > 0 and self.audio_array[last_sample] == 0:
            last_sample -= 1

        # Trim the array to actual content + small buffer
        trimmed_array = self.audio_array[
            : last_sample + int(self.sample_rate * 2)
        ]  # +2s buffer

        # Initialize WAV writer
        wav_writer = wave.open(self.output_file, "wb")
        wav_writer.setnchannels(1)
        wav_writer.setsampwidth(2)  # 16-bit
        wav_writer.setframerate(self.sample_rate)

        # Write the entire array
        wav_writer.writeframes(trimmed_array.tobytes())
        wav_writer.close()

        # Calculate statistics
        total_duration = len(trimmed_array) / self.sample_rate
        non_zero_samples = np.count_nonzero(trimmed_array)
        audio_duration = non_zero_samples / self.sample_rate
        pause_duration = total_duration - audio_duration

        logger.info(
            f"📊 Total: {total_duration:.1f}s (Audio: {audio_duration:.1f}s, Silence: {pause_duration:.1f}s)"
        )
        logger.info(
            f"📈 Original: {self.original_duration:.1f}s, Generated: {total_duration:.1f}s"
        )
        logger.info(
            f"🎵 Transcriptions processed: {len(self.chunk_stats)} ({self.transcription_count} final)"
        )
        logger.info(f"🔢 Total chunks received: {self.total_chunks_received}")
        logger.info(f"⏱️ Total drift accumulated: {self.total_drift_seconds:.1f}s")
        logger.info(f"📊 Maximum drift: {self.max_drift_seconds:.1f}s")
        logger.info(
            f"📏 Final audio position: {self.last_audio_end_position / self.sample_rate:.1f}s"
        )
        logger.info(f"🔧 Logging mode: verbose (file: {self.original_duration:.1f}s)")

        # Calculate average drift per transcription
        if self.transcription_drifts:
            avg_drifts = {
                tid: sum(drifts) / len(drifts)
                for tid, drifts in self.transcription_drifts.items()
            }
            max_drift_transcription = max(avg_drifts.items(), key=lambda x: x[1])
            logger.info(
                f"📈 Worst drift transcription: {max_drift_transcription[0][:8]} (avg {max_drift_transcription[1]:.2f}s)"
            )

            # Count chunks with high drift
            high_drift_count = sum(
                1
                for drifts in self.transcription_drifts.values()
                for d in drifts
                if d > 2.0
            )
            logger.info(
                f"⚠️ Chunks with >2s drift: {high_drift_count}/{self.total_chunks_received} ({100*high_drift_count/max(1,self.total_chunks_received):.1f}%)"
            )

        # Show top 5 transcriptions by chunk count with drift info
        if self.chunk_stats:
            top_transcriptions = sorted(
                self.chunk_stats.items(), key=lambda x: x[1], reverse=True
            )[:5]
            logger.info("📊 Top transcriptions by chunk count:")
            for trans_id, count in top_transcriptions:
                avg_drift = 0.0
                if (
                    trans_id in self.transcription_drifts
                    and self.transcription_drifts[trans_id]
                ):
                    avg_drift = sum(self.transcription_drifts[trans_id]) / len(
                        self.transcription_drifts[trans_id]
                    )
                logger.info(
                    f"   {trans_id}: {count} chunks (avg drift: {avg_drift:.2f}s)"
                )

    async def close(self):
        """Close WebSocket and save audio"""
        logger.info("🔒 Closing WebSocket connection...")

        if self.ws and not self.ws_closed:
            try:
                await asyncio.wait_for(self.ws.close(), timeout=5.0)
                logger.info("✅ WebSocket closed gracefully")
            except Exception as e:
                logger.warning(f"⚠️ Error closing WebSocket: {e}")

        self._save_audio_array()


### Step 3: Read and Send Audio File ###
async def send_audio_file(ws: FileWebSocket, input_file: str):
    sample_rate = 24000
    chunk_duration = 0.1  # 320ms
    chunk_samples = int(sample_rate * chunk_duration)

    # Open audio file with PyAV (supports MP3, WAV, etc.)
    try:
        container = av.open(input_file)
        audio_stream = container.streams.audio[0]

        logger.info(f"📖 Reading from: {input_file}")
        logger.info(
            f"🎵 Format: {audio_stream.channels}ch, {audio_stream.rate}Hz, codec={audio_stream.codec.name}"
        )

        # Calculate original duration
        original_duration = float(audio_stream.duration * audio_stream.time_base)
        logger.info(f"⏱️ Original duration: {original_duration:.1f}s")

        # Decode audio to numpy array
        audio_samples = []
        for frame in container.decode(audio_stream):
            # Convert to numpy array and flatten to mono if needed
            audio_data = frame.to_ndarray()
            if len(audio_data.shape) > 1:  # Multi-channel
                audio_data = np.mean(audio_data, axis=0)  # Convert to mono
            audio_samples.append(audio_data)

        container.close()

        if not audio_samples:
            logger.error("❌ No audio data found in file")
            return None

        # Concatenate all samples
        audio_array = np.concatenate(audio_samples).astype(np.float32)
        logger.info(
            f"🔍 Raw audio before resampling: shape={audio_array.shape}, min={audio_array.min():.6f}, max={audio_array.max():.6f}"
        )

        # Resample if needed
        if audio_stream.rate != sample_rate:
            logger.info(f"🔄 Resampling from {audio_stream.rate}Hz to {sample_rate}Hz")
            ratio = sample_rate / audio_stream.rate
            new_length = int(len(audio_array) * ratio)
            indices = np.linspace(0, len(audio_array) - 1, new_length)
            audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)
            logger.info(
                f"🔍 After resampling: shape={audio_array.shape}, min={audio_array.min():.6f}, max={audio_array.max():.6f}"
            )

        # Convert to int16 PCM with proper normalization to prevent clipping
        # First normalize to [-1, 1] range based on actual peak values
        max_abs = max(abs(audio_array.min()), abs(audio_array.max()))
        if max_abs > 0:
            audio_array = audio_array / max_abs  # Normalize to [-1, 1]

        # Then convert to int16 range
        audio_array = (audio_array * 32767).astype(np.int16)

        # Debug: log audio statistics after conversion
        logger.info(f"🔍 Audio statistics after PyAV processing:")
        logger.info(
            f"   Shape: {audio_array.shape}, Duration: {len(audio_array)/sample_rate:.1f}s"
        )
        logger.info(f"   Min: {audio_array.min()}, Max: {audio_array.max()}")
        logger.info(
            f"   Mean: {np.mean(audio_array):.2f}, Std: {np.std(audio_array):.2f}"
        )
        logger.info(
            f"   RMS: {np.sqrt(np.mean(audio_array.astype(np.float64)**2)):.2f}"
        )
        logger.info(
            f"   Non-zero samples: {np.count_nonzero(audio_array)}/{len(audio_array)} ({100*np.count_nonzero(audio_array)/len(audio_array):.1f}%)"
        )
        logger.info(f"   First 10 samples: {audio_array[:10].tolist()}")
        logger.info(f"   Last 10 samples: {audio_array[-10:].tolist()}")

    except Exception as e:
        logger.error(f"❌ Error reading audio file: {e}")
        return None

    # Send audio chunks with real-time pacing and batching
    logger.info("🎤 Sending audio chunks...")
    total_chunks = len(audio_array) // chunk_samples
    batch_size = 50  # Send in batches to be more stable

    # Set global start time when sending first chunk
    ws.global_start_time = time.perf_counter()
    send_start_time = ws.global_start_time

    for i in range(0, len(audio_array), chunk_samples):
        chunk = audio_array[i : i + chunk_samples]

        # Pad last chunk if needed
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant")

        # Calculate expected send time for this chunk
        chunk_num = i // chunk_samples + 1
        expected_send_time = send_start_time + (chunk_num - 1) * chunk_duration
        current_time = time.perf_counter()

        # If we're behind, catch up with burst sending
        if current_time > expected_send_time + chunk_duration:
            logger.warning(
                f"⚡ Catching up - behind by {current_time - expected_send_time:.1f}s"
            )
        else:
            # Wait until it's time to send this chunk
            wait_time = max(0, expected_send_time - current_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        # Send via WebSocket with retry logic
        message = {
            "message_type": "input_audio_data",
            "data": {"data": base64.b64encode(chunk.tobytes()).decode("utf-8")},
        }

        success = await ws.send(message)
        if not success:
            logger.error(f"❌ Failed to send chunk {chunk_num}/{total_chunks}")
            return original_duration

        # Log EVERY chunk sent
        actual_send_time = time.perf_counter() - send_start_time
        expected_elapsed = (chunk_num - 1) * chunk_duration
        timing_drift = actual_send_time - expected_elapsed
        logger.info(
            f"🎤 Sent chunk {chunk_num}/{total_chunks}: time={actual_send_time:.3f}s, expected={expected_elapsed:.3f}s, drift={timing_drift:+.3f}s"
        )

        # Small pause every batch for stability (but don't break timing)
        if chunk_num % batch_size == 0:
            await asyncio.sleep(0.05)

    logger.info(f"✅ Finished sending {total_chunks} chunks")
    return original_duration


### Step 4: Settings ###
def create_translation_settings(source_lang: str, target_lang: str) -> dict:
    return {
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
            "translation_queue_configs": {
                "global": {
                    "desired_queue_level_ms": 4000,
                    "max_queue_level_ms": 15000,
                    "auto_tempo": True,
                    "min_tempo": 1.5,
                    "max_tempo": 1.9,
                }
            },
            "transcription": {"source_language": source_lang},
            "translations": [
                {
                    "target_language": target_lang,
                    "speech_generation": {},
                }
            ],
        },
    }


### RUNNER ###


def analyze_rms_comparison(
    original_file: str, translated_file: str, window_seconds: int = 10
):
    """Compare RMS levels of original and translated files to verify pause preservation"""
    try:
        # Load original file using PyAV
        container = av.open(original_file)
        audio_stream = container.streams.audio[0]
        original_rate = audio_stream.rate

        audio_samples = []
        for frame in container.decode(audio_stream):
            audio_data = frame.to_ndarray()
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
            audio_samples.append(audio_data)
        container.close()

        if audio_samples:
            original_audio = np.concatenate(audio_samples).astype(np.float32)
        else:
            logger.error("No audio data in original file")
            return

        # Load translated file
        with wave.open(translated_file, "rb") as wav:
            translated_rate = wav.getframerate()
            translated_data = wav.readframes(wav.getnframes())
            translated_audio = (
                np.frombuffer(translated_data, dtype=np.int16).astype(np.float32)
                / 32768.0
            )

        # Resample original to match translated if needed
        if original_rate != translated_rate:
            ratio = translated_rate / original_rate
            new_length = int(len(original_audio) * ratio)
            indices = np.linspace(0, len(original_audio) - 1, new_length)
            original_audio = np.interp(
                indices, np.arange(len(original_audio)), original_audio
            )

        logger.info(f"\n📊 RMS Analysis (10s windows):")
        logger.info(
            f"Original: {len(original_audio)/translated_rate:.1f}s, Translated: {len(translated_audio)/translated_rate:.1f}s"
        )

        window_samples = window_seconds * translated_rate
        min_length = min(len(original_audio), len(translated_audio))

        for i in range(0, int(min_length), int(window_samples)):
            end = min(i + int(window_samples), min_length)
            orig_window = original_audio[i:end]
            trans_window = translated_audio[i:end]

            orig_rms = np.sqrt(np.mean(orig_window**2))
            trans_rms = np.sqrt(np.mean(trans_window**2))

            time_mark = i / translated_rate
            logger.info(
                f"  {time_mark:6.1f}s: Orig RMS={orig_rms:.4f}, Trans RMS={trans_rms:.4f}, Ratio={trans_rms/max(orig_rms,1e-6):.2f}"
            )

    except Exception as e:
        logger.error(f"❌ RMS analysis failed: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="File-to-file translation using WebSocket"
    )
    parser.add_argument("source_lang", help="Source language code (e.g., 'en')")
    parser.add_argument("target_lang", help="Target language code (e.g., 'es')")
    parser.add_argument("input_file", help="Path to input WAV file")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory (default: current directory)",
    )
    return parser.parse_args()


async def main():
    try:
        args = parse_args()

        # Input and output files
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"❌ Input file not found: {input_path}")
            return

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_path.stem}_{args.source_lang}2{args.target_lang}_ws_{timestamp}.wav"
        output_file = Path(args.output_dir) / output_filename

        # Setup logging
        log_file = output_file.with_suffix(".log")
        logger.remove()  # Remove default handler
        # Same format for both file and console
        log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}"
        logger.add(log_file, format=log_format)
        logger.add(sys.stderr, format=log_format)  # Use stderr for console output

        logger.info("🚀 Palabra File2File WebSocket Client")
        logger.info(f"📝 Translation: {args.source_lang} → {args.target_lang}")
        logger.info(f"📖 Input: {input_path}")
        logger.info(f"📝 Output: {output_file}")
        logger.info(f"📋 Log: {log_file}")

        # Get original file duration using PyAV
        try:
            container = av.open(str(input_path))
            audio_stream = container.streams.audio[0]
            original_duration = float(audio_stream.duration * audio_stream.time_base)
            container.close()
            logger.info(f"⏱️ Original duration: {original_duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ Error reading input file: {e}")
            return

        # Create session
        session = await create_session(
            os.getenv("PALABRA_CLIENT_ID"), os.getenv("PALABRA_CLIENT_SECRET")
        )
        ws_url = session["data"]["ws_url"]
        publisher = session["data"]["publisher"]

        # Connect WebSocket
        ws = FileWebSocket(ws_url, publisher, str(output_file), original_duration)
        await ws.connect()

        try:
            # Create translation settings
            settings = create_translation_settings(args.source_lang, args.target_lang)

            # Send settings and wait
            await ws.send({"message_type": "set_task", "data": settings})
            langs = [t["target_language"] for t in settings["pipeline"]["translations"]]
            logger.info(f"⚙️ Settings sent: {args.source_lang} → {langs}")

            # Wait for settings to process
            logger.info("⏳ Waiting for settings to process...")
            await asyncio.sleep(3)

            # Send audio file
            audio_duration = await send_audio_file(ws, str(input_path))
            if audio_duration is None:
                return

            # Fixed wait time for all files
            max_wait_time = 60  # Always wait 60 seconds regardless of file size
            logger.info(
                f"⏳ Waiting for translation to complete (max {max_wait_time:.0f}s)..."
            )

            # Wait for translation completion or timeout
            start_wait = time.time()
            last_progress_log = 0

            while time.time() - start_wait < max_wait_time:
                current_wait = time.time() - start_wait

                # Log progress every 10 seconds
                if current_wait - last_progress_log >= 10:
                    logger.info(
                        f"⏱️ Waiting progress: {current_wait:.0f}s/{max_wait_time:.0f}s"
                    )
                    last_progress_log = current_wait

                if ws.translation_complete:
                    logger.info("✅ Translation complete!")
                    break

                # Also exit if WebSocket disconnected and we've waited 30s
                if not ws.ws_connected and time.time() - start_wait > 30:
                    logger.info(
                        "⏰ WebSocket disconnected, waited 30s for remaining chunks"
                    )
                    break

                await asyncio.sleep(1)
            else:
                logger.warning("⏰ Translation timeout reached")

            # Give a final grace period for any remaining chunks
            logger.info("⏳ Final grace period...")
            await asyncio.sleep(3)

        finally:
            logger.info("🧹 Cleaning up...")
            await ws.close()

        # Analyze RMS comparison
        if output_file.exists():
            analyze_rms_comparison(str(input_path), str(output_file))

    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        logger.info("🏁 Done")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown complete")
        os._exit(0)
