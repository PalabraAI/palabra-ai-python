import base64
import ctypes
import io
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from livekit.rtc import AudioFrame as RtcAudioFrame

from palabra_ai.constant import AUDIO_BUFFER_PADDING_SEC, BYTES_PER_SAMPLE
from palabra_ai.util.logger import error
from palabra_ai.util.orjson import from_json, to_json
from palabra_ai.util.timing import get_perf_ts


def save_wav(np_audio: np.typing.NDArray, output_path: Path, sr: int, ch: int):
    """Save audio chunks to WAV file"""
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(ch)
        wav.setframerate(sr)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.writeframes(np_audio.tobytes())


class AudioFrame:
    """Lightweight AudioFrame replacement with __slots__ for performance"""

    __slots__ = (
        "data",
        "sample_rate",
        "num_channels",
        "samples_per_channel",
        "_dbg",
        "last_chunk",
        "transcription_id",
        "language",
        "perf_ts",
    )

    def __init__(
        self,
        data: np.ndarray | bytes,
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int,
        language: str | None = None,
        transcription_id: str | None = None,
        last_chunk: bool | None = None,
        perf_ts: float | None = None,
    ):
        if isinstance(data, bytes):
            # Convert bytes to numpy array
            self.data = np.frombuffer(data, dtype=np.int16)
        else:
            self.data = data

        self.sample_rate = sample_rate
        self.num_channels = num_channels

        self.last_chunk = last_chunk
        self.transcription_id = transcription_id
        self.language = language
        self.perf_ts = perf_ts if perf_ts is not None else get_perf_ts()

        if samples_per_channel is None:
            self.samples_per_channel = len(self.data) // num_channels
        else:
            self.samples_per_channel = samples_per_channel

    @property
    def dbg_delta(self) -> str:
        if getattr(self, "_dbg", None):
            return str(self._dbg.delta)
        else:
            return "n/a"

    @property
    def duration(self) -> float:
        """Duration of the audio frame in seconds"""
        return self.samples_per_channel / self.sample_rate

    @classmethod
    def create(
        cls,
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int,
        perf_ts: float | None = None,
    ) -> "AudioFrame":
        """
        Create a new empty AudioFrame instance with specified sample rate, number of channels,
        and samples per channel.

        Args:
            sample_rate (int): The sample rate of the audio in Hz.
            num_channels (int): The number of audio channels (e.g., 1 for mono, 2 for stereo).
            samples_per_channel (int): The number of samples per channel.
            perf_ts (float|None): Performance timestamp, defaults to current time if None.

        Returns:
            AudioFrame: A new AudioFrame instance with uninitialized (zeroed) data.
        """
        size = num_channels * samples_per_channel * ctypes.sizeof(ctypes.c_int16)
        data = bytearray(size)
        return cls(
            data, sample_rate, num_channels, samples_per_channel, perf_ts=perf_ts
        )

    @classmethod
    def create_silence(
        cls,
        donor_frame: "AudioFrame",
        duration_seconds: float,
        pause_start_perf_ts: float,
    ) -> "AudioFrame|None":
        """
        Create AudioFrame filled with silence of specified duration.

        Args:
            donor_frame: Frame to copy audio parameters from (sample_rate, num_channels)
            duration_seconds: Duration of silence in seconds
            pause_start_perf_ts: Performance timestamp when the pause should start

        Returns:
            AudioFrame with silence or None if invalid duration
        """
        if duration_seconds <= 0:
            return None

        samples_per_channel = int(donor_frame.sample_rate * duration_seconds)
        if samples_per_channel <= 0:
            return None

        total_samples = samples_per_channel * donor_frame.num_channels
        silence_data = np.zeros(total_samples, dtype=np.int16)

        return cls(
            data=silence_data,
            sample_rate=donor_frame.sample_rate,
            num_channels=donor_frame.num_channels,
            samples_per_channel=samples_per_channel,
            perf_ts=pause_start_perf_ts,
        )

    def __repr__(self):
        return f"🗣️<AF(dur={self.duration:.3f}s, s={self.samples_per_channel}, sr={self.sample_rate}, ch={self.num_channels})>"

    def __bool__(self):
        """Return False if data is empty, True otherwise"""
        if self.data is None:
            return False
        if hasattr(self.data, "__len__"):
            return len(self.data) > 0
        return True

    @classmethod
    def from_rtc(
        cls, frame: RtcAudioFrame, perf_ts: float | None = None
    ) -> "AudioFrame":
        """Create AudioFrame from LiveKit's RtcAudioFrame"""
        return cls(
            data=frame.data,
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
            perf_ts=perf_ts,
        )

    @classmethod
    def from_ws(
        cls,
        raw_msg: bytes | str,
        sample_rate: int,
        num_channels: int,
        perf_ts: float | None = None,
    ) -> Optional["AudioFrame"]:
        """Create AudioFrame from WebSocket message

        Expected format:
        {
            "message_type": "output_audio_data",
            "data": {
                "data": "<base64_encoded_audio>"
            }
        }
        """

        if not isinstance(raw_msg, bytes | str):
            return None
        elif isinstance(raw_msg, str) and "output_audio_data" not in raw_msg:
            return None
        elif isinstance(raw_msg, bytes) and b"output_audio_data" not in raw_msg:
            return None

        msg = from_json(raw_msg)
        if msg.get("message_type") != "output_audio_data":
            return None

        if "data" not in msg:
            return None

        if isinstance(msg["data"], str):
            # If data is a string, decode it
            msg["data"] = from_json(msg["data"])

        if "data" not in msg["data"]:
            return None

        msg_top_level = msg["data"]

        # Extract base64 data
        base64_data = msg_top_level["data"]

        try:
            # Decode base64 to bytes
            audio_bytes = base64.b64decode(base64_data)

            samples_per_channel = len(audio_bytes) // (BYTES_PER_SAMPLE * num_channels)

            return cls(
                data=audio_bytes,
                sample_rate=sample_rate,
                num_channels=num_channels,
                samples_per_channel=samples_per_channel,
                language=msg_top_level["language"],
                transcription_id=msg_top_level["transcription_id"],
                last_chunk=msg_top_level["last_chunk"],
                perf_ts=perf_ts,
            )
        except Exception as e:
            error(f"Failed to decode audio data: {e}")

    def to_rtc(self) -> RtcAudioFrame:
        return RtcAudioFrame(
            data=self.data,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            samples_per_channel=self.samples_per_channel,
        )

    def to_ws(self) -> bytes:
        """Convert AudioFrame to WebSocket message format

        Returns:
        {
            "message_type": "input_audio_data",
            "data": {
                "data": "<base64_encoded_audio>"
            }
        }
        """

        return to_json(
            {
                "message_type": "input_audio_data",
                "data": {"data": base64.b64encode(self.data)},
            }
        )

    def to_bench(self):
        result = {
            "message_type": "__$bench_audio_frame",
            "__dbg": {
                "size": len(self.data),
                "sample_rate": self.sample_rate,
                "num_channels": self.num_channels,
                "samples_per_channel": self.samples_per_channel,
            },
            "data": {},
        }

        # Include transcription metadata only for output frames (those with transcription_id)
        if self.transcription_id:
            result["data"].update(
                {
                    "transcription_id": self.transcription_id,
                    "language": self.language,
                    "last_chunk": self.last_chunk,
                }
            )

        # Replace base64 audio data with "..." to avoid log pollution
        if "data" in result["data"] and isinstance(result["data"]["data"], str):
            result["data"]["data"] = "..."

        return result


@dataclass
class AudioBuffer:
    sample_rate: int
    num_channels: int
    original_duration: float = field(default=60.0)  # Default 60s buffer
    drop_empty_frames: bool = field(default=False)
    audio_array: np.ndarray = field(default=None, init=False)
    last_audio_end_position: int = field(default=0, init=False)
    global_start_time: float | None = field(default=None, init=False)
    external_buffer: io.BytesIO | None = field(default=None, init=False)

    def __post_init__(self):
        # Create numpy array like file2file - duration + buffer padding
        total_samples = int(
            (self.original_duration + AUDIO_BUFFER_PADDING_SEC) * self.sample_rate
        )
        self.audio_array = np.zeros(total_samples, dtype=np.int16)

    def set_start_time(self, start_time: float):
        """Set global start time for timing calculations"""
        if self.global_start_time is None:
            self.global_start_time = start_time

    def replace_buffer(self, external_buffer: io.BytesIO):
        """Replace internal buffer with external BytesIO buffer for direct writing"""
        self.external_buffer = external_buffer

    def to_wav_bytes(self) -> bytes:
        """Convert buffer to WAV format"""
        if self.audio_array is None or len(self.audio_array) == 0:
            from palabra_ai.util.logger import warning

            warning("Buffer is empty, returning empty WAV data")
            return b""

        # Find the last non-zero sample to trim trailing silence
        last_sample = len(self.audio_array) - 1
        while last_sample > 0 and self.audio_array[last_sample] == 0:
            last_sample -= 1

        # Trim the array to actual content + small buffer
        trimmed_array = self.audio_array[
            : last_sample + int(self.sample_rate * 2)
        ]  # +2s buffer

        # If external buffer is set, write directly to it
        if self.external_buffer is not None:
            self.external_buffer.seek(0)
            with wave.open(self.external_buffer, "wb") as wav:
                wav.setnchannels(self.num_channels)
                wav.setframerate(self.sample_rate)
                wav.setsampwidth(BYTES_PER_SAMPLE)
                wav.writeframes(trimmed_array.tobytes())
            self.external_buffer.seek(0)
            return self.external_buffer.getvalue()
        else:
            # Default behavior: return bytes
            with io.BytesIO() as wav_file:
                with wave.open(wav_file, "wb") as wav:
                    wav.setnchannels(self.num_channels)
                    wav.setframerate(self.sample_rate)
                    wav.setsampwidth(BYTES_PER_SAMPLE)
                    wav.writeframes(trimmed_array.tobytes())
                return wav_file.getvalue()

    async def write(self, frame: AudioFrame):
        """Write frame to buffer using file2file approach"""
        if self.audio_array is None:
            return

        # Convert frame data to numpy array if needed
        if isinstance(frame.data, bytes):
            audio_samples = np.frombuffer(frame.data, dtype=np.int16)
        else:
            audio_samples = frame.data

        # Skip empty frames if drop_empty_frames is enabled
        if self.drop_empty_frames and np.all(audio_samples == 0):
            return

        # Calculate position like file2file
        if self.global_start_time is not None and frame.perf_ts is not None:
            # Calculate ideal position based on timing
            elapsed_time = frame.perf_ts - self.global_start_time
            ideal_position = int(elapsed_time * self.sample_rate)

            # Find earliest available position without overlaps
            actual_position = max(ideal_position, self.last_audio_end_position)
            end_position = actual_position + len(audio_samples)

            # Write to array if it fits
            if end_position <= len(self.audio_array):
                self.audio_array[actual_position:end_position] = audio_samples
                self.last_audio_end_position = end_position

                # Calculate and store drift for metrics
                drift_samples = actual_position - ideal_position
                drift_seconds = drift_samples / self.sample_rate

                # Store drift info in frame for TTS metrics
                if hasattr(frame, "_drift_seconds"):
                    frame._drift_seconds = drift_seconds
        else:
            # Fallback: just append at current position
            end_position = self.last_audio_end_position + len(audio_samples)
            if end_position <= len(self.audio_array):
                self.audio_array[self.last_audio_end_position : end_position] = (
                    audio_samples
                )
                self.last_audio_end_position = end_position
