import base64
import ctypes
import io
import wave
from asyncio import to_thread
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from livekit.rtc import AudioFrame as RtcAudioFrame

from palabra_ai.constant import BYTES_PER_SAMPLE
from palabra_ai.enum import Direction
from palabra_ai.util.logger import error
from palabra_ai.util.orjson import from_json, to_json


class AudioFrame:
    """Lightweight AudioFrame replacement with __slots__ for performance"""

    __slots__ = ("_dbg", "data", "sample_rate", "num_channels", "samples_per_channel")

    def __init__(
        self,
        data: np.ndarray | bytes,
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int,
    ):
        if isinstance(data, bytes):
            # Convert bytes to numpy array
            self.data = np.frombuffer(data, dtype=np.int16)
        else:
            self.data = data

        self.sample_rate = sample_rate
        self.num_channels = num_channels

        if samples_per_channel is None:
            self.samples_per_channel = len(self.data) // num_channels
        else:
            self.samples_per_channel = samples_per_channel

    @classmethod
    def create(
        cls, sample_rate: int, num_channels: int, samples_per_channel: int
    ) -> "AudioFrame":
        """
        Create a new empty AudioFrame instance with specified sample rate, number of channels,
        and samples per channel.

        Args:
            sample_rate (int): The sample rate of the audio in Hz.
            num_channels (int): The number of audio channels (e.g., 1 for mono, 2 for stereo).
            samples_per_channel (int): The number of samples per channel.

        Returns:
            AudioFrame: A new AudioFrame instance with uninitialized (zeroed) data.
        """
        size = num_channels * samples_per_channel * ctypes.sizeof(ctypes.c_int16)
        data = bytearray(size)
        return cls(data, sample_rate, num_channels, samples_per_channel)

    def __repr__(self):
        return f"🗣️<AF(s={self.samples_per_channel}, sr={self.sample_rate}, ch={self.num_channels})>"

    def __bool__(self):
        """Return False if data is empty, True otherwise"""
        if self.data is None:
            return False
        if hasattr(self.data, "__len__"):
            return len(self.data) > 0
        return True

    @classmethod
    def from_rtc(cls, frame: RtcAudioFrame) -> "AudioFrame":
        """Create AudioFrame from LiveKit's RtcAudioFrame"""
        return cls(
            data=frame.data,
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
        )

    @classmethod
    def from_ws(
        cls,
        raw_msg: bytes | str,
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int,
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

        # Extract base64 data
        base64_data = msg["data"]["data"]

        try:
            # Decode base64 to bytes
            audio_bytes = base64.b64decode(base64_data)

            return cls(
                data=audio_bytes,
                sample_rate=sample_rate,
                num_channels=num_channels,
                samples_per_channel=samples_per_channel,
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

    def model_dump(self):
        return {
            "message_type": "input_audio_data" if self._dbg.direction == Direction.IN else "output_audio_data",
            "data": {
                "data_size": len(self.data),
                "sample_rate": self.sample_rate,
                "num_channels": self.num_channels,
                "samples_per_channel": self.samples_per_channel,
            }
        }



@dataclass
class AudioBuffer:
    sample_rate: int
    num_channels: int
    b: io.BytesIO = field(default_factory=io.BytesIO, init=False)
    drop_empty_frames: bool = field(default=False)

    def to_wav_bytes(self) -> bytes:
        """Convert buffer to WAV format"""
        if self.b.getbuffer().nbytes == 0:
            from palabra_ai.util.logger import warning

            warning("Buffer is empty, returning empty WAV data")
            return b""

        with io.BytesIO() as wav_file:
            with wave.open(wav_file, "wb") as wav:
                wav.setnchannels(self.num_channels)
                wav.setframerate(self.sample_rate)
                wav.setsampwidth(BYTES_PER_SAMPLE)
                wav.writeframes(self.b.getvalue())
            return wav_file.getvalue()

    async def write(self, frame: AudioFrame):
        frame_bytes = frame.data.tobytes()
        if self.drop_empty_frames and all(byte == 0 for byte in frame_bytes):
            return
        await to_thread(self.b.write, frame_bytes)

    def replace_buffer(self, new_buffer: io.BytesIO):
        self.b = new_buffer
