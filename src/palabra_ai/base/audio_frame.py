"""Custom AudioFrame dataclass for efficient audio data handling"""
from __future__ import annotations

import numpy as np


class AudioFrame:
    """Lightweight AudioFrame replacement with __slots__ for performance"""
    __slots__ = ('data', 'sample_rate', 'num_channels', 'samples_per_channel')
    
    def __init__(self, data: np.ndarray | bytes, sample_rate: int = 48000, 
                 num_channels: int = 1, samples_per_channel: int | None = None):
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
    def from_bytes(cls, data: bytes, sample_rate: int = 48000, num_channels: int = 1) -> AudioFrame:
        """Create AudioFrame from raw bytes"""
        return cls(data, sample_rate, num_channels)
    
    def __repr__(self):
        return f"AudioFrame(samples={self.samples_per_channel}, rate={self.sample_rate}, ch={self.num_channels})"