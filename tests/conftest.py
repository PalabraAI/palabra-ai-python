"""Common fixtures and utilities for all tests."""
import asyncio
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

# Suppress Pydantic deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*class-based `cfg`.*")

# Configure pytest-asyncio
pytest_plugins = ['pytest_asyncio']


# Base mock implementations
from palabra_ai.base.adapter import Reader

class MockReader(Reader):
    """Mock Reader implementation for tests."""
    async def boot(self): pass
    async def do(self): pass
    async def exit(self): pass
    async def read(self, size=None): return b""


# Common fixtures
@pytest.fixture
def mock_reader():
    """Mock reader that satisfies Reader interface."""
    return MockReader()


# @pytest.fixture(autouse=False)
# def mock_config():
#     """Mock config object."""
#     from palabra_ai.config import Config
#     return MagicMock(spec=Config)


@pytest.fixture
def valid_source(mock_reader):
    """Valid SourceLang for tests."""
    from palabra_ai.config import SourceLang
    from palabra_ai.adapter.dummy import DummyReader
    return SourceLang(lang="en", reader=DummyReader(return_data=b"\x00\x00", eof_after_reads=10))


@pytest.fixture
def valid_target():
    """Valid TargetLang for tests."""
    from palabra_ai.config import TargetLang
    from palabra_ai.adapter.dummy import DummyWriter
    return TargetLang(lang="es", writer=DummyWriter())


@pytest.fixture
def minimal_config(valid_source, valid_target):
    """Minimal valid config."""
    from palabra_ai.config import Config
    return Config(source=valid_source, targets=valid_target)


@pytest.fixture
def mock_manager():
    """Mock manager with task."""
    manager = MagicMock()
    manager.task = asyncio.Future()
    manager.task.set_result(None)
    return manager


@pytest.fixture
def mock_av():
    """Mock av library."""
    with patch("palabra_ai.internal.audio.av") as mock:
        # Mock AudioFormat
        mock.AudioFormat.return_value = MagicMock()

        # Mock AudioResampler
        resampled_frame = MagicMock()
        resampled_frame.samples = 1024
        mock.AudioResampler.return_value.resample.return_value = [resampled_frame]

        # Mock filter
        mock.filter.Graph.return_value.configure = MagicMock()

        # Mock AVError
        mock.AVError = Exception

        yield mock


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice."""
    with patch("palabra_ai.internal.device.sd") as mock:
        # Create proper device list
        devices = [
            {"name": "test", "max_input_channels": 2, "max_output_channels": 2, "index": 0}
        ]

        mock.query_devices.return_value = devices
        mock.query_hostapis.return_value = [{"name": "test", "devices": [0]}]

        # Create stream mock
        stream = MagicMock()
        stream.latency = 0.01
        stream.active = True
        mock.RawInputStream.return_value.__enter__.return_value = stream
        mock.RawOutputStream.return_value.__enter__.return_value = stream

        yield mock


@pytest.fixture
def mock_livekit():
    """Mock livekit rtc."""
    with patch("palabra_ai.internal.webrtc.rtc") as mock:
        # Mock audio components
        mock.AudioSource.return_value = MagicMock()
        mock.LocalAudioTrack.create_audio_track.return_value = MagicMock()
        mock.AudioFrame.create.return_value = MagicMock(data=b"\x00" * 1024)

        # Mock Room
        mock_room = AsyncMock()
        mock_room.connect = AsyncMock()
        mock_room.disconnect = AsyncMock()
        mock_room.local_participant = MagicMock()
        mock_room.local_participant.publish_track = AsyncMock(return_value=MagicMock(sid="test"))
        mock_room.remote_participants = {}
        mock_room.name = ""
        mock_room._room = None

        mock.Room = MagicMock(return_value=mock_room)
        mock.RoomOptions = MagicMock

        yield mock

@pytest.fixture
def mock_aiohttp():
    """Mock aiohttp for REST client."""
    with patch("palabra_ai.internal.rest.aiohttp") as mock:
        resp = AsyncMock()
        resp.raise_for_status = MagicMock()
        resp.json = AsyncMock(return_value={
            "ok": True,
            "data": {
                "publisher": ["token1"],
                "subscriber": ["token2"],
                "room_name": "test",
                "stream_url": "wss://test",
                "control_url": "wss://control"
            }
        })

        session = AsyncMock()
        session.post = AsyncMock(return_value=resp)
        session.close = AsyncMock()
        mock.ClientSession.return_value = session
        yield mock








@pytest.fixture
def mock_audio_frame():
    """Mock rtc.AudioFrame."""
    from livekit import rtc
    frame = MagicMock(spec=rtc.AudioFrame)
    frame.data = MagicMock()
    frame.data.tobytes.return_value = b"\x00" * 1024
    frame.num_channels = 1
    frame.sample_rate = 48000
    return frame



# Base test classes with common patterns
class BaseTaskTest:
    """Base class for task tests with common patterns."""

    @pytest.fixture
    def mock_task_group(self):
        """Mock TaskGroup for task tests."""
        return MagicMock()

    async def assert_cancelled(self, coro):
        """Assert that coroutine raises CancelledError."""
        with pytest.raises(asyncio.CancelledError):
            await coro

    async def run_with_timeout(self, coro, timeout=0.1):
        """Run coroutine with timeout."""
        try:
            await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            pass
