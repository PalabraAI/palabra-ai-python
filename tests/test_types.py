import asyncio
from typing import get_args
from palabra_ai.types import T_ON_TRANSCRIPTION
from palabra_ai.message import TranscriptionMessage


def test_t_on_transcription_type():
    """Test T_ON_TRANSCRIPTION type alias"""
    # Just importing it achieves coverage
    assert T_ON_TRANSCRIPTION is not None

    # Verify the type structure
    args = get_args(T_ON_TRANSCRIPTION)
    assert len(args) == 2

    # Test sync callback
    def sync_callback(msg: TranscriptionMessage) -> None:
        pass

    # Test async callback
    async def async_callback(msg: TranscriptionMessage) -> None:
        pass

    # These should work with the type (for documentation purposes)
    assert callable(sync_callback)
    assert asyncio.iscoroutinefunction(async_callback)
