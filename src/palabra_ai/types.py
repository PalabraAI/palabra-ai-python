from collections.abc import Awaitable, Callable
from typing import Union

from palabra_ai.message import TranscriptionMessage

T_ON_TRANSCRIPTION = Union[
    Callable[[TranscriptionMessage], None],
    Callable[[TranscriptionMessage], Awaitable[None]],
]
