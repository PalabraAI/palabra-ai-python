from enum import StrEnum


class MessageType(StrEnum):
    PARTIAL_TRANSCRIPTION = "partial_transcription"
    TRANSLATED_TRANSCRIPTION = "translated_transcription"
    VALIDATED_TRANSCRIPTION = "validated_transcription"
    PARTIAL_TRANSLATED_TRANSCRIPTION = "partial_translated_transcription"
    PIPELINE_TIMINGS = "pipeline_timings"
    _QUEUE_STATUS = "queue_status"  # For "es" messages
    _EMPTY = "empty"  # For empty {} messages
    _UNKNOWN = "unknown"  # For unrecognized message formats


TRANSCRIPTION_MESSAGE_TYPES = {
    mt.value
    for mt in (
        MessageType.PARTIAL_TRANSCRIPTION,
        MessageType.TRANSLATED_TRANSCRIPTION,
        MessageType.VALIDATED_TRANSCRIPTION,
        MessageType.PARTIAL_TRANSLATED_TRANSCRIPTION,
    )
}
ALLOWED_MESSAGE_TYPES = {
    MessageType.PIPELINE_TIMINGS.value
} | TRANSCRIPTION_MESSAGE_TYPES


class Channel(StrEnum):
    """Channel names for WebRTC messages."""

    WS = "ws"
    WEBRTC = "webrtc"


class Direction(StrEnum):
    """Direction of messages."""

    IN = "in"
    OUT = "out"
