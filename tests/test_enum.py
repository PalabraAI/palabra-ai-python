from palabra_ai.enum import MessageType, Channel, Direction, TRANSCRIPTION_MESSAGE_TYPES, ALLOWED_MESSAGE_TYPES


def test_message_type_values():
    """Test MessageType enum values"""
    assert MessageType.PARTIAL_TRANSCRIPTION == "partial_transcription"
    assert MessageType.TRANSLATED_TRANSCRIPTION == "translated_transcription"
    assert MessageType.VALIDATED_TRANSCRIPTION == "validated_transcription"
    assert MessageType.PARTIAL_TRANSLATED_TRANSCRIPTION == "partial_translated_transcription"
    assert MessageType.PIPELINE_TIMINGS == "pipeline_timings"
    assert MessageType._QUEUE_STATUS == "queue_status"
    assert MessageType._EMPTY == "empty"
    assert MessageType._UNKNOWN == "unknown"


def test_channel_values():
    """Test Channel enum values"""
    assert Channel.WS == "ws"
    assert Channel.WEBRTC == "webrtc"


def test_direction_values():
    """Test Direction enum values"""
    assert Direction.IN == "in"
    assert Direction.OUT == "out"


def test_transcription_message_types():
    """Test TRANSCRIPTION_MESSAGE_TYPES constant"""
    expected = {
        "partial_transcription",
        "translated_transcription",
        "validated_transcription",
        "partial_translated_transcription",
    }
    assert TRANSCRIPTION_MESSAGE_TYPES == expected


def test_allowed_message_types():
    """Test ALLOWED_MESSAGE_TYPES constant"""
    expected = {
        "pipeline_timings",
        "partial_transcription",
        "translated_transcription",
        "validated_transcription",
        "partial_translated_transcription",
    }
    assert ALLOWED_MESSAGE_TYPES == expected


def test_enum_string_representation():
    """Test that enums convert to strings properly"""
    assert str(MessageType.PARTIAL_TRANSCRIPTION) == "partial_transcription"
    assert str(Channel.WS) == "ws"
    assert str(Direction.IN) == "in"
