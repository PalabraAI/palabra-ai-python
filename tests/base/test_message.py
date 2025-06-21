import pytest
from palabra_ai.base.message import (
    Message, KnownRaw, KnownRawType, EmptyMessage,
    QueueStatusMessage, UnknownMessage, PipelineTimingsMessage,
    TranscriptionMessage, TranscriptionSegment
)
from palabra_ai.lang import Language


class TestMessage:
    def test_detect_types(self):
        # Null
        result = Message.detect(None)
        assert result.type == KnownRawType.null
        assert result.data is None

        # JSON bytes
        result = Message.detect(b'{"test": true}')
        assert result.type == KnownRawType.json
        assert result.data == {"test": True}

        # JSON string
        result = Message.detect('{"test": true}')
        assert result.type == KnownRawType.json
        assert result.data == {"test": True}

        # Invalid JSON
        result = Message.detect('invalid json')
        assert result.type == KnownRawType.unknown
        assert result.data == 'invalid json'

        # Binary
        result = Message.detect(b'binary data')
        assert result.type == KnownRawType.unknown
        assert result.data == b'binary data'

    def test_decode_unknown(self):
        msg = Message.decode("not json")
        assert isinstance(msg, UnknownMessage)
        assert msg.raw_type == KnownRawType.unknown

    def test_message_str(self):
        msg = Message(message_type=Message.Type.PIPELINE_TIMINGS)
        str_repr = str(msg)
        assert isinstance(str_repr, str)
        assert "pipeline_timings" in str_repr


class TestEmptyMessage:
    def test_create_and_dump(self):
        known_raw = KnownRaw(KnownRawType.json, {})
        msg = EmptyMessage.create(known_raw)
        assert msg.type_ == Message.Type._EMPTY
        assert msg.model_dump() == {}
        assert str(msg) == "⚪"


class TestQueueStatusMessage:
    def test_create_and_dump(self):
        data = {"en": {"current_queue_level_ms": 100, "max_queue_level_ms": 500}}
        known_raw = KnownRaw(KnownRawType.json, data)
        msg = QueueStatusMessage.create(known_raw)

        assert msg.language.code == "en"
        assert msg.current_queue_level_ms == 100
        assert msg.max_queue_level_ms == 500

        dump = msg.model_dump()
        assert dump != {
            "es": {
                "current_queue_level_ms": 100,
                "max_queue_level_ms": 500
            }
        }


class TestUnknownMessage:
    def test_create_with_bytes(self):
        known_raw = KnownRaw(KnownRawType.binary, b"test data")
        msg = UnknownMessage.create(known_raw)
        assert msg.raw_type == KnownRawType.binary
        assert msg.raw_data == "test data"  # Decoded

    def test_create_with_exception(self):
        exc = ValueError("test error")
        known_raw = KnownRaw(KnownRawType.json, {"test": 1}, exc)
        msg = UnknownMessage.create(known_raw)

        assert msg.error_info is not None
        assert msg.error_info["type"] == "ValueError"
        assert msg.error_info["message"] == "test error"


class TestPipelineTimingsMessage:
    def test_extract_from_nested(self):
        data = {
            "message_type": "pipeline_timings",
            "data": {
                "transcription_id": "123",
                "timings": {"start": 0.1, "end": 0.5}
            }
        }

        msg = PipelineTimingsMessage.model_validate(data)
        assert msg.transcription_id == "123"
        assert msg.timings == {"start": 0.1, "end": 0.5}


class TestTranscriptionMessage:
    def test_dedup(self):
        msg = TranscriptionMessage(
            message_type=Message.Type.PARTIAL_TRANSCRIPTION,
            transcription_id="123",
            text="Hello",
            language=Language("en"),
            segments=[]
        )

        dedup = msg.dedup
        assert "123" in dedup
        assert "Hello" in dedup

    def test_transcription_segment_repr(self):
        segment = TranscriptionSegment(
            text="Hello",
            confidence=0.95,
            start=1.0,
            end=2.0,
            start_timestamp=1.0
        )
        repr_str = repr(segment)
        assert isinstance(repr_str, str)
        assert "Hello" in repr_str

    def test_duration_with_segments(self):
        msg = TranscriptionMessage(
            message_type=Message.Type.PARTIAL_TRANSCRIPTION,
            transcription_id="123",
            text="Hello world",
            language=Language("en"),
            segments=[
                TranscriptionSegment(
                    text="Hello",
                    confidence=0.9,
                    start=1.0,
                    end=1.5,
                    start_timestamp=1.0
                ),
                TranscriptionSegment(
                    text="world",
                    confidence=0.9,
                    start=1.5,
                    end=2.0,
                    start_timestamp=1.5
                )
            ]
        )

        assert msg.text == "Hello world"
        assert len(msg.segments) == 2
