import pytest
import time
import json
import copy
from pathlib import Path
from dataclasses import dataclass
from palabra_ai.message import (
    Message, KnownRaw, KnownRawType, Dbg,
    EmptyMessage, EndTaskMessage, EosMessage, SetTaskMessage, GetTaskMessage,
    QueueStatusMessage, ErrorMessage, UnknownMessage, PipelineTimingsMessage,
    TranscriptionMessage, TranscriptionSegment, CurrentTaskMessage,
    Channel, Direction
)
from palabra_ai.lang import Language, EN, ES
from palabra_ai.exc import ApiError, ApiValidationError, TaskNotFoundError
from palabra_ai.config import Config, SourceLang, TargetLang
from palabra_ai.util.orjson import to_json


def test_dbg_creation():
    """Test Dbg dataclass creation"""
    from palabra_ai.enum import Kind
    dbg = Dbg(kind=Kind.AUDIO, ch=Channel.WS, dir=Direction.IN)
    assert dbg.kind == Kind.AUDIO
    assert dbg.ch == Channel.WS
    assert dbg.dir == Direction.IN
    assert isinstance(dbg.perf_ts, float)
    assert isinstance(dbg.utc_ts, float)


def test_dbg_empty():
    """Test Dbg.empty() method"""
    dbg = Dbg.empty()
    assert dbg.kind is None
    assert dbg.ch is None
    assert dbg.dir is None
    assert isinstance(dbg.perf_ts, float)
    assert isinstance(dbg.utc_ts, float)


def test_known_raw_creation():
    """Test KnownRaw dataclass creation"""
    kr = KnownRaw(type=KnownRawType.json, data={"key": "value"})
    assert kr.type == KnownRawType.json
    assert kr.data == {"key": "value"}
    assert kr.exc is None


def test_message_get_transcription_message_types():
    """Test Message.get_transcription_message_types()"""
    types = Message.get_transcription_message_types()
    assert Message.Type.PARTIAL_TRANSCRIPTION in types
    assert Message.Type.TRANSLATED_TRANSCRIPTION in types
    assert Message.Type.VALIDATED_TRANSCRIPTION in types
    assert Message.Type.PARTIAL_TRANSLATED_TRANSCRIPTION in types


def test_message_get_allowed_message_types():
    """Test Message.get_allowed_message_types()"""
    types = Message.get_allowed_message_types()
    assert Message.Type.PIPELINE_TIMINGS in types
    assert Message.Type.PARTIAL_TRANSCRIPTION in types


def test_message_detect_null():
    """Test Message.detect with None"""
    kr = Message.detect(None)
    assert kr.type == KnownRawType.null
    assert kr.data is None


def test_message_detect_json_bytes():
    """Test Message.detect with JSON bytes"""
    kr = Message.detect(b'{"key": "value"}')
    assert kr.type == KnownRawType.json
    assert kr.data == {"key": "value"}


def test_message_detect_json_str():
    """Test Message.detect with JSON string"""
    kr = Message.detect('{"key": "value"}')
    assert kr.type == KnownRawType.json
    assert kr.data == {"key": "value"}


def test_message_detect_invalid_json():
    """Test Message.detect with invalid JSON"""
    kr = Message.detect('{"invalid": json}')
    assert kr.type == KnownRawType.string
    assert kr.data == '{"invalid": json}'
    assert kr.exc is not None


def test_message_detect_binary():
    """Test Message.detect with binary data"""
    kr = Message.detect(b'\x00\x01\x02')
    assert kr.type == KnownRawType.unknown
    assert kr.data == b'\x00\x01\x02'


def test_message_decode_empty():
    """Test Message.decode with empty message"""
    msg = Message.decode('{}')
    assert isinstance(msg, EmptyMessage)
    assert msg.type_ == Message.Type._EMPTY


def test_empty_message():
    """Test EmptyMessage"""
    msg = EmptyMessage()
    assert msg.model_dump() == {}
    assert str(msg) == "⚪"


def test_end_task_message():
    """Test EndTaskMessage"""
    msg = EndTaskMessage(force=True, eos_timeout=10)
    dump = msg.model_dump()
    assert dump["message_type"] == "end_task"
    assert dump["data"]["force"] is True
    assert dump["data"]["eos_timeout"] == 10


def test_eos_message():
    """Test EosMessage"""
    msg = EosMessage()
    dump = msg.model_dump()
    assert dump["message_type"] == "eos"
    assert dump["data"] == {}


def test_set_task_message():
    """Test SetTaskMessage"""
    msg = SetTaskMessage(data={"config": "value"})
    dump = msg.model_dump()
    assert dump["message_type"] == "set_task"
    assert dump["data"] == {"config": "value"}


def test_get_task_message():
    """Test GetTaskMessage"""
    msg = GetTaskMessage()
    dump = msg.model_dump()
    assert dump["message_type"] == "get_task"
    assert dump["data"] == {}


def test_queue_status_message():
    """Test QueueStatusMessage"""
    lang = Language.get_or_create("es")
    msg = QueueStatusMessage(
        language=lang,
        current_queue_level_ms=100,
        max_queue_level_ms=1000
    )
    dump = msg.model_dump()
    assert dump == {
        "es": {
            "current_queue_level_ms": 100,
            "max_queue_level_ms": 1000
        }
    }
    assert "📊[es]:" in str(msg)
    assert "cur=100ms" in str(msg)
    assert "max=1000ms" in str(msg)


def test_queue_status_message_from_detected():
    """Test QueueStatusMessage creation from detected data"""
    kr = KnownRaw(
        type=KnownRawType.json,
        data={"es": {"current_queue_level_ms": 200, "max_queue_level_ms": 2000}}
    )
    msg = Message.from_detected(kr)
    assert isinstance(msg, QueueStatusMessage)
    assert msg.language.code == "es"
    assert msg.current_queue_level_ms == 200
    assert msg.max_queue_level_ms == 2000


def test_error_message_validation():
    """Test ErrorMessage with validation error"""
    kr = KnownRaw(
        type=KnownRawType.json,
        data={
            "message_type": "error",
            "data": {"code": "VALIDATION_ERROR", "desc": "Invalid input"}
        }
    )
    msg = Message.from_detected(kr)
    assert isinstance(msg, ErrorMessage)
    assert isinstance(msg._exc, ApiValidationError)
    with pytest.raises(ApiValidationError):
        msg.raise_()


def test_error_message_not_found():
    """Test ErrorMessage with not found error"""
    kr = KnownRaw(
        type=KnownRawType.json,
        data={
            "message_type": "error",
            "data": {"code": "NOT_FOUND", "desc": "Task not found"}
        }
    )
    msg = Message.from_detected(kr)
    assert isinstance(msg, ErrorMessage)
    assert isinstance(msg._exc, TaskNotFoundError)


def test_error_message_generic():
    """Test ErrorMessage with generic error"""
    kr = KnownRaw(
        type=KnownRawType.json,
        data={
            "message_type": "error",
            "data": {"code": "UNKNOWN", "desc": "Something went wrong"}
        }
    )
    msg = Message.from_detected(kr)
    assert isinstance(msg, ErrorMessage)
    assert isinstance(msg._exc, ApiError)


def test_unknown_message():
    """Test UnknownMessage"""
    kr = KnownRaw(
        type=KnownRawType.string,
        data="not json",
        exc=ValueError("parse error")
    )
    msg = UnknownMessage.create(kr)
    assert msg.raw_type == KnownRawType.string
    assert msg.raw_data == "not json"
    assert msg.error_info["type"] == "ValueError"
    assert "⚠️" in str(msg)


def test_unknown_message_bytes():
    """Test UnknownMessage with bytes data"""
    kr = KnownRaw(
        type=KnownRawType.binary,
        data=b"hello"
    )
    msg = UnknownMessage.create(kr)
    assert msg.raw_data == "hello"  # Decoded to string


def test_pipeline_timings_message():
    """Test PipelineTimingsMessage"""
    data = {
        "message_type": "pipeline_timings",
        "data": {
            "transcription_id": "123",
            "timings": {"step1": 0.1, "step2": 0.2}
        }
    }
    msg = PipelineTimingsMessage.model_validate(data)
    assert msg.transcription_id == "123"
    assert msg.timings == {"step1": 0.1, "step2": 0.2}

    dump = msg.model_dump()
    assert dump == data


def test_transcription_message():
    """Test TranscriptionMessage"""
    data = {
        "message_type": "partial_transcription",
        "data": {
            "transcription": {
                "transcription_id": "456",
                "language": "es",
                "text": "Hola mundo",
                "segments": [
                    {
                        "text": "Hola",
                        "start": 0.0,
                        "end": 0.5,
                        "start_timestamp": 1000.0,
                        "end_timestamp": 1500.0
                    }
                ]
            }
        }
    }
    msg = TranscriptionMessage.model_validate(data)
    assert msg.id_ == "456"
    assert msg.language.code == "es"
    assert msg.text == "Hola mundo"
    assert len(msg.segments) == 1
    assert msg.segments[0].text == "Hola"

    assert "🇪🇸es" in repr(msg)
    assert str(msg) == "Hola mundo"

    # Test dedup property
    assert "456" in msg.dedup

    # Test model_dump
    dump = msg.model_dump()
    assert dump == data


def test_transcription_message_from_detected():
    """Test TranscriptionMessage creation from detected data"""
    kr = KnownRaw(
        type=KnownRawType.json,
        data={
            "message_type": "translated_transcription",
            "data": {
                "transcription": {
                    "transcription_id": "789",
                    "language": "en",
                    "text": "Hello world",
                    "segments": []
                }
            }
        }
    )
    msg = Message.from_detected(kr)
    assert isinstance(msg, TranscriptionMessage)
    assert msg.type_ == Message.Type.TRANSLATED_TRANSCRIPTION
    assert msg.language.code == "en"


def test_current_task_message():
    """Test CurrentTaskMessage"""
    data = {
        "message_type": "current_task",
        "data": {"task": "config"}
    }
    msg = CurrentTaskMessage.model_validate(data)
    assert msg.data == {"task": "config"}

    dump = msg.model_dump()
    assert dump == data


def test_current_task_message_from_detected():
    """Test CurrentTaskMessage creation from detected data"""
    kr = KnownRaw(
        type=KnownRawType.json,
        data={
            "message_type": "current_task",
            "data": {"some": "data"}
        }
    )
    msg = Message.from_detected(kr)
    assert isinstance(msg, CurrentTaskMessage)
    assert msg.data == {"some": "data"}


def test_message_decode_nested_json():
    """Test Message.decode with nested JSON in data field"""
    raw = '{"data": "{\\"nested\\": \\"json\\"}", "message_type": "unknown"}'
    msg = Message.decode(raw)
    # Should be UnknownMessage since no pattern matches
    assert isinstance(msg, UnknownMessage)


def test_message_decode_non_json():
    """Test Message.decode with non-JSON input"""
    msg = Message.decode("not json")
    assert isinstance(msg, UnknownMessage)
    assert msg.raw_type == KnownRawType.unknown


def test_message_from_detected_exception():
    """Test Message.from_detected handles exceptions"""
    # Create a KnownRaw that will cause an exception during processing
    kr = KnownRaw(
        type=KnownRawType.json,
        data=None  # This will cause issues when accessed as dict
    )
    msg = Message.from_detected(kr)
    assert isinstance(msg, UnknownMessage)


def assert_dicts_identical(actual: dict, expected: dict, path: str = "root"):
    """Recursively check that two dicts are absolutely identical"""
    missing_keys = set(expected.keys()) - set(actual.keys())
    assert not missing_keys, f"Missing keys in actual at {path}: {missing_keys}"

    extra_keys = set(actual.keys()) - set(expected.keys())
    assert not extra_keys, f"Extra keys in actual at {path}: {extra_keys}"

    for key in expected.keys():
        expected_value = expected[key]
        actual_value = actual[key]
        current_path = f"{path}.{key}"

        assert type(actual_value) == type(expected_value), \
            f"Type mismatch at {current_path}: {type(actual_value).__name__} != {type(expected_value).__name__}"

        if isinstance(expected_value, dict):
            assert_dicts_identical(actual_value, expected_value, current_path)
        elif isinstance(expected_value, list):
            assert len(actual_value) == len(expected_value), \
                f"List length mismatch at {current_path}: {len(actual_value)} != {len(expected_value)}"
            for i, (act_item, exp_item) in enumerate(zip(actual_value, expected_value)):
                item_path = f"{current_path}[{i}]"
                if isinstance(exp_item, dict):
                    assert_dicts_identical(act_item, exp_item, item_path)
                elif isinstance(exp_item, list):
                    assert act_item == exp_item, f"List item mismatch at {item_path}"
                else:
                    assert act_item == exp_item, f"Value mismatch at {item_path}: {act_item} != {exp_item}"
        else:
            assert actual_value == expected_value, \
                f"Value mismatch at {current_path}: {actual_value!r} != {expected_value!r}"


def test_set_task_message_from_config_basic():
    """SetTaskMessage.from_config() should contain Config.to_dict()"""
    config = Config(source=SourceLang(lang=EN), targets=[TargetLang(lang=ES)])

    msg = SetTaskMessage.from_config(config)

    assert msg.type_ == Message.Type.SET_TASK
    assert "pipeline" in msg.data
    assert msg.data["pipeline"]["transcription"]["source_language"] == "en"
    assert msg.data["pipeline"]["translations"][0]["target_language"] == "es"


def test_set_task_message_preserves_modified_fields():
    """SetTaskMessage must contain ALL modified fields (set + unset)"""
    minimal_json = {
        "input_stream": {"content_type": "audio", "source": {"type": "ws", "sample_rate": 16000, "channels": 1}},
        "output_stream": {"content_type": "audio", "target": {"type": "ws", "sample_rate": 24000, "channels": 1}},
        "pipeline": {
            "transcription": {
                "source_language": "en",
                "segment_confirmation_silence_threshold": 0.7,
            },
            "translations": [{
                "target_language": "es",
                "translate_partial_transcriptions": False,
            }]
        }
    }

    config = Config.from_json(copy.deepcopy(minimal_json))

    # Modify SET field (was explicitly set)
    config.source.transcription.segment_confirmation_silence_threshold = 0.8

    # Modify UNSET field (was default False)
    config.source.transcription.only_confirm_by_silence = True

    msg = SetTaskMessage.from_config(config)

    # CHECK that BOTH changes are in msg.data
    assert msg.data["pipeline"]["transcription"]["segment_confirmation_silence_threshold"] == 0.8, \
        "Modified set field was not preserved in SetTaskMessage.data"
    assert msg.data["pipeline"]["transcription"]["only_confirm_by_silence"] == True, \
        "Modified unset field was not preserved in SetTaskMessage.data"


def test_set_task_message_roundtrip_with_benchmark_config():
    """SetTaskMessage with benchmark config should preserve all fields on roundtrip"""
    fixture_path = Path(__file__).parent / "fixtures" / "benchmark_config.json"
    original_json = json.loads(fixture_path.read_text())

    config = Config.from_json(copy.deepcopy(original_json))

    # Modify some fields
    config.translation_queue_configs.global_.auto_tempo = False
    config.translation_queue_configs.global_.min_tempo = 1.2

    msg = SetTaskMessage.from_config(config)

    # Verify modifications present
    assert msg.data["pipeline"]["translation_queue_configs"]["global"]["auto_tempo"] == False
    assert msg.data["pipeline"]["translation_queue_configs"]["global"]["min_tempo"] == 1.2

    # Roundtrip: recreate Config from message.data
    config2 = Config.from_json(copy.deepcopy(msg.data))
    msg2 = SetTaskMessage.from_config(config2)

    # Both messages should have identical data
    assert_dicts_identical(msg2.data, msg.data)
