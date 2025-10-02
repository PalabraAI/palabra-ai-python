from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Union

import orjson
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from palabra_ai.enum import Channel, Direction, Kind
from palabra_ai.exc import ApiError, ApiValidationError, TaskNotFoundError
from palabra_ai.lang import Language
from palabra_ai.util.logger import debug
from palabra_ai.util.orjson import from_json
from palabra_ai.util.timing import get_perf_ts, get_utc_ts

if TYPE_CHECKING:
    from palabra_ai.config import Config


class KnownRawType(StrEnum):
    null = "null"
    binary = "binary"
    string = "string"
    json = "json"
    unknown = "unknown"


@dataclass
class Dbg:
    kind: Kind | None
    ch: Channel | None
    dir: Direction | None
    dawn_ts: float | None = field(default=None)  # ts from global start
    perf_ts: float = field(default_factory=get_perf_ts)
    utc_ts: float = field(default_factory=get_utc_ts)
    idx: int | None = field(default=None)
    num: int | None = field(default=None)
    dur_s: float | None = field(default=None)
    rms_db: float | None = field(default=None)

    def calc_dawn_ts(self, global_start_ts: float | None):
        if global_start_ts is not None:
            self.dawn_ts = self.perf_ts - global_start_ts

    @classmethod
    def empty(cls):
        """Create an empty debug object"""
        return cls(kind=None, ch=None, dir=None)

    @property
    def delta(self) -> float:
        return get_perf_ts() - self.perf_ts

    @classmethod
    def now_utc_ts(cls):
        return get_utc_ts()

    @classmethod
    def now_perf_ts(cls):
        return get_perf_ts()


@dataclass
class IoEvent:
    head: "Dbg"
    body: str | bytes | dict | list
    tid: str | None = field(default=None)
    mtype: str | None = field(default=None)

    def __post_init__(self):
        if isinstance(self.body, bytes):
            self.body = self.body.decode("utf-8")
        self.convert_raw_to_body()

        self.mtype = self.body.get("message_type")
        if self.mtype == "output_audio_data":
            self.tid = self.body.get("data", {}).get("transcription_id")
        elif self.mtype in {
            "partial_transcription",
            "validated_transcription",
            "translated_transcription",
            "partial_translated_transcription",
        }:
            self.tid = (
                self.body.get("data", {})
                .get("transcription", {})
                .get("transcription_id")
            )

    def convert_raw_to_body(self):
        self.body = from_json(self.body) if self.body else None
        if "data" in self.body and isinstance(self.body["data"], str):
            try:
                self.body["data"] = from_json(self.body["data"])
            except orjson.JSONDecodeError:
                debug("Failed to decode nested JSON in 'data' field")


@dataclass
class KnownRaw:
    type: KnownRawType
    data: str | bytes | dict | None
    exc: Exception | None = None


class Message(BaseModel):
    """Base class for all message types"""

    type_: "Message.Type" = Field(alias="message_type")
    _known_raw: KnownRaw | None = PrivateAttr(default=None)
    _dbg: Dbg | None = PrivateAttr(default=None)

    class Type(StrEnum):
        PARTIAL_TRANSCRIPTION = "partial_transcription"
        TRANSLATED_TRANSCRIPTION = "translated_transcription"
        VALIDATED_TRANSCRIPTION = "validated_transcription"
        PARTIAL_TRANSLATED_TRANSCRIPTION = "partial_translated_transcription"
        PIPELINE_TIMINGS = "pipeline_timings"
        ERROR = "error"  # For error messages
        END_TASK = "end_task"  # For end_task messages
        SET_TASK = "set_task"  # For set_task messages
        GET_TASK = "get_task"  # For get_task messages
        CURRENT_TASK = "current_task"  # For current_task messages
        EOS = "eos"  # End of stream marker
        _QUEUE_LEVEL = "queue_level"
        _EMPTY = "empty"  # For empty {} messages
        _UNKNOWN = "unknown"  # For unrecognized message formats

    TRANSCRIPTION_TYPES: ClassVar[set[Type]] = {
        Type.PARTIAL_TRANSCRIPTION,
        Type.TRANSLATED_TRANSCRIPTION,
        Type.VALIDATED_TRANSCRIPTION,
        Type.PARTIAL_TRANSLATED_TRANSCRIPTION,
    }

    IN_PROCESS_TYPES: ClassVar[set[Type]] = TRANSCRIPTION_TYPES

    ALLOWED_TYPES: ClassVar[set[Type]] = {Type.PIPELINE_TIMINGS} | TRANSCRIPTION_TYPES

    STR_TRANSCRIPTION_TYPES: ClassVar[set[str]] = {
        mt.value for mt in TRANSCRIPTION_TYPES
    }

    @property
    def dbg_delta(self) -> str:
        if self._dbg:
            return str(self._dbg.delta)
        return "n/a"

    @classmethod
    def get_transcription_message_types(cls) -> set["Message.Type"]:
        """Get set of all transcription message types"""
        return {
            Message.Type.PARTIAL_TRANSCRIPTION,
            Message.Type.TRANSLATED_TRANSCRIPTION,
            Message.Type.VALIDATED_TRANSCRIPTION,
            Message.Type.PARTIAL_TRANSLATED_TRANSCRIPTION,
        }

    @classmethod
    def get_allowed_message_types(cls) -> set["Message.Type"]:
        return {Message.Type.PIPELINE_TIMINGS} | cls.get_transcription_message_types()

    @classmethod
    def from_detected(
        cls, known_raw: KnownRaw
    ) -> Union[
        "EmptyMessage",
        "QueueStatusMessage",
        "PipelineTimingsMessage",
        "TranscriptionMessage",
        "UnknownMessage",
        "ErrorMessage",
        "CurrentTaskMessage",
        "EosMessage",
    ]:
        """Factory method to create appropriate message type using pattern matching"""
        data = known_raw.data
        try:
            # Parse nested JSON in 'data' field if present
            if (
                isinstance(data, dict)
                and "data" in data
                and isinstance(data["data"], str)
            ):
                try:
                    data["data"] = from_json(data["data"])
                except orjson.JSONDecodeError:
                    debug("Failed to decode nested JSON in 'data' field")

            match data:
                # Empty message first - exactly empty dict
                case dict() if len(data) == 0:
                    return EmptyMessage.create(known_raw)

                # TranscriptionMessage messages
                case {"message_type": msg_type, "data": {"transcription": _}} if (
                    msg_type in cls.STR_TRANSCRIPTION_TYPES
                ):
                    return TranscriptionMessage.create(known_raw)

                # Error
                case {"message_type": Message.Type.ERROR.value, "data": _}:
                    return ErrorMessage.create(known_raw)

                # Pipeline timings
                case {"message_type": Message.Type.PIPELINE_TIMINGS.value, "data": _}:
                    return PipelineTimingsMessage.create(known_raw)

                case {"message_type": Message.Type.CURRENT_TASK.value, "data": _}:
                    return CurrentTaskMessage.create(known_raw)

                case {"message_type": Message.Type.EOS.value, "data": _}:
                    return EosMessage.create(known_raw)

                # Queue status: non-empty dict without message_type
                case dict() as d if d and "message_type" not in d and len(d) == 1:
                    [(lang, val)] = d.items()
                    match val:
                        case {
                            "current_queue_level_ms": int() as current,  # noqa: F841
                            "max_queue_level_ms": int() as max_,  # noqa: F841
                        } if len(val) == 2:
                            return QueueStatusMessage.create(known_raw)
                        case _:
                            debug(
                                f"Invalid queue status format. Expected {{current_queue_level_ms: int, max_queue_level_ms: int}}. Got: {val}"
                            )
                            return UnknownMessage.create(known_raw)

                # Unknown format
                case _:
                    debug(f"Unknown message format: {known_raw}")
                    return UnknownMessage.create(known_raw)

        except Exception as e:
            debug(f"Failed to parse message: {e}. Data: {known_raw}")
            return UnknownMessage.create(known_raw)

    @classmethod
    def create(cls, known_raw: KnownRaw) -> "Message":
        """Create a message instance from KnownRaw"""
        obj = cls.model_validate(known_raw.data)
        obj._known_raw = known_raw
        return obj

    @classmethod
    def detect(cls, raw_msg: str | bytes | None) -> KnownRaw:
        match raw_msg:
            case None:
                return KnownRaw(KnownRawType.null, None)

            case bytes() as b if b.startswith(b"{") and b.endswith(b"}"):
                try:
                    return KnownRaw(KnownRawType.json, from_json(b))
                except Exception as e:
                    return KnownRaw(KnownRawType.binary, b, e)

            case str() as s if s.startswith("{") and s.endswith("}"):
                try:
                    return KnownRaw(KnownRawType.json, from_json(s))
                except Exception as e:
                    return KnownRaw(KnownRawType.string, s, e)

            case _:
                return KnownRaw(KnownRawType.unknown, raw_msg)
        return KnownRaw(KnownRawType.unknown, raw_msg)

    @classmethod
    def decode(cls, raw_msg: str | bytes | None) -> "Message":
        # debug(raw_msg)
        known_msg = cls.detect(raw_msg)
        # debug(known_msg)
        if known_msg.type == KnownRawType.json:
            return cls.from_detected(known_msg)
        else:
            return UnknownMessage.create(known_msg)


class EmptyMessage(Message):
    """Empty message"""

    type_: Message.Type = Field(default=Message.Type._EMPTY, alias="message_type")

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return {}

    def __str__(self) -> str:
        return "⚪"


class EndTaskMessage(Message):
    ### {"message_type": "end_task", "data": {"force": True}}
    """End task message"""

    type_: Message.Type = Field(default=Message.Type.END_TASK, alias="message_type")
    force: bool = Field(default=False, description="Force end the task")
    eos_timeout: int | None = Field(
        default=5, description="Timeout for end of stream in seconds"
    )

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return {
            "message_type": self.type_.value,
            "data": {"force": self.force, "eos_timeout": self.eos_timeout},
        }


class EosMessage(Message):
    """End of stream message"""

    type_: Message.Type = Field(default=Message.Type.EOS, alias="message_type")

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return {"message_type": self.type_.value, "data": {}}


class SetTaskMessage(Message):
    """Set task message"""

    type_: Message.Type = Field(default=Message.Type.SET_TASK, alias="message_type")
    data: dict[str, Any] = Field(
        default_factory=dict, description="Task configuration data"
    )

    @classmethod
    def from_config(cls, cfg: "Config"):
        return cls(data=cfg.to_dict())

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return {
            "message_type": self.type_.value,
            "data": self.data,
        }


class GetTaskMessage(Message):
    """Get task message"""

    type_: Message.Type = Field(default=Message.Type.GET_TASK, alias="message_type")

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return {
            "message_type": self.type_.value,
            "data": {},
        }


class QueueStatusMessage(Message):
    """Queue status message with language-specific queue data"""

    type_: Message.Type = Field(default=Message.Type._QUEUE_LEVEL, alias="message_type")
    language: Language
    current_queue_level_ms: int
    max_queue_level_ms: int

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return {
            self.language.code: {
                "current_queue_level_ms": self.current_queue_level_ms,
                "max_queue_level_ms": self.max_queue_level_ms,
            }
        }

    @classmethod
    def create(cls, known_raw: KnownRaw) -> "QueueStatusMessage":
        """Create QueueStatusMessage from KnownRaw with proper data conversion"""
        if not isinstance(known_raw.data, dict):
            raise ValueError("QueueStatusMessage requires a dictionary data format")

        lang_code, queue_data = next(iter(known_raw.data.items()))
        obj = cls.model_validate(
            {
                "language": Language.get_or_create(lang_code),
                "current_queue_level_ms": queue_data["current_queue_level_ms"],
                "max_queue_level_ms": queue_data["max_queue_level_ms"],
            }
        )
        obj._known_raw = known_raw
        return obj

    def __str__(self) -> str:
        return (
            f"📊[{self.language.code}]: "
            f"cur={self.current_queue_level_ms}ms, "
            f"max={self.max_queue_level_ms}ms"
        )


class ErrorMessage(Message):
    raw: Any
    data: dict
    _exc: ApiError | None = PrivateAttr(default=None)

    @classmethod
    def create(cls, known_raw: KnownRaw) -> "ErrorMessage":
        """Create ErrorMessage from KnownRaw with proper data conversion"""
        obj = cls(
            message_type=Message.Type.ERROR, raw=known_raw, data={"raw": known_raw}
        )
        obj._known_raw = known_raw
        match known_raw.data:
            case {"data": {"code": "VALIDATION_ERROR", "desc": desc}}:
                # Format validation errors properly
                if isinstance(desc, list):
                    # Pydantic-style validation errors (list of dicts)
                    error_msgs = []
                    for err in desc:
                        if isinstance(err, dict):
                            loc = err.get("loc", [])
                            msg = err.get("msg", "validation error")
                            loc_str = (
                                " -> ".join(str(_loc) for _loc in loc)
                                if loc
                                else "unknown"
                            )
                            error_msgs.append(f"{loc_str}: {msg}")
                    error_str = "; ".join(error_msgs)
                else:
                    error_str = str(desc)
                obj._exc = ApiValidationError(error_str)
                obj.data = known_raw.data
            case {"data": {"code": "NOT_FOUND", "desc": desc}}:
                obj._exc = TaskNotFoundError(str(desc))
                obj.data = known_raw.data
            case _:
                obj._exc = ApiError(str(known_raw.data))
                print(f"Not a dict: {type(known_raw).__name__}")
        return obj

    def raise_(self):
        raise self._exc or ApiError("Unknown error occurred")


class UnknownMessage(Message):
    """Unknown/unrecognized message format"""

    raw_type: KnownRawType
    raw_data: str | dict | None  # Already processed data
    error_info: dict[str, Any] | None = None

    @classmethod
    def create(cls, known_raw: KnownRaw) -> "UnknownMessage":
        """Create UnknownMessage from KnownRaw with proper data conversion"""
        # Handle bytes data
        data = known_raw.data
        if isinstance(data, bytes):
            try:
                # Try to decode as UTF-8 string first
                data = data.decode("utf-8")
            except UnicodeDecodeError:
                # If fails, encode as hex
                data = data.hex()

        # Handle exception
        error_info = None
        if known_raw.exc is not None:
            error_info = {
                "type": type(known_raw.exc).__name__,
                "message": str(known_raw.exc),
                "args": known_raw.exc.args,
            }

        obj = cls(
            message_type=Message.Type._UNKNOWN,
            raw_type=known_raw.type,
            raw_data=data,
            error_info=error_info,
        )
        obj._known_raw = known_raw
        return obj

    def model_dump(self, **kwargs) -> Any:
        return self.raw_data

    def __str__(self) -> str:
        return f"⚠️[{self.raw_type},{len(self.raw_data)}]: {str(self.raw_data)[:100]}{self.error_info}"


class PipelineTimingsMessage(Message):
    """Pipeline timing information"""

    type_: Message.Type = Message.Type.PIPELINE_TIMINGS
    transcription_id: str
    timings: dict[str, float]

    @model_validator(mode="before")
    @classmethod
    def extract_from_nested(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "data" in values and "message_type" in values:
            data = values["data"]
            return {
                "message_type": values["message_type"],
                "transcription_id": data["transcription_id"],
                "timings": data["timings"],
            }
        return values

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return {
            "message_type": self.type_.value,
            "data": {
                "transcription_id": self.transcription_id,
                "timings": self.timings,
            },
        }


class TranscriptionSegment(BaseModel):
    text: str
    start: float
    end: float
    start_timestamp: float
    end_timestamp: float | None = None


class TranscriptionMessage(Message):
    """Transcription message"""

    type_: Message.Type = Field(alias="message_type")
    id_: str = Field(alias="transcription_id")
    text: str
    language: Language
    segments: list[TranscriptionSegment]

    model_config = ConfigDict(populate_by_name=True)

    @property
    def dedup(self) -> str:
        """Deduplication key for this message"""
        return f"{self.id_} {repr(self)}"

    @model_validator(mode="before")
    @classmethod
    def extract_from_nested(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Extract data from nested API structure"""
        if "data" in values and "message_type" in values:
            # Extract transcription data
            transcription = values["data"]["transcription"]
            # Convert language string to Language object
            lang_code = transcription["language"]
            return {
                "message_type": values["message_type"],
                "transcription_id": transcription["transcription_id"],
                "language": Language.get_or_create(lang_code),
                "segments": transcription["segments"],
                "text": transcription["text"],
            }
        return values

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump to nested API format"""
        segments_data = [seg.model_dump() for seg in self.segments]

        return {
            "message_type": self.type_.value,
            "data": {
                "transcription": {
                    "transcription_id": self.id_,
                    "language": self.language.code,
                    "text": self.text,
                    "segments": segments_data,
                }
            },
        }

    def __repr__(self) -> str:
        return f"{self.language.flag}{self.language.code} [{self.type_}]: {self.text}"

    def __str__(self) -> str:
        return self.text


class CurrentTaskMessage(Message):
    """Current task message"""

    type_: Message.Type = Field(default=Message.Type.CURRENT_TASK, alias="message_type")
    data: dict[str, Any] = Field(default_factory=dict, description="Current task data")

    @model_validator(mode="before")
    @classmethod
    def extract_from_nested(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "data" in values and "message_type" in values:
            return {
                "message_type": values["message_type"],
                "data": values["data"],
            }
        return values

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return {
            "message_type": self.type_.value,
            "data": self.data,
        }
