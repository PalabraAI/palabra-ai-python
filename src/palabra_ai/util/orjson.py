"""Universal JSON encoder/decoder that never fails."""

from typing import Any

import orjson

from palabra_ai.util.logger import debug


def _default(obj: Any) -> Any:
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return str(obj)
    except Exception as e:
        debug("Failed to serialize object", exc_info=e)
        return str(obj)


def to_json(obj: Any, indent: bool = False, sort_keys: bool = True) -> bytes:
    """Convert anything to JSON bytes. Never fails."""
    opts = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
    if indent:
        opts |= orjson.OPT_INDENT_2
    if sort_keys:
        opts |= orjson.OPT_SORT_KEYS

    return orjson.dumps(
        obj,
        default=_default,
        option=opts,
    )


def from_json(data: str | bytes) -> Any:
    """Parse JSON from string or bytes."""
    return orjson.loads(data)
