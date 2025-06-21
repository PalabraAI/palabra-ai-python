import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from palabra_ai.util.orjson import to_json, _default


class TestOrjson:
    def test_basic_types(self):
        data = {"key": "value", "number": 123, "list": [1, 2, 3]}

        json_bytes = to_json(data)
        assert isinstance(json_bytes, bytes)

        result = json.loads(json_bytes)
        assert result == data

    def test_special_types(self):
        data = {"path": Path("/tmp/test")}

        json_bytes = to_json(data)
        result = json.loads(json_bytes)
        assert result["path"] == "/tmp/test"

    def test_default_function(self):
        # With model_dump
        obj = MagicMock()
        obj.model_dump.return_value = {"key": "value"}

        result = _default(obj)
        assert result == {"key": "value"}
        obj.model_dump.assert_called_once()

        # With dict method
        obj = MagicMock()
        del obj.model_dump
        obj.dict.return_value = {"key": "value"}

        result = _default(obj)
        assert result == {"key": "value"}
        obj.dict.assert_called_once()

        # Fallback to str
        obj = object()
        result = _default(obj)
        assert isinstance(result, str)
        assert "object" in result

    def test_to_json_options(self):
        data = {"b": 2, "a": 1}

        # With indent
        result = to_json(data, indent=True, sort_keys=True)
        assert isinstance(result, bytes)
        assert b"\n" in result  # Indented

        # Without sort_keys
        result = to_json(data, indent=False, sort_keys=False)
        assert isinstance(result, bytes)
