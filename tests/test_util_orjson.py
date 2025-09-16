import pytest
from pydantic import BaseModel
from palabra_ai.util.orjson import to_json, from_json, _default


class PydanticModel(BaseModel):
    name: str
    value: int


class DictModel:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def dict(self):
        return {"name": self.name, "value": self.value}


class CustomObject:
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return f"CustomObject({self.data})"


def test_to_json_basic():
    """Test to_json with basic types"""
    assert to_json({"key": "value"}) == b'{"key":"value"}'
    assert to_json([1, 2, 3]) == b'[1,2,3]'
    assert to_json("string") == b'"string"'
    assert to_json(123) == b'123'
    assert to_json(True) == b'true'
    assert to_json(None) == b'null'


def test_to_json_indent():
    """Test to_json with indentation"""
    result = to_json({"key": "value"}, indent=True)
    assert b'{\n  "key": "value"\n}' in result


def test_to_json_sort_keys():
    """Test to_json with sorted keys"""
    result = to_json({"b": 2, "a": 1}, sort_keys=True)
    assert result == b'{"a":1,"b":2}'

    result = to_json({"b": 2, "a": 1}, sort_keys=False)
    # Order might vary, just check both keys are present
    assert b'"a":1' in result
    assert b'"b":2' in result


def test_from_json():
    """Test from_json with various inputs"""
    assert from_json('{"key": "value"}') == {"key": "value"}
    assert from_json(b'{"key": "value"}') == {"key": "value"}
    assert from_json('[1, 2, 3]') == [1, 2, 3]
    assert from_json('"string"') == "string"
    assert from_json('123') == 123
    assert from_json('true') is True
    assert from_json('null') is None


def test_default_memoryview():
    """Test _default with memoryview"""
    mv = memoryview(b"hello")
    assert _default(mv) == "hello"


def test_default_bytes():
    """Test _default with bytes and bytearray"""
    assert _default(b"hello") == "hello"
    assert _default(bytearray(b"hello")) == "hello"


def test_default_pydantic_model():
    """Test _default with Pydantic model"""
    model = PydanticModel(name="test", value=123)
    result = _default(model)
    assert result == {"name": "test", "value": 123}


def test_default_dict_model():
    """Test _default with object that has dict() method"""
    model = DictModel(name="test", value=456)
    result = _default(model)
    assert result == {"name": "test", "value": 456}


def test_default_custom_object():
    """Test _default with custom object (fallback to str)"""
    obj = CustomObject("data")
    result = _default(obj)
    assert result == "CustomObject(data)"


def test_default_exception_handling():
    """Test _default handles exceptions gracefully"""
    # Create object that raises exception on model_dump and dict
    class BadObject:
        def model_dump(self):
            raise ValueError("Cannot dump")

        def dict(self):
            raise ValueError("Cannot dict")

        def __str__(self):
            return "BadObject"

    obj = BadObject()
    # Should fall back to str() after exceptions
    result = _default(obj)
    assert result == "BadObject"


def test_to_json_with_complex_objects():
    """Test to_json with complex nested objects"""
    data = {
        "pydantic": PydanticModel(name="test", value=123),
        "dict_model": DictModel(name="test2", value=456),
        "custom": CustomObject("data"),
        "bytes": b"binary",
        "list": [1, 2, 3],
        "nested": {
            "key": "value"
        }
    }

    result = to_json(data)
    parsed = from_json(result)

    assert parsed["pydantic"] == {"name": "test", "value": 123}
    assert parsed["dict_model"] == {"name": "test2", "value": 456}
    assert "CustomObject(data)" in parsed["custom"]
    assert parsed["bytes"] == "binary"
    assert parsed["list"] == [1, 2, 3]
    assert parsed["nested"] == {"key": "value"}
