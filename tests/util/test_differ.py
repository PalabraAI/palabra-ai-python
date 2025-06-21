"""Tests for dict subset checker."""

import pytest
from palabra_ai.util.differ import is_dict_subset


@pytest.mark.parametrize("subset,superset,expected", [
    # Basic subset cases - should pass
    ({}, {"a": 1}, True),
    ({"a": 1}, {"a": 1}, True),
    ({"a": 1}, {"a": 1, "b": 2}, True),
    ({"a": {"x": 1}}, {"a": {"x": 1, "y": 2}}, True),
    ({"a": {}}, {"a": {"b": {"c": 3}}}, True),
    ({"a": [1, 2, 3]}, {"a": [1, 2, 3]}, True),  # lists matter!

    # Lists with dicts - subset check
    ({"a": [{}]}, {"a": [{"x": 1}]}, True),
    ({"a": [{"x": 1}]}, {"a": [{"x": 1, "y": 2}]}, True),
    ({"a": [{}, {"b": 2}]}, {"a": [{"x": 1}, {"b": 2, "c": 3}]}, True),

    # Not subset cases - should fail
    ({"a": []}, {"a": [1, 2, 3]}, False),  # lists matter!
    ({"a": 1}, {"a": "hello"}, False),
    ({"a": 1}, {"a": None}, False),
    ({"a": 1}, {"a": 2}, False),
    ({"a": 1}, {}, False),
    ({"a": 1}, {"b": 1}, False),
    ({"a": 1, "b": 2}, {"a": 1}, False),
    ({"a": {"x": 1}}, {"a": {"x": 2}}, False),
    ({"a": [1, 2]}, {"a": [2, 1]}, False),  # order matters for lists

    # Lists with dicts - not subset
    ({"a": [{"x": 1}]}, {"a": [{"x": 2}]}, False),
    ({"a": [{"x": 1}, {"y": 2}]}, {"a": [{"x": 1}]}, False),  # different length
])
def test_is_dict_subset(subset, superset, expected):
    """Test is_dict_subset with various cases."""
    assert is_dict_subset(subset, superset) == expected


def test_is_dict_subset_type_error():
    """Test that TypeError is raised for non-dict arguments."""
    with pytest.raises(TypeError, match="Both arguments must be dictionaries"):
        is_dict_subset("not a dict", {})

    with pytest.raises(TypeError, match="Both arguments must be dictionaries"):
        is_dict_subset({}, "not a dict")

    with pytest.raises(TypeError, match="Both arguments must be dictionaries"):
        is_dict_subset("not", "dict")


def test_is_dict_subset_nested_complex():
    """Test complex nested structures."""
    subset = {
        "config": {
            "database": {
                "host": "localhost",
                "port": 5432
            },
            "cache": {
                "enabled": True
            },
            "something": [{}]
        }
    }

    superset = {
        "config": {
            "database": {
                "host": "localhost",
                "port": 5432,
                "user": "admin",
                "password": "secret"
            },
            "cache": {
                "enabled": True,
                "ttl": 3600
            },
            "logging": {
                "level": "INFO"
            },
            "something": [{"hello": "world"}]
        },
        "version": "1.0"
    }

    assert is_dict_subset(subset, superset) is True

    # Change one nested value
    subset["config"]["database"]["port"] = 3306
    assert is_dict_subset(subset, superset) is False

    # Test list with nested dicts
    assert is_dict_subset(
        {"items": [{"a": 1}, {"b": 2}]},
        {"items": [{"a": 1, "x": 10}, {"b": 2, "y": 20}]}
    ) is True

    # Test list with different dict values
    assert is_dict_subset(
        {"items": [{"a": 1}, {"b": 2}]},
        {"items": [{"a": 1}, {"b": 3}]}
    ) is False