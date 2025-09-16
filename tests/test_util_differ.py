import pytest
from palabra_ai.util.differ import is_dict_subset

def test_is_dict_subset_empty():
    """Test empty dict is subset of any dict"""
    assert is_dict_subset({}, {})
    assert is_dict_subset({}, {"a": 1})
    assert is_dict_subset({}, {"a": 1, "b": 2})

def test_is_dict_subset_simple():
    """Test simple dict subset"""
    superset = {"a": 1, "b": 2, "c": 3}

    assert is_dict_subset({"a": 1}, superset)
    assert is_dict_subset({"a": 1, "b": 2}, superset)
    assert is_dict_subset({"a": 1, "b": 2, "c": 3}, superset)

def test_is_dict_subset_not_subset():
    """Test when not a subset"""
    superset = {"a": 1, "b": 2}

    # Missing key
    assert not is_dict_subset({"c": 3}, superset)

    # Wrong value
    assert not is_dict_subset({"a": 2}, superset)

    # Extra key
    assert not is_dict_subset({"a": 1, "b": 2, "c": 3}, superset)

def test_is_dict_subset_nested():
    """Test nested dict subset"""
    superset = {
        "a": 1,
        "b": {
            "x": 10,
            "y": 20,
            "z": {"deep": True}
        }
    }

    # Nested subset
    assert is_dict_subset({"b": {"x": 10}}, superset)
    assert is_dict_subset({"b": {"x": 10, "y": 20}}, superset)
    assert is_dict_subset({"b": {"z": {"deep": True}}}, superset)

    # Not subset - wrong nested value
    assert not is_dict_subset({"b": {"x": 11}}, superset)
    assert not is_dict_subset({"b": {"z": {"deep": False}}}, superset)

def test_is_dict_subset_with_lists():
    """Test dict subset with lists"""
    superset = {
        "a": [1, 2, 3],
        "b": ["x", "y", "z"]
    }

    # Exact list match
    assert is_dict_subset({"a": [1, 2, 3]}, superset)
    assert is_dict_subset({"b": ["x", "y", "z"]}, superset)

    # Different list length
    assert not is_dict_subset({"a": [1, 2]}, superset)
    assert not is_dict_subset({"a": [1, 2, 3, 4]}, superset)

    # Different list values
    assert not is_dict_subset({"a": [1, 2, 4]}, superset)
    assert not is_dict_subset({"b": ["x", "z", "y"]}, superset)  # Order matters

def test_is_dict_subset_list_with_dicts():
    """Test dict subset with lists containing dicts"""
    superset = {
        "items": [
            {"id": 1, "name": "a", "extra": "data"},
            {"id": 2, "name": "b", "extra": "data"}
        ]
    }

    # Subset of dicts in list
    subset = {
        "items": [
            {"id": 1, "name": "a"},
            {"id": 2, "name": "b"}
        ]
    }
    assert is_dict_subset(subset, superset)

    # Wrong value in nested dict
    subset_wrong = {
        "items": [
            {"id": 1, "name": "wrong"},
            {"id": 2, "name": "b"}
        ]
    }
    assert not is_dict_subset(subset_wrong, superset)

def test_is_dict_subset_type_errors():
    """Test type errors"""
    with pytest.raises(TypeError) as exc_info:
        is_dict_subset("not a dict", {})
    assert "Both arguments must be dictionaries" in str(exc_info.value)

    with pytest.raises(TypeError) as exc_info:
        is_dict_subset({}, "not a dict")
    assert "Both arguments must be dictionaries" in str(exc_info.value)

    with pytest.raises(TypeError) as exc_info:
        is_dict_subset([], [])
    assert "Both arguments must be dictionaries" in str(exc_info.value)
    """Test complex nested structure"""
    superset = {
        "config": {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "user": "admin",
                    "pass": "secret"
                }
            },
            "features": ["auth", "api", "ui"],
            "settings": [
                {"key": "timeout", "value": 30},
                {"key": "retries", "value": 3}
            ]
        }
    }

    # Deep nested subset
    subset = {
        "config": {
            "database": {
                "credentials": {
                    "user": "admin"
                }
            }
        }
    }
    assert is_dict_subset(subset, superset)

    # List subset with dict elements
    subset2 = {
        "config": {
            "settings": [
                {"key": "timeout"},
                {"key": "retries"}
            ]
        }
    }
    assert is_dict_subset(subset2, superset)
