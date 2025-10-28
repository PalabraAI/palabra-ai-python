import pytest
from pydantic import BaseModel, Field
from palabra_ai.util.pydantic import mark_fields_as_set


class NestedModel(BaseModel):
    enabled: bool = True
    value: str = "default"


class ArrayItem(BaseModel):
    name: str = "item"
    config: NestedModel = Field(default_factory=NestedModel)


class DeepArrayItem(BaseModel):
    items: list[ArrayItem] = Field(default_factory=list)


class TestModel(BaseModel):
    # Simple fields
    simple_field: str = "default"
    simple_bool: bool = False

    # Nested object
    nested: NestedModel = Field(default_factory=NestedModel)

    # Arrays
    array_items: list[ArrayItem] = Field(default_factory=list)

    # Deep nested arrays
    deep_array: list[DeepArrayItem] = Field(default_factory=list)


def test_mark_simple_field():
    """Test marking simple fields as set"""
    model = TestModel()

    # Initially field should not be in __pydantic_fields_set__
    assert "simple_field" not in model.__pydantic_fields_set__
    assert "simple_bool" not in model.__pydantic_fields_set__

    # Mark fields as set
    mark_fields_as_set(model, ["simple_field", "simple_bool"])

    # Now fields should be marked as set
    assert "simple_field" in model.__pydantic_fields_set__
    assert "simple_bool" in model.__pydantic_fields_set__

    # Verify exclude_unset behavior
    data = model.model_dump(exclude_unset=True)
    assert "simple_field" in data
    assert "simple_bool" in data
    assert data["simple_field"] == "default"
    assert data["simple_bool"] == False


def test_mark_nested_field():
    """Test marking nested object fields as set"""
    model = TestModel()

    # Mark nested field as set
    mark_fields_as_set(model, ["nested.enabled", "nested.value"])

    # Verify serialization includes nested fields
    data = model.model_dump(exclude_unset=True)
    assert "nested" in data
    assert "enabled" in data["nested"]
    assert "value" in data["nested"]
    assert data["nested"]["enabled"] == True
    assert data["nested"]["value"] == "default"


def test_mark_array_fields():
    """Test marking fields in array elements"""
    model = TestModel()

    # Add some items to array
    model.array_items = [ArrayItem(), ArrayItem(name="custom")]

    # Mark array field as set
    mark_fields_as_set(model, ["array_items[].name", "array_items[].config.enabled"])

    # Verify serialization includes array fields
    data = model.model_dump(exclude_unset=True)
    assert "array_items" in data
    assert len(data["array_items"]) == 2

    for item in data["array_items"]:
        assert "name" in item
        assert "config" in item
        assert "enabled" in item["config"]


def test_mark_deep_nested_arrays():
    """Test marking fields in deeply nested arrays"""
    model = TestModel()

    # Create deep nested structure
    deep_item = DeepArrayItem()
    deep_item.items = [ArrayItem(name="deep1"), ArrayItem(name="deep2")]
    model.deep_array = [deep_item]

    # Mark deep nested fields as set
    mark_fields_as_set(model, ["deep_array[].items[].name", "deep_array[].items[].config.value"])

    # Verify serialization includes deep nested fields
    data = model.model_dump(exclude_unset=True)
    assert "deep_array" in data
    assert len(data["deep_array"]) == 1
    assert "items" in data["deep_array"][0]
    assert len(data["deep_array"][0]["items"]) == 2

    for item in data["deep_array"][0]["items"]:
        assert "name" in item
        assert "config" in item
        assert "value" in item["config"]


def test_nonexistent_fields_ignored():
    """Test that nonexistent fields are silently ignored"""
    model = TestModel()

    # These paths don't exist but should not raise errors
    mark_fields_as_set(model, [
        "nonexistent_field",
        "nested.nonexistent",
        "array_items[].nonexistent",
        "deep_array[].nonexistent[].field"
    ])

    # Should not crash and should not affect existing fields
    data = model.model_dump(exclude_unset=True)
    # Only explicitly set fields should appear


def test_empty_arrays_handled():
    """Test that empty arrays are handled gracefully"""
    model = TestModel()

    # Arrays are empty by default
    assert len(model.array_items) == 0
    assert len(model.deep_array) == 0

    # Mark array fields - should not crash
    mark_fields_as_set(model, ["array_items[].name", "deep_array[].items[].name"])

    # Arrays themselves should be marked as set
    data = model.model_dump(exclude_unset=True)
    assert "array_items" in data
    assert "deep_array" in data
    assert data["array_items"] == []
    assert data["deep_array"] == []


def test_single_object_as_array():
    """Test handling single object when expecting array"""
    model = TestModel()

    # Create model with single item instead of list
    class SingleItemModel(BaseModel):
        item: ArrayItem = Field(default_factory=ArrayItem)

    single_model = SingleItemModel()

    # Mark as if it's an array (should handle gracefully)
    mark_fields_as_set(single_model, ["item[].name"])

    # Should mark the single item field
    data = single_model.model_dump(exclude_unset=True)
    assert "item" in data


def test_complex_mixed_paths():
    """Test complex combination of different path types"""
    model = TestModel()

    # Set up complex structure
    model.array_items = [ArrayItem(name="item1"), ArrayItem(name="item2")]
    deep_item = DeepArrayItem()
    deep_item.items = [ArrayItem(name="deep")]
    model.deep_array = [deep_item]

    # Mark various combinations
    paths = [
        "simple_field",
        "nested.enabled",
        "array_items[].name",
        "array_items[].config.value",
        "deep_array[].items[].config.enabled"
    ]

    mark_fields_as_set(model, paths)

    # Verify all paths work correctly
    data = model.model_dump(exclude_unset=True)

    # Simple field
    assert "simple_field" in data

    # Nested field
    assert "nested" in data
    assert "enabled" in data["nested"]

    # Array fields
    assert "array_items" in data
    for item in data["array_items"]:
        assert "name" in item
        assert "config" in item
        assert "value" in item["config"]

    # Deep nested fields
    assert "deep_array" in data
    for deep_item in data["deep_array"]:
        assert "items" in deep_item
        for item in deep_item["items"]:
            assert "config" in item
            assert "enabled" in item["config"]


def test_universal_with_different_models():
    """Test that utility works with any pydantic model"""
    class SimpleModel(BaseModel):
        name: str = "test"
        count: int = 0

    class ComplexModel(BaseModel):
        title: str = "complex"
        items: list[SimpleModel] = Field(default_factory=list)
        config: SimpleModel = Field(default_factory=SimpleModel)

    # Test with different model types
    simple = SimpleModel()
    complex_model = ComplexModel()
    complex_model.items = [SimpleModel(name="item1"), SimpleModel(name="item2")]

    # Mark fields in different models
    mark_fields_as_set(simple, ["name", "count"])
    mark_fields_as_set(complex_model, [
        "title",
        "items[].name",
        "config.count"
    ])

    # Verify both models work
    simple_data = simple.model_dump(exclude_unset=True)
    assert "name" in simple_data
    assert "count" in simple_data

    complex_data = complex_model.model_dump(exclude_unset=True)
    assert "title" in complex_data
    assert "items" in complex_data
    assert "config" in complex_data
    assert "count" in complex_data["config"]

    for item in complex_data["items"]:
        assert "name" in item