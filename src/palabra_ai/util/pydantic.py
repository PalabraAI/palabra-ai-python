from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def mark_fields_as_set(model: BaseModel, paths: list[str]) -> None:
    """
    Universal utility to mark fields as 'set' in any pydantic model.

    This makes fields appear in serialization even with exclude_unset=True.
    Supports complex path navigation including nested objects and arrays.

    Args:
        model: Any pydantic BaseModel instance
        paths: List of materialized paths like:
            - "field" - simple field
            - "nested.field" - nested object field
            - "array[].field" - field in each array element
            - "deep[].nested[].field" - multi-level arrays

    Examples:
        mark_fields_as_set(config, [
            "source.transcription.enabled",
            "targets[].translation.voice_cloning",
            "queue_configs[].items[].value"
        ])
    """
    for path in paths:
        try:
            _navigate_and_mark(model, _parse_path(path))
        except (AttributeError, IndexError, TypeError):
            # Field may not exist in current configuration - skip silently
            continue


def _parse_path(path: str) -> list[str]:
    """Parse materialized path into parts, preserving array notation."""
    return path.split(".")


def _navigate_and_mark(obj: Any, path_parts: list[str]) -> None:
    """
    Recursively navigate object using path parts and mark final field as set.

    Args:
        obj: Current object being navigated
        path_parts: Remaining path parts to navigate
    """
    if not path_parts or obj is None:
        return

    current_part = path_parts[0]
    remaining_parts = path_parts[1:]

    # Check if this part represents an array
    if current_part.endswith("[]"):
        field_name = current_part[:-2]  # Remove "[]" suffix
        array_obj = getattr(obj, field_name, None)

        if array_obj is None:
            return

        # Handle different array types
        if isinstance(array_obj, list):
            # Iterate through list elements
            for item in array_obj:
                _navigate_and_mark(item, remaining_parts)
        else:
            # Single object wrapped as array
            _navigate_and_mark(array_obj, remaining_parts)

        # Mark the array field itself as set
        try:
            setattr(obj, field_name, array_obj)
        except (AttributeError, ValueError):
            # Field doesn't exist or can't be set - skip silently
            pass

    else:
        # Regular field navigation
        if not remaining_parts:
            # Final field - mark as set using self-assignment
            try:
                current_value = getattr(obj, current_part, None)
                setattr(obj, current_part, current_value)
            except (AttributeError, ValueError):
                # Field doesn't exist or can't be set - skip silently
                pass
        else:
            # Navigate deeper
            nested_obj = getattr(obj, current_part, None)
            if nested_obj is not None:
                _navigate_and_mark(nested_obj, remaining_parts)
                # Mark the nested object field as set too
                try:
                    setattr(obj, current_part, nested_obj)
                except (AttributeError, ValueError):
                    # Field doesn't exist or can't be set - skip silently
                    pass
