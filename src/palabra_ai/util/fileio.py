from pathlib import Path
from typing import Any

from palabra_ai.util.logger import warning
from palabra_ai.util.orjson import to_json


def save_json(
    path: Path | None, obj: Any, indent: bool = False, sort_keys: bool = True
) -> None:
    """Save object as JSON file."""
    if path is None:
        warning(f"save_json: path is None, skipping save {obj=}")
        return
    path.write_bytes(to_json(obj, indent=indent, sort_keys=sort_keys))


def save_text(path: Path | None, text: str) -> None:
    """Save text to file."""
    if path is None:
        warning(
            f"save_text: path is None, skipping save {text[:30] if text else '???'}..."
        )
        return
    path.write_text(text)
