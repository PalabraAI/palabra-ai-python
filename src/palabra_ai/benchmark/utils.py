from typing import Any, List, Tuple, Union

Container = Union[dict[str, Any], list[Any]]

def _join(prefix: str, segment: str) -> str:
    return segment if not prefix else f"{prefix}.{segment}"

def _format_value(v: Any) -> str:
    if v is True:
        return "{true}"
    elif v is False:
        return "{false}"
    elif v is None:
        return "{null}"
    return repr(v)

def flatten_container_to_paths(container: Container, prefix: str = "") -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []

    if isinstance(container, dict):
        if not container:
            out.append((prefix, {}))
            return out
        for k, v in container.items():
            path = _join(prefix, str(k))
            if isinstance(v, dict):
                if v:
                    out.extend(flatten_container_to_paths(v, path))
                else:
                    out.append((path, {}))
            elif isinstance(v, list):
                if v:
                    out.extend(flatten_container_to_paths(v, path))
                else:
                    out.append((path, []))
            else:
                out.append((path, v))
        return out

    if isinstance(container, list):
        if not container:
            out.append((prefix, []))
            return out
        for i, v in enumerate(container):
            path = _join(prefix, str(i))
            if isinstance(v, dict):
                if v:
                    out.extend(flatten_container_to_paths(v, path))
                else:
                    out.append((path, {}))
            elif isinstance(v, list):
                if v:
                    out.extend(flatten_container_to_paths(v, path))
                else:
                    out.append((path, []))
            else:
                out.append((path, v))
        return out

    out.append((prefix, container))
    return out
