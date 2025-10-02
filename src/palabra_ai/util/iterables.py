import bisect
from typing import TypeVar

T = TypeVar("T")


def nearest_left(arr: list[T], x: T) -> tuple[int, T] | None:
    i = bisect.bisect_right(arr, x)
    if i == 0:
        return None
    return i - 1, arr[i - 1]
