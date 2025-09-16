"""High-precision timing utilities for Palabra AI."""

import time


def get_utc_ts() -> float:
    return time.time_ns() / 1_000_000_000


def get_perf_ts() -> float:
    return time.perf_counter()
