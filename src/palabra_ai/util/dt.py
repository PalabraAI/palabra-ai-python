from datetime import UTC, datetime


def get_now_strftime() -> str:
    """Get current UTC time as a formatted string."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
