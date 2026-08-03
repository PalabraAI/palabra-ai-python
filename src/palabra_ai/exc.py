class PalabraError(Exception):
    """Base error for the Palabra client."""


class AuthError(PalabraError):
    """Missing or invalid API Key / token."""


class SessionError(PalabraError):
    """WebSocket connection or session failure."""


class TaskError(PalabraError):
    """The server rejected a command (``error`` message).

    Attributes:
        code: short server error code (e.g. ``VALIDATION_ERROR``).
        desc: human-readable description from the server.
    """

    def __init__(self, code: str, desc: str = ""):
        super().__init__(f"{code}: {desc}" if desc else code)
        self.code = code
        self.desc = desc


class NotReadyError(PalabraError):
    """The pipeline did not confirm the task in time."""


class ApiError(PalabraError):
    """A management REST API request (voices/glossaries) failed.

    Attributes:
        status: HTTP status code, or None for non-HTTP failures
            (network errors, voice processing failure, poll timeout).
    """

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status
