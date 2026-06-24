class PalabraError(Exception):
    """Base error for the Palabra client."""


class AuthError(PalabraError):
    """Missing or invalid credentials / token."""


class SessionError(PalabraError):
    """Session creation or management failed."""


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
