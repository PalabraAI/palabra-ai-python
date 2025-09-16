class PalabraException(Exception):
    pass


class ConfigurationError(PalabraException):
    """Raised when there is a configuration error."""

    pass


class InvalidCredentialsError(PalabraException):  # TODO
    """Raised when credentials are invalid or missing."""

    pass


class NotSufficientFundsError(PalabraException):  # TODO
    """Raised when there are not enough funds to perform an operation."""

    pass


class ApiError(PalabraException):
    """Base class for API-related errors."""

    ...


class ApiValidationError(ApiError):
    """Raised when the API response is invalid or does not match the expected schema."""

    ...


class TaskNotFoundError(ApiError): ...


def unwrap_exceptions(exc):
    """Recursively unwrap all non-ExceptionGroup exceptions into a flat list"""
    if isinstance(exc, ExceptionGroup):
        result = []
        for e in exc.exceptions:
            result.extend(unwrap_exceptions(e))
        return result
    return [exc]
