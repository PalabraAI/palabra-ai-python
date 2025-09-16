import pytest
from palabra_ai.exc import (
    PalabraException,
    ConfigurationError,
    InvalidCredentialsError,
    NotSufficientFundsError,
    ApiError,
    ApiValidationError,
    TaskNotFoundError,
    unwrap_exceptions,
)


def test_palabra_exception():
    """Test base PalabraException"""
    exc = PalabraException("test error")
    assert str(exc) == "test error"
    assert isinstance(exc, Exception)


def test_configuration_error():
    """Test ConfigurationError"""
    exc = ConfigurationError("config error")
    assert str(exc) == "config error"
    assert isinstance(exc, PalabraException)


def test_invalid_credentials_error():
    """Test InvalidCredentialsError"""
    exc = InvalidCredentialsError("invalid creds")
    assert str(exc) == "invalid creds"
    assert isinstance(exc, PalabraException)


def test_not_sufficient_funds_error():
    """Test NotSufficientFundsError"""
    exc = NotSufficientFundsError("no funds")
    assert str(exc) == "no funds"
    assert isinstance(exc, PalabraException)


def test_api_error():
    """Test ApiError"""
    exc = ApiError("api error")
    assert str(exc) == "api error"
    assert isinstance(exc, PalabraException)


def test_api_validation_error():
    """Test ApiValidationError"""
    exc = ApiValidationError("validation error")
    assert str(exc) == "validation error"
    assert isinstance(exc, ApiError)
    assert isinstance(exc, PalabraException)


def test_task_not_found_error():
    """Test TaskNotFoundError"""
    exc = TaskNotFoundError("task not found")
    assert str(exc) == "task not found"
    assert isinstance(exc, ApiError)
    assert isinstance(exc, PalabraException)


def test_unwrap_exceptions_single():
    """Test unwrap_exceptions with a single exception"""
    exc = ValueError("test")
    result = unwrap_exceptions(exc)
    assert result == [exc]


def test_unwrap_exceptions_group():
    """Test unwrap_exceptions with ExceptionGroup"""
    exc1 = ValueError("test1")
    exc2 = RuntimeError("test2")
    exc3 = TypeError("test3")

    # Create nested ExceptionGroups
    group1 = ExceptionGroup("group1", [exc1, exc2])
    group2 = ExceptionGroup("group2", [group1, exc3])

    result = unwrap_exceptions(group2)
    assert len(result) == 3
    assert exc1 in result
    assert exc2 in result
    assert exc3 in result


def test_unwrap_exceptions_deeply_nested():
    """Test unwrap_exceptions with deeply nested ExceptionGroups"""
    exc1 = ValueError("test1")
    exc2 = RuntimeError("test2")

    # Create deeply nested structure
    group1 = ExceptionGroup("group1", [exc1])
    group2 = ExceptionGroup("group2", [group1])
    group3 = ExceptionGroup("group3", [group2, exc2])

    result = unwrap_exceptions(group3)
    assert len(result) == 2
    assert exc1 in result
    assert exc2 in result
