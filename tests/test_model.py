import pytest
from palabra_ai.model import RunResult


def test_run_result_success():
    """Test RunResult model for successful run"""
    result = RunResult(ok=True)

    assert result.ok is True
    assert result.exc is None


def test_run_result_with_exception():
    """Test RunResult model with exception"""
    test_exception = ValueError("Test error")
    result = RunResult(ok=False, exc=test_exception)

    assert result.ok is False
    assert result.exc == test_exception


def test_run_result_arbitrary_types():
    """Test RunResult allows arbitrary types for exc field"""
    # Test with different exception types
    custom_exception = Exception("Custom error")
    result = RunResult(ok=False, exc=custom_exception)

    assert result.exc == custom_exception
    assert isinstance(result.exc, Exception)


def test_run_result_eos_field_default():
    """Test RunResult eos field defaults to False"""
    result = RunResult(ok=True)

    assert hasattr(result, 'eos')
    assert result.eos is False


def test_run_result_eos_field_true():
    """Test RunResult eos field can be set to True"""
    result = RunResult(ok=True, eos=True)

    assert result.eos is True


def test_run_result_eos_field_with_all_params():
    """Test RunResult eos field works with all parameters"""
    result = RunResult(ok=True, exc=None, eos=True)

    assert result.ok is True
    assert result.exc is None
    assert result.eos is True