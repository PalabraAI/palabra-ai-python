import pytest
from palabra_ai.model import LogData, RunResult


def test_log_data_creation():
    """Test LogData model creation"""
    log_data = LogData(
        version="1.0.0",
        sysinfo={"os": "darwin"},
        messages=[{"type": "test", "data": "example"}],
        start_ts=1234567890.0,
        cfg={"mode": "test"},
        log_file="test.log",
        trace_file="test.trace",
        debug=True,
        logs=["log1", "log2"]
    )

    assert log_data.version == "1.0.0"
    assert log_data.sysinfo["os"] == "darwin"
    assert len(log_data.messages) == 1
    assert log_data.start_ts == 1234567890.0
    assert log_data.cfg["mode"] == "test"
    assert log_data.log_file == "test.log"
    assert log_data.trace_file == "test.trace"
    assert log_data.debug is True
    assert len(log_data.logs) == 2


def test_run_result_success():
    """Test RunResult model for successful run"""
    result = RunResult(ok=True)

    assert result.ok is True
    assert result.exc is None
    assert result.log_data is None


def test_run_result_with_exception():
    """Test RunResult model with exception"""
    test_exception = ValueError("Test error")
    result = RunResult(ok=False, exc=test_exception)

    assert result.ok is False
    assert result.exc == test_exception
    assert result.log_data is None


def test_run_result_with_log_data():
    """Test RunResult model with log data"""
    log_data = LogData(
        version="1.0.0",
        sysinfo={},
        messages=[],
        start_ts=0.0,
        cfg={},
        log_file="",
        trace_file="",
        debug=False,
        logs=[]
    )

    result = RunResult(ok=True, log_data=log_data)

    assert result.ok is True
    assert result.exc is None
    assert result.log_data == log_data


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
    log_data = LogData(
        version="1.0.0",
        sysinfo={},
        messages=[],
        start_ts=0.0,
        cfg={},
        log_file="",
        trace_file="",
        debug=False,
        logs=[]
    )

    result = RunResult(ok=True, exc=None, log_data=log_data, eos=True)

    assert result.ok is True
    assert result.exc is None
    assert result.log_data == log_data
    assert result.eos is True