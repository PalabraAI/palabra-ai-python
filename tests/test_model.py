import pytest
from palabra_ai.model import LogData, RunResult


def test_log_data_creation():
    """Test LogData model creation without messages field"""
    log_data = LogData(
        version="1.0.0",
        sysinfo={"os": "darwin"},
        start_ts=1234567890.0,
        cfg={"mode": "test"},
        log_file="test.log",
        trace_file="test.trace",
        debug=True,
        logs=["log1", "log2"]
    )

    assert log_data.version == "1.0.0"
    assert log_data.sysinfo["os"] == "darwin"
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


def test_run_result_with_exception():
    """Test RunResult model with exception"""
    test_exception = ValueError("Test error")
    result = RunResult(ok=False, exc=test_exception)

    assert result.ok is False
    assert result.exc == test_exception


def test_run_result_with_log_data():
    """Test RunResult model with log data"""
    log_data = LogData(
        version="1.0.0",
        sysinfo={},
        start_ts=0.0,
        cfg={},
        log_file="",
        trace_file="",
        debug=False,
        logs=[]
    )

    result = RunResult(ok=True)

    assert result.ok is True
    assert result.exc is None


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


# ═══════════════════════════════════════════════════
# NEW TESTS: LogData without messages field (TDD - RED)
# ═══════════════════════════════════════════════════

def test_log_data_does_not_have_messages_field():
    """Test that LogData does not have messages attribute"""
    log_data = LogData(
        version="0.5.13",
        sysinfo={},
        start_ts=0.0,
        cfg={},
        log_file="",
        trace_file="",
        debug=False,
        logs=[]
    )

    assert not hasattr(log_data, "messages")


def test_log_data_serialization_without_messages():
    """Test that LogData.model_dump() does not include messages"""
    log_data = LogData(
        version="0.5.13",
        sysinfo={},
        start_ts=0.0,
        cfg={},
        log_file="",
        trace_file="",
        debug=False,
        logs=[]
    )

    dumped = log_data.model_dump()
    assert "messages" not in dumped