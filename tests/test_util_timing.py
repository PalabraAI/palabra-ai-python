import time
from palabra_ai.util.timing import get_utc_ts


def test_get_utc_ts_returns_float():
    """Test that get_utc_ts returns a float timestamp"""
    result = get_utc_ts()
    assert isinstance(result, float)


def test_get_utc_ts_increasing():
    """Test that consecutive calls return increasing timestamps"""
    ts1 = get_utc_ts()
    time.sleep(0.001)  # Sleep 1ms
    ts2 = get_utc_ts()

    assert ts2 > ts1


def test_get_utc_ts_precision():
    """Test timestamp has reasonable precision"""
    ts = get_utc_ts()

    # Should be close to current time
    current_time = time.time()
    assert abs(ts - current_time) < 1.0  # Within 1 second
