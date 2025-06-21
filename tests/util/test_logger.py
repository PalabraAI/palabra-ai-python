import pytest
from palabra_ai.util.logger import set_logging, debug, info, warning, error, exception, logger


class TestUtilLogger:
    def test_set_logging_variants(self, capfd, tmp_path):
        # Silent mode
        set_logging(silent=True, debug=False, log_file=None)
        info("test message")
        captured = capfd.readouterr()
        assert "test message" not in captured.out

        # Debug mode
        set_logging(silent=False, debug=True, log_file=None)
        debug("debug message")
        info("info message")
        warning("warning message")
        error("error message")

        # Exception logging
        try:
            raise ValueError("test")
        except ValueError:
            exception("exception message")

        assert logger is not None

        # With file
        log_file = tmp_path / "test.log"
        set_logging(silent=False, debug=True, log_file=log_file)
        info("test file logging")
