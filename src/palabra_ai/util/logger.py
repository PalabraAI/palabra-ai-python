import sys
from logging import DEBUG, INFO, WARNING
from pathlib import Path

from loguru import logger


def set_logging(silent: bool, debug: bool, log_file: Path):
    logger.remove()

    screen = WARNING if silent else INFO
    screen = DEBUG if debug else screen

    logger.add(
        sys.stderr,
        level=screen,
        colorize=True,  # Keep default colors
    )
    if log_file:
        logger.add(
            str(log_file.absolute()),
            level=DEBUG,
            enqueue=True,
            buffering=1,  # Line buffering for immediate write
            # Additional options for reliability:
            catch=True,  # Catch errors in logging itself
            backtrace=True,  # Full traceback on errors
            diagnose=True,  # Extra diagnostic info
        )


debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical
exception = logger.exception
log = logger.log
trace = logger.trace


__all__ = [
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception",
    "log",
    "trace",
    "set_logging",
]
