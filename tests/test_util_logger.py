import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from loguru import logger
from palabra_ai.util.logger import (
    Library,
    set_logging,
    DEBUG, INFO, WARNING, ERROR, SUCCESS,
    debug, info, warning, error, critical, exception, trace, success
)


class TestLibrary:
    """Test Library class"""

    def test_init_defaults(self):
        """Test Library initialization with defaults"""
        lib = Library()

        assert lib.name == "palabra_ai"
        assert lib.level == INFO
        assert lib.handlers == []

    def test_init_custom_values(self):
        """Test Library initialization with custom values"""
        lib = Library(name="test_lib", level=DEBUG)

        assert lib.name == "test_lib"
        assert lib.level == DEBUG
        assert lib.handlers == []

    def test_call_method(self):
        """Test __call__ method for setting level"""
        lib = Library()
        lib(DEBUG)

        assert lib.level == DEBUG

    def test_set_level_debug(self):
        """Test set_level with debug=True"""
        lib = Library()
        lib.set_level(silent=False, debug=True)

        assert lib.level == DEBUG

    def test_set_level_silent(self):
        """Test set_level with silent=True"""
        lib = Library()
        lib.set_level(silent=True, debug=False)

        assert lib.level == SUCCESS

    def test_set_level_normal(self):
        """Test set_level with both flags False"""
        lib = Library()
        lib.set_level(silent=False, debug=False)

        assert lib.level == INFO

    def test_is_library_record_true(self):
        """Test _is_library_record with library record"""
        lib = Library(name="test_lib")
        record = {"name": "test_lib.module"}

        assert lib._is_library_record(record) is True

    def test_is_library_record_false(self):
        """Test _is_library_record with non-library record"""
        lib = Library(name="test_lib")
        record = {"name": "other_module"}

        assert lib._is_library_record(record) is False

    def test_is_library_record_no_name(self):
        """Test _is_library_record with no name field"""
        lib = Library(name="test_lib")
        record = {}

        assert lib._is_library_record(record) is False

    def test_should_log_non_library(self):
        """Test should_log for non-library records"""
        lib = Library()
        record = {"name": "other_module", "level": MagicMock(no=ERROR)}

        assert lib.should_log(record) is True

    def test_should_log_library_above_level(self):
        """Test should_log for library record above threshold"""
        lib = Library(level=INFO)
        mock_level = MagicMock()
        mock_level.no = ERROR
        record = {"name": "palabra_ai.module", "level": mock_level}

        assert lib.should_log(record) is True

    def test_should_log_library_below_level(self):
        """Test should_log for library record below threshold"""
        lib = Library(level=WARNING)
        mock_level = MagicMock()
        mock_level.no = INFO
        record = {"name": "palabra_ai.module", "level": mock_level}

        assert lib.should_log(record) is False

    def test_create_console_filter_pass(self):
        """Test create_console_filter with passing record"""
        lib = Library()
        original_filter = lambda r: True
        combined_filter = lib.create_console_filter(original_filter)

        # Mock should_log to return True
        with patch.object(lib, 'should_log', return_value=True):
            record = {"test": "data"}
            assert combined_filter(record) is True

    def test_create_console_filter_fail_lib(self):
        """Test create_console_filter with library filter failing"""
        lib = Library()
        original_filter = lambda r: True
        combined_filter = lib.create_console_filter(original_filter)

        # Mock should_log to return False
        with patch.object(lib, 'should_log', return_value=False):
            record = {"test": "data"}
            assert combined_filter(record) is False

    def test_create_console_filter_fail_original(self):
        """Test create_console_filter with original filter failing"""
        lib = Library()
        original_filter = lambda r: False
        combined_filter = lib.create_console_filter(original_filter)

        # Mock should_log to return True
        with patch.object(lib, 'should_log', return_value=True):
            record = {"test": "data"}
            assert combined_filter(record) is False

    def test_create_console_filter_no_original(self):
        """Test create_console_filter with no original filter"""
        lib = Library()
        combined_filter = lib.create_console_filter(None)

        # Mock should_log to return True
        with patch.object(lib, 'should_log', return_value=True):
            record = {"test": "data"}
            assert combined_filter(record) is True

    def test_create_file_filter(self):
        """Test create_file_filter"""
        lib = Library(name="test_lib")
        file_filter = lib.create_file_filter()

        # Test with library record
        lib_record = {"name": "test_lib.module"}
        assert file_filter(lib_record) is True

        # Test with non-library record
        other_record = {"name": "other.module"}
        assert file_filter(other_record) is False

    def test_cleanup_handlers_success(self):
        """Test cleanup_handlers with successful removal"""
        lib = Library()
        lib.handlers = [1, 2, 3]

        with patch.object(logger, 'remove') as mock_remove:
            lib.cleanup_handlers()

            assert mock_remove.call_count == 3
            assert lib.handlers == []

    def test_cleanup_handlers_with_errors(self):
        """Test cleanup_handlers with some removal errors"""
        lib = Library()
        lib.handlers = [1, 2, 3]

        def side_effect(handler_id):
            if handler_id == 2:
                raise ValueError("Handler not found")

        with patch.object(logger, 'remove', side_effect=side_effect):
            lib.cleanup_handlers()

            assert lib.handlers == []

    def test_setup_console_handler_modify_existing(self):
        """Test setup_console_handler with existing default handler"""
        lib = Library()

        # Mock existing handler
        mock_handler = MagicMock()
        mock_handler._filter = lambda r: True
        mock_handlers = {0: mock_handler}

        with patch.object(logger._core, 'handlers', mock_handlers):
            lib.setup_console_handler()

            # Should modify existing handler filter
            assert mock_handler._filter is not None
            # Handlers list should remain empty (no new handler added)
            assert lib.handlers == []

    def test_setup_console_handler_create_new(self):
        """Test setup_console_handler with no existing handler"""
        lib = Library()

        with patch.object(logger._core, 'handlers', {}), \
             patch.object(logger, 'add', return_value=999) as mock_add:

            lib.setup_console_handler()

            mock_add.assert_called_once()
            assert 999 in lib.handlers

    def test_setup_console_handler_prevent_recursion(self):
        """Test setup_console_handler prevents recursion on repeated calls"""
        lib = Library()

        # Mock existing handler
        mock_handler = MagicMock()
        original_filter = lambda r: True
        mock_handler._filter = original_filter
        mock_handlers = {0: mock_handler}

        with patch.object(logger._core, 'handlers', mock_handlers):
            # First call should save original filter
            lib.setup_console_handler()
            assert lib._original_console_filter == original_filter

            # Second call should use saved original filter, not the new combined filter
            lib.setup_console_handler()
            # Original filter reference should remain unchanged
            assert lib._original_console_filter == original_filter
            # Filter should still be functional (not recursive)
            assert mock_handler._filter is not None
    def test_setup_file_handler_no_file(self):
        """Test setup_file_handler with no log file"""
        lib = Library()

        lib.setup_file_handler(None)

        assert lib.handlers == []

    def test_setup_file_handler_with_file(self):
        """Test setup_file_handler with log file"""
        lib = Library()
        log_file = Path("/tmp/test.log")

        with patch.object(logger, 'add', return_value=123) as mock_add:
            lib.setup_file_handler(log_file)

            mock_add.assert_called_once()
            call_args, call_kwargs = mock_add.call_args
            assert call_args[0] == str(log_file.absolute())
            assert call_kwargs['level'] == DEBUG
            assert 123 in lib.handlers


class TestSetLogging:
    """Test set_logging function"""

    def test_set_logging_debug(self):
        """Test set_logging with debug mode"""
        from io import StringIO
        text_io = StringIO()
        with patch('palabra_ai.util.logger._lib') as mock_lib:
            set_logging(silent=False, debug=True, text_io=text_io, log_file=None)

            mock_lib.set_level.assert_called_once_with(False, True)
            mock_lib.cleanup_handlers.assert_called_once()
            mock_lib.setup_console_handler.assert_called_once()
            mock_lib.setup_textio_handler.assert_called_once_with(text_io)
            # setup_file_handler is not called when log_file is None
            mock_lib.setup_file_handler.assert_not_called()

    def test_set_logging_with_file(self):
        """Test set_logging with log file"""
        from io import StringIO
        text_io = StringIO()
        log_file = Path("/tmp/test.log")

        with patch('palabra_ai.util.logger._lib') as mock_lib:
            set_logging(silent=True, debug=False, text_io=text_io, log_file=log_file)

            mock_lib.set_level.assert_called_once_with(True, False)
            mock_lib.cleanup_handlers.assert_called_once()
            mock_lib.setup_console_handler.assert_called_once()
            mock_lib.setup_textio_handler.assert_called_once_with(text_io)
            mock_lib.setup_file_handler.assert_called_once_with(log_file)

    def test_set_logging_repeated_calls_no_recursion(self):
        """Test repeated calls to set_logging don't cause recursion"""
        from io import StringIO

        # Test with actual Library instance (not mocked) to verify no recursion
        text_io1 = StringIO()
        text_io2 = StringIO()
        text_io3 = StringIO()

        # Multiple calls should not raise RecursionError
        try:
            set_logging(silent=True, debug=False, text_io=text_io1, log_file=None)
            set_logging(silent=False, debug=True, text_io=text_io2, log_file=None)
            set_logging(silent=True, debug=False, text_io=text_io3, log_file=None)

            # Try logging something to verify the filter chain works
            logger.info("Test message for recursion check")

        except RecursionError:
            pytest.fail("set_logging caused RecursionError on repeated calls")


class TestLoggerExports:
    """Test exported logger functions"""

    def test_direct_exports_exist(self):
        """Test that all direct exports exist and are callable"""
        exports = [debug, info, warning, error, critical, exception, trace, success]

        for func in exports:
            assert callable(func)

    def test_constants_exist(self):
        """Test that logging constants are defined"""
        assert DEBUG == 10
        assert INFO == 20
        assert SUCCESS == 25
        assert WARNING == 30
        assert ERROR == 40
        assert isinstance(DEBUG, int)
        assert isinstance(INFO, int)
        assert isinstance(WARNING, int)
        assert isinstance(ERROR, int)
        assert isinstance(SUCCESS, int)


class TestLoggerBraceHandling:
    """Test logger behavior with curly braces in messages."""

    def test_simple_braces_no_exc_info(self):
        """Test that simple braces work without exc_info."""
        # Should work fine
        debug("Message with {'key': 'value'}")
        info("Single brace: {")
        warning("Error with {unknown} placeholder")

    def test_validation_error_message_without_exc_info(self):
        """Test the exact validation error that caused KeyError (without exc_info)."""
        validation_error = "ValidationError(model='SetTaskRequestMessage', errors=[{'loc': ('output_stream', 'target', 'type'), 'msg': \"unexpected value; permitted: 'livekit', 'webrtc'\"}])"

        try:
            raise Exception(validation_error)
        except Exception as e:
            # This should not raise KeyError (without exc_info)
            error(f"Task failed: {e}")

    def test_dict_like_content_without_exc_info(self):
        """Test dict-like content in exception without exc_info."""
        try:
            raise Exception("Error with {'loc': 'test', 'msg': 'test message'}")
        except Exception as e:
            # This should not raise KeyError
            error(f"Exception occurred: {e}")

    def test_single_brace_without_exc_info(self):
        """Test single brace in exception without exc_info."""
        try:
            raise Exception("Single brace: {")
        except Exception as e:
            # This should not raise ValueError or KeyError
            error(f"Error: {e}")

    def test_unknown_placeholder_without_exc_info(self):
        """Test unknown placeholder in exception without exc_info."""
        try:
            raise Exception("Error with {unknown} placeholder")
        except Exception as e:
            # This should not raise KeyError
            error(f"Failed: {e}")

    def test_nested_dict_in_exception(self):
        """Test nested dict structures in exception."""
        try:
            complex_error = "ValidationError(errors=[{'loc': ('field',), 'ctx': {'given': 'ws', 'permitted': ('webrtc',)}}])"
            raise Exception(complex_error)
        except Exception as e:
            # This should not raise KeyError
            error(f"Complex error: {e}")

    def test_all_log_levels_with_problematic_content(self):
        """Test all log levels with content that could cause formatting issues."""
        problematic_msg = "Content with {'key': 'value'} and {placeholder}"

        try:
            raise Exception(problematic_msg)
        except Exception as e:
            # All of these should work without KeyError (no exc_info)
            debug(f"Debug: {e}")
            info(f"Info: {e}")
            warning(f"Warning: {e}")
            error(f"Error: {e}")
            critical(f"Critical: {e}")
            exception(f"Exception: {e}")

    def test_normal_formatting_still_works(self):
        """Test that normal loguru formatting still works."""
        # These should still work as before
        debug("Normal message")
        info("Value: {}", 42)
        warning("Multiple values: {} and {}", "first", "second")

    def test_exact_task_base_scenario(self):
        """Test the exact scenario from task/base.py after removing exc_info."""
        try:
            # Simulate the exact error from the server
            server_error = "ValidationError(model='SetTaskRequestMessage', errors=[{'loc': ('output_stream', 'target', 'type'), 'msg': \"unexpected value; permitted: 'livekit', 'webrtc'\"}, {'loc': ('output_stream', 'target', 'sample_rate'), 'msg': 'ensure this value is less than or equal to 24000'}])"
            raise Exception(server_error)
        except Exception as e:
            # This is exactly line 100 from task/base.py (without exc_info)
            error(f"Task.run() failed with error: {e}, exiting...")

    def test_output_looks_normal(self):
        """Test that logger works with problematic braces without error."""
        # This test just ensures no KeyError is raised
        error("Error with {'loc': 'test'}")
        debug("Debug with {unknown} placeholder")
        warning("Warning with {'key': 'value'} dict")

        # If we reach here without exception, the test passes
        assert True
