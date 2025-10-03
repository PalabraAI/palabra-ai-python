"""Tests for benchmark config loading"""
import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from palabra_ai.config import Config


def test_benchmark_loads_config_from_json():
    """Test that benchmark correctly loads and applies config from JSON file"""
    # Create test config with specific auto_tempo settings
    test_config = {
        "input_stream": {
            "content_type": "audio",
            "source": {
                "type": "ws",
                "format": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1
            }
        },
        "output_stream": {
            "content_type": "audio",
            "target": {
                "type": "ws",
                "format": "pcm_s16le",
                "sample_rate": 24000,
                "channels": 1
            }
        },
        "pipeline": {
            "transcription": {
                "source_language": "en"
            },
            "translations": [
                {
                    "target_language": "es"
                }
            ],
            "translation_queue_configs": {
                "global": {
                    "auto_tempo": True,
                    "min_tempo": 2.0,
                    "max_tempo": 2.0,
                    "desired_queue_level_ms": 5000,
                    "max_queue_level_ms": 20000
                }
            }
        }
    }

    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        config_path = Path(f.name)

    try:
        # Load config using Config.from_json (simulating what benchmark does)
        loaded_config = Config.from_json(config_path.read_text())

        # Verify auto_tempo settings are applied
        assert loaded_config.translation_queue_configs is not None
        global_config = loaded_config.translation_queue_configs.global_
        assert global_config.auto_tempo is True
        assert global_config.min_tempo == 2.0
        assert global_config.max_tempo == 2.0
        assert global_config.desired_queue_level_ms == 5000
        assert global_config.max_queue_level_ms == 20000

        # Verify languages
        assert loaded_config.source.lang.code == "en"
        assert loaded_config.targets[0].lang.code == "es"

    finally:
        # Cleanup
        config_path.unlink()


def test_benchmark_config_to_dict():
    """Test that config settings are preserved in dict representation"""
    test_config = {
        "input_stream": {
            "content_type": "audio",
            "source": {
                "type": "ws",
                "format": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1
            }
        },
        "output_stream": {
            "content_type": "audio",
            "target": {
                "type": "ws",
                "format": "pcm_s16le",
                "sample_rate": 24000,
                "channels": 1
            }
        },
        "pipeline": {
            "transcription": {
                "source_language": "en"
            },
            "translations": [
                {
                    "target_language": "es"
                }
            ],
            "translation_queue_configs": {
                "global": {
                    "auto_tempo": False,
                    "min_tempo": 1.5,
                    "max_tempo": 1.8
                }
            }
        }
    }

    config = Config.from_json(test_config)

    # Verify translation_queue_configs are preserved
    assert config.translation_queue_configs.global_.auto_tempo is False
    assert config.translation_queue_configs.global_.min_tempo == 1.5
    assert config.translation_queue_configs.global_.max_tempo == 1.8

    # Verify config can be serialized back (for set_task)
    config_dict = config.to_dict()
    queue_configs = config_dict["pipeline"]["translation_queue_configs"]["global"]
    assert queue_configs["auto_tempo"] is False
    assert queue_configs["min_tempo"] == 1.5
    assert queue_configs["max_tempo"] == 1.8


def test_benchmark_exception_propagation():
    """Test that benchmark properly propagates exceptions with full context"""
    from palabra_ai.model import RunResult
    from palabra_ai.benchmark.__main__ import main

    # Mock PalabraAI to return failed result with exception
    original_exc = ValueError("Original error message")
    failed_result = RunResult(ok=False, exc=original_exc, io_data=None)

    with patch('palabra_ai.benchmark.__main__.PalabraAI') as mock_palabra_class:
        mock_palabra = MagicMock()
        mock_palabra.run.return_value = failed_result
        mock_palabra_class.return_value = mock_palabra

        with patch('sys.argv', ['benchmark', 'dummy.wav', 'en', 'es']):
            with patch('palabra_ai.benchmark.__main__.Path') as mock_path:
                mock_path_instance = MagicMock()
                mock_path_instance.exists.return_value = True
                mock_path.return_value = mock_path_instance

                with patch('av.open'):
                    with patch('palabra_ai.benchmark.__main__.FileReader'):
                        with patch('palabra_ai.benchmark.__main__.tqdm'):
                            try:
                                main()
                                assert False, "main() should have raised RuntimeError"
                            except RuntimeError as e:
                                # Check that exception message contains type and message
                                assert "ValueError" in str(e)
                                assert "Original error message" in str(e)
                                # Check that original exception is chained
                                assert e.__cause__ is original_exc


def test_benchmark_exception_without_message():
    """Test that benchmark handles exceptions without message properly"""
    from palabra_ai.model import RunResult
    from palabra_ai.benchmark.__main__ import main

    # Create exception without message (empty string when converted to str)
    original_exc = RuntimeError()
    failed_result = RunResult(ok=False, exc=original_exc, io_data=None)

    with patch('palabra_ai.benchmark.__main__.PalabraAI') as mock_palabra_class:
        mock_palabra = MagicMock()
        mock_palabra.run.return_value = failed_result
        mock_palabra_class.return_value = mock_palabra

        with patch('sys.argv', ['benchmark', 'dummy.wav', 'en', 'es']):
            with patch('palabra_ai.benchmark.__main__.Path') as mock_path:
                mock_path_instance = MagicMock()
                mock_path_instance.exists.return_value = True
                mock_path.return_value = mock_path_instance

                with patch('av.open'):
                    with patch('palabra_ai.benchmark.__main__.FileReader'):
                        with patch('palabra_ai.benchmark.__main__.tqdm'):
                            try:
                                main()
                                assert False, "main() should have raised RuntimeError"
                            except RuntimeError as e:
                                # Even without message, should show exception type
                                assert "RuntimeError" in str(e)
                                # Check that original exception is chained
                                assert e.__cause__ is original_exc
