import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import json

from palabra_ai.config import Config, SourceLang, TargetLang, WsMode, WebrtcMode
from palabra_ai.benchmark.runner import BenchmarkRunner
from palabra_ai.lang import Language
from palabra_ai.exc import ApiValidationError


def test_benchmark_runner_with_base_config():
    """Test BenchmarkRunner with base_config parameter"""
    # Create a mock base config with source and target languages
    base_config = Config(
        source=SourceLang(lang="es"),
        targets=[TargetLang(lang="en")]
    )
    base_config.preprocessing.enable_vad = False
    base_config.preprocessing.pre_vad_denoise = True

    # Create temp audio file (just for testing, doesn't need to be real audio)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        audio_file = f.name
        f.write(b"dummy audio data")

    try:
        # Mock librosa and PalabraAI to avoid actual audio loading and client creation
        with patch('palabra_ai.benchmark.runner.librosa.load') as mock_load, \
             patch('palabra_ai.benchmark.runner.PalabraAI') as mock_palabra:

            mock_load.return_value = ([0.0] * 1000, 16000)  # 1000 samples at 16kHz
            mock_palabra.return_value = Mock()  # Mock PalabraAI instance

            runner = BenchmarkRunner(
                audio_file=audio_file,
                source_lang="es",
                target_lang="en",
                silent=True,
                mode="ws",
                chunk_duration_ms=100,
                base_config=base_config
            )

            assert runner.base_config == base_config
            assert runner.source_lang.code == "es"
            assert runner.target_lang.code == "en"
            assert runner.mode == "ws"
            assert runner.chunk_duration_ms == 100
    finally:
        Path(audio_file).unlink()


def test_benchmark_runner_without_base_config():
    """Test BenchmarkRunner without base_config (backward compatibility)"""
    # Create temp audio file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        audio_file = f.name
        f.write(b"dummy audio data")

    try:
        # Mock librosa and PalabraAI to avoid actual audio loading and client creation
        with patch('palabra_ai.benchmark.runner.librosa.load') as mock_load, \
             patch('palabra_ai.benchmark.runner.PalabraAI') as mock_palabra:

            mock_load.return_value = ([0.0] * 1000, 16000)
            mock_palabra.return_value = Mock()  # Mock PalabraAI instance

            runner = BenchmarkRunner(
                audio_file=audio_file,
                source_lang="es",
                target_lang="en",
                silent=True,
                mode="ws",
                chunk_duration_ms=100
            )

            assert runner.base_config is None
            assert runner.source_lang.code == "es"
            assert runner.target_lang.code == "en"
    finally:
        Path(audio_file).unlink()


def test_benchmark_runner_run_with_base_config():
    """Test BenchmarkRunner.run() with base config override"""
    # Create a mock base config with specific settings
    base_config = Config(
        source=SourceLang(lang="es"),
        targets=[TargetLang(lang="en")]
    )
    base_config.preprocessing.enable_vad = False
    base_config.preprocessing.auto_tempo = True

    # Create temp audio file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        audio_file = f.name
        f.write(b"dummy audio data")

    try:
        # Mock all the dependencies
        with patch('palabra_ai.benchmark.runner.librosa.load') as mock_load, \
             patch('palabra_ai.benchmark.runner.PalabraAI') as mock_palabra, \
             patch('palabra_ai.benchmark.runner.FileReader') as mock_reader, \
             patch('palabra_ai.benchmark.runner.DummyWriter') as mock_writer:

            mock_load.return_value = ([0.0] * 1000, 16000)

            # Create a mock result
            mock_result = Mock()
            mock_result.ok = True
            mock_result.exc = None
            mock_result.log_data = Mock()
            mock_result.log_data.messages = []

            mock_palabra_instance = MagicMock()
            mock_palabra_instance.run.return_value = mock_result
            mock_palabra.return_value = mock_palabra_instance

            runner = BenchmarkRunner(
                audio_file=audio_file,
                source_lang="es",
                target_lang="en",
                silent=True,
                mode="ws",
                chunk_duration_ms=100,
                base_config=base_config
            )

            result = runner.run(show_progress=False)

            # Verify the config was used properly
            assert mock_palabra_instance.run.called
            call_args = mock_palabra_instance.run.call_args
            used_config = call_args[0][0]

            # Check that base config settings were preserved
            assert used_config.preprocessing.auto_tempo == True
            # Check that override settings were applied
            assert used_config.silent == True
            assert used_config.benchmark == True
            assert isinstance(used_config.mode, WsMode)
            assert used_config.mode.chunk_duration_ms == 100

    finally:
        Path(audio_file).unlink()


def test_config_from_json_in_benchmark():
    """Test loading config from JSON for benchmark"""
    # Create a sample config JSON
    config_data = {
        "pipeline": {
            "preprocessing": {
                "enable_vad": False,
                "auto_tempo": True,
                "vad_threshold": 0.7
            },
            "transcription": {
                "source_language": "es",
                "asr_model": "auto",
                "denoise": "low"
            },
            "translations": [
                {
                    "target_language": "en",
                    "translation_model": "auto"
                }
            ],
            "translation_queue_configs": {
                "global": {
                    "desired_queue_level_ms": 10000,
                    "max_queue_level_ms": 30000
                }
            },
            "allowed_message_types": []
        }
    }

    # Convert to JSON string
    json_str = json.dumps(config_data)

    # Test Config.from_json
    config = Config.from_json(json_str)

    assert config.preprocessing.enable_vad == False
    assert config.preprocessing.auto_tempo == True
    assert config.preprocessing.vad_threshold == 0.7
    assert config.source.lang.code == "es"
    assert config.source.transcription.denoise == "low"
    assert len(config.targets) == 1
    assert config.targets[0].lang.code == "en"
    assert config.translation_queue_configs.global_.desired_queue_level_ms == 10000


def test_validation_error_parsing():
    """Test that ValidationError from server is parsed correctly"""
    from palabra_ai.message import ErrorMessage, KnownRaw, KnownRawType

    # Simulate server error response for validation errors
    server_error = {
        "message_type": "error",
        "data": {
            "code": "VALIDATION_ERROR",
            "desc": "ValidationError(model='SetTaskRequestMessage', errors=[{'loc': ('pipeline', 'transcription', 'denoise'), 'msg': \"value is not a valid enumeration member; permitted: 'none', 'alpha', 'beta'\", 'type': 'type_error.enum'}, {'loc': ('pipeline', 'transcription', 'priority'), 'msg': \"value is not a valid enumeration member; permitted: 'speed', 'normal', 'quality'\", 'type': 'type_error.enum'}])"
        }
    }

    # Create ErrorMessage
    known_raw = KnownRaw(type=KnownRawType.json, data=server_error, exc=None)
    error_msg = ErrorMessage.create(known_raw)

    # Check that error is properly formatted
    assert error_msg._exc is not None
    assert isinstance(error_msg._exc, ApiValidationError)

    error_str = str(error_msg._exc)
    # Our simplified approach just returns the string as-is
    assert "ValidationError" in error_str
    assert "denoise" in error_str
    assert "priority" in error_str
    assert "'none', 'alpha', 'beta'" in error_str
    assert "'speed', 'normal', 'quality'" in error_str


def test_config_with_correct_enum_values():
    """Test that config with correct enum values works"""
    config_data = {
        "pipeline": {
            "preprocessing": {
                "enable_vad": True,
                "vad_threshold": 0.6,
                "auto_tempo": True
            },
            "transcription": {
                "source_language": "es",
                "asr_model": "auto",
                "denoise": "none",  # Valid: none, alpha, beta
                "priority": "quality"  # Valid: speed, normal, quality
            },
            "translations": [{
                "target_language": "en",
                "translation_model": "auto"
            }],
            "allowed_message_types": [
                "partial_transcription",
                "translated_transcription",
                "validated_transcription",
                "partial_translated_transcription",
                "pipeline_timings"
            ]
        }
    }

    # Should load without errors
    config = Config.from_json(json.dumps(config_data))
    assert config.source.lang.code == "es"
    assert config.targets[0].lang.code == "en"
    assert config.source.transcription.denoise == "none"
    assert config.source.transcription.priority == "quality"
    assert len(config.allowed_message_types) == 5
