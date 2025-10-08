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


def test_benchmark_saves_error_to_file_with_out():
    """Test that benchmark saves error.txt when --out is specified"""
    from palabra_ai.model import RunResult
    from palabra_ai.benchmark.__main__ import main
    from pathlib import Path
    import tempfile

    original_exc = ValueError("Test error for saving")
    failed_result = RunResult(ok=False, exc=original_exc, io_data=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with patch('palabra_ai.benchmark.__main__.PalabraAI') as mock_palabra_class:
            mock_palabra = MagicMock()
            mock_palabra.run.return_value = failed_result
            mock_palabra_class.return_value = mock_palabra

            with patch('sys.argv', ['benchmark', 'dummy.wav', 'en', 'es', '--out', str(output_dir)]):
                with patch('palabra_ai.benchmark.__main__.Path') as mock_path_class:
                    def path_side_effect(path_str):
                        if 'dummy.wav' in str(path_str):
                            mock_path = MagicMock()
                            mock_path.exists.return_value = True
                            return mock_path
                        return Path(path_str)
                    mock_path_class.side_effect = path_side_effect

                    with patch('av.open'):
                        with patch('palabra_ai.benchmark.__main__.FileReader'):
                            with patch('palabra_ai.benchmark.__main__.tqdm'):
                                with patch('palabra_ai.benchmark.__main__.get_system_info', return_value={"test": "info"}):
                                    try:
                                        main()
                                        assert False, "main() should have raised RuntimeError"
                                    except RuntimeError:
                                        # Check that error file was created
                                        error_files = list(output_dir.glob("*_bench_error.txt"))
                                        assert len(error_files) == 1, f"Expected 1 error file, found {len(error_files)}"

                                        error_content = error_files[0].read_text()
                                        assert "ValueError" in error_content
                                        assert "Test error for saving" in error_content
                                        assert "Traceback" in error_content or "traceback" in error_content


def test_benchmark_saves_sysinfo_on_start():
    """Test that benchmark saves sysinfo.json immediately when --out is specified"""
    from palabra_ai.model import RunResult, IoData
    from palabra_ai.benchmark.__main__ import main
    from pathlib import Path
    import tempfile

    # Create a successful result to avoid hitting error paths
    io_data = IoData(
        start_perf_ts=0.0,
        start_utc_ts=0.0,
        in_sr=16000,
        out_sr=16000,
        mode="ws",
        channels=1,
        events=[],
        count_events=0
    )
    successful_result = RunResult(ok=True, exc=None, io_data=io_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with patch('palabra_ai.benchmark.__main__.PalabraAI') as mock_palabra_class:
            mock_palabra = MagicMock()
            mock_palabra.run.return_value = successful_result
            mock_palabra_class.return_value = mock_palabra

            with patch('sys.argv', ['benchmark', 'dummy.wav', 'en', 'es', '--out', str(output_dir)]):
                with patch('palabra_ai.benchmark.__main__.Path') as mock_path_class:
                    def path_side_effect(path_str):
                        if 'dummy.wav' in str(path_str):
                            mock_path = MagicMock()
                            mock_path.exists.return_value = True
                            return mock_path
                        return Path(path_str)
                    mock_path_class.side_effect = path_side_effect

                    with patch('av.open'):
                        with patch('palabra_ai.benchmark.__main__.FileReader'):
                            with patch('palabra_ai.benchmark.__main__.tqdm'):
                                with patch('palabra_ai.benchmark.__main__.get_system_info', return_value={"test": "sysinfo"}):
                                    with patch('palabra_ai.benchmark.__main__.Report.parse', return_value=(MagicMock(), MagicMock(), MagicMock())):
                                        with patch('palabra_ai.benchmark.__main__.format_report', return_value="Test report"):
                                            with patch('palabra_ai.benchmark.__main__.save_wav'):
                                                try:
                                                    main()
                                                except Exception:
                                                    pass  # We don't care if it fails, just checking sysinfo was saved

                                                # Check that sysinfo file was created
                                                sysinfo_files = list(output_dir.glob("*_bench_sysinfo.json"))
                                                assert len(sysinfo_files) >= 1, f"Expected at least 1 sysinfo file, found {len(sysinfo_files)}"


def test_benchmark_handles_cancelled_error():
    """Test that benchmark properly handles CancelledError with context"""
    from palabra_ai.model import RunResult, LogData
    from palabra_ai.benchmark.__main__ import main
    import asyncio
    from pathlib import Path
    import tempfile
    from io import StringIO
    import sys

    # Create CancelledError with traceback
    cancelled_exc = asyncio.CancelledError()

    # Create log data with many entries to test "all logs" output
    log_entries = [f"Entry {i}: Log line {i}\n" for i in range(200)]
    log_entries.extend([
        "2025-10-03 15:10:43.128 | SUCCESS  | Starting...\n",
        "2025-10-03 15:10:47.623 | INFO     | Processing...\n",
        "2025-10-03 15:10:50.090 | ERROR    | Something went wrong\n",
        "2025-10-03 15:10:50.327 | INFO     | Cancelling...\n",
    ])

    log_data = LogData(
        version="1.0.0",
        sysinfo={"platform": "test"},
        messages=[],
        start_ts=0.0,
        cfg={"test": "config"},
        log_file="test.log",
        trace_file="",
        debug=True,
        logs=log_entries
    )

    failed_result = RunResult(ok=False, exc=cancelled_exc, io_data=None, log_data=log_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with patch('palabra_ai.benchmark.__main__.PalabraAI') as mock_palabra_class:
            mock_palabra = MagicMock()
            mock_palabra.run.return_value = failed_result
            mock_palabra_class.return_value = mock_palabra

            with patch('sys.argv', ['benchmark', 'dummy.wav', 'en', 'es', '--out', str(output_dir)]):
                with patch('palabra_ai.benchmark.__main__.Path') as mock_path_class:
                    def path_side_effect(path_str):
                        if 'dummy.wav' in str(path_str):
                            mock_path = MagicMock()
                            mock_path.exists.return_value = True
                            return mock_path
                        return Path(path_str)
                    mock_path_class.side_effect = path_side_effect

                    with patch('av.open'):
                        with patch('palabra_ai.benchmark.__main__.FileReader'):
                            with patch('palabra_ai.benchmark.__main__.tqdm'):
                                with patch('palabra_ai.benchmark.__main__.get_system_info', return_value={"test": "info"}):
                                    # Capture stdout to check that ALL logs are printed
                                    captured_output = StringIO()
                                    try:
                                        with patch('sys.stdout', captured_output):
                                            main()
                                        assert False, "main() should have raised RuntimeError"
                                    except RuntimeError as e:
                                        assert "CancelledError" in str(e)

                                        # Check that RunResult debug file was saved
                                        runresult_files = list(output_dir.glob("*_bench_runresult_debug.json"))
                                        assert len(runresult_files) >= 1, f"Expected RunResult debug file, found {len(runresult_files)}"

                                        # Check that error file was saved
                                        error_files = list(output_dir.glob("*_bench_error.txt"))
                                        assert len(error_files) >= 1, f"Expected error file, found {len(error_files)}"

                                        # Check that output mentions "cascade cancellation"
                                        output = captured_output.getvalue()
                                        assert "cascade cancellation" in output, "Should mention cascade cancellation"

                                        # Check that ALL logs were printed (not just last 100)
                                        assert f"Full logs (all {len(log_entries)} entries)" in output
                                        # Check that first entry was printed (would not be if only last 100)
                                        assert "Entry 0: Log line 0" in output


def test_sysinfo_contains_command():
    """Test that sysinfo.json contains command line information"""
    from palabra_ai.benchmark.__main__ import main
    from pathlib import Path
    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with patch('sys.argv', ['benchmark', 'test.wav', 'en', 'es', '--out', str(output_dir)]):
            with patch('palabra_ai.benchmark.__main__.Path') as mock_path_class:
                def path_side_effect(path_str):
                    if 'test.wav' in str(path_str):
                        mock_path = MagicMock()
                        mock_path.exists.return_value = True
                        return mock_path
                    return Path(path_str)
                mock_path_class.side_effect = path_side_effect

                with patch('av.open'):
                    # Main should save sysinfo immediately
                    try:
                        main()
                    except Exception:
                        pass  # We expect it to fail, just checking sysinfo was saved

                    # Check that sysinfo file was created
                    sysinfo_files = list(output_dir.glob("*_bench_sysinfo.json"))
                    assert len(sysinfo_files) >= 1, f"Expected sysinfo file, found {len(sysinfo_files)}"

                    # Check content
                    sysinfo = json.loads(sysinfo_files[0].read_text())
                    assert "command" in sysinfo
                    assert "argv" in sysinfo
                    assert "cwd" in sysinfo
                    assert "benchmark" in sysinfo["command"]
                    assert isinstance(sysinfo["argv"], list)


def test_manager_has_graceful_completion_flag():
    """Test that Manager class has _graceful_completion flag"""
    from palabra_ai.task.manager import Manager
    from dataclasses import fields

    # Check that Manager dataclass has _graceful_completion field
    field_names = {f.name for f in fields(Manager)}
    assert '_graceful_completion' in field_names, "Manager should have _graceful_completion field"


def test_benchmark_handles_result_none():
    """Test that benchmark handles result=None (Ctrl+C) without AttributeError"""
    from palabra_ai.benchmark.__main__ import main
    from pathlib import Path
    import tempfile
    from io import StringIO

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with patch('palabra_ai.benchmark.__main__.PalabraAI') as mock_palabra_class:
            mock_palabra = MagicMock()
            # Simulate KeyboardInterrupt returning None
            mock_palabra.run.return_value = None
            mock_palabra_class.return_value = mock_palabra

            with patch('sys.argv', ['benchmark', 'dummy.wav', 'en', 'es', '--out', str(output_dir)]):
                with patch('palabra_ai.benchmark.__main__.Path') as mock_path_class:
                    def path_side_effect(path_str):
                        if 'dummy.wav' in str(path_str):
                            mock_path = MagicMock()
                            mock_path.exists.return_value = True
                            return mock_path
                        return Path(path_str)
                    mock_path_class.side_effect = path_side_effect

                    with patch('av.open'):
                        with patch('palabra_ai.benchmark.__main__.FileReader'):
                            with patch('palabra_ai.benchmark.__main__.tqdm'):
                                with patch('palabra_ai.benchmark.__main__.get_system_info', return_value={"test": "info"}):
                                    captured_output = StringIO()
                                    with patch('sys.stdout', captured_output):
                                        # Should NOT raise AttributeError
                                        main()  # Should return gracefully, not raise

                                    output = captured_output.getvalue()
                                    # Check that appropriate message was printed
                                    assert "INTERRUPTED BY USER" in output
                                    assert "Ctrl+C" in output
                                    assert "No results were generated" in output


def test_benchmark_parse_handles_part_suffixes():
    """Test that _part_0 gets metrics but _part_1+ only gets text"""
    from palabra_ai.benchmark.__main__ import Report
    from palabra_ai.message import IoEvent, Dbg
    from palabra_ai.model import IoData
    from palabra_ai.enum import Kind
    from datetime import datetime

    # Create mock events for different tid patterns
    base_ts = 0.0  # Use relative timestamps starting from 0

    def make_event(idx, tid, mtype, dawn_ts, text="test"):
        import orjson
        import base64
        import numpy as np

        # Create dummy audio data (empty PCM samples)
        if mtype in ("input_audio_data", "output_audio_data"):
            audio_samples = np.zeros(160, dtype=np.int16)  # 10ms @ 16khz
            audio_b64 = base64.b64encode(audio_samples.tobytes()).decode('utf-8')

            if mtype == "input_audio_data":
                body_dict = {
                    "message_type": mtype,
                    "data": {"data": audio_b64}
                }
            else:  # output_audio_data
                body_dict = {
                    "message_type": mtype,
                    "data": {"data": audio_b64, "transcription_id": tid}
                }
        else:  # transcription messages
            body_dict = {
                "message_type": mtype,
                "data": {
                    "transcription": {
                        "text": text,
                        "segments": [{"start": dawn_ts, "end": dawn_ts + 1.0}],
                        "transcription_id": tid
                    }
                }
            }

        body_bytes = orjson.dumps(body_dict)

        return IoEvent(
            head=Dbg(kind=Kind.MESSAGE if "transcription" in mtype or mtype == "input_audio_data" else Kind.AUDIO,
                     ch=None, dir=None, idx=idx, dawn_ts=dawn_ts, dur_s=0.1),
            body=body_bytes,
            tid=None,
            mtype=None
        )

    events = [
        # Input audio events
        make_event(0, None, "input_audio_data", base_ts),
        make_event(1, None, "input_audio_data", base_ts + 0.1),
        make_event(2, None, "input_audio_data", base_ts + 0.2),

        # sentence_1 (no _part suffix) - should have metrics
        make_event(10, "sentence_1", "partial_transcription", base_ts + 0.5, "Hello"),
        make_event(11, "sentence_1", "validated_transcription", base_ts + 0.6, "Hello"),
        make_event(12, "sentence_1", "translated_transcription", base_ts + 0.7, "Hola"),
        make_event(13, "sentence_1", "output_audio_data", base_ts + 0.8),

        # sentence_2_part_0 - should have metrics
        make_event(20, "sentence_2_part_0", "partial_transcription", base_ts + 1.0, "World"),
        make_event(21, "sentence_2_part_0", "validated_transcription", base_ts + 1.1, "World"),
        make_event(22, "sentence_2_part_0", "translated_transcription", base_ts + 1.2, "Mundo"),
        make_event(23, "sentence_2_part_0", "output_audio_data", base_ts + 1.3),

        # sentence_2_part_1 - should NOT have metrics, only text
        make_event(30, "sentence_2_part_1", "validated_transcription", base_ts + 1.5, "Part one"),
        make_event(31, "sentence_2_part_1", "translated_transcription", base_ts + 1.6, "Parte uno"),

        # sentence_2_part_2 - should NOT have metrics, only text
        make_event(40, "sentence_2_part_2", "validated_transcription", base_ts + 2.0, "Part two"),
        make_event(41, "sentence_2_part_2", "translated_transcription", base_ts + 2.1, "Parte dos"),
    ]

    io_data = IoData(
        start_perf_ts=base_ts,
        start_utc_ts=base_ts,
        in_sr=16000,
        out_sr=24000,
        mode="ws",
        channels=1,
        events=events,
        count_events=len(events)
    )

    # Parse the report
    report, _, _ = Report.parse(io_data)

    # Check that we have all 4 sentences
    assert len(report.sentences) == 4, f"Expected 4 sentences, got {len(report.sentences)}"

    # sentence_1 should have metrics
    s1 = report.sentences["sentence_1"]
    assert s1.metric_partial is not None, "sentence_1 should have metric_partial"
    assert s1.metric_validated is not None, "sentence_1 should have metric_validated"
    assert s1.metric_translated is not None, "sentence_1 should have metric_translated"
    assert s1.validated_text == "Hello"
    assert s1.translated_text == "Hola"

    # sentence_2_part_0 should have metrics
    s2p0 = report.sentences["sentence_2_part_0"]
    assert s2p0.metric_partial is not None, "sentence_2_part_0 should have metric_partial"
    assert s2p0.metric_validated is not None, "sentence_2_part_0 should have metric_validated"
    assert s2p0.metric_translated is not None, "sentence_2_part_0 should have metric_translated"
    assert s2p0.validated_text == "World"
    assert s2p0.translated_text == "Mundo"

    # sentence_2_part_1 should NOT have metrics, only text
    s2p1 = report.sentences["sentence_2_part_1"]
    assert s2p1.metric_partial is None, "sentence_2_part_1 should NOT have metric_partial"
    assert s2p1.metric_validated is None, "sentence_2_part_1 should NOT have metric_validated"
    assert s2p1.metric_translated is None, "sentence_2_part_1 should NOT have metric_translated"
    assert s2p1.validated_text == "Part one"
    assert s2p1.translated_text == "Parte uno"

    # sentence_2_part_2 should NOT have metrics, only text
    s2p2 = report.sentences["sentence_2_part_2"]
    assert s2p2.metric_partial is None, "sentence_2_part_2 should NOT have metric_partial"
    assert s2p2.metric_validated is None, "sentence_2_part_2 should NOT have metric_validated"
    assert s2p2.metric_translated is None, "sentence_2_part_2 should NOT have metric_translated"
    assert s2p2.validated_text == "Part two"
    assert s2p2.translated_text == "Parte dos"

    # Check that metrics_summary only includes sentences WITH metrics
    assert "metric_partial" in report.metrics_summary
    # Should only have 2 values (sentence_1 and sentence_2_part_0)
    partial_count = len([s for s in report.sentences.values() if s.metric_partial is not None])
    assert partial_count == 2, f"Expected 2 sentences with metrics, got {partial_count}"
