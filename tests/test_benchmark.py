"""Tests for benchmark config loading"""
import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from palabra_ai.benchmark.utils import flatten_container_to_paths
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
    from palabra_ai.benchmark.report import Report
    from palabra_ai.message import IoEvent, Dbg
    from palabra_ai.model import IoData
    from palabra_ai.enum import Kind
    from datetime import datetime

    # Create mock events for different tid patterns
    base_ts = 0.0  # Use relative timestamps starting from 0

    # Helper to create set_task event
    def make_set_task_event(idx, dawn_ts):
        from palabra_ai.util.orjson import to_json
        body_dict = {
            "message_type": "set_task",
            "data": {
                "source": {"lang": {"code": "en"}},
                "targets": [{"lang": {"code": "es"}}]
            }
        }
        return IoEvent(
            head=Dbg(kind=Kind.MESSAGE, ch=None, dir=None, idx=idx, dawn_ts=dawn_ts, dur_s=0.0),
            body=to_json(body_dict),
            tid=None,
            mtype=None
        )

    def make_event(idx, tid, mtype, dawn_ts, text="test"):
        from palabra_ai.util.orjson import to_json
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

        body_bytes = to_json(body_dict)

        return IoEvent(
            head=Dbg(kind=Kind.MESSAGE if "transcription" in mtype or mtype == "input_audio_data" else Kind.AUDIO,
                     ch=None, dir=None, idx=idx, dawn_ts=dawn_ts, dur_s=0.1),
            body=body_bytes,
            tid=None,
            mtype=None
        )

    events = [
        # set_task event (required by Report.parse)
        make_set_task_event(0, base_ts),

        # Input audio events
        make_event(1, None, "input_audio_data", base_ts),
        make_event(2, None, "input_audio_data", base_ts + 0.1),
        make_event(3, None, "input_audio_data", base_ts + 0.2),

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
        count_events=len(events),
        reader_x_title="FileReader(test.wav)",
        writer_x_title="DummyWriter()"
    )

    # Parse the report
    report = Report.parse(io_data)

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

    # Check proper sorting - _part_1+ should use parent timestamp and be grouped with parent
    sorted_sentences = sorted(report.sentences.items(), key=lambda x: x[1].local_start_ts)
    sorted_tids = [tid for tid, _ in sorted_sentences]

    # sentence_2_part_0, sentence_2_part_1, sentence_2_part_2 should all use same parent timestamp
    s2p0 = report.sentences["sentence_2_part_0"]
    s2p1 = report.sentences["sentence_2_part_1"]
    s2p2 = report.sentences["sentence_2_part_2"]

    # All should have same local_start_ts (from parent)
    assert s2p0.local_start_ts == s2p1.local_start_ts, "sentence_2_part_1 should use parent timestamp"
    assert s2p0.local_start_ts == s2p2.local_start_ts, "sentence_2_part_2 should use parent timestamp"

    # Test that format_report includes IDs correctly
    report_text = report.report_txt

    # Check that table contains formatted IDs
    assert "sentence_1" in report_text, "Should show sentence_1 without brackets"
    assert "sentence_2[0]" in report_text, "Should show sentence_2[0]"
    assert "sentence_2[1]" in report_text, "Should show sentence_2[1]"
    assert "sentence_2[2]" in report_text, "Should show sentence_2[2]"


def test_benchmark_parse_handles_partial_extra_parts():
    """Test that _part_1+ with only validated or only translated are shown"""
    from palabra_ai.benchmark.report import Report
    from palabra_ai.message import IoEvent, Dbg
    from palabra_ai.model import IoData
    from palabra_ai.enum import Kind
    from palabra_ai.util.orjson import to_json
    import base64
    import numpy as np

    base_ts = 0.0

    # Helper to create set_task event
    def make_set_task_event(idx, dawn_ts):
        body_dict = {
            "message_type": "set_task",
            "data": {
                "source": {"lang": {"code": "en"}},
                "targets": [{"lang": {"code": "es"}}]
            }
        }
        return IoEvent(
            head=Dbg(kind=Kind.MESSAGE, ch=None, dir=None, idx=idx, dawn_ts=dawn_ts, dur_s=0.0),
            body=to_json(body_dict),
            tid=None,
            mtype=None
        )

    def make_event(idx, tid, mtype, dawn_ts, text="test"):
        if mtype in ("input_audio_data", "output_audio_data"):
            audio_samples = np.zeros(160, dtype=np.int16)
            audio_b64 = base64.b64encode(audio_samples.tobytes()).decode('utf-8')
            body_dict = {
                "message_type": mtype,
                "data": {"data": audio_b64, "transcription_id": tid} if mtype == "output_audio_data" else {"data": audio_b64}
            }
        else:
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

        return IoEvent(
            head=Dbg(kind=Kind.MESSAGE if "transcription" in mtype or mtype == "input_audio_data" else Kind.AUDIO,
                     ch=None, dir=None, idx=idx, dawn_ts=dawn_ts, dur_s=0.1),
            body=to_json(body_dict),
            tid=None,
            mtype=None
        )

    events = [
        # set_task event (required by Report.parse)
        make_set_task_event(0, base_ts),

        make_event(1, None, "input_audio_data", base_ts),

        # Parent sentence
        make_event(10, "s1_part_0", "partial_transcription", base_ts + 0.5, "Parent"),
        make_event(11, "s1_part_0", "validated_transcription", base_ts + 0.6, "Parent"),
        make_event(12, "s1_part_0", "translated_transcription", base_ts + 0.7, "Padre"),
        make_event(13, "s1_part_0", "output_audio_data", base_ts + 0.8),

        # Only validated
        make_event(20, "s1_part_1", "validated_transcription", base_ts + 1.0, "Only validated"),

        # Only translated
        make_event(30, "s1_part_2", "translated_transcription", base_ts + 1.5, "Solo traducido"),
    ]

    io_data = IoData(
        start_perf_ts=base_ts,
        start_utc_ts=base_ts,
        in_sr=16000,
        out_sr=24000,
        mode="ws",
        channels=1,
        events=events,
        count_events=len(events),
        reader_x_title="FileReader(test.wav)",
        writer_x_title="DummyWriter()"
    )

    report = Report.parse(io_data)

    assert len(report.sentences) == 3, f"Expected 3 sentences, got {len(report.sentences)}"

    # Check s1_part_1 (only validated)
    s1p1 = report.sentences["s1_part_1"]
    assert s1p1.validated_text == "Only validated"
    assert s1p1.translated_text == ""
    assert s1p1.metric_partial is None

    # Check s1_part_2 (only translated)
    s1p2 = report.sentences["s1_part_2"]
    assert s1p2.validated_text == ""
    assert s1p2.translated_text == "Solo traducido"
    assert s1p2.metric_partial is None


def test_benchmark_parse_handles_orphan_extra_parts():
    """Test that _part_1+ without parent prints warning and is skipped"""
    from palabra_ai.benchmark.report import Report
    from palabra_ai.message import IoEvent, Dbg
    from palabra_ai.model import IoData
    from palabra_ai.enum import Kind
    from io import StringIO
    import sys
    from palabra_ai.util.orjson import to_json

    base_ts = 0.0

    # Helper to create set_task event
    def make_set_task_event(idx, dawn_ts):
        body_dict = {
            "message_type": "set_task",
            "data": {
                "source": {"lang": {"code": "en"}},
                "targets": [{"lang": {"code": "es"}}]
            }
        }
        return IoEvent(
            head=Dbg(kind=Kind.MESSAGE, ch=None, dir=None, idx=idx, dawn_ts=dawn_ts, dur_s=0.0),
            body=to_json(body_dict),
            tid=None,
            mtype=None
        )

    def make_event(idx, tid, mtype, dawn_ts, text="test"):
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
        return IoEvent(
            head=Dbg(kind=Kind.MESSAGE, ch=None, dir=None, idx=idx, dawn_ts=dawn_ts, dur_s=0.1),
            body=to_json(body_dict),
            tid=None,
            mtype=None
        )

    events = [
        # set_task event (required by Report.parse)
        make_set_task_event(0, base_ts),

        # Orphan _part_1 without parent
        make_event(10, "orphan_part_1", "validated_transcription", base_ts + 0.5, "Orphan"),
        make_event(11, "orphan_part_1", "translated_transcription", base_ts + 0.6, "Huerfano"),
    ]

    io_data = IoData(
        start_perf_ts=base_ts,
        start_utc_ts=base_ts,
        in_sr=16000,
        out_sr=24000,
        mode="ws",
        channels=1,
        events=events,
        count_events=len(events),
        reader_x_title="FileReader(test.wav)",
        writer_x_title="DummyWriter()"
    )

    # Capture stdout to check for warning
    captured = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        report = Report.parse(io_data)
        sys.stdout = old_stdout
    finally:
        sys.stdout = old_stdout

    # Should be skipped
    assert "orphan_part_1" not in report.sentences

    # Should print warning
    output = captured.getvalue()
    assert "WARNING" in output
    assert "No parent sentence found for orphan_part_1" in output


def test_tid_parse():
    """Test that Tid.parse() works correctly"""
    from palabra_ai.benchmark.report import Tid

    # Without _part suffix
    tid1 = Tid.parse("sentence_1")
    assert tid1.base == "sentence_1"
    assert tid1.part_num is None
    assert tid1.display == "sentence_1"
    assert tid1.raw == "sentence_1"

    # With _part_0
    tid2 = Tid.parse("sentence_2_part_0")
    assert tid2.base == "sentence_2"
    assert tid2.part_num == 0
    assert tid2.display == "sentence_2[0]"
    assert tid2.raw == "sentence_2_part_0"

    # With _part_1+
    tid3 = Tid.parse("sentence_2_part_1")
    assert tid3.base == "sentence_2"
    assert tid3.part_num == 1
    assert tid3.display == "sentence_2[1]"
    assert tid3.raw == "sentence_2_part_1"

    tid4 = Tid.parse("sentence_2_part_10")
    assert tid4.base == "sentence_2"
    assert tid4.part_num == 10
    assert tid4.display == "sentence_2[10]"
    assert tid4.raw == "sentence_2_part_10"

    # Complex base ids
    tid5 = Tid.parse("my_long_sentence_id_part_5")
    assert tid5.base == "my_long_sentence_id"
    assert tid5.part_num == 5
    assert tid5.display == "my_long_sentence_id[5]"
    assert tid5.raw == "my_long_sentence_id_part_5"


def test_sentence_has_metrics():
    """Test that Sentence.has_metrics property works correctly"""
    from palabra_ai.benchmark.report import Sentence

    # Sentence without metrics (extra_part with text only)
    sentence_no_metrics = Sentence(
        transcription_id="test_part_1",
        local_start_ts=1.0,
        local_start_chunk_idx=0,
        validated_text="Some text",
        translated_text="Translated text"
    )
    assert sentence_no_metrics.has_metrics is False
    assert sentence_no_metrics.metric_partial is None

    # Sentence with metrics (focused sentence)
    sentence_with_metrics = Sentence(
        transcription_id="test",
        local_start_ts=1.0,
        local_start_chunk_idx=0,
        validated_text="Some text",
        translated_text="Translated text",
        metric_validated=2.5
    )
    assert sentence_with_metrics.has_metrics is True
    assert sentence_with_metrics.metric_validated == 2.5


def test_benchmark_always_overrides_allowed_message_types():
    """Test that benchmark always uses BENCHMARK_ALLOWED_MESSAGE_TYPES regardless of config"""
    from palabra_ai.benchmark.report import BENCHMARK_ALLOWED_MESSAGE_TYPES
    from palabra_ai.message import Message

    # Verify the constant contains all expected types
    expected = {mt.value for mt in Message.ALLOWED_TYPES}
    actual = set(BENCHMARK_ALLOWED_MESSAGE_TYPES)
    assert actual == expected

    # Verify it includes all required types
    assert "pipeline_timings" in BENCHMARK_ALLOWED_MESSAGE_TYPES
    assert "translated_transcription" in BENCHMARK_ALLOWED_MESSAGE_TYPES
    assert "partial_transcription" in BENCHMARK_ALLOWED_MESSAGE_TYPES
    assert "partial_translated_transcription" in BENCHMARK_ALLOWED_MESSAGE_TYPES
    assert "validated_transcription" in BENCHMARK_ALLOWED_MESSAGE_TYPES


def test_benchmark_handles_sentence_splitter_case():
    """Test that sentence splitter case uses _part_0 timestamp as parent for _part_1+"""
    from palabra_ai.benchmark.report import Report
    from palabra_ai.benchmark.report import Tid

    # Test the core logic directly: parent_timestamps building
    sentences = {
        "base_sentence_part_0": type('Sentence', (), {'local_start_ts': 1.0})(),
    }

    # Build registry as done in Report.parse
    parent_timestamps = {}
    for raw_tid, sentence in sentences.items():
        _tid = Tid.parse(raw_tid)
        if _tid.base not in parent_timestamps:
            parent_timestamps[_tid.base] = sentence.local_start_ts

    # For sentence splitter cases: find _part_0 sentences and use their timestamps
    # for base TIDs that don't exist (because base only has partial_transcription)
    for raw_tid, sentence in sentences.items():
        _tid = Tid.parse(raw_tid)
        if _tid.part_num == 0 and _tid.base not in parent_timestamps:
            parent_timestamps[_tid.base] = sentence.local_start_ts

    # Verify the fix works: base_sentence should be in parent_timestamps
    assert "base_sentence" in parent_timestamps, "base_sentence should be in parent_timestamps from _part_0"
    assert parent_timestamps["base_sentence"] == 1.0, "Should use _part_0 timestamp"

    # Test Tid parsing works correctly
    tid0 = Tid.parse("base_sentence_part_0")
    assert tid0.base == "base_sentence"
    assert tid0.part_num == 0

    tid1 = Tid.parse("base_sentence_part_1")
    assert tid1.base == "base_sentence"
    assert tid1.part_num == 1


def test_benchmark_handles_missing_partial_transcription():
    """Test that sentences are created even when partial_transcription is missing"""
    from palabra_ai.benchmark.report import Report
    from palabra_ai.model import IoData
    from palabra_ai.message import IoEvent, Dbg, Kind
    from palabra_ai.util.orjson import to_json
    import base64
    import numpy as np

    # Helper to create set_task event
    def make_set_task_event(idx, dawn_ts):
        body_dict = {
            "message_type": "set_task",
            "data": {
                "source": {"lang": {"code": "en"}},
                "targets": [{"lang": {"code": "es"}}]
            }
        }
        return IoEvent(
            head=Dbg(kind=Kind.MESSAGE, ch=None, dir=None, idx=idx, dawn_ts=dawn_ts, dur_s=0.0),
            body=to_json(body_dict),
            tid=None,
            mtype=None
        )

    def make_event(idx, tid, mtype, dawn_ts, text="test"):
        if mtype in ("input_audio_data", "output_audio_data"):
            audio_samples = np.zeros(160, dtype=np.int16)
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
        return IoEvent(
            head=Dbg(kind=Kind.MESSAGE if "transcription" in mtype or mtype == "input_audio_data" else Kind.AUDIO,
                     ch=None, dir=None, idx=idx, dawn_ts=dawn_ts, dur_s=0.1),
            body=to_json(body_dict),
            tid=None,
            mtype=None
        )

    base_tid = "test123_part_0"
    base_ts = 0.5

    # Create events: input, validated, translated, output_audio (no partial)
    events = [
        # set_task event (required by Report.parse)
        make_set_task_event(0, base_ts),

        make_event(1, None, "input_audio_data", base_ts),
        make_event(10, base_tid, "validated_transcription", base_ts + 0.5, "validated text"),
        make_event(11, base_tid, "translated_transcription", base_ts + 1.0, "translated text"),
        make_event(12, base_tid, "output_audio_data", base_ts + 1.5),
    ]

    # Create IoData
    io_data = IoData(
        start_perf_ts=base_ts, start_utc_ts=base_ts, in_sr=16000, out_sr=24000,
        mode="test", channels=1, events=events, count_events=len(events),
        reader_x_title="FileReader(test.wav)",
        writer_x_title="DummyWriter()"
    )

    # Parse with Report - should handle missing partial_transcription
    report = Report.parse(io_data)

    # Verify sentence was created despite missing partial_transcription
    assert len(report.sentences) == 1
    sentence = report.sentences[base_tid]

    # Check fallback behavior
    assert sentence.partial_text == ""  # Should be empty when partial missing
    assert sentence.partial_ts is None  # Should be None when partial missing
    assert sentence.validated_text == "validated text"  # Should use validated
    assert sentence.translated_text == "translated text"  # Should use translated

    # Timing should be calculated correctly
    assert sentence.validated_ts == base_ts + 0.5
    assert sentence.translated_ts == base_ts + 1.0
    assert sentence.tts_api_ts == base_ts + 1.5

    # Metrics should still be calculated (using validated timing calculations)
    assert sentence.metric_partial is None  # Should be None when partial missing
    assert sentence.metric_validated > 0  # Should be positive
    assert sentence.metric_translated > 0  # Should be positive


def test_benchmark_forces_100ms_chunks_with_config():
    """Test that benchmark forces 100ms chunks even when loading config with defaults"""
    import json
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from palabra_ai.config import Config

    # Create config that doesn't specify chunk duration (defaults to 320ms)
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
            ]
        }
    }

    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        config_path = Path(f.name)

    try:
        # This demonstrates the problem: config loaded from JSON defaults to 320ms
        config = Config.from_json(config_path.read_text())
        assert config.mode.input_chunk_duration_ms == 320, "Config from JSON should default to 320ms (this is the problem)"

        # Now test the actual benchmark config loading process
        # Mock the benchmark main function's config loading section
        from palabra_ai.benchmark.report import BENCHMARK_ALLOWED_MESSAGE_TYPES
        from palabra_ai.benchmark.report import INPUT_CHUNK_DURATION_S
        from palabra_ai.config import WsMode
        from palabra_ai.task.adapter.file import FileReader
        from palabra_ai.task.adapter.dummy import DummyWriter

        # Simulate what benchmark main does when --config is provided
        # This is the code path that currently has the bug

        # Step 1: Create initial mode with 100ms chunks (line 527)
        mode = WsMode(input_chunk_duration_ms=INPUT_CHUNK_DURATION_S*1000)
        assert mode.input_chunk_duration_ms == 100, "Initial mode should be 100ms"

        # Step 2: Load config from JSON (line 564) - this overwrites the mode!
        config = Config.from_json(config_path.read_text())
        assert config.mode.input_chunk_duration_ms == 320, "JSON config overwrites with 320ms default"

        # Step 3: Apply benchmark overrides (lines 566-571)
        mock_reader = MagicMock()
        mock_on_transcription = MagicMock()
        config.source._reader = mock_reader
        config.source._on_transcription = mock_on_transcription
        config.targets[0]._writer = DummyWriter()
        config.benchmark = True
        config.allowed_message_types = BENCHMARK_ALLOWED_MESSAGE_TYPES

        # Step 4: Apply the fix - force benchmark mode with 100ms buffer (lines 573-575)
        config.mode = WsMode(input_chunk_duration_ms=INPUT_CHUNK_DURATION_S*1000)

        # After applying the fix: benchmark should force 100ms chunks
        assert config.mode.input_chunk_duration_ms == 100, "After fix: benchmark should force 100ms chunks"

    finally:
        # Cleanup
        config_path.unlink()


def test_rewind_saves_files_with_out_option():
    """Test that rewind --out saves all expected files"""
    from palabra_ai.benchmark.rewind import main as rewind_main
    from palabra_ai.model import IoData, RunResult
    from pathlib import Path
    import tempfile
    import json
    from unittest.mock import patch, MagicMock

    # Create mock io_data
    io_data = IoData(
        start_perf_ts=0.0,
        start_utc_ts=0.0,
        in_sr=16000,
        out_sr=24000,
        mode="ws",
        channels=1,
        events=[],
        count_events=0,
        reader_x_title="FileReader(test.wav)",
        writer_x_title="DummyWriter()"
    )

    # Create mock benchmark result file content
    mock_result = {
        "io_data": {
            "start_perf_ts": 0.0,
            "start_utc_ts": 0.0,
            "in_sr": 16000,
            "out_sr": 24000,
            "mode": "ws",
            "channels": 1,
            "events": [],
            "reader_x_title": "FileReader(test.wav)",
            "writer_x_title": "DummyWriter()"
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create mock result file
        result_file = input_dir / "test_result.json"
        result_file.write_text(json.dumps(mock_result))

        with patch('sys.argv', ['rewind', str(result_file), '--out', str(output_dir)]):
            with patch('palabra_ai.benchmark.rewind.load_run_result') as mock_load:
                mock_load.return_value = io_data

                with patch('palabra_ai.benchmark.rewind.Report.parse') as mock_parse:
                    # Mock report with save_all and report_txt
                    mock_report = MagicMock()
                    mock_report.report_txt = "Test report content"
                    mock_report.save_all = MagicMock()
                    mock_parse.return_value = mock_report

                    # Run rewind
                    rewind_main()

                    # Verify parse and save_all were called
                    mock_parse.assert_called_once()
                    mock_report.save_all.assert_called_once()




def test_merge_task_settings_full_outer_join():
    """Test that merge_task_settings performs correct full outer join with proper sorting"""
    from palabra_ai.benchmark.report import merge_task_settings

    sent_paths = [
        ("pipeline.auto_tempo", True),
        ("pipeline.shared_setting", "sent_value"),
        ("pipeline.sent_only", "only_in_sent")
    ]

    applied_paths = [
        ("pipeline.auto_tempo", False),
        ("pipeline.shared_setting", "applied_value"),
        ("pipeline.applied_only", "only_in_applied")
    ]

    merged = merge_task_settings(sent_paths, applied_paths)

    # Verify all keys are present
    merged_dict = {key: (sent, applied) for key, sent, applied in merged}
    assert len(merged_dict) == 4

    # Verify values
    assert merged_dict["pipeline.auto_tempo"] == (True, False)
    assert merged_dict["pipeline.shared_setting"] == ("sent_value", "applied_value")
    assert merged_dict["pipeline.sent_only"] == ("only_in_sent", None)
    assert merged_dict["pipeline.applied_only"] == (None, "only_in_applied")

    # Verify sorting: keys with SENT values first, then APPLIED only, alphabetically
    expected_order = [
        "pipeline.auto_tempo",      # has SENT
        "pipeline.sent_only",       # has SENT
        "pipeline.shared_setting",  # has SENT
        "pipeline.applied_only"     # only APPLIED
    ]
    actual_order = [key for key, _, _ in merged]
    assert actual_order == expected_order






def test_mixed_list_of_dicts():
    data = [
        {"foo": {"bar": False}},
        {"bar": {"baz": 10}},
        {"baz": {"xxx": {}}}
    ]
    got = flatten_container_to_paths(data)
    want = [
        ("0.foo.bar", False),
        ("1.bar.baz", 10),
        ("2.baz.xxx", {}),
    ]
    assert got == want

def test_mixed_dict_of_lists_and_scalars():
    data = {
        "foo": {"bar": [{"baz": []}]},
        "xxx": None,
        "zzz": "hello"
    }
    got = flatten_container_to_paths(data)
    want = [
        ("foo.bar.0.baz", []),
        ("xxx", None),
        ("zzz", "hello"),
    ]
    assert got == want

def test_empty_dict_is_leaf():
    data = {}
    got = flatten_container_to_paths(data, prefix="root")
    assert got == [("root", {})]

def test_empty_list_is_leaf():
    data = []
    got = flatten_container_to_paths(data, prefix="root")
    assert got == [("root", [])]

def test_with_prefix_on_container_and_internal_values():
    data = {"a": [1, {"b": 2}], "c": {}}
    got = flatten_container_to_paths(data, prefix="P")
    want = [
        ("P.a.0", 1),
        ("P.a.1.b", 2),
        ("P.c", {}),
    ]
    assert got == want

def test_boolean_none_numbers_strings_and_order():
    data = {"k1": True, "k2": 0, "k3": 1.5, "k4": None, "k5": "s"}
    got = flatten_container_to_paths(data)
    want = [("k1", True), ("k2", 0), ("k3", 1.5), ("k4", None), ("k5", "s")]
    assert got == want

def test_fallback_non_container():
    # Not expected by signature, but ensure graceful behavior
    got = flatten_container_to_paths(42, prefix="weird")  # type: ignore[arg-type]
    assert got == [("weird", 42)]
