import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from deepdiff import DeepDiff

from palabra_ai.config import (
    Config, SourceLang, TargetLang, LanguageField,
    Preprocessing, Transcription, Translation, SpeechGen,
    Splitter, SplitterAdvanced, Verification, FillerPhrases,
    TranscriptionAdvanced, TTSAdvanced, TimbreDetection,
    QueueConfig, QueueConfigs, InputStream, OutputStream,
    validate_language, serialize_language
)
from palabra_ai.lang import Language
from palabra_ai.exc import ConfigurationError
from palabra_ai.adapter.dummy import DummyWriter
from palabra_ai.base.message import Message
from palabra_ai.util.differ import is_dict_subset


class TestLanguageField:
    def test_validate_language(self):
        assert validate_language("en").code == "en"

        lang = Language("es")
        assert validate_language(lang) is lang

    def test_serialize_language(self):
        assert serialize_language(Language("fr")) == "fr"


class TestSourceLang:
    def test_init_valid(self, mock_reader):
        source = SourceLang(lang="en", reader=mock_reader, on_transcription=MagicMock())
        assert source.lang.code == "en"

    def test_init_invalid_callback(self, mock_reader):
        with pytest.raises(ConfigurationError, match="on_transcription should be a callable"):
            SourceLang(lang="en", reader=mock_reader, on_transcription="not_callable")


class TestConfig:
    def test_reconstruct_from_legacy(self):
        assert Config.reconstruct_from_serialized("not a dict") == "not a dict"

        data = {
            "pipeline": {"preprocessing": {"enable_vad": True}},
            "transcription": {"source_language": "en", "asr_model": "auto"},
            "translations": [{"target_language": "es", "translation_model": "auto"}]
        }
        result = Config.reconstruct_from_serialized(data)
        assert result["source"]["lang"] == "en"
        assert result["targets"][0]["lang"] == "es"

    @pytest.mark.parametrize("json_file", [
        "fixtures/minimal_settings.json",
        "fixtures/full_settings.json"
    ])
    def test_through_json_equality(self, json_file):
        with open(Path(__file__).parent / json_file) as f:
            orig_json = f.read()
            orig_dict = json.loads(orig_json)
        result_dict = json.loads(Config.from_json(orig_json).to_json())
        if not is_dict_subset(orig_dict, result_dict):
            breakpoint()
        assert is_dict_subset(orig_dict, result_dict), DeepDiff(orig_dict, result_dict, ignore_order=True).pretty()

class TestConfigComponents:
    def test_preprocessing_defaults(self):
        prep = Preprocessing()
        assert prep.enable_vad is True
        assert prep.vad_threshold == 0.5

    def test_queue_configs_alias(self):
        configs = QueueConfigs.model_validate({"global": {"desired_queue_level_ms": 100}})
        assert configs.global_.desired_queue_level_ms == 100




