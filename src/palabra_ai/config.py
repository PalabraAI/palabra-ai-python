from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from environs import Env
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    PrivateAttr,
    model_validator,
)

from palabra_ai.base.message import Message
from palabra_ai.constant import (
    CONTEXT_SIZE_DEFAULT,
    DESIRED_QUEUE_LEVEL_MS_DEFAULT,
    ENERGY_VARIANCE_FACTOR_DEFAULT,
    F0_VARIANCE_FACTOR_DEFAULT,
    FORCE_END_OF_SEGMENT_DEFAULT,
    MAX_ALIGNMENT_CER_DEFAULT,
    MAX_QUEUE_LEVEL_MS_DEFAULT,
    MAX_STEPS_WITHOUT_EOS_DEFAULT,
    MIN_ALIGNMENT_SCORE_DEFAULT,
    MIN_SENTENCE_CHARACTERS_DEFAULT,
    MIN_SENTENCE_SECONDS_DEFAULT,
    MIN_SPLIT_INTERVAL_DEFAULT,
    MIN_TRANSCRIPTION_LEN_DEFAULT,
    MIN_TRANSCRIPTION_TIME_DEFAULT,
    PHRASE_CHANCE_DEFAULT,
    SEGMENT_CONFIRMATION_SILENCE_THRESHOLD_DEFAULT,
    SEGMENTS_AFTER_RESTART_DEFAULT,
    SPEECH_TEMPO_ADJUSTMENT_FACTOR_DEFAULT,
    STEP_SIZE_DEFAULT,
    VAD_LEFT_PADDING_DEFAULT,
    VAD_RIGHT_PADDING_DEFAULT,
    VAD_THRESHOLD_DEFAULT,
)
from palabra_ai.exc import ConfigurationError
from palabra_ai.lang import Language
from palabra_ai.types import T_ON_TRANSCRIPTION
from palabra_ai.util.logger import set_logging
from palabra_ai.util.orjson import from_json, to_json

if TYPE_CHECKING:
    from palabra_ai.base.adapter import Reader, Writer

env = Env(prefix="PALABRA_")
env.read_env()
CLIENT_ID = env.str("CLIENT_ID", default=None)
CLIENT_SECRET = env.str("CLIENT_SECRET", default=None)
SILENT = env.bool("SILENT", default=False)
DEBUG = env.bool("DEBUG", default=False)
DEEP_DEBUG = env.bool("DEEP_DEBUG", default=False)
DEEPEST_DEBUG = env.bool("DEEPEST_DEBUG", default=False)
TIMEOUT = env.int("TIMEOUT", default=0)
LOG_FILE = env.path("LOG_FILE", default=None)


class Mode(StrEnum):
    MIXED = "mixed"
    WEBSOCKET = "websocket"
    WEBRTC = "webrtc"


def validate_language(v):
    if isinstance(v, str):
        return Language.get_by_bcp47(v)
    return v


def serialize_language(lang: Language) -> str:
    return lang.bcp47


LanguageField = Annotated[
    Language, BeforeValidator(validate_language), PlainSerializer(serialize_language)
]


class Stream(BaseModel):
    content_type: str = "audio"


class InputStream(Stream):
    source: dict[str, str] = {"type": "livekit"}


class OutputStream(Stream):
    target: dict[str, str] = {"type": "livekit"}


class Preprocessing(BaseModel):
    enable_vad: bool = True
    vad_threshold: float = VAD_THRESHOLD_DEFAULT
    vad_left_padding: int = VAD_LEFT_PADDING_DEFAULT
    vad_right_padding: int = VAD_RIGHT_PADDING_DEFAULT
    pre_vad_denoise: bool = False
    pre_vad_dsp: bool = True
    record_tracks: list[str] = []
    auto_tempo: bool = False


class SplitterAdvanced(BaseModel):
    min_sentence_characters: int = MIN_SENTENCE_CHARACTERS_DEFAULT
    min_sentence_seconds: int = MIN_SENTENCE_SECONDS_DEFAULT
    min_split_interval: float = MIN_SPLIT_INTERVAL_DEFAULT
    context_size: int = CONTEXT_SIZE_DEFAULT
    segments_after_restart: int = SEGMENTS_AFTER_RESTART_DEFAULT
    step_size: int = STEP_SIZE_DEFAULT
    max_steps_without_eos: int = MAX_STEPS_WITHOUT_EOS_DEFAULT
    force_end_of_segment: float = FORCE_END_OF_SEGMENT_DEFAULT


class Splitter(BaseModel):
    enabled: bool = True
    splitter_model: str = "auto"
    advanced: SplitterAdvanced = Field(default_factory=SplitterAdvanced)


class Verification(BaseModel):
    verification_model: str = "auto"
    allow_verification_glossaries: bool = True
    auto_transcription_correction: bool = False
    transcription_correction_style: str | None = None


class FillerPhrases(BaseModel):
    enabled: bool = False
    min_transcription_len: int = MIN_TRANSCRIPTION_LEN_DEFAULT
    min_transcription_time: int = MIN_TRANSCRIPTION_TIME_DEFAULT
    phrase_chance: float = PHRASE_CHANCE_DEFAULT


class TranscriptionAdvanced(BaseModel):
    filler_phrases: FillerPhrases = Field(default_factory=FillerPhrases)
    ignore_languages: list[str] = []


class Transcription(BaseModel):
    detectable_languages: list[str] = []
    asr_model: str = "auto"
    denoise: str = "none"
    allow_hotwords_glossaries: bool = True
    supress_numeral_tokens: bool = False
    diarize_speakers: bool = False
    priority: str = "normal"
    min_alignment_score: float = MIN_ALIGNMENT_SCORE_DEFAULT
    max_alignment_cer: float = MAX_ALIGNMENT_CER_DEFAULT
    segment_confirmation_silence_threshold: float = (
        SEGMENT_CONFIRMATION_SILENCE_THRESHOLD_DEFAULT
    )
    only_confirm_by_silence: bool = False
    batched_inference: bool = False
    force_detect_language: bool = False
    calculate_voice_loudness: bool = False
    sentence_splitter: Splitter = Field(default_factory=Splitter)
    verification: Verification = Field(default_factory=Verification)
    advanced: TranscriptionAdvanced = Field(default_factory=TranscriptionAdvanced)


class TimbreDetection(BaseModel):
    enabled: bool = False
    high_timbre_voices: list[str] = ["default_high"]
    low_timbre_voices: list[str] = ["default_low"]


class TTSAdvanced(BaseModel):
    f0_variance_factor: float = F0_VARIANCE_FACTOR_DEFAULT
    energy_variance_factor: float = ENERGY_VARIANCE_FACTOR_DEFAULT
    with_custom_stress: bool = True


class SpeechGen(BaseModel):
    tts_model: str = "auto"
    voice_cloning: bool = False
    voice_cloning_mode: str = "static_10"
    denoise_voice_samples: bool = True
    voice_id: str = "default_low"
    voice_timbre_detection: TimbreDetection = Field(default_factory=TimbreDetection)
    speech_tempo_auto: bool = True
    speech_tempo_timings_factor: int = 0
    speech_tempo_adjustment_factor: float = SPEECH_TEMPO_ADJUSTMENT_FACTOR_DEFAULT
    advanced: TTSAdvanced = Field(default_factory=TTSAdvanced)


class TranslationAdvanced(BaseModel):
    pass  # Empty for now, can be extended later


class Translation(BaseModel):
    allowed_source_languages: list[str] = []
    translation_model: str = "auto"
    allow_translation_glossaries: bool = True
    style: str | None = None
    translate_partial_transcriptions: bool = False
    speech_generation: SpeechGen = Field(default_factory=SpeechGen)
    advanced: TranslationAdvanced = Field(default_factory=TranslationAdvanced)


class QueueConfig(BaseModel):
    desired_queue_level_ms: int = DESIRED_QUEUE_LEVEL_MS_DEFAULT
    max_queue_level_ms: int = MAX_QUEUE_LEVEL_MS_DEFAULT
    auto_tempo: bool = False


class QueueConfigs(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    global_: QueueConfig = Field(alias="global", default_factory=QueueConfig)


class SourceLang(BaseModel):
    lang: LanguageField

    _reader: Reader | None = PrivateAttr(default=None)
    _on_transcription: T_ON_TRANSCRIPTION | None = PrivateAttr(default=None)

    transcription: Transcription = Field(default_factory=Transcription)

    def __init__(
        self,
        lang: LanguageField,
        reader: Reader | None = None,
        on_transcription: T_ON_TRANSCRIPTION | None = None,
        **kwargs,
    ):
        super().__init__(lang=lang, **kwargs)
        if on_transcription and not callable(on_transcription):
            raise ConfigurationError(
                f"on_transcription should be a callable function, got {type(on_transcription)}"
            )
        self._on_transcription = on_transcription
        self._reader = reader

    @property
    def reader(self) -> Reader:
        return self._reader

    @property
    def on_transcription(self) -> T_ON_TRANSCRIPTION | None:
        return self._on_transcription


class TargetLang(BaseModel):
    lang: LanguageField

    _writer: Writer | None = PrivateAttr(default=None)
    _on_transcription: T_ON_TRANSCRIPTION | None = PrivateAttr(default=None)

    translation: Translation = Field(default_factory=Translation)

    def __init__(
        self,
        lang: LanguageField,
        writer: Writer | None = None,
        on_transcription: T_ON_TRANSCRIPTION | None = None,
        **kwargs,
    ):
        super().__init__(lang=lang, **kwargs)
        if on_transcription and not callable(on_transcription):
            raise ConfigurationError(
                f"on_transcription should be a callable function, got {type(on_transcription)}"
            )
        self._writer = writer
        self._on_transcription = on_transcription

    @property
    def writer(self) -> Writer | None:
        return self._writer

    @property
    def on_transcription(self) -> T_ON_TRANSCRIPTION | None:
        return self._on_transcription


class Config(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source: SourceLang | None = Field(default=None)
    # TODO: SIMULTANEOUS TRANSLATION!!!
    targets: list[TargetLang] | None = Field(default=None)

    input_stream: InputStream = Field(default_factory=InputStream)
    output_stream: OutputStream = Field(default_factory=OutputStream)
    preprocessing: Preprocessing = Field(default_factory=Preprocessing)
    translation_queue_configs: QueueConfigs = Field(default_factory=QueueConfigs)
    allowed_message_types: list[str] = [mt.value for mt in Message.ALLOWED_TYPES]

    silent: bool = Field(default=SILENT, exclude=True)
    log_file: Path | str | None = Field(default=LOG_FILE, exclude=True)
    debug: bool = Field(default=DEBUG, exclude=True)
    deep_debug: bool = Field(default=DEEP_DEBUG, exclude=True)
    timeout: int = Field(default=TIMEOUT, exclude=True)  # TODO!
    mode: Mode = Field(default=Mode.MIXED, exclude=True)

    trace_file: Path | str | None = Field(default=None, exclude=True)

    def __init__(
        self,
        source: SourceLang | None = None,
        targets: list[TargetLang] | TargetLang | None = None,
        **kwargs,
    ):
        super().__init__(__from_init=True, **kwargs)
        if not self.source:
            self.source = source
        if not self.targets:
            self.targets = targets

    def model_post_init(self, context: Any, /) -> None:
        if self.targets is None:
            self.targets = []
        elif isinstance(self.targets, TargetLang):
            self.targets = [self.targets]
        if self.log_file:
            self.log_file = Path(self.log_file).absolute()
            self.log_file.parent.mkdir(exist_ok=True, parents=True)
            self.trace_file = self.log_file.with_suffix(".trace.json")
        set_logging(self.silent, self.debug, self.log_file)
        super().model_post_init(context)

    @model_validator(mode="before")
    @classmethod
    def reconstruct_from_serialized(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        __from_init = data.pop("__from_init", False)

        # Extract pipeline if exists
        if "pipeline" in data:
            pipeline = data.pop("pipeline")
            data.update(pipeline)

        raw_source = data.pop("transcription", {})
        source_lang = raw_source.pop("source_language", None)
        if not __from_init and not source_lang:
            raise ConfigurationError(
                f"Source language must be specified in the transcription section. {raw_source}"
            )
        if source_lang:
            data["source"] = {"lang": source_lang, "transcription": raw_source}

        # Reconstruct targets if not present
        raw_targets = data.pop("translations", {})
        targets = []
        for raw_target in raw_targets:
            target_lang = raw_target.pop("target_language", None)
            if not __from_init and not target_lang:
                raise ConfigurationError(
                    f"Target language must be specified in the translation section. {raw_target}"
                )
            targets.append({"lang": target_lang, "translation": raw_target})
        data["targets"] = targets

        return data

    def model_dump(self, by_alias=True, exclude_none=False, **kwargs) -> dict[str, Any]:
        # Get base dump
        data = super().model_dump(
            by_alias=by_alias, exclude_none=exclude_none, **kwargs
        )

        # Extract source and targets
        source = data.pop("source")
        targets = data.pop("targets")

        # Build transcription with source_language
        transcription = source["transcription"].copy()
        transcription["source_language"] = source["lang"]

        # Build translations with target_language
        translations = []
        for target in targets:
            translation = target["translation"].copy()
            translation["target_language"] = target["lang"]
            translations.append(translation)

        # Build pipeline structure
        pipeline = {
            "preprocessing": data.pop("preprocessing"),
            "transcription": transcription,
            "translations": translations,
            "translation_queue_configs": data.pop("translation_queue_configs"),
            "allowed_message_types": data.pop("allowed_message_types"),
        }

        # Final structure
        result = {
            "input_stream": data.pop("input_stream"),
            "output_stream": data.pop("output_stream"),
            "pipeline": pipeline,
        }

        # Add any remaining fields
        result.update(data)

        return result

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self) -> str:
        return to_json(self.model_dump()).decode("utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        return cls.from_json(data)

    @classmethod
    def from_json(cls, data: str | dict) -> Config:
        if isinstance(data, str):
            data = from_json(data)
        return cls.model_validate(data)
