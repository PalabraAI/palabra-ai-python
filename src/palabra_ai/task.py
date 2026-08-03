from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# server default for pipeline.allowed_message_types
DEFAULT_MESSAGE_TYPES = (
    "translated_transcription",
    "partial_transcription",
    "validated_transcription",
)

# a glossary option: True = ALL your glossaries of the kind matching the
# language apply, False = none, sequence = only these (ids or Glossary
# objects), None = server default (same as True)
GlossaryOption = bool | Sequence[Any] | None


def _glossary_settings(value: Any, prefix: str) -> dict[str, Any]:
    """allow_*_glossaries / allowed_*_glossary_ids pair for one glossary kind."""
    if isinstance(value, bool):
        return {f"allow_{prefix}_glossaries": value}
    ids = [g.id if hasattr(g, "id") else str(g) for g in value]
    return {f"allow_{prefix}_glossaries": True, f"allowed_{prefix}_glossary_ids": ids}


def build_task(
    source: str,
    targets: str | Sequence[str] | Mapping[str, dict[str, Any]],
    *,
    input_format: str = "pcm_s16le",  # pcm_s16le | wav | opus
    input_sample_rate: int = 24000,  # 16000-48000
    input_channels: int = 1,
    output_format: str = "pcm_s16le",  # pcm_s16le | zlib_pcm_s16le
    voice_id: str | None = None,
    voice_cloning: bool = False,
    translate_partials: bool = False,
    silence_threshold: float | None = None,
    message_types: Sequence[str] | None = None,
    transcription: dict[str, Any] | None = None,
    hotwords_glossaries: GlossaryOption = None,
    verification_glossaries: GlossaryOption = None,
    translation_glossaries: GlossaryOption = None,
) -> dict[str, Any]:
    """Build a set_task "data" payload for the WebSocket audio transport.

    source may be "auto". voice_id/voice_cloning apply to every target;
    per-target overrides win. With translate_partials the
    partial_translated_transcription type is enabled automatically.
    Extra transcription keys are merged into pipeline.transcription.

    The *_glossaries options control which of your glossaries apply
    (create them via Palabra.glossaries). True -- ALL your enabled
    glossaries of the kind matching the language apply (hotwords and
    verification: the source language; translation: the source->target
    pair). False -- none. A list of ids (or Glossary objects) -- only
    those. None -- server default, which is the same as True.
    translation_glossaries applies to every target; a per-target
    "allowed_translation_glossary_ids" override wins.
    """
    if isinstance(targets, str):
        targets = [targets]
    if isinstance(targets, Mapping):
        target_items = list(targets.items())
    else:
        target_items = [(lang, {}) for lang in targets]

    translations = []
    for lang, overrides in target_items:
        speech_gen: dict[str, Any] = {}
        if voice_cloning:
            speech_gen["voice_cloning"] = True
        elif voice_id is not None:
            speech_gen["voice_id"] = voice_id
        tr: dict[str, Any] = {"target_language": lang, "speech_generation": speech_gen}
        if translate_partials:
            tr["translate_partial_transcriptions"] = True
        if translation_glossaries is not None:
            tr.update(_glossary_settings(translation_glossaries, "translation"))
        for k, v in overrides.items():
            if k == "speech_generation" and isinstance(v, dict):
                tr["speech_generation"] = {**speech_gen, **v}
            else:
                tr[k] = v
        translations.append(tr)

    transcription_cfg: dict[str, Any] = {"source_language": source}
    if silence_threshold is not None:
        transcription_cfg["segment_confirmation_silence_threshold"] = silence_threshold
    if transcription:
        transcription_cfg.update(transcription)
    if hotwords_glossaries is not None:
        transcription_cfg.update(_glossary_settings(hotwords_glossaries, "hotwords"))
    if verification_glossaries is not None:
        verification = transcription_cfg.setdefault("verification", {})
        verification.update(_glossary_settings(verification_glossaries, "verification"))

    if message_types is None:
        message_types = list(DEFAULT_MESSAGE_TYPES)
        if translate_partials:
            message_types.append("partial_translated_transcription")

    return {
        "input_stream": {
            "content_type": "audio",
            "source": {
                "type": "ws",
                "format": input_format,
                "sample_rate": input_sample_rate,
                "channels": input_channels,
            },
        },
        "output_stream": {
            "content_type": "audio",
            "target": {"type": "ws", "format": output_format},
        },
        "pipeline": {
            "transcription": transcription_cfg,
            "translations": translations,
            "allowed_message_types": list(message_types),
        },
    }
