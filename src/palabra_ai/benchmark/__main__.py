"""Palabra AI Benchmark - Data Collection Only"""

import argparse
import bisect
import re
import wave
from audioop import byteswap
from base64 import b64decode
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Self
from typing import TypeVar

import numpy as np

from palabra_ai import Config, PalabraAI, SourceLang, TargetLang
from palabra_ai.audio import save_wav
from palabra_ai.constant import BYTES_PER_SAMPLE
from palabra_ai.enum import Kind
from palabra_ai.lang import Language
from palabra_ai.message import IoEvent
from palabra_ai.model import IoData
from palabra_ai.task.adapter.dummy import DummyWriter
from palabra_ai.task.adapter.file import FileReader
from palabra_ai.util.orjson import to_json

FOCUSED = re.compile(r".+(_part_0)?$") # without part_1+ suffix

T = TypeVar("T")

@dataclass
class Sentence:
    """
    Complete sentence data with timestamps and metrics

    Timestamps:
    - global_start_ts: when first input audio chunk was sent to API (t=0 for whole session)
    - local_start_ts: when input audio chunk containing this sentence start was sent

    Metrics (all calculated):
    - metric_partial: local_start → first partial transcription
    - metric_validated: local_start → validated transcription
    - metric_translated: local_start → translated transcription
    - metric_tts_api: local_start → first TTS output chunk arrived from API
    - metric_tts_playback: local_start → when TTS can actually play (accounting for queue)
    """
    transcription_id: str

    # Core timestamps
    local_start_ts: float   # Input chunk where this sentence started
    local_start_chunk_idx: int

    # Event timestamps (when events occurred)
    partial_ts: float | None = None
    validated_ts: float | None = None
    translated_ts: float | None = None
    tts_api_ts: float | None = None  # When first output chunk with this transcription_id arrived

    # Calculated metrics (populated by analyze stage)
    metric_partial: float | None = None
    metric_validated: float | None = None
    metric_translated: float | None = None
    metric_tts_api: float | None = None
    metric_tts_playback: float | None = None

    in_deltas: dict[int, float] = field(default_factory=dict) # chunk idx -> delta to apply
    out_deltas: dict[int, float] = field(default_factory=dict) # chunk idx
    out_tids_with_playback: dict[str, float] = field(default_factory=dict) # tid -> actual playback start pos

    # Text content
    partial_text: str = ""
    validated_text: str = ""
    translated_text: str = ""

@dataclass
class AudioStat:
    length_s: float
    tids_with_actual_tts_playback: dict[str, float] # tid -> actual playback start pos
    deltas: dict[int, float] # chunk idx -> delta to apply


@dataclass
class Reg:
    sentences: dict[str, Sentence] = field(default_factory=dict) # transcription_id -> Sentence
    in_audio_stat: AudioStat | None = None
    out_audio_stat: AudioStat | None = None

    @staticmethod
    def predecessor(d: dict[float, T], x: float) -> tuple[float, T] | None:
        keys = list(d.keys())
        i = bisect.bisect_right(keys, x)
        if i == 0:
            return None
        k = keys[i - 1]
        return k, d[k]

    @classmethod
    def put_audio_to_canvas(cls, audio_canvas: np.typing.NDArray, start_idx: int, e: IoEvent):
        raw_samples = b64decode(e.body["data"]["data"])
        chunk = np.frombuffer(raw_samples, dtype=np.int16)
        audio_canvas[start_idx:start_idx + len(chunk)] += chunk

    @classmethod
    def playback(cls, events: list[IoEvent], sr: int, ch: int):
        playback_pos = 0.0
        tids_with_actual_tts_playback: dict[str, float] = {} # tid -> actual playback start pos
        deltas: dict[int, float] = {} # chunk idx -> delta to apply
        audio_map: dict[float, IoEvent] = {}
        for e in events:
            deltas[e.head.idx] = playback_pos - e.head.dawn_ts
            start_pos = max(playback_pos, e.head.dawn_ts)
            if e.tid and e.tid not in tids_with_actual_tts_playback:
                tids_with_actual_tts_playback[e.tid] = start_pos
            audio_map[start_pos] = e
            playback_pos = start_pos + e.head.dur_s
        audio_canvas = np.zeros(sr * int(playback_pos + 1), dtype=np.int16)
        for start_pos, e in sorted(audio_map.items()):
            start_idx_rough = int(start_pos * sr * ch)
            start_idx_aligned = round(start_idx_rough / ch) * ch
            cls.put_audio_to_canvas(audio_canvas, start_idx_aligned, e)
        return audio_canvas, AudioStat(playback_pos, tids_with_actual_tts_playback, deltas)
        return playback_pos, audio_canvas, deltas, tids_with_actual_tts_playback


    @classmethod
    def parse(cls, io_data: IoData) -> Self:
        playback_pos = 0.0
        sentences = {}
        focused = [e for e in io_data.events if e.tid and FOCUSED.fullmatch(e.tid)]
        focused_by_tid = defaultdict(list)
        for fe in focused:
            focused_by_tid[fe.tid].append(fe)

        in_evs = [e for e in io_data.events if e.mtype == "input_audio_data"]
        out_evs = [e for e in io_data.events if e.mtype == "output_audio_data"]
        in_evs_by_dawn = {e.head.dawn_ts:e for e in in_evs}
        # out_by_idx = {e.head.idx:e for e in out_evs}
        in_audio_canvas, in_audio_stat = cls.playback(in_evs, io_data.in_sr, io_data.channels)
        out_audio_canvas, out_audio_stat = cls.playback(out_evs, io_data.out_sr, io_data.channels)

        for tid, fes in focused_by_tid.items():
            mtypes = {}
            for fe in fes:
                if not fe.mtype in mtypes:
                    mtypes[fe.mtype] = fe
            # mtypes = {e.mtype:e for e in reversed(fes)} # first event of each type
            partial = mtypes.get("partial_transcription")
            validated = mtypes.get("validated_transcription")
            translated = mtypes.get("translated_transcription")
            out_audio = mtypes.get("output_audio_data")
            if not all([partial, validated, translated, out_audio]):
                continue

            asr_start = partial.body["data"]["transcription"]["segments"][0]["start"]
            nearest_in = cls.predecessor(in_evs_by_dawn, asr_start)
            if not nearest_in:
                continue
            _, nearest_in_ev = nearest_in
            local_start_ts = nearest_in_ev.head.dawn_ts

            playback_tts_ts = out_audio_stat.tids_with_actual_tts_playback.get(tid)

            sentences[tid] = Sentence(
                transcription_id=tid,
                local_start_ts=local_start_ts,
                local_start_chunk_idx=nearest_in_ev.head.idx,
                partial_ts=partial.head.dawn_ts,
                validated_ts=validated.head.dawn_ts,
                translated_ts=translated.head.dawn_ts,
                tts_api_ts=out_audio.head.dawn_ts,
                partial_text=partial.body["data"]["transcription"]["text"],
                validated_text=validated.body["data"]["transcription"]["text"],
                translated_text=translated.body["data"]["transcription"]["text"],
                metric_partial=partial.head.dawn_ts - local_start_ts,
                metric_validated=validated.head.dawn_ts - local_start_ts,
                metric_translated=translated.head.dawn_ts - local_start_ts,
                metric_tts_api=out_audio.head.dawn_ts - local_start_ts,
                metric_tts_playback=(playback_tts_ts - local_start_ts) if playback_tts_ts else None,
            )


        return cls(sentences=sentences, in_audio_stat=in_audio_stat, out_audio_stat=out_audio_stat), in_audio_canvas, out_audio_canvas


def main():
    parser = argparse.ArgumentParser(description="Palabra AI Benchmark - Data Collection")
    parser.add_argument("audio", help="Audio file path")
    parser.add_argument("source_lang", nargs="?", help="Source language")
    parser.add_argument("target_lang", nargs="?", help="Target language")
    parser.add_argument("--config", type=Path, help="JSON config file")
    parser.add_argument("--output-dir", type=Path, default=Path("1Oct25"), help="Output directory")

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    if args.config:
        config = Config.from_json(args.config.read_text())
        source_lang = config.source.lang.code
        target_lang = config.targets[0].lang.code
    else:
        if not args.source_lang or not args.target_lang:
            parser.error("source_lang and target_lang required without --config")
        source_lang = args.source_lang
        target_lang = args.target_lang

        config = Config(
            source=SourceLang(Language.get_or_create(source_lang), FileReader(str(audio_path))),
            targets=[TargetLang(Language.get_or_create(target_lang), DummyWriter())],
            benchmark=True
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    in_wav_path = output_dir / f"{timestamp}_bench_in_{source_lang}.wav"
    out_wav_path = output_dir / f"{timestamp}_bench_out_{target_lang}.wav"
    raw_result_path = output_dir  / f"{timestamp}_bench_raw_result.json"
    io_data_path = output_dir / f"{timestamp}_bench_io_data.json"
    reg_path = output_dir / f"{timestamp}_bench_reg.json"

    print(f"Running benchmark: {source_lang} → {target_lang}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    palabra = PalabraAI()
    result = palabra.run(config, no_raise=True, )

    if not result.ok or not result.io_data:
        raise RuntimeError(f"Benchmark failed: {result.exc}")

    print("Collecting data...")

    # Save raw result
    raw_result_path.write_bytes(
        to_json(result.model_dump(), True)
    )
    print(f"✓ Saved raw result")

    # Save io_data summary
    io_data_path.write_bytes(
        to_json(result.io_data, True)
    )
    print(f"✓ Saved io_data summary")

    # Save analyzed registry
    reg, in_audio_canvas, out_audio_canvas = Reg.parse(result.io_data)
    reg_path.write_bytes(
        to_json(reg, True)
    )


    save_wav(in_audio_canvas,in_wav_path,result.io_data.in_sr,
        result.io_data.channels
    )
    print(f"✓ Saved input audio")

    save_wav(
        out_audio_canvas,
        out_wav_path,
        result.io_data.out_sr,
        result.io_data.channels
    )
    print(f"✓ Saved output audio")

    print(f"\n✅ Done! Files in {output_dir}/")
    print(f"   - benchmark_raw_result_{timestamp}.json")
    print(f"   - benchmark_io_data_{timestamp}.json")
    print(f"   - benchmark_reg_{timestamp}.json")
    print(f"   - benchmark_in_{source_lang}_{timestamp}.wav")
    print(f"   - benchmark_out_{target_lang}_{timestamp}.wav")


if __name__ == "__main__":
    main()
