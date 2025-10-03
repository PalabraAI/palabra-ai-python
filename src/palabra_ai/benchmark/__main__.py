"""Palabra AI Benchmark - Data Collection Only"""

import argparse
import bisect
import re
import wave
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

import av
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

from palabra_ai import Config, PalabraAI, SourceLang, TargetLang
from palabra_ai.audio import save_wav
from palabra_ai.config import WsMode
from palabra_ai.constant import BYTES_PER_SAMPLE
from palabra_ai.enum import Kind
from palabra_ai.lang import Language
from palabra_ai.message import IoEvent
from palabra_ai.model import IoData
from palabra_ai.task.adapter.dummy import DummyWriter
from palabra_ai.task.adapter.file import FileReader
from palabra_ai.util.orjson import to_json

INPUT_CHUNK_DURATION_S = 0.1 # 100ms
FOCUSED = re.compile(r".+(_part_0)?$") # without part_1+ suffix

T = TypeVar("T")

def calculate_stats(values: list[float]) -> dict[str, float]:
    """Calculate min, max, avg, p50, p90, p95 for a list of values"""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "avg": sum(sorted_vals) / n,
        "p50": sorted_vals[int(n * 0.5)],
        "p90": sorted_vals[int(n * 0.9)],
        "p95": sorted_vals[int(n * 0.95)],
    }

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
class Report:
    sentences: dict[str, Sentence] = field(default_factory=dict) # transcription_id -> Sentence
    in_audio_stat: AudioStat | None = None
    out_audio_stat: AudioStat | None = None
    metrics_summary: dict[str, dict[str, float]] = field(default_factory=dict) # metric_name -> {min, max, avg, p50, p90, p95}

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
        focused = [e for e in sorted(io_data.events, key=lambda x: x.head.idx) if e.tid and FOCUSED.fullmatch(e.tid)]
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
                if fe.mtype not in mtypes:
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

        # Calculate metrics summary
        metrics_summary = {}
        for metric_name in ["metric_partial", "metric_validated", "metric_translated", "metric_tts_api", "metric_tts_playback"]:
            values = [getattr(s, metric_name) for s in sentences.values() if getattr(s, metric_name) is not None]
            if values:
                metrics_summary[metric_name] = calculate_stats(values)

        return cls(sentences=sentences, in_audio_stat=in_audio_stat, out_audio_stat=out_audio_stat, metrics_summary=metrics_summary), in_audio_canvas, out_audio_canvas


def create_histogram(values: list[float], bins: int = 20, width: int = 50) -> str:
    """Create simple ASCII histogram"""
    if not values:
        return "No data"
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        return f"All values: {min_val:.3f}s"

    bin_width = (max_val - min_val) / bins
    bin_counts = [0] * bins

    for val in values:
        bin_idx = min(int((val - min_val) / bin_width), bins - 1)
        bin_counts[bin_idx] += 1

    max_count = max(bin_counts)
    lines = []
    for i, count in enumerate(bin_counts):
        bin_start = min_val + i * bin_width
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "█" * bar_len
        lines.append(f"{bin_start:6.2f}s {bar} {count}")

    return "\n".join(lines)

def truncate_text(text: str, max_len: int = 25) -> str:
    """Truncate text to max_len chars, showing remaining count"""
    if len(text) <= max_len:
        return text
    remaining = len(text) - max_len
    return f"{text[:max_len]}...(+{remaining})"

def format_report(report: Report, io_data: IoData, source_lang: str, target_lang: str, in_file: str, out_file: str, config: Config) -> str:
    """Format report as text with tables and histogram"""
    lines = []
    lines.append("=" * 80)
    lines.append("PALABRA AI BENCHMARK REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Mode and audio info
    mode_name = "WebRTC" if io_data.mode == "webrtc" else "Websocket"
    lines.append(f"Mode: {mode_name}")

    # Input/Output info
    in_dur = f"{report.in_audio_stat.length_s:.1f}s" if report.in_audio_stat else "?.?s"
    out_dur = f"{report.out_audio_stat.length_s:.1f}s" if report.out_audio_stat else "?.?s"
    lines.append(f"Input:  [{in_dur}, {io_data.in_sr}hz, 16bit, PCM] {in_file}")
    lines.append(f"Output: [{out_dur}, {io_data.out_sr}hz, 16bit, PCM] {out_file}")

    # TTS autotempo info
    queue_config = config.translation_queue_configs.global_ if config.translation_queue_configs else None
    if queue_config:
        if queue_config.auto_tempo:
            lines.append(f"TTS autotempo: ✅ on ({queue_config.min_tempo}-{queue_config.max_tempo})")
        else:
            lines.append(f"TTS autotempo: ❌ off")
    lines.append("")

    # Metrics summary table
    if report.metrics_summary:
        lines.append("METRICS SUMMARY")
        lines.append("-" * 80)
        table = PrettyTable()
        table.field_names = ["Metric", "Min", "Max", "Avg", "P50", "P90", "P95"]

        metric_labels = {
            "metric_partial": "Partial",
            "metric_validated": "Validated",
            "metric_translated": "Translated",
            "metric_tts_api": "TTS API",
            "metric_tts_playback": "TTS Playback"
        }

        for metric_name, stats in report.metrics_summary.items():
            label = metric_labels.get(metric_name, metric_name)
            table.add_row([
                label,
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}",
                f"{stats['avg']:.3f}",
                f"{stats['p50']:.3f}",
                f"{stats['p90']:.3f}",
                f"{stats['p95']:.3f}"
            ])

        lines.append(str(table))
        lines.append("")

    # Sentences breakdown
    if report.sentences:
        lines.append("SENTENCES BREAKDOWN")
        lines.append("-" * 80)
        table = PrettyTable()
        table.field_names = ["Start", "Validated", "Translated", "Part", "Valid", "Trans", "TTS API", "TTS Play"]
        table.align["Validated"] = "l"
        table.align["Translated"] = "l"

        sorted_sentences = sorted(report.sentences.items(), key=lambda x: x[1].local_start_ts)
        global_start = sorted_sentences[0][1].local_start_ts if sorted_sentences else 0

        for tid, sentence in sorted_sentences:
            start_time = sentence.local_start_ts - global_start
            table.add_row([
                f"{start_time:.1f}s",
                truncate_text(sentence.validated_text),
                truncate_text(sentence.translated_text),
                f"{sentence.metric_partial:.2f}" if sentence.metric_partial else "-",
                f"{sentence.metric_validated:.2f}" if sentence.metric_validated else "-",
                f"{sentence.metric_translated:.2f}" if sentence.metric_translated else "-",
                f"{sentence.metric_tts_api:.2f}" if sentence.metric_tts_api else "-",
                f"{sentence.metric_tts_playback:.2f}" if sentence.metric_tts_playback else "-"
            ])

        lines.append(str(table))
        lines.append("")

    # Histogram for TTS playback
    if "metric_tts_playback" in report.metrics_summary:
        lines.append("TTS PLAYBACK HISTOGRAM")
        lines.append("-" * 80)
        playback_values = [s.metric_tts_playback for s in report.sentences.values() if s.metric_tts_playback is not None]
        lines.append(create_histogram(playback_values))
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Palabra AI Benchmark - Data Collection")
    parser.add_argument("audio", help="Audio file path")
    parser.add_argument("source_lang", nargs="?", help="Source language")
    parser.add_argument("target_lang", nargs="?", help="Target language")
    parser.add_argument("--config", type=Path, help="JSON config file")
    parser.add_argument("--out", type=Path, help="Output directory for files (if not specified, only prints to console)")

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    mode = WsMode(input_chunk_duration_ms=INPUT_CHUNK_DURATION_S*1000)

    # Get audio duration for progress tracking
    with av.open(str(audio_path)) as container:
        audio_duration = container.duration / 1000000  # convert microseconds to seconds

    # Create reader
    reader = FileReader(str(audio_path))

    # Create progress bar placeholder (will update desc after config loaded)
    last_timestamp = [0.0]  # mutable to allow updates in nested function
    progress_bar = [None]  # mutable reference

    def on_transcription(msg):
        if hasattr(msg, 'segments') and msg.segments:
            end_ts = msg.segments[-1].end
            if end_ts > last_timestamp[0]:
                last_timestamp[0] = end_ts
                progress_pct = min(100, (end_ts / audio_duration) * 100)
                if progress_bar[0]:
                    progress_bar[0].update(progress_pct - progress_bar[0].n)

    if args.config:
        # Load full config from JSON
        config = Config.from_json(args.config.read_text())

        # Override benchmark-specific settings (using private attrs)
        config.source._reader = reader
        config.source._on_transcription = on_transcription
        config.targets[0]._writer = DummyWriter()
        config.benchmark = True

        source_lang = config.source.lang.code
        target_lang = config.targets[0].lang.code
    else:
        if not args.source_lang or not args.target_lang:
            parser.error("source_lang and target_lang required without --config")
        source_lang = args.source_lang
        target_lang = args.target_lang

        config = Config(
            source=SourceLang(Language.get_or_create(source_lang), reader, on_transcription=on_transcription),
            targets=[TargetLang(Language.get_or_create(target_lang), DummyWriter())],
            benchmark=True,
            mode=mode,
        )

    # Create progress bar with language info
    progress_bar[0] = tqdm(
        total=100,
        desc=f"Processing {source_lang}→{target_lang}",
        unit="%",
        mininterval=7.0,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]"
    )

    print(f"Running benchmark: {source_lang} → {target_lang}")
    if args.out:
        print(f"Files will be saved to {args.out}")
    print("-" * 60)

    palabra = PalabraAI()
    result = palabra.run(config, no_raise=True)

    # Complete and close progress bar
    if progress_bar[0]:
        progress_bar[0].update(100 - progress_bar[0].n)
        progress_bar[0].close()

    if not result.ok or not result.io_data:
        raise RuntimeError(f"Benchmark failed: {result.exc}")

    # Parse report
    report, in_audio_canvas, out_audio_canvas = Report.parse(result.io_data)

    # Create file paths (used in report and optionally saved with --out)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    in_wav_name = f"{timestamp}_bench_in_{source_lang}.wav"
    out_wav_name = f"{timestamp}_bench_out_{target_lang}.wav"

    # Generate text report
    report_text = format_report(
        report,
        result.io_data,
        source_lang,
        target_lang,
        str(audio_path),
        out_wav_name,
        config
    )

    if args.out:
        # Save all files to output directory
        output_dir = args.out
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_result_path = output_dir / f"{timestamp}_bench_raw_result.json"
        io_data_path = output_dir / f"{timestamp}_bench_io_data.json"
        report_path = output_dir / f"{timestamp}_bench_report.json"
        report_txt_path = output_dir / f"{timestamp}_bench_report.txt"
        in_wav_path = output_dir / in_wav_name
        out_wav_path = output_dir / out_wav_name

        raw_result_path.write_bytes(to_json(result.model_dump(), True))
        io_data_path.write_bytes(to_json(result.io_data, True))
        report_path.write_bytes(to_json(report, True))
        report_txt_path.write_text(report_text)

        save_wav(in_audio_canvas, in_wav_path, result.io_data.in_sr, result.io_data.channels)
        save_wav(out_audio_canvas, out_wav_path, result.io_data.out_sr, result.io_data.channels)

    # Always print report to console
    print("\n" + report_text)


if __name__ == "__main__":
    main()
