import bisect
import re
from base64 import b64decode

from collections import defaultdict

from dataclasses import dataclass

from dataclasses import field
from pathlib import Path
from typing import Any
from typing import NamedTuple
from typing import Self
from typing import TypeVar

import numpy as np
from prettytable import PrettyTable

from palabra_ai import Config
from palabra_ai import Message
from palabra_ai.audio import save_wav
from palabra_ai.benchmark.utils import flatten_container_to_paths, _format_value

from palabra_ai.message import IoEvent

from palabra_ai.model import IoData

from palabra_ai.util.orjson import to_json

INPUT_CHUNK_DURATION_S = 0.1 # 100ms
FOCUSED = re.compile(r"^(?!.*_part_[1-9]\d*).*$") # without part_1+ suffix
BENCHMARK_ALLOWED_MESSAGE_TYPES = [mt.value for mt in Message.ALLOWED_TYPES]
T = TypeVar("T")


class Tid(NamedTuple):
    """Parsed transcription ID with base and optional part number"""
    base: str
    part_num: int | None

    @property
    def display(self) -> str:
        """Format for display: base[N] if has part_num, else just base"""
        if self.part_num is not None:
            return f"{self.base}[{self.part_num}]"
        return self.base

    @property
    def raw(self) -> str:
        """Reconstruct original raw transcription_id"""
        if self.part_num is not None:
            return f"{self.base}_part_{self.part_num}"
        return self.base

    @classmethod
    def parse(cls, raw_tid: str) -> "Tid":
        """Parse raw transcription_id into base and part_num

        Examples:
            sentence_1 -> Tid(base='sentence_1', part_num=None)
            sentence_2_part_0 -> Tid(base='sentence_2', part_num=0)
            sentence_2_part_1 -> Tid(base='sentence_2', part_num=1)
        """
        match = re.search(r'^(.+)_part_(\d+)$', raw_tid)
        if match:
            return cls(base=match.group(1), part_num=int(match.group(2)))
        return cls(base=raw_tid, part_num=None)


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

    @property
    def has_metrics(self) -> bool:
        # Consider sentence to have metrics if it has validated/translated/tts timing,
        # even if partial_transcription is missing (fallback case)
        return self.metric_validated is not None


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
    set_task_e: IoEvent | None = None
    current_task_e: IoEvent | None = None

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
        sentences = {}

        all_with_tid: list[IoEvent] = []
        focused: list[IoEvent] = []
        extra_parts: list[IoEvent] = []
        in_evs: list[IoEvent] = []
        out_evs: list[IoEvent] = []
        set_task_e: IoEvent|None = None
        current_task_e: IoEvent|None = None
        for e in sorted(io_data.events, key=lambda x: x.head.idx):
            if e.tid:
                all_with_tid.append(e)
                if FOCUSED.fullmatch(e.tid):
                    focused.append(e)
                else:
                    extra_parts.append(e)
            if e.mtype == "input_audio_data":
                in_evs.append(e)
            elif e.mtype == "output_audio_data":
                out_evs.append(e)
            elif e.mtype == "set_task":
                if set_task_e is not None:
                    raise ValueError("Multiple set_task events found, old: {}, new: {}".format(set_task_e, e))
                set_task_e = e
            elif e.mtype == "current_task" and not current_task_e:
                if current_task_e is not None:
                    raise ValueError("Multiple current_task events found, old: {}, new: {}".format(current_task_e, e))
                current_task_e = e

        focused_by_tid = defaultdict(list)
        for fe in focused:
            focused_by_tid[fe.tid].append(fe)

        in_evs_by_dawn = {e.head.dawn_ts:e for e in in_evs}
        in_audio_canvas, in_audio_stat = cls.playback(in_evs, io_data.in_sr, io_data.channels)
        out_audio_canvas, out_audio_stat = cls.playback(out_evs, io_data.out_sr, io_data.channels)

        for raw_tid, fes in focused_by_tid.items():
            mtypes = {}
            for fe in fes:
                if fe.mtype not in mtypes:
                    mtypes[fe.mtype] = fe

            partial = mtypes.get("partial_transcription")
            validated = mtypes.get("validated_transcription")
            translated = mtypes.get("translated_transcription")
            out_audio = mtypes.get("output_audio_data")

            # Require validated, translated, out_audio + (partial OR validated)
            if not all([validated, translated, out_audio]):
                continue

            # Handle missing partial_transcription by falling back to validated_transcription
            if not partial:
                # Use validated_transcription as fallback for timing calculations
                timing_source = validated
                partial_text = ""  # No partial text available
                partial_ts = None  # No partial timestamp available
            else:
                timing_source = partial
                partial_text = partial.body["data"]["transcription"]["text"]
                partial_ts = partial.head.dawn_ts

            asr_start = timing_source.body["data"]["transcription"]["segments"][0]["start"]
            nearest_in = cls.predecessor(in_evs_by_dawn, asr_start)
            if not nearest_in:
                continue
            _, nearest_in_ev = nearest_in
            local_start_ts = nearest_in_ev.head.dawn_ts

            playback_tts_ts = out_audio_stat.tids_with_actual_tts_playback.get(raw_tid)

            sentences[raw_tid] = Sentence(
                transcription_id=raw_tid,
                local_start_ts=local_start_ts,
                local_start_chunk_idx=nearest_in_ev.head.idx,
                partial_ts=partial_ts,
                validated_ts=validated.head.dawn_ts,
                translated_ts=translated.head.dawn_ts,
                tts_api_ts=out_audio.head.dawn_ts,
                partial_text=partial_text,
                validated_text=validated.body["data"]["transcription"]["text"],
                translated_text=translated.body["data"]["transcription"]["text"],
                metric_partial=partial_ts - local_start_ts if partial_ts else None,
                metric_validated=validated.head.dawn_ts - local_start_ts,
                metric_translated=translated.head.dawn_ts - local_start_ts,
                metric_tts_api=out_audio.head.dawn_ts - local_start_ts,
                metric_tts_playback=(playback_tts_ts - local_start_ts) if playback_tts_ts else None,
            )

        # Build registry of base_tid -> local_start_ts for parent sentences
        parent_timestamps = {}
        for raw_tid, sentence in sentences.items():
            _tid = Tid.parse(raw_tid)
            if _tid.base not in parent_timestamps:
                parent_timestamps[_tid.base] = sentence.local_start_ts

        # Process extra_parts (_part_1+) - text only, no metrics
        extra_parts_by_tid = defaultdict(list)
        for ep in extra_parts:
            extra_parts_by_tid[ep.tid].append(ep)

        # EXTRA SENTENCES (no metrics, just text)
        for raw_tid, eps in extra_parts_by_tid.items():
            mtypes = {}
            for ep in eps:
                if ep.mtype not in mtypes:
                    mtypes[ep.mtype] = ep

            validated = mtypes.get("validated_transcription")
            translated = mtypes.get("translated_transcription")

            # Show what we have - need at least one of validated or translated
            if not validated and not translated:
                continue

            # Get parent timestamp for proper sorting
            _tid = Tid.parse(raw_tid)
            parent_ts = parent_timestamps.get(_tid.base)

            if parent_ts is None:
                # DEPRECATED: This warning should no longer occur after fallback implementation
                print(f"⚠️  WARNING: No parent sentence found for {raw_tid} (base: {_tid.base})")
                continue

            sentences[raw_tid] = Sentence(
                transcription_id=raw_tid,
                local_start_ts=parent_ts,  # Use parent's timestamp for sorting
                local_start_chunk_idx=0,  # Not used for extra_parts
                validated_text=validated.body["data"]["transcription"]["text"] if validated else "",
                translated_text=translated.body["data"]["transcription"]["text"] if translated else "",
                # All metrics stay None
            )

        # Note: sentences may be empty in some cases, which is valid

        # Calculate metrics summary
        metrics_summary = {}
        for metric_name in ["metric_partial", "metric_validated", "metric_translated", "metric_tts_api", "metric_tts_playback"]:
            values = [getattr(s, metric_name) for s in sentences.values() if s.has_metrics and getattr(s, metric_name) is not None]
            if values:
                metrics_summary[metric_name] = calculate_stats(values)

        return cls(sentences=sentences, in_audio_stat=in_audio_stat, out_audio_stat=out_audio_stat, metrics_summary=metrics_summary, set_task_e=set_task_e, current_task_e=current_task_e), in_audio_canvas, out_audio_canvas


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


def merge_task_settings(sent_paths: list[tuple[str, Any]], applied_paths: list[tuple[str, Any]]) -> list[tuple[str, Any, Any]]:
    """Merge sent and applied task settings into full outer join.

    Args:
        sent_paths: Flattened paths from set_task
        applied_paths: Flattened paths from current_task

    Returns:
        List of (key, sent_value, applied_value) tuples.
        None values indicate missing data in that column.
        Sorted: keys with SENT values first, then keys with APPLIED values, alphabetically.
    """
    sent_dict = dict(sent_paths)
    applied_dict = dict(applied_paths)

    # Get all unique keys
    all_keys = set(sent_dict.keys()) | set(applied_dict.keys())

    # Create merged tuples
    result = []
    for key in all_keys:
        sent_value = sent_dict.get(key)
        applied_value = applied_dict.get(key)
        result.append((key, sent_value, applied_value))

    # Sort: keys with SENT values first, then keys with APPLIED values, alphabetically
    def sort_key(item):
        key, sent_value, applied_value = item
        if sent_value is not None:
            # Has SENT value - priority 0
            return (0, key)
        else:
            # Only has APPLIED value - priority 1
            return (1, key)

    result.sort(key=sort_key)
    return result


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

    # CONFIG - comparison of sent vs applied settings
    lines.append("")
    lines.append("CONFIG (sent vs applied)")
    lines.append("-" * 80)

    set_task_data = report.set_task_e.body["data"] if report.set_task_e else {}
    current_task_data = report.current_task_e.body["data"] if report.current_task_e else {}

    sent_paths = [(k, _format_value(v)) for k,v in flatten_container_to_paths(set_task_data)]
    applied_paths = [(k, _format_value(v)) for k,v in flatten_container_to_paths(current_task_data)]

    # Merge settings using full outer join
    merged_settings = merge_task_settings(sent_paths, applied_paths)

    table = PrettyTable()
    table.field_names = ["Key", "Sent", "Applied"]
    table.align["Key"] = "l"
    table.align["Sent"] = "l"
    table.align["Applied"] = "l"

    for key, sent_value, applied_value in merged_settings:
        sent_str = sent_value if sent_value is not None else ""
        applied_str = applied_value if applied_value is not None else ""
        table.add_row([key, sent_str, applied_str])

    lines.append(str(table))
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
        table.field_names = ["Start", "ID", "Validated", "Translated", "Part", "Valid", "Trans", "TTS API", "TTS Play"]
        table.align["ID"] = "l"
        table.align["Validated"] = "l"
        table.align["Translated"] = "l"

        sorted_sentences = sorted(report.sentences.items(), key=lambda x: x[1].local_start_ts)
        global_start = sorted_sentences[0][1].local_start_ts if sorted_sentences else 0

        for raw_tid, sentence in sorted_sentences:
            tid = Tid.parse(raw_tid)

            if sentence.has_metrics:
                start_time = sentence.local_start_ts - global_start
                table.add_row([
                    f"{start_time:.1f}s",
                    tid.display,
                    truncate_text(sentence.validated_text),
                    truncate_text(sentence.translated_text),
                    f"{sentence.metric_partial:.2f}" if sentence.metric_partial is not None else "-",
                    f"{sentence.metric_validated:.2f}" if sentence.metric_validated else "-",
                    f"{sentence.metric_translated:.2f}" if sentence.metric_translated else "-",
                    f"{sentence.metric_tts_api:.2f}" if sentence.metric_tts_api else "-",
                    f"{sentence.metric_tts_playback:.2f}" if sentence.metric_tts_playback else "-"
                ])
            else:
                # Text-only row for extra_parts (_part_1+)
                table.add_row([
                    "",  # no start time
                    tid.display,
                    truncate_text(sentence.validated_text),
                    truncate_text(sentence.translated_text),
                    "", "", "", "", ""  # no metrics
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


def save_benchmark_files(
    output_dir: Path,
    timestamp: str,
    report: Report,
    io_data: IoData,
    config: Config,
    result,  # RunResult
    in_audio_canvas,  # np.ndarray
    out_audio_canvas,  # np.ndarray
    source_lang: str,
    target_lang: str,
    report_text: str,
    input_file_path: str,
    file_prefix: str = "bench"
) -> None:
    """Save benchmark files to output directory"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create file names based on prefix
    in_wav_name = f"{timestamp}_{file_prefix}_in_{source_lang}.wav"
    out_wav_name = f"{timestamp}_{file_prefix}_out_{target_lang}.wav"

    # Define file paths
    raw_result_path = output_dir / f"{timestamp}_{file_prefix}_raw_result.json" if result else None
    io_data_path = output_dir / f"{timestamp}_{file_prefix}_io_data.json"
    report_path = output_dir / f"{timestamp}_{file_prefix}_report.json"
    report_txt_path = output_dir / f"{timestamp}_{file_prefix}_report.txt"
    in_wav_path = output_dir / in_wav_name
    out_wav_path = output_dir / out_wav_name

    # Save JSON files
    if result and raw_result_path:
        raw_result_path.write_bytes(to_json(result.model_dump(), True))
    io_data_path.write_bytes(to_json(io_data, True))
    report_path.write_bytes(to_json(report, True))
    report_txt_path.write_text(report_text)

    # Save audio files
    save_wav(in_audio_canvas, in_wav_path, io_data.in_sr, io_data.channels)
    save_wav(out_audio_canvas, out_wav_path, io_data.out_sr, io_data.channels)
