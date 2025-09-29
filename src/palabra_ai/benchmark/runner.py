"""
Main runner for Palabra AI benchmark
Handles audio processing with DummyWriter and progress tracking
"""

import asyncio
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass  # uvloop not available, use default event loop

from palabra_ai import Config, PalabraAI, SourceLang, TargetLang
from palabra_ai.lang import Language, is_valid_source_language, is_valid_target_language
from palabra_ai.task.adapter.dummy import DummyWriter
from palabra_ai.task.adapter.file import FileReader, FileWriter
from palabra_ai.util.logger import debug, error, info, warning
from palabra_ai.util.orjson import from_json, to_json

from .analyzer import analyze_latency
from .reporter import generate_html_report, generate_json_report, generate_text_report


def create_stereo_debug_mix(original_path: Path, tts_path: Path, output_path: Path):
    """
    Create stereo mix with original audio in left channel and TTS in right channel.
    Both are resampled to 48kHz and padded to same length.
    """
    try:
        # Load original audio (16kHz)
        original_audio, original_sr = librosa.load(str(original_path), sr=None)
        debug(f"Loaded original audio: {len(original_audio)} samples at {original_sr}Hz")

        # Load TTS audio (24kHz)
        tts_audio, tts_sr = librosa.load(str(tts_path), sr=None)
        debug(f"Loaded TTS audio: {len(tts_audio)} samples at {tts_sr}Hz")

        # Resample both to 48kHz
        target_sr = 48000
        original_resampled = librosa.resample(original_audio, orig_sr=original_sr, target_sr=target_sr)
        tts_resampled = librosa.resample(tts_audio, orig_sr=tts_sr, target_sr=target_sr)

        debug(f"Resampled original: {len(original_resampled)} samples at {target_sr}Hz")
        debug(f"Resampled TTS: {len(tts_resampled)} samples at {target_sr}Hz")

        # Pad to same length
        max_length = max(len(original_resampled), len(tts_resampled))
        original_padded = np.pad(original_resampled, (0, max_length - len(original_resampled)), mode='constant')
        tts_padded = np.pad(tts_resampled, (0, max_length - len(tts_resampled)), mode='constant')

        # Create stereo array: [left, right] = [original, tts]
        stereo_audio = np.vstack([original_padded, tts_padded]).T

        # Save stereo mix
        sf.write(str(output_path), stereo_audio, target_sr)
        debug(f"Saved stereo debug mix to: {output_path}")

    except Exception as e:
        error(f"Failed to create stereo debug mix: {e}")
        import traceback
        traceback.print_exc()



class BenchmarkRunner:
    """Run Palabra AI benchmark with progress tracking"""

    def __init__(self, audio_file: str, source_lang: str | None = None, target_lang: str | None = None,
                 silent: bool = True, mode: str | None = None, chunk_duration_ms: int | None = None,
                 base_config: Config | None = None, palabra_client: PalabraAI | None = None,
                 save_audio: bool = False, output_dir: Path | None = None):
        from datetime import datetime

        self.palabra_client = palabra_client or PalabraAI()
        self.audio_file = Path(audio_file)
        self.base_config = base_config

        # Generate single timestamp for entire benchmark run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # If config provided, extract values from it
        if self.base_config:
            # Use languages from config
            if self.base_config.source:
                self.source_lang = self.base_config.source.lang
            else:
                raise ValueError("Config must have a source language defined")

            if self.base_config.targets and len(self.base_config.targets) > 0:
                self.target_lang = self.base_config.targets[0].lang
            else:
                raise ValueError("Config must have at least one target language defined")

            # Use mode from config if not overridden
            if mode is None:
                # Mode is already in config
                self.mode = None  # Will use config's mode
            else:
                self.mode = mode

            # Use chunk_duration_ms from config if not overridden
            if chunk_duration_ms is None:
                self.chunk_duration_ms = None  # Will use config's chunk_duration_ms
            else:
                self.chunk_duration_ms = chunk_duration_ms
        else:
            # No config, require all parameters
            if not source_lang or not target_lang:
                raise ValueError("source_lang and target_lang are required when not using config")

            # Get language objects using existing functionality
            self.source_lang = Language.get_or_create(source_lang)
            self.target_lang = Language.get_or_create(target_lang)

            # Validate languages
            if not is_valid_source_language(self.source_lang):
                raise ValueError(f"Language '{source_lang}' is not a valid source language for Palabra API")
            if not is_valid_target_language(self.target_lang):
                raise ValueError(f"Language '{target_lang}' is not a valid target language for Palabra API")

            self.mode = mode or "ws"
            self.chunk_duration_ms = chunk_duration_ms or 100

        self.silent = silent
        self.progress_bar = None
        self.audio_duration = None
        self.last_timestamp = 0.0
        self.save_audio = save_audio
        self.output_dir = output_dir or Path.cwd()

        # Validate audio file
        if not self.audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Get audio duration for progress tracking
        try:
            audio_data, sr = librosa.load(str(self.audio_file), sr=None)
            self.audio_duration = len(audio_data) / sr
        except Exception as e:
            warning(f"Could not determine audio duration: {e}")
            self.audio_duration = None

    def _on_transcription(self, msg):
        """Callback for transcription messages to track progress"""
        if self.progress_bar and self.audio_duration:
            # Extract timestamp from transcription
            if hasattr(msg, 'segments') and msg.segments:
                # Get the end timestamp of the last segment
                end_timestamp = msg.segments[-1].end
                if end_timestamp > self.last_timestamp:
                    self.last_timestamp = end_timestamp
                    # Update progress bar
                    progress_pct = min(100, (end_timestamp / self.audio_duration) * 100)
                    self.progress_bar.update(progress_pct - self.progress_bar.n)

    def run(self, show_progress: bool = True) -> dict[str, Any]:
        """
        Run the benchmark and return the result

        Args:
            show_progress: Whether to show progress bar

        Returns:
            Dictionary containing the benchmark result with log_data
        """
        # Create progress bar
        if show_progress and self.audio_duration:
            self.progress_bar = tqdm(
                total=100,
                desc="Processing audio",
                unit="%",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total:.0f} [{elapsed}<{remaining}]"
            )

        try:
            # Create reader and writer
            reader = FileReader(str(self.audio_file))

            # Create writer based on save_audio flag
            if self.save_audio:
                audio_output = self.output_dir / f"{self.audio_file.stem}_{self.timestamp}.wav"

                writer = FileWriter(path=audio_output, delete_on_error=False)
            else:
                writer = DummyWriter()

            # Configure with benchmark mode
            if self.base_config:
                # Use base config as template but create new source/target with readers/writers
                config = self.base_config

                # Create new source with reader and callback, preserving transcription settings
                source_transcription = config.source.transcription if config.source else None
                config.source = SourceLang(
                    self.source_lang,
                    reader,
                    on_transcription=self._on_transcription,
                    transcription=source_transcription
                )

                # Create new target with writer and callback, preserving translation settings
                if config.targets and len(config.targets) > 0:
                    target_translation = config.targets[0].translation
                    config.targets = [TargetLang(
                        self.target_lang,
                        writer,
                        on_transcription=self._on_transcription,
                        translation=target_translation
                    )]
                else:
                    config.targets = [TargetLang(self.target_lang, writer, on_transcription=self._on_transcription)]

                # Only override mode if explicitly provided via CLI
                if self.mode is not None:
                    from palabra_ai.config import IoMode
                    config.mode = IoMode.from_string(
                        self.mode,
                        chunk_duration_ms=self.chunk_duration_ms or (10 if self.mode == "webrtc" else 100)
                    )
                elif self.chunk_duration_ms is not None:
                    # If only chunk_duration_ms is overridden, update existing mode
                    from palabra_ai.config import IoMode
                    config.mode = IoMode.from_string(
                        config.mode.mode_type,
                        chunk_duration_ms=self.chunk_duration_ms
                    )
                else:
                    # For benchmark, ensure WsMode has 100ms chunks (not the default 320ms)
                    if config.mode.mode_type == "ws":
                        from palabra_ai.config import IoMode
                        config.mode = IoMode.from_string("ws", chunk_duration_ms=100)
                # Set benchmark-specific settings
                config.silent = self.silent
                config.benchmark = True
                config.estimated_duration = self.audio_duration
            else:
                # Create appropriate IoMode based on mode parameter
                from palabra_ai.config import IoMode
                io_mode = IoMode.from_string(
                    self.mode or "ws",  # default to ws
                    chunk_duration_ms=self.chunk_duration_ms
)

                # Create config from scratch
                config = Config(
                    SourceLang(self.source_lang, reader, on_transcription=self._on_transcription),
                    [TargetLang(self.target_lang, writer, on_transcription=self._on_transcription)],
                    silent=self.silent,
                    benchmark=True,
                    mode=io_mode,
                    estimated_duration=self.audio_duration,
                )

            # Run the processing
            # Note: When running from subprocess, we're in main thread so signal handlers work
            # When running from threads, use without_signal_handlers=True
            import threading
            if threading.current_thread() == threading.main_thread():
                result = self.palabra_client.run(config, no_raise=True)
            else:
                result = self.palabra_client.run(config, without_signal_handlers=True, no_raise=True)

            # Detailed diagnostics
            debug(f"Result type: {type(result)}")
            debug(f"Result is None: {result is None}")

            if result:
                debug(f"Result.ok: {result.ok}")
                debug(f"Result.exc: {result.exc}")
                debug(f"Result.log_data: {result.log_data}")
                debug(f"Has log_data: {result.log_data is not None}")

                if result.log_data:
                    debug(f"Messages count: {len(result.log_data.messages)}")

                if result.exc:
                    debug(f"Exception type: {type(result.exc)}")
                    if isinstance(result.exc, asyncio.CancelledError):
                        debug("Task was cancelled but we might have log_data")
                    import traceback
                    debug("Exception traceback:")
                    traceback.print_exception(type(result.exc), result.exc, result.exc.__traceback__)
            else:
                error("palabra.run() returned None!")
                # Create empty result
                from palabra_ai.model import RunResult
                result = RunResult(ok=False, exc=Exception("No result from palabra.run()"))

            # Close progress bar in any case
            if self.progress_bar:
                self.progress_bar.update(100 - self.progress_bar.n)  # Complete to 100%
                self.progress_bar.close()

            # Log saved audio file
            if self.save_audio and audio_output and audio_output.exists():
                info(f"✅ Audio saved to: {audio_output}")

                # Create stereo debug mix with original + TTS
                debug_output = audio_output.with_name(audio_output.stem + '_debug_in_out.wav')
                create_stereo_debug_mix(self.audio_file, audio_output, debug_output)
                if debug_output.exists():
                    info(f"✅ Stereo debug mix saved to: {debug_output}")

            return result

        except Exception as e:
            if self.progress_bar:
                self.progress_bar.close()
            raise e


class BenchmarkAnalyzer:
    """Analyze and generate reports from benchmark results"""

    def __init__(self, result: dict[str, Any], timestamp: str = None):
        """
        Initialize analyzer with benchmark result

        Args:
            result: Result from BenchmarkRunner.run()
            timestamp: Timestamp for consistent file naming
        """
        from datetime import datetime

        self.result = result
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Debug output
        debug(f"Result type: {type(result)}")

        # Handle different result scenarios
        if result is None:
            error("Result is None!")
            self.messages = []
        elif hasattr(result, 'exc') and result.exc:
            error(f"Benchmark failed with exception: {result.exc}")
            import traceback
            traceback.print_exception(type(result.exc), result.exc, result.exc.__traceback__)
            # Try to extract log_data even with exception
            self.messages = result.log_data.messages if result.log_data else []
            debug(f"Extracted {len(self.messages)} messages despite exception")
        elif hasattr(result, 'log_data'):
            debug("Has log_data: True")
            debug(f"log_data is None: {result.log_data is None}")
            if result.log_data:
                debug(f"Messages count: {len(result.log_data.messages)}")
                self.messages = result.log_data.messages
            else:
                warning("log_data is None!")
                self.messages = []
        else:
            warning("Result has no log_data attribute!")
            self.messages = []

        debug(f"Final extracted messages count: {len(self.messages)}")

        self.analysis = None

    def analyze(self) -> dict[str, Any]:
        """
        Perform latency analysis on the messages

        Returns:
            Analysis results dictionary
        """
        if not self.messages:
            raise ValueError("No messages to analyze")

        self.analysis = analyze_latency(self.messages)
        return self.analysis

    def get_text_report(self, max_chunks: int = -1, show_empty: bool = False) -> str:
        """
        Get text report for console output

        Args:
            max_chunks: Maximum number of chunks to display in detail (-1 for all)
            show_empty: Whether to include empty chunks in the detailed view

        Returns:
            Formatted text report
        """
        if not self.analysis:
            self.analyze()

        return generate_text_report(self.analysis, max_chunks, show_empty)

    def get_result(self) -> dict[str, Any]:
        return from_json(to_json(self.result))

    def get_html_report(self) -> str:
        """
        Get HTML report

        Returns:
            HTML report content
        """
        if not self.analysis:
            self.analyze()

        return generate_html_report(self.analysis)

    def get_json_report(self, raw_result: bool = False) -> bytes:
        """
        Get JSON report

        Args:
            raw_result: Whether to include full raw result data

        Returns:
            JSON report content
        """
        if not self.analysis:
            self.analyze()

        return generate_json_report(self.analysis, raw_result, self.get_result() if raw_result else None)

    def save_reports(self, output_dir: Path | None = None,
                 html: bool = False, json: bool = False, raw_result: bool = False) -> dict[str, Path]:
        """
        Save reports to files

        Args:
            output_dir: Directory to save reports (default: current directory)
            html: Whether to save HTML report
            json: Whether to save JSON report
            raw_result: Whether to include full raw result data in JSON

        Returns:
            Dictionary with paths to saved files
        """
        if not self.analysis:
            self.analyze()

        output_dir = output_dir or Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        if html:
            html_file = output_dir / f"benchmark_report_{self.timestamp}.html"
            html_file.write_text(self.get_html_report())
            saved_files['html'] = html_file

        if json:
            json_file = output_dir / f"benchmark_analysis_{self.timestamp}.json"
            json_file.write_bytes(self.get_json_report(raw_result))
            saved_files['json'] = json_file

            result_file = output_dir / f"benchmark_result_{self.timestamp}.json"
            result_file.write_bytes(to_json(self.result))
            saved_files['result'] = result_file

        return saved_files


def run_benchmark(audio_file: str, source_lang: str | None = None, target_lang: str | None = None,
                 silent: bool = True, show_progress: bool = True,
                 mode: str | None = None, chunk_duration_ms: int | None = None,
                 base_config: Config | None = None,
                 palabra_client: PalabraAI | None = None,
                 save_audio: bool = False, output_dir: Path | None = None) -> BenchmarkAnalyzer:
    """
    Convenience function to run benchmark and return analyzer

    Args:
        audio_file: Path to audio file
        source_lang: Source language code (ignored if base_config provided)
        target_lang: Target language code (ignored if base_config provided)
        silent: Whether to run Palabra in silent mode
        show_progress: Whether to show progress bar
        mode: Connection mode - "ws" or "webrtc" (ignored if base_config provided)
        chunk_duration_ms: Audio chunk duration in milliseconds (ignored if base_config provided)
        base_config: Optional base Config to preload settings from

    Returns:
        BenchmarkAnalyzer instance with results
    """
    runner = BenchmarkRunner(audio_file, source_lang, target_lang, silent, mode, chunk_duration_ms, base_config, palabra_client, save_audio, output_dir)
    result = runner.run(show_progress)
    return BenchmarkAnalyzer(result, runner.timestamp)
