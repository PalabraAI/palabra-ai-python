"""
Main runner for Palabra AI benchmark
Handles audio processing with DummyWriter and progress tracking
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import librosa
from tqdm import tqdm

from palabra_ai import PalabraAI, Config, SourceLang, TargetLang
from palabra_ai.lang import Language, is_valid_source_language, is_valid_target_language
from palabra_ai.task.adapter.file import FileReader
from palabra_ai.task.adapter.dummy import DummyWriter
from palabra_ai.util.orjson import to_json, from_json
from palabra_ai.util.logger import debug, error, warning, info

from .analyzer import analyze_latency
from .reporter import generate_text_report, generate_html_report, generate_json_report



class BenchmarkRunner:
    """Run Palabra AI benchmark with progress tracking"""
    
    def __init__(self, audio_file: str, source_lang: str, target_lang: str, silent: bool = True):
        self.audio_file = Path(audio_file)
        
        # Get language objects using existing functionality
        self.source_lang = Language.get_or_create(source_lang)
        self.target_lang = Language.get_or_create(target_lang)
        
        # Validate languages
        if not is_valid_source_language(self.source_lang):
            raise ValueError(f"Language '{source_lang}' is not a valid source language for Palabra API")
        if not is_valid_target_language(self.target_lang):
            raise ValueError(f"Language '{target_lang}' is not a valid target language for Palabra API")
        
        self.silent = silent
        self.progress_bar = None
        self.audio_duration = None
        self.last_timestamp = 0.0
        
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
    
    def run(self, show_progress: bool = True) -> Dict[str, Any]:
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
            # Initialize Palabra AI
            palabra = PalabraAI()
            
            # Create reader and writer
            reader = FileReader(str(self.audio_file))
            writer = DummyWriter()
            
            # Configure with benchmark mode
            config = Config(
                SourceLang(self.source_lang, reader, on_transcription=self._on_transcription),
                [TargetLang(self.target_lang, writer, on_transcription=self._on_transcription)],
                silent=self.silent,
                benchmark=True,
            )
            
            # Run the processing
            # Note: When running from subprocess, we're in main thread so signal handlers work
            # When running from threads, use without_signal_handlers=True
            import threading
            if threading.current_thread() == threading.main_thread():
                result = palabra.run(config, no_raise=True)
            else:
                result = palabra.run(config, without_signal_handlers=True, no_raise=True)
            
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
                    debug(f"Exception traceback:")
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
            
            return result
            
        except Exception as e:
            if self.progress_bar:
                self.progress_bar.close()
            raise e


class BenchmarkAnalyzer:
    """Analyze and generate reports from benchmark results"""
    
    def __init__(self, result: Dict[str, Any]):
        """
        Initialize analyzer with benchmark result
        
        Args:
            result: Result from BenchmarkRunner.run()
        """
        self.result = result
        # Debug output
        debug(f"Result type: {type(result)}")
        
        # Handle different result scenarios
        if result is None:
            error(f"Result is None!")
            self.messages = []
        elif hasattr(result, 'exc') and result.exc:
            error(f"Benchmark failed with exception: {result.exc}")
            import traceback
            traceback.print_exception(type(result.exc), result.exc, result.exc.__traceback__)
            # Try to extract log_data even with exception
            self.messages = result.log_data.messages if result.log_data else []
            debug(f"Extracted {len(self.messages)} messages despite exception")
        elif hasattr(result, 'log_data'):
            debug(f"Has log_data: True")
            debug(f"log_data is None: {result.log_data is None}")
            if result.log_data:
                debug(f"Messages count: {len(result.log_data.messages)}")
                self.messages = result.log_data.messages
            else:
                warning(f"log_data is None!")
                self.messages = []
        else:
            warning(f"Result has no log_data attribute!")
            self.messages = []
        
        debug(f"Final extracted messages count: {len(self.messages)}")
        
        self.analysis = None
    
    def analyze(self) -> Dict[str, Any]:
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

    def get_result(self) -> Dict[str, Any]:
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

    def get_json_report(self, include_raw_data: bool = False) -> str:
        """
        Get JSON report
        
        Args:
            include_raw_data: Whether to include full raw result data
        
        Returns:
            JSON report content
        """
        if not self.analysis:
            self.analyze()
        
        return generate_json_report(self.analysis, include_raw_data, self.get_result() if include_raw_data else None)
    
    def save_reports(self, output_dir: Optional[Path] = None, 
                     html: bool = False, json: bool = False, include_raw_data: bool = False) -> Dict[str, Path]:
        """
        Save reports to files
        
        Args:
            output_dir: Directory to save reports (default: current directory)
            html: Whether to save HTML report
            json: Whether to save JSON report
            include_raw_data: Whether to include full raw result data in JSON
            
        Returns:
            Dictionary with paths to saved files
        """
        if not self.analysis:
            self.analyze()
        
        output_dir = output_dir or Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        if html:
            html_file = output_dir / "benchmark_report.html"
            html_file.write_text(self.get_html_report())
            saved_files['html'] = html_file
        
        if json:
            json_file = output_dir / "benchmark_analysis.json"
            json_file.write_text(self.get_json_report(include_raw_data))
            saved_files['json'] = json_file
        
        return saved_files


def run_benchmark(audio_file: str, source_lang: str, target_lang: str,
                 silent: bool = True, show_progress: bool = True) -> BenchmarkAnalyzer:
    """
    Convenience function to run benchmark and return analyzer
    
    Args:
        audio_file: Path to audio file
        source_lang: Source language code
        target_lang: Target language code
        silent: Whether to run Palabra in silent mode
        show_progress: Whether to show progress bar
        
    Returns:
        BenchmarkAnalyzer instance with results
    """
    runner = BenchmarkRunner(audio_file, source_lang, target_lang, silent)
    result = runner.run(show_progress)
    return BenchmarkAnalyzer(result)