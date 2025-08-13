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
from palabra_ai.util.orjson import to_json

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
            print(f"Warning: Could not determine audio duration: {e}")
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
            result = palabra.run(config)
            
            # Close progress bar
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
        self.messages = result.log_data.messages if hasattr(result, 'log_data') else []
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
    
    def get_text_report(self) -> str:
        """
        Get text report for console output
        
        Returns:
            Formatted text report
        """
        if not self.analysis:
            self.analyze()
        
        return generate_text_report(self.analysis)
    
    def get_html_report(self) -> str:
        """
        Get HTML report
        
        Returns:
            HTML report content
        """
        if not self.analysis:
            self.analyze()
        
        return generate_html_report(self.analysis)
    
    def get_json_report(self) -> str:
        """
        Get JSON report
        
        Returns:
            JSON report content
        """
        if not self.analysis:
            self.analyze()
        
        return generate_json_report(self.analysis)
    
    def save_reports(self, output_dir: Optional[Path] = None, 
                     html: bool = False, json: bool = False) -> Dict[str, Path]:
        """
        Save reports to files
        
        Args:
            output_dir: Directory to save reports (default: current directory)
            html: Whether to save HTML report
            json: Whether to save JSON report
            
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
            json_file.write_text(self.get_json_report())
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