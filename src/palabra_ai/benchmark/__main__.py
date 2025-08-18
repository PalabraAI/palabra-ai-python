"""
CLI entry point for Palabra AI Benchmark
Usage: python -m palabra_ai.benchmark <audio> <source_lang> <target_lang> [options]
"""

import argparse
import sys
from pathlib import Path

from palabra_ai.util.logger import error
from .runner import run_benchmark


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Palabra AI Benchmark - Analyze latency and performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m palabra_ai.benchmark audio.mp3 es en
  python -m palabra_ai.benchmark audio.mp3 es en --html --json
  python -m palabra_ai.benchmark audio.mp3 es en --output-dir results/
  python -m palabra_ai.benchmark audio.mp3 es en --chunks 5  # Show only 5 chunks
  python -m palabra_ai.benchmark audio.mp3 es en --chunks 10 --show-empty
  python -m palabra_ai.benchmark audio.mp3 es en --json --raw-result  # Include raw data
  python -m palabra_ai.benchmark audio.mp3 es en --no-progress
  python -m palabra_ai.benchmark audio.mp3 es en --mode webrtc  # Use WebRTC mode
  python -m palabra_ai.benchmark audio.mp3 es en --chunk-duration-ms 50  # 50ms chunks
  python -m palabra_ai.benchmark audio.mp3 es en --mode webrtc --chunk-duration-ms 20
        """
    )
    
    # Required arguments
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("source_lang", help="Source language code (e.g., es, en, fr)")
    parser.add_argument("target_lang", help="Target language code (e.g., en, es, fr)")
    
    # Optional arguments
    parser.add_argument("--html", action="store_true",
                       help="Save HTML report to file")
    parser.add_argument("--json", action="store_true",
                       help="Save JSON report to file")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Directory to save reports (default: current directory)")
    parser.add_argument("--no-progress", action="store_true",
                       help="Disable progress bar")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output (disable silent mode)")
    parser.add_argument("--chunks", type=int, default=-1,
                       help="Number of chunks to show in detail (default: all, use positive number to limit)")
    parser.add_argument("--show-empty", action="store_true",
                       help="Include empty chunks in detailed view")
    parser.add_argument("--raw-result", action="store_true",
                       help="Include full raw result data in JSON report")
    parser.add_argument("--mode", choices=["ws", "webrtc"], default="ws",
                       help="Connection mode: ws (WebSocket) or webrtc (default: ws)")
    parser.add_argument("--chunk-duration-ms", type=int, default=100,
                       help="Audio chunk duration in milliseconds (default: 100)")
    
    args = parser.parse_args()
    
    # Validate audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        error(f"Audio file not found: {args.audio}")
        sys.exit(1)
    
    try:
        # Run benchmark
        print(f"Running benchmark on: {args.audio}")
        print(f"Languages: {args.source_lang} → {args.target_lang}")
        print("-" * 60)
        
        analyzer = run_benchmark(
            str(audio_path),
            args.source_lang,
            args.target_lang,
            silent=not args.verbose,
            show_progress=not args.no_progress,
            mode=args.mode,
            chunk_duration_ms=args.chunk_duration_ms
        )
        
        # Analyze results
        print("\nAnalyzing results...")
        analyzer.analyze()
        
        # Print text report to console (always)
        print("\n" + analyzer.get_text_report(args.chunks, args.show_empty))
        
        # Save additional reports if requested
        if args.html or args.json:
            saved_files = analyzer.save_reports(
                output_dir=args.output_dir,
                html=args.html,
                json=args.json,
                raw_result=args.raw_result
            )
            
            print("\nReports saved:")
            for report_type, path in saved_files.items():
                print(f"  {report_type.upper()}: {path}")
        
    except KeyboardInterrupt:
        error("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        error(f"Error during benchmark: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()