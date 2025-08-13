"""
CLI entry point for Palabra AI Benchmark
Usage: python -m palabra_ai.benchmark <audio> <source_lang> <target_lang> [options]
"""

import argparse
import sys
from pathlib import Path

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
  python -m palabra_ai.benchmark audio.mp3 es en --no-progress
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
    
    args = parser.parse_args()
    
    # Validate audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
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
            show_progress=not args.no_progress
        )
        
        # Analyze results
        print("\nAnalyzing results...")
        analyzer.analyze()
        
        # Print text report to console (always)
        print("\n" + analyzer.get_text_report())
        
        # Save additional reports if requested
        if args.html or args.json:
            saved_files = analyzer.save_reports(
                output_dir=args.output_dir,
                html=args.html,
                json=args.json
            )
            
            print("\nReports saved:")
            for report_type, path in saved_files.items():
                print(f"  {report_type.upper()}: {path}")
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError during benchmark: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()