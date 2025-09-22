"""
CLI entry point for Palabra AI Benchmark
Usage: python -m palabra_ai.benchmark <audio> <source_lang> <target_lang> [options]
"""

import argparse
import sys
from pathlib import Path

from palabra_ai.util.logger import error
from palabra_ai.config import Config
from palabra_ai.util.orjson import from_json
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
  python -m palabra_ai.benchmark audio.mp3 --config config.json  # Use JSON config (languages from config)
        """
    )

    # Required arguments
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("source_lang", nargs='?', default=None,
                       help="Source language code (e.g., es, en, fr) - ignored if --config is provided")
    parser.add_argument("target_lang", nargs='?', default=None,
                       help="Target language code (e.g., en, es, fr) - ignored if --config is provided")

    # Optional arguments
    parser.add_argument("--config", type=Path, default=None,
                       help="Path to JSON config file to preload settings from")
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
    parser.add_argument("--save-audio", action="store_true",
                       help="Save translated audio output to WAV file")

    args = parser.parse_args()

    # Load base config from JSON if provided
    base_config = None
    source_lang = None
    target_lang = None
    mode = None
    chunk_duration_ms = None

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            error(f"Config file not found: {args.config}")
            sys.exit(1)
        try:
            config_data = config_path.read_text()
            base_config = Config.from_json(config_data)
            # Extract languages from config for display
            if base_config.source:
                source_lang = base_config.source.lang.code
            if base_config.targets and len(base_config.targets) > 0:
                target_lang = base_config.targets[0].lang.code
            # Don't extract mode - let runner use config's mode
            # Only override if CLI args are provided
            if args.mode != "ws":  # "ws" is the default, so only override if explicitly changed
                mode = args.mode
            if args.chunk_duration_ms != 100:  # 100 is the default
                chunk_duration_ms = args.chunk_duration_ms
        except Exception as e:
            error(f"Failed to load config from {args.config}: {e}")
            sys.exit(1)
    else:
        # No config, use CLI arguments
        if not args.source_lang or not args.target_lang:
            error("source_lang and target_lang are required when not using --config")
            parser.print_help()
            sys.exit(1)
        source_lang = args.source_lang
        target_lang = args.target_lang
        mode = args.mode
        chunk_duration_ms = args.chunk_duration_ms

    # Validate audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        error(f"Audio file not found: {args.audio}")
        sys.exit(1)

    try:
        # Run benchmark
        print(f"Running benchmark on: {args.audio}")
        if source_lang and target_lang:
            print(f"Languages: {source_lang} → {target_lang}")
        if base_config:
            print(f"Using config: {args.config}")
        print("-" * 60)

        analyzer = run_benchmark(
            str(audio_path),
            source_lang=source_lang,
            target_lang=target_lang,
            silent=not args.verbose,
            show_progress=not args.no_progress,
            mode=mode,
            chunk_duration_ms=chunk_duration_ms,
            base_config=base_config,
            save_audio=args.save_audio,
            output_dir=args.output_dir
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
