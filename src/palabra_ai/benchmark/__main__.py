"""Palabra AI Benchmark - Data Collection Only"""

import argparse
import asyncio
import sys
import traceback
from datetime import datetime
from pathlib import Path

import av
from tqdm import tqdm

from palabra_ai import Config, PalabraAI, SourceLang, TargetLang
from palabra_ai.audio import save_wav
from palabra_ai.benchmark.report import BENCHMARK_ALLOWED_MESSAGE_TYPES
from palabra_ai.benchmark.report import INPUT_CHUNK_DURATION_S
from palabra_ai.benchmark.report import Report
from palabra_ai.config import WsMode
from palabra_ai.lang import Language
from palabra_ai.task.adapter.dummy import DummyWriter
from palabra_ai.task.adapter.file import FileReader
from palabra_ai.util.fileio import save_text
from palabra_ai.util.orjson import to_json
from palabra_ai.util.sysinfo import get_system_info


# Benchmark always uses all message types for complete data collection


def main():
    parser = argparse.ArgumentParser(description="Palabra AI Benchmark - Data Collection")
    parser.add_argument("audio", help="Audio file path")
    parser.add_argument("source_lang", nargs="?", help="Source language")
    parser.add_argument("target_lang", nargs="?", help="Target language")
    parser.add_argument("--config", type=Path, help="JSON config file")
    parser.add_argument("--out", type=Path, help="Output directory for files (if not specified, only prints to console)")

    args = parser.parse_args()

    # Initialize variables for error handling
    output_dir = None
    timestamp = None
    result = None
    config = None
    progress_bar = [None]

    try:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {args.audio}")
        mode = WsMode(input_chunk_duration_ms=INPUT_CHUNK_DURATION_S * 1000)

        # Get audio duration for progress tracking
        with av.open(str(audio_path)) as container:
            audio_duration = container.duration / 1000000  # convert microseconds to seconds

        # Create reader
        reader = FileReader(str(audio_path))

        # Create progress bar placeholder (will update desc after config loaded)
        last_timestamp = [0.0]  # mutable to allow updates in nested function

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
            config.allowed_message_types = BENCHMARK_ALLOWED_MESSAGE_TYPES

            # Force benchmark mode with 100ms buffer regardless of config
            # Config loaded from JSON defaults to 320ms chunks, but benchmark needs 100ms for optimal performance
            config.mode = WsMode(input_chunk_duration_ms=INPUT_CHUNK_DURATION_S * 1000)
        else:
            if not args.source_lang or not args.target_lang:
                parser.error("source_lang and target_lang required without --config")

            config = Config(
                source=SourceLang(Language.get_or_create(args.source_lang), reader, on_transcription=on_transcription),
                targets=[TargetLang(Language.get_or_create(args.target_lang), DummyWriter())],
                benchmark=True,
                mode=mode,
                allowed_message_types=BENCHMARK_ALLOWED_MESSAGE_TYPES,
            )

        # Enable debug mode and output directory when --out is specified
        # Core will auto-save log, trace, result.json, and audio files
        if args.out:
            # config.debug = True
            config.output_dir = Path(args.out)
            print(f"Files will be saved to {args.out}")

        # Create progress bar with language info
        progress_bar[0] = tqdm(
            total=100,
            desc=f"Processing {config.source_lang}→{config.target_lang}",
            unit="%",
            mininterval=7.0,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]"
        )

        print(f"Running benchmark: {config.source_lang} → {config.target_lang}")
        print("-" * 60)

        palabra = PalabraAI()
        result = palabra.run(config, no_raise=True)

        # Complete and close progress bar
        if progress_bar[0]:
            progress_bar[0].update(100 - progress_bar[0].n)
            progress_bar[0].close()

        if result is None or not result.ok or not result.io_data:
            if result is None:
                print(f"\n{'='*80}")
                print("BENCHMARK INTERRUPTED BY USER (Ctrl+C)")
                print(f"{'='*80}\n")
                print("The benchmark was interrupted before completion.")
                print("No results were generated.")
                return
            if result.exc:
                exc_type = type(result.exc).__name__
                exc_msg = str(result.exc) or "(no message)"

                # Special handling for CancelledError
                if isinstance(result.exc, asyncio.CancelledError):
                    print(f"\n{'='*80}")
                    print(f"BENCHMARK WAS CANCELLED")
                    print(f"{'='*80}\n")
                    print("This usually means:")
                    print("  - User interrupted with Ctrl+C")
                    print("  - Task was cancelled by timeout")
                    print("  - Internal cancellation due to error")
                    print("  - One of the subtasks failed and caused cascade cancellation\n")
                else:
                    print(f"\n{'='*80}")
                    print(f"BENCHMARK FAILED: {exc_type}: {exc_msg}")
                    print(f"{'='*80}\n")

                # Print traceback from exception if available
                if hasattr(result.exc, '__traceback__') and result.exc.__traceback__:
                    print("\nOriginal exception traceback:")
                    traceback.print_exception(type(result.exc), result.exc, result.exc.__traceback__)
                    print()

                raise RuntimeError(f"Benchmark failed: {exc_type}: {exc_msg}") from result.exc
            raise RuntimeError("Benchmark failed: no io_data")

        # Parse report
        if args.out:
            report = Report.parse(result.io_data, Path(args.out))
            report.save_all()
        else:
            report = Report.parse(result.io_data)
        print("\n" + report.report_txt)

    except Exception as e:
        # Capture traceback IMMEDIATELY - must be done in except block!
        tb_string = traceback.format_exc()

        # Print full traceback to console
        print(f"\n{'='*80}")
        print("BENCHMARK CRASHED - FULL TRACEBACK:")
        print(f"{'='*80}\n")
        print(tb_string)

        if config and args.out:
            save_text(config.get_out_path(".error.txt"), f"Benchmark Error:\n\n{tb_string}")

        # Try to save partial report/audio even on error (for debugging)
        if result and result.io_data:
            try:
                print("\nAttempting to save partial results for debugging...")

                # Try to parse report
                if args.out:
                    output_dir = Path(args.out)
                    report = Report.parse(result.io_data, output_dir)
                    report.save_all()
                print(f"✓ Something saved to: {args.out}")

            except Exception as save_err:
                print(f"Could not save partial results: {save_err}")

        # Re-raise the exception
        raise

    finally:
        # Always try to close progress bar
        if progress_bar[0]:
            try:
                progress_bar[0].close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
