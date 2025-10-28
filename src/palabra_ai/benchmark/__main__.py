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
from palabra_ai.benchmark.report import format_report
from palabra_ai.benchmark.report import INPUT_CHUNK_DURATION_S
from palabra_ai.benchmark.report import Report
from palabra_ai.benchmark.report import save_benchmark_files
from palabra_ai.config import WsMode
from palabra_ai.lang import Language
from palabra_ai.task.adapter.dummy import DummyWriter
from palabra_ai.task.adapter.file import FileReader
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

        # Setup output directory and timestamp if --out is specified
        if args.out:
            output_dir = args.out
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save sysinfo immediately at startup
            sysinfo = get_system_info()
            sysinfo["command"] = " ".join(sys.argv)
            sysinfo["argv"] = sys.argv
            sysinfo["cwd"] = str(Path.cwd())
            sysinfo_path = output_dir / f"{timestamp}_bench_sysinfo.json"
            sysinfo_path.write_bytes(to_json(sysinfo, True))

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
                allowed_message_types=BENCHMARK_ALLOWED_MESSAGE_TYPES,
            )

        # Enable debug mode and logging when --out is specified
        if output_dir and timestamp:
            config.debug = True
            config.log_file = str(output_dir / f"{timestamp}_bench.log")

            # Save exact config that goes to set_task (SetTaskMessage.from_config uses to_dict)
            config_dict = config.to_dict()
            config_path = output_dir / f"{timestamp}_bench_config.json"
            config_path.write_bytes(to_json(config_dict, True))

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

        # Save RunResult in debug mode when --out is specified
        if output_dir and timestamp and result is not None:
            try:
                result_debug_path = output_dir / f"{timestamp}_bench_runresult_debug.json"
                result_debug_path.write_bytes(to_json(result.model_dump(), True))
            except Exception as e:
                # If serialization fails, save error info
                error_path = output_dir / f"{timestamp}_bench_runresult_error.txt"
                error_path.write_text(
                    f"Failed to serialize RunResult: {e}\n\n"
                    f"RunResult repr:\n{repr(result)}\n\n"
                    f"Exception: {result.exc if result else 'N/A'}"
                )

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

                    # For CancelledError, show ALL logs to understand what happened
                    if result.log_data and result.log_data.logs:
                        print(f"Full logs (all {len(result.log_data.logs)} entries):")
                        for log_line in result.log_data.logs:
                            print(log_line, end='')
                        print()
                else:
                    print(f"\n{'='*80}")
                    print(f"BENCHMARK FAILED: {exc_type}: {exc_msg}")
                    print(f"{'='*80}\n")

                    # For other errors, show last 100
                    if result.log_data and result.log_data.logs:
                        print("Last 100 log entries:")
                        for log_line in result.log_data.logs[-100:]:
                            print(log_line, end='')
                        print()

                # Print traceback from exception if available
                if hasattr(result.exc, '__traceback__') and result.exc.__traceback__:
                    print("\nOriginal exception traceback:")
                    traceback.print_exception(type(result.exc), result.exc, result.exc.__traceback__)
                    print()

                raise RuntimeError(f"Benchmark failed: {exc_type}: {exc_msg}") from result.exc
            raise RuntimeError("Benchmark failed: no io_data")

        # Parse report
        report, in_audio_canvas, out_audio_canvas = Report.parse(result.io_data)

        # Create file paths (used in report and optionally saved with --out)
        if not timestamp:
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
            # Use the shared save function
            if not output_dir:
                output_dir = args.out
            save_benchmark_files(
                output_dir=output_dir,
                timestamp=timestamp,
                report=report,
                io_data=result.io_data,
                config=config,
                result=result,
                in_audio_canvas=in_audio_canvas,
                out_audio_canvas=out_audio_canvas,
                source_lang=source_lang,
                target_lang=target_lang,
                report_text=report_text,
                input_file_path=str(audio_path),
                file_prefix="bench"
            )

        # Always print report to console
        print("\n" + report_text)

    except Exception as e:
        # Capture traceback IMMEDIATELY - must be done in except block!
        tb_string = traceback.format_exc()

        # Print full traceback to console
        print(f"\n{'='*80}")
        print("BENCHMARK CRASHED - FULL TRACEBACK:")
        print(f"{'='*80}\n")
        print(tb_string)

        # Save error to file if output directory exists
        if output_dir and timestamp:
            try:
                error_file = output_dir / f"{timestamp}_bench_error.txt"
                error_file.write_text(f"Benchmark Error:\n\n{tb_string}")
                print(f"\nError details saved to: {error_file}")
            except Exception as save_error:
                print(f"Failed to save error file: {save_error}")

        # Try to save partial report/audio even on error (for debugging)
        if output_dir and timestamp and result and result.io_data:
            try:
                print("\nAttempting to save partial results for debugging...")

                # Try to parse report
                report, in_audio, out_audio = Report.parse(result.io_data)

                # Save report files
                report_path = output_dir / f"{timestamp}_bench_report_partial.json"
                report_path.write_bytes(to_json(report, True))
                print(f"✓ Partial report saved to: {report_path}")

                # Save audio (always when --out is specified)
                in_wav = output_dir / f"{timestamp}_bench_in_partial.wav"
                out_wav = output_dir / f"{timestamp}_bench_out_partial.wav"
                save_wav(in_audio, in_wav, result.io_data.in_sr, result.io_data.channels)
                save_wav(out_audio, out_wav, result.io_data.out_sr, result.io_data.channels)
                print(f"✓ Partial audio saved: {in_wav.name}, {out_wav.name}")

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
