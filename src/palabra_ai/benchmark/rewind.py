#!/usr/bin/env python3
"""
Palabra AI Benchmark Rewind - Analyze existing run_result.json files
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from palabra_ai.util.orjson import from_json, to_json
from palabra_ai.model import IoData
from palabra_ai.message import IoEvent, Dbg
from palabra_ai.benchmark.__main__ import Report, format_report, save_benchmark_files
from palabra_ai import Config

def extract_config_from_result(file_path: Path) -> tuple[dict, str, str]:
    """Extract config data and languages from run_result.json using jq"""
    print(f"Extracting config from {file_path}...")

    # Extract config data from set_task event using jq
    cmd = ['jq', '.io_data.events[] | select(.body.message_type == "set_task") | .body.data', str(file_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    config_data = from_json(result.stdout)

    # Extract source language using jq
    cmd = ['jq', '-r', '.io_data.events[] | select(.body.message_type == "set_task") | .body.data.pipeline.transcription.source_language', str(file_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    source_lang = result.stdout.strip()

    # Extract target language using jq
    cmd = ['jq', '-r', '.io_data.events[] | select(.body.message_type == "set_task") | .body.data.pipeline.translations[0].target_language', str(file_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    target_lang = result.stdout.strip()

    return config_data, source_lang, target_lang

def load_run_result(file_path: Path) -> tuple[IoData, dict]:
    """Load and validate run_result.json file using jq for efficiency"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not (file_path.name.endswith('_result.json') or file_path.name.endswith('.result.json')):
        raise ValueError(f"File must be a result.json file, got: {file_path.name}")

    print(f"Loading IoData from {file_path}...")

    # Extract only io_data using jq to avoid loading full file
    cmd = ['jq', '.io_data', str(file_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    io_data_dict = from_json(result.stdout)

    events = []
    print(f"Processing {len(io_data_dict.get('events', []))} events...")
    for event_dict in io_data_dict['events']:
        # Convert body dict back to bytes (same as original rewind implementation)
        body_bytes = to_json(event_dict['body']) if event_dict['body'] else b'{}'

        head_dict = event_dict['head']
        head = Dbg(
            kind=head_dict['kind'],
            ch=head_dict['ch'],
            dir=head_dict['dir'],
            dawn_ts=head_dict['dawn_ts'],
            perf_ts=head_dict['perf_ts'],
            utc_ts=head_dict['utc_ts'],
            idx=head_dict['idx'],
            num=head_dict['num'],
            dur_s=head_dict['dur_s'],
            rms_db=head_dict['rms_db']
        )

        event = IoEvent(
            head=head,
            body=body_bytes,
            tid=event_dict['tid'],
            mtype=event_dict['mtype']
        )
        events.append(event)

    io_data = IoData(
        start_perf_ts=io_data_dict['start_perf_ts'],
        start_utc_ts=io_data_dict['start_utc_ts'],
        in_sr=io_data_dict['in_sr'],
        out_sr=io_data_dict['out_sr'],
        mode=io_data_dict['mode'],
        channels=io_data_dict['channels'],
        events=events,
        count_events=len(events)
    )

    return io_data, io_data_dict

def generate_rewind_report(report: Report, io_data: IoData, config: Config, source_lang: str, target_lang: str, file_path: Path) -> str:
    """Generate text report using existing format_report function"""
    try:
        print("Generating report...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        in_file = str(file_path)
        out_file = f"{timestamp}_rewind_out_{target_lang}.wav"

        # Use the exact same format_report function as main benchmark
        return format_report(report, io_data, source_lang, target_lang, in_file, out_file, config)

    except Exception as e:
        import traceback
        return f"Error generating report: {e}\n\nTraceback:\n{traceback.format_exc()}"

def main():
    parser = argparse.ArgumentParser(description="Analyze Palabra AI benchmark run_result.json files")
    parser.add_argument("run_result", help="Path to run_result.json file")
    parser.add_argument("--out", type=Path, help="Output directory for reconstructed files (if not specified, only prints to console)")
    args = parser.parse_args()

    try:
        file_path = Path(args.run_result)

        # Extract config and languages using jq
        config_data, source_lang, target_lang = extract_config_from_result(file_path)

        # Reconstruct Config object from the extracted data
        config = Config.from_dict(config_data)

        # Load IoData
        io_data, _ = load_run_result(file_path)

        # Parse report (same as main benchmark)
        report, in_audio_canvas, out_audio_canvas = Report.parse(io_data)

        # Generate report using existing format_report function
        report_text = generate_rewind_report(report, io_data, config, source_lang, target_lang, file_path)

        # Save files if --out option is specified
        if args.out:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_benchmark_files(
                output_dir=args.out,
                timestamp=timestamp,
                report=report,
                io_data=io_data,
                config=config,
                result=None,  # No RunResult in rewind
                in_audio_canvas=in_audio_canvas,
                out_audio_canvas=out_audio_canvas,
                source_lang=source_lang,
                target_lang=target_lang,
                report_text=report_text,
                input_file_path=str(file_path),
                file_prefix="rewind"
            )
            print(f"\nFiles saved to: {args.out}")

        # Always print report to console
        print("\n" + report_text)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()