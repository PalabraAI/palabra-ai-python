#!/usr/bin/env python3
"""
Palabra AI Benchmark Rewind - Analyze existing run_result.json files
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from palabra_ai.util.orjson import from_json, to_json
from palabra_ai.model import IoData
from palabra_ai.message import IoEvent, Dbg
from palabra_ai.benchmark.report import save_benchmark_files
from palabra_ai.benchmark.report import Report
from palabra_ai import Config

def load_run_result(file_path: Path) -> IoData:
    """Load and validate run_result.json file"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not (file_path.name.endswith('_result.json') or file_path.name.endswith('.result.json')):
        raise ValueError(f"File must be a result.json file, got: {file_path.name}")

    print(f"Loading IoData from {file_path}...")

    # Load full JSON file
    data = from_json(file_path.read_text())
    io_data_dict = data['io_data']

    events = []
    print(f"Processing {len(io_data_dict.get('events', []))} events...")
    for event_dict in io_data_dict['events']:
        head_dict = event_dict['head']
        head = Dbg(**head_dict)

        event = IoEvent(
            head=head,
            body=to_json(event_dict['body']) if event_dict['body'] else b'{}',
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
        count_events=len(events),
        reader_x_title=io_data_dict.get('reader_x_title', "n/a"),
        writer_x_title=io_data_dict.get('writer_x_title', "n/a")
    )

    return io_data

def main():
    parser = argparse.ArgumentParser(description="Analyze Palabra AI benchmark run_result.json files")
    parser.add_argument("run_result", help="Path to run_result.json file")
    parser.add_argument("--out", type=Path, help="Output directory for reconstructed files (if not specified, only prints to console)")
    args = parser.parse_args()


    file_path = Path(args.run_result)

    # Load IoData
    io_data = load_run_result(file_path)

    # Parse report (same as main benchmark)
    report = Report.parse(io_data)

    # Save files if --out option is specified
    if args.out:
        report.cfg.output_dir = Path(args.out)
        report.save_all()
        print(f"\nFiles saved to: {args.out}")

    # Always print report to console
    print("\n" + report.format_report())


if __name__ == "__main__":
    main()