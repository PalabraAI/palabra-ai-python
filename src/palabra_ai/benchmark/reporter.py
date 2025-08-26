"""
Report generation for Palabra AI benchmark results
Supports text (console), HTML, and JSON formats
"""

import statistics
import datetime
from typing import List, Dict, Any
from pathlib import Path
from jinja2 import Template

from palabra_ai.util.orjson import to_json


def create_ascii_histogram(values: List[float], bins: int = 20, width: int = 50, title: str = "") -> str:
    """Create ASCII histogram for console output"""
    if not values:
        return "No data"
    
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        return f"All values: {min_val:.3f}s"
    
    # Create bins
    bin_width = (max_val - min_val) / bins
    bin_counts = [0] * bins
    
    for val in values:
        bin_idx = min(int((val - min_val) / bin_width), bins - 1)
        bin_counts[bin_idx] += 1
    
    max_count = max(bin_counts)
    
    # Build histogram
    lines = []
    if title:
        lines.append(title)
        lines.append("-" * (width + 20))
    
    for i, count in enumerate(bin_counts):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "█" * bar_len + "░" * (width - bar_len)
        lines.append(f"{bin_start:6.2f}s |{bar}| {count:3d}")
    
    return "\n".join(lines)


def create_ascii_box_plot(values: List[float], width: int = 60, label: str = "") -> str:
    """Create ASCII box plot for console output"""
    if not values:
        return "No data"
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    min_val = sorted_vals[0]
    q1 = sorted_vals[int(n * 0.25)]
    median = sorted_vals[int(n * 0.5)]
    q3 = sorted_vals[int(n * 0.75)]
    max_val = sorted_vals[-1]
    
    # Normalize to width
    range_val = max_val - min_val
    if range_val == 0:
        return f"{label}: All values = {min_val:.3f}s"
    
    def pos(val):
        return int((val - min_val) / range_val * width)
    
    # Create the plot
    line = [" "] * (width + 1)
    
    # Min to Q1
    for i in range(pos(min_val), pos(q1)):
        line[i] = "─"
    
    # Q1 to Q3 (box)
    for i in range(pos(q1), pos(q3) + 1):
        line[i] = "█"
    
    # Q3 to Max
    for i in range(pos(q3) + 1, pos(max_val) + 1):
        line[i] = "─"
    
    # Mark special points
    line[pos(min_val)] = "├"
    line[pos(max_val)] = "┤"
    line[pos(median)] = "┃"
    
    result = f"{label:20s} {''.join(line)}\n"
    
    # Build labels line with proper spacing
    labels_line = " " * 20  # Start with label indent
    labels_line += f"{min_val:.2f}s"
    
    # Calculate positions and add spacing
    q1_pos = 20 + pos(q1)
    median_pos = 20 + pos(median)
    q3_pos = 20 + pos(q3)
    max_pos = 20 + pos(max_val)
    
    # Add Q1 label with spacing
    q1_label = f"Q1:{q1:.2f}s"
    spacing_to_q1 = max(2, pos(q1) - len(f"{min_val:.2f}s") - 2)
    labels_line += " " * spacing_to_q1 + q1_label
    
    # Add Median label with spacing
    median_label = f"M:{median:.2f}s"
    spacing_to_median = max(2, pos(median) - pos(q1) - len(q1_label) - 2)
    labels_line += " " * spacing_to_median + median_label
    
    # Add Q3 label with spacing
    q3_label = f"Q3:{q3:.2f}s"
    spacing_to_q3 = max(2, pos(q3) - pos(median) - len(median_label) - 2)
    labels_line += " " * spacing_to_q3 + q3_label
    
    # Add Max label with spacing
    spacing_to_max = max(2, pos(max_val) - pos(q3) - len(q3_label) - 2)
    labels_line += " " * spacing_to_max + f"{max_val:.2f}s"
    
    result += labels_line
    
    return result


def create_ascii_time_series(x_values: List[float], y_values: List[float], 
                            width: int = 70, height: int = 15, title: str = "") -> str:
    """Create ASCII line chart for time series data"""
    if not x_values or not y_values:
        return "No data"
    
    min_y, max_y = min(y_values), max(y_values)
    min_x, max_x = min(x_values), max(x_values)
    
    # Create grid
    grid = [[" " for _ in range(width)] for _ in range(height)]
    
    # Plot points
    for x, y in zip(x_values, y_values):
        x_pos = int((x - min_x) / (max_x - min_x) * (width - 1)) if max_x > min_x else 0
        y_pos = height - 1 - int((y - min_y) / (max_y - min_y) * (height - 1)) if max_y > min_y else height // 2
        
        if 0 <= x_pos < width and 0 <= y_pos < height:
            grid[y_pos][x_pos] = "●"
    
    # Connect points with lines
    for i in range(len(x_values) - 1):
        x1 = int((x_values[i] - min_x) / (max_x - min_x) * (width - 1)) if max_x > min_x else 0
        x2 = int((x_values[i + 1] - min_x) / (max_x - min_x) * (width - 1)) if max_x > min_x else 0
        y1 = height - 1 - int((y_values[i] - min_y) / (max_y - min_y) * (height - 1)) if max_y > min_y else height // 2
        y2 = height - 1 - int((y_values[i + 1] - min_y) / (max_y - min_y) * (height - 1)) if max_y > min_y else height // 2
        
        # Simple line interpolation
        if x2 > x1:
            for x in range(x1, x2):
                y = y1 + (y2 - y1) * (x - x1) // (x2 - x1)
                if 0 <= x < width and 0 <= y < height and grid[y][x] == " ":
                    grid[y][x] = "·"
    
    # Build output
    lines = []
    if title:
        lines.append(title)
        lines.append("─" * width)
    
    # Add Y axis labels
    for i, row in enumerate(grid):
        y_val = max_y - (i / (height - 1)) * (max_y - min_y) if height > 1 else max_y
        lines.append(f"{y_val:6.2f}s │{''.join(row)}")
    
    # X axis
    lines.append(f"{'':8s}└" + "─" * width)
    lines.append(f"{'':8s}{min_x:.1f}s" + " " * (width - 12) + f"{max_x:.1f}s")
    
    return "\n".join(lines)


def generate_text_report(analysis: Dict[str, Any], max_chunks: int = -1, show_empty: bool = False) -> str:
    """
    Generate formatted text report for console output
    
    Args:
        analysis: Analysis data from analyze_latency
        max_chunks: Maximum number of chunks to display in detail (-1 for all)
        show_empty: Whether to include empty chunks in the detailed view
    """
    
    lines = []
    lines.append("=" * 80)
    lines.append("PALABRA BENCHMARK LATENCY ANALYSIS REPORT")
    lines.append("=" * 80)
    
    # Summary
    summary = analysis["summary"]
    lines.append(f"\nSUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total audio chunks:      {summary['total_chunks']}")
    lines.append(f"Chunks with sound:       {summary.get('chunks_with_sound', 0)} (RMS > {summary.get('rms_threshold_db', -40.0):.1f} dB)")
    lines.append(f"Silent chunks:           {summary.get('silent_chunks', 0)} ({summary.get('silent_percentage', 0):.1f}%)")
    lines.append(f"Chunk duration:          {summary.get('chunk_duration_ms', 100):.0f} ms")
    lines.append(f"Total duration:          {summary['total_duration']:.1f} seconds")
    lines.append(f"")
    lines.append(f"Events processed:")
    lines.append(f"  Partial transcriptions:    {summary['chunks_with_partial']}")
    lines.append(f"  Validated transcriptions:  {summary['chunks_with_validated']}")
    lines.append(f"  Translations:              {summary['chunks_with_translation']}")
    lines.append(f"  TTS audio outputs:         {summary['chunks_with_tts']}")
    if summary.get('out_audio_chunks', 0) > 0:
        lines.append(f"  OUT audio chunks:          {summary['out_audio_chunks']}")
    
    
    # Statistics
    lines.append("\n" + "=" * 80)
    lines.append("LATENCY STATISTICS (seconds)")
    lines.append("=" * 80)
    lines.append("\nOne-way latency from sending audio chunk to receiving response")
    lines.append("")
    
    # Simplified table format
    lines.append("Category                          Count    Min    Avg    Max    p50    p90    p95")
    lines.append("-" * 78)
    
    metrics_order = ["partial_transcription", "validated_transcription", "translated_transcription", "tts_audio"]
    metric_names = {
        "partial_transcription": "Partial Transcription Latency",
        "validated_transcription": "Validated Transcription Latency",
        "translated_transcription": "Translated Transcription Latency",
        "tts_audio": "Translated Speech (TTS) Latency"
    }
    
    for metric in metrics_order:
        if metric in analysis["statistics"]:
            stats = analysis["statistics"][metric]
            name = metric_names[metric]
            # Format as table row
            lines.append(f"{name:33s} {stats['count']:5d} {stats['min']:6.3f} {stats['mean']:6.3f} {stats['max']:6.3f} {stats['p50']:6.3f} {stats['p90']:6.3f} {stats['p95']:6.3f}")
    
    # Detailed statistics
    lines.append("\n" + "=" * 80)
    lines.append("DETAILED LATENCY STATISTICS")
    lines.append("=" * 80)
    
    for metric in metrics_order:
        if metric in analysis["statistics"]:
            stats = analysis["statistics"][metric]
            lines.append(f"\n{metric_names[metric]} ({stats['count']} samples)")
            lines.append("-" * 40)
            lines.append(f"  Min:    {stats['min']:.3f}s")
            lines.append(f"  P25:    {stats['p25']:.3f}s")
            lines.append(f"  P50:    {stats['p50']:.3f}s (median)")
            lines.append(f"  P75:    {stats['p75']:.3f}s")
            lines.append(f"  P90:    {stats['p90']:.3f}s")
            lines.append(f"  P95:    {stats['p95']:.3f}s")
            lines.append(f"  P99:    {stats['p99']:.3f}s")
            lines.append(f"  Max:    {stats['max']:.3f}s")
            lines.append(f"  Mean:   {stats['mean']:.3f}s")
            lines.append(f"  StDev:  {stats['stdev']:.3f}s")
    
    # Box plots for each metric (if measurements available)
    if "measurements" in analysis:
        lines.append("\n" + "=" * 80)
        lines.append("LATENCY DISTRIBUTION (Box Plots)")
        lines.append("=" * 80)
        lines.append("\nEach metric showing: Min ├──[Q1 ███ Median ███ Q3]──┤ Max\n")
        
        measurements = analysis["measurements"]
        for metric in metrics_order:
            if metric in measurements and measurements[metric]:
                label = metric_names[metric]
                lines.append(create_ascii_box_plot(measurements[metric], width=50, label=label))
                lines.append("")
    
    # Histogram for Validated Transcription latency
    if "measurements" in analysis and "validated_transcription" in analysis["measurements"]:
        validated_latencies = analysis["measurements"]["validated_transcription"]
        if validated_latencies:
            lines.append("\n" + "=" * 80)
            lines.append("VALIDATED TRANSCRIPTION LATENCY DISTRIBUTION (Histogram)")
            lines.append("=" * 80)
            lines.append("\n" + create_ascii_histogram(
                validated_latencies, 
                bins=15, 
                width=50, 
                title="Frequency Distribution"
            ))
    
    # Sample chunk details with latencies and texts - show ALL validated chunks
    if "chunk_details" in analysis and analysis["chunk_details"]:
        lines.append("\n" + "=" * 80)
        lines.append("TRANSCRIPTION DETAILS (All Validated Chunks)")
        lines.append("=" * 80)
        
        # Sort chunk details by chunk index
        sorted_chunks = sorted(analysis["chunk_details"].items(), key=lambda x: x[0])
        
        # Show ALL chunks that have validated transcription
        displayed_count = 0
        for chunk_idx, details in sorted_chunks:
            # Only show chunks with validated transcription
            if details["validated_latency"] is None:
                continue
            
            displayed_count += 1
            lines.append(f"\nChunk #{chunk_idx} (Audio time: {details['audio_time']:.2f}s)")
            lines.append("-" * 60)
            
            # Show partial transcription
            if details["partial_latency"] is not None:
                text = details["partial_text"][:80] + "..." if len(details["partial_text"]) > 80 else details["partial_text"]
                lines.append(f"  Partial ({details['partial_latency']:.3f}s): {text}")
            
            # Show validated transcription
            text = details["validated_text"][:80] + "..." if len(details["validated_text"]) > 80 else details["validated_text"]
            lines.append(f"  Validated ({details['validated_latency']:.3f}s): {text}")
            
            # Show translation
            if details["translated_latency"] is not None:
                text = details["translated_text"][:80] + "..." if len(details["translated_text"]) > 80 else details["translated_text"]
                lines.append(f"  Translation ({details['translated_latency']:.3f}s): {text}")
            
            # Show TTS
            if details.get("tts_latency") is not None:
                lines.append(f"  TTS ({details['tts_latency']:.3f}s): Audio output started")
        
        if displayed_count == 0:
            lines.append("\nNo chunks with validated transcription found.")
    elif "measurements" in analysis:
        # Fallback to simpler display if chunk_details not available
        lines.append("\n" + "=" * 80)
        lines.append("SAMPLE CHUNK DETAILS")
        lines.append("=" * 80)
        
        # Show latency ranges for each event type
        lines.append("\nLatency Ranges by Event Type:")
        lines.append("-" * 60)
        
        event_names = {
            "partial_transcription": "Partial Transcription",
            "validated_transcription": "Validated Transcription",
            "translated_transcription": "Translation",
            "tts_audio": "TTS Audio"
        }
        
        for event_type, name in event_names.items():
            if event_type in analysis["measurements"] and analysis["measurements"][event_type]:
                latencies = analysis["measurements"][event_type]
                if latencies:
                    min_lat = min(latencies)
                    max_lat = max(latencies)
                    lines.append(f"{name:25s}: {min_lat:6.3f}s (min) - {max_lat:6.3f}s (max)")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def generate_html_report(analysis: Dict[str, Any]) -> str:
    """Generate HTML report with interactive charts using Jinja2 template"""
    
    # Load template
    template_path = Path(__file__).parent / "templates" / "report.html"
    with open(template_path, 'r') as f:
        template = Template(f.read())
    
    # Prepare data for charts
    stats = analysis["statistics"]
    
    # Time series data
    time_series_labels = []
    time_series_mean = []
    time_series_median = []
    
    for window_id in sorted(analysis["time_progression"].keys()):
        window = analysis["time_progression"][window_id]
        if window["mean"] is not None:
            time_series_labels.append(window["start_time"])
            time_series_mean.append(window["mean"])
            time_series_median.append(window["median"])
    
    # Percentile comparison data
    percentile_data = {
        "categories": ["P25", "P50", "P75", "P90", "P95", "P99"],
        "series": []
    }
    
    metric_labels = {
        "partial_transcription": "Partial Transcription",
        "validated_transcription": "Validated Transcription",
        "translated_transcription": "Translation",
        "tts_audio": "TTS Audio"
    }
    
    for metric, label in metric_labels.items():
        if metric in stats:
            s = stats[metric]
            percentile_data["series"].append({
                "name": label,
                "data": [s["p25"], s["p50"], s["p75"], s["p90"], s["p95"], s["p99"]]
            })
    
    # Render template
    return template.render(
        generated_at=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        summary=analysis['summary'],
        statistics=stats,
        metric_labels=metric_labels,
        percentile_data=percentile_data,
        time_series_labels=time_series_labels,
        time_series_mean=time_series_mean,
        time_series_median=time_series_median
    )
    


def generate_json_report(analysis: Dict[str, Any], raw_result: bool = False, raw_result_data: Dict[str, Any] = None) -> bytes:
    """
    Generate JSON report
    
    Args:
        analysis: Analysis data from analyze_latency
        raw_result: Whether to include full raw result data
        raw_result_data: Raw result data to include (if raw_result is True)
    """
    report_data = analysis.copy()
    
    if raw_result and raw_result_data is not None:
        report_data["raw_result"] = raw_result_data
    
    return to_json(report_data, indent=True)


def save_html_report(analysis: Dict[str, Any], output_file: Path) -> None:
    """Save HTML report to file"""
    html_content = generate_html_report(analysis)
    output_file.write_text(html_content)


def save_json_report(analysis: Dict[str, Any], output_file: Path) -> None:
    """Save JSON report to file"""
    json_content = generate_json_report(analysis)
    output_file.write_bytes(json_content)