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
    lines.append(f"Total audio chunks:     {summary['total_chunks']}")
    lines.append(f"Total duration:         {summary['total_duration']:.1f} seconds")
    # Handle backward compatibility for new fields
    if 'average_completion' in summary:
        lines.append(f"Average completion:     {summary['average_completion']:.1%}")
    
    lines.append(f"Chunks transcribed:     {summary['chunks_with_validated']} ({summary['chunks_with_validated']/summary['total_chunks']*100:.1f}%)")
    lines.append(f"Chunks translated:      {summary['chunks_with_translation']} ({summary['chunks_with_translation']/summary['total_chunks']*100:.1f}%)")
    lines.append(f"Chunks with TTS:        {summary['chunks_with_tts']} ({summary['chunks_with_tts']/summary['total_chunks']*100:.1f}%)")
    
    # Handle backward compatibility for empty chunks count
    empty_chunks = summary.get('empty_chunks', 0)
    lines.append(f"Empty chunks:           {empty_chunks} ({empty_chunks/summary['total_chunks']*100:.1f}%)")
    
    # Pipeline stage breakdown (if available)
    if 'pipeline_stages' in summary:
        lines.append(f"\nPIPELINE STAGE BREAKDOWN")
        lines.append("-" * 40)
        pipeline_stages = summary["pipeline_stages"]
        stage_names = {"empty": "Empty (no data)", "partial": "Partial ASR", "validated": "Validated ASR", "translated": "Translated", "complete": "Complete (with TTS)"}
        for stage, count in pipeline_stages.items():
            percentage = count / summary['total_chunks'] * 100
            lines.append(f"{stage_names[stage]:20s} {count:3d} ({percentage:5.1f}%)")
    
    # Statistics
    lines.append("\n" + "=" * 80)
    lines.append("LATENCY STATISTICS (seconds)")
    lines.append("=" * 80)
    lines.append("\nLatency = Time from sending audio chunk to receiving response from API")
    
    metrics_order = ["asr_first_partial", "asr_validated", "translation", "tts_audio"]
    metric_names = {
        "asr_first_partial": "ASR First Partial",
        "asr_validated": "ASR Validated",
        "translation": "Translation (S2TT)",
        "tts_audio": "TTS Audio Output"
    }
    
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
    
    # Box plots for each metric
    lines.append("\n" + "=" * 80)
    lines.append("LATENCY DISTRIBUTION (Box Plots)")
    lines.append("=" * 80)
    lines.append("\nEach metric showing: Min ├──[Q1 ███ Median ███ Q3]──┤ Max\n")
    
    # Collect values for each metric
    metric_values = {}
    for chunk in analysis["chunks"].values():
        for metric in ["asr_first_partial", "asr_validated", "translation", "tts_audio"]:
            if chunk[metric] is not None:
                if metric not in metric_values:
                    metric_values[metric] = []
                metric_values[metric].append(chunk[metric])
    
    # Create box plots
    for metric in ["asr_first_partial", "asr_validated", "translation", "tts_audio"]:
        if metric in metric_values:
            label = metric_names[metric]
            lines.append(create_ascii_box_plot(metric_values[metric], width=50, label=label))
            lines.append("")
    
    # Histogram for ASR Validated latency
    if "asr_validated" in metric_values:
        lines.append("\n" + "=" * 80)
        lines.append("ASR VALIDATED LATENCY DISTRIBUTION (Histogram)")
        lines.append("=" * 80)
        lines.append("\n" + create_ascii_histogram(
            metric_values["asr_validated"], 
            bins=15, 
            width=50, 
            title="Frequency Distribution"
        ))
    
    # Time series chart
    lines.append("\n" + "=" * 80)
    lines.append("LATENCY PROGRESSION OVER TIME")
    lines.append("=" * 80)
    
    # Prepare time series data
    x_vals = []
    y_vals = []
    for window_id in sorted(analysis["time_progression"].keys()):
        window = analysis["time_progression"][window_id]
        if window["mean"] is not None:
            x_vals.append(window["start_time"])
            y_vals.append(window["mean"])
    
    if x_vals and y_vals:
        lines.append("\n" + create_ascii_time_series(
            x_vals, y_vals, 
            width=70, height=12, 
            title="Mean Latency Over Time"
        ))
    
    lines.append("\nTime Window Details:")
    lines.append("-" * 60)
    lines.append("Time Window        Chunks  Mean Latency  Median Latency")
    lines.append("-" * 60)
    
    time_windows = sorted(analysis["time_progression"].items())
    for window_id, window_data in time_windows[:10]:  # Show first 10 windows
        if window_data["mean"] is not None:
            lines.append(f"[{window_data['start_time']:5.1f}s - {window_data['end_time']:5.1f}s]    "
                        f"{window_data['count']:3d}     {window_data['mean']:6.3f}s      {window_data['median']:6.3f}s")
    
    if len(time_windows) > 10:
        lines.append(f"... and {len(time_windows) - 10} more time windows")
    
    # Sample chunk details
    lines.append("\n" + "=" * 80)
    
    # Filter chunks based on show_empty parameter
    chunk_items = list(analysis["chunks"].items())
    if not show_empty:
        chunk_items = [(chunk_id, chunk) for chunk_id, chunk in chunk_items if not chunk.get("is_empty", False)]
    
    total_chunks = len(chunk_items)
    total_all_chunks = len(analysis["chunks"])
    
    # Select chunks to display
    if max_chunks == -1 or total_chunks <= max_chunks:
        # Show all chunks
        selected_chunks = chunk_items
        if max_chunks == -1:
            lines.append(f"ALL CHUNK DETAILS ({total_chunks} chunks shown)")
        else:
            lines.append(f"CHUNK DETAILS (all {total_chunks} chunks)")
        if not show_empty and total_all_chunks != total_chunks:
            lines.append(f"Note: {total_all_chunks - total_chunks} empty chunks hidden (use --show-empty to display)")
    else:
        # Select representative sample
        if max_chunks <= 3:
            # For very small numbers, just take first chunks
            selected_chunks = chunk_items[:max_chunks]
        else:
            # Select first, last, and evenly distributed middle chunks
            selected_indices = [0]  # First chunk
            
            # Add evenly distributed middle chunks
            if max_chunks > 2:
                step = (total_chunks - 1) / (max_chunks - 1)
                for i in range(1, max_chunks - 1):
                    selected_indices.append(int(i * step))
            
            selected_indices.append(total_chunks - 1)  # Last chunk
            
            # Remove duplicates and sort
            selected_indices = sorted(set(selected_indices))
            selected_chunks = [chunk_items[i] for i in selected_indices]
        
        lines.append(f"SAMPLE CHUNK DETAILS (showing {len(selected_chunks)} of {total_chunks} chunks)")
        if not show_empty and total_all_chunks != total_chunks:
            lines.append(f"Note: {total_all_chunks - total_chunks} empty chunks hidden")
    
    lines.append("=" * 80)
    lines.append("\nNote: Numbers show latency (seconds from chunk send to response receipt)")
    lines.append("      Offset is the chunk's position in the audio file")
    lines.append("-" * 60)
    
    for chunk_id, chunk in selected_chunks:
        # Handle backward compatibility for new fields
        stage = chunk.get("pipeline_stage", "unknown")
        completion = chunk.get("completion_score", 0.0)
        
        # Determine if chunk is empty based on available data
        is_empty = chunk.get("is_empty", False)
        if not is_empty and stage == "unknown":
            # For backward compatibility, calculate if empty
            is_empty = (chunk.get("asr_first_partial") is None and 
                       chunk.get("asr_validated") is None and
                       not chunk.get("partial_text", "") and
                       not chunk.get("validated_text", ""))
        
        if stage != "unknown" or completion > 0.0:
            lines.append(f"\nChunk {chunk_id} (audio position: {chunk['time_offset']:.2f}s, stage: {stage}, completion: {completion:.0%})")
        else:
            lines.append(f"\nChunk {chunk_id} (audio position: {chunk['time_offset']:.2f}s)")
        
        if is_empty:
            lines.append("  Status: Empty (no transcription data)")
        else:
            if chunk.get("asr_first_partial"):
                lines.append(f"  ASR Partial:  {chunk['asr_first_partial']:.3f}s latency → \"{chunk.get('partial_text', '')}\"")
            if chunk.get("asr_validated"):
                lines.append(f"  ASR Valid:    {chunk['asr_validated']:.3f}s latency → \"{chunk.get('validated_text', '')}\"")
            if chunk.get("translation"):
                lines.append(f"  Translation:  {chunk['translation']:.3f}s latency → \"{chunk.get('translated_text', '')}\"")
            if chunk.get("tts_audio"):
                lines.append(f"  TTS Audio:    {chunk['tts_audio']:.3f}s latency (audio output started)")
    
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
        "asr_first_partial": "ASR First Partial",
        "asr_validated": "ASR Validated",
        "translation": "Translation",
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