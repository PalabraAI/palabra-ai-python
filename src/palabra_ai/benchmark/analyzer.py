"""
Latency analyzer for Palabra AI messages
Provides data analysis for benchmark results
"""

import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ChunkMetrics:
    """Metrics for a single audio chunk"""
    chunk_id: int
    send_timestamp: float
    time_offset: float  # Seconds from start
    
    # Latencies from chunk send time
    asr_first_partial: Optional[float] = None
    asr_validated: Optional[float] = None
    translation: Optional[float] = None
    tts_audio: Optional[float] = None
    
    # Text content
    partial_text: str = ""
    validated_text: str = ""
    translated_text: str = ""
    
    # Associated IDs
    transcription_ids: List[str] = field(default_factory=list)
    
    @property
    def is_empty(self) -> bool:
        """Check if chunk has no transcription data"""
        return (self.asr_first_partial is None and 
                self.asr_validated is None and 
                not self.partial_text and 
                not self.validated_text)
    
    @property
    def pipeline_stage(self) -> str:
        """Get the furthest pipeline stage reached"""
        if self.tts_audio is not None:
            return "complete"
        elif self.translation is not None:
            return "translated"
        elif self.asr_validated is not None:
            return "validated"
        elif self.asr_first_partial is not None:
            return "partial"
        else:
            return "empty"
    
    @property
    def completion_score(self) -> float:
        """Calculate completion score (0.0 to 1.0)"""
        score = 0.0
        if self.asr_first_partial is not None:
            score += 0.25
        if self.asr_validated is not None:
            score += 0.25
        if self.translation is not None:
            score += 0.25
        if self.tts_audio is not None:
            score += 0.25
        return score


def calculate_percentiles(data: List[float]) -> Dict[str, float]:
    """Calculate statistical percentiles for a dataset"""
    if not data:
        return {}
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    return {
        "min": sorted_data[0],
        "p25": sorted_data[min(int(n * 0.25), n-1)],
        "p50": sorted_data[min(int(n * 0.50), n-1)],
        "p75": sorted_data[min(int(n * 0.75), n-1)],
        "p90": sorted_data[min(int(n * 0.90), n-1)],
        "p95": sorted_data[min(int(n * 0.95), n-1)],
        "p99": sorted_data[min(int(n * 0.99), n-1)],
        "max": sorted_data[-1],
        "mean": statistics.mean(sorted_data),
        "stdev": statistics.stdev(sorted_data) if n > 1 else 0,
        "count": n
    }


def analyze_latency(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze message trace for latency metrics
    
    Returns comprehensive latency analysis including:
    - Per-chunk latencies
    - Aggregate statistics
    - Time progression analysis
    """
    
    # Constants
    CHUNK_DURATION = 0.32  # 320ms per chunk at 24kHz
    
    # Collect all input audio chunks
    chunks = {}
    for msg in messages:
        if msg.get("dir") == "in" and msg.get("kind") == "audio":
            chunk_num = msg.get("num")
            if chunk_num is not None:
                chunks[chunk_num] = ChunkMetrics(
                    chunk_id=chunk_num,
                    send_timestamp=msg.get("ts"),
                    time_offset=chunk_num * CHUNK_DURATION
                )
    
    if not chunks:
        raise ValueError("No audio chunks found in messages")
    
    # Process transcription messages
    for msg in messages:
        if msg.get("dir") != "out":
            continue
        
        msg_type = msg.get("msg", {}).get("message_type")
        msg_ts = msg.get("ts")
        data = msg.get("msg", {}).get("data", {})
        
        if msg_type in ["partial_transcription", "validated_transcription", "translated_transcription"]:
            transcription = data.get("transcription", {})
            trans_id = transcription.get("transcription_id")
            segments = transcription.get("segments", [])
            text = transcription.get("text", "")
            
            if segments:
                seg_start = segments[0].get("start", 0)
                seg_end = segments[-1].get("end", 0)
                
                # Map to chunks based on timing overlap
                for chunk_id, chunk in chunks.items():
                    chunk_start = chunk.time_offset
                    chunk_end = chunk_start + CHUNK_DURATION
                    
                    # Check overlap
                    if seg_start < chunk_end and seg_end > chunk_start:
                        latency = msg_ts - chunk.send_timestamp
                        
                        if trans_id and trans_id not in chunk.transcription_ids:
                            chunk.transcription_ids.append(trans_id)
                        
                        if msg_type == "partial_transcription":
                            if chunk.asr_first_partial is None or latency < chunk.asr_first_partial:
                                chunk.asr_first_partial = latency
                                chunk.partial_text = text
                        
                        elif msg_type == "validated_transcription":
                            chunk.asr_validated = latency
                            chunk.validated_text = text
                        
                        elif msg_type == "translated_transcription":
                            chunk.translation = latency
                            chunk.translated_text = text
        
        # Handle TTS audio output
        elif msg.get("kind") == "audio" and msg.get("dir") == "out":
            # Simple mapping - correlate based on timing
            for chunk in chunks.values():
                if chunk.tts_audio is None and chunk.translation:
                    time_since_send = msg_ts - chunk.send_timestamp
                    if chunk.translation < time_since_send < chunk.translation + 2.0:
                        chunk.tts_audio = time_since_send
                        break
    
    # Collect all latencies
    all_metrics = {
        "asr_first_partial": [],
        "asr_validated": [],
        "translation": [],
        "tts_audio": []
    }
    
    for chunk in chunks.values():
        if chunk.asr_first_partial is not None:
            all_metrics["asr_first_partial"].append(chunk.asr_first_partial)
        if chunk.asr_validated is not None:
            all_metrics["asr_validated"].append(chunk.asr_validated)
        if chunk.translation is not None:
            all_metrics["translation"].append(chunk.translation)
        if chunk.tts_audio is not None:
            all_metrics["tts_audio"].append(chunk.tts_audio)
    
    # Calculate statistics
    statistics_data = {}
    for metric_name, values in all_metrics.items():
        if values:
            statistics_data[metric_name] = calculate_percentiles(values)
    
    # Time progression analysis (group by 5-second windows)
    time_windows = {}
    window_size = 5.0  # seconds
    
    for chunk in chunks.values():
        window = int(chunk.time_offset / window_size)
        if window not in time_windows:
            time_windows[window] = {
                "start_time": window * window_size,
                "end_time": (window + 1) * window_size,
                "latencies": []
            }
        
        # Use validated transcription as primary metric
        if chunk.asr_validated is not None:
            time_windows[window]["latencies"].append(chunk.asr_validated)
    
    # Calculate window statistics
    for window_data in time_windows.values():
        if window_data["latencies"]:
            window_data["mean"] = statistics.mean(window_data["latencies"])
            window_data["median"] = statistics.median(window_data["latencies"])
            window_data["count"] = len(window_data["latencies"])
        else:
            window_data["mean"] = None
            window_data["median"] = None
            window_data["count"] = 0
    
    # Convert chunks to serializable format
    chunks_data = {}
    pipeline_stage_counts = {"empty": 0, "partial": 0, "validated": 0, "translated": 0, "complete": 0}
    
    for chunk_id, chunk in sorted(chunks.items()):
        chunks_data[chunk_id] = {
            "time_offset": chunk.time_offset,
            "asr_first_partial": chunk.asr_first_partial,
            "asr_validated": chunk.asr_validated,
            "translation": chunk.translation,
            "tts_audio": chunk.tts_audio,
            "partial_text": chunk.partial_text[:50] + "..." if len(chunk.partial_text) > 50 else chunk.partial_text,
            "validated_text": chunk.validated_text[:50] + "..." if len(chunk.validated_text) > 50 else chunk.validated_text,
            "translated_text": chunk.translated_text[:50] + "..." if len(chunk.translated_text) > 50 else chunk.translated_text,
            "transcription_ids": chunk.transcription_ids,
            "is_empty": chunk.is_empty,
            "pipeline_stage": chunk.pipeline_stage,
            "completion_score": chunk.completion_score
        }
        
        # Count pipeline stages
        pipeline_stage_counts[chunk.pipeline_stage] += 1
    
    # Find first audio timestamp as reference
    first_audio_ts = min(chunk.send_timestamp for chunk in chunks.values())
    
    # Calculate average completion score
    completion_scores = [chunk.completion_score for chunk in chunks.values()]
    avg_completion = sum(completion_scores) / len(completion_scores) if completion_scores else 0.0
    
    # Build results
    return {
        "summary": {
            "total_chunks": len(chunks),
            "chunks_with_partial": len(all_metrics["asr_first_partial"]),
            "chunks_with_validated": len(all_metrics["asr_validated"]),
            "chunks_with_translation": len(all_metrics["translation"]),
            "chunks_with_tts": len(all_metrics["tts_audio"]),
            "empty_chunks": pipeline_stage_counts["empty"],
            "pipeline_stages": pipeline_stage_counts,
            "average_completion": avg_completion,
            "first_audio_timestamp": first_audio_ts,
            "total_duration": max(chunk.time_offset for chunk in chunks.values()) + CHUNK_DURATION
        },
        "statistics": statistics_data,
        "time_progression": time_windows,
        "chunks": chunks_data,
        "metric_descriptions": {
            "asr_first_partial": "Latency from audio chunk send to first partial transcription",
            "asr_validated": "Latency from audio chunk send to validated transcription",
            "translation": "Latency from audio chunk send to text translation",
            "tts_audio": "Latency from audio chunk send to TTS audio output"
        }
    }