"""
Latency analyzer for Palabra AI messages
Provides data analysis for benchmark results
"""

import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class AudioChunk:
    """Metadata for audio chunk"""
    index: int
    timestamp: float  # perf_counter timestamp when sent/received
    rms_db: float  # RMS in decibels
    chunk_duration_ms: float
    direction: str  # 'in' or 'out'
    
    @property
    def audio_time_sec(self) -> float:
        """Audio time position in seconds from start"""
        return self.index * (self.chunk_duration_ms / 1000.0)


@dataclass 
class LatencyMeasurement:
    """Single latency measurement"""
    event_type: str  # partial_transcription, validated_transcription, etc
    transcription_id: str
    latency_sec: float
    chunk_index: int
    segment_start_sec: float
    text: str = ""  # Text content from the transcription


def find_nearest_sound_chunk(chunks: List[AudioChunk], target_index: int, 
                            rms_threshold_db: float = -40.0,
                            max_lookahead_ms: int = 10000) -> Optional[AudioChunk]:
    """
    Find nearest chunk with sound (RMS > threshold) starting from target_index.
    Searches forward up to max_lookahead_ms.
    """
    if not chunks or target_index < 0:
        return None
    
    # Ensure target_index is within bounds
    target_index = min(target_index, len(chunks) - 1)
    
    # Get chunk duration from first chunk
    chunk_duration_ms = chunks[0].chunk_duration_ms if chunks else 100
    max_steps = int(max_lookahead_ms / chunk_duration_ms)
    
    # Search forward from target_index
    for step in range(max_steps + 1):
        idx = target_index + step
        if idx >= len(chunks):
            break
            
        chunk = chunks[idx]
        if chunk.rms_db > rms_threshold_db:
            return chunk
    
    # If no sound found, return original target
    if target_index < len(chunks):
        return chunks[target_index]
    
    return None


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
    Analyze message trace for latency metrics with proper time synchronization.
    
    The key insight: segment.start values are relative to the audio file start,
    but we need to sync this with our chunk send timestamps to calculate accurate latencies.
    
    Synchronization approach:
    1. Use the first partial transcription as a synchronization point
    2. Map its segment.start to the corresponding chunk to establish the time relationship
    3. Use this mapping for all subsequent latency calculations
    
    Calculates one-way latency from sending audio chunk to receiving events:
    - partial_transcription
    - validated_transcription  
    - translated_transcription
    - TTS audio output (first non-silent OUT audio chunk)
    
    Returns comprehensive latency analysis.
    """
    
    # Constants
    RMS_THRESHOLD_DB = -40.0  # Threshold in dB for detecting sound
    MAX_LOOKAHEAD_MS = 10000  # Max 10 seconds lookahead for sound
    
    # Collect IN audio chunks
    in_audio_chunks = []
    in_chunks_by_index = {}
    
    # Collect OUT audio chunks (for TTS)
    out_audio_chunks = []
    
    # Time synchronization state
    sync_established = False
    sync_reference_chunk_ts = None  # Timestamp of the reference chunk
    sync_reference_segment_start = None  # segment.start of the reference transcription
    sync_reference_chunk_idx = None  # Index of the reference chunk
    
    # Extract chunk_duration_ms from first audio message
    chunk_duration_ms = None
    
    for msg in messages:
        if msg.get("kind") == "audio":
            # Extract chunk duration from first audio message
            if chunk_duration_ms is None:
                chunk_duration_ms = msg.get("chunk_duration_ms")
                if chunk_duration_ms:
                    chunk_duration_ms = float(chunk_duration_ms)
            
            direction = msg.get("dir")
            chunk_num = msg.get("num")
            timestamp = msg.get("ts")
            rms_db = msg.get("rms")
            
            if chunk_num is not None and timestamp is not None:
                chunk = AudioChunk(
                    index=chunk_num,
                    timestamp=timestamp,
                    rms_db=rms_db if rms_db is not None else -100.0,
                    chunk_duration_ms=chunk_duration_ms or 100.0,
                    direction=direction
                )
                
                if direction == "in":
                    in_audio_chunks.append(chunk)
                    in_chunks_by_index[chunk_num] = chunk
                elif direction == "out":
                    out_audio_chunks.append(chunk)
    
    # Sort chunks by index for IN, by timestamp for OUT
    in_audio_chunks.sort(key=lambda c: c.index)
    out_audio_chunks.sort(key=lambda c: c.timestamp)  # Sort by timestamp to find first in time
    
    if not in_audio_chunks:
        raise ValueError("No input audio chunks found in messages")
    
    # Use detected chunk_duration_ms or default
    if chunk_duration_ms is None:
        chunk_duration_ms = 100.0  # Default fallback
    
    # Process transcription events with deduplication
    # For each message type, track which base transcription_ids we've seen
    processed_ids = {
        "partial_transcription": set(),
        "validated_transcription": set(),
        "translated_transcription": set()
    }
    latency_measurements = []
    
    # Track translation events for TTS correlation
    translation_events = []  # List of (timestamp, transcription_id, chunk_index)
    
    
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
            text = transcription.get("text", "")  # Extract text content
            
            # Skip if has _part_ suffix (we only want base IDs)
            if not trans_id or '_part_' in trans_id:
                continue
            
            # Deduplication - only process first occurrence of each transcription_id per message type
            if trans_id in processed_ids[msg_type]:
                continue
            processed_ids[msg_type].add(trans_id)
            
            if segments and len(segments) > 0:
                # Get first segment start time  
                segment_start = segments[0].get("start", 0)
                
                # Establish synchronization on first partial transcription
                if msg_type == "partial_transcription" and not sync_established:
                    # This is our synchronization point
                    target_idx = round(segment_start * 1000 / chunk_duration_ms)  # round, not floor!
                    
                    # Find the chunk with sound that corresponds to this segment
                    sync_chunk = find_nearest_sound_chunk(
                        in_audio_chunks,
                        target_idx,
                        RMS_THRESHOLD_DB,
                        MAX_LOOKAHEAD_MS
                    )
                    
                    if sync_chunk:
                        sync_established = True
                        sync_reference_chunk_ts = sync_chunk.timestamp
                        sync_reference_segment_start = segment_start
                        sync_reference_chunk_idx = sync_chunk.index
                
                # Calculate latency using synchronized time reference
                if sync_established:
                    # Calculate target chunk index (using round as per spec)
                    target_index = round(segment_start * 1000 / chunk_duration_ms)  # round, not floor!
                    
                    # Find nearest chunk with sound
                    nearest_chunk = find_nearest_sound_chunk(
                        in_audio_chunks, 
                        target_index,
                        RMS_THRESHOLD_DB,
                        MAX_LOOKAHEAD_MS
                    )
                    
                    if nearest_chunk:
                        # Calculate latency
                        latency = msg_ts - nearest_chunk.timestamp
                        
                        # Skip only negative latencies (impossible)
                        if latency >= 0:
                            measurement = LatencyMeasurement(
                                event_type=msg_type,
                                transcription_id=trans_id,
                                latency_sec=latency,
                                chunk_index=nearest_chunk.index,
                                segment_start_sec=segment_start,
                                text=text  # Save text content
                            )
                            latency_measurements.append(measurement)
                        
                        # Track translation events for TTS correlation
                        if msg_type == "translated_transcription":
                            translation_events.append((msg_ts, trans_id, nearest_chunk.index))
        
    
    # Process TTS audio using transcription_id from OUT audio messages
    tts_measurements = []
    
    # Create mapping of OUT audio chunks by transcription_id
    out_audio_by_transcription_id = {}
    for msg in messages:
        if (msg.get("msg", {}).get("message_type") == "__$bench_audio_frame" 
            and msg.get("dir") == "out"):
            data = msg.get("msg", {}).get("data", {})
            transcription_id = data.get("transcription_id")
            if transcription_id:
                # Store first OUT audio chunk for each transcription_id
                if transcription_id not in out_audio_by_transcription_id:
                    out_audio_by_transcription_id[transcription_id] = msg.get("ts")
    
    for trans_ts, trans_id, in_chunk_idx in translation_events:
        # Use direct transcription_id matching from OUT audio
        if trans_id in out_audio_by_transcription_id:
            out_audio_ts = out_audio_by_transcription_id[trans_id]
            if in_chunk_idx in in_chunks_by_index:
                in_chunk = in_chunks_by_index[in_chunk_idx]
                tts_latency = out_audio_ts - in_chunk.timestamp
                
                measurement = LatencyMeasurement(
                    event_type="tts_audio",
                    transcription_id=trans_id,
                    latency_sec=tts_latency,
                    chunk_index=in_chunk_idx,
                    segment_start_sec=in_chunk.audio_time_sec
                )
                tts_measurements.append(measurement)
    
    # Combine all measurements
    all_measurements = latency_measurements + tts_measurements
    
    # Group measurements by event type
    measurements_by_type = {
        "partial_transcription": [],
        "validated_transcription": [],
        "translated_transcription": [],
        "tts_audio": []
    }
    
    # Also create detailed chunk information with texts
    chunk_details = {}
    
    for m in all_measurements:
        if m.event_type in measurements_by_type:
            measurements_by_type[m.event_type].append(m.latency_sec)
            
            # Store detailed info for each chunk
            if m.chunk_index not in chunk_details:
                chunk_details[m.chunk_index] = {
                    "chunk_index": m.chunk_index,
                    "audio_time": m.segment_start_sec,
                    "partial_text": "",
                    "validated_text": "",
                    "translated_text": "",
                    "partial_latency": None,
                    "validated_latency": None,
                    "translated_latency": None,
                    "tts_latency": None
                }
            
            # Update with specific event data
            if m.event_type == "partial_transcription":
                chunk_details[m.chunk_index]["partial_text"] = m.text
                chunk_details[m.chunk_index]["partial_latency"] = m.latency_sec
            elif m.event_type == "validated_transcription":
                chunk_details[m.chunk_index]["validated_text"] = m.text
                chunk_details[m.chunk_index]["validated_latency"] = m.latency_sec
            elif m.event_type == "translated_transcription":
                chunk_details[m.chunk_index]["translated_text"] = m.text
                chunk_details[m.chunk_index]["translated_latency"] = m.latency_sec
            elif m.event_type == "tts_audio":
                chunk_details[m.chunk_index]["tts_latency"] = m.latency_sec
    
    # Calculate statistics for each type
    statistics_data = {}
    for event_type, latencies in measurements_by_type.items():
        if latencies:
            statistics_data[event_type] = calculate_percentiles(latencies)
    
    # Calculate summary statistics
    total_in_chunks = len(in_audio_chunks)
    chunks_with_sound = sum(1 for c in in_audio_chunks if c.rms_db > RMS_THRESHOLD_DB)
    silent_chunks = total_in_chunks - chunks_with_sound
    
    # Audio duration
    if in_audio_chunks:
        total_duration = max(c.audio_time_sec for c in in_audio_chunks) + chunk_duration_ms/1000
    else:
        total_duration = 0
    
    # Calculate time progression (for charts)
    time_progression = {}
    window_size_sec = 5.0  # 5-second windows
    
    # Group measurements by time window for validated transcription
    if measurements_by_type["validated_transcription"]:
        validated_with_time = [(m.segment_start_sec, m.latency_sec) for m in all_measurements if m.event_type == "validated_transcription"]
        validated_with_time.sort(key=lambda x: x[0])  # Sort by audio time
        
        for audio_time, latency in validated_with_time:
            window_id = int(audio_time / window_size_sec)
            if window_id not in time_progression:
                time_progression[window_id] = {
                    "start_time": window_id * window_size_sec,
                    "end_time": (window_id + 1) * window_size_sec,
                    "latencies": []
                }
            time_progression[window_id]["latencies"].append(latency)
        
        # Calculate stats for each window
        for window_id, window_data in time_progression.items():
            latencies = window_data["latencies"]
            if latencies:
                window_data["mean"] = statistics.mean(latencies)
                window_data["median"] = statistics.median(latencies)
                window_data["min"] = min(latencies)
                window_data["max"] = max(latencies)
                window_data["count"] = len(latencies)
            else:
                window_data["mean"] = None
                window_data["median"] = None
                window_data["min"] = None
                window_data["max"] = None
                window_data["count"] = 0
            # Remove raw latencies to save space
            del window_data["latencies"]
    
    # Build results
    return {
        "summary": {
            "total_chunks": total_in_chunks,
            "chunks_with_sound": chunks_with_sound,
            "silent_chunks": silent_chunks,
            "silent_percentage": (silent_chunks / total_in_chunks * 100) if total_in_chunks > 0 else 0,
            "chunk_duration_ms": chunk_duration_ms,
            "rms_threshold_db": RMS_THRESHOLD_DB,
            "total_duration": total_duration,
            "chunks_with_partial": len(measurements_by_type["partial_transcription"]),
            "chunks_with_validated": len(measurements_by_type["validated_transcription"]),
            "chunks_with_translation": len(measurements_by_type["translated_transcription"]),
            "chunks_with_tts": len(measurements_by_type["tts_audio"]),
            "out_audio_chunks": len(out_audio_chunks),
            "sync_established": sync_established,
            "sync_reference": {
                "segment_start": sync_reference_segment_start,
                "chunk_index": sync_reference_chunk_idx,
                "chunk_ts": sync_reference_chunk_ts
            } if sync_established else None
        },
        "statistics": statistics_data,
        "measurements": {
            "partial_transcription": measurements_by_type["partial_transcription"],
            "validated_transcription": measurements_by_type["validated_transcription"],
            "translated_transcription": measurements_by_type["translated_transcription"],
            "tts_audio": measurements_by_type["tts_audio"]
        },
        "metric_descriptions": {
            "partial_transcription": "Latency from audio chunk send to first partial transcription",
            "validated_transcription": "Latency from audio chunk send to validated transcription",
            "translated_transcription": "Latency from audio chunk send to text translation",
            "tts_audio": "Latency from audio chunk send to first TTS audio with sound"
        },
        "time_progression": time_progression,
        "chunk_details": chunk_details,  # Add detailed chunk information with texts
        "chunks": {
            "in_audio": [
                {
                    "index": c.index,
                    "audio_time": c.audio_time_sec,
                    "rms_db": c.rms_db,
                    "has_sound": c.rms_db > RMS_THRESHOLD_DB
                }
                for c in in_audio_chunks[:10]  # First 10 for debugging
            ],
            "out_audio": [
                {
                    "index": c.index,
                    "audio_time": c.audio_time_sec,
                    "rms_db": c.rms_db,
                    "has_sound": c.rms_db > RMS_THRESHOLD_DB
                }
                for c in out_audio_chunks[:10]  # First 10 for debugging
            ]
        }
    }