"""Tests for palabra_ai.benchmark.analyzer module"""

import pytest
from dataclasses import dataclass
from palabra_ai.benchmark.analyzer import AudioChunk, LatencyMeasurement, calculate_percentiles


def test_audio_chunk_creation():
    """Test AudioChunk dataclass creation"""
    chunk = AudioChunk(
        index=0,
        timestamp=1.0,
        rms_db=-20.0,
        chunk_duration_ms=100.0,
        direction="in"
    )
    assert chunk.index == 0
    assert chunk.timestamp == 1.0
    assert chunk.rms_db == -20.0
    assert chunk.chunk_duration_ms == 100.0
    assert chunk.direction == "in"


def test_latency_measurement_creation():
    """Test LatencyMeasurement dataclass creation"""
    measurement = LatencyMeasurement(
        event_type="partial_transcription",
        transcription_id="test_123",
        latency_sec=0.5,
        chunk_index=0,
        segment_start_sec=1.5
    )
    assert measurement.event_type == "partial_transcription"
    assert measurement.transcription_id == "test_123"
    assert measurement.latency_sec == 0.5
    assert measurement.chunk_index == 0
    assert measurement.segment_start_sec == 1.5


def test_calculate_percentiles_empty():
    """Test calculate_percentiles with empty data"""
    result = calculate_percentiles([])
    assert result == {}


def test_calculate_percentiles_single_value():
    """Test calculate_percentiles with single value"""
    result = calculate_percentiles([5.0])
    assert result["min"] == 5.0
    assert result["max"] == 5.0
    assert result["mean"] == 5.0
    assert result["count"] == 1
    assert result["stdev"] == 0  # Single value has no standard deviation


def test_calculate_percentiles_multiple_values():
    """Test calculate_percentiles with multiple values"""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    result = calculate_percentiles(data)

    assert result["min"] == 1.0
    assert result["max"] == 10.0
    assert result["count"] == 10
    assert result["mean"] == 5.5
    # The implementation uses int(n * percentile) for index calculation
    assert result["p50"] == 6.0  # index 5 -> value 6.0
    assert result["p25"] == 3.0  # index 2 -> value 3.0
    assert result["p75"] == 8.0  # index 7 -> value 8.0
    assert result["p90"] == 10.0  # index 9 -> value 10.0
    assert result["p95"] == 10.0  # index 9 -> value 10.0
    assert result["p99"] == 10.0  # index 9 -> value 10.0
    assert result["stdev"] > 2.8 and result["stdev"] < 3.1  # Approximately 2.87
