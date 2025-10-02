import pytest
import numpy as np
import base64
import io
from unittest.mock import MagicMock, patch
from palabra_ai.audio import AudioFrame, AudioBuffer
from palabra_ai.util.orjson import to_json, from_json

def test_audio_frame_creation_with_numpy():
    """Test AudioFrame creation with numpy array"""
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int16)
    frame = AudioFrame(data, sample_rate=48000, num_channels=2, samples_per_channel=3)

    assert frame.sample_rate == 48000
    assert frame.num_channels == 2
    assert frame.samples_per_channel == 3
    assert np.array_equal(frame.data, data)

def test_audio_frame_creation_with_bytes():
    """Test AudioFrame creation with bytes"""
    # Create some int16 data as bytes
    data_array = np.array([100, 200, 300, 400], dtype=np.int16)
    data_bytes = data_array.tobytes()

    frame = AudioFrame(data_bytes, sample_rate=24000, num_channels=1, samples_per_channel=4)

    assert frame.sample_rate == 24000
    assert frame.num_channels == 1
    assert frame.samples_per_channel == 4
    assert np.array_equal(frame.data, data_array)

def test_audio_frame_samples_per_channel_auto():
    """Test AudioFrame auto-calculates input_samples_per_channel when None"""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int16)

    # With 2 channels, should get 4 samples per channel
    frame = AudioFrame(data, sample_rate=48000, num_channels=2, samples_per_channel=None)
    assert frame.samples_per_channel == 4

    # With 1 channel, should get 8 samples per channel
    frame2 = AudioFrame(data, sample_rate=48000, num_channels=1, samples_per_channel=None)
    assert frame2.samples_per_channel == 8

def test_audio_frame_create_classmethod():
    """Test AudioFrame.create class method"""
    frame = AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=160)

    assert frame.sample_rate == 16000
    assert frame.num_channels == 1
    assert frame.samples_per_channel == 160
    # create() returns bytearray, which is NOT converted by __init__ (only bytes are)
    assert isinstance(frame.data, bytearray)
    assert len(frame.data) == 320  # 160 samples * 2 bytes per sample
    # Check all bytes are zero
    assert all(b == 0 for b in frame.data)
    """Test AudioFrame to_bytes method"""
    data = np.array([100, -200, 300, -400], dtype=np.int16)
    frame = AudioFrame(data, sample_rate=16000, num_channels=1, samples_per_channel=4)

    # AudioFrame doesn't have to_bytes method, it has to_ws method which returns JSON bytes
    # Let's test what we actually have - direct byte conversion
    byte_data = frame.data.tobytes()
    assert isinstance(byte_data, bytes)

    # Convert back and check
    recovered = np.frombuffer(byte_data, dtype=np.int16)
    assert np.array_equal(recovered, data)
    """Test AudioFrame __repr__ method"""
    data = np.array([1, 2, 3, 4], dtype=np.int16)
    frame = AudioFrame(data=data, sample_rate=16000, num_channels=1, samples_per_channel=4)

    repr_str = repr(frame)
    assert "🗣️<AF(" in repr_str
    assert "s=4" in repr_str
    assert "sr=16000" in repr_str
    assert "ch=1" in repr_str

def test_audio_frame_bool_true():
    """Test AudioFrame __bool__ returns True for non-empty data"""
    data = np.array([1, 2, 3, 4], dtype=np.int16)
    frame = AudioFrame(data=data, sample_rate=16000, num_channels=1, samples_per_channel=4)

    assert bool(frame) is True

def test_audio_frame_bool_false_none():
    """Test AudioFrame __bool__ returns False for None data"""
    frame = AudioFrame(data=None, sample_rate=16000, num_channels=1, samples_per_channel=0)
    frame.data = None  # Force None

    assert bool(frame) is False

def test_audio_frame_bool_false_empty():
    """Test AudioFrame __bool__ returns False for empty data"""
    data = np.array([], dtype=np.int16)
    frame = AudioFrame(data=data, sample_rate=16000, num_channels=1, samples_per_channel=0)

    assert bool(frame) is False

def test_audio_frame_bool_bytearray():
    """Test AudioFrame __bool__ with bytearray data"""
    frame = AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=160)
    assert bool(frame) is True  # bytearray has length > 0

def test_audio_frame_from_rtc():
    """Test creating AudioFrame from RtcAudioFrame"""
    mock_rtc_frame = MagicMock()
    mock_rtc_frame.data = np.array([1, 2, 3, 4], dtype=np.int16)
    mock_rtc_frame.sample_rate = 48000
    mock_rtc_frame.num_channels = 2
    mock_rtc_frame.samples_per_channel = 2

    frame = AudioFrame.from_rtc(mock_rtc_frame)

    assert frame.sample_rate == 48000
    assert frame.num_channels == 2
    assert frame.samples_per_channel == 2
    assert np.array_equal(frame.data, mock_rtc_frame.data)
    assert isinstance(frame.perf_ts, float)  # Auto-generated perf_ts

def test_audio_frame_from_rtc_with_perf_ts():
    """Test creating AudioFrame from RtcAudioFrame with explicit perf_ts"""
    mock_rtc_frame = MagicMock()
    mock_rtc_frame.data = np.array([1, 2, 3, 4], dtype=np.int16)
    mock_rtc_frame.sample_rate = 48000
    mock_rtc_frame.num_channels = 2
    mock_rtc_frame.samples_per_channel = 2

    custom_perf_ts = 9876.543
    frame = AudioFrame.from_rtc(mock_rtc_frame, perf_ts=custom_perf_ts)

    assert frame.perf_ts == custom_perf_ts

def test_audio_frame_to_rtc():
    """Test converting AudioFrame to RtcAudioFrame"""
    data = np.array([1, 2, 3, 4], dtype=np.int16)
    frame = AudioFrame(data=data, sample_rate=48000, num_channels=2, samples_per_channel=2)

    with patch('palabra_ai.audio.RtcAudioFrame') as mock_rtc_class:
        rtc_frame = frame.to_rtc()

        mock_rtc_class.assert_called_once_with(
            data=data,
            sample_rate=48000,
            num_channels=2,
            samples_per_channel=2
        )

def test_audio_frame_from_ws_valid():
    """Test creating AudioFrame from valid WebSocket message"""
    audio_bytes = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    base64_data = base64.b64encode(audio_bytes).decode()

    ws_msg = to_json({
        "message_type": "output_audio_data",
        "data": {
            "data": base64_data,
            "language": "en",
            "transcription_id": "trans1",
            "last_chunk": False
        }
    })

    frame = AudioFrame.from_ws(ws_msg, 16000, 1)

    assert frame is not None
    assert frame.sample_rate == 16000
    assert frame.num_channels == 1
    assert frame.samples_per_channel == 4
    assert np.array_equal(frame.data, np.array([1, 2, 3, 4], dtype=np.int16))
    assert isinstance(frame.perf_ts, float)  # Auto-generated perf_ts

def test_audio_frame_from_ws_with_perf_ts():
    """Test creating AudioFrame from WebSocket message with explicit perf_ts"""
    audio_bytes = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    base64_data = base64.b64encode(audio_bytes).decode()

    ws_msg = to_json({
        "message_type": "output_audio_data",
        "data": {
            "data": base64_data,
            "language": "en",
            "transcription_id": "trans1",
            "last_chunk": False
        }
    })

    custom_perf_ts = 12345.678
    frame = AudioFrame.from_ws(ws_msg, 16000, 1, perf_ts=custom_perf_ts)

    assert frame is not None
    assert frame.perf_ts == custom_perf_ts

def test_audio_frame_from_ws_string_data():
    """Test creating AudioFrame from WebSocket message with string data"""
    audio_bytes = b"\x01\x00\x02\x00"
    base64_data = base64.b64encode(audio_bytes).decode()

    # data field is a JSON string
    ws_msg = to_json({
        "message_type": "output_audio_data",
        "data": to_json({
            "data": base64_data,
            "language": "en",
            "transcription_id": "trans1",
            "last_chunk": False
        }).decode()
    })

    frame = AudioFrame.from_ws(ws_msg, 16000, 1)

    assert frame is not None
    assert np.array_equal(frame.data, np.array([1, 2], dtype=np.int16))

def test_audio_frame_from_ws_invalid_type():
    """Test from_ws with invalid message type"""
    ws_msg = to_json({
        "message_type": "some_other_type",
        "data": {"data": "test"}
    })

    frame = AudioFrame.from_ws(ws_msg, 16000, 1)
    assert frame is None

def test_audio_frame_from_ws_not_json():
    """Test from_ws with non-JSON input"""
    frame = AudioFrame.from_ws(123, 16000, 1)
    assert frame is None

def test_audio_frame_from_ws_str_without_type():
    """Test from_ws with string not containing output_audio_data"""
    frame = AudioFrame.from_ws("some random string", 16000, 1)
    assert frame is None

def test_audio_frame_from_ws_bytes_without_type():
    """Test from_ws with bytes not containing output_audio_data"""
    frame = AudioFrame.from_ws(b"some random bytes", 16000, 1)
    assert frame is None

def test_audio_frame_from_ws_missing_data():
    """Test from_ws with missing data field"""
    ws_msg = to_json({
        "message_type": "output_audio_data"
    })

    frame = AudioFrame.from_ws(ws_msg, 16000, 1)
    assert frame is None

def test_audio_frame_from_ws_missing_inner_data():
    """Test from_ws with missing inner data field"""
    ws_msg = to_json({
        "message_type": "output_audio_data",
        "data": {}
    })

    frame = AudioFrame.from_ws(ws_msg, 16000, 1)
    assert frame is None

def test_audio_frame_from_ws_invalid_base64():
    """Test from_ws with invalid base64 data"""
    ws_msg = to_json({
        "message_type": "output_audio_data",
        "data": {"data": "invalid_base64!@#"}
    })

    with patch('palabra_ai.audio.error') as mock_error:
        frame = AudioFrame.from_ws(ws_msg, 16000, 1)
        assert frame is None
        mock_error.assert_called_once()

def test_audio_frame_to_ws():
    """Test converting AudioFrame to WebSocket message"""
    data = np.array([1, 2, 3, 4], dtype=np.int16)
    frame = AudioFrame(data=data, sample_rate=16000, num_channels=1, samples_per_channel=4)

    ws_msg = frame.to_ws()
    msg = from_json(ws_msg)

    assert msg["message_type"] == "input_audio_data"
    assert "data" in msg
    assert "data" in msg["data"]

    # Decode and verify audio data
    decoded_bytes = base64.b64decode(msg["data"]["data"])
    decoded_array = np.frombuffer(decoded_bytes, dtype=np.int16)
    assert np.array_equal(decoded_array, data)

def test_audio_buffer_init():
    """Test AudioBuffer initialization"""
    buffer = AudioBuffer(sample_rate=16000, num_channels=1)

    assert buffer.sample_rate == 16000
    assert buffer.num_channels == 1

def test_audio_buffer_to_wav_bytes_empty():
    """Test to_wav_bytes with empty buffer"""
    buffer = AudioBuffer(sample_rate=16000, num_channels=1)

    wav_bytes = buffer.to_wav_bytes()

    # Should return proper WAV file format with zero data
    assert wav_bytes.startswith(b'RIFF')
    assert b'WAVEfmt' in wav_bytes

def test_audio_buffer_to_wav_bytes_with_data():
    """Test to_wav_bytes with data in buffer"""
    buffer = AudioBuffer(sample_rate=16000, num_channels=1)

    # Write some data directly to the audio array
    import numpy as np
    buffer.audio_array[0:4] = np.array([1, 2, 3, 4], dtype=np.int16)

    wav_bytes = buffer.to_wav_bytes()

    assert wav_bytes != b""
    assert wav_bytes.startswith(b"RIFF")  # WAV file header
    assert b"WAVE" in wav_bytes

@pytest.mark.asyncio
async def test_audio_buffer_write():
    """Test writing AudioFrame to buffer"""
    buffer = AudioBuffer(sample_rate=16000, num_channels=1)

    data = np.array([1, 2, 3, 4], dtype=np.int16)
    frame = AudioFrame(data=data, sample_rate=16000, num_channels=1, samples_per_channel=4)

    await buffer.write(frame)

    # Check that data was written to the audio array
    assert buffer.audio_array[0] == 1
    assert buffer.audio_array[1] == 2
    assert buffer.audio_array[2] == 3
    assert buffer.audio_array[3] == 4

@pytest.mark.asyncio
async def test_audio_buffer_write_timing():
    """Test writing AudioFrame with timing info"""
    buffer = AudioBuffer(sample_rate=16000, num_channels=1)
    buffer.set_start_time(1000.0)  # Set start time

    data = np.array([1, 2, 3, 4], dtype=np.int16)
    frame = AudioFrame(data=data, sample_rate=16000, num_channels=1, samples_per_channel=4, perf_ts=1000.5)

    await buffer.write(frame)

    # Data should be written at calculated position based on timing
    # Check that some data was written (exact position depends on timing calculation)
    assert np.any(buffer.audio_array != 0)

@pytest.mark.asyncio
async def test_audio_buffer_write_positioning():
    """Test writing frame data with proper positioning"""
    buffer = AudioBuffer(sample_rate=16000, num_channels=1)

    data = np.array([1, 0, 0, 0], dtype=np.int16)
    frame = AudioFrame(data=data, sample_rate=16000, num_channels=1, samples_per_channel=4)

    await buffer.write(frame)

    # Check that data was written to the audio array
    assert buffer.audio_array[0] == 1
    assert buffer.audio_array[1] == 0
    assert buffer.audio_array[2] == 0
    assert buffer.audio_array[3] == 0

def test_audio_buffer_duration():
    """Test buffer duration calculations"""
    buffer = AudioBuffer(sample_rate=16000, num_channels=1, original_duration=30.0)

    # Should have created array for 30s + 60s buffer = 90s total
    expected_samples = int(90 * 16000)  # 90 seconds at 16kHz
    assert len(buffer.audio_array) == expected_samples


def test_audio_buffer_init_with_drop_empty():
    """Test AudioBuffer with drop_empty_frames enabled"""
    buffer = AudioBuffer(sample_rate=16000, num_channels=1, drop_empty_frames=True)

    assert buffer.sample_rate == 16000
    assert buffer.num_channels == 1
    assert buffer.drop_empty_frames is True


@pytest.mark.asyncio
async def test_audio_buffer_write_drop_empty_frames():
    """Test writing empty frame with drop_empty_frames=True"""
    buffer = AudioBuffer(sample_rate=16000, num_channels=1, drop_empty_frames=True)

    # Create frame with all zeros (empty)
    data = np.zeros(4, dtype=np.int16)
    frame = AudioFrame(data=data, sample_rate=16000, num_channels=1, samples_per_channel=4)

    await buffer.write(frame)

    # Buffer should remain empty (no data written)
    assert np.all(buffer.audio_array == 0)


@pytest.mark.asyncio
async def test_audio_buffer_write_keep_empty_frames():
    """Test writing empty frame with drop_empty_frames=False (default)"""
    buffer = AudioBuffer(sample_rate=16000, num_channels=1, drop_empty_frames=False)

    # Create frame with all zeros (empty)
    data = np.zeros(4, dtype=np.int16)
    frame = AudioFrame(data=data, sample_rate=16000, num_channels=1, samples_per_channel=4)

    await buffer.write(frame)

    # Buffer should contain the zeros (data was written)
    assert buffer.audio_array[0] == 0
    assert buffer.audio_array[1] == 0
    assert buffer.audio_array[2] == 0
    assert buffer.audio_array[3] == 0
    assert buffer.last_audio_end_position == 4


@pytest.mark.asyncio
async def test_audio_buffer_write_mixed_frames_with_drop():
    """Test writing mixed empty and non-empty frames with drop_empty_frames=True"""
    buffer = AudioBuffer(sample_rate=16000, num_channels=1, drop_empty_frames=True)

    # Write non-empty frame first
    non_empty_data = np.array([1, 2, 3, 4], dtype=np.int16)
    non_empty_frame = AudioFrame(data=non_empty_data, sample_rate=16000, num_channels=1, samples_per_channel=4)
    await buffer.write(non_empty_frame)

    # Write empty frame (should be dropped)
    empty_data = np.zeros(4, dtype=np.int16)
    empty_frame = AudioFrame(data=empty_data, sample_rate=16000, num_channels=1, samples_per_channel=4)
    await buffer.write(empty_frame)

    # Write another non-empty frame
    non_empty_data2 = np.array([5, 6, 7, 8], dtype=np.int16)
    non_empty_frame2 = AudioFrame(data=non_empty_data2, sample_rate=16000, num_channels=1, samples_per_channel=4)
    await buffer.write(non_empty_frame2)

    # Check that only non-empty frames were written
    assert buffer.audio_array[0] == 1
    assert buffer.audio_array[1] == 2
    assert buffer.audio_array[2] == 3
    assert buffer.audio_array[3] == 4
    assert buffer.audio_array[4] == 5
    assert buffer.audio_array[5] == 6
    assert buffer.audio_array[6] == 7
    assert buffer.audio_array[7] == 8
    assert buffer.last_audio_end_position == 8
