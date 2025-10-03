from palabra_ai import constant


def test_audio_constants():
    """Test audio processing constants"""
    assert constant.SINGLE_TARGET_SUPPORTED_COUNT == 1
    assert constant.SAMPLE_RATE_DEFAULT == 48000
    assert constant.SAMPLE_RATE_HALF == 24000
    assert constant.CHANNELS_MONO == 1
    assert constant.OUTPUT_DEVICE_BLOCK_SIZE == 1024
    assert constant.AUDIO_CHUNK_SECONDS == 0.5
    assert constant.BYTES_PER_SAMPLE == 2


def test_timing_constants():
    """Test timing constants"""
    assert constant.BOOT_TIMEOUT == 30.0
    assert constant.SHUTDOWN_TIMEOUT == 5.0
    assert constant.LOGGER_SHUTDOWN_TIMEOUT == 20.0
    assert constant.SAFE_PUBLICATION_END_DELAY == 3.0
    assert constant.MONITOR_TIMEOUT == 0.1
    assert constant.DEFAULT_PROCESS_TIMEOUT == 300.0
    assert constant.TRACK_WAIT_TIMEOUT == 30.0
    assert constant.TRACK_CLOSE_TIMEOUT == 5.0
    assert constant.FINALIZE_WAIT_TIME == 5.0
    assert constant.SLEEP_INTERVAL_SHORT == 0.01
    assert constant.SLEEP_INTERVAL_DEFAULT == 0.1
    assert constant.SLEEP_INTERVAL_MEDIUM == 0.3
    assert constant.SLEEP_INTERVAL_LONG == 1.0
    assert constant.SLEEP_INTERVAL_BUFFER_CHECK == 5.0
    assert constant.QUEUE_READ_TIMEOUT == 1.0
    assert constant.QUEUE_WAIT_TIMEOUT == 0.5
    assert constant.WS_TIMEOUT == 5.0
    assert constant.DEBUG_TASK_CHECK_INTERVAL == 30.0



def test_buffer_constants():
    """Test buffer and queue constants"""
    assert constant.THREADPOOL_MAX_WORKERS == 32
    assert constant.DEVICE_ID_HASH_LENGTH == 8
    assert constant.MONITOR_MESSAGE_PREVIEW_LENGTH == 100
    assert constant.AUDIO_PROGRESS_LOG_INTERVAL == 100000


def test_eof_constants():
    """Test EOF and completion constants"""
    assert constant.EMPTY_MESSAGE_THRESHOLD == 10
    assert constant.EOF_DRAIN_TIMEOUT == 5.0
    assert constant.COMPLETION_WAIT_TIMEOUT == 2.0
    assert constant.STATS_LOG_INTERVAL == 5.0


def test_preprocessing_constants():
    """Test preprocessing constants"""
    assert constant.MIN_SENTENCE_CHARACTERS_DEFAULT == 80
    assert constant.MIN_SENTENCE_SECONDS_DEFAULT == 4
    assert constant.MIN_SPLIT_INTERVAL_DEFAULT == 0.6
    assert constant.CONTEXT_SIZE_DEFAULT == 30
    assert constant.SEGMENTS_AFTER_RESTART_DEFAULT == 15
    assert constant.STEP_SIZE_DEFAULT == 5
    assert constant.MAX_STEPS_WITHOUT_EOS_DEFAULT == 3
    assert constant.FORCE_END_OF_SEGMENT_DEFAULT == 0.5


def test_filler_constants():
    """Test filler phrases constants"""
    assert constant.MIN_TRANSCRIPTION_LEN_DEFAULT == 40
    assert constant.MIN_TRANSCRIPTION_TIME_DEFAULT == 3
    assert constant.PHRASE_CHANCE_DEFAULT == 0.5


def test_tts_constants():
    """Test TTS constants"""
    assert constant.F0_VARIANCE_FACTOR_DEFAULT == 1.2
    assert constant.ENERGY_VARIANCE_FACTOR_DEFAULT == 1.5
    assert constant.SPEECH_TEMPO_ADJUSTMENT_FACTOR_DEFAULT == 0.75


def test_queue_config_constants():
    """Test queue config constants"""
    assert constant.DESIRED_QUEUE_LEVEL_MS_DEFAULT == 5000
    assert constant.MAX_QUEUE_LEVEL_MS_DEFAULT == 24000


def test_transcription_constants():
    """Test transcription constants"""
    assert constant.MIN_ALIGNMENT_SCORE_DEFAULT == 0.2
    assert constant.MAX_ALIGNMENT_CER_DEFAULT == 0.8
    assert constant.SEGMENT_CONFIRMATION_SILENCE_THRESHOLD_DEFAULT == 0.7


def test_vad_constants():
    """Test VAD constants"""
    assert constant.VAD_THRESHOLD_DEFAULT == 0.5
    assert constant.VAD_LEFT_PADDING_DEFAULT == 1
    assert constant.VAD_RIGHT_PADDING_DEFAULT == 1
