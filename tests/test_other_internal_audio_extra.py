import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import pytest
import numpy as np
from palabra_ai.internal.audio import (
    resample_pcm, 
    pull_until_blocked
)

@pytest.mark.skip(reason="Focus on high-coverage functions first")
class TestWriteToDisk:
    """Test write_to_disk function"""
    pass

@pytest.mark.skip(reason="Focus on high-coverage functions first")  
class TestReadFromDisk:
    """Test read_from_disk function"""
    pass

class TestResamplePCM:
    """Test resample_pcm function"""
    
    def test_resample_same_rate_same_channels(self):
        """Test when input and output are the same"""
        # Create test audio data (16-bit PCM)
        audio_data = np.array([1000, 2000, 3000, 4000], dtype=np.int16).tobytes()
        
        result = resample_pcm(audio_data, 44100, 44100, 1, 1)
        
        # Should return the same data (approximately, due to float conversion)
        assert len(result) == len(audio_data)
    
    def test_resample_stereo_to_mono(self):
        """Test converting stereo to mono"""
        # Create stereo test data (interleaved L/R)
        audio_data = np.array([1000, 2000, 3000, 4000], dtype=np.int16).tobytes()
        
        result = resample_pcm(audio_data, 44100, 44100, 2, 1)
        
        # Should be half the length (mono)
        assert len(result) == len(audio_data) // 2
    
    def test_resample_stereo_to_mono_odd_samples(self):
        """Test stereo to mono with odd number of samples"""
        # Create stereo data with odd number of samples
        audio_data = np.array([1000, 2000, 3000], dtype=np.int16).tobytes()
        
        result = resample_pcm(audio_data, 44100, 44100, 2, 1)
        
        # Should handle odd samples correctly
        assert len(result) == 2  # One sample after dropping last odd sample
    
    @patch('palabra_ai.internal.audio.librosa.resample')
    def test_resample_different_sample_rate(self, mock_resample):
        """Test resampling to different sample rate"""
        audio_data = np.array([1000, 2000], dtype=np.int16).tobytes()
        mock_resample.return_value = np.array([1500, 2500], dtype=np.float32)
        
        result = resample_pcm(audio_data, 44100, 22050, 1, 1)
        
        mock_resample.assert_called_once()
        assert len(result) == 4  # 2 samples * 2 bytes per sample
    
    def test_resample_2d_stereo_to_mono(self):
        """Test 2D stereo array to mono conversion"""
        # Create 2D audio data (channels already separated)
        audio_array = np.array([[1000, 2000], [3000, 4000]], dtype=np.float32)
        audio_data = audio_array.astype(np.int16).tobytes()
        
        with patch('numpy.frombuffer') as mock_frombuffer:
            mock_frombuffer.return_value = audio_array
            
            result = resample_pcm(audio_data, 44100, 44100, 2, 1)
            
            assert len(result) > 0

@pytest.mark.skip(reason="Complex av library mocking, focus on simpler functions first")
class TestConvertAnyToPCM16:
    """Test convert_any_to_pcm16 function"""
    pass

class TestPullUntilBlocked:
    """Test pull_until_blocked function"""
    
    def test_pull_success(self):
        """Test successful frame pulling"""
        mock_graph = MagicMock()
        mock_frame1 = MagicMock()
        mock_frame2 = MagicMock()
        
        # Mock to return two frames then block
        from av.error import BlockingIOError as AvBlockingIOError
        mock_graph.pull.side_effect = [mock_frame1, mock_frame2, AvBlockingIOError("test", "test", "test")]
        
        result = pull_until_blocked(mock_graph)
        
        assert len(result) == 2
        assert result[0] == mock_frame1
        assert result[1] == mock_frame2
    
    def test_pull_ffmpeg_error(self):
        """Test FFmpeg error propagation"""
        from av.error import FFmpegError
        
        mock_graph = MagicMock()
        mock_graph.pull.side_effect = FFmpegError("Test error", "test")
        
        with pytest.raises(FFmpegError):
            pull_until_blocked(mock_graph)
    
    def test_pull_immediate_block(self):
        """Test immediate blocking"""
        from av.error import BlockingIOError as AvBlockingIOError
        
        mock_graph = MagicMock()
        mock_graph.pull.side_effect = AvBlockingIOError("test", "test", "test")
        
        result = pull_until_blocked(mock_graph)
        
        assert result == []