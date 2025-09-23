import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import pytest
import numpy as np
from palabra_ai.internal.audio import (
    pull_until_blocked
)




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
