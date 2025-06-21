import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
import numpy as np

from palabra_ai.internal.webrtc import (
    AudioTrackSettings, RoomClient, AudioPublication,
    _PALABRA_TRANSLATOR_PARTICIPANT_IDENTITY_PREFIX
)


class TestAudioTrackSettings:
    def test_chunk_size_calculation(self):
        settings = AudioTrackSettings(sample_rate=48000, chunk_duration_ms=20)
        assert settings.chunk_size == 960  # 48000 * 0.02

    def test_new_frame(self, mock_livekit):
        settings = AudioTrackSettings()
        frame = settings.new_frame()
        mock_livekit.AudioFrame.create.assert_called_once()


class TestAudioPublication:
    @pytest.mark.asyncio
    async def test_create(self, mock_livekit):
        room = MagicMock()
        room.local_participant.publish_track = AsyncMock(return_value=MagicMock(sid="test"))

        settings = AudioTrackSettings()
        pub = await AudioPublication.create(room, settings)

        assert pub.publication is not None
        room.local_participant.publish_track.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_cancelled(self, mock_livekit):
        room = MagicMock()
        room.local_participant.publish_track = AsyncMock(side_effect=asyncio.CancelledError)

        settings = AudioTrackSettings()

        with pytest.raises(asyncio.CancelledError):
            await AudioPublication.create(room, settings)

    @pytest.mark.asyncio
    async def test_close_variants(self, mock_livekit):
        room = MagicMock()
        room.local_participant.unpublish_track = AsyncMock()

        pub = AudioPublication(room, AudioTrackSettings())
        pub.publication = MagicMock()
        pub.publication.track.sid = "test_sid"

        # Normal close
        await pub.close()
        room.local_participant.unpublish_track.assert_called_once_with("test_sid")
        assert pub.publication is None

        # Close with cancellation
        room.local_participant.unpublish_track = AsyncMock(side_effect=asyncio.CancelledError)
        pub.publication = MagicMock()
        pub.publication.track.sid = "test_sid"
        await pub.close()

        # Close with error
        room.local_participant.unpublish_track = AsyncMock(side_effect=Exception("Test"))
        pub.publication = MagicMock()
        pub.publication.track.sid = "test_sid"
        await pub.close()

    @pytest.mark.asyncio
    async def test_push_variants(self, mock_livekit):
        room = MagicMock()
        settings = AudioTrackSettings(sample_rate=48000, chunk_duration_ms=10)
        settings.audio_source.capture_frame = AsyncMock()

        pub = AudioPublication(room, settings)

        mock_frame = MagicMock()
        mock_frame.data = np.zeros(480, dtype=np.int16)
        settings.new_frame = MagicMock(return_value=mock_frame)

        # Normal push
        await pub.push(b"\x01\x00\x02\x00" * 240)
        settings.audio_source.capture_frame.assert_called_once()

        # Push with padding
        settings.audio_source.capture_frame.reset_mock()
        await pub.push(b"\x01\x00")  # Only 1 sample
        settings.audio_source.capture_frame.assert_called_once()

        # Push cancelled
        settings.audio_source.capture_frame = AsyncMock(side_effect=asyncio.CancelledError)
        mock_frame.data = np.zeros(480, dtype=np.int16).data  # Real memoryview
        settings.new_frame = MagicMock(return_value=mock_frame)

        with pytest.raises(asyncio.CancelledError):
            await pub.push(b"\x00\x00")


class TestRoomClient:
    @pytest.mark.asyncio
    async def test_connect(self):
        with patch("palabra_ai.internal.webrtc.rtc.Room.connect", new_callable=AsyncMock) as mock_connect:
            room = RoomClient()
            await room.connect("wss://test", "token")

            mock_connect.assert_called_once()
            assert mock_connect.call_args[0][0] == "wss://test"
            assert mock_connect.call_args[0][1] == "token"

    @pytest.mark.asyncio
    async def test_connect_cancelled(self):
        with patch("palabra_ai.internal.webrtc.rtc.Room.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = asyncio.CancelledError

            room = RoomClient()
            with pytest.raises(asyncio.CancelledError):
                await room.connect("wss://test", "token")

    @pytest.mark.asyncio
    async def test_close(self):
        with patch("palabra_ai.internal.webrtc.rtc.Room.disconnect", new_callable=AsyncMock) as mock_disconnect:
            room = RoomClient()

            pub = MagicMock()
            pub.close = AsyncMock()
            room._publications = [pub]

            await room.close()

            pub.close.assert_called_once()
            mock_disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_cancelled(self):
        with patch("palabra_ai.internal.webrtc.rtc.Room.disconnect", new_callable=AsyncMock) as mock_disconnect:
            mock_disconnect.side_effect = asyncio.CancelledError

            room = RoomClient()
            await room.close()

    @pytest.mark.asyncio
    async def test_new_publication(self):
        room = RoomClient()

        participant = MagicMock()
        participant.identity = _PALABRA_TRANSLATOR_PARTICIPANT_IDENTITY_PREFIX + "test"
        room.wait_for_participant_join = AsyncMock(return_value=participant)

        with patch("palabra_ai.internal.webrtc.AudioPublication.create") as mock_create:
            mock_pub = MagicMock()
            mock_create.return_value = mock_pub

            settings = AudioTrackSettings()
            pub = await room.new_publication(settings)

            assert pub == mock_pub
            assert pub in room._publications

    @pytest.mark.asyncio
    async def test_new_publication_timeout(self):
        room = RoomClient()

        with patch("palabra_ai.internal.webrtc.AudioPublication.create") as mock_create:
            mock_create.return_value = MagicMock()
            room.wait_for_participant_join = AsyncMock(side_effect=TimeoutError)

            with pytest.raises(RuntimeError) as exc_info:
                await room.new_publication(AudioTrackSettings())

            assert "Palabra translator did not appear" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_wait_for_participant_join(self):
        room = RoomClient()

        participant = MagicMock()
        participant.identity = "test_participant"

        with patch.object(type(room), 'remote_participants', new_callable=lambda: {"p1": participant}):
            result = await room.wait_for_participant_join("test_")
            assert result == participant

    @pytest.mark.asyncio
    async def test_wait_for_participant_join_timeout(self):
        room = RoomClient()

        with patch.object(type(room), 'remote_participants', new_callable=lambda: {}):
            with pytest.raises(asyncio.TimeoutError):
                await room.wait_for_participant_join("test", timeout=0.1)

    @pytest.mark.asyncio
    async def test_wait_for_track_publish(self):
        room = RoomClient()

        track = MagicMock()
        track.name = "test_track"
        track.kind = 1  # KIND_AUDIO
        track.track = MagicMock()

        participant = MagicMock()
        participant.track_publications = {"t1": track}

        with patch("palabra_ai.internal.webrtc.rtc.TrackKind.KIND_AUDIO", 1):
            result = await room.wait_for_track_publish(participant, "test_")
            assert result == track

    def test_event_handlers(self):
        room = RoomClient()

        participant = MagicMock()
        participant.sid = "p1"
        participant.identity = "test"

        publication = MagicMock()
        publication.sid = "pub1"

        track = MagicMock()

        # Call all handlers - should not raise
        room.on_participant_connected(participant)
        room.on_participant_disconnected(participant)
        room.on_track_published(publication, participant)
        room.on_track_unpublished(publication, participant)
        room.on_local_track_published(publication, track)
        room.on_local_track_unpublished(publication)
        room.on_active_speakers_changed([participant])
        room.on_track_muted(publication, participant)
        room.on_track_unmuted(publication, participant)
        room.on_connection_quality_changed(participant, MagicMock())
        room.on_track_subscription_failed(participant, "sid", "error")
        room.on_connection_state_changed(MagicMock())
        room.on_connected()
        room.on_disconnected()
        room.on_reconnecting()
        room.on_reconnected()

    def test_on_track_subscribed(self, mock_livekit):
        room = RoomClient()

        # Video track
        video_track = MagicMock()
        video_track.kind = 0  # KIND_VIDEO

        with patch("palabra_ai.internal.webrtc.rtc.TrackKind.KIND_VIDEO", 0):
            with patch("palabra_ai.internal.webrtc.rtc.VideoStream") as mock_video:
                room.on_track_subscribed(video_track, MagicMock(), MagicMock())
                mock_video.assert_called_once()

        # Audio track
        audio_track = MagicMock()
        audio_track.kind = 1  # KIND_AUDIO

        with patch("palabra_ai.internal.webrtc.rtc.TrackKind.KIND_AUDIO", 1):
            with patch("palabra_ai.internal.webrtc.rtc.AudioStream") as mock_audio:
                room.on_track_subscribed(audio_track, MagicMock(), MagicMock())
                mock_audio.assert_called_once()
