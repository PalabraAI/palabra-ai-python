import asyncio
import time
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import av
import librosa
import numpy as np
from aiofile import async_open
from av.error import (
    BlockingIOError as AvBlockingIOError,
)
from av.error import (
    EOFError as AvEOFError,
)
from av.error import (
    FFmpegError,
)
from av.filter import Graph as FilterGraph

from palabra_ai.util.logger import debug, error


async def write_to_disk(file_path: str | Path, body: bytes) -> int:
    try:
        async with async_open(file_path, "wb") as f:
            return await f.write(body)
    except asyncio.CancelledError:
        debug(f"write_to_disk cancelled for {file_path}")
        raise


async def read_from_disk(file_path: str | Path) -> bytes:
    try:
        async with async_open(file_path, "rb") as afp:
            return await afp.read()
    except asyncio.CancelledError:
        debug(f"read_from_disk cancelled for {file_path}")
        raise


def resample_pcm(
    audio_data: bytes,
    input_sample_rate: int,
    output_sample_rate: int,
    input_channels: int,
    output_channels: int,
) -> bytes:
    incoming_audio_data = np.frombuffer(audio_data, dtype=np.int16)
    incoming_audio_data = incoming_audio_data.astype(np.float32) / (
        np.iinfo(np.int16).max or 1
    )

    if input_channels == 2 and output_channels == 1:
        if incoming_audio_data.ndim == 1:
            # if audio is 1D, it means the channels are interleaved
            if incoming_audio_data.size % 2 != 0:
                incoming_audio_data = incoming_audio_data[:-1]
            incoming_audio_data = incoming_audio_data.reshape(-1, 2).mean(axis=1)
        else:
            # channels are already separated
            incoming_audio_data = np.mean(
                incoming_audio_data, axis=tuple(range(incoming_audio_data.ndim - 1))
            )

    if input_sample_rate != output_sample_rate:
        incoming_audio_data = librosa.resample(
            incoming_audio_data, orig_sr=input_sample_rate, target_sr=output_sample_rate
        )

    return (incoming_audio_data * np.iinfo(np.int16).max).astype(np.int16).tobytes()


def convert_any_to_pcm16(
    audio_data: bytes,
    sample_rate: int,
    layout: str = "mono",
    normalize: bool = True,
) -> bytes:
    before_conversion = time.perf_counter()
    try:
        input_buffer = BytesIO(audio_data)
        input_container, _ = open_audio_container(input_buffer)

        output_buffer = BytesIO()
        output_container, audio_stream = create_pcm_output_container(
            output_buffer, sample_rate, layout
        )

        filter_graph_buffer, filter_graph_sink = None, None
        if normalize:
            _, filter_graph_buffer, filter_graph_sink = (
                create_normalization_filter_graph(
                    audio_stream.format.name,
                    audio_stream.rate,
                    audio_stream.layout,
                    audio_stream.time_base,
                )
            )

        resampler = av.AudioResampler(
            format=av.AudioFormat("s16"), layout=layout, rate=sample_rate
        )

        dts = process_audio_frames(
            input_container,
            audio_stream,
            resampler,
            filter_graph_buffer,
            filter_graph_sink,
        )

        flush_filters_and_encoder(
            filter_graph_buffer, filter_graph_sink, audio_stream, dts
        )

        output_container.close()
        input_container.close()

        output_buffer.seek(0)
        return output_buffer.read()
    except FFmpegError as e:
        error("Failed to convert audio using libav with: %s", str(e))
        raise
    finally:
        debug(f"Conversion took {time.perf_counter() - before_conversion:.3f} seconds")


def pull_until_blocked(graph):
    frames = []
    while True:
        try:
            frames.append(graph.pull())
        except AvBlockingIOError:
            return frames
        except FFmpegError:
            raise


def open_audio_container(path_or_buffer, timeout=None):
    """Open audio container and return container and first audio stream."""
    container = av.open(path_or_buffer, timeout=timeout, metadata_errors="ignore")
    audio_streams = [s for s in container.streams if s.type == "audio"]
    if not audio_streams:
        container.close()
        raise ValueError("No audio streams found")
    return container, audio_streams[0]


def get_audio_stream_info(audio_stream):
    """Get audio stream information (duration, codec, sample_rate, channels)."""
    duration = (
        float(audio_stream.duration * audio_stream.time_base)
        if audio_stream.duration
        else 0
    )
    return {
        "duration": duration,
        "codec": audio_stream.codec.name,
        "sample_rate": audio_stream.sample_rate,
        "channels": audio_stream.channels,
    }


def create_normalization_filter_graph(format_name, sample_rate, layout, time_base):
    """Create filter graph with loudnorm and speechnorm filters."""
    filter_graph = FilterGraph()
    filter_buffer = filter_graph.add_abuffer(
        format=format_name,
        sample_rate=sample_rate,
        layout=layout,
        time_base=time_base,
    )
    loudnorm_filter = filter_graph.add("loudnorm", "I=-23:LRA=5:TP=-1")
    speechnorm_filter = filter_graph.add("speechnorm", "e=50:r=0.0005:l=1")
    filter_sink = filter_graph.add("abuffersink")

    filter_buffer.link_to(loudnorm_filter)
    loudnorm_filter.link_to(speechnorm_filter)
    speechnorm_filter.link_to(filter_sink)
    filter_graph.configure()

    return filter_graph, filter_buffer, filter_sink


def create_pcm_output_container(buffer, sample_rate, layout="mono"):
    """Create PCM output container and stream."""
    output_container = av.open(buffer, mode="w", format="s16le")
    output_stream = output_container.add_stream("pcm_s16le", rate=sample_rate)
    output_stream.layout = layout
    output_stream.time_base = Fraction(1, sample_rate)
    return output_container, output_stream


def process_audio_frames(
    input_container,
    output_stream,
    resampler,
    filter_buffer=None,
    filter_sink=None,
    progress_callback=None,
):
    """Process all audio frames with optional filters and progress callback."""
    dts = 0

    for frame in input_container.decode(audio=0):
        if frame is not None:
            for resampled_frame in resampler.resample(frame):
                if filter_buffer and filter_sink:
                    filter_buffer.push(resampled_frame)
                    processed_frames = pull_until_blocked(filter_sink)
                else:
                    processed_frames = [resampled_frame]

                for processed_frame in processed_frames:
                    processed_frame.pts = dts
                    dts += processed_frame.samples

                    for packet in output_stream.encode(processed_frame):
                        output_stream.container.mux(packet)

                    if progress_callback:
                        progress_callback(processed_frame.samples)

    return dts


def flush_filters_and_encoder(filter_buffer, filter_sink, output_stream, start_dts=0):
    """Flush filters and encoder, return final dts."""
    dts = start_dts

    # Flush filters
    if filter_buffer and filter_sink:
        try:
            filter_buffer.push(None)
            while True:
                try:
                    filtered_frame = filter_sink.pull()
                    filtered_frame.pts = dts
                    dts += filtered_frame.samples

                    for packet in output_stream.encode(filtered_frame):
                        output_stream.container.mux(packet)

                except (AvBlockingIOError, AvEOFError):
                    break
        except AvEOFError:
            pass

    # Flush encoder
    try:
        for packet in output_stream.encode(None):
            output_stream.container.mux(packet)
    except AvEOFError:
        pass

    return dts


def create_audio_resampler(target_rate, audio_format="s16", layout="mono"):
    """Create audio resampler - used in multiple places."""
    return av.AudioResampler(
        format=av.AudioFormat(audio_format), layout=layout, rate=target_rate
    )
