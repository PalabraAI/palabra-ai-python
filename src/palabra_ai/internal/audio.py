import asyncio
import errno
import time
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import av
import librosa
import numpy as np
from aiofile import async_open

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
    sample_rate: int = 16000,
    layout: str = "mono",
    normalize: bool = True,
) -> bytes:
    before_conversion = time.perf_counter()
    try:
        input_buffer = BytesIO(audio_data)
        input_container = av.open(input_buffer, metadata_errors="ignore")

        output_buffer = BytesIO()
        output_container = av.open(output_buffer, mode="w", format="s16le")
        audio_stream = output_container.add_stream("pcm_s16le", rate=sample_rate)
        audio_stream.layout = layout
        audio_stream.time_base = Fraction(1, sample_rate)

        filter_graph_buffer, filter_graph_sink = None, None
        if normalize:
            # create filter graph for `loudnorm` and `speechnorm` filters
            filter_graph = av.filter.Graph()
            filter_graph_buffer = filter_graph.add_abuffer(
                format=audio_stream.format.name,
                sample_rate=audio_stream.rate,
                layout=audio_stream.layout,
                time_base=audio_stream.time_base,
            )
            loudnorm_filter_node = filter_graph.add("loudnorm", "I=-23:LRA=5:TP=-1")
            speechnorm_filter_node = filter_graph.add("speechnorm", "e=50:r=0.0005:l=1")
            filter_graph_sink = filter_graph.add("abuffersink")
            filter_graph_buffer.link_to(loudnorm_filter_node)
            loudnorm_filter_node.link_to(speechnorm_filter_node)
            speechnorm_filter_node.link_to(filter_graph_sink)
            filter_graph.configure()

        resampler = av.AudioResampler(
            format=av.AudioFormat("s16"), layout=layout, rate=sample_rate
        )

        dts = 0
        for frame in input_container.decode(audio=0):
            if frame is not None:
                for resampled_frame in resampler.resample(frame):
                    if filter_graph_buffer and filter_graph_sink:
                        filter_graph_buffer.push(resampled_frame)
                        processed_frames = pull_until_blocked(filter_graph_buffer)
                    else:
                        processed_frames = [resampled_frame]

                    for processed_frame in processed_frames:
                        processed_frame.pts = dts
                        dts += processed_frame.samples

                        for packet in audio_stream.encode(processed_frame):
                            output_container.mux(packet)

        # flush filters
        if filter_graph_buffer and filter_graph_sink:
            try:
                # signal the end of the stream
                filter_graph_buffer.push(None)
                while True:
                    try:
                        processed_frame = filter_graph_sink.pull()
                        processed_frame.pts = dts
                        dts += processed_frame.samples
                        for packet in audio_stream.encode(processed_frame):
                            output_container.mux(packet)
                    except av.AVError as e:
                        if e.errno == errno.EAGAIN:
                            break
                        raise
            except av.AVError as e:
                if e.type != "EOF":
                    raise

        # flush encoder
        try:
            for packet in audio_stream.encode(None):
                output_container.mux(packet)
        except av.AVError as e:
            if e.type != "EOF":
                raise

        output_container.close()

        output_buffer.seek(0)
        return output_buffer.read()
    except av.AVError as e:
        error("Failed to convert audio using libav with: %s", str(e))
        raise
    finally:
        debug(f"Conversion took {time.perf_counter() - before_conversion:.3f} seconds")


def pull_until_blocked(graph):
    frames = []
    while True:
        try:
            frames.append(graph.pull())
        except av.AVError as e:
            if e.errno != errno.EAGAIN:
                raise
            return frames
