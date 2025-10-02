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

from palabra_ai.exc import ApiError
from palabra_ai.util.logger import debug, error, info


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


# NEW SIMPLE AUDIO PROCESSING PIPELINE
def open_audio_file(audio_data: bytes, sample_rate: int) -> np.ndarray:
    """Simple audio processing pipeline: librosa first, PyAV fallback."""
    try:
        audio_data, _ = librosa.load(BytesIO(audio_data), mono=True, sr=sample_rate)
    except Exception as e:
        try:
            info(
                'Librosa failed to open audio file with: "{}". Using libav for conversion.',
                e,
            )
            audio_data = convert_any_to_wav(
                audio_data=audio_data,
                layout="mono",
                sample_rate=sample_rate,
                s16le=True,
            )
            audio_data = np.frombuffer(audio_data, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
        except Exception as e:
            error_msg = "Failed to open audio sample file"
            debug(f"{error_msg} with:")
            raise ApiError(error_msg) from e

    return audio_data


def convert_any_to_wav(
    audio_data: bytes,
    sample_rate: int,
    layout: str = "mono",
    s16le: bool = False,
) -> bytes:
    """Convert any audio format to WAV using PyAV."""
    before_conversion_time = time.perf_counter()
    try:
        input_buffer = BytesIO(audio_data)
        input_container = av.open(input_buffer, metadata_errors="ignore")

        output_buffer = BytesIO()
        output_container = av.open(
            output_buffer, mode="w", format="wav" if not s16le else "s16le"
        )
        audio_stream = output_container.add_stream("pcm_s16le", rate=sample_rate)
        audio_stream.layout = layout
        audio_stream.time_base = Fraction(1, sample_rate)

        resampler = av.AudioResampler(
            format=av.AudioFormat("s16"), layout="mono", rate=sample_rate
        )

        dts = 0
        for frame in input_container.decode(audio=0):
            frame.pts = dts
            dts += frame.samples

            for resampled_frame in resampler.resample(frame):
                for packet in audio_stream.encode(resampled_frame):
                    output_container.mux(packet)

        for packet in audio_stream.encode():
            output_container.mux(packet)

        output_container.close()

        output_buffer.seek(0)
        return output_buffer.read()
    except av.AVError as e:
        error("Failed to convert audio using libav with: {}", str(e))
        raise
    finally:
        info(
            "Wav conversion took: {:.3f}s", time.perf_counter() - before_conversion_time
        )


def simple_preprocess_audio_file(
    file_path: str | Path,
    target_rate: int,
    normalize: bool = False,
    progress_callback=None,
    eof_silence_duration_s: float = 0.0,
) -> tuple[bytes, dict]:
    """Simple preprocessing: load with librosa/PyAV, resample only if not 16kHz."""
    debug(f"Simple preprocessing audio file {file_path}...")

    # Read file as bytes
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    # Get original audio info first
    try:
        container = av.open(str(file_path), metadata_errors="ignore")
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            container.close()
            raise ValueError("No audio streams found")

        audio_stream = audio_streams[0]
        original_rate = audio_stream.sample_rate
        duration = (
            float(audio_stream.duration * audio_stream.time_base)
            if audio_stream.duration
            else 0
        )
        channels = audio_stream.channels
        container.close()

        debug(f"Original audio: {original_rate}Hz, {channels}ch, {duration:.1f}s")
    except Exception as e:
        debug(f"Could not get audio info: {e}")
        original_rate = None
        duration = 0
        channels = 1

    # Simple resampling logic: resample to 16kHz unless already 16kHz
    if original_rate == target_rate:
        debug(f"Audio already at {target_rate}Hz, no resampling needed")
        final_rate = original_rate
        needs_resample = False
    else:
        debug(f"Resampling {original_rate}Hz -> {target_rate}Hz")
        final_rate = target_rate
        needs_resample = True

    # Process audio
    audio_array = open_audio_file(audio_bytes, final_rate)

    # Convert back to bytes
    audio_int16 = (audio_array * np.iinfo(np.int16).max).astype(np.int16)

    # Add silence padding at the end
    if eof_silence_duration_s > 0:
        silence_samples = int(eof_silence_duration_s * final_rate)
        audio_int16 = np.concatenate(
            [audio_int16, np.zeros(silence_samples, dtype=np.int16)]
        )
        debug(
            f"Added {eof_silence_duration_s}s ({silence_samples} samples) of silence padding"
        )

    processed_data = audio_int16.tobytes()

    if progress_callback:
        progress_callback(len(audio_int16))

    metadata = {
        "original_rate": original_rate or final_rate,
        "final_rate": final_rate,
        "resampled": needs_resample,
        "duration": duration,
        "size": len(processed_data),
    }

    debug(f"Simple preprocessing complete: {len(processed_data)} bytes")
    return processed_data, metadata


def simple_setup_streaming_audio(
    file_path: str | Path,
    target_rate: int,
    timeout: float = None,
) -> tuple["av.Container", "av.AudioResampler", int, dict]:
    """Simple streaming setup with configurable sample rate."""
    debug(f"Simple streaming setup for {file_path}...")

    # Open container for streaming
    container = av.open(str(file_path), timeout=timeout, metadata_errors="ignore")

    # Find audio stream
    audio_streams = [s for s in container.streams if s.type == "audio"]
    if not audio_streams:
        container.close()
        raise ValueError(f"No audio streams found in {file_path}")

    audio_stream = audio_streams[0]
    original_rate = audio_stream.sample_rate
    duration = (
        float(audio_stream.duration * audio_stream.time_base)
        if audio_stream.duration
        else 0
    )
    channels = audio_stream.channels

    debug(f"Original audio: {original_rate}Hz, {channels}ch, {duration:.1f}s")

    # Simple logic: always use target_rate (16kHz)
    needs_resample = original_rate != target_rate
    final_rate = target_rate

    debug(
        f"Simple streaming: {original_rate}Hz -> {final_rate}Hz (resample: {needs_resample})"
    )

    # Create resampler
    resampler = av.AudioResampler(
        format=av.AudioFormat("s16"), layout="mono", rate=final_rate
    )

    # Enable threading for faster decode
    audio_stream.codec_context.thread_type = av.codec.context.ThreadType.FRAME

    metadata = {
        "original_rate": original_rate,
        "final_rate": final_rate,
        "resampled": needs_resample,
        "duration": duration,
    }

    debug(f"Simple streaming setup complete: {final_rate}Hz")
    return container, resampler, final_rate, metadata
