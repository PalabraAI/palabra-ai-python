"""Palabra AI Benchmark - Data Collection Only"""

import argparse
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from palabra_ai import Config, PalabraAI, SourceLang, TargetLang
from palabra_ai.constant import BYTES_PER_SAMPLE
from palabra_ai.enum import Kind
from palabra_ai.lang import Language
from palabra_ai.task.adapter.dummy import DummyWriter
from palabra_ai.task.adapter.file import FileReader
from palabra_ai.util.orjson import to_json


def save_wav(chunks: list, output_path: Path, sample_rate: int, channels: int):
    """Save audio chunks to WAV file"""
    chunks.sort(key=lambda c: c['index'])
    audio_data = b"".join(c['audio'] for c in chunks)

    # Trim to int16 boundary
    samples_count = len(audio_data) // BYTES_PER_SAMPLE
    audio_data = audio_data[:samples_count * BYTES_PER_SAMPLE]

    # Convert to int16 numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Save using wave module
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setframerate(sample_rate)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.writeframes(audio_array.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Palabra AI Benchmark - Data Collection")
    parser.add_argument("audio", help="Audio file path")
    parser.add_argument("source_lang", nargs="?", help="Source language")
    parser.add_argument("target_lang", nargs="?", help="Target language")
    parser.add_argument("--config", type=Path, help="JSON config file")
    parser.add_argument("--output-dir", type=Path, default=Path("1Oct25"), help="Output directory")

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    if args.config:
        config = Config.from_json(args.config.read_text())
        source_lang = config.source.lang.code
        target_lang = config.targets[0].lang.code
    else:
        if not args.source_lang or not args.target_lang:
            parser.error("source_lang and target_lang required without --config")
        source_lang = args.source_lang
        target_lang = args.target_lang

        config = Config(
            source=SourceLang(Language.get_or_create(source_lang), FileReader(str(audio_path))),
            targets=[TargetLang(Language.get_or_create(target_lang), DummyWriter())],
            benchmark=True
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running benchmark: {source_lang} → {target_lang}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    palabra = PalabraAI()
    result = palabra.run(config, no_raise=True)

    if not result.ok or not result.io_data:
        raise RuntimeError(f"Benchmark failed: {result.exc}")

    print("Collecting data...")

    # Save raw result
    (output_dir / f"benchmark_raw_result_{timestamp}.json").write_bytes(
        to_json(result.model_dump())
    )
    print(f"✓ Saved raw result")

    # Save io_data summary
    (output_dir / f"benchmark_io_data_{timestamp}.json").write_bytes(
        to_json(result.io_data)
    )
    print(f"✓ Saved io_data summary")

    # # Extract audio chunks from io_events
    # input_chunks = []
    # output_chunks = []

    # for event in result.io_data.events:
    #     if event.header.kind == Kind.AUDIO:
    #         raw_audio = event.raw.encode('utf-8') if isinstance(event.raw, str) else event.raw
    #
    #         chunk_data = {
    #             'index': event.header.num,
    #             'audio': raw_audio
    #         }
    #
    #         if event.header.dir.value == "in":
    #             input_chunks.append(chunk_data)
    #         else:
    #             output_chunks.append(chunk_data)

    # # Save audio files
    # if input_chunks:
    #     save_wav(
    #         input_chunks,
    #         output_dir / f"benchmark_in_{source_lang}_{timestamp}.wav",
    #         result.io_data.input_sample_rate,
    #         result.io_data.channels
    #     )
    #     print(f"✓ Saved input audio ({len(input_chunks)} chunks)")
    #
    # if output_chunks:
    #     save_wav(
    #         output_chunks,
    #         output_dir / f"benchmark_out_{target_lang}_{timestamp}.wav",
    #         result.io_data.output_sample_rate,
    #         result.io_data.channels
    #     )
    #     print(f"✓ Saved output audio ({len(output_chunks)} chunks)")

    print(f"\n✅ Done! Files in {output_dir}/")
    print(f"   - benchmark_raw_result_{timestamp}.json")
    print(f"   - benchmark_io_data_{timestamp}.json")
    print(f"   - benchmark_in_{source_lang}_{timestamp}.wav")
    print(f"   - benchmark_out_{target_lang}_{timestamp}.wav")


if __name__ == "__main__":
    main()
