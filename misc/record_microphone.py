from __future__ import annotations

import argparse
from pathlib import Path
import time
import wave

import numpy as np
import sounddevice as sd


def to_pcm16(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    if samples.ndim != 2:
        raise ValueError("audio must be a 1D mono array or 2D frames-by-channels array")
    if samples.shape[1] < 1:
        raise ValueError("audio must contain at least one channel")

    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * np.iinfo(np.int16).max).astype(np.int16)


def write_wav(output_path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = to_pcm16(audio)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(pcm.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return path


def write_pcm16_frames(wav_file: wave.Wave_write, audio: np.ndarray) -> None:
    wav_file.writeframes(to_pcm16(audio).tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record microphone audio to a PCM16 WAV file."
    )
    parser.add_argument("output", nargs="?", type=Path, help="Destination WAV path.")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--device", help="Optional sounddevice input device index or name.")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available audio devices and exit.",
    )
    args = parser.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return 0
    if args.output is None:
        parser.error("output is required unless --list-devices is used")

    if args.sample_rate <= 0:
        parser.error("--sample-rate must be greater than zero")
    if args.channels <= 0:
        parser.error("--channels must be greater than zero")

    device: int | str | None = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("Recording. Press Ctrl+C to stop.")
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(args.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(args.sample_rate)

        def callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
            if status:
                print(status)
            write_pcm16_frames(wav_file, indata)

        try:
            with sd.InputStream(
                samplerate=args.sample_rate,
                channels=args.channels,
                dtype="float32",
                device=device,
                callback=callback,
            ):
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
