# src/sound/FileAudioSource.py
from __future__ import annotations
import queue
import numpy as np
import time
import logging
import threading
from typing import Dict, Any, List, Tuple
from pathlib import Path


class FileAudioSource:
    """File-based audio source for testing timestamp accuracy.

    FileAudioSource simulates real-time audio capture by reading from a WAV file
    and feeding chunks to queue at real-time rate (32ms per chunk). This enables
    reproducible testing of timestamp accuracy and STT model behavior.

    The interface matches AudioSource (start/stop methods) for drop-in replacement.

    Processing steps:
    1. Load audio file using soundfile
    2. Convert to mono if stereo (take first channel)
    3. Resample to 16kHz if needed
    4. Split into 32ms chunks (512 samples)
    5. Generate timestamps based on chunk positions
    6. Feed chunks to queue at real-time rate when started

    Args:
        chunk_queue: Queue to send audio chunks (dict with audio, timestamp)
        config: Configuration dictionary loaded from stt_config.json
        file_path: Path to WAV file to load
        verbose: Enable verbose logging
    """

    def __init__(self,
                 chunk_queue: queue.Queue,
                 config: Dict[str, Any],
                 file_path: str,
                 verbose: bool = False):

        self.chunk_queue: queue.Queue = chunk_queue
        self.file_path: str = file_path
        self.verbose: bool = verbose

        self.sample_rate: int = config['audio']['sample_rate']
        chunk_duration = config['audio']['chunk_duration']
        self.chunk_size: int = int(self.sample_rate * chunk_duration)

        # Load and prepare audio
        self.audio: List[np.ndarray]
        self.timestamps: List[float]
        self.audio, self.timestamps = self._load_audio()

        self.is_running: bool = False
        self.thread: threading.Thread | None = None

        if self.verbose:
            logging.info(f"FileAudioSource: loaded {len(self.audio)} chunks from {file_path}")


    def _load_audio(self) -> Tuple[List[np.ndarray], List[float]]:
        """Load audio file, resample if needed, split into chunks.

        Loads WAV file using soundfile, converts to mono if stereo,
        resamples to 16kHz if needed, splits into 512-sample chunks,
        and generates accurate timestamps for each chunk.

        Returns:
            Tuple of (chunks, timestamps):
                - chunks: List of audio chunks (np.ndarray, shape (512,), float32)
                - timestamps: List of timestamps in seconds (float)
        """
        import soundfile as sf

        # Load audio file
        audio, sr = sf.read(self.file_path, dtype='float32')

        # Convert to mono if stereo (take first channel, matching AudioSource)
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        # Resample to 16kHz if needed
        if sr != self.sample_rate:
            from scipy import signal
            num_samples = int(len(audio) * self.sample_rate / sr)
            audio = signal.resample(audio, num_samples).astype(np.float32)

        # Split into chunks and create timestamps
        chunks: List[np.ndarray] = []
        timestamps: List[float] = []
        chunk_duration = self.chunk_size / self.sample_rate

        for i in range(0, len(audio), self.chunk_size):
            chunk = audio[i:i+self.chunk_size]

            # Pad last chunk if shorter than chunk_size
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))

            chunks.append(chunk)
            timestamps.append(i / self.sample_rate)

        return chunks, timestamps


    def start(self) -> None:
        """Start feeding chunks to queue at real-time rate.

        Launches a background thread that feeds audio chunks to chunk_queue
        at approximately 32ms intervals to simulate real-time capture.
        """
        self.is_running = True
        self.thread = threading.Thread(target=self._feed_chunks, daemon=True)
        self.thread.start()

        if self.verbose:
            logging.info("FileAudioSource: started feeding chunks")


    def _feed_chunks(self) -> None:
        """Feed chunks to queue at real-time rate (32ms per chunk).

        This method runs in a background thread and simulates real-time audio
        capture by sending chunks at 32ms intervals. Uses time.time() for
        timestamps to match AudioSource behavior.
        """
        chunk_duration = self.chunk_size / self.sample_rate
        start_time = time.time()

        for i, (chunk, original_timestamp) in enumerate(zip(self.audio, self.timestamps)):
            if not self.is_running:
                break

            # Calculate when this chunk should be sent
            target_time = start_time + (i * chunk_duration)
            current_time = time.time()
            sleep_time = target_time - current_time

            if sleep_time > 0:
                time.sleep(sleep_time)

            # Use current time for timestamp to match AudioSource behavior
            chunk_data = {
                'audio': chunk,
                'timestamp': time.time()
            }

            try:
                self.chunk_queue.put_nowait(chunk_data)

            except queue.Full:
                logging.warning("chunk_queue full, dropping chunk")

        self.is_running = False

        if self.verbose:
            logging.info("FileAudioSource: finished feeding all chunks")


    def stop(self) -> None:
        """Stop feeding chunks to queue.

        Sets is_running flag to False and waits for background thread
        to terminate (up to 1 second timeout).
        """
        self.is_running = False

        if self.thread:
            self.thread.join(timeout=1.0)

        if self.verbose:
            logging.info("FileAudioSource: stopped")
