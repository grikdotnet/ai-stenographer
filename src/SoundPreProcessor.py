# src/SoundPreProcessor.py
from __future__ import annotations
import queue
import threading
import numpy as np
import logging
from collections import deque
from typing import Dict, Any, List, TYPE_CHECKING
from src.types import AudioSegment

if TYPE_CHECKING:
    from src.VoiceActivityDetector import VoiceActivityDetector
    from src.AdaptiveWindower import AdaptiveWindower


class SoundPreProcessor:
    """Processes raw audio chunks through VAD and buffering.

    Runs in dedicated thread to avoid blocking audio capture.
    Orchestrates VAD, speech buffering, and AdaptiveWindower.

    Context Buffer Strategy:
    - Maintains circular buffer of last 48 chunks (1.536s @ 32ms/chunk)
    - Accumulates both speech and silence chunks for context extraction
    - left_context: chunks before speech
    - right_context: trailing silence after speech ends
    - Enables better STT quality for short words and prevents hallucinations

    Dual-Path Audio Processing:
    - Normalized audio → VAD (Silero is volume-dependent, needs consistent levels)
    - Raw audio → speech_buffer → STT (Parakeet requires original amplitude)
    - This ensures accurate VAD detection AND maximum STT accuracy

    RMS Normalization (for VAD only):
    - Normalizes to target RMS level (default: 0.05)
    - Skips normalization for silence (RMS < silence_threshold)
    - Prevents boosting background noise
    - Uses temporal smoothing to prevent sudden gain changes

    Silence Energy Logic:
    - High speech prob (over vad threshold): reset energy to 0.0
    - Lower prob is added up (1.0 - prob)
    - Finalize when energy >= silence_energy_threshold (e.g., 1.5)
    - Strong silence (prob=0.1) finalizes faster than weak silence (prob=0.3)

    Silence Timeout Logic:
    - After speech ends, track silence duration from timestamp parameter
    - If silence exceeds timeout AND speech was detected before, flush windower
    - Reset speech_before_silence flag after flush to prevent repeated flushes

    Args:
        chunk_queue: Queue to read raw audio chunks from AudioSource
        speech_queue: Queue to write AudioSegments (preliminary + finalized)
        vad: VoiceActivityDetector instance
        windower: AdaptiveWindower instance (called synchronously)
        config: Configuration dictionary
        verbose: Enable verbose logging
    """

    CONTEXT_BUFFER_SIZE = 48  # 1.536s @ 32ms/chunk

    def __init__(self,
                 chunk_queue: queue.Queue,
                 speech_queue: queue.Queue,
                 vad: VoiceActivityDetector,
                 windower: AdaptiveWindower,
                 config: Dict[str, Any],
                 verbose: bool = False):

        self.chunk_queue: queue.Queue = chunk_queue      # INPUT: raw audio
        self.speech_queue: queue.Queue = speech_queue    # OUTPUT: preliminary segments
        self.vad: VoiceActivityDetector = vad            # Called for speech detection
        self.windower: AdaptiveWindower = windower       # Called synchronously

        # Configuration
        self.sample_rate: int = config['audio']['sample_rate']
        self.frame_duration_ms: int = config['vad']['frame_duration_ms']
        self.max_speech_duration_ms: int = config['windowing']['max_speech_duration_ms']
        self.silence_energy_threshold: float = config['audio']['silence_energy_threshold']
        self.silence_timeout: float = config['windowing']['silence_timeout']

        # RMS normalization (AGC)
        rms_config = config['audio']['rms_normalization']
        self.target_rms: float = rms_config['target_rms']
        self.rms_silence_threshold: float = rms_config['silence_threshold']
        self.gain_smoothing: float = rms_config['gain_smoothing']
        self.current_gain: float = 1.0  # Start with unity gain

        # Speech buffering state
        # Each buffer item: {'audio': np.ndarray, 'timestamp': float, 'chunk_id': int}
        self.speech_buffer: List[Dict[str, Any]] = []
        self.is_speech_active: bool = False
        self.speech_start_time: float = 0.0
        self.silence_energy: float = 0.0
        self.last_speech_timestamp: float = 0.0
        self.speech_before_silence: bool = False

        # Consecutive speech chunk counter for false positive prevention
        # Requires 3 consecutive chunks above threshold before starting segment
        self.consecutive_speech_count: int = 0
        self.CONSECUTIVE_SPEECH_CHUNKS: int = 3

        # Context buffer: circular buffer for gathering sound context
        # Each item: {'audio': np.ndarray, 'timestamp': float, 'chunk_id': int|None, 'is_speech': bool}
        self.context_buffer: deque = deque(maxlen=self.CONTEXT_BUFFER_SIZE)
        # Captured left context when speech starts
        self.left_context_snapshot: List[np.ndarray] | None = None  

        self.chunk_id_counter: int = 0

        self.is_running: bool = False
        self.thread: threading.Thread | None = None
        self.verbose: bool = verbose


    def start(self) -> None:
        """Start processing thread"""
        self.is_running = True
        self.thread = threading.Thread(target=self.process, daemon=True)
        self.thread.start()


    def stop(self) -> None:
        """Stop processing thread and flush pending segments"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.flush()


    def process(self) -> None:
        """Main loop: read chunk_queue → emit to speech_queue"""
        while self.is_running:
            try:
                chunk_data = self.chunk_queue.get(timeout=0.1)
                self._process_chunk(chunk_data)
            except queue.Empty:
                continue


    def _process_chunk(self, chunk_data: Dict[str, Any]) -> None:
        """Process single raw audio chunk (was process_chunk_with_vad in AudioSource).

        Dual-path processing:
        1. Normalize audio for VAD → ensures consistent speech detection
        2. Call VAD with normalized audio → get speech probability
        3. Buffer raw audio for STT → preserves original quality

        Args:
            chunk_data: Dict with 'audio', 'timestamp' (no chunk_id yet - assigned after VAD)
        """
        audio = chunk_data['audio']
        timestamp = chunk_data['timestamp']

        normalized_audio = self._normalize_rms(audio)

        vad_result = self.vad.process_frame(normalized_audio)
        speech_prob = vad_result['speech_probability']

        if vad_result['is_speech']:
            self.silence_energy = 0.0
            self.last_speech_timestamp = timestamp
            self.speech_before_silence = True
            self._handle_speech_frame(audio, timestamp, speech_prob)
        else:
            # Reset consecutive counter on silence
            self.consecutive_speech_count = 0

            if self.speech_before_silence:
                self.silence_energy += (1.0 - speech_prob)
                silence_duration = timestamp - self.last_speech_timestamp
                if silence_duration >= self.silence_timeout:
                    if self.verbose:
                        logging.debug(f"SoundPreProcessor: silence_timeout reached ({silence_duration:.2f}s)")
                    self.windower.flush()
                    self.speech_before_silence = False

            self._handle_silence_frame(audio, timestamp)

        if self.verbose and speech_prob > 0.2:
            logging.debug(f"SoundPreProcessor: speech_prob {speech_prob:.2f} gain {self.current_gain}")


    def _normalize_rms(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Normalize audio chunk to target RMS level with temporal smoothing (AGC).

        Purpose: Creates normalized audio for VAD only. The raw audio is preserved
        for STT downstream (Parakeet requires original amplitude per NeMo design).

        Automatic Gain Control that boosts quiet audio and reduces loud audio
        to maintain consistent volume level for VAD. Uses temporal smoothing to
        prevent sudden gain changes and transient noise amplification.

        Temporal Smoothing Algorithm:
        - Calculate target gain for current chunk: target_gain = target_rms / current_rms
        - Smooth with previous gain: current_gain = α * current_gain + (1-α) * target_gain
        - Apply smoothed gain to chunk
        - Prevents sudden volume jumps between chunks (e.g., speech → pause → speech)

        Args:
            audio_chunk: Raw audio data to normalize

        Returns:
            Normalized audio chunk with smoothed gain applied (for VAD use only)
        """
        rms = np.sqrt(np.mean(audio_chunk**2))

        # Don't normalize silence
        if rms < self.rms_silence_threshold:
            return audio_chunk

        target_gain = self.target_rms / rms

        # Apply temporal smoothing to prevent sudden gain changes
        # current_gain = α * old_gain + (1-α) * new_gain
        # Higher α (e.g., 0.9) = slower, smoother changes
        # Lower α (e.g., 0.5) = faster response
        self.current_gain = self.gain_smoothing * self.current_gain + (1 - self.gain_smoothing) * target_gain

        return audio_chunk * self.current_gain


    def _handle_speech_frame(self, audio_chunk: np.ndarray, timestamp: float, speech_prob: float) -> None:
        """Buffer a speech frame, managing consecutive speech logic internally.

        Consecutive Speech Logic:
        - Requires 3 consecutive speech chunks to start segment (prevents false positives)
        - Chunks 1-3: Buffered in context_buffer as unconfirmed speech
        - Chunk 3: Triggers segment start, extracts last 3 chunks from buffer (KEEP in buffer)
        - Subsequent chunks: Added to both context_buffer AND speech_buffer

        Context Buffer Management:
        - Circular buffer (deque with maxlen=48) maintains ALL recent audio history
        - Speech chunks remain in buffer (not removed) for continuous history
        - On segment start: snapshot left_context from everything BEFORE last 3 chunks
        - Circular buffer automatically evicts old chunks when full

        Args:
            audio_chunk: Audio frame data (raw, not normalized)
            timestamp: Timestamp of this frame
            speech_prob: Speech probability from VAD (for logging)
        """
        self.consecutive_speech_count += 1

        # Start segment on Nth consecutive chunk
        if self.consecutive_speech_count == self.CONSECUTIVE_SPEECH_CHUNKS:
            if self.verbose:
                logging.debug(f"SoundPreProcessor: {self.CONSECUTIVE_SPEECH_CHUNKS} consecutive chunks, starting segment. "
                             f"speech_prob {speech_prob:.2f} gain {self.current_gain}")

            buffer_list = list(self.context_buffer)

            # Snapshot left_context
            chunks_already_in_buffer = self.CONSECUTIVE_SPEECH_CHUNKS - 1
            self.left_context_snapshot = [chunk['audio'] for chunk in buffer_list[:-chunks_already_in_buffer]]
            last_chunks = buffer_list[-chunks_already_in_buffer:]

            if self.verbose:
                logging.debug(f"SoundPreProcessor: Starting segment, "
                             f"captured {len(self.left_context_snapshot)} left context chunks")

            self.is_speech_active = True
            self.speech_start_time = last_chunks[0]['timestamp']
            self.speech_buffer = []

            # Add first speech chunks to speech_buffer
            for chunk in last_chunks:
                self.speech_buffer.append(chunk)

        # Always add to context_buffer (no chunk_id yet)
        buffer_chunk = {
            'audio': audio_chunk,
            'timestamp': timestamp,
            'chunk_id': self.chunk_id_counter,
            'is_speech': True
        }
        self.context_buffer.append(buffer_chunk)

        if self.is_speech_active: # will skip single false negatives from VAD
            # Add to speech_buffer with chunk_id
            speech_chunk = buffer_chunk
            self.speech_buffer.append(speech_chunk)

        self.chunk_id_counter += 1

        # Check max duration
        if self.is_speech_active:
            current_duration_ms = len(self.speech_buffer) * self.frame_duration_ms
            if current_duration_ms >= self.max_speech_duration_ms:
                if self.verbose:
                    logging.debug(f"SoundPreProcessor: max_speech_duration reached, starting new segment")
                self._finalize_segment()
                # Start new segment
                self.is_speech_active = True
                self.speech_start_time = timestamp
                self.speech_buffer = []


    def _handle_silence_frame(self, audio_chunk: np.ndarray, timestamp: float) -> None:
        """Handle silence, buffer silence chunks during energy accumulation, finalize when threshold reached.

        Context Buffer Logic:
        - Always append to context_buffer with is_speech=False
        - During active speech: accumulate in speech_buffer as trailing silence (right_context)
        - Trailing silence chunks have no chunk_id (not counted in chunk_ids list)

        Args:
            audio_chunk: Audio frame data (buffered to capture release transients)
            timestamp: Timestamp of this frame
        """
        if self.is_speech_active:
            chunk = {
                'audio': audio_chunk,
                'timestamp': timestamp,
                'chunk_id': self.chunk_id_counter,
                'is_speech': False,
            }
            self.speech_buffer.append(chunk)
            self.chunk_id_counter += 1

            if self.verbose:
                logging.debug(f"SoundPreProcessor: silence_energy={self.silence_energy:.2f}")

            if self.silence_energy >= self.silence_energy_threshold:
                if self.verbose:
                    logging.debug(f"SoundPreProcessor: silence_energy_threshold reached")
                self.is_speech_active = False
                self._finalize_segment()
        else:
            chunk = {
                'audio': audio_chunk,
                'timestamp': timestamp,
                'chunk_id': None,
                'is_speech': False,
            }

        # Always add to circular context buffer
        self.context_buffer.append(chunk)


    def _finalize_segment(self) -> None:
        """Emit preliminary AudioSegment and call windower.

        Creates AudioSegment from buffered chunks and:
        1. Uses left_context_snapshot
        2. Extracts data from speech_buffer (speech chunks only)
        3. Assigns trailing silence to right_context
        4. Emits to speech_queue (preliminary segment)
        5. Calls windower.process_segment() synchronously
        6. Resets buffering state

        """
        if not self.speech_buffer:
            return

        left_context = (np.concatenate(self.left_context_snapshot)
                       if self.left_context_snapshot
                       else np.array([], dtype=np.float32))

        data_chunks = [chunk['audio'] for chunk in self.speech_buffer
                      if chunk.get('is_speech', True)]
        data = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float32)

        right_context_chunks = []
        for chunk in reversed(self.speech_buffer):
            if not chunk.get('is_speech', True):
                right_context_chunks.insert(0, chunk['audio'])
            else:
                break  # Stop at the first speech chunk from the end

        right_context = (np.concatenate(right_context_chunks)
                        if right_context_chunks
                        else np.array([], dtype=np.float32))

        chunk_ids = [chunk['chunk_id'] for chunk in self.speech_buffer
                    if chunk.get('is_speech', True)]

        end_time = self.speech_start_time + (len(data) / self.sample_rate)

        segment = AudioSegment(
            type='preliminary',
            data=data,
            left_context=left_context,
            right_context=right_context,
            start_time=self.speech_start_time,
            end_time=end_time,
            chunk_ids=chunk_ids
        )

        if self.verbose:
            logging.debug(f"SoundPreProcessor: emitting preliminary segment, chunk_ids={chunk_ids}, "
                         f"left_context={len(left_context)} samples, right_context={len(right_context)} samples")

        self.speech_queue.put(segment)

        self.windower.process_segment(segment)

        self._reset_segment_state()


    def _reset_segment_state(self) -> None:
        """Reset state after segment finalization.
        """
        self.speech_buffer = []
        self.silence_energy = 0.0
        self.left_context_snapshot = None


    def flush(self) -> None:
        """Emit pending segment and flush windower.

        Useful for end-of-stream or testing scenarios.
        """
        if self.is_speech_active and len(self.speech_buffer) > 0:
            self._finalize_segment()

        self.windower.flush()

        if self.verbose:
            logging.debug("SoundPreProcessor: flush()")
