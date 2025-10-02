# tests/test_audiosource_vad.py
import pytest
import queue
import time
from src.AudioSource import AudioSource


class TestAudioSourceVADIntegration:
    """Tests for AudioSource with VAD integration.

    Tests that AudioSource correctly processes audio through VAD, emitting
    variable-length speech segments instead of fixed-size chunks.
    """

    @pytest.fixture
    def config(self):
        """Standard configuration for AudioSource with VAD."""
        return {
            'audio': {
                'sample_rate': 16000,
                'chunk_duration': 0.032  # 32ms chunks match VAD frame size
            },
            'vad': {
                'model_path': './models/silero_vad/silero_vad.onnx',
                'threshold': 0.5,
                'frame_duration_ms': 32,
                'min_speech_duration_ms': 64,  # 2 frames minimum
                'silence_timeout_ms': 32,  # 1 frame = word-level cuts
                'max_speech_duration_ms': 3000
            }
        }

    def test_processes_chunks_through_vad(self, config, speech_audio):
        """AudioSource should process audio chunks through VAD and emit segments."""
        chunk_queue = queue.Queue()

        # Create AudioSource (VAD always enabled)
        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            config=config
        )

        # Simulate feeding audio chunks (like microphone would)
        # Use 100ms chunks (1600 samples @ 16kHz)
        chunk_size = int(config['audio']['sample_rate'] * config['audio']['chunk_duration'])

        for i in range(0, len(speech_audio[:16000]), chunk_size):
            chunk = speech_audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                break

            # Simulate what sounddevice callback would do
            audio_source.process_chunk_with_vad(chunk, time.time())

        # Flush any pending segments (simulates end of stream)
        audio_source.flush_vad()

        # Should have emitted at least one speech segment
        assert not chunk_queue.empty()

        segment = chunk_queue.get()
        assert 'is_speech' in segment
        assert 'data' in segment
        assert 'start_time' in segment
        assert 'end_time' in segment


    def test_emits_variable_length_segments(self, config, speech_with_pause_audio):
        """AudioSource with VAD should emit segments during streaming, not just after flush.

        Pattern: 1s continuous speech + 0.5s silence + 1s continuous speech
        Expected: First segment appears in queue after silence (streaming behavior)
        """
        chunk_queue = queue.Queue()

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            config=config
        )

        chunk_size = int(config['audio']['sample_rate'] * config['audio']['chunk_duration'])

        # Feed first 1s of speech - should not emit yet (no silence boundary)
        chunks_count = 16000 // chunk_size  # ~31 chunks for 1 second
        for i in range(chunks_count):
            start = i * chunk_size
            chunk = speech_with_pause_audio[start:start + chunk_size]
            audio_source.process_chunk_with_vad(chunk, time.time())

        # Should have no segments yet (speech still ongoing)
        assert chunk_queue.empty(), "No segments should be emitted during continuous speech"

        # Feed 0.5s silence - should trigger first segment emission
        silence_start = 16000  # After 1s speech
        silence_chunks = 8000 // chunk_size  # ~15 chunks for 0.5s
        for i in range(silence_chunks):
            start = silence_start + i * chunk_size
            chunk = speech_with_pause_audio[start:start + chunk_size]
            audio_source.process_chunk_with_vad(chunk, time.time())

        # STREAMING CHECK: First segment should appear NOW (before flush)
        assert not chunk_queue.empty(), "First segment should be emitted after silence timeout"
        first_segment = chunk_queue.get()
        assert first_segment['is_speech'] is True

        # Verify segment is approximately 1s (VAD frame alignment may cause slight variance)
        duration = first_segment['end_time'] - first_segment['start_time']
        expected_duration = 1.0  # 1 second of speech
        tolerance = 0.2  # ±200ms tolerance for VAD frame alignment
        assert abs(duration - expected_duration) < tolerance, \
            f"Expected {expected_duration}s ±{tolerance}s, got {duration}s"

        # Feed remaining 1s speech
        remaining_start = 24000  # After 1s speech + 0.5s silence
        for i in range(0, len(speech_with_pause_audio) - remaining_start, chunk_size):
            start = remaining_start + i
            if start + chunk_size > len(speech_with_pause_audio):
                break
            chunk = speech_with_pause_audio[start:start + chunk_size]
            audio_source.process_chunk_with_vad(chunk, time.time())

        # Flush to get second segment
        audio_source.flush_vad()

        # Should have second segment
        assert not chunk_queue.empty(), "Second segment should be emitted after flush"
        second_segment = chunk_queue.get()
        assert second_segment['is_speech'] is True

        # Verify segment is approximately 1s
        duration = second_segment['end_time'] - second_segment['start_time']
        expected_duration = 1.0
        tolerance = 0.2
        assert abs(duration - expected_duration) < tolerance, \
            f"Expected {expected_duration}s ±{tolerance}s, got {duration}s"


    def test_respects_min_speech_duration(self, config, speech_audio):
        """AudioSource VAD should filter out speech segments shorter than min_duration."""
        config['vad']['min_speech_duration_ms'] = 500  # 500ms minimum

        chunk_queue = queue.Queue()
        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            config=config
        )

        # Feed only 300ms of speech (below threshold)
        chunk_size = int(config['audio']['sample_rate'] * config['audio']['chunk_duration'])
        short_speech = speech_audio[:4800]  # 300ms at 16kHz

        for i in range(0, len(short_speech), chunk_size):
            chunk = short_speech[i:i + chunk_size]
            audio_source.process_chunk_with_vad(chunk, time.time())

        # Should not emit segment (too short)
        speech_segments = []
        while not chunk_queue.empty():
            seg = chunk_queue.get()
            if seg.get('is_speech'):
                speech_segments.append(seg)

        assert len(speech_segments) == 0
