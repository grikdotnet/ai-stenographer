# tests/test_audiosource_vad.py
import pytest
import queue
import time
from src.AudioSource import AudioSource
from src.types import AudioSegment


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
                'chunk_duration': 0.032,  # 32ms chunks match VAD frame size
                'silence_energy_threshold': 0.9  # Lower for faster word-level cuts
            },
            'vad': {
                'model_path': './models/silero_vad/silero_vad.onnx',
                'threshold': 0.5,
                'frame_duration_ms': 32
            },
            'windowing': {
                'window_duration': 3.0,
                'step_size': 1.0,
                'max_speech_duration_ms': 3000
            }
        }

    def test_processes_chunks_through_vad(self, config, real_vad, speech_audio, mock_windower):
        """AudioSource should process audio chunks through VAD and emit segments."""
        chunk_queue = queue.Queue()

        # Create AudioSource (VAD always enabled)
        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=real_vad,
            windower=mock_windower,
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
        audio_source.flush()

        # Should have emitted at least one speech segment
        assert not chunk_queue.empty()

        segment = chunk_queue.get()
        assert isinstance(segment, AudioSegment)
        assert segment.type == 'preliminary'
        assert len(segment.data) > 0
        assert segment.start_time >= 0.0
        assert segment.end_time > segment.start_time
        assert len(segment.chunk_ids) == 1  # Preliminary segments have single chunk ID


    def test_emits_variable_length_segments(self, config, real_vad, speech_with_pause_audio, mock_windower):
        """AudioSource with VAD should emit segments during streaming, not just after flush.

        Pattern: 1s continuous speech + 0.5s silence + 1s continuous speech
        Expected: First segment appears in queue after silence (streaming behavior)
        """
        chunk_queue = queue.Queue()

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=real_vad,
            windower=mock_windower,
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
        assert isinstance(first_segment, AudioSegment)
        assert first_segment.type == 'preliminary'

        # Verify segment is approximately 1s (VAD frame alignment may cause slight variance)
        duration = first_segment.end_time - first_segment.start_time
        expected_duration = 1.0  # 1 second of speech
        tolerance = 0.25  # ±250ms tolerance for VAD frame alignment
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
        audio_source.flush()

        # Should have second segment
        assert not chunk_queue.empty(), "Second segment should be emitted after flush"
        second_segment = chunk_queue.get()
        assert isinstance(second_segment, AudioSegment)
        assert second_segment.type == 'preliminary'

        # Verify segment is approximately 1s
        duration = second_segment.end_time - second_segment.start_time
        expected_duration = 1.0
        tolerance = 0.25
        assert abs(duration - expected_duration) < tolerance, \
            f"Expected {expected_duration}s ±{tolerance}s, got {duration}s"




    def test_calls_windower_for_each_segment(self, config, real_vad, speech_audio):
        """AudioSource should call windower.process_segment() for each preliminary segment."""
        from unittest.mock import Mock

        chunk_queue = queue.Queue()
        mock_windower = Mock()

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=real_vad,
            windower=mock_windower,
            config=config
        )

        # Feed audio that will produce segments
        chunk_size = int(config['audio']['sample_rate'] * config['audio']['chunk_duration'])
        for i in range(0, len(speech_audio[:16000]), chunk_size):
            chunk = speech_audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                break
            audio_source.process_chunk_with_vad(chunk, time.time())

        # Flush to get final segment
        audio_source.flush()

        # Verify windower.process_segment was called
        assert mock_windower.process_segment.called
        assert mock_windower.process_segment.call_count >= 1

        # Verify each call received an AudioSegment
        for call in mock_windower.process_segment.call_args_list:
            segment = call[0][0]
            assert isinstance(segment, AudioSegment)
            assert segment.type == 'preliminary'


    def test_calls_windower_flush_on_stop(self, config, real_vad, speech_audio):
        """AudioSource.stop() should call windower.flush() to emit final window."""
        from unittest.mock import Mock

        chunk_queue = queue.Queue()
        mock_windower = Mock()

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=real_vad,
            windower=mock_windower,
            config=config
        )

        # Feed some audio
        chunk_size = int(config['audio']['sample_rate'] * config['audio']['chunk_duration'])
        for i in range(0, len(speech_audio[:8000]), chunk_size):
            chunk = speech_audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                break
            audio_source.process_chunk_with_vad(chunk, time.time())

        # Stop should call windower.flush()
        audio_source.stop()

        # Verify windower.flush() was called
        assert mock_windower.flush.called
        assert mock_windower.flush.call_count == 1
