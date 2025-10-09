# tests/test_audiosource_buffering.py
import pytest
import queue
import numpy as np
from unittest.mock import Mock
from src.AudioSource import AudioSource
from src.types import AudioSegment


class TestAudioSourceBuffering:
    """Tests for AudioSource speech buffering and segmentation logic.

    Tests that AudioSource correctly buffers speech frames based on VAD results,
    applies timing constraints, and emits segments at appropriate times.
    """

    @pytest.fixture
    def config(self):
        """Standard configuration for AudioSource."""
        return {
            'audio': {
                'sample_rate': 16000,
                'chunk_duration': 0.032,
                'silence_energy_threshold': 2.0  # ~4 frames for tests
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

    @pytest.fixture
    def mock_vad(self):
        """Mock VAD that returns controllable results."""
        vad = Mock()
        vad.process_frame = Mock()
        vad.threshold = 0.5  # Add threshold attribute for silence energy logic
        return vad

    def test_accumulates_speech_frames(self, config, mock_vad, mock_windower):
        """AudioSource should buffer consecutive speech frames."""
        chunk_queue = queue.Queue()

        # VAD returns speech for all frames
        mock_vad.process_frame.return_value = {'is_speech': True, 'speech_probability': 0.9}

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=config
        )

        # Feed 5 speech frames (160ms total)
        frame_size = 512  # 32ms @ 16kHz
        for i in range(5):
            frame = np.random.randn(frame_size).astype(np.float32) * 0.1
            audio_source.process_chunk_with_vad(frame, float(i) * 0.032)

        # Should not emit yet (no silence boundary)
        assert chunk_queue.empty(), "Should not emit during continuous speech"

        # Verify internal buffering (after refactoring, AudioSource will have speech_buffer)
        assert hasattr(audio_source, 'speech_buffer'), "AudioSource should have speech_buffer"
        assert len(audio_source.speech_buffer) == 5, "Should have buffered 5 frames"

    def test_emits_segment_after_silence_timeout(self, config, mock_vad, mock_windower):
        """AudioSource should emit segment when silence_timeout_ms is reached."""
        chunk_queue = queue.Queue()

        # Sequence: 3 speech frames + 4 silence frames (silence_timeout = 128ms = 4 frames)
        call_count = [0]
        def vad_side_effect(frame):
            call_count[0] += 1
            is_speech = call_count[0] <= 3  # First 3 are speech
            return {'is_speech': is_speech, 'speech_probability': 0.9 if is_speech else 0.1}

        mock_vad.process_frame.side_effect = vad_side_effect

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=config
        )

        frame_size = 512
        # Feed 7 frames: 3 speech + 4 silence
        for i in range(7):
            frame = np.random.randn(frame_size).astype(np.float32) * 0.1
            audio_source.process_chunk_with_vad(frame, float(i) * 0.032)

        # After 4 silence frames (128ms), segment should be emitted
        assert not chunk_queue.empty(), "Segment should be emitted after silence timeout"

        segment = chunk_queue.get()
        assert isinstance(segment, AudioSegment)
        assert segment.type == 'preliminary'
        # Segment should contain 3 speech frames (96ms)
        expected_samples = 3 * frame_size
        assert len(segment.data) == expected_samples


    def test_handles_max_speech_duration(self, config, mock_vad, mock_windower):
        """AudioSource should split segments exceeding max_speech_duration_ms."""
        config['windowing']['max_speech_duration_ms'] = 160  # 5 frames max (160ms)
        chunk_queue = queue.Queue()

        # All speech - should auto-split at 5 frames
        mock_vad.process_frame.return_value = {'is_speech': True, 'speech_probability': 0.9}

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=config
        )

        frame_size = 512
        # Feed 6 speech frames - should split when frame 5 reaches exact max duration
        for i in range(6):
            frame = np.random.randn(frame_size).astype(np.float32) * 0.1
            audio_source.process_chunk_with_vad(frame, float(i) * 0.032)

        # Should have emitted first segment (5 frames = 160ms, split at exact max duration)
        assert not chunk_queue.empty(), "Should emit segment at max duration"

        first_segment = chunk_queue.get()
        assert len(first_segment.data) == 5 * frame_size, \
            f"First segment should be 5 frames (160ms = max), got {len(first_segment.data) // frame_size} frames"

        # Buffer now has frame 6 (second segment started)
        assert audio_source.speech_buffer is not None
        assert len(audio_source.speech_buffer) == 1, \
            f"Buffer should have 1 frame (frame 6), but has {len(audio_source.speech_buffer)} frames"

    def test_flush_emits_pending_segment(self, config, mock_vad, mock_windower):
        """flush() should emit any buffered speech segment."""
        chunk_queue = queue.Queue()

        # 3 speech frames, no silence
        mock_vad.process_frame.return_value = {'is_speech': True, 'speech_probability': 0.9}

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=config
        )

        frame_size = 512
        for i in range(3):
            frame = np.random.randn(frame_size).astype(np.float32) * 0.1
            audio_source.process_chunk_with_vad(frame, float(i) * 0.032)

        # No segment yet (no silence)
        assert chunk_queue.empty()

        # Flush should emit pending segment
        audio_source.flush()

        assert not chunk_queue.empty(), "flush() should emit buffered segment"
        segment = chunk_queue.get()
        assert len(segment.data) == 3 * frame_size

    def test_maintains_chunk_ids(self, config, mock_vad, mock_windower):
        """AudioSource should assign sequential chunk IDs to segments."""
        chunk_queue = queue.Queue()

        # Sequence: 3 speech + 4 silence + 3 speech + 4 silence (2 segments)
        call_count = [0]
        def vad_side_effect(frame):
            call_count[0] += 1
            cycle_pos = (call_count[0] - 1) % 7
            is_speech = cycle_pos < 3
            return {'is_speech': is_speech, 'speech_probability': 0.9 if is_speech else 0.1}

        mock_vad.process_frame.side_effect = vad_side_effect

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=config
        )

        frame_size = 512
        for i in range(14):  # Two complete cycles
            frame = np.random.randn(frame_size).astype(np.float32) * 0.1
            audio_source.process_chunk_with_vad(frame, float(i) * 0.032)

        # Should have 2 segments
        segments = []
        while not chunk_queue.empty():
            segments.append(chunk_queue.get())

        assert len(segments) == 2, "Should have emitted 2 segments"

        # Each segment should have single chunk_id (preliminary)
        assert len(segments[0].chunk_ids) == 1
        assert len(segments[1].chunk_ids) == 1

        # IDs should be sequential
        assert segments[1].chunk_ids[0] > segments[0].chunk_ids[0]

    def test_timing_accuracy(self, config, mock_vad, mock_windower):
        """AudioSource should accurately track segment start/end times."""
        chunk_queue = queue.Queue()

        # 5 speech frames starting at t=0.1s
        mock_vad.process_frame.return_value = {'is_speech': True, 'speech_probability': 0.9}

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=config
        )

        frame_size = 512
        frame_duration = 0.032
        start_time = 0.1

        # Feed 5 speech frames
        for i in range(5):
            frame = np.random.randn(frame_size).astype(np.float32) * 0.1
            audio_source.process_chunk_with_vad(frame, start_time + i * frame_duration)

        # Flush to emit
        audio_source.flush()

        segment = chunk_queue.get()

        # Check timing
        expected_start = start_time
        expected_end = start_time + 5 * frame_duration

        assert abs(segment.start_time - expected_start) < 0.001, \
            f"Start time should be {expected_start}, got {segment.start_time}"
        assert abs(segment.end_time - expected_end) < 0.001, \
            f"End time should be {expected_end}, got {segment.end_time}"
