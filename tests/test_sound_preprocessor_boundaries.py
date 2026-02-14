import queue
from unittest.mock import Mock

import numpy as np

from src.sound.SoundPreProcessor import SoundPreProcessor
from src.types import SpeechEndSignal


def _config() -> dict:
    return {
        "audio": {
            "sample_rate": 16000,
            "silence_energy_threshold": 1.5,
            "rms_normalization": {
                "target_rms": 0.05,
                "silence_threshold": 0.001,
                "gain_smoothing": 0.9,
            },
        },
        "vad": {
            "frame_duration_ms": 32,
        },
        "windowing": {
            "max_speech_duration_ms": 5000,
            "silence_timeout": 0.5,
        },
    }


def _chunk(ts: float) -> dict:
    return {"audio": np.ones(512, dtype=np.float32) * 0.1, "timestamp": ts}


def test_emits_boundary_when_utterance_ends() -> None:
    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()
    vad = Mock()
    windower = Mock()
    windower.process_segment = Mock()
    windower.flush = Mock()

    # 3 speech chunks -> active speech, then 2 silences -> silence threshold exceeded.
    vad.process_frame.side_effect = [
        {"is_speech": True, "speech_probability": 0.9},
        {"is_speech": True, "speech_probability": 0.9},
        {"is_speech": True, "speech_probability": 0.9},
        {"is_speech": False, "speech_probability": 0.0},
        {"is_speech": False, "speech_probability": 0.0},
    ]

    spp = SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=vad,
        windower=windower,
        config=_config(),
        app_state=Mock(),
    )

    timestamps = [0.0, 0.032, 0.064, 0.096, 0.128]
    for ts in timestamps:
        spp._process_chunk(_chunk(ts))

    boundary = speech_queue.get_nowait()
    assert isinstance(boundary, SpeechEndSignal)
    assert boundary.utterance_id == 1
    assert boundary.end_time > 0


def test_flush_without_open_utterance_does_not_emit_boundary() -> None:
    spp = SoundPreProcessor(
        chunk_queue=queue.Queue(),
        speech_queue=queue.Queue(),
        vad=Mock(),
        windower=Mock(flush=Mock(), process_segment=Mock()),
        config=_config(),
        app_state=Mock(),
    )
    spp.flush()
    assert spp.speech_queue.empty()
