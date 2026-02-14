import queue

import numpy as np
import pytest

from src.sound.GrowingWindowAssembler import GrowingWindowAssembler
from src.types import AudioSegment


def _config() -> dict:
    return {
        "audio": {"sample_rate": 16000},
        "windowing": {"max_window_duration": 7.0},
    }


def _segment(chunk_id: int, utterance_id: int, right_context: np.ndarray | None = None) -> AudioSegment:
    return AudioSegment(
        type="incremental",
        data=np.ones(1600, dtype=np.float32) * 0.1,
        left_context=np.array([], dtype=np.float32),
        right_context=right_context if right_context is not None else np.array([], dtype=np.float32),
        start_time=0.0,
        end_time=0.1,
        utterance_id=utterance_id,
        chunk_ids=[chunk_id],
    )


def test_flush_emits_incremental_with_right_context() -> None:
    q = queue.Queue()
    assembler = GrowingWindowAssembler(speech_queue=q, config=_config())
    right = np.ones(10, dtype=np.float32) * 0.2
    assembler.flush(_segment(1, utterance_id=7, right_context=right))
    window = q.get_nowait()
    assert window.type == "incremental"
    assert window.utterance_id == 7
    assert np.array_equal(window.right_context, right)


def test_mixed_utterance_ids_raise() -> None:
    q = queue.Queue()
    assembler = GrowingWindowAssembler(speech_queue=q, config=_config())
    assembler.process_segment(_segment(1, utterance_id=1))
    _ = q.get_nowait()
    with pytest.raises(ValueError):
        assembler.process_segment(_segment(2, utterance_id=2))
