# TDD Plan: Single-Queue Architecture with Dataclass Types

## Architecture Overview
```
AudioSource+VAD → PreliminarySegment
                ↓
                ├─→ chunk_queue.put(PreliminarySegment) [instant]
                └─→ windower.process_segment() [direct call]
                                ↓
                    chunk_queue.put(FinalizedWindow) [high quality]
                                ↓
                    Recognizer reads chunk_queue (ChunkQueueItem)
                                ↓
                    text_queue RecognitionResult
```

---

## Data Type Definitions

### New File: `src/types.py`

```python
"""Type definitions for STT pipeline queue messages."""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import numpy.typing as npt

@dataclass
class AudioSegment:
    """Speech segment for recognition - either preliminary word or finalized window."""
    type: Literal['preliminary', 'finalized']
    data: npt.NDArray[np.float32]
    start_time: float
    end_time: float
    chunk_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        # Validate: must have at least one chunk ID
        if len(self.chunk_ids) < 1:
            raise ValueError("Must have at least one chunk_id")

ChunkQueueItem = AudioSegment

@dataclass
class RecognitionResult:
    """Speech recognition output with timing and type."""
    text: str
    start_time: float
    end_time: float
    is_preliminary: bool
```

**Benefits:**
- `item.type == 'preliminary'` or `item.type == 'finalized'` checks type
- Single class simplifies code (no need for multiple isinstance checks)
- `__post_init__` validates chunk_ids field (at least one required)
- `chunk_ids`: preliminary = [single_id], finalized = [id1, id2, id3, ...]
- Immutable with `@dataclass(frozen=True)` if needed

---

## Phase 1: Write Tests (TDD)

### 0. Create Type Definitions
**New File:** `src/types.py`

Create dataclass definitions as shown above.

### 1.1 Update AdaptiveWindower Tests
**File:** `tests/test_adaptive_windower.py`

**New Test 1:** `test_emits_finalized_windows`
```python
from src.types import AudioSegment

def test_emits_finalized_windows(self, config, word_segments):
    """AdaptiveWindower should emit finalized AudioSegment instances."""
    chunk_queue = queue.Queue()
    windower = AdaptiveWindower(chunk_queue=chunk_queue, config=config)

    for segment in word_segments:
        windower.process_segment(segment)
    windower.flush()

    window = chunk_queue.get()

    # Type check replaces all field assertions
    assert isinstance(window, AudioSegment)
    assert window.type == 'finalized'
    assert len(window.chunk_ids) > 1  # Finalized windows aggregate multiple chunks
```

**Update existing 6 tests:**
- Change `window_queue` → `chunk_queue`
- Replace field assertions with `type` field checks
- Access fields with dot notation: `window.start_time` instead of `window['start_time']`

### 1.2 Update AudioSource Tests
**File:** `tests/test_audiosource_vad.py`

**New Test 2:** `test_emits_preliminary_segments`
```python
from src.types import AudioSegment

def test_emits_preliminary_segments(self, config, speech_with_pause_audio):
    """AudioSource should emit preliminary AudioSegment instances."""
    chunk_queue = queue.Queue()

    class MockWindower:
        def process_segment(self, segment): pass
        def flush(self): pass

    audio_source = AudioSource(
        chunk_queue=chunk_queue,
        windower=MockWindower(),
        config=config
    )

    chunk_size = int(config['audio']['sample_rate'] * config['audio']['chunk_duration'])
    for i in range(len(speech_with_pause_audio) // chunk_size):
        chunk = speech_with_pause_audio[i*chunk_size:(i+1)*chunk_size]
        audio_source.process_chunk_with_vad(chunk, i * config['audio']['chunk_duration'])

    audio_source.flush_vad()

    segments = []
    while not chunk_queue.empty():
        segments.append(chunk_queue.get())

    assert len(segments) >= 1
    for segment in segments:
        # Type check replaces all assertions
        assert isinstance(segment, AudioSegment)
        assert segment.type == 'preliminary'
        assert len(segment.chunk_ids) == 1  # Preliminary segments have single chunk ID
```

**New Test 3:** `test_calls_windower_for_each_segment`
```python
def test_calls_windower_for_each_segment(self, config, speech_with_pause_audio):
    """AudioSource should call windower.process_segment() for each segment."""
    chunk_queue = queue.Queue()

    class MockWindower:
        def __init__(self):
            self.segments_received = []

        def process_segment(self, segment):
            self.segments_received.append(segment)

        def flush(self):
            pass

    mock_windower = MockWindower()
    audio_source = AudioSource(
        chunk_queue=chunk_queue,
        windower=mock_windower,
        config=config
    )

    chunk_size = int(config['audio']['sample_rate'] * config['audio']['chunk_duration'])
    for i in range(len(speech_with_pause_audio) // chunk_size):
        chunk = speech_with_pause_audio[i*chunk_size:(i+1)*chunk_size]
        audio_source.process_chunk_with_vad(chunk, i * config['audio']['chunk_duration'])

    audio_source.flush_vad()

    # Verify all are preliminary AudioSegment instances
    assert len(mock_windower.segments_received) >= 1
    for seg in mock_windower.segments_received:
        assert isinstance(seg, AudioSegment)
        assert seg.type == 'preliminary'
        assert len(seg.chunk_ids) == 1
```

**New Test 4:** `test_calls_windower_flush_on_stop`
```python
def test_calls_windower_flush_on_stop(self, config):
    """AudioSource.stop() should call windower.flush()."""
    chunk_queue = queue.Queue()

    class MockWindower:
        def __init__(self):
            self.flush_called = False

        def process_segment(self, segment):
            pass

        def flush(self):
            self.flush_called = True

    mock_windower = MockWindower()
    audio_source = AudioSource(
        chunk_queue=chunk_queue,
        windower=mock_windower,
        config=config
    )

    audio_source.stop()
    assert mock_windower.flush_called
```

**Update existing 3 tests:** Add mock windower, use type field checks

### 1.3 Create Integration Tests
**New File:** `tests/test_audiosource_windower_integration.py`

```python
import pytest
import numpy as np
import queue
from src.AudioSource import AudioSource
from src.AdaptiveWindower import AdaptiveWindower
from src.types import AudioSegment
```

### 1.4 Create Recognizer Tests
**New File:** `tests/test_recognizer.py`

```python
import pytest
import numpy as np
import queue
from src.Recognizer import Recognizer
from src.types import AudioSegment, RecognitionResult
```

**Test 7:** `test_tags_preliminary_segments`
```python
def test_tags_preliminary_segments():
    """Recognizer should tag preliminary AudioSegment with is_preliminary=True."""
    chunk_queue = queue.Queue()
    text_queue = queue.Queue()

    class MockModel:
        def recognize(self, audio):
            return "word"

    recognizer = Recognizer(
        chunk_queue=chunk_queue,
        text_queue=text_queue,
        model=MockModel()
    )

    segment = AudioSegment(
        type='preliminary',
        data=np.random.randn(3200).astype(np.float32) * 0.1,
        start_time=0.0,
        end_time=0.2,
        chunk_ids=[1]
    )

    recognizer.recognize_window(segment, is_preliminary=True)

    result = text_queue.get()
    assert isinstance(result, RecognitionResult)
    assert result.text == "word"
    assert result.is_preliminary == True
    assert result.start_time == 0.0
    assert result.end_time == 0.2
```

**Test 8:** `test_tags_finalized_windows`
```python
def test_tags_finalized_windows():
    """Recognizer should tag finalized AudioSegment with is_preliminary=False."""
    chunk_queue = queue.Queue()
    text_queue = queue.Queue()

    class MockModel:
        def recognize(self, audio):
            return "complete sentence"

    recognizer = Recognizer(
        chunk_queue=chunk_queue,
        text_queue=text_queue,
        model=MockModel()
    )

    window = AudioSegment(
        type='finalized',
        data=np.random.randn(48000).astype(np.float32) * 0.1,
        start_time=0.0,
        end_time=3.0,
        chunk_ids=[1, 2, 3]  # Finalized windows have multiple chunk IDs
    )

    recognizer.recognize_window(window, is_preliminary=False)

    result = text_queue.get()
    assert isinstance(result, RecognitionResult)
    assert result.text == "complete sentence"
    assert result.is_preliminary == False
```

### 1.5 Create Integration Test
**New File:** `tests/test_integration_preliminary_finalized.py`

**Test 10:** `test_end_to_end_flow`
```python
import pytest
import numpy as np
import queue
from src.AudioSource import AudioSource
from src.AdaptiveWindower import AdaptiveWindower
from src.Recognizer import Recognizer
from src.types import AudioSegment, RecognitionResult

def test_end_to_end_flow(config, speech_audio):
    """End-to-end: preliminary AudioSegment → preliminary text, finalized → finalized text."""
    chunk_queue = queue.Queue()
    text_queue = queue.Queue()

    windower = AdaptiveWindower(chunk_queue=chunk_queue, config=config)
    audio_source = AudioSource(
        chunk_queue=chunk_queue,
        windower=windower,
        config=config
    )

    class MockModel:
        def recognize(self, audio):
            if len(audio) < 10000:
                return "word"
            else:
                return "word complete sentence"

    recognizer = Recognizer(
        chunk_queue=chunk_queue,
        text_queue=text_queue,
        model=MockModel()
    )

    # Process 3.2s
    chunk_size = int(config['audio']['sample_rate'] * config['audio']['chunk_duration'])
    num_chunks = int(3.2 / config['audio']['chunk_duration'])

    for i in range(num_chunks):
        idx = (i * chunk_size) % len(speech_audio)
        chunk = speech_audio[idx:idx + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        audio_source.process_chunk_with_vad(chunk, i * config['audio']['chunk_duration'])

    audio_source.flush_vad()

    # Process recognition using type field checks
    recognizer.is_running = True
    while not chunk_queue.empty():
        try:
            item = chunk_queue.get_nowait()
            is_preliminary = (item.type == 'preliminary')
            recognizer.recognize_window(item, is_preliminary=is_preliminary)
        except queue.Empty:
            break

    # Collect and verify results
    results = []
    while not text_queue.empty():
        result = text_queue.get()
        assert isinstance(result, RecognitionResult)
        results.append(result)

    preliminary_results = [r for r in results if r.is_preliminary == True]
    finalized_results = [r for r in results if r.is_preliminary == False]

    assert len(preliminary_results) >= 1
    assert len(finalized_results) >= 1
```

---

## Phase 2: Implement Code to Pass Tests

### 2.0 Create Type Definitions
**New File:** `src/types.py`

```python
"""Type definitions for STT pipeline queue messages."""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import numpy.typing as npt

@dataclass
class AudioSegment:
    """Speech segment for recognition - either preliminary word or finalized window."""
    type: Literal['preliminary', 'finalized']
    data: npt.NDArray[np.float32]
    start_time: float
    end_time: float
    chunk_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        # Validate: must have at least one chunk ID
        if len(self.chunk_ids) < 1:
            raise ValueError("Must have at least one chunk_id")

ChunkQueueItem = AudioSegment

@dataclass
class RecognitionResult:
    """Speech recognition output with timing and type."""
    text: str
    start_time: float
    end_time: float
    is_preliminary: bool
```

### 2.1 Update AdaptiveWindower
**File:** `src/AdaptiveWindower.py`

**Changes:**
- Line 1: Add: `from src.types import AudioSegment`
- Line 24: Rename: `window_queue` → `chunk_queue`
- Line 38: Update signature: `def process_segment(self, segment: AudioSegment) -> None:`
- Line 107-116: Create finalized AudioSegment:
  ```python
  # Collect unique chunk_ids from all segments in window
  chunk_ids = []
  for seg in window_segments:
      chunk_ids.extend(seg.chunk_ids)
  unique_chunk_ids = sorted(set(chunk_ids))

  window = AudioSegment(
      type='finalized',
      data=window_audio,
      start_time=actual_start,
      end_time=actual_end,
      chunk_ids=unique_chunk_ids
  )
  self.chunk_queue.put(window)
  ```
- Line 146-155: Create finalized AudioSegment in flush():
  ```python
  # Collect unique chunk_ids from all segments
  chunk_ids = []
  for seg in self.segments:
      chunk_ids.extend(seg.chunk_ids)
  unique_chunk_ids = sorted(set(chunk_ids))

  window = AudioSegment(
      type='finalized',
      data=window_audio,
      start_time=actual_start,
      end_time=actual_end,
      chunk_ids=unique_chunk_ids
  )
  self.chunk_queue.put(window)
  ```

### 2.2 Update AudioSource
**File:** `src/AudioSource.py`

**Changes:**
- Line 1: Add: `from __future__ import annotations`
- Line 6: Add: `from src.types import AudioSegment`
- Line 7: Add: `from typing import TYPE_CHECKING`
- Line 8: Add: `if TYPE_CHECKING: from src.AdaptiveWindower import AdaptiveWindower`
- Line 23: Add parameter: `windower: AdaptiveWindower | None = None`
- Line 30: Store: `self.windower = windower`
- Line 92-97: Create preliminary AudioSegment with chunk ID:
  ```python
  segments = self.vad.process_audio(audio_chunk, reset_state=False)
  for seg in segments:
      chunk_id = self.chunk_id_counter
      self.chunk_id_counter += 1

      segment = AudioSegment(
          type='preliminary',
          data=seg['data'],
          start_time=seg['start_time'],
          end_time=seg['end_time'],
          chunk_ids=[chunk_id]  # Preliminary segments have single chunk ID
      )
      self.chunk_queue.put(segment)

      if self.windower:
          self.windower.process_segment(segment)
  ```
- Line 112-125: Update stop():
  ```python
  self.is_running = False
  if self.stream:
      self.stream.stop()

  if self.windower:
      self.windower.flush()

  segments = self.vad.flush()
  for seg in segments:
      chunk_id = self.chunk_id_counter
      self.chunk_id_counter += 1

      segment = AudioSegment(
          type='preliminary',
          data=seg['data'],
          start_time=seg['start_time'],
          end_time=seg['end_time'],
          chunk_ids=[chunk_id]
      )
      self.chunk_queue.put(segment)
      if self.windower:
          self.windower.process_segment(segment)
  ```
- Line 127-132: Update flush_vad() similarly

### 2.3 Update Recognizer
**File:** `src/Recognizer.py`

**Changes:**
- Line 1: Add: `from src.types import ChunkQueueItem, RecognitionResult`
- Line 23: Rename: `window_queue` → `chunk_queue`
- Line 29: Update: `self.chunk_queue = chunk_queue`
- Line 41: Update signature: `def recognize_window(self, window_data: ChunkQueueItem, is_preliminary: bool = False) -> None:`
- Line 68-72: Create RecognitionResult:
  ```python
  text_data = RecognitionResult(
      text=text,
      start_time=window_data.start_time,
      end_time=window_data.end_time,
      is_preliminary=is_preliminary
  )
  self.text_queue.put(text_data)
  ```
- Line 84-86: Update process():
  ```python
  window_data: ChunkQueueItem = self.chunk_queue.get(timeout=0.1)
  is_preliminary = (window_data.type == 'preliminary')
  self.recognize_window(window_data, is_preliminary=is_preliminary)
  ```

### 2.4 Update Pipeline
**File:** `src/pipeline.py`

**Changes:**
- Line 4: Add: `from .AdaptiveWindower import AdaptiveWindower`
- Line 25: Remove: `self.window_queue`
- Line 38-43: Update components:
  ```python
  self.adaptive_windower = AdaptiveWindower(
      chunk_queue=self.chunk_queue,
      config=self.config
  )

  self.audio_source = AudioSource(
      chunk_queue=self.chunk_queue,
      windower=self.adaptive_windower,
      config=self.config
  )

  self.recognizer = Recognizer(
      chunk_queue=self.chunk_queue,
      text_queue=self.text_queue,
      model=self.model,
      verbose=verbose
  )
  ```
- Remove old Windower
- Add start/stop to AdaptiveWindower:
  ```python
  def start(self) -> None:
      pass  # No thread

  def stop(self) -> None:
      pass  # No thread
  ```

---

## Phase 3: Run Tests and Verify

1. `pytest tests/test_adaptive_windower.py -v` (7 tests)
2. `pytest tests/test_audiosource_vad.py -v` (6 tests)
3. `pytest tests/test_audiosource_windower_integration.py -v` (2 tests)
4. `pytest tests/test_recognizer.py -v` (3 tests)
5. `pytest tests/test_integration_preliminary_finalized.py -v` (1 test)

**Total: 19 tests**

---

## Summary

**Key Improvements:**

1. **Unified dataclass** replaces dict-based messages:
   - Single `AudioSegment` class with `type` discriminator
   - `RecognitionResult` class
   - Simpler than separate classes for preliminary/finalized

2. **Type field checks** replace manual assertions:
   - `item.type == 'preliminary'` → is preliminary
   - `item.type == 'finalized'` → is finalized
   - No need for multiple isinstance checks

3. **Type safety**:
   - `__post_init__` validates `chunk_ids` field (at least one required)
   - Type hints enable IDE/mypy checking
   - Dot notation: `item.start_time` not `item['start_time']`
   - Chunk ID tracking: preliminary = [single_id], finalized = [multiple_ids]

4. **Cleaner tests**:
   - Replace 5-line assertions with type field checks
   - Type system catches structural errors
   - More readable and maintainable

**Tests:** 10 new, 9 updated (remove dict assertions)
**New Files:** 5 test files + `src/types.py`
**Modified Files:** AdaptiveWindower, AudioSource, Recognizer, Pipeline
