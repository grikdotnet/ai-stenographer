# Implementation Plan: Context-Aware Speech Recognition with Token Filtering

## Overview
Implement a three-phase refactoring to add left/right audio context for better STT quality while filtering hallucinated tokens based on timestamps.

---

## Task 1: AudioSegment Refactoring for Context Support

### 1.1 Update AudioSegment Dataclass
**File:** `src/types.py`

**Changes:**
```python
@dataclass
class AudioSegment:
    type: Literal['preliminary', 'finalized', 'flush']
    data: npt.NDArray[np.float32]           # Speech audio (concatenated)
    left_context: npt.NDArray[np.float32]   # NEW: Pre-speech context
    right_context: npt.NDArray[np.float32]  # NEW: Post-speech context
    start_time: float                        # Start of data (not context)
    end_time: float                          # End of data (not context)
    chunk_ids: list[int] = field(default_factory=list)
```

**Decision:** Keep as concatenated numpy arrays (NOT list of chunks) because:
- Simpler implementation
- Model needs concatenated array anyway
- Memory savings from chunk reuse are minimal (~50KB max)
- No need for complex deque indexing logic

### 1.2 Update SoundPreProcessor
**File:** `src/SoundPreProcessor.py`

**Changes:**
1. **Remove `last_silence_chunk`** property and all related logic
2. **Add circular buffer** (48 chunks = 1.536s @ 32ms/chunk):
   ```python
   CONTEXT_BUFFER_SIZE = 48  # Class constant
   self.context_buffer: deque = deque(maxlen=48)
   ```
3. **Update `_process_chunk()`**: Always append to context_buffer (speech or silence)
4. **Update `_finalize_segment()`**:
   ```python
   # Extract left context from buffer (everything before speech_start_index)
   left_context_chunks = [buffer items before speech start]
   left_context = np.concatenate(left_context_chunks) if left_context_chunks else np.array([], dtype=np.float32)

   # Extract right context (accumulated silence chunks after speech)
   right_context_chunks = [silence chunks at end of speech_buffer]
   right_context = np.concatenate(right_context_chunks) if right_context_chunks else np.array([], dtype=np.float32)

   # Extract data (only speech chunks)
   data_chunks = [speech chunks only]
   data = np.concatenate(data_chunks)

   segment = AudioSegment(
       type='preliminary',
       data=data,
       left_context=left_context,
       right_context=right_context,
       start_time=speech_start_time,
       end_time=end_time,
       chunk_ids=chunk_ids
   )
   ```

### 1.3 Update AdaptiveWindower
**File:** `src/AdaptiveWindower.py`

**Changes:**
- Update `_emit_window()` and `flush()` to pass empty contexts:
  ```python
  window = AudioSegment(
      type='finalized',  # or 'flush'
      data=window_audio,
      left_context=np.array([], dtype=np.float32),   # No context for windows
      right_context=np.array([], dtype=np.float32),  # No context for windows
      start_time=start_time,
      end_time=end_time,
      chunk_ids=chunk_ids
  )
  ```

**Rationale:** Finalized windows already have proper context from preliminary segments - no additional context needed.

---

## Task 2: Timestamped Recognition

### 2.1 Update Model Loading
**File:** `src/pipeline.py`

**Change line 52:**
```python
# Before:
self.model: TextResultsAsrAdapter = onnx_asr.load_model(...)

# After:
base_model = onnx_asr.load_model(...)
self.model: TimestampedResultsAsrAdapter = base_model.with_timestamps()
```

**Add import:**
```python
from onnx_asr.adapters import TimestampedResultsAsrAdapter
from onnx_asr.asr import TimestampedResult
```

### 2.2 Update Recognizer
**File:** `src/Recognizer.py`

**Changes to `__init__()`:**
```python
def __init__(self,
             speech_queue: queue.Queue,
             text_queue: queue.Queue,
             model: TimestampedResultsAsrAdapter,  # Type changed
             sample_rate: int = 16000,  # NEW parameter
             verbose: bool = False) -> None:
    # ... existing fields ...
    self.sample_rate = sample_rate
```

**Changes to `recognize_window()`:**
```python
def recognize_window(self, window_data: ChunkQueueItem) -> Optional[RecognitionResult]:
    # Concatenate context + data + context for recognition
    audio_parts = []
    if window_data.left_context.size > 0:
        audio_parts.append(window_data.left_context)
    audio_parts.append(window_data.data)
    if window_data.right_context.size > 0:
        audio_parts.append(window_data.right_context)

    full_audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else window_data.data

    # Recognize with timestamps
    result: TimestampedResult = self.model.recognize(full_audio)

    if not result.text or not result.text.strip():
        return None

    # Calculate context boundaries in seconds
    left_context_duration = len(window_data.left_context) / self.sample_rate
    data_duration = len(window_data.data) / self.sample_rate
    data_start = left_context_duration
    data_end = left_context_duration + data_duration

    # Filter tokens within data region
    filtered_text = self._filter_tokens_by_timestamp(
        result.text,
        result.tokens,
        result.timestamps,
        data_start,
        data_end
    )

    if not filtered_text or not filtered_text.strip():
        return None

    # Create result with filtered text, original timing
    status = {'preliminary': 'preliminary', 'finalized': 'final', 'flush': 'flush'}[window_data.type]

    return RecognitionResult(
        text=filtered_text,
        start_time=window_data.start_time,
        end_time=window_data.end_time,
        status=status,
        chunk_ids=window_data.chunk_ids
    )
```

**New private method:**
```python
def _filter_tokens_by_timestamp(self,
                                 text: str,
                                 tokens: list[str] | None,
                                 timestamps: list[float] | None,
                                 data_start: float,
                                 data_end: float) -> str:
    """Filter tokens to only those within data region (exclude context).

    Args:
        text: Full recognized text
        tokens: Token list from TimestampedResult (subword units with � prefix)
        timestamps: Timestamp list (in seconds, relative to audio start)
        data_start: Start of data region in seconds
        data_end: End of data region in seconds

    Returns:
        Filtered text containing only tokens from data region
    """
    if not tokens or not timestamps:
        # No timestamps available - return full text (fallback)
        return text

    # Filter tokens within data boundaries
    filtered_tokens = []
    for token, ts in zip(tokens, timestamps):
        if data_start <= ts <= data_end:
            filtered_tokens.append(token)

    if not filtered_tokens:
        return ""

    # Reconstruct text from filtered tokens
    # Tokens use � (U+2581) for word boundaries
    reconstructed = ''.join(filtered_tokens)
    reconstructed = reconstructed.replace('�', ' ')
    reconstructed = reconstructed.strip()

    return reconstructed
```

---

## Task 3: Context Buffer Implementation

### 3.1 SoundPreProcessor Circular Buffer
**File:** `src/SoundPreProcessor.py`

**Implementation details:**

```python
class SoundPreProcessor:
    CONTEXT_BUFFER_SIZE = 48  # 1.536s @ 32ms/chunk

    def __init__(self, ...):
        # Remove: self.last_silence_chunk
        self.context_buffer: deque = deque(maxlen=self.CONTEXT_BUFFER_SIZE)
        self.speech_start_index: int | None = None  # Index in buffer when speech started

    def _handle_speech_frame(self, audio_chunk, timestamp, chunk_id):
        # Add to circular buffer
        self.context_buffer.append({
            'audio': audio_chunk,
            'timestamp': timestamp,
            'chunk_id': chunk_id,
            'is_speech': True
        })

        if not self.is_speech_active:
            # Mark where speech started in buffer
            self.speech_start_index = len(self.context_buffer) - 1
            self.is_speech_active = True
            self.speech_start_time = timestamp

        # Add to speech buffer (for data extraction)
        self.speech_buffer.append({
            'audio': audio_chunk,
            'timestamp': timestamp,
            'chunk_id': chunk_id
        })

        # Max duration check (unchanged)
        if (timestamp - self.speech_start_time) * 1000 >= self.max_speech_duration_ms:
            self._finalize_segment()

    def _handle_silence_frame(self, audio_chunk, timestamp, chunk_id):
        # Always add to circular buffer
        self.context_buffer.append({
            'audio': audio_chunk,
            'timestamp': timestamp,
            'chunk_id': chunk_id,
            'is_speech': False
        })

        if self.is_speech_active:
            # Accumulate trailing silence in speech_buffer
            self.speech_buffer.append({
                'audio': audio_chunk,
                'timestamp': timestamp
                # No chunk_id for silence
            })

            # Check silence energy threshold (unchanged)
            speech_prob = self.vad.process_frame(normalized_audio)
            self.silence_energy += (1.0 - speech_prob)

            if self.silence_energy >= self.silence_energy_threshold:
                self._finalize_segment()

    def _finalize_segment(self):
        if not self.speech_buffer:
            return

        # Extract left context from circular buffer
        left_context_chunks = []
        if self.speech_start_index is not None:
            # Get all chunks before speech start (available in buffer)
            buffer_list = list(self.context_buffer)
            for i in range(self.speech_start_index):
                if i < len(buffer_list):
                    left_context_chunks.append(buffer_list[i]['audio'])

        left_context = (np.concatenate(left_context_chunks)
                       if left_context_chunks
                       else np.array([], dtype=np.float32))

        # Extract data (speech chunks only from speech_buffer)
        data_chunks = [chunk['audio'] for chunk in self.speech_buffer
                      if 'chunk_id' in chunk]
        data = np.concatenate(data_chunks)

        # Extract right context (trailing silence from speech_buffer)
        right_context_chunks = [chunk['audio'] for chunk in self.speech_buffer
                               if 'chunk_id' not in chunk]
        right_context = (np.concatenate(right_context_chunks)
                        if right_context_chunks
                        else np.array([], dtype=np.float32))

        # Extract chunk_ids (speech only)
        chunk_ids = [chunk['chunk_id'] for chunk in self.speech_buffer
                    if 'chunk_id' in chunk]

        # Create segment with contexts
        segment = AudioSegment(
            type='preliminary',
            data=data,
            left_context=left_context,
            right_context=right_context,
            start_time=self.speech_start_time,
            end_time=end_time,
            chunk_ids=chunk_ids
        )

        self.speech_queue.put(segment)

        # Reset state
        self.speech_buffer.clear()
        self.is_speech_active = False
        self.speech_start_index = None
        self.silence_energy = 0.0

        # Call windower (unchanged)
        self.windower.process_segment(segment)
```

---

## Test Strategy

### Phase 1: Task 1 Tests (AudioSegment + Context Buffer)

**New Tests:**
1. `tests/test_sound_preprocessor.py`:
   - `test_circular_buffer_initialization()` - Verify buffer created with size 48
   - `test_context_buffer_fills_before_speech()` - Verify buffer accumulates silence before speech
   - `test_left_context_extraction()` - Verify left context extracted from buffer
   - `test_right_context_extraction()` - Verify right context from trailing silence
   - `test_early_speech_small_left_context()` - Verify handling when speech starts within first 18 chunks
   - `test_no_left_context_at_startup()` - Verify empty left context when speech starts immediately

**Tests to Remove:**
2. `tests/test_sound_preprocessor.py`:
   - `test_last_silence_chunk_prepended()` (lines 622-721) - Feature removed
   - `test_last_silence_chunk_cleared()` - Feature removed
   - Any other tests checking `last_silence_chunk` property

**Tests to Update:**
3. `tests/test_sound_preprocessor.py`:
   - All tests creating AudioSegment must add `left_context` and `right_context` fields
   - Update assertions to check context fields

4. `tests/test_adaptive_windower.py`:
   - Update all AudioSegment creations to include empty contexts
   - Verify finalized windows have empty contexts

5. `tests/test_recognizer.py`:
   - Update all AudioSegment creations to include context fields

### Phase 2: Task 2 Tests (Timestamped Recognition)

**New Tests:**
6. `tests/test_recognizer.py`:
   - `test_recognize_with_timestamps()` - Verify model returns TimestampedResult
   - `test_filter_tokens_by_timestamp()` - Unit test for token filtering
   - `test_filter_tokens_all_in_range()` - All tokens within data region
   - `test_filter_tokens_partial_overlap()` - Some tokens in context, some in data
   - `test_filter_tokens_none_in_range()` - All tokens in context (return empty)
   - `test_filter_tokens_no_timestamps()` - Fallback when timestamps unavailable
   - `test_context_concatenation()` - Verify left+data+right concatenated correctly

**Tests to Update:**
7. `tests/test_recognizer.py`:
   - Update mock model to return TimestampedResult instead of str
   - Update 3 existing tests for new signature

### Phase 3: Integration Tests

**New Tests:**
8. `tests/test_integration_context_filtering.py`:
   - `test_end_to_end_with_context()` - Full pipeline with real audio
   - `test_hallucination_filtering()` - Verify context prevents "yeah"/"mm-hmm" hallucinations
   - `test_short_word_with_context()` - Verify 50-128ms words recognized correctly

---

## Implementation Order (TDD: Tests First!)

### Phase 1: AudioSegment + Context Buffer

1. **Phase 1a:** Update AudioSegment dataclass (add context fields) - *minimal change to enable tests*
2. **Phase 1b:** Update AdaptiveWindower (pass empty contexts) - *minimal change to enable tests*
3. **Phase 1c:** Update all existing tests to use new AudioSegment fields (left_context, right_context)
4. **Phase 1d:** Remove obsolete tests (last_silence_chunk related)
5. **Phase 1e:** Write NEW Phase 1 tests (context buffer) - *tests will FAIL*
6. **Phase 1f:** Implement circular buffer in SoundPreProcessor - *make tests PASS*
7. **Phase 1g:** Run Phase 1 tests, verify all GREEN

### Phase 2: Timestamped Recognition

8. **Phase 2a:** Write Phase 2 tests (token filtering) - *tests will FAIL*
9. **Phase 2b:** Update pipeline.py model loading (with_timestamps())
10. **Phase 2c:** Implement token filtering in Recognizer - *make tests PASS*
11. **Phase 2d:** Run Phase 2 tests, verify all GREEN

### Phase 3: Integration & Validation

12. **Phase 3a:** Write integration tests - *tests will FAIL*
13. **Phase 3b:** Fix any integration issues - *make tests PASS*
14. **Phase 3c:** Run FULL test suite, verify all GREEN
15. **Phase 3d:** Manual testing with real audio

---

## Design Constants (No Config Changes)

```python
# In SoundPreProcessor
CONTEXT_BUFFER_SIZE = 48  # chunks (1.536s @ 32ms/chunk)

# In Recognizer
# sample_rate passed from Pipeline (16000)
```

---

## Risks and Mitigations

### Risk 1: Token filtering too aggressive
- **Mitigation:** Log filtered vs original text in verbose mode
- **Fallback:** Return full text if no tokens in range

### Risk 2: 80ms timestamp resolution insufficient
- **Mitigation:** Use timestamp boundaries with tolerance (�40ms)
- **Monitor:** Track filtering accuracy in tests

### Risk 3: Circular buffer index tracking complex
- **Mitigation:** Thorough unit tests for edge cases
- **Simplification:** Convert deque to list for indexing when needed

### Risk 4: Memory increase from context storage
- **Impact:** ~50KB per segment (1.5s @ 16kHz = 24KB per context)
- **Acceptable:** Small compared to model memory (600MB+)

---

## Success Criteria

1.  All existing tests pass with updated AudioSegment
2.  Context buffer correctly extracts left/right contexts
3.  Token filtering removes context-related tokens
4.  No regression in recognition quality for normal speech
5.  Improved quality for short words (50-128ms) - measured manually
6.  No hallucinations ("yeah", "mm-hmm") on silence - verified in tests

---

## Validation Plan

After implementation:
1. Run full test suite: `python -m pytest tests/ -v`
2. Manual test with verbose mode: `python main.py -v`
3. Speak short words ("yes", "no", "hi") and verify recognition
4. Test with silence periods - verify no hallucinations
5. Check logs for context sizes and filtered tokens

