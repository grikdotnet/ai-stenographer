# Plan: Fix flush() signal propagation using status discriminator

## Problem Statement

When `AudioSource.flush()` is called on silence timeout, finalized text hangs in TextMatcher's `previous_finalized_text` because `finalize_pending()` is not called.

**Current Flow:**
1. AudioSource detects silence timeout → calls `flush()`
2. AudioSource.flush() → calls `windower.flush()`
3. AdaptiveWindower.flush() → creates AudioSegment with `type='finalized'`, puts to queue
4. Recognizer → processes segment, creates RecognitionResult, puts to text_queue
5. TextMatcher.process_finalized() → receives text, stores in `previous_finalized_text`
6. **Problem:** TextMatcher doesn't know this is a flush event, so doesn't call `finalize_pending()`
7. **Result:** Pending text sits waiting for next window indefinitely

## Solution Design

Use type discriminators to propagate flush signal through the pipeline:

- **AudioSegment:** Extend `type` to include `'flush'` (currently `'preliminary' | 'finalized'`)
- **RecognitionResult:** Replace `is_preliminary: bool` with `status: Literal['preliminary', 'final', 'flush']`
- **Mapping:** Recognizer maps AudioSegment.type → RecognitionResult.status
- **Action:** TextMatcher checks status and calls `finalize_pending()` on flush

**Benefits:**
- Single discriminator field (cleaner than multiple boolean flags)
- Type-safe with Literal types
- Follows existing pattern (AudioSegment.type)
- Maintains queue-based architecture

---

## TDD Implementation Steps

### Phase 1: Write Tests (will fail initially)

#### 1.1 Test: AdaptiveWindower.flush() creates type='flush'
**File:** `tests/test_adaptive_windower.py`
**Action:** Update existing test
```python
def test_flush_emits_partial_window(self, config, word_segments):
    # ... existing setup ...

    window = chunk_queue.get()
    assert isinstance(window, AudioSegment)
    assert window.type == 'flush'  # UPDATE: was 'finalized'
    # ... rest of assertions ...
```

#### 1.2 Test: Recognizer maps AudioSegment.type → RecognitionResult.status
**File:** `tests/test_recognizer.py`
**Action:** Add new test
```python
def test_recognizer_maps_segment_types_to_status():
    """Recognizer should map AudioSegment.type to RecognitionResult.status"""
    # Test mapping:
    # type='preliminary' → status='preliminary'
    # type='finalized' → status='final'
    # type='flush' → status='flush'
```

#### 1.3 Test: TextMatcher calls finalize_pending() on flush
**File:** `tests/test_text_matcher.py`
**Action:** Update all tests + add new test

**Updates needed:**
- Replace all `is_preliminary=True/False` with `status='preliminary'/'final'`
- Update assertions that check `is_preliminary`

**New test:**
```python
def test_process_flush_finalizes_pending():
    """Flush result should trigger finalize_pending()"""
    text_matcher = TextMatcher(self.text_queue, self.mock_gui)

    # 1. Process final text (gets stored in previous_finalized_text)
    final_text = RecognitionResult('hello world', 1.0, 1.5, status='final', chunk_ids=[1, 2])
    text_matcher.process_text(final_text)

    # Verify stored but not finalized
    assert text_matcher.previous_finalized_text == 'hello world'
    self.mock_gui.finalize_text.assert_not_called()

    # 2. Process flush result
    flush_text = RecognitionResult('', 2.0, 2.0, status='flush', chunk_ids=[])
    text_matcher.process_text(flush_text)

    # Verify finalize_pending() was triggered
    self.mock_gui.finalize_text.assert_called_once()
    finalized = self.mock_gui.finalize_text.call_args[0][0]
    assert finalized.text == 'hello world'
    assert text_matcher.previous_finalized_text == ''
```

---

### Phase 2: Update Type Definitions

#### 2.1 Update AudioSegment.type
**File:** [src/types.py:23](src/types.py#L23)
```python
# Before:
type: Literal['preliminary', 'finalized']

# After:
type: Literal['preliminary', 'finalized', 'flush']
```

#### 2.2 Update RecognitionResult
**File:** [src/types.py:39-54](src/types.py#L39-54)
```python
# Before:
@dataclass
class RecognitionResult:
    text: str
    start_time: float
    end_time: float
    is_preliminary: bool
    chunk_ids: list[int] = field(default_factory=list)

# After:
@dataclass
class RecognitionResult:
    text: str
    start_time: float
    end_time: float
    status: Literal['preliminary', 'final', 'flush']
    chunk_ids: list[int] = field(default_factory=list)
```

---

### Phase 3: Update AdaptiveWindower

#### 3.1 Update flush() method
**File:** [src/AdaptiveWindower.py:161](src/AdaptiveWindower.py#L161)
```python
# Before:
window = AudioSegment(
    type='finalized',
    data=window_audio,
    start_time=actual_start,
    end_time=actual_end,
    chunk_ids=unique_chunk_ids
)

# After:
window = AudioSegment(
    type='flush',  # CHANGE: was 'finalized'
    data=window_audio,
    start_time=actual_start,
    end_time=actual_end,
    chunk_ids=unique_chunk_ids
)
```

---

### Phase 4: Update Recognizer

#### 4.1 Update recognize_window()
**File:** [src/Recognizer.py:33-64](src/Recognizer.py#L33-64)

**Current signature:**
```python
def recognize_window(self, window_data: ChunkQueueItem, is_preliminary: bool = False) -> Optional[RecognitionResult]:
```

**New signature:**
```python
def recognize_window(self, window_data: ChunkQueueItem) -> Optional[RecognitionResult]:
    """Recognize audio and emit RecognitionResult.

    Maps AudioSegment.type to RecognitionResult.status:
    - 'preliminary' → 'preliminary'
    - 'finalized' → 'final'
    - 'flush' → 'flush'
    """
    audio: np.ndarray = window_data.data

    # Map AudioSegment.type to RecognitionResult.status
    status_map = {
        'preliminary': 'preliminary',
        'finalized': 'final',
        'flush': 'flush'
    }
    status = status_map[window_data.type]

    # Recognize
    text: str = self.model.recognize(audio)

    if self.verbose:
        print(f"recognize(): chunk_ids={window_data.chunk_ids}")
        print(f"recognize(): text: '{text}'")

    if not text or not text.strip():
        return None

    # Create RecognitionResult with mapped status
    result = RecognitionResult(
        text=text,
        start_time=window_data.start_time,
        end_time=window_data.end_time,
        status=status,
        chunk_ids=window_data.chunk_ids
    )
    self.text_queue.put(result)
    return result
```

#### 4.2 Update process() method
**File:** [src/Recognizer.py:73-80](src/Recognizer.py#L73-80)
```python
# Before:
while self.is_running:
    try:
        window_data: ChunkQueueItem = self.chunk_queue.get(timeout=0.1)
        is_preliminary: bool = (window_data.type == 'preliminary')
        self.recognize_window(window_data, is_preliminary=is_preliminary)
    except queue.Empty:
        continue

# After:
while self.is_running:
    try:
        window_data: ChunkQueueItem = self.chunk_queue.get(timeout=0.1)
        self.recognize_window(window_data)  # No is_preliminary parameter
    except queue.Empty:
        continue
```

---

### Phase 5: Update TextMatcher

#### 5.1 Update process_text() routing
**File:** [src/TextMatcher.py:124-136](src/TextMatcher.py#L124-136)
```python
# Before:
def process_text(self, result: RecognitionResult) -> None:
    if result.is_preliminary:
        self.process_preliminary(result)
    else:
        self.process_finalized(result)

# After:
def process_text(self, result: RecognitionResult) -> None:
    """Process a single text segment using appropriate path.

    Routes based on status:
    - 'preliminary': Pass through to GUI directly
    - 'final': Overlap resolution, then to GUI
    - 'flush': Trigger finalize_pending() to flush buffered text
    """
    if result.status == 'preliminary':
        self.process_preliminary(result)
    elif result.status == 'final':
        self.process_finalized(result)
    elif result.status == 'flush':
        self.process_flush(result)
```

#### 5.2 Add process_flush() method
**File:** [src/TextMatcher.py](src/TextMatcher.py) (add after process_finalized)
```python
def process_flush(self, result: RecognitionResult) -> None:
    """Process flush signal by finalizing any pending text.

    Flush signals indicate end of speech segment (e.g., silence timeout).
    Any text waiting in previous_finalized_text should be finalized now.

    Args:
        result: RecognitionResult with status='flush'
    """
    if self.verbose:
        print(f"TextMatcher.process_flush() received, pending='{self.previous_finalized_text}'")

    # Finalize any pending text
    self.finalize_pending()
```

#### 5.3 Update state variable comments
**File:** [src/TextMatcher.py:44-45](src/TextMatcher.py#L44-45)
```python
# Before:
# State only needed for finalized path (overlap resolution)
# Preliminary path has no state - VAD guarantees distinct segments

# After:
# State only needed for final/flush paths (overlap resolution)
# Preliminary path has no state - VAD guarantees distinct segments
```

---

### Phase 6: Check Other Files

#### 6.1 Check GuiWindow
**File:** `src/GuiWindow.py`
**Action:** Search for `is_preliminary` references and update if needed

#### 6.2 Check integration tests
**File:** `tests/test_integration_preliminary_finalized.py`
**Action:** Update to use `status` instead of `is_preliminary`

---

### Phase 7: Run Tests

```bash
python -m pytest tests/ -v
```

**Expected:**
- All existing tests pass with updated status field
- New flush test passes
- AdaptiveWindower flush test passes with type='flush'

---

## Edge Cases to Consider

1. **Empty flush:** What if flush is called with no pending text?
   - `finalize_pending()` already handles this (checks `if self.previous_finalized_text`)

2. **Flush with text:** What if flush AudioSegment has actual recognized text?
   - Current plan ignores flush text, just triggers finalize_pending()
   - Alternative: Process flush text first, then finalize_pending()

3. **Multiple flushes:** What if multiple flush signals arrive?
   - `finalize_pending()` is idempotent (clears state after finalizing)

4. **Flush during preliminary:** Can flush interrupt preliminary flow?
   - Preliminary path is stateless, so flush only affects finalized path

5. **Type safety:** What if unknown status value?
   - Literal types provide compile-time safety
   - Could add runtime validation in process_text()

---

## Testing Strategy

1. **Unit tests:** Each component maps types/status correctly
2. **Integration test:** End-to-end flush signal propagation
3. **Manual test:** Run with `-v`, trigger silence timeout, verify pending text appears

---

## Rollback Plan

If issues arise:
1. Revert type changes (restore `is_preliminary: bool`)
2. Revert AdaptiveWindower.flush() to `type='finalized'`
3. Git commit history provides clean rollback point
