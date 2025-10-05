# TDD Plan: Remove Silence Detection from Recognizer (SRP Fix)

## Problem Analysis

### Current SRP Violation
- **Recognizer** should only do: Audio → Text recognition
- **Current violation:** Recognizer tracks silence timing and tells TextMatcher when to finalize
- **Root cause:** Silence signal generation is not related to voice recognition

### Solution: Separation of Concerns

**TextMatcher** manages its own finalization:
- Pure, synchronous text processing
- No threads, no timeouts in TextMatcher
- Exposes `finalize_pending()` method for explicit finalization

**Pipeline** orchestrates finalization:
- Calls `text_matcher.finalize_pending()` in `stop()`

**Future: SilenceObserver** (separate component, added later):
- Monitors VAD activity for phrase-level silence
- Triggers finalization when needed
- Observer pattern: extends behavior without modifying existing classes

## SOLID Compliance

✅ **SRP:** Each class has one reason to change
- Recognizer: recognition logic changes
- TextMatcher: text processing logic changes
- Future SilenceObserver: silence detection logic changes

✅ **OCP:** Closed to modification, open to extension
- TextMatcher doesn't change when adding timeout
- Add SilenceObserver as separate component

✅ **DIP:** TextMatcher.finalize_pending() is interface for external triggers

---

## Phase 1: Update Tests (TDD First)

### 1.1 Update Recognizer Tests
**File:** `tests/test_recognizer.py`

**Remove 2 tests:**
- `test_silence_timeout_signal` (lines 153-207)
- `test_silence_flag_reset_on_new_speech` (lines 209-243)

**Keep 6 tests unchanged:**
- `test_recognizes_preliminary_segments`
- `test_recognizes_finalized_segments`
- `test_filters_empty_text`
- `test_process_method_with_type_detection`
- `test_preserves_timing_information`
- (1 more existing test)

### 1.2 Update TextMatcher Tests
**File:** `tests/test_text_matcher.py`

**Remove 4 silence-related tests (lines 269-371):**
- `test_silence_window_finalizes_partial_text`
- `test_speech_after_silence_starts_new_phrase`
- `test_multiple_speech_then_silence_pattern`
- `test_silence_only_no_crash`

**Add 2 new tests:**

**Test 1: `test_finalize_pending_method`**
```python
def test_finalize_pending_method(self):
    """Test explicit finalize_pending() call finalizes partial text"""
    text_matcher = TextMatcher(
        self.text_queue,
        self.final_queue,
        self.partial_queue
    )

    # Process some speech
    speech_text = {'text': 'hello world', 'timestamp': 1.0}
    text_matcher.process_text(speech_text)

    # Verify it went to partial queue only
    self.assertTrue(self.final_queue.empty())
    self.assertFalse(self.partial_queue.empty())
    partial = self.partial_queue.get()
    self.assertEqual(partial['text'], 'hello world')

    # Explicitly finalize
    text_matcher.finalize_pending()

    # Verify finalized
    self.assertFalse(self.final_queue.empty())
    final = self.final_queue.get()
    self.assertEqual(final['text'], 'hello world')

    # State should be reset
    self.assertEqual(text_matcher.previous_text, "")
```

**Test 2: `test_finalize_pending_when_empty`**
```python
def test_finalize_pending_when_empty(self):
    """Test finalize_pending() with no pending text does nothing"""
    text_matcher = TextMatcher(
        self.text_queue,
        self.final_queue,
        self.partial_queue
    )

    # Call finalize with no pending text
    text_matcher.finalize_pending()

    # Should not crash, queues remain empty
    self.assertTrue(self.final_queue.empty())
    self.assertTrue(self.partial_queue.empty())
```

**Keep unchanged:**
- All `resolve_overlap` tests (9 tests)
- `test_process_first_text`
- `test_process_texts_with_overlap`

---

## Phase 2: Implement Code (Make Tests Pass)

### 2.1 Update Recognizer
**File:** `src/Recognizer.py`

**Remove:**
- Line ~35: `silence_timeout` parameter from `__init__`
- Line ~40: `self.silence_timeout = silence_timeout`
- Line ~42: `self._silence_signal_sent = False`
- Line ~43: `self.last_speech_time = 0`
- Lines ~70-80: Silence signal emission logic in `recognize_window()`
- Lines ~90-110: Silence timeout check in `process()` loop

**Result:** Clean recognition-only class
```python
class Recognizer:
    """Pure speech recognition: AudioSegment → RecognitionResult

    Single Responsibility: Convert audio to text using STT model.
    Does not handle silence detection or text finalization logic.
    """

    def __init__(self, chunk_queue: queue.Queue,
                 text_queue: queue.Queue,
                 model,
                 verbose: bool = False) -> None:
        self.chunk_queue: queue.Queue = chunk_queue
        self.text_queue: queue.Queue = text_queue
        self.model = model
        self.verbose: bool = verbose
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None

    def recognize_window(self, window_data: ChunkQueueItem,
                        is_preliminary: bool = False) -> Optional[RecognitionResult]:
        """Recognize audio and emit RecognitionResult.

        Args:
            window_data: AudioSegment to recognize
            is_preliminary: True for instant results, False for high-quality

        Returns:
            RecognitionResult if text recognized, None if empty/silence
        """
        if self.verbose:
            print(f"recognize(): processing {window_data.type} segment")

        text: str = self.model.recognize(window_data.data)

        if not text or not text.strip():
            return None

        result = RecognitionResult(
            text=text,
            start_time=window_data.start_time,
            end_time=window_data.end_time,
            is_preliminary=is_preliminary
        )
        self.text_queue.put(result)
        return result

    def process(self) -> None:
        """Process AudioSegments from queue continuously."""
        while self.is_running:
            try:
                window_data: ChunkQueueItem = self.chunk_queue.get(timeout=0.1)
                is_preliminary: bool = (window_data.type == 'preliminary')
                self.recognize_window(window_data, is_preliminary=is_preliminary)
            except queue.Empty:
                continue

    def start(self) -> None:
        """Start recognition in background thread."""
        self.is_running = True
        self.thread = threading.Thread(target=self.process, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop recognition thread."""
        self.is_running = False
```

### 2.2 Update TextMatcher
**File:** `src/TextMatcher.py`

**Remove `is_silence` handling from `process_text()`:**

Delete lines 138-151:
```python
# REMOVE THIS BLOCK:
# Handle silence signal - finalize any remaining partial text
if is_silence:
    if self.previous_text:
        final_data: Dict[str, Union[str, float]] = {
            'text': self.previous_text,
            'timestamp': text_data['timestamp']
        }
        if self.verbose:
            print(f"sending to final_queue: {final_data}")
        self.final_queue.put(final_data)
    # Reset state for new phrase after silence
    self.previous_text = ""
    self.previous_text_timestamp = 0.0
    return
```

**Add `finalize_pending()` method (after `process_text`):**
```python
def finalize_pending(self) -> None:
    """Finalize any remaining partial text.

    Called explicitly by Pipeline.stop() or external observers
    to flush pending partial text to final queue.

    This implements explicit finalization control, allowing
    external components to trigger finalization without
    TextMatcher managing its own timeout logic (SRP).
    """
    if self.previous_text:
        final_data: Dict[str, Union[str, float]] = {
            'text': self.previous_text,
            'timestamp': self.previous_text_timestamp
        }
        if self.verbose:
            print(f"TextMatcher.finalize_pending(): {final_data}")
        self.final_queue.put(final_data)
        self.previous_text = ""
        self.previous_text_timestamp = 0.0
```

**Update docstring for `process_text():`**
```python
def process_text(self, text_data: Dict[str, Union[str, float, bool]]) -> None:
    """Process a single text segment using window-based overlap resolution.

    Uses the resolve_overlap algorithm to properly handle STT window overlaps
    by finding reliable common subsequences and finalizing the reliable parts
    while discarding boundary errors.

    Note: Does not handle silence signals. Use finalize_pending() to
    explicitly finalize remaining partial text.

    Args:
        text_data: Dictionary containing 'text' and 'timestamp'
    """
```

### 2.3 Update Pipeline
**File:** `src/pipeline.py`

**Remove `silence_timeout` from Recognizer initialization:**

Find and remove the `silence_timeout` parameter:
```python
self.recognizer = Recognizer(
    chunk_queue=self.chunk_queue,
    text_queue=self.text_queue,
    model=self.model,
    verbose=verbose
    # REMOVE: silence_timeout=2.0
)
```

**Update `stop()` method to call `finalize_pending()`:**

```python
def stop(self) -> None:
    """Stop all pipeline components.

    Stops components in proper order:
    1. AudioSource (stop capturing audio)
    2. Recognizer (stop processing audio)
    3. TextMatcher finalization (flush pending text)
    4. TextMatcher (stop processing)
    5. DisplayHandler (stop displaying)
    """
    if self.verbose:
        print("\nStopping pipeline...")

    # Stop audio capture and recognition
    self.audio_source.stop()
    self.recognizer.stop()

    # Finalize any pending partial text before stopping TextMatcher
    if self.verbose:
        print("Finalizing pending text...")
    self.text_matcher.finalize_pending()

    # Stop text processing and display
    self.text_matcher.stop()
    self.display_handler.stop()

    if self.verbose:
        print("Pipeline stopped.")
```

---

## Phase 3: Run Tests and Verify

### 3.1 Run Individual Test Files
```bash
# 1. Recognizer tests (6 remaining tests)
source venv/Scripts/activate
python -m pytest tests/test_recognizer.py -v

# 2. TextMatcher tests (updated tests)
python -m pytest tests/test_text_matcher.py -v
```

### 3.2 Run All Tests
```bash
# Run entire test suite
python -m pytest tests/ -v
```

### 3.3 Manual Test
```bash
# Test real-time pipeline
python main.py -v

# Speak a phrase and stop - verify final text appears
# (finalize_pending() should be called in Pipeline.stop())
```

---

## Summary

### Changes Made

**Removed:**
- 2 Recognizer tests (silence-related)
- 4 TextMatcher tests (silence-related)
- ~60 lines of silence logic from Recognizer
- `is_silence` handling from TextMatcher.process_text()

**Added:**
- 2 TextMatcher tests (finalize_pending)
- `finalize_pending()` method in TextMatcher
- Call to `finalize_pending()` in Pipeline.stop()

### SOLID Compliance

✅ **Single Responsibility Principle:**
- Recognizer: Pure audio → text recognition
- TextMatcher: Pure text processing and overlap resolution
- Pipeline: Orchestration and lifecycle management

✅ **Open/Closed Principle:**
- Can add SilenceObserver later without modifying TextMatcher or Recognizer
- TextMatcher.finalize_pending() is extension point

✅ **Dependency Inversion Principle:**
- TextMatcher doesn't depend on concrete timeout implementation
- Exposes interface (finalize_pending) for external control

### Future Extension: SilenceObserver

**Can be added later without modifying existing classes:**

```python
class SilenceObserver:
    """Monitors VAD activity and triggers text finalization on silence.

    Single Responsibility: Detect phrase-level silence from VAD activity.
    """

    def __init__(self, chunk_queue, text_matcher, timeout=2.0):
        self.chunk_queue = chunk_queue
        self.text_matcher = text_matcher
        self.timeout = timeout
        self.last_segment_time = time.time()

    def monitor(self):
        """Monitor VAD activity and trigger finalization on timeout."""
        while self.is_running:
            if not self.chunk_queue.empty():
                self.last_segment_time = time.time()
            elif time.time() - self.last_segment_time > self.timeout:
                self.text_matcher.finalize_pending()
                self.last_segment_time = time.time()
            time.sleep(0.1)
```

This can be added to Pipeline later without changing TextMatcher or Recognizer.

---

## Test Counts

**Before:**
- Recognizer: 8 tests
- TextMatcher: 15 tests

**After:**
- Recognizer: 6 tests (-2)
- TextMatcher: 13 tests (-4 +2)

**Total:** 19 tests
