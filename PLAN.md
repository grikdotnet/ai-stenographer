# TextMatcher Redesign Plan - TDD Approach

## Problem Statement

TextMatcher was designed for fixed-length sliding windows with arbitrary time-based cuts that could truncate words. With the new VAD-based architecture:

- **Preliminary results** (word-level VAD): Clean boundaries, no truncation, no overlap between segments
- **Finalized results** (3s sliding windows): Still have overlap, still need resolution

Current TextMatcher applies overlap resolution to ALL results, which is:
- Unnecessary for preliminary results (wasted effort, potential corruption)
- Still necessary for finalized results (windows overlap by design)

## Solution: Merged TextMatcher + DisplayHandler

**Architecture change:** Merge TextMatcher and TwoStageDisplayHandler into single threaded component.

**Rationale:**
- Both are lightweight operations (text processing + GUI marshalling)
- Eliminates `partial_queue` and `final_queue` (simplification)
- Guaranteed ordering between preliminary and finalized results
- Direct GUI calls via `root.after()` (thread-safe)

**New flow:**
```
TextMatcher thread → process_text() → gui_window.update_partial()/finalize_text()
                                   ↓
                          root.after() → GUI main thread
```

**Two-path processing:**
1. **Preliminary path**: Simple duplicate detection → `gui_window.update_partial()` → temporary display
2. **Finalized path**: Overlap resolution → `gui_window.finalize_text()` → permanent display

---

## Phase 1: Test Analysis and Planning

### Current Test Coverage (test_text_matcher.py)

**Tests to KEEP (finalized path only):**
1. `test_resolve_overlap_with_boundary_errors` - Finalized windows still have overlap
2. `test_resolve_overlap_with_punctuation` - Finalized overlap handling
3. `test_resolve_overlap_with_case_variations` - Finalized overlap handling
4. `test_resolve_overlap_with_unicode` - Finalized overlap handling
5. `test_resolve_overlap_edge_cases` - Finalized overlap handling
6. `test_resolve_overlap_preserves_original_text` - Finalized overlap handling

**Tests to UPDATE (add preliminary variant):**
7. `test_process_first_text` - Add preliminary vs finalized scenarios
8. `test_process_texts_with_overlap` - Split into preliminary and finalized tests
9. `test_finalize_pending_method` - Test both paths
10. `test_finalize_pending_when_empty` - Keep as-is (path-agnostic)

**Tests to ADD (new preliminary path):**
11. `test_process_preliminary_no_overlap_resolution` - Verify preliminary skips overlap logic
12. `test_process_preliminary_to_gui_partial` - Calls `gui_window.update_partial()` directly
13. `test_process_finalized_to_gui_final` - Calls `gui_window.finalize_text()` after overlap resolution
14. `test_process_mixed_preliminary_finalized` - Real-world scenario with both types
15. `test_preliminary_ignores_boundary_errors` - Preliminary doesn't use resolve_overlap

**Note:** Removed `test_process_preliminary_duplicate_detection` - VAD guarantees distinct non-overlapping segments, so preliminary duplicates indicate a bug in AudioSource/Recognizer rather than expected behavior to handle.

### Delete obsolete tests

Review each test and remove any that:
- Test implementation details rather than behavior
- Are redundant with new tests
- Test scenarios that can't happen with VAD architecture
- tests that are useless for streaming realtime voice recognition


---

## Phase 2: Test Implementation

### 2.1 Update Test Fixtures

Remove queues, add mock GUI window:

```python
def setUp(self):
    self.text_queue = queue.Queue()
    self.mock_gui = Mock(spec=['update_partial', 'finalize_text'])

    # Helper to create test data
    def make_text_data(text, timestamp, is_preliminary=False):
        return {
            'text': text,
            'timestamp': timestamp,
            'is_preliminary': is_preliminary
        }
    self.make_text_data = make_text_data
```

### 2.2 Tests to ADD

#### Test: Preliminary results skip overlap resolution
```python
def test_process_preliminary_no_overlap_resolution(self):
    """Preliminary results should NOT use resolve_overlap()"""
    text_matcher = TextMatcher(self.text_queue, self.mock_gui)

    # Two preliminary results with no overlap
    prelim1 = {'text': 'hello world', 'timestamp': 1.0, 'is_preliminary': True}
    prelim2 = {'text': 'how are you', 'timestamp': 2.0, 'is_preliminary': True}

    text_matcher.process_text(prelim1)
    text_matcher.process_text(prelim2)

    # Both should call update_partial() directly (no overlap resolution)
    self.assertEqual(self.mock_gui.update_partial.call_count, 2)
    self.mock_gui.update_partial.assert_any_call('hello world')
    self.mock_gui.update_partial.assert_any_call('how are you')
```

#### Test: Mixed preliminary and finalized
```python
def test_process_mixed_preliminary_finalized(self):
    """System should handle mix of preliminary and finalized results"""
    text_matcher = TextMatcher(self.text_queue, self.mock_gui)

    # Preliminary result (instant, word-level)
    prelim1 = {'text': 'hello', 'timestamp': 1.0, 'is_preliminary': True}

    # Finalized results (sliding windows with overlap)
    final1 = {'text': 'hello world how', 'timestamp': 1.5, 'is_preliminary': False}
    final2 = {'text': 'world how are', 'timestamp': 2.0, 'is_preliminary': False}

    text_matcher.process_text(prelim1)
    text_matcher.process_text(final1)
    text_matcher.process_text(final2)

    # Verify preliminary called update_partial()
    self.mock_gui.update_partial.assert_any_call('hello')

    # Verify finalized called finalize_text() after overlap resolution
    self.assertGreater(self.mock_gui.finalize_text.call_count, 0)
```

### 2.3 Tests to UPDATE

#### Update: test_process_first_text
```python
def test_process_first_preliminary_text(self):
    """Test processing first preliminary text"""
    text_matcher = TextMatcher(self.text_queue, self.mock_gui)

    first_text = {'text': 'hello world', 'timestamp': 1.0, 'is_preliminary': True}
    text_matcher.process_text(first_text)

    # Preliminary should call update_partial() immediately
    self.mock_gui.update_partial.assert_called_once_with('hello world')

def test_process_first_finalized_text(self):
    """Test processing first finalized text"""
    text_matcher = TextMatcher(self.text_queue, self.mock_gui)

    first_text = {'text': 'hello world', 'timestamp': 1.0, 'is_preliminary': False}
    text_matcher.process_text(first_text)

    # Finalized should call update_partial() (awaiting overlap resolution)
    self.mock_gui.update_partial.assert_called_once_with('hello world')
    self.mock_gui.finalize_text.assert_not_called()
```

#### Update: test_process_texts_with_overlap
Split into:
- `test_process_finalized_texts_with_overlap` - Keep existing logic, add `is_preliminary=False`
- `test_process_preliminary_texts_no_overlap` - New test for preliminary path

---

## Phase 3: Implementation

### 3.1 Update TextMatcher.process_text()

Add routing logic:

```python
def process_text(self, text_data: Dict[str, Union[str, float, bool]]) -> None:
    """Process a single text segment using appropriate path.

    Routes to preliminary or finalized processing based on is_preliminary flag.

    Args:
        text_data: Dictionary containing 'text', 'timestamp', 'is_preliminary'
    """
    is_preliminary = text_data.get('is_preliminary', False)

    if is_preliminary:
        self.process_preliminary(text_data)
    else:
        self.process_finalized(text_data)
```

### 3.2 Add TextMatcher.process_preliminary()

New method for preliminary results:

```python
def process_preliminary(self, text_data: Dict[str, Union[str, float, bool]]) -> None:
    """Process preliminary result - no overlap resolution needed.

    Preliminary results come from word-level VAD segmentation with clean
    boundaries. They don't overlap, so we skip the overlap resolution
    logic and only apply duplicate detection.

    Args:
        text_data: Dictionary containing 'text' and 'timestamp'
    """
    current_text: str = text_data['text']

    if self.verbose:
        print(f"TextMatcher.process_preliminary() received '{current_text}'")

    # No duplicate detection needed - VAD guarantees distinct segments
    # If duplicates appear, it's a bug in AudioSource/Recognizer

    # Direct GUI call (no overlap resolution needed)
    if self.verbose:
        print(f"TextMatcher.process_preliminary() → update_partial('{current_text}')")
    self.gui_window.update_partial(current_text)
```

### 3.3 Rename existing logic to process_finalized()

```python
def process_finalized(self, text_data: Dict[str, Union[str, float, bool]]) -> None:
    """Process finalized result using window-based overlap resolution.

    Finalized results come from sliding windows with overlap. Uses the
    resolve_overlap algorithm to find reliable common subsequences and
    finalize the reliable parts while discarding boundary errors.

    Args:
        text_data: Dictionary containing 'text' and 'timestamp'
    """
    # Move all existing process_text() logic here
    # (lines 134-222 from current implementation)
    # BUT replace queue.put() calls with gui_window method calls:
    # - self.final_queue.put(...) → self.gui_window.finalize_text(text)
    # - self.partial_queue.put(...) → self.gui_window.update_partial(text)
```

### 3.4 Update constructor and state management

Update constructor signature and state:

```python
def __init__(self, text_queue: queue.Queue,
             gui_window,  # GuiWindow instance with update_partial() and finalize_text()
             text_normalizer: Optional[TextNormalizer] = None,
             time_threshold: float = 0.6,
             verbose: bool = False) -> None:
    self.text_queue = text_queue
    self.gui_window = gui_window  # Replace final_queue and partial_queue
    # ... other initialization ...

    # State only needed for finalized path (overlap resolution)
    # Preliminary path has no state - VAD guarantees distinct segments
    self.previous_finalized_text: str = ""
    self.previous_finalized_timestamp: float = 0.0
```

---

## Phase 4: Test Execution and Analysis

### 4.1 Run existing tests
```bash
python -m pytest tests/test_text_matcher.py -v
```

Expected: Most tests should fail (missing `is_preliminary` field)

### 4.2 Update test data

Add `is_preliminary: False` to all existing finalized tests

### 4.3 Run updated tests
```bash
python -m pytest tests/test_text_matcher.py -v
```

Expected: All existing tests pass, new tests fail (not implemented yet)

### 4.4 Implement preliminary path

Implement `process_preliminary()` and routing logic

### 4.5 Run all tests again
```bash
python -m pytest tests/test_text_matcher.py -v
```

Expected: All tests pass

---

## Phase 5: Integration Testing

### 5.1 Update pipeline construction

Update Pipeline to pass `gui_window` instead of queues:

```python
# In pipeline.py
class STTPipeline:
    def __init__(self, ...):
        # ... audio setup ...

        # Remove these queues
        # self.final_queue = queue.Queue()
        # self.partial_queue = queue.Queue()

        # TextMatcher gets gui_window directly
        self.text_matcher = TextMatcher(
            text_queue=self.text_queue,
            gui_window=self.gui_window,  # Pass GuiWindow instance
            verbose=verbose
        )

        # Remove TwoStageDisplayHandler initialization
        # (its functionality is now in TextMatcher + GuiWindow)
```

### 5.2 Update RecognitionResult usage

Verify that RecognitionResult → dict conversion includes `is_preliminary`:

```python
# In Recognizer.py or wherever RecognitionResult is converted to dict
text_data = {
    'text': result.text,
    'timestamp': result.end_time,
    'is_preliminary': result.is_preliminary
}
```

### 5.3 Run integration tests
```bash
python -m pytest tests/test_integration_preliminary_finalized.py -v
```

### 5.4 Manual testing
```bash
python main.py -v
```

Observe verbose output to verify:
- Preliminary results call `update_partial()` directly
- Finalized results use overlap resolution → `finalize_text()`
- No duplicates
- No boundary errors
- Preliminary text appears in gray/italic (temporary)
- Finalized text appears in black/normal (permanent)

---

## Success Criteria

✅ All unit tests pass (test_text_matcher.py)
✅ All integration tests pass (test_integration_preliminary_finalized.py)
✅ Preliminary results call `update_partial()` directly (verify in logs)
✅ Finalized results use overlap resolution → `finalize_text()` (verify in logs)
✅ No duplicates in output (verify in manual testing)
✅ Clean separation between paths (code review)
✅ GUI displays preliminary (gray/italic) and final (black/normal) correctly
✅ Documentation updated (docstrings, ARCHITECTURE.md)
✅ TwoStageDisplayHandler removed (merged into TextMatcher)
✅ `partial_queue` and `final_queue` removed from pipeline

---

## Risks and Mitigations

### Risk: State contamination between paths
**Mitigation:** Use separate state variables for preliminary vs finalized

### Risk: Breaking existing behavior
**Mitigation:** TDD approach ensures existing tests keep passing

### Risk: Duplicate detection too aggressive/lenient
**Mitigation:** Existing time threshold (0.6s) should work, adjust if needed

### Risk: Integration issues with Recognizer
**Mitigation:** Integration tests will catch this early

---

## Rollback Plan

If issues arise:

1. Keep `process_text()` as router
2. Make both paths call existing logic temporarily
3. Fix issues incrementally
4. Re-enable optimized paths

Git commits should be atomic:
- Commit 1: Update PLAN.md with architecture changes
- Commit 2: Add/update tests (failing) with mock GUI window
- Commit 3: Update TextMatcher constructor (remove queues, add gui_window)
- Commit 4: Implement routing and preliminary path
- Commit 5: Refactor finalized path (replace queue.put with GUI calls)
- Commit 6: Update Pipeline (remove TwoStageDisplayHandler, remove queues)
- Commit 7: Update documentation (ARCHITECTURE.md, docstrings)

Each commit can be reverted independently.
