# GuiWindow Partial Finalization Plan - TDD Approach

## Problem Statement

Current UX issue with preliminary/finalized text display:

1. Chunks arrive: "hello" "world" "how" "are" "you" (all preliminary, gray)
2. Window 1 (chunks 1-5) finalizes: "hello world how"
3. **BUG**: `finalize_text()` replaces ALL preliminary text, so "are you" disappears
4. Window 2 arrives later → "are you" reappears
5. **Bad UX**: Text flickers - preliminary disappears before being replaced

**Root cause:** `GuiWindow.finalize_text()` doesn't know which chunks are finalized vs. which are still preliminary.

**Solution:** Use `chunk_ids` (integers) instead of text matching.

## Solution: Chunk-ID-Based Partial Finalization

### Architecture Changes

**1. Add `chunk_ids` to RecognitionResult:**
```python
@dataclass
class RecognitionResult:
    text: str
    is_preliminary: bool
    chunk_ids: List[int]  # NEW: tracks which chunks contributed
```

**2. Modify GuiWindow to track chunk IDs:**
```python
class GuiWindow:
    def __init__(...):
        self.preliminary_results: List[RecognitionResult] = []  # Store objects, not strings
        self.finalized_chunk_ids: Set[int] = set()  # Track finalized chunks

    def update_partial(self, result: RecognitionResult):
        """Store and display preliminary result."""
        self.preliminary_results.append(result)
        # Append result.text to widget with "preliminary" tag

    def finalize_text(self, result: RecognitionResult):
        """Finalize chunks, re-render remaining preliminary."""
        # Mark chunks as finalized
        self.finalized_chunk_ids.update(result.chunk_ids)

        # Delete ALL text from widget
        # Re-add finalized text (black) from result.text
        # Re-add preliminary text (gray) from unfinalized chunks

        remaining = [
            r for r in self.preliminary_results
            if not any(cid in self.finalized_chunk_ids for cid in r.chunk_ids)
        ]
```

**3. Modify Recognizer to copy chunk_ids:**
```python
# In Recognizer.recognize()
result = RecognitionResult(
    text=recognized_text,
    is_preliminary=audio_segment.is_preliminary,
    chunk_ids=audio_segment.chunk_ids  # Copy from AudioSegment
)
```

**4. Modify TextMatcher to pass RecognitionResult objects:**
```python
# OLD: self.gui_window.finalize_text("hello world how")
# NEW: self.gui_window.finalize_text(RecognitionResult(..., chunk_ids=[1,2,3,4,5]))

# OLD: self.gui_window.update_partial("are you")
# NEW: self.gui_window.update_partial(RecognitionResult(..., chunk_ids=[6,7]))
```

### Data Flow

```
AudioSource + VAD → AudioSegment(chunk_ids=[1], preliminary=True)
                 ↓
              Recognizer → RecognitionResult(text="hello", chunk_ids=[1], preliminary=True)
                 ↓
           TextMatcher → gui_window.update_partial(RecognitionResult)
                 ↓
            GuiWindow → stores result, displays "hello" (gray)

--- Later ---

AdaptiveWindower → AudioSegment(chunk_ids=[1,2,3,4,5], preliminary=False)
                ↓
           Recognizer → RecognitionResult(text="hello world how", chunk_ids=[1,2,3,4,5], preliminary=False)
                ↓
          TextMatcher → overlap resolution → gui_window.finalize_text(RecognitionResult)
                ↓
           GuiWindow → marks chunks 1-5 as finalized
                    → re-renders: "hello world how" (black) + "are you" (gray, from chunks 6-7)
```

### Why This Works

✅ **Fast**: Matching integers in sets (10 values) is O(1) per chunk
✅ **Reliable**: Chunk IDs don't change like text does
✅ **Accurate**: Knows exactly which chunks are finalized
✅ **No flickering**: Preliminary text from future windows stays visible

---

## Phase 1: GuiWindow Tests (TDD)

### Tests to ADD to test_gui_window.py

These tests demonstrate the desired behavior for chunk-ID-based partial finalization:

**1. `test_partial_finalization_with_chunk_ids`**
- Multiple preliminary results with chunk_ids
- Finalize only some chunks
- Verify finalized chunks become black, unfinalized stay gray

**2. `test_overlapping_window_workflow_with_chunk_ids`**
- Complete realistic workflow: preliminary → finalized Window 1 → more preliminary → finalized Window 2
- Verify chunks are correctly tracked and displayed

**3. `test_finalize_tracks_chunk_ids_globally`**
- Multiple finalization calls
- Verify `finalized_chunk_ids` set grows correctly
- No chunk finalized twice

**4. `test_preliminary_results_stored_as_objects`**
- Verify GuiWindow stores `RecognitionResult` objects, not strings
- Check internal state after multiple preliminary results

**Steps:**
1. ✅ Update test fixtures to use `RecognitionResult` dataclass
2. ✅ Write failing tests demonstrating chunk-ID-based behavior
3. ✅ Verify tests fail (implementation not ready)

---

## Phase 2: Recognizer Update

### 2.1 Update Recognizer Tests (test_recognizer.py)

**Modify existing tests to verify `chunk_ids` are copied:**

1. **`test_recognize_preliminary_segment`**
   - Verify `RecognitionResult.chunk_ids` matches `AudioSegment.chunk_ids`
   - Example: `AudioSegment(chunk_ids=[1])` → `RecognitionResult(chunk_ids=[1])`

2. **`test_recognize_finalized_segment`**
   - Verify `RecognitionResult.chunk_ids` matches `AudioSegment.chunk_ids`
   - Example: `AudioSegment(chunk_ids=[1,2,3,4,5])` → `RecognitionResult(chunk_ids=[1,2,3,4,5])`

3. **Add: `test_chunk_ids_preserved_in_recognition`**
   - Test that chunk_ids are preserved through recognition process
   - Verify no chunk_ids are lost or duplicated

**Steps:**
1. Read existing Recognizer tests
2. Add assertions to verify `chunk_ids` are copied from `AudioSegment` to `RecognitionResult`
3. Run tests - expect failures (not implemented yet)

### 2.2 Implement Recognizer Changes

**Modify `Recognizer.recognize()`:**
```python
def recognize(self, audio_segment: AudioSegment) -> RecognitionResult:
    """Recognize speech from audio segment."""
    # ... existing recognition logic ...

    result = RecognitionResult(
        text=recognized_text,
        start_time=audio_segment.start_time,
        end_time=audio_segment.end_time,
        is_preliminary=(audio_segment.type == 'preliminary'),
        chunk_ids=audio_segment.chunk_ids  # NEW: Copy chunk_ids
    )
    return result
```

**Steps:**
1. Update `Recognizer.recognize()` to copy `chunk_ids`
2. Run tests - expect all Recognizer tests to pass
3. Commit: "feat(Recognizer): copy chunk_ids from AudioSegment to RecognitionResult"

---

## Phase 3: TextMatcher Update

### 3.1 Update TextMatcher Tests (test_text_matcher.py)

**Current issue:** TextMatcher works with dict/queue messages, not `RecognitionResult` objects.

**Changes needed:**

1. **Update test fixtures** - Create `RecognitionResult` objects instead of dicts:
   ```python
   # OLD
   text_data = {'text': 'hello', 'timestamp': 1.0, 'is_preliminary': True}

   # NEW
   result = RecognitionResult(
       text='hello',
       start_time=0.0,
       end_time=1.0,
       is_preliminary=True,
       chunk_ids=[1]
   )
   ```

2. **Update mock GUI calls** - Expect `RecognitionResult` objects, not strings:
   ```python
   # OLD
   self.mock_gui.update_partial.assert_called_with('hello')

   # NEW
   call_args = self.mock_gui.update_partial.call_args[0][0]
   self.assertIsInstance(call_args, RecognitionResult)
   self.assertEqual(call_args.text, 'hello')
   self.assertEqual(call_args.chunk_ids, [1])
   ```

3. **Add tests for chunk_ids preservation:**
   - `test_preliminary_preserves_chunk_ids` - Verify preliminary path preserves chunk_ids
   - `test_finalized_preserves_chunk_ids` - Verify finalized path preserves chunk_ids
   - `test_overlap_resolution_merges_chunk_ids` - When Window 2 overlaps Window 1, verify chunk_ids are merged correctly

**Steps:**
1. Read existing TextMatcher tests
2. Update fixtures to use `RecognitionResult` objects
3. Update assertions to check `RecognitionResult` fields
4. Add chunk_ids preservation tests
5. Run tests - expect failures (not implemented yet)

### 3.2 Implement TextMatcher Changes

**Modify TextMatcher to pass `RecognitionResult` objects to GUI:**

```python
def process_preliminary(self, result: RecognitionResult) -> None:
    """Process preliminary result - pass RecognitionResult to GUI."""
    if self.verbose:
        print(f"TextMatcher.process_preliminary() received '{result.text}' chunk_ids={result.chunk_ids}")

    # Pass RecognitionResult object (not just text string)
    self.gui_window.update_partial(result)

def process_finalized(self, result: RecognitionResult) -> None:
    """Process finalized result using overlap resolution."""
    # ... existing overlap resolution logic ...

    # Create new RecognitionResult with resolved text but SAME chunk_ids
    finalized_result = RecognitionResult(
        text=finalized_text,
        start_time=result.start_time,
        end_time=result.end_time,
        is_preliminary=False,
        chunk_ids=result.chunk_ids  # Preserve chunk_ids through overlap resolution
    )

    self.gui_window.finalize_text(finalized_result)
```

**Steps:**
1. Update `process_preliminary()` to pass `RecognitionResult` objects
2. Update `process_finalized()` to create and pass `RecognitionResult` objects
3. Verify chunk_ids are preserved through overlap resolution
4. Run tests - expect all TextMatcher tests to pass
5. Commit: "feat(TextMatcher): pass RecognitionResult objects to GuiWindow"

---

## Phase 4: GuiWindow Implementation

### 4.1 Tests Already Written (Phase 1)

The tests in `TestGuiWindowWithChunkIds` are already written and failing.

### 4.2 Implement GuiWindow Changes

**Modify GuiWindow to track chunk IDs:**

```python
class GuiWindow:
    def __init__(self, text_widget, root):
        # ... existing initialization ...
        self.preliminary_results: List[RecognitionResult] = []  # NEW: Store objects
        self.finalized_chunk_ids: Set[int] = set()  # NEW: Track finalized chunks
        self.finalized_text: str = ""  # Track finalized text

    def update_partial(self, result: RecognitionResult) -> None:
        """Store and display preliminary result."""
        # Store preliminary result
        self.preliminary_results.append(result)

        # Thread-safe GUI update
        self.root.after(0, self._update_partial_safe, result.text)

    def finalize_text(self, result: RecognitionResult) -> None:
        """Finalize chunks, re-render remaining preliminary."""
        # Mark chunks as finalized
        self.finalized_chunk_ids.update(result.chunk_ids)

        # Store finalized text
        self.finalized_text += (" " if self.finalized_text else "") + result.text

        # Thread-safe GUI update - re-render everything
        self.root.after(0, self._rerender_all)

    def _rerender_all(self) -> None:
        """Re-render all text: finalized (black) + remaining preliminary (gray)."""
        # Clear everything
        self.text_widget.delete("1.0", tk.END)

        # Add finalized text (black)
        if self.finalized_text:
            self.text_widget.insert(tk.END, self.finalized_text + " ", "final")

        # Update preliminary_start_pos
        self.preliminary_start_pos = self.text_widget.index("end-1c")

        # Find preliminary results with unfinalized chunks
        remaining = [
            r for r in self.preliminary_results
            if not any(cid in self.finalized_chunk_ids for cid in r.chunk_ids)
        ]

        # Add remaining preliminary text (gray)
        for i, r in enumerate(remaining):
            separator = " " if i > 0 else ""
            self.text_widget.insert(tk.END, separator + r.text, "preliminary")

        # Auto-scroll
        self.text_widget.see(tk.END)
```

**Steps:**
1. Add `preliminary_results` and `finalized_chunk_ids` to `__init__`
2. Update `update_partial()` to accept `RecognitionResult` and store it
3. Update `finalize_text()` to accept `RecognitionResult` and use chunk_ids
4. Add `_rerender_all()` method to re-render finalized + remaining preliminary
5. Update old tests to use `RecognitionResult` instead of strings
6. Run tests - expect all GuiWindow tests to pass
7. Commit: "feat(GuiWindow): implement chunk-ID-based partial finalization"

---

## Phase 5: Integration Testing

### 5.1 Update Pipeline

No changes needed - Pipeline already passes objects through queues.

### 5.2 Manual Testing

```bash
python main.py -v
```

Observe:
- Preliminary chunks appear immediately (gray)
- Window 1 finalized → matched chunks turn black, future chunks stay gray
- No text flickering
- Verbose logs show chunk_ids at each stage

---

## Success Criteria

✅ Phase 1: GuiWindow tests written and failing
✅ Phase 2: Recognizer copies chunk_ids, tests pass
✅ Phase 3: TextMatcher passes RecognitionResult objects, tests pass
✅ Phase 4: GuiWindow implements chunk-ID tracking, tests pass
✅ Phase 5: Integration testing shows no flickering
✅ All existing tests updated to use RecognitionResult (no backward compatibility)
✅ Documentation updated

---

No Git Commits