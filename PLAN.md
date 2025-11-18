# State Machine Refactoring Plan - TDD Approach

## Overview

Refactor `SoundPreProcessor` from implicit boolean-flag-based state management to an explicit enum-based state machine using `match/case` pattern. Follow TDD: write tests first, then implement.

**Goal:** Make the code clearer and more maintainable. 
---

## Phase 1: Create State Machine Tests

### Test File
**File:** `tests/test_sound_preprocessor_state_machine.py` (new file)

### Test 1: State Transitions - IDLE
**Test:** `test_state_idle_to_waiting_on_speech`
- Initial state: IDLE
- Feed 1 speech chunk
- Verify: `state == WAITING_CONFIRMATION`, `consecutive_speech_count == 1`

**Test:** `test_state_idle_stays_idle_on_silence`
- Initial state: IDLE
- Feed silence chunks
- Verify: `state == IDLE`

### Test 2: State Transitions - WAITING_CONFIRMATION
**Test:** `test_state_waiting_to_active_on_third_speech`
- State: WAITING_CONFIRMATION, count = 2
- Feed 3rd speech chunk
- Verify:
  - `state == ACTIVE_SPEECH`
  - `speech_buffer` has 3 chunks
  - `left_context_snapshot` captured

**Test:** `test_state_waiting_to_idle_on_silence`
- State: WAITING_CONFIRMATION, count = 2
- Feed silence chunk
- Verify: `state == IDLE`, `consecutive_speech_count == 0`

**Test:** `test_state_waiting_continues_on_second_speech`
- State: WAITING_CONFIRMATION, count = 1
- Feed 2nd speech chunk
- Verify: `state == WAITING_CONFIRMATION`, `consecutive_speech_count == 2`

### Test 3: State Transitions - ACTIVE_SPEECH
**Test:** `test_state_active_stays_active_on_speech`
- State: ACTIVE_SPEECH
- Feed speech chunks (below max duration)
- Verify: `state == ACTIVE_SPEECH`, buffer grows

**Test:** `test_state_active_to_accumulating_on_silence`
- State: ACTIVE_SPEECH
- Feed silence chunk
- Verify:
  - `state == ACCUMULATING_SILENCE`
  - `silence_energy > 0`

**Test:** `test_state_active_stays_active_on_max_duration_with_breakpoint`
- State: ACTIVE_SPEECH, buffer at max duration
- Mock `_find_silence_breakpoint()` to return a breakpoint
- Feed chunk that triggers max_duration
- Verify:
  - `state == ACTIVE_SPEECH` (continuation, speech continues)
  - `_build_audio_segment(breakpoint_idx)` called
  - Segment emitted to speech_queue
  - windower.process_segment() called
  - `_keep_remainder_after_breakpoint(breakpoint_idx)` called

**Test:** `test_state_active_stays_active_on_max_duration_hard_cut`
- State: ACTIVE_SPEECH, buffer at max duration
- Mock `_find_silence_breakpoint()` to return None
- Feed chunk that triggers max_duration
- Verify:
  - `state == ACTIVE_SPEECH` (continuation, speech continues)
  - `_build_audio_segment(None)` called
  - Segment emitted to speech_queue
  - windower.process_segment() called
  - `_reset_segment_state()` called
  - Current chunk re-processed (starts new segment)

### Test 4: State Transitions - ACCUMULATING_SILENCE
**Test:** `test_state_accumulating_to_active_on_speech_resume`
- State: ACCUMULATING_SILENCE, energy < threshold
- Feed speech chunk
- Verify:
  - `state == ACTIVE_SPEECH`
  - `silence_energy == 0`

**Test:** `test_state_accumulating_stays_accumulating`
- State: ACCUMULATING_SILENCE, energy < threshold
- Feed silence chunk (doesn't reach threshold)
- Verify:
  - `state == ACCUMULATING_SILENCE`
  - `silence_energy` increased

**Test:** `test_state_accumulating_to_idle_on_threshold`
- State: ACCUMULATING_SILENCE, energy approaching threshold
- Feed silence chunk that exceeds threshold
- Verify:
  - `state == IDLE`
  - Finalization method called (segment emitted)

### Test 5: State Consistency
**Test:** `test_state_property_is_speech_active`
- Verify property works correctly:
  - `IDLE` � `is_speech_active == False`
  - `WAITING_CONFIRMATION` � `is_speech_active == False`
  - `ACTIVE_SPEECH` � `is_speech_active == True`
  - `ACCUMULATING_SILENCE` � `is_speech_active == True`

---

## Phase 2: Helper Methods Design

### Helper Methods to Extract

These methods will be extracted from existing code to keep the `match/case` logic clean:

#### Buffering Methods
1. **`_append_to_context_buffer(audio, timestamp, is_speech, chunk_id)`**
   - Append chunk dict to circular context_buffer
   - Used in all states

2. **`_append_to_speech_buffer(audio, timestamp, is_speech, chunk_id)`**
   - Append chunk dict to speech_buffer
   - Used in ACTIVE_SPEECH and ACCUMULATING_SILENCE states

#### Segment Initialization Methods
3. **`_capture_left_context_from_buffer()`**
   - Extract left context from context_buffer (exclude last 2 chunks)
   - Set `self.left_context_snapshot`
   - Called when transitioning WAITING_CONFIRMATION → ACTIVE_SPEECH

4. **`_initialize_speech_buffer_from_context()`**
   - Extract last 2 chunks from context_buffer
   - Assign chunk_ids and populate speech_buffer
   - Set `self.speech_start_time`
   - Called when transitioning WAITING_CONFIRMATION → ACTIVE_SPEECH

#### Segment Finalization Methods
5. **`_build_audio_segment(breakpoint_idx) -> AudioSegment`**
   - Build AudioSegment from speech_buffer
   - Extract left_context from left_context_snapshot
   - Extract data (speech chunks only, from 0 to breakpoint_idx or all)
   - Extract right_context (trailing silence or from buffer after breakpoint)
   - Return AudioSegment
   - Called from state machine, which then emits to queue and windower

6. **`_keep_remainder_after_breakpoint(breakpoint_idx)`**
   - Keep remainder chunks in speech_buffer (after breakpoint_idx)
   - Update speech_start_time for remainder
   - Reset silence_energy
   - Called from state machine after breakpoint split

7. **`_reset_segment_state()`**
   - Reset speech_buffer, silence_energy, left_context_snapshot
   - Called from state machine when transitioning to IDLE

#### Existing Methods to Keep
8. **`_find_silence_breakpoint()`** - unchanged
9. **`_normalize_rms(audio)`** - unchanged

### Tests for Helper Methods

**File:** `tests/test_sound_preprocessor_helpers.py` (new file)

#### Test: Buffering Methods
**Test:** `test_append_to_context_buffer`
- Call `_append_to_context_buffer()` with chunk data
- Verify chunk dict added to context_buffer with correct structure

**Test:** `test_append_to_speech_buffer`
- Call `_append_to_speech_buffer()` with chunk data
- Verify chunk dict added to speech_buffer with correct structure

#### Test: Segment Initialization
**Test:** `test_capture_left_context_from_buffer`
- Populate context_buffer with 10 chunks
- Set consecutive_speech_count = 3 (last 2 chunks already in buffer)
- Call `_capture_left_context_from_buffer()`
- Verify `left_context_snapshot` contains first 8 chunks (10 - 2)

**Test:** `test_initialize_speech_buffer_from_context`
- Populate context_buffer with last 2 speech chunks
- Call `_initialize_speech_buffer_from_context()`
- Verify:
  - speech_buffer has 2 chunks
  - chunk_ids assigned correctly
  - speech_start_time set to first chunk timestamp

#### Test: Segment Finalization
**Test:** `test_build_audio_segment_without_breakpoint`
- Set left_context_snapshot
- Populate speech_buffer with speech and trailing silence chunks
- Call `segment = _build_audio_segment(breakpoint_idx=None)`
- Verify returned AudioSegment:
  - left_context from snapshot
  - data has speech chunks only
  - right_context has trailing silence
  - chunk_ids correct
  - type == 'preliminary'

**Test:** `test_build_audio_segment_with_breakpoint`
- Set left_context_snapshot
- Populate speech_buffer with 60 chunks
- Call `segment = _build_audio_segment(breakpoint_idx=40)`
- Verify returned AudioSegment:
  - data from chunks 0-40 only
  - right_context extracted from chunks 41-46 (up to 6 chunks)
  - chunk_ids from 0-40

**Test:** `test_keep_remainder_after_breakpoint`
- Populate speech_buffer with 60 chunks
- Call `_keep_remainder_after_breakpoint(40)`
- Verify:
  - speech_buffer has remainder (41-59)
  - speech_start_time updated to chunk 41 timestamp
  - silence_energy reset to 0

**Test:** `test_reset_segment_state`
- Populate speech_buffer, silence_energy, left_context_snapshot
- Call `_reset_segment_state()`
- Verify all cleared

---

## Implementation Order

1. **Create state machine tests** (Phase 1)
2. **Create helper method tests** (Phase 2)
3. **Add ProcessingState enum**
4. **Add state field and property**
5. **Extract helper methods** (implement and verify tests pass)
6. **Refactor `_process_chunk` to match/case**
7. **Remove old methods**
8. **Run all tests**
9. **Cleanup**

---

## Success Criteria

- [ ] All new state machine tests pass
- [ ] All existing tests pass (100% backward compatible)
- [ ] Code is more readable (state machine visible)
- [ ] No change to public API
- [ ] No change to behavior

---

## Key Design Decisions

1. **Use `match/case`** for state machine (Python 3.10+)
2. **All state transitions in `_process_chunk`** - single source of truth
3. **Explicit state assignments** - `self.state = ProcessingState.X`
4. **Helper methods for actions** - keep match/case readable
5. **Pure refactoring** - no behavior changes, no bug fixes
