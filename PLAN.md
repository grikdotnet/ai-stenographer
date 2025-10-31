# Speech Recognition Quality Improvement Plan

## Problem Statement

Short words (<100ms) are poorly recognized.
1. **No pre-speech padding**: Attack transients are lost due to RMS gain ramp-up (�=0.85, takes 3-4 chunks to stabilize)
2. **No post-speech padding**: Release transients are cut off when silence energy threshold is reached
3. **Segments can be very short**: Below NeMo's 100ms training minimum
4. **Aggressive detection parameters**: Low threshold (0.3) and quick finalization (1.2 energy) lead to premature segmentation

## Solution Overview

Split into two tasks:
- **Task 1**: Prepend/append real sound chunks (no artificial silence padding)
- **Task 2**: Add silence padding in Recognizer for fixed-duration inference with shader reuse

## Config Changes (User-Tunable Parameters)

```json
{
  "audio": {
    "silence_energy_threshold": 1.5,  // Was 1.2 � requires 3 chunks instead of 2
    "rms_normalization": {
      "target_rms": 0.07,              // Was 0.05 � better VAD response
      "silence_threshold": 0.0001,     // Unchanged
      "gain_smoothing": 0.7            // Was 0.85 � faster gain ramp-up
    }
  },
  "vad": {
    "threshold": 0.4                   // Was 0.3 � more conservative
  }
}
```

---

# Task 1: Prepend/Append Real Sound Chunks

## Objective
Capture attack and release transients by prepending the last silence chunk and appending silence chunks during energy accumulation.

## Design Changes

### 1. Add `last_silence_chunk` Variable
Store the most recent silence chunk for prepending when speech starts.

### 2. Prepend Last Silence Chunk on Speech Start
When `is_speech=True` and `is_speech_active=False`, prepend the stored silence chunk to recover lost speech onset.

### 3. Buffer Silence Chunks During Energy Accumulation
Modify existing `_handle_silence_frame()` to append silence chunks to `self.speech_buffer` (same as speech chunks) while accumulating energy, until threshold is reached.

### 4. Update Config Parameters
- `silence_energy_threshold`: 1.2 � 1.5 (ensures ~2-3 trailing chunks)
- `target_rms`: 0.05 � 0.07
- `gain_smoothing`: 0.85 � 0.7
- `vad.threshold`: 0.3 � 0.4

---

## TDD Implementation Plan for Task 1

### Phase 1: Design New Tests

#### Test 1.1: `test_last_silence_chunk_stored`
**Purpose**: Verify that silence chunks are stored in `last_silence_chunk` variable.

**Setup**:
- Mock VAD to return `is_speech=False` for first 2 chunks
- Feed 2 silence chunks

**Assertion**:
- `preprocessor.last_silence_chunk` is not None
- Contains the second chunk's audio and timestamp

**Location**: `tests/test_sound_preprocessor.py`

---

#### Test 1.2: `test_last_silence_chunk_prepended_on_speech_start`
**Purpose**: Verify that the last silence chunk is prepended when speech starts.

**Setup**:
- Mock VAD: silence, silence, speech, speech
- Feed 4 chunks

**Assertion**:
- `speech_buffer` has 3 items (1 prepended + 2 speech chunks)
- First item in buffer matches `last_silence_chunk`
- AudioSegment's `chunk_ids` contains only the 2 speech chunk IDs (prepended silence not included)

**Location**: `tests/test_sound_preprocessor.py`

---

#### Test 1.3: `test_silence_chunks_buffered_during_energy_accumulation`
**Purpose**: Verify that silence chunks are buffered while counting energy.

**Setup**:
- Mock VAD: speech (3 chunks), then silence with prob=0.25
- silence_energy_threshold = 1.5
- Feed chunks until energy reaches threshold

**Assertion**:
- `speech_buffer` contains speech chunks + trailing silence chunks
- AudioSegment's audio data includes trailing silence
- AudioSegment's `chunk_ids` contains only speech chunk IDs (trailing silence not included)

**Location**: `tests/test_sound_preprocessor.py`

---

#### Test 1.4: skip

---

#### Test 1.5: `test_no_prepend_if_no_previous_silence`
**Purpose**: Verify no prepending if speech starts immediately (no prior silence chunk).

**Setup**:
- Mock VAD: speech, speech, speech (from start)
- Feed 3 speech chunks

**Assertion**:
- `speech_buffer` has exactly 3 items

**Location**: `tests/test_sound_preprocessor.py`

---

#### Test 1.6: `test_last_silence_chunk_cleared_after_prepend`
**Purpose**: Verify `last_silence_chunk` is cleared after prepending to avoid reuse.

**Setup**:
- Mock VAD: silence, speech, silence, speech (two speech segments)
- Feed chunks

**Assertion**:
- After first speech start: `last_silence_chunk` is None
- After second silence: `last_silence_chunk` stores new chunk
- Second speech segment prepends the NEW silence chunk

**Location**: `tests/test_sound_preprocessor.py`

---

### Phase 2: Update Existing Tests

#### Test 2.1: Update `test_preliminary_segment_emission_on_silence`
**Current behavior**: Emits segment with 3 speech chunks only.

**Change needed**: With trailing buffering, segment audio will include trailing silence chunks.

**Update**:
- Keep existing assertion: `len(segment.chunk_ids) == 3` (only speech chunks)
- Add assertion: Verify audio data length is longer than 3 chunks (includes trailing silence)
- Expected audio length: ~5 chunks worth (3 speech + 2 trailing silence)

**Location**: `tests/test_sound_preprocessor.py:198`

---

#### Test 2.2: skip

---

#### Test 2.3: Update `test_segment_finalization_at_energy_threshold`
**Current behavior**: Tests finalization when threshold is reached.

**Change needed**: Adjust VAD probabilities for new threshold value (1.5 instead of 1.2).

**Update**:
- Update VAD probabilities to reach new threshold (1.5)

**Location**: `tests/test_sound_preprocessor.py:298`

---

#### Test 2.4: Update `test_timestamp_accuracy`
**Current behavior**: Expects end_time based only on speech chunks.

**Change needed**: With prepended and trailing silence chunks, end_time must include total audio duration.

**Update**:
- Calculate expected end_time = start_time + (total_audio_length / sample_rate)
- Total audio = prepended chunk + speech chunks + trailing chunks
- Example: 1 prepended + 3 speech + 2 trailing = 6 chunks = 192ms
- Expected: start_time=1.0, end_time=1.192 (not 1.096)

**Location**: `tests/test_sound_preprocessor.py:506`

---

### Phase 3: Implementation Steps (TDD)

1. **Write new tests (1.1-1.6)** - All tests should FAIL initially
2. **Run pytest** - Confirm all new tests fail with expected errors
3. **Add `last_silence_chunk` variable** to `SoundPreProcessor.__init__()`
4. **Implement storing logic** in `_process_chunk()` when `is_speech=False`
5. **Run Test 1.1** - Should PASS
6. **Implement prepending logic** in `_handle_speech_frame()` for `not self.is_speech_active`
7. **Run Test 1.2, 1.5, 1.6** - Should PASS
8. **Implement buffering silence chunks** in `_handle_silence_frame()` (change line 234-249)
9. **Run Test 1.3** - Should PASS
10. **Update config file** with new parameters
12. **Update existing tests (2.1, 2.3, 2.4)** to match new behavior
13. **Run all tests** - All should PASS
14. **Manual testing** with `python main.py -v` - Verify short words recognized better

---

### Code Changes Summary for Task 1

**File**: `src/SoundPreProcessor.py`

**Changes**:
1. Add `self.last_silence_chunk = None` to `__init__()`
2. In `_process_chunk()`:
   - When `is_speech=False`: Store chunk in `last_silence_chunk`
3. In `_handle_speech_frame()`:
   - When `not self.is_speech_active` and `last_silence_chunk is not None`:
     - Prepend it to `speech_buffer`
     - Set `last_silence_chunk = None`
4. **Modify** existing `_handle_silence_frame()`:
   - When `self.is_speech_active`:
     - Append silence chunks to buffer: `self.speech_buffer.append({'audio': audio_chunk, 'timestamp': timestamp})`
     - Continue with existing energy accumulation and finalization logic
5. In `_finalize_segment()`:
   - Extract audio data from all buffered chunks (speech + silence)
   - This way prepended/trailing silence is in audio data

**File**: `config/stt_config.json`

**Changes**:
```json
{
  "audio": {
    "silence_energy_threshold": 1.5,
    "rms_normalization": {
      "target_rms": 0.07,
      "gain_smoothing": 0.7
    }
  },
  "vad": {
    "threshold": 0.4
  }
}
```

---

# Task 2: Add Silence Padding in Recognizer

## Objective
Add silence padding to audio segments before STT inference to:
1. Meet NeMo's 100ms minimum duration requirement
2. Create fixed-duration batches for shader reuse (performance optimization)

## Design Approach

### Strategy
- **No changes to SoundPreProcessor**: It emits raw segments as-is
- **Padding in Recognizer**: Add silence before/after audio in `recognize_window()` method
- **No metadata needed**: Recognizer applies padding logic directly based on audio length

### Padding Logic

```python
def _pad_audio_for_recognition(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Pad audio to optimal duration for STT model inference.

    Logic:
    - Pre-padding: Always add 96ms (3 chunks) silence before audio
    - Post-padding: Pad to minimum 100ms total duration
    - Batch-size optimization: Round up to nearest 500ms for shader reuse

    Args:
        audio: Raw audio data from AudioSegment
        sample_rate: Audio sample rate (16000 Hz)

    Returns:
        Padded audio ready for STT inference
    """
    # Pre-padding: 96ms silence (3 chunks � 32ms)
    pre_padding_samples = int(0.096 * sample_rate)  # 1536 samples
    pre_silence = np.zeros(pre_padding_samples, dtype=np.float32)

    # Combine
    padded_audio = np.concatenate([pre_silence, audio])

    # Minimum duration: 100ms (NeMo requirement)
    min_samples = int(0.1 * sample_rate)  # 1600 samples
    if len(padded_audio) < min_samples:
        post_padding_samples = min_samples - len(padded_audio)
        post_silence = np.zeros(post_padding_samples, dtype=np.float32)
        padded_audio = np.concatenate([padded_audio, post_silence])

    # Optional: Round to nearest 500ms for batch optimization
    # (This can be enabled later for performance tuning)

    return padded_audio
```

---

## TDD Implementation Plan for Task 2

### Phase 1: Design New Tests

#### Test 2.1: `test_recognizer_pads_short_audio`
**Purpose**: Verify Recognizer pads audio shorter than 100ms.

**Setup**:
- Mock model that captures audio length
- Create AudioSegment with 50ms audio (800 samples)

**Assertion**:
- Model receives audio >= 1600 samples (100ms minimum)
- Pre-padding of ~1536 samples (96ms) is present
- Audio starts with near-zero values (silence padding)

**Location**: `tests/test_recognizer.py`

---

#### Test 2.2: `test_recognizer_adds_pre_padding_to_all_segments`
**Purpose**: Verify all segments receive 96ms pre-padding, regardless of length.

**Setup**:
- Mock model that captures audio
- Create segments: 50ms, 100ms, 200ms, 1000ms

**Assertion**:
- All segments receive 96ms (1536 samples) pre-padding
- First 1536 samples are near-zero (silence)

**Location**: `tests/test_recognizer.py`

---

#### Test 2.3: `test_recognizer_preserves_original_audio_data`
**Purpose**: Verify padding doesn't modify original audio, only prepends/appends.

**Setup**:
- Create AudioSegment with distinct audio pattern (sine wave)
- Mock model captures audio

**Assertion**:
- After skipping pre-padding samples, original audio is intact
- No amplitude changes or modifications to speech content

**Location**: `tests/test_recognizer.py`

---

#### Test 2.4: `test_recognizer_no_padding_affects_timing`
**Purpose**: Verify RecognitionResult timestamps reflect ORIGINAL segment times, not padded audio.

**Setup**:
- Create AudioSegment with start_time=1.0, end_time=1.2
- Recognizer pads audio

**Assertion**:
- RecognitionResult.start_time == 1.0 (unchanged)
- RecognitionResult.end_time == 1.2 (unchanged)
- Timing metadata preserved despite audio padding

**Location**: `tests/test_recognizer.py`

---

#### Test 2.5: `test_recognizer_padding_with_preliminary_segments`
**Purpose**: Verify preliminary segments (short, single-word) get proper padding.

**Setup**:
- Create preliminary AudioSegment with 60ms audio
- Mock model

**Assertion**:
- Padded to >= 100ms
- Pre-padding included
- RecognitionResult.status == 'preliminary' (unchanged)

**Location**: `tests/test_recognizer.py`

---

#### Test 2.6: `test_recognizer_padding_with_finalized_segments`
**Purpose**: Verify finalized segments (3s windows) also receive pre-padding.

**Setup**:
- Create finalized AudioSegment with 3s audio (48000 samples)
- Mock model

**Assertion**:
- Pre-padding of 96ms added
- Total length ~3.096s
- RecognitionResult.status == 'final' (unchanged)

**Location**: `tests/test_recognizer.py`

---

#### Test 2.7: `test_padding_does_not_create_hallucination`
**Purpose**: Verify silence padding doesn't cause model to hallucinate text.

**Setup**:
- Real Parakeet model (not mock)
- Create AudioSegment with ONLY silence (no speech)
- Add padding

**Assertion**:
- Model returns empty string or None
- Recognizer filters it out (returns None)
- No phantom text generated from padding

**Location**: `tests/test_recognizer.py` (integration test, may be slow)

---

### Phase 2: Update Existing Tests

#### Test 2.8: Update existing Recognizer tests
**Current behavior**: Tests assume model receives original audio unchanged.

**Change needed**: Tests must account for padding.

**Tests to update**:
- `test_recognizes_preliminary_segments`
- `test_recognizes_finalized_segments`
- `test_preserves_timing_information`

**Update**:
- Mock model should expect padded audio length
- Or mock the padding method to return original audio (isolate padding logic)

**Location**: `tests/test_recognizer.py`

---

### Phase 3: Implementation Steps (TDD)

1. **Write new tests (2.1-2.7)** - All tests should FAIL initially
2. **Run pytest** - Confirm failures
3. **Implement `_pad_audio_for_recognition()` method** in `src/Recognizer.py`
4. **Update `recognize_window()` method** to call padding before `model.recognize()`
5. **Run Test 2.1, 2.2** - Should PASS
6. **Verify pre-padding logic**
7. **Run Test 2.3, 2.4** - Should PASS (timing preserved)
8. **Run Test 2.5, 2.6** - Should PASS (works for both segment types)
9. **Run Test 2.7** - May need real model, should PASS (no hallucination)
10. **Update existing tests (2.8)** to work with padding
11. **Run all Recognizer tests** - All should PASS
12. **Integration testing** - Run full pipeline, verify quality improvement

---

### Code Changes Summary for Task 2

**File**: `src/Recognizer.py`

**Changes**:
1. Add `_pad_audio_for_recognition()` method (see logic above)
2. In `recognize_window()` method:
   - Call `audio = self._pad_audio_for_recognition(window_data.data, sample_rate=16000)`
   - Pass padded audio to `model.recognize(audio)`
3. Timing metadata (start_time, end_time) remains unchanged
4. No changes to AudioSegment or RecognitionResult data structures

---

## Testing Strategy

### Unit Tests
- **SoundPreProcessor**: 8 new tests + 4 updated tests
- **Recognizer**: 7 new tests + update existing tests

### Integration Tests
- Run `python main.py -v` with short word utterances:
  - "go"
  - "yes"
  - "no"
  - "I"
  - "a cat"
- Verify:
  - Preliminary text appears correctly (no missing letters)
  - Final text matches ground truth
  - No hallucinated text from padding

### Performance Tests (Optional)
- Measure recognition latency before/after changes
- Expected: Minimal impact (<10ms increase due to padding)
- If shader reuse is implemented: Measure throughput improvement

---

## Expected Outcomes

### Task 1 Results
- Last silence chunk (32ms) prepended to segments � recovers attack transients
- Trailing silence chunks (64-96ms) appended � captures release transients
- Faster gain ramp-up (�=0.7) � fewer lost chunks at speech onset
- Higher VAD threshold (0.4) � fewer false positives from noise
- Higher silence energy (1.5) � more natural segmentation

### Task 2 Results
- All segments meet NeMo's 100ms minimum � better model accuracy
- Pre-padding (96ms silence) � consistent acoustic context
- No hallucination from artificial silence � clean recognition
- Timing metadata preserved � GUI displays correct timestamps

### Combined Impact
- **Short words**: "go", "no", "yes" recognized with complete phonemes
- **Preliminary quality**: Faster, more accurate instant feedback
- **Final quality**: High-quality recognition maintained
- **User experience**: Smoother real-time transcription, fewer corrections needed

---

## Implementation Order

1.  Generate and review this plan
2. **Task 1 - Phase 1**: Write new tests (1.1-1.6)
3. **Task 1 - Phase 2**: Update existing tests (2.1-2.4)
4. **Task 1 - Phase 3**: Implement SoundPreProcessor changes (TDD cycle)
5. **Task 1 - Testing**: Manual verification with short words
6. **Task 2 - Phase 1**: Write new tests (2.1-2.7)
7. **Task 2 - Phase 2**: Update existing tests (2.8)
8. **Task 2 - Phase 3**: Implement Recognizer padding (TDD cycle)
9. **Integration Testing**: Full pipeline validation
10. **Documentation**: Update CLAUDE.md with new parameters and logic

---

## Risks and Mitigations

### Risk 1: Prepended silence chunk contains previous speech
**Mitigation**: Test with rapid speech scenarios (pause <300ms). If issues occur, add conditional logic based on RMS level.

### Risk 2: Silence padding causes hallucination
**Mitigation**: Test with real Parakeet model (Test 2.7). If hallucination occurs, reduce padding amount or adjust padding pattern.

### Risk 3: Trailing silence chunks add latency
**Mitigation**: With threshold=1.5, only ~64-96ms added. Acceptable trade-off for quality. Monitor in integration tests.

### Risk 4: Config changes break existing behavior
**Mitigation**: All config changes are backward-compatible (no removed parameters). Run full test suite before deployment.

---

## Success Criteria

-  All unit tests pass (both new and updated)
-  Integration tests show improved short word recognition
-  No regression in long utterance recognition quality
-  Latency increase < 100ms
-  No hallucinated text from padding
-  Code follows SOLID principles and TDD approach
-  Documentation updated with new parameters and behavior

---

## Notes

- This plan follows TDD strictly: tests written before implementation
- Each phase builds on the previous phase
- Tests are designed to verify behavior, not implementation details
