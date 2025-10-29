# Refactoring Plan: Threaded Audio Processing (VAD-to-Windower Thread)

## Goal
Separate audio capture from audio processing to prevent blocking the sounddevice callback, which can cause buffer overflows and audio dropouts.

## Architecture Changes

### Current Flow (Branching!)
```
AudioSource.audio_callback() [sounddevice thread]
  � process_chunk_with_vad()
  � VAD processing (blocking!)
  � buffering logic
  � _finalize_segment() � chunk_queue (preliminary AudioSegment)
  � AdaptiveWindower.process_segment() � chunk_queue (finalized AudioSegment)
                                              �
                                         Both write here!

chunk_queue � Recognizer (gets both preliminary and finalized)
            � text_queue � TextMatcher
```

### New Flow (After Refactoring)
```
AudioSource.audio_callback() [sounddevice thread]
  � chunk_queue (raw audio chunks with metadata)
  � FAST RETURN (no blocking!)

SoundPreProcessor thread [NEW dedicated thread]
  reads chunk_queue (raw audio)
  � _normalize_rms()
  � VAD.process_frame()
  � buffering logic (_handle_speech_frame, _handle_silence_frame)
  � _finalize_segment() � speech_queue (preliminary AudioSegment)
  � AdaptiveWindower.process_segment() [synchronous call]
      � _emit_window() � speech_queue (finalized AudioSegment)
                              �
                         Both write here!

Recognizer thread [existing]
  reads speech_queue (AudioSegments: preliminary + finalized)
  � text_queue � TextMatcher
```

### Key Changes Summary

| Component | Before | After |
|-----------|--------|-------|
| **AudioSource** | Audio capture + VAD + buffering | Audio capture only (minimal callback) |
| **SoundPreProcessor** | N/A | NEW: VAD + buffering + windowing orchestration |
| **chunk_queue** | Carries AudioSegments (preliminary + finalized) | Carries raw audio dicts |
| **speech_queue** | N/A | NEW: Carries AudioSegments (preliminary + finalized) |
| **AdaptiveWindower** | Reads/writes chunk_queue | Writes to speech_queue (called by SoundPreProcessor) |
| **Recognizer** | Reads chunk_queue | Reads speech_queue |

## Test-Driven Development Plan

### Phase 1: Design Test Suite

#### 1.1 `tests/test_sound_preprocessor.py` - Core Processing Logic (NEW)

**Test fixtures:**
```python
@pytest.fixture
def sample_audio_chunk():
    """32ms chunk of audio at 16kHz (512 samples)"""
    return {
        'audio': np.random.randn(512).astype(np.float32) * 0.1,
        'timestamp': 1.0,
        'chunk_id': 1
    }

@pytest.fixture
def preprocessor_config():
    """Minimal config for SoundPreProcessor"""
    # Returns full config dict with audio/vad/windowing sections

@pytest.fixture
def mock_vad():
    """Mock VAD that returns configurable speech probabilities"""

@pytest.fixture
def mock_windower():
    """Mock AdaptiveWindower to verify synchronous calls"""
```

**Test cases (14 total):**

1. skip

2. **`test_rms_normalization_applied_to_audio()`**
   - Setup: Chunk with RMS=0.02
   - Assert: Output RMS closer to target (0.05)
   - Focus: RMS normalization logic

3. **`test_rms_normalization_skips_silence()`**
   - Setup: Silent chunk (RMS < 0.001)
   - Assert: Gain unchanged, no boosting
   - Focus: Silence detection in normalization

4. **`test_vad_called_with_normalized_audio()`**
   - Setup: Mock VAD
   - Assert: VAD.process_frame() receives normalized audio (not raw)
   - Focus: Dual-path processing

5. **`test_speech_frame_buffering()`**
   - Setup: VAD returns is_speech=True for 3 chunks
   - Assert: Speech buffer accumulates all 3
   - Focus: Speech buffering

6. **`test_preliminary_segment_emission_on_silence()`**
   - Setup: 3 speech chunks � silence chunk
   - Assert:
     - Preliminary AudioSegment in speech_queue
     - type='preliminary'
     - chunk_ids=[0,1,2]
   - Focus: Segment finalization

7. **`test_silence_energy_accumulation()`**
   - Setup: VAD probabilities [0.4, 0.3, 0.2]
   - Assert: Silence energy = (1-0.4) + (1-0.3) + (1-0.2) = 1.8
   - Focus: Cumulative silence detection

8. **`test_segment_finalization_at_energy_threshold()`**
   - Setup: Speech + silence until energy >= 1.5
   - Assert: Segment finalized when threshold reached
   - Focus: Energy threshold triggering

9. **`test_max_speech_duration_forces_split()`**
   - Setup: Continuous speech for > 3000ms
   - Assert: Segment finalized at max duration boundary
   - Focus: Max duration splitting

10. **`test_adaptive_windower_called_synchronously()`**
    - Setup: Mock windower
    - Assert: windower.process_segment() called with preliminary AudioSegment
    - Focus: Synchronous AdaptiveWindower invocation

11. **`test_silence_timeout_triggers_windower_flush()`**
    - Setup: Speech � silence > 0.5s
    - Assert: windower.flush() called
    - Focus: Silence timeout � flush logic

12. **`test_flush_emits_pending_segment()`**
    - Setup: Active speech buffer with 2 chunks
    - Action: Call flush()
    - Assert:
      - Preliminary AudioSegment emitted
      - windower.flush() called
      - Buffer cleared
    - Focus: Manual flush behavior

13. **`test_chunk_id_tracking_in_segments()`**
    - Setup: Chunks with IDs [10, 11, 12]
    - Assert: AudioSegment.chunk_ids == [10, 11, 12]
    - Focus: Chunk ID propagation

14. **`test_timestamp_accuracy()`**
    - Setup: Chunks at [1.0, 1.032, 1.064]
    - Assert:
      - start_time == 1.0
      - end_time == 1.064 + chunk_duration
    - Focus: Timestamp tracking

15. **`test_threading_starts_and_stops_cleanly()`**
    - Action: start() � stop()
    - Assert: Thread lifecycle correct
    - Focus: Thread management

#### 1.2 `tests/test_audiosource.py` - Simplified AudioSource (UPDATE)

**Test cases (4 total):**

1. **`test_audio_callback_puts_raw_chunk_to_queue()`**
   - Setup: Mock chunk_queue
   - Assert: Mock queue.put_nowait() called with dict {audio, timestamp, chunk_id}
   - Focus: Queue writing

2. **`test_chunk_id_increments()`**
   - Action: Trigger callback 3 times
   - Assert: chunk_ids are [0, 1, 2]
   - Focus: Chunk ID counter

3. **`test_no_processing_in_callback()`**
   - Assert: Callback completes in < 1ms
   - Focus: Fast callback return (no blocking)

4. **`test_audio_source_start_stop()`**
   - Action: start() � stop()
   - Assert: Stream lifecycle correct
   - Focus: Stream management

#### 1.3 `tests/test_integration_threaded_pipeline.py` - End-to-End (NEW)

**Test cases (3 total):**

1. **`test_raw_audio_flows_to_speech_queue()`**
   - Setup: Full pipeline with real VAD
   - Assert: Raw chunks � speech_queue contains AudioSegments
   - Focus: Queue flow correctness

2. **`test_both_preliminary_and_finalized_in_speech_queue()`**
   - Action: Process speech with silence gap
   - Assert:
     - Preliminary AudioSegment from SoundPreProcessor
     - Finalized AudioSegment from AdaptiveWindower
   - Focus: Branching output verification

3. **`test_recognizer_receives_from_speech_queue()`**
   - Setup: Mock Recognizer
   - Assert: Recognizer.recognize_window() called with speech_queue items
   - Focus: Recognizer integration

#### 1.4 Update Existing Tests

**`tests/test_adaptive_windower.py`** (RENAME QUEUE):
- Change `chunk_queue` � `speech_queue` in all tests
- Verify windower still writes finalized segments correctly

**`tests/test_recognizer.py`** (RENAME QUEUE):
- Change `chunk_queue` � `speech_queue` in all tests
- Verify recognizer reads from speech_queue

**`tests/test_audiosource_buffering.py`** (REMOVE):
- Logic moved to SoundPreProcessor
- Tests covered by `test_sound_preprocessor.py`
- **Action**: Delete this file

**`tests/test_audiosource_vad.py`** (REMOVE):
- VAD integration moved to SoundPreProcessor
- Tests covered by `test_sound_preprocessor.py`
- **Action**: Delete this file

**`tests/test_audiosource_rms_normalization.py`** (REMOVE):
- RMS normalization moved to SoundPreProcessor
- Tests covered by `test_sound_preprocessor.py`
- **Action**: Delete this file

**`tests/test_silence_energy.py`** (REMOVE):
- Silence energy logic moved to SoundPreProcessor
- Tests covered by `test_sound_preprocessor.py`
- **Action**: Delete this file

**`tests/test_integration_preliminary_finalized.py`** (UPDATE):
- Currently tests AudioSource → AdaptiveWindower → Recognizer
- Update to test SoundPreProcessor → AdaptiveWindower → Recognizer
- Change queue references: chunk_queue → speech_queue
- Update component instantiation to use SoundPreProcessor

### Phase 2: Implementation Order

1. **Create `tests/test_sound_preprocessor.py`** with all 15 test cases (TDD - tests fail initially)
2. **Create `src/SoundPreProcessor.py`** - implement to pass tests
3. **Update `src/AudioSource.py`** - simplify to minimal callback
4. **Update `tests/test_audiosource.py`** - verify simplified behavior with mocked queue
5. **Rename queues** in `AdaptiveWindower`, `Recognizer`
6. **Update `src/pipeline.py`** - wire up new components and queues
7. **Create `tests/test_integration_threaded_pipeline.py`** - verify end-to-end flow
8. **Update existing tests** - fix queue name changes, remove obsolete tests
9. **Run full test suite** - ensure all tests pass
10. **Manual testing** - verify with microphone, check for audio dropouts

### Phase 3: Implementation Details

#### 3.1 Raw Audio Chunk Format (chunk_queue)

```python
{
    'audio': np.ndarray,      # float32, 512 samples (32ms @ 16kHz)
    'timestamp': float,       # time.time() when captured
    'chunk_id': int          # monotonic counter starting at 0
}
```

#### 3.2 SoundPreProcessor Class Structure

**File**: `src/SoundPreProcessor.py`

```python
class SoundPreProcessor:
    """Processes raw audio chunks through VAD and buffering.

    Runs in dedicated thread to avoid blocking audio capture.
    Orchestrates VAD, speech buffering, and AdaptiveWindower.

    Args:
        chunk_queue: Queue to read raw audio chunks from AudioSource
        speech_queue: Queue to write AudioSegments (preliminary + finalized)
        vad: VoiceActivityDetector instance
        windower: AdaptiveWindower instance (called synchronously)
        config: Configuration dictionary
        verbose: Enable verbose logging
    """

    def __init__(self, chunk_queue, speech_queue, vad, windower, config, verbose):
        self.chunk_queue = chunk_queue      # INPUT: raw audio
        self.speech_queue = speech_queue    # OUTPUT: preliminary segments
        self.vad = vad                      # Called for speech detection
        self.windower = windower            # Called synchronously

        # Configuration
        self.sample_rate = config['audio']['sample_rate']
        self.frame_duration_ms = config['vad']['frame_duration_ms']
        self.max_speech_duration_ms = config['windowing']['max_speech_duration_ms']
        self.silence_energy_threshold = config['audio']['silence_energy_threshold']
        self.silence_timeout = config['windowing']['silence_timeout']

        # RMS normalization
        rms_config = config['audio']['rms_normalization']
        self.target_rms = rms_config['target_rms']
        self.rms_silence_threshold = rms_config['silence_threshold']
        self.gain_smoothing = rms_config['gain_smoothing']
        self.current_gain = 1.0

        # Speech buffering state
        self.speech_buffer = []
        self.is_speech_active = False
        self.speech_start_time = 0.0
        self.silence_energy = 0.0
        self.last_speech_timestamp = 0.0
        self.speech_before_silence = False

        # Threading
        self.is_running = False
        self.thread = None
        self.verbose = verbose

    def start(self):
        """Start processing thread"""
        self.is_running = True
        self.thread = threading.Thread(target=self.process, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop processing thread and flush pending segments"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.flush()

    def process(self):
        """Main loop: read chunk_queue � emit to speech_queue"""
        while self.is_running:
            try:
                chunk_data = self.chunk_queue.get(timeout=0.1)
                self._process_chunk(chunk_data)
            except queue.Empty:
                continue

    def _process_chunk(self, chunk_data: dict):
        """Process single raw audio chunk (was process_chunk_with_vad)"""
        audio = chunk_data['audio']
        timestamp = chunk_data['timestamp']
        chunk_id = chunk_data['chunk_id']

        # 1. Normalize audio for VAD
        normalized_audio = self._normalize_rms(audio)

        # 2. Call VAD with normalized audio
        vad_result = self.vad.process_frame(normalized_audio)
        speech_prob = vad_result['speech_probability']

        # 3. Handle speech/silence
        if vad_result['is_speech']:
            self.silence_energy = 0.0
            self.last_speech_timestamp = timestamp
            self.speech_before_silence = True
            self._handle_speech_frame(audio, timestamp, chunk_id)
        else:
            self.silence_energy += (1.0 - speech_prob)
            self._handle_silence_frame(audio, timestamp, chunk_id)

        # 4. Check silence timeout � flush windower
        if self.speech_before_silence:
            silence_duration = timestamp - self.last_speech_timestamp
            if silence_duration >= self.silence_timeout:
                self.windower.flush()
                self.speech_before_silence = False

    def _normalize_rms(self, audio: np.ndarray) -> np.ndarray:
        """Apply RMS normalization (AGC) to audio"""
        # ... (move from AudioSource) ...

    def _handle_speech_frame(self, audio: np.ndarray, timestamp: float, chunk_id: int):
        """Buffer speech frame, check max duration"""
        # ... (move from AudioSource) ...

    def _handle_silence_frame(self, audio: np.ndarray, timestamp: float, chunk_id: int):
        """Handle silence, finalize segment if threshold reached"""
        # ... (move from AudioSource) ...

    def _finalize_segment(self):
        """Emit preliminary AudioSegment and call windower"""
        if not self.speech_buffer:
            return

        # Create preliminary AudioSegment
        segment = AudioSegment(
            type='preliminary',
            data=np.concatenate([chunk['audio'] for chunk in self.speech_buffer]),
            start_time=self.speech_buffer[0]['timestamp'],
            end_time=self.speech_buffer[-1]['timestamp'] + frame_duration,
            chunk_ids=[chunk['chunk_id'] for chunk in self.speech_buffer]
        )

        # Emit to speech_queue
        self.speech_queue.put(segment)

        # Call windower synchronously
        self.windower.process_segment(segment)

        # Reset state
        self._reset_segment_state()

    def _reset_segment_state(self):
        """Reset buffering state"""
        # ... (move from AudioSource) ...

    def flush(self):
        """Emit pending segment and flush windower"""
        self._finalize_segment()
        self.windower.flush()
```

#### 3.3 Simplified AudioSource

**File**: `src/AudioSource.py`

**Keep:**
- Audio stream management
- Chunk ID counter
- Minimal `audio_callback()`

**Remove:**
- All VAD references
- All buffering logic
- All normalization logic
- AdaptiveWindower reference

**New constructor:**
```python
def __init__(self, chunk_queue, config, verbose=False):
    self.chunk_queue = chunk_queue  # OUTPUT: raw audio chunks
    self.sample_rate = config['audio']['sample_rate']
    self.chunk_size = int(self.sample_rate * config['audio']['chunk_duration'])
    self.chunk_id_counter = 0
    self.is_running = False
    self.stream = None
    self.verbose = verbose
```

**Simplified callback:**
```python
def audio_callback(self, indata, frames, time_info, status):
    """Minimal callback: capture audio and put to queue"""
    if status:
        logging.error(f"Audio error: {status}")

    audio_float = indata[:, 0].astype(np.float32)
    current_time = time.time()

    chunk_data = {
        'audio': audio_float,
        'timestamp': current_time,
        'chunk_id': self.chunk_id_counter
    }

    self.chunk_id_counter += 1

    try:
        self.chunk_queue.put_nowait(chunk_data)
    except queue.Full:
        logging.warning("chunk_queue full, dropping audio chunk")
```

#### 3.4 AdaptiveWindower Changes

**File**: `src/AdaptiveWindower.py`

**Changes:**
- Constructor parameter: `chunk_queue` � `speech_queue`
- Internal references: `self.chunk_queue` � `self.speech_queue`
- `_emit_window()`: Write to `self.speech_queue`
- `flush()`: Write to `self.speech_queue`

**No logic changes**, just queue renaming.

#### 3.5 Recognizer Changes

**File**: `src/Recognizer.py`

**Changes:**
- Constructor parameter: `chunk_queue` � `speech_queue`
- Internal references: `self.chunk_queue` � `self.speech_queue`
- `process()`: Read from `self.speech_queue`

**No logic changes**, just queue renaming.

#### 3.6 Pipeline Orchestration

**File**: `src/pipeline.py`

**Changes:**
```python
def __init__(self, ...):
    # Load config
    self.config = self._load_config(config_path)

    # Create queues
    self.chunk_queue = queue.Queue(maxsize=200)     # Raw audio chunks
    self.speech_queue = queue.Queue(maxsize=200)    # AudioSegments (prelim + final)
    self.text_queue = queue.Queue(maxsize=50)       # RecognitionResults

    # Create VAD (unchanged)
    self.vad = VoiceActivityDetector(
        config=self.config,
        model_path=vad_model_path,
        verbose=verbose
    )

    # Create AdaptiveWindower (renamed queue)
    self.adaptive_windower = AdaptiveWindower(
        speech_queue=self.speech_queue,
        config=self.config,
        verbose=verbose
    )

    # Create SoundPreProcessor (NEW)
    self.sound_preprocessor = SoundPreProcessor(
        chunk_queue=self.chunk_queue,
        speech_queue=self.speech_queue,
        vad=self.vad,
        windower=self.adaptive_windower,
        config=self.config,
        verbose=verbose
    )

    # Create simplified AudioSource
    self.audio_source = AudioSource(
        chunk_queue=self.chunk_queue,
        config=self.config,
        verbose=verbose
    )

    # Create Recognizer (renamed queue)
    self.recognizer = Recognizer(
        speech_queue=self.speech_queue,
        text_queue=self.text_queue,
        model=self.model,
        verbose=verbose
    )

    # Update components list
    self.components = [
        self.audio_source,
        self.sound_preprocessor,  # NEW
        self.recognizer,
        self.text_matcher
    ]
```

**Stop order (updated):**
```python
def stop(self):
    if self._is_stopped:
        return
    self._is_stopped = True

    logging.info("Stopping pipeline...")

    # 1. Stop audio capture
    self.audio_source.stop()

    # 2. Stop preprocessing (flushes pending segments)
    self.sound_preprocessor.stop()

    # 3. Stop recognition
    self.recognizer.stop()

    # 4. Finalize text
    logging.info("Finalizing pending text...")
    self.text_matcher.finalize_pending()

    # 5. Stop text processing
    self.text_matcher.stop()

    logging.info("Pipeline stopped.")
```

## Benefits

1. **Non-blocking audio callback**: Prevents buffer overflows and audio dropouts
2. **Better separation of concerns**:
   - AudioSource: Pure audio capture
   - SoundPreProcessor: VAD, normalization, buffering
   - AdaptiveWindower: Windowing logic (unchanged)
3. **Improved testability**: Can test preprocessing independently with mock queues
4. **Better performance**: Audio callback has minimal latency (<1ms)
5. **Foundation for optimizations**: Could add GPU-accelerated preprocessing in future

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Queue overflow if preprocessing is slow | Audio dropouts | Set maxsize=200, monitor in verbose mode |
| Timestamp accuracy issues | Text timing off | Pass timestamps with chunks, verify in tests |
| Test complexity increases | Slower development | Use TDD, create focused unit tests |
| Existing integration tests break | Broken CI/CD | Update tests incrementally, verify all pass |

## Testing Strategy

1. **Write tests first** (TDD approach)
2. **Implement SoundPreProcessor** to pass tests
3. **Simplify AudioSource** and verify tests still pass
4. **Update existing tests** for queue renaming
5. **Integration tests** to verify end-to-end flow
6. **Manual testing** with microphone to verify no audio issues

## Estimated Effort

- **Day 1**: Write all test cases (Phase 1)
- **Day 2-3**: Implement SoundPreProcessor and AudioSource changes (Phase 2)
- **Day 4**: Update pipeline, existing tests, integration tests (Phase 2)
- **Day 5**: Manual testing, debugging, verification (Phase 2-3)

**Total: 4-5 days**

## Success Criteria

- [ ] All 15 SoundPreProcessor tests pass
- [ ] All existing tests pass with queue renaming
- [ ] Integration tests verify end-to-end flow
- [ ] Audio callback executes in <1ms (measured)
- [ ] No audio buffer overflows during manual testing
- [ ] Recognition quality unchanged (baseline comparison)

---

**Status**: Planning complete, ready for implementation

**Next Step**: Phase 1 - Create `tests/test_sound_preprocessor.py`
