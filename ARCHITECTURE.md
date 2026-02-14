# STT Pipeline Architecture Documentation

## System Overview

The Speech-to-Text pipeline is a multi-threaded, queue-based architecture that processes audio from microphone to text output in real-time.

Application state management with pause/resume functionality is handled through an observer pattern, allowing components to react independently to state changes.

## Router Protocol Overview

`SpeechEndRouter` mediates flow between `SoundPreProcessor`, `Recognizer`, and `IncrementalTextMatcher`.
It enforces one in-flight `AudioSegment` at a time, assigns monotonic `message_id` values, forwards
`RecognitionTextMessage` payloads to matcher input, and unblocks on terminal `RecognizerAck`.
`SpeechEndSignal` boundaries are held until all in-flight audio for that utterance is acknowledged.

---

## Module Organization

The codebase is organized into domain-based subfolders within `src/`:

### Root (`src/`)
- **ApplicationState.py**: State manager with observer pattern
- **RecognitionResultPublisher.py**: Observer pattern publisher
- **Pipeline.py**: Main pipeline orchestration
- **types.py**: Data type definitions (AudioSegment, RecognitionResult)

### Audio Processing (`src/sound/`)
- **AudioSource.py**: Microphone audio capture with sounddevice callback
- **FileAudioSource.py**: WAV file audio source for testing
- **SoundPreProcessor.py**: VAD processing, RMS normalization, speech buffering

### Speech Recognition (`src/asr/`)
- **Recognizer.py**: STT model inference with context-aware recognition
- **AdaptiveWindower.py**: Aggregates segments into 3s recognition windows
- **VoiceActivityDetector.py**: Detects speech vs silence
- **ModelManager.py**: Model download and management
- **DownloadProgressReporter.py**: Model download progress tracking
- **ExecutionProviderManager.py**: GPU/CPU detection and selection
- **SessionOptionsFactory.py**: ONNX Runtime session configuration
- **SessionOptionsStrategy.py**: Hardware-specific optimization strategies

### Post-Processing (`src/postprocessing/`)
- **TextMatcher.py**: Overlap resolution, duplicate detection, result publishing
- **TextNormalizer.py**: Text normalization for fuzzy matching

### GUI (`src/gui/`)
- **TextDisplayWidget.py**: Tkinter text display with thread-safety
- **TextFormatter.py**: Text formatting logic (MVC controller)
- **ControlPanel.py**: Pause/Resume button widget
- **ModelDownloadDialog.py**: Model download dialog

### Controllers (`src/controllers/`)
- **PauseController.py**: MVC controller for pause/resume
- **InsertionController.py**: MVC controller for text insertion mode

### Quick Entry (`src/quickentry/`)
- **QuickEntryService.py**: Quick entry service orchestration
- **QuickEntryController.py**: MVC controller for quick entry
- **QuickEntrySubscriber.py**: Subscriber to recognition results
- **QuickEntryPopup.py**: Popup window for quick entry
- **GlobalHotkeyListener.py**: Global hotkey detection
- **FocusTracker.py**: Window focus tracking
- **windows_effects.py**: Windows-specific visual effects

---

## Architecture Diagram

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STT PIPELINE ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────┘

Microphone Input (16kHz, mono)
        │
        ├─ 32ms frames (512 samples)
        ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  TIER 1: Audio Capture                                                   │
│  Thread: AudioSource (daemon, sounddevice callback)                      │
├──────────────────────────────────────────────────────────────────────────┤
│  AudioSource.audio_callback()                                            │
│    ├─→ Captures raw audio (32ms frames, 512 samples)                    │
│    ├─→ Assigns chunk_id (monotonic counter)                             │
│    ├─→ chunk_queue.put_nowait({audio, timestamp, chunk_id})             │
│    ├─→ Observer: ApplicationState (component observer)                 │
│    │   └─→ on_state_change(): pause → close stream, resume → start()   │
│    └─→ FAST RETURN (<1ms) - no blocking operations!                     │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ├─ ApplicationState observes: state changes
        ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  APPLICATION STATE MANAGEMENT                                            │
├──────────────────────────────────────────────────────────────────────────┤
│  ApplicationState (shared object, dependency injection)                  │
│    ├─→ States: starting → running ⟷ paused → shutdown                   │
│    ├─→ Component observers: AudioSource, SoundPreProcessor, Recognizer, TextMatcher │
│    ├─→ GUI observers: ControlPanel                                      │
│    └─→ Thread-safe mutations (threading.Lock)                           │
│                                                                           │
│  ControlPanel (GUI widget, top of window)                               │
│    ├─→ Observes ApplicationState (GUI observer)                         │
│    ├─→ Button states: "Loading..." | "Pause" | "Resume"                 │
│    └─→ Delegates to PauseController.toggle()                            │
│                                                                           │
│  PauseController (MVC controller)                                       │
│    └─→ ONLY updates ApplicationState.set_state()                        │
│        └─→ NO component manipulation (proper encapsulation)             │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ├─ Raw audio dicts: {audio, timestamp, chunk_id}
        ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  Queue: chunk_queue (maxsize=200)                                        │
│  Data Type: dict                                                         │
│    - audio: npt.NDArray[np.float32]  (512 samples)                      │
│    - timestamp: float                                                    │
│    - chunk_id: int                                                       │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  TIER 2: Audio Processing                                                │
│  Thread: SoundPreProcessor (daemon)                                      │
├──────────────────────────────────────────────────────────────────────────┤
│  SoundPreProcessor.process()                                             │
│    ├─→ chunk_queue.get(timeout=0.1)  [raw audio dict]                   │
│    ├─→ _normalize_rms(audio)  [RMS normalization for VAD]               │
│    ├─→ VoiceActivityDetector.process_frame(normalized_audio)            │
│    │     └─→ Detects speech vs silence (threshold: 0.5)                 │
│    │         └─→ Segments speech at word boundaries                     │
│    │                                                                      │
│    ├─→ Context buffer management (circular buffer, 20 chunks = 640ms) │
│    │     ├─→ All chunks stored: speech + silence                       │
│    │     ├─→ Left context: snapshot captured when speech starts        │
│    │     └─→ Right context: extracted from trailing silence            │
│    │                                                                      │
│    ├─→ Speech buffering (_handle_speech_frame, _handle_silence_frame)   │
│    │     ├─→ Consecutive speech logic: 3 chunks to start segment       │
│    │     └─→ Accumulates chunks until finalization                      │
│    │                                                                      │
│    ├─→ _finalize_segment() [on silence threshold or max duration]       │
│    │     ├─→ Creates preliminary AudioSegment with context              │
│    │     │     └─→ type='preliminary'                                    │
│    │     │     └─→ chunk_ids=[id1, id2, ...]                            │
│    │     │     └─→ data: np.ndarray (speech only)                       │
│    │     │     └─→ left_context: pre-speech audio                       │
│    │     │     └─→ right_context: trailing silence                      │
│    │     │                                                                │
│    │     ├─→ Intelligent breakpoint splitting (max duration)            │
│    │     │     ├─→ _find_silence_breakpoint(): backward search          │
│    │     │     ├─→ Skips last 6 chunks for right_context                │
│    │     │     └─→ Preserves buffer continuity with overlap             │
│    │     │                                                                │
│    │     ├─→ speech_queue.put(preliminary_segment)  [instant output]    │
│    │     └─→ AdaptiveWindower.process_segment(segment) [sync call]      │
│    │           └─→ Aggregates into finalized windows                    │
│    │                                                                      │
│    ├─→ Observer: ApplicationState (component observer)                 │
│    │   └─→ on_state_change(): pause → flush() pending segments         │
│    │                                                                      │
│    └─→ Silence timeout detection → windower.flush()                     │
└──────────────────────────────────────────────────────────────────────────┘
        │
        │  AdaptiveWindower (NO THREAD - synchronous calls from SoundPreProcessor)
        │  ├─→ Aggregates preliminary segments into 3s windows
        │  ├─→ Creates finalized AudioSegment
        │  │     └─→ type='finalized'
        │  │     └─→ chunk_ids=[id1, id2, id3, ...]  (merged IDs)
        │  └─→ speech_queue.put(finalized_window)
        │
        ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  Queue: speech_queue (maxsize=200)                                       │
│  Data Type: AudioSegment (dataclass)                                     │
│    - type: 'preliminary' | 'finalized' | 'flush'                         │
│    - data: npt.NDArray[np.float32] (speech chunks only)                  │
│    - left_context: npt.NDArray[np.float32] (pre-speech audio)            │
│    - right_context: npt.NDArray[np.float32] (trailing silence)           │
│    - start_time: float                                                   │
│    - end_time: float                                                     │
│    - chunk_ids: list[int]                                                │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ├─ Preliminary AudioSegment (instant, low-quality)
        │  Finalized AudioSegment (delayed, high-quality)
        ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  TIER 3: Speech Recognition                                              │
│  Thread: Recognizer (daemon)                                             │
├──────────────────────────────────────────────────────────────────────────┤
│  Recognizer.process()                                                    │
│    ├─→ speech_queue.get(timeout=0.1)                                    │
│    ├─→ Context-aware recognition:                                       │
│    │     ├─→ Concatenate: left_context + data + right_context           │
│    │     ├─→ Recognize with timestamps (TimestampedResultsAsrAdapter)   │
│    │     └─→ Filter tokens by timestamp (exclude context hallucinations)│
│    │         └─→ Keep only tokens within data boundaries                │
│    │         └─→ 0.37s tolerance for VAD warm-up delay                  │
│    │                                                                      │
│    ├─→ Parakeet ONNX Model (nemo-parakeet-tdt-0.6b-v3)                 │
│    │     └─→ audio + context → text recognition with timestamps         │
│    │                                                                      │
│    └─→ Emits protocol messages                                           │
│          └─→ RecognitionTextMessage(result, message_id)                 │
│          └─→ RecognizerAck(message_id, ok/error)                        │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ├─ RecognitionTextMessage / RecognizerAck
        ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  Queue: recognizer_output_queue (maxsize=50)                             │
│  Data Type: RecognitionTextMessage | RecognizerAck                       │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  TIER 4: Text Processing + Display (MVC + Observer Pattern)             │
│  Thread: TextMatcher (daemon)                                            │
├──────────────────────────────────────────────────────────────────────────┤
│  TextMatcher.process()                                                   │
│    ├─→ text_queue.get(timeout=0.1)                                      │
│    ├─→ Routes by message type:                                          │
│    │                                                                      │
│    ├─ TEXT PATH (recognition messages):                                 │
│    │   └─→ process_item()                                               │
│    │       └─→ publisher.publish_partial_update(result)  [publishes]    │
│    │           └─→ RecognitionResultPublisher notifies subscribers      │
│    │               └─→ TextFormatter.on_partial_update(result)          │
│    │                   └─→ _calculate_partial_update() → DisplayInstructions │
│    │                       └─→ display.apply_instructions(instructions) │
│    │                           └─→ Thread-safe: schedules on main thread│
│    │                                                                      │
│    ├─ FINAL PATH (3s sliding windows):                                  │
│    │   └─→ process_finalized()                                          │
│    │       ├─→ Duplicate detection (time + normalized text)             │
│    │       ├─→ resolve_overlap() - handles windowed text overlaps       │
│    │       │     └─→ Uses TextNormalizer for fuzzy matching             │
│    │       │     └─→ Finds longest common subsequence                   │
│    │       └─→ publisher.publish_finalization(finalized_result)         │
│    │           └─→ RecognitionResultPublisher notifies subscribers      │
│    │               └─→ TextFormatter.on_finalization(result)            │
│    │                   └─→ _calculate_finalization() → DisplayInstructions │
│    │                       └─→ display.apply_instructions(instructions) │
│    │                           └─→ Thread-safe: schedules on main thread│
│    │                                                                      │
│    └─ BOUNDARY PATH (SpeechEndSignal):                                  │
│        └─→ process_speech_end()                                         │
│            └─→ publisher.publish_finalization(pending_result)           │
│                                                                           │
│  TextMatcher.finalize_pending()                                          │
│    └─→ Called by Pipeline.stop() to flush remaining partial text        │
│        └─→ publisher.publish_finalization(previous_result)              │
│                                                                           │
│  RecognitionResultPublisher (observer pattern, no thread):               │
│    ├─→ subscribe(subscriber) - register TextRecognitionSubscriber       │
│    ├─→ unsubscribe(subscriber) - unregister subscriber                  │
│    ├─→ publish_partial_update(result) - notify all subscribers          │
│    │     └─→ Thread-safe: copy subscriber list under lock               │
│    │         └─→ Call subscriber.on_partial_update(result)              │
│    │             └─→ Error isolation: exceptions logged, don't affect others │
│    ├─→ publish_finalization(result) - notify all subscribers            │
│    │     └─→ Thread-safe: copy subscriber list under lock               │
│    │         └─→ Call subscriber.on_finalization(result)                │
│    │             └─→ Error isolation: exceptions logged, don't affect others │
│    └─→ Subscribers: TextFormatter (GUI), Future: APISubscriber, etc.    │
│                                                                           │
│  TextFormatter (controller + subscriber, no thread):                     │
│    ├─→ Implements TextRecognitionSubscriber protocol                    │
│    ├─→ on_partial_update(result) → partial_update(result)               │
│    │     └─→ _calculate_partial_update() → DisplayInstructions          │
│    │         └─→ display.apply_instructions()                           │
│    ├─→ on_finalization(result) → finalization(result)                   │
│    │     └─→ _calculate_finalization() → DisplayInstructions            │
│    │         └─→ display.apply_instructions() (if not duplicate)        │
│    └─→ State: preliminary_results, finalized_chunk_ids, finalized_text  │
│                                                                           │
│  TextDisplayWidget (view, technical layer):                             │
│    ├─→ apply_instructions(instructions) - thread-safe entry point       │
│    │     └─→ Checks thread: main? direct call : root.after()            │
│    │         └─→ _apply_instructions_safe() - renders to widget         │
│    └─→ Handles GUI rendering + thread-safety internally                 │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ├─ Observer Pattern: TextMatcher → RecognitionResultPublisher → Subscribers
        ├─ MVC Pattern: TextFormatter → TextDisplayWidget
        ↓
   Tkinter GUI Window (main thread)
```

---

## Threading Model

### Thread Overview

| Component | Thread Type | Purpose | Blocking Operations |
|-----------|-------------|---------|---------------------|
| **AudioSource** | Daemon (sounddevice callback) | Captures microphone input only | `chunk_queue.put_nowait()` (non-blocking, drops if full) |
| **SoundPreProcessor** | Daemon | VAD processing, RMS normalization, speech buffering | `chunk_queue.get(timeout=0.1)`, `speech_queue.put()` |
| **AdaptiveWindower** | **NO THREAD** | Aggregates segments into windows | Called synchronously by SoundPreProcessor |
| **Recognizer** | Daemon | Runs STT model on audio segments | `speech_queue.get(timeout=0.1)`, model inference |
| **TextMatcher** | Daemon | Routes preliminary/finalized, overlap resolution, publishes results | `text_queue.get(timeout=0.1)`, calls publisher methods |
| **RecognitionResultPublisher** | **NO THREAD** | Observer pattern publisher (manages subscribers) | Called by TextMatcher, notifies subscribers with thread-safe copy |
| **TextFormatter** | **NO THREAD** | Formatting logic (controller + subscriber) | Implements TextRecognitionSubscriber protocol, triggers display updates |
| **TextDisplayWidget** | **NO THREAD** | GUI rendering (view) + thread-safety | Called by TextFormatter, schedules on main thread |
| **ApplicationState** | **NO THREAD** | Shared state manager (dependency injection) | Thread-safe mutations via `threading.Lock` |
| **PauseController** | **NO THREAD** | MVC controller (called by ControlPanel) | Updates ApplicationState only |
| **ControlPanel** | **NO THREAD** | GUI widget (runs on main thread) | Observes ApplicationState, delegates to controller |
| **Main Thread** | Main | Runs Tkinter event loop | `root.mainloop()` |

### Thread Communication

```
┌─────────────────────────────────────────────────────────────────────────┐
│  THREAD COMMUNICATION DIAGRAM                                            │
└─────────────────────────────────────────────────────────────────────────┘

Main Thread        AudioSource Thread    SoundPreProcessor Thread    Recognizer Thread
    │                     │                        │                         │
    │ start()             │                        │                         │
    ├────────────────────→│                        │                         │
    │                     │ start()                │                         │
    │                     ├───────────────────────→│                         │
    │                     │                        │ start()                 │
    │                     │                        ├────────────────────────→│
    │                     │                        │                         │
    │                     │ audio_callback()       │                         │
    │                     │   └─→ chunk_queue      │                         │
    │                     │       .put_nowait()    │                         │
    │                     │       (raw dict)       │                         │
    │                     │                        │                         │
    │                     │      chunk_queue       │                         │
    │                     │   ──────────────────→  │                         │
    │                     │                        │ process()               │
    │                     │                        │  ├─→ get raw chunk      │
    │                     │                        │  ├─→ normalize_rms()    │
    │                     │                        │  ├─→ VAD.process_frame()│
    │                     │                        │  ├─→ buffering logic    │
    │                     │                        │  ├─→ finalize_segment() │
    │                     │                        │  │   └─→ speech_queue   │
    │                     │                        │  │       .put(prelim)   │
    │                     │                        │  └─→ windower.process() │
    │                     │                        │      └─→ speech_queue   │
    │                     │                        │          .put(finalized)│
    │                     │                        │                         │
    │                     │                        │      speech_queue       │
    │                     │                        │   ──────────────────→   │
    │                     │                        │                         │ process()
    │                     │                        │                         │  ├─→ get segment
    │                     │                        │                         │  ├─→ recognize
    │                     │                        │                         │  └─→ text_queue.put()
    │                     │                        │                         │
    │                     │                        │                         ↓
    │                   TextMatcher Thread                            GUI Main Thread
    │                         │                                             │
    │                         │ process()                                   │ mainloop()
    │                         │  ├─→ get result                            │
    │                         │  ├─→ route by type                          │
    │                         │  ├─→ preliminary:                           │
    │                         │  │   publisher.publish_partial_update()     │
    │                         │  │     └─→ formatter.on_partial_update()    │
    │                         │  │         └─→ display.apply_instructions()─→│ root.after()
    │                         │  └─→ finalized:                             │
    │                         │      ├─resolve_overlap                      │
    │                         │      └─publisher.publish_finalization()     │
    │                         │          └─→ formatter.on_finalization()    │
    │                         │              └─→ display.apply_instructions()→│ root.after()
    │                         │                                             │
    │                         │      ApplicationState.set_state('paused')   │
    │                         │   ─────────────────────────────────────────→│
    │                         │                                             │
    │                         │   ← Notify observers ──┤                   │
    │                         │     ├─ AudioSource.on_state_change()        │
    │                         │     │   └─ stream.close()                   │
    │                         │     └─ SoundPreProcessor.on_state_change()  │
    │                         │         └─ flush()                          │
    │                         │                                             │
    │ stop()                  │                                             │
    ├─────────────────────────┤                                             │
    │  ├─→ audio_source.stop()│                                             │
    │  ├─→ sound_preprocessor.stop()                                        │
    │  ├─→ recognizer.stop()  │                                             │
    │  ├─→ text_matcher.finalize_pending()                                  │
    │  └─→ text_matcher.stop()│                                             │
    │                         │                                             │
```

---

## Data Flow with Types

### Type Definitions (`src/types.py`)

```python
@dataclass
class AudioSegment:
    """Unified type for preliminary/finalized audio segments + flush signals"""
    type: Literal['preliminary', 'finalized', 'flush']  # Discriminator
    data: npt.NDArray[np.float32]              # Audio samples [-1.0, 1.0], speech only
    left_context: npt.NDArray[np.float32]      # Pre-speech context audio
    right_context: npt.NDArray[np.float32]     # Post-speech context audio
    start_time: float                           # Segment start (seconds, data only)
    end_time: float                             # Segment end (seconds, data only)
    chunk_ids: list[int]                        # Tracking IDs

    # preliminary: chunk_ids = [id1, id2, ...]  (silence-bounded segments)
    # finalized:   chunk_ids = [id1, id2, id3, ...]  (aggregated windows)
    # flush:       chunk_ids = [last_id]  (end-of-segment signal with final chunk ID)

    # Note: AudioSegment.__post_init__ validates len(chunk_ids) >= 1

@dataclass
class RecognitionResult:
    """Speech recognition output with timing and context-filtered text"""
    text: str                    # Recognized text (tokens filtered by timestamp)
    start_time: float            # Recognition start
    end_time: float              # Recognition end
    chunk_ids: list[int]         # Tracking IDs from AudioSegment
```

### Data Transformation Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DATA TRANSFORMATION PIPELINE                                            │
└─────────────────────────────────────────────────────────────────────────┘

Raw Audio Frame - AudioSource
  └─→ dict: {audio, timestamp, chunk_id}
        │
        ↓ chunk_queue
        │
SoundPreProcessor
  ├─→ RMS Normalization + VAD Processing
  ├─→ Context Buffer (20 chunks = 640ms)
  └─→ Speech Buffering (3-chunk confirmation)
        │
        ↓ Segment Finalization
        │
AudioSegment (preliminary)
  └─→ type='preliminary', data + left_context + right_context
  └─→ chunk_ids: [42, 43, 44]
        │
        ├─→ speech_queue (instant)
        └─→ AdaptiveWindower.process_segment() (sync)
              │
              ↓ Window Aggregation (≥3s)
              │
        AudioSegment (finalized)
          └─→ type='finalized', data (3s window)
          └─→ chunk_ids: [42, ..., 94] (merged)
                │
                ↓ speech_queue
                │
        Recognizer
          ├─→ Concatenate contexts
          ├─→ Recognize with timestamps
          └─→ Filter tokens by timestamp
                │
                ↓
        RecognitionResult
          └─→ text: "hello world"
          └─→ chunk_ids: [42, 43, 44]
                │
                ↓ matcher_queue (via router)
                │
        TextMatcher
          ├─→ Preliminary: publish_partial_update()
          └─→ Final: resolve_overlap() → publish_finalization()
                │
                ↓ RecognitionResultPublisher
                │
        TextFormatter (subscriber)
          └─→ Calculate DisplayInstructions
                │
                ↓
        TextDisplayWidget
          └─→ Thread-safe render to tkinter
```

---

## Component Details

### Hardware Acceleration Architecture

Uses **Strategy Pattern** for hardware-specific ONNX Runtime optimizations.

**Components:**
- **ExecutionProviderManager**: DXGI-based GPU detection (discrete > integrated priority)
- **SessionOptionsFactory**: Creates hardware-specific strategy
- **SessionOptionsStrategy**: Configures session options per hardware type

**Hardware Configurations:**
- **Integrated GPU** (UMA): Zero-copy shared memory, disable CPU arena
- **Discrete GPU** (PCIe): Staging buffer with CPU arena
- **CPU**: Multi-threaded execution (auto-detect cores)

**Configuration:** `recognition.inference = "auto" | "directml" | "cpu"` with DirectML GPU → CPU fallback

---

### ApplicationState (State Manager)

**File:** `src/ApplicationState.py`

**Responsibility:** Central state manager with observer pattern for pause/resume

**State Machine:** `starting → running ⟷ paused → shutdown`

**Observer Types:**
- **Component observers**: Called directly (AudioSource, SoundPreProcessor)
- **GUI observers**: Scheduled via `root.after()` for thread safety (ControlPanel)

**Thread Safety:** Lock-protected state mutations, observers notified outside lock to prevent deadlocks

---

### PauseController (MVC Controller)

**File:** `src/controllers/PauseController.py`

**Responsibility:** Minimal controller that ONLY updates ApplicationState (proper MVC separation)

**Methods:** `pause()`, `resume()`, `toggle()` with guarded state transitions

---

### ControlPanel (GUI Widget)

**File:** `src/gui/ControlPanel.py`

**Responsibility:** Pause/Resume button widget observing ApplicationState

**Behavior:** Updates button text/state based on application state ("Loading..." → "Pause" → "Resume"), delegates clicks to PauseController

---

### 1. AudioSource

**Responsibility:** Capture 32ms audio frames (16kHz, 512 samples) from microphone

**Output:** Raw audio dicts `{audio, timestamp, chunk_id}` → `chunk_queue`

**Design:** Non-blocking callback (<1ms) prevents buffer overflows

**Observer Behavior:** On pause: close stream (release microphone). On resume: create fresh stream

### 2. SoundPreProcessor

**Responsibility:** VAD processing, RMS normalization, context-aware speech buffering, windower orchestration

**Key Features:**
- **RMS Normalization**: Temporal smoothing (AGC) before VAD
- **Context Buffer**: 20-chunk circular buffer (640ms) captures left/right context for better STT
- **Consecutive Speech Logic**: 3-chunk confirmation prevents false positives
- **Intelligent Breakpoint Splitting**: Backward search for natural silence boundaries when max duration reached
- **Observer Behavior**: On pause: flush pending segments to windower

**Input:** Raw audio dicts from `chunk_queue`
**Output:** Preliminary AudioSegments with context → `speech_queue`
**Side Effect:** Synchronously calls `windower.process_segment()` for finalized windows

### 3. AdaptiveWindower (NO THREAD)

**Responsibility:** Aggregate word-level segments into 3s recognition windows

**Strategy:**
- Accumulates preliminary segments until total duration ≥3s
- Emits finalized window when threshold reached
- Preserves trailing segments (≥1s) for overlap with next window

**Called synchronously by:** SoundPreProcessor
**Output:** Finalized AudioSegments → `speech_queue`

### 4. Recognizer

**Responsibility:** Convert audio to text using hardware-accelerated STT model

**Model:** Parakeet ONNX (nemo-parakeet-tdt-0.6b-v3, FP16) with `TimestampedResultsAsrAdapter`

**Hardware Acceleration:** DirectML GPU (auto-select) → CPU fallback (see Hardware Acceleration Architecture)

**Context-Aware Recognition:**
- Concatenates left_context + data + right_context for better acoustic modeling
- Filters tokens by timestamp to remove hallucinations from context regions
- 0.37s tolerance for VAD warm-up delay
- Benefits: Better short-word recognition (50-128ms), eliminates silence padding artifacts

**Input:** AudioSegments from `speech_queue`
**Output:** `RecognitionTextMessage` + terminal `RecognizerAck` via recognizer output queue

### 5. TextMatcher

**Responsibility:** Handle overlapping text from sliding windows, publish results via Observer pattern

**Key Features:**
- Overlap resolution using normalized text matching and longest common subsequence
- Duplicate detection (0.6s time threshold, normalized text comparison)
- Routes by message type: `RecognitionResult` overlap handling and `SpeechEndSignal` boundary finalization

**Input:** RecognitionResults from `text_queue`
**Output:** Publishes to `RecognitionResultPublisher` → notifies all subscribers

### 6. RecognitionResultPublisher (Observer Pattern)

**File:** `src/RecognitionResultPublisher.py`

**Responsibility:** Manage subscribers and publish text recognition events

**Protocol:** Subscribers implement `TextRecognitionSubscriber` (`on_partial_update()`, `on_finalization()`)

**Thread Safety:** Copy-on-notify pattern (lock-protected list, callbacks outside lock), error isolation per subscriber

**Subscribers:** TextFormatter (GUI), Future: APISubscriber, VoiceInputWriter, CommandAgent

### 7. TextFormatter (MVC Controller + Subscriber)

**File:** `src/gui/TextFormatter.py`

**Responsibility:** Text formatting logic (NO GUI knowledge), implements `TextRecognitionSubscriber`

**Logic:**
- Calculates DisplayInstructions from RecognitionResults
- Space insertion, paragraph breaks (2.0s threshold), chunk-ID filtering
- Duplicate detection and prevention

**Output:** Calls `display.apply_instructions()` to trigger GUI updates

### 8. TextDisplayWidget (MVC View)

**File:** `src/gui/TextDisplayWidget.py`

**Responsibility:** Render DisplayInstructions to tkinter widget with thread-safety

**Thread-Safety:** Detects calling thread, schedules on main thread via `root.after()` if needed

**Styles:** Final text (black, normal), Partial text (gray, italic)

---

## Design Principles

### Speech Recognition Pipeline Flow

1. **AudioSource** → captures 32ms frames (non-blocking) → raw audio dict → `chunk_queue`
2. **SoundPreProcessor** → reads `chunk_queue` → RMS normalization + VAD + buffering → preliminary `AudioSegment` → `speech_queue`
3. **SoundPreProcessor** → calls **AdaptiveWindower** synchronously → aggregates into finalized `AudioSegment` → `speech_queue`
4. **Recognizer** → reads `speech_queue` → processes both preliminary and finalized AudioSegments → `RecognitionResult` → `text_queue`
5. **TextMatcher** → filters/processes text → publishes via **RecognitionResultPublisher** → notifies subscribers (TextFormatter, etc.)

**Key Architecture:**
- Two-queue design: `chunk_queue` (raw audio) → `speech_queue` (AudioSegments)
- AudioSource callback is non-blocking (<1ms) to prevent buffer overflows
- SoundPreProcessor runs in dedicated thread, handling all audio processing
- AdaptiveWindower is called synchronously by SoundPreProcessor (not a separate thread)

### 1. Single Responsibility Principle (SRP)

Each component has one clear responsibility:
- **AudioSource:** Audio capture only (minimal callback)
- **SoundPreProcessor:** VAD, RMS normalization, speech buffering
- **AdaptiveWindower:** Window aggregation
- **Recognizer:** Speech recognition
- **TextMatcher:** Text processing and overlap resolution (publishes results)
- **RecognitionResultPublisher:** Subscriber management and event notification (Observer pattern)
- **TextFormatter:** Formatting logic (MVC controller + subscriber, NO GUI knowledge)
- **TextDisplayWidget:** GUI rendering + thread-safety (MVC view)

### 2. Open/Closed Principle (OCP)

Extension without modification via Observer pattern (add new subscribers without changing publisher) and interface-based design

### 3. Dependency Inversion Principle (DIP)

Components depend on abstractions (protocols, interfaces) not concrete implementations (e.g., TextRecognitionSubscriber protocol, windower interface)

### 4. Type Safety

Strongly-typed dataclasses with discriminated unions (`type: 'preliminary' | 'finalized'`) enable compile-time checking

### 5. Hardware Abstraction

Strategy Pattern isolates hardware optimizations (IntegratedGPUStrategy, DiscreteGPUStrategy, CPUStrategy) from main pipeline

### 6. Observer Pattern for State Management

ApplicationState notifies component observers (AudioSource, SoundPreProcessor) and GUI observers (ControlPanel) independently. PauseController only updates state, components react autonomously. Thread-safe: mutations locked, notifications outside lock.

### 7. Observer Pattern for Text Recognition Distribution

RecognitionResultPublisher decouples TextMatcher from subscribers (TextFormatter, future: APISubscriber). Uses Python Protocol for structural subtyping. Copy-on-notify prevents deadlocks, error isolation per subscriber.

### 8. MVC Pattern for Pause/Resume

**Model**: ApplicationState | **View**: ControlPanel | **Controller**: PauseController (state-only, no component manipulation)

Flow: `User → View → Controller → Model → Observers → Components`

### 9. MVC Pattern for Text Display

**Model**: RecognitionResult | **Controller**: TextFormatter (no GUI knowledge) | **View**: TextDisplayWidget (thread-safe rendering)

Flow: `TextMatcher → TextFormatter → TextDisplayWidget → tkinter`

---

## Conclusion

This architecture provides:

### Key Innovations

**Context Buffer Strategy:**
- 20-chunk circular buffer (640ms) captures pre/post-speech audio
- Left context snapshot immune to wraparound
- Right context extraction from trailing silence
- Enables better STT for short words (<100ms) without hallucinations

**Intelligent Breakpoint Splitting:**
- Backward search for natural silence boundaries
- Skips last 6 chunks to preserve right_context
- 6-chunk overlap maintains buffer continuity
- Eliminates mid-word cuts from max_speech_duration

**Timestamped Token Filtering:**
- Context concatenation: left_context + data + right_context
- Token-level timestamp alignment from TimestampedResultsAsrAdapter
- 0.37s tolerance for VAD warm-up delay
- Removes hallucinations ("yeah", "mm-hmm") from silence padding

**Consecutive Speech Logic:**
- 3-chunk confirmation prevents false positives
- Reduces transient noise triggering
- Preserves responsiveness while improving accuracy

**Observer-Based State Management:**
- ApplicationState as central subject with dual observer types
- Component observers called directly for background operations
- GUI observers scheduled via `root.after()` for thread safety
- Lock-protected state mutations with deadlock prevention
- Enables pause/resume without tight coupling between controller and components

**Observer-Based Text Recognition Distribution:**
- RecognitionResultPublisher decouples TextMatcher from consumers
- Protocol-based subscriber interface (TextRecognitionSubscriber)
- Thread-safe copy-on-notify pattern prevents deadlocks
- Error isolation: subscriber exceptions don't affect others
- Enables multiple output targets (GUI, API, voice input) from single pipeline
- Open for extension: add new subscribers without modifying existing code
