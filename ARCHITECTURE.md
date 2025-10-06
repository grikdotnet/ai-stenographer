# STT Pipeline Architecture Documentation

## System Overview

The Speech-to-Text pipeline is a multi-threaded, queue-based architecture that processes audio from microphone to text output in real-time.

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
│  TIER 1: Audio Capture + Voice Activity Detection                        │
│  Thread: AudioSource (daemon)                                            │
├──────────────────────────────────────────────────────────────────────────┤
│  AudioSource.process_chunk_with_vad()                                    │
│    ├─→ VoiceActivityDetector (Silero VAD ONNX)                          │
│    │     └─→ Detects speech vs silence (threshold: 0.5)                 │
│    │         └─→ Segments speech at word boundaries (32ms silence)      │
│    │                                                                      │
│    ├─→ Creates preliminary AudioSegment                                  │
│    │     └─→ type='preliminary'                                          │
│    │     └─→ chunk_ids=[unique_id]  (single ID)                         │
│    │     └─→ data: np.ndarray (float32, -1.0 to 1.0)                   │
│    │                                                                      │
│    ├─→ chunk_queue.put(preliminary_segment)  [instant output]           │
│    └─→ AdaptiveWindower.process_segment(preliminary_segment) [sync call]│
└──────────────────────────────────────────────────────────────────────────┘
        │
        ├─ Preliminary AudioSegment (instant, low-quality)
        │  Finalized AudioSegment (delayed, high-quality)
        ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  Queue: chunk_queue (maxsize=100)                                        │
│  Data Type: AudioSegment (dataclass)                                     │
│    - type: 'preliminary' | 'finalized'                                   │
│    - data: npt.NDArray[np.float32]                                       │
│    - start_time: float                                                   │
│    - end_time: float                                                     │
│    - chunk_ids: list[int]                                                │
└──────────────────────────────────────────────────────────────────────────┘
        │
        │  AdaptiveWindower (NO THREAD - synchronous calls from AudioSource)
        │  ├─→ Aggregates preliminary segments into 3s windows
        │  ├─→ Creates finalized AudioSegment
        │  │     └─→ type='finalized'
        │  │     └─→ chunk_ids=[id1, id2, id3, ...]  (merged IDs)
        │  └─→ chunk_queue.put(finalized_window)
        │
        ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  TIER 2: Speech Recognition                                              │
│  Thread: Recognizer (daemon)                                             │
├──────────────────────────────────────────────────────────────────────────┤
│  Recognizer.process()                                                    │
│    ├─→ chunk_queue.get(timeout=0.1)                                     │
│    ├─→ is_preliminary = (segment.type == 'preliminary')                 │
│    ├─→ Parakeet ONNX Model (nemo-parakeet-tdt-0.6b-v3)                 │
│    │     └─→ audio → text recognition                                   │
│    └─→ Creates RecognitionResult                                        │
│          └─→ text: str                                                  │
│          └─→ is_preliminary: bool                                       │
│          └─→ start_time, end_time: float                                │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ├─ RecognitionResult (dataclass)
        ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  Queue: text_queue (maxsize=50)                                          │
│  Data Type: RecognitionResult (dataclass)                                │
│    - text: str                                                           │
│    - start_time: float                                                   │
│    - end_time: float                                                     │
│    - is_preliminary: bool                                                │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  TIER 3: Text Processing + Display                                       │
│  Thread: TextMatcher (daemon)                                            │
├──────────────────────────────────────────────────────────────────────────┤
│  TextMatcher.process()                                                   │
│    ├─→ text_queue.get(timeout=0.1)                                      │
│    ├─→ Routes based on is_preliminary flag:                             │
│    │                                                                      │
│    ├─ PRELIMINARY PATH (word-level VAD results):                        │
│    │   └─→ process_preliminary()                                        │
│    │       └─→ gui_window.update_partial(text)  [direct, no overlap]   │
│    │           └─→ root.after() → GUI main thread                       │
│    │                                                                      │
│    └─ FINALIZED PATH (3s sliding windows):                              │
│        └─→ process_finalized()                                          │
│            ├─→ Duplicate detection (time + normalized text)             │
│            ├─→ resolve_overlap() - handles windowed text overlaps       │
│            │     └─→ Uses TextNormalizer for fuzzy matching             │
│            │     └─→ Finds longest common subsequence                   │
│            ├─→ gui_window.finalize_text(finalized_part)                 │
│            │     └─→ root.after() → GUI main thread                     │
│            └─→ gui_window.update_partial(remaining_part)                │
│                └─→ root.after() → GUI main thread                       │
│                                                                           │
│  TextMatcher.finalize_pending()                                          │
│    └─→ Called by Pipeline.stop() to flush remaining partial text        │
│                                                                           │
│  GuiWindow (helper, no thread):                                          │
│    ├─→ update_partial(text) - gray/italic (temporary)                   │
│    └─→ finalize_text(text) - black/normal (permanent)                   │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ├─ Direct GUI calls (no queues)
        ↓
   Tkinter GUI Window (main thread)
```

---

## Threading Model

### Thread Overview

| Component | Thread Type | Purpose | Blocking Operations |
|-----------|-------------|---------|---------------------|
| **AudioSource** | Daemon | Captures microphone input, runs VAD | `chunk_queue.put()` (blocks if full) |
| **AdaptiveWindower** | **NO THREAD** | Aggregates segments into windows | Called synchronously by AudioSource |
| **Recognizer** | Daemon | Runs STT model on audio segments | `chunk_queue.get(timeout=0.1)`, model inference |
| **TextMatcher** | Daemon | Routes preliminary/finalized, overlap resolution, GUI calls | `text_queue.get(timeout=0.1)`, GUI updates via `root.after()` |
| **GuiWindow** | **NO THREAD** | Helper for GUI updates (update_partial, finalize_text) | Called by TextMatcher |
| **Main Thread** | Main | Runs Tkinter event loop | `root.mainloop()` |

### Thread Communication

```
┌─────────────────────────────────────────────────────────────────────────┐
│  THREAD COMMUNICATION DIAGRAM                                            │
└─────────────────────────────────────────────────────────────────────────┘

Main Thread                AudioSource Thread         Recognizer Thread
    │                            │                            │
    │ start()                    │                            │
    ├───────────────────────────→│                            │
    │                            │ start()                    │
    │                            ├───────────────────────────→│
    │                            │                            │
    │                            │ process_chunk_with_vad()   │
    │                            │   ├─→ VAD                  │
    │                            │   ├─→ create preliminary   │
    │                            │   ├─→ chunk_queue.put()    │
    │                            │   └─→ windower.process()   │
    │                            │        (sync call)         │
    │                            │                            │
    │                            │         chunk_queue        │
    │                            │   ──────────────────────→  │
    │                            │                            │ process()
    │                            │                            │  ├─→ get segment
    │                            │                            │  ├─→ recognize
    │                            │                            │  └─→ text_queue.put()
    │                            │                            │
    │                            │                            ↓
    │                         TextMatcher Thread           GUI Main Thread
    │                               │                          │
    │                               │ process()                │ mainloop()
    │                               │  ├─→ get result         │
    │                               │  ├─→ route by type       │
    │                               │  ├─→ preliminary:        │
    │                               │  │   update_partial()────┼─→ root.after()
    │                               │  └─→ finalized:          │
    │                               │      ├─resolve_overlap   │
    │                               │      ├─finalize_text()───┼─→ root.after()
    │                               │      └─update_partial()──┼─→ root.after()
    │                               │                          │
    │ stop()                        │                          │
    ├───────────────────────────────┤                          │
    │  ├─→ audio_source.stop()     │                          │
    │  ├─→ recognizer.stop()       │                          │
    │  ├─→ text_matcher.finalize_pending()                    │
    │  └─→ text_matcher.stop()     │                          │
    │                               │                          │
```

---

## Data Flow with Types

### Type Definitions (`src/types.py`)

```python
@dataclass
class AudioSegment:
    """Unified type for both preliminary and finalized audio segments"""
    type: Literal['preliminary', 'finalized']  # Discriminator
    data: npt.NDArray[np.float32]              # Audio samples [-1.0, 1.0]
    start_time: float                           # Segment start (seconds)
    end_time: float                             # Segment end (seconds)
    chunk_ids: list[int]                        # Tracking IDs

    # preliminary: chunk_ids = [single_id]
    # finalized:   chunk_ids = [id1, id2, id3, ...]

@dataclass
class RecognitionResult:
    """Speech recognition output with timing"""
    text: str                    # Recognized text
    start_time: float            # Recognition start
    end_time: float              # Recognition end
    is_preliminary: bool         # True=instant, False=high-quality
```

### Data Transformation Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DATA TRANSFORMATION PIPELINE                                            │
└─────────────────────────────────────────────────────────────────────────┘

Raw Audio Frame (32ms)
  └─→ np.ndarray: shape=(512,), dtype=float32, range=[-1.0, 1.0]
        │
        ↓ VAD Processing
        │
AudioSegment (preliminary)
  └─→ type='preliminary'
  └─→ data: np.ndarray (variable length, 64ms-3000ms)
  └─→ start_time: 1.234
  └─→ end_time: 1.456
  └─→ chunk_ids: [42]  ← single unique ID
        │
        ├─→ chunk_queue (instant path)
        │
        └─→ AdaptiveWindower.process_segment() (sync)
              │
              ↓ Window Aggregation (3s windows, 1s step)
              │
        AudioSegment (finalized)
          └─→ type='finalized'
          └─→ data: np.ndarray (48000 samples = 3s @ 16kHz)
          └─→ start_time: 1.000
          └─→ end_time: 4.000
          └─→ chunk_ids: [42, 43, 44, 45, 46, 47]  ← merged IDs
                │
                ↓ chunk_queue (quality path)
                │
        RecognitionResult (preliminary OR finalized)
          └─→ text: "hello world"
          └─→ start_time: 1.234
          └─→ end_time: 1.456
          └─→ is_preliminary: True/False
                │
                ↓ text_queue
                │
        TextMatcher (routes by is_preliminary)
          ├─→ Preliminary: gui_window.update_partial()
          └─→ Finalized: resolve_overlap() → gui_window.finalize_text() + update_partial()
                │
                ↓ root.after() → GUI main thread
                │
        GUI Display (GuiWindow)
          └─→ Final text: black, normal (permanent)
          └─→ Partial text: gray, italic (temporary)
```

---

## Component Details

### 1. AudioSource + VAD

**Responsibility:** Capture audio and detect speech at word level

**Configuration:**
```json
{
  "audio": {
    "sample_rate": 16000,
    "chunk_duration": 0.032  // 32ms = 512 samples
  },
  "vad": {
    "threshold": 0.5,              // 50% speech probability
    "min_speech_duration_ms": 64,  // 2 frames minimum
    "max_speech_duration_ms": 3000, // 3s maximum
    "silence_timeout_ms": 32        // 1 frame = word boundary
  }
}
```

**Key Methods:**
- `process_chunk_with_vad(chunk, timestamp)` - Process 32ms frame
- `flush_vad()` - Flush remaining VAD buffer
- `stop()` - Stop capture, flush VAD, call windower.flush()

**Output:** Preliminary AudioSegments with `chunk_ids=[single_id]`

### 2. AdaptiveWindower (NO THREAD)

**Responsibility:** Aggregate word-level segments into recognition windows

**Configuration:**
```json
{
  "windowing": {
    "window_duration": 3.0,  // 3 second windows
    "step_size": 1.0         // 1 second step = 33% overlap
  }
}
```

**Key Methods:**
- `process_segment(segment: AudioSegment)` - Add segment to buffer
- `flush()` - Emit final partial window

**Windowing Logic:**
```
Time: 0s        1s        2s        3s        4s        5s
      |---------|---------|---------|---------|---------|

Window 1: [0s ──────────────────── 3s]
Window 2:       [1s ──────────────────── 4s]
Window 3:             [2s ──────────────────── 5s]

Overlap:  [0s-1s] [1s-2s] [2s-3s] [3s-4s] [4s-5s]
          └─w1──┘ └─w1,w2┘ └─w1,w2,w3┘ └─w2,w3┘ └─w3──┘
```

**Output:** Finalized AudioSegments with `chunk_ids=[id1, id2, ...]`

### 3. Recognizer

**Responsibility:** Convert audio to text using STT model

**Model:** Parakeet ONNX (nemo-parakeet-tdt-0.6b-v3)

**Key Methods:**
- `recognize_window(segment, is_preliminary)` - Run STT inference
- `process()` - Main loop reading from chunk_queue

**Processing:**
```python
segment = chunk_queue.get(timeout=0.1)
is_preliminary = (segment.type == 'preliminary')
text = model.recognize(segment.data)

result = RecognitionResult(
    text=text,
    start_time=segment.start_time,
    end_time=segment.end_time,
    is_preliminary=is_preliminary
)
text_queue.put(result)
```

**Output:** RecognitionResults with `is_preliminary` flag

### 4. TextMatcher

**Responsibility:** Handle overlapping text from sliding windows

**Key Methods:**
- `process_text(text_data)` - Process single recognition result
- `resolve_overlap(window1, window2)` - Find common subsequence
- `finalize_pending()` - Flush remaining partial text

**Overlap Resolution:**
```
Window 1: "hello world how"
Window 2: "how are you"

Common: "how" (found via normalized matching)

Output:
  Final:   "hello world how"
  Partial: "are you"
```

**Duplicate Detection:**
- Normalizes text (punctuation, case, Unicode)
- Time threshold: 0.6s
- Skips duplicates from overlapping windows

**Output:** Direct GUI calls via `gui_window.update_partial()` and `gui_window.finalize_text()`

### 5. GuiWindow

**Responsibility:** Manage two-stage text display in GUI

**Key Methods:**
- `update_partial(text)` - Display temporary gray/italic text
- `finalize_text(text)` - Convert to permanent black/normal text
- Uses `root.after()` to marshal calls to main GUI thread

**Display Styles:**
- Final text: black, normal (permanent)
- Partial text: gray, italic (temporary, gets replaced)

---

## Queue Specifications

| Queue | Producer | Consumer | Data Type | Maxsize | Purpose |
|-------|----------|----------|-----------|---------|---------|
| `chunk_queue` | AudioSource, AdaptiveWindower | Recognizer | `AudioSegment` | 100 | Audio segments (preliminary + finalized) |
| `text_queue` | Recognizer | TextMatcher | `RecognitionResult` | 50 | Recognition results |

---

## Control Flow

### Startup Sequence

```
1. Pipeline.__init__()
   ├─→ Load config from config/stt_config.json
   ├─→ Load Parakeet model
   ├─→ Create Tkinter GUI window
   ├─→ Create GuiWindow (helper for TextMatcher)
   ├─→ Create queues (chunk, text)
   ├─→ Create components:
   │     ├─→ AdaptiveWindower (no thread)
   │     ├─→ AudioSource (with windower reference)
   │     ├─→ Recognizer
   │     └─→ TextMatcher (with gui_window reference)
   └─→ Setup signal handlers (Ctrl+C)

2. Pipeline.start()
   ├─→ audio_source.start()      → spawns daemon thread
   ├─→ recognizer.start()         → spawns daemon thread
   └─→ text_matcher.start()       → spawns daemon thread

3. Pipeline.run()
   └─→ root.mainloop()            → Tkinter event loop
```

### Shutdown Sequence

```
1. User presses Ctrl+C or closes window

2. Pipeline.stop()
   ├─→ audio_source.stop()
   │     ├─→ Stop microphone stream
   │     ├─→ flush_vad() → emit remaining segments
   │     └─→ windower.flush() → emit final window
   │
   ├─→ recognizer.stop()
   │     └─→ Set is_running=False → thread exits
   │
   ├─→ text_matcher.finalize_pending()
   │     └─→ Flush remaining partial text to GUI (gui_window.finalize_text())
   │
   └─→ text_matcher.stop()
         └─→ Set is_running=False → thread exits

3. All daemon threads exit
4. Main thread exits
```

---

## Performance Characteristics

### Latency Breakdown

| Stage | Latency | Type |
|-------|---------|------|
| Audio capture (32ms frame) | 32ms | Fixed |
| VAD processing | <5ms | Variable |
| Preliminary recognition | ~100-200ms | Variable (model) |
| Finalized recognition (3s window) | ~300-500ms | Variable (model) |
| Text processing | <10ms | Variable |
| GUI update | <5ms | Variable |
| **Total (preliminary path)** | **~150-250ms** | End-to-end |
| **Total (finalized path)** | **~350-550ms** | End-to-end |

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| Parakeet model | ~600MB | ONNX model in RAM |
| Silero VAD model | ~5MB | Quantized f16 model |
| Audio buffers | ~50KB | Ring buffers, queues |
| GUI | ~20MB | Tkinter overhead |
| **Total** | **~700MB** | Approximate |

---

## Design Principles

### 1. Single Responsibility Principle (SRP)

Each component has one clear responsibility:
- **AudioSource:** Audio capture + VAD
- **AdaptiveWindower:** Window aggregation
- **Recognizer:** Speech recognition
- **TextMatcher:** Text processing
- **DisplayHandler:** GUI updates

### 2. Open/Closed Principle (OCP)

System is open for extension, closed for modification:
- Can add SilenceObserver without modifying existing classes
- TextMatcher exposes `finalize_pending()` as extension point

### 3. Dependency Inversion Principle (DIP)

High-level modules don't depend on low-level details:
- AudioSource receives windower interface, not concrete class
- TextMatcher doesn't know about timeout implementation

### 4. Type Safety

Strong typing throughout:
- Dataclasses replace dicts
- Type discriminators (`type: 'preliminary' | 'finalized'`)
- Type hints enable IDE/mypy checking

---

## Future Enhancements

### SilenceObserver (Planned)

```python
class SilenceObserver:
    """Monitor VAD activity for phrase-level silence detection.

    Observes chunk_queue activity and triggers text finalization
    when no segments received for timeout period.
    """

    def __init__(self, chunk_queue, text_matcher, timeout=2.0):
        self.chunk_queue = chunk_queue
        self.text_matcher = text_matcher
        self.timeout = timeout

    def monitor(self):
        last_segment_time = time.time()
        while self.is_running:
            if not self.chunk_queue.empty():
                last_segment_time = time.time()
            elif time.time() - last_segment_time > self.timeout:
                # Trigger finalization
                self.text_matcher.finalize_pending()
                last_segment_time = time.time()
            time.sleep(0.1)
```

This can be added without modifying AudioSource, Recognizer, or TextMatcher (OCP compliance).

---

## Testing Strategy

### Test Pyramid

```
                    ┌───────────────┐
                    │  Integration  │  5 tests
                    │     Tests     │
                    └───────────────┘
                   ┌─────────────────┐
                   │  Component      │  20 tests
                   │  Integration    │
                   └─────────────────┘
                  ┌───────────────────┐
                  │   Unit Tests      │  20 tests
                  │   (Mocked deps)   │
                  └───────────────────┘
```

### Test Coverage

| Component | Unit Tests | Integration Tests | Total |
|-----------|------------|-------------------|-------|
| VoiceActivityDetector | 3 | - | 3 |
| AudioSource + VAD | - | 5 | 5 |
| AdaptiveWindower | 6 | - | 6 |
| Recognizer | 5 | - | 5 |
| TextMatcher | 10 | - | 10 |
| TextNormalizer | 2 | - | 2 |
| TwoStageDisplay | - | 6 | 6 |
| Full Pipeline | - | 5 | 5 |
| Enhanced Duplicate | - | 3 | 3 |
| **Total** | **26** | **19** | **45** |

### Test Data

- Real audio samples from Silero VAD (en.wav)
- No synthetic audio in tests
- Deterministic values (no random data)
- Type-safe dataclasses throughout

---

## Conclusion

This architecture provides:

✅ **Real-time performance** - Sub-250ms latency for preliminary results
✅ **High accuracy** - 3s windows for quality results
✅ **Clean separation** - Each component has single responsibility
✅ **Type safety** - Strongly-typed dataclasses throughout
✅ **Extensibility** - Open for enhancement without modification
✅ **Testability** - 45 tests covering all major paths
✅ **Maintainability** - Clear architecture, well-documented

The dual-path design (preliminary + finalized) balances responsiveness with accuracy, making the system suitable for real-time transcription use cases.
