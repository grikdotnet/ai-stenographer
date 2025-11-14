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
│  TIER 1: Audio Capture                                                   │
│  Thread: AudioSource (daemon, sounddevice callback)                      │
├──────────────────────────────────────────────────────────────────────────┤
│  AudioSource.audio_callback()                                            │
│    ├─→ Captures raw audio (32ms frames, 512 samples)                    │
│    ├─→ Assigns chunk_id (monotonic counter)                             │
│    ├─→ chunk_queue.put_nowait({audio, timestamp, chunk_id})             │
│    └─→ FAST RETURN (<1ms) - no blocking operations!                     │
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
│    ├─→ Context buffer management (circular buffer, 48 chunks = 1.536s) │
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
│    │     │     ├─→ Skips last 3 chunks for right_context                │
│    │     │     └─→ Preserves buffer continuity with overlap             │
│    │     │                                                                │
│    │     ├─→ speech_queue.put(preliminary_segment)  [instant output]    │
│    │     └─→ AdaptiveWindower.process_segment(segment) [sync call]      │
│    │           └─→ Aggregates into finalized windows                    │
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
│    │         └─→ 0.3s tolerance for VAD warm-up delay                   │
│    │                                                                      │
│    ├─→ Parakeet ONNX Model (nemo-parakeet-tdt-0.6b-v3)                 │
│    │     └─→ audio + context → text recognition with timestamps         │
│    │                                                                      │
│    └─→ Creates RecognitionResult                                        │
│          └─→ text: str (filtered)                                       │
│          └─→ status: 'preliminary' | 'final' | 'flush'                  │
│          └─→ start_time, end_time: float                                │
│          └─→ chunk_ids: list[int]                                       │
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
│    - status: 'preliminary' | 'final' | 'flush'                           │
│    - chunk_ids: list[int]                                                │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  TIER 4: Text Processing + Display                                       │
│  Thread: TextMatcher (daemon)                                            │
├──────────────────────────────────────────────────────────────────────────┤
│  TextMatcher.process()                                                   │
│    ├─→ text_queue.get(timeout=0.1)                                      │
│    ├─→ Routes based on status field:                                    │
│    │                                                                      │
│    ├─ PRELIMINARY PATH (silence-bounded segments with context):         │
│    │   └─→ process_preliminary()                                        │
│    │       └─→ gui_window.update_partial(text)  [direct, no overlap]   │
│    │           └─→ root.after() → GUI main thread                       │
│    │                                                                      │
│    ├─ FINAL PATH (3s sliding windows):                                  │
│    │   └─→ process_finalized()                                          │
│    │       ├─→ Duplicate detection (time + normalized text)             │
│    │       ├─→ resolve_overlap() - handles windowed text overlaps       │
│    │       │     └─→ Uses TextNormalizer for fuzzy matching             │
│    │       │     └─→ Finds longest common subsequence                   │
│    │       ├─→ gui_window.finalize_text(finalized_part)                 │
│    │       │     └─→ root.after() → GUI main thread                     │
│    │       └─→ gui_window.update_partial(remaining_part)                │
│    │           └─→ root.after() → GUI main thread                       │
│    │                                                                      │
│    └─ FLUSH PATH (end of speech segment):                               │
│        └─→ process_flush()                                              │
│            └─→ Finalize pending partial text                            │
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
| **AudioSource** | Daemon (sounddevice callback) | Captures microphone input only | `chunk_queue.put_nowait()` (non-blocking, drops if full) |
| **SoundPreProcessor** | Daemon | VAD processing, RMS normalization, speech buffering | `chunk_queue.get(timeout=0.1)`, `speech_queue.put()` |
| **AdaptiveWindower** | **NO THREAD** | Aggregates segments into windows | Called synchronously by SoundPreProcessor |
| **Recognizer** | Daemon | Runs STT model on audio segments | `speech_queue.get(timeout=0.1)`, model inference |
| **TextMatcher** | Daemon | Routes preliminary/finalized, overlap resolution, GUI calls | `text_queue.get(timeout=0.1)`, GUI updates via `root.after()` |
| **GuiWindow** | **NO THREAD** | Helper for GUI updates (update_partial, finalize_text) | Called by TextMatcher |
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
    │                         │  │   update_partial()─────────────────────→│ root.after()
    │                         │  └─→ finalized:                             │
    │                         │      ├─resolve_overlap                      │
    │                         │      ├─finalize_text()─────────────────────→│ root.after()
    │                         │      └─update_partial()────────────────────→│ root.after()
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
    # flush:       chunk_ids = []  (end-of-segment signal)

@dataclass
class RecognitionResult:
    """Speech recognition output with timing and context-filtered text"""
    text: str                    # Recognized text (tokens filtered by timestamp)
    start_time: float            # Recognition start
    end_time: float              # Recognition end
    status: Literal['preliminary', 'final', 'flush']  # Type discriminator
    chunk_ids: list[int]         # Tracking IDs from AudioSegment
```

### Data Transformation Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DATA TRANSFORMATION PIPELINE                                            │
└─────────────────────────────────────────────────────────────────────────┘

Raw Audio Frame (32ms) - AudioSource
  └─→ np.ndarray: shape=(512,), dtype=float32, range=[-1.0, 1.0]
  └─→ dict: {audio, timestamp, chunk_id}
        │
        ↓ chunk_queue (raw audio)
        │
SoundPreProcessor Processing
  ├─→ RMS Normalization (for VAD input)
  ├─→ VAD Processing (Silero)
  ├─→ Circular Context Buffer (48 chunks = 1.536s)
  │     ├─→ left_context: snapshot before speech starts
  │     └─→ right_context: trailing silence extraction
  ├─→ Consecutive Speech Logic (3 chunks to start)
  └─→ Speech Buffering
        │
        ↓ Segment Finalization (silence-bounded or max duration)
        │
AudioSegment (preliminary)
  └─→ type='preliminary'
  └─→ data: np.ndarray (speech only, 64ms-3000ms)
  └─→ left_context: np.ndarray (pre-speech audio)
  └─→ right_context: np.ndarray (trailing silence)
  └─→ start_time: 1.234
  └─→ end_time: 1.456
  └─→ chunk_ids: [42, 43, 44]  ← accumulated chunk IDs
        │
        ├─→ speech_queue (instant path)
        │
        └─→ AdaptiveWindower.process_segment() (sync call)
              │
              ↓ Window Aggregation (3s windows, 1s step)
              │
        AudioSegment (finalized)
          └─→ type='finalized'
          └─→ data: np.ndarray (48000 samples = 3s @ 16kHz)
          └─→ left_context: np.ndarray (empty for finalized)
          └─→ right_context: np.ndarray (empty for finalized)
          └─→ start_time: 1.000
          └─→ end_time: 4.000
          └─→ chunk_ids: [42, 43, 44, 45, 46, 47, ...]  ← merged IDs
                │
                ↓ speech_queue (quality path)
                │
        Recognizer (Context-Aware Recognition)
          ├─→ Concatenate: left_context + data + right_context
          ├─→ Recognize with timestamps (TimestampedResultsAsrAdapter)
          └─→ Filter tokens by timestamp (exclude context)
                │
                ↓
        RecognitionResult (preliminary OR final OR flush)
          └─→ text: "hello world" (context-filtered)
          └─→ start_time: 1.234
          └─→ end_time: 1.456
          └─→ status: 'preliminary' | 'final' | 'flush'
          └─→ chunk_ids: [42, 43, 44]
                │
                ↓ text_queue
                │
        TextMatcher (routes by status)
          ├─→ Preliminary: gui_window.update_partial()
          ├─→ Final: resolve_overlap() → gui_window.finalize_text() + update_partial()
          └─→ Flush: finalize pending partial text
                │
                ↓ root.after() → GUI main thread
                │
        GUI Display (GuiWindow)
          └─→ Final text: black, normal (permanent)
          └─→ Partial text: gray, italic (temporary)
```

---

## Component Details

### Hardware Acceleration Architecture

The system uses a **Strategy Pattern** for hardware-specific optimizations:

**Components:**
- **ExecutionProviderManager**: Detects GPU type and builds ONNX Runtime provider list
- **SessionOptionsFactory**: Creates appropriate strategy based on GPU type
- **SessionOptionsStrategy**: Configures ONNX Runtime session options per hardware

**GPU Detection (DXGI-based):**
- Native DXGI API enumeration (IDXGIFactory::EnumAdapters) for ground-truth device_id mapping
- Multi-GPU priority: discrete > integrated
- Virtual GPU filtering (Microsoft Basic Render Driver, Remote Desktop adapters)
- Cached results for performance

**Hardware-Specific Configurations:**

| Hardware Type | Memory Architecture | Session Options |
|---------------|---------------------|-----------------|
| **Integrated GPU** | Shared RAM (UMA) | `enable_cpu_mem_arena=False` (zero-copy), threads=(1,1) |
| **Discrete GPU** | Dedicated VRAM (PCIe) | `enable_cpu_mem_arena=True` (staging buffer), threads=(1,1) |
| **CPU** | System RAM | `enable_cpu_mem_arena=True`, threads=(≤8,≤4) auto-detect |

**Configuration Options (`config/stt_config.json`):**
```json
{
  "recognition": {
    "inference": "auto"  // "auto" | "directml" | "cpu"
  }
}
```

**Fallback Chain:** DirectML GPU → CPU

---

### 1. AudioSource (Simplified)

**Responsibility:** Capture audio from microphone only

**Configuration:**
```json
{
  "audio": {
    "sample_rate": 16000,
    "chunk_duration": 0.032  // 32ms = 512 samples
  }
}
```

**Key Methods:**
- `audio_callback(indata, frames, time_info, status)` - Minimal callback (<1ms)
- `start()` - Start audio stream
- `stop()` - Stop audio stream

**Output:** Raw audio dicts to `chunk_queue`: `{audio, timestamp, chunk_id}`

**Design:** Non-blocking callback prevents buffer overflows

### 2. SoundPreProcessor

**Responsibility:** VAD processing, RMS normalization, context-aware speech buffering, windower orchestration

**Configuration:**
```json
{
  "audio": {
    "sample_rate": 16000,
    "rms_normalization": {
      "target_rms": 0.07,               // Boosted for better VAD response
      "silence_threshold": 0.001,
      "gain_smoothing": 0.7             // Faster gain ramp-up
    },
    "silence_energy_threshold": 3.0     // Higher threshold for more trailing silence
  },
  "vad": {
    "threshold": 0.55,                  // Tuned for confident speech detection
    "max_speech_duration_ms": 1500,     // 1.5s maximum
    "frame_duration_ms": 32
  },
  "windowing": {
    "silence_timeout": 0.5              // Trigger windower.flush()
  }
}
```

**Key Methods:**
- `process()` - Main loop reading from chunk_queue
- `_process_chunk(chunk_data)` - Process single raw chunk (dual-path: normalized→VAD, raw→STT)
- `_normalize_rms(audio)` - Apply RMS normalization with temporal smoothing (AGC)
- `_handle_speech_frame(audio, timestamp, speech_prob)` - Buffer speech with consecutive logic
- `_handle_silence_frame(audio, timestamp)` - Handle silence, buffer trailing context
- `_find_silence_breakpoint()` - **NEW:** Backward search for natural split points
- `_finalize_segment(breakpoint_idx)` - **ENHANCED:** Emit AudioSegment with context + intelligent splitting
- `flush()` - Emit pending segment and flush windower
- `start()` - Start processing thread
- `stop()` - Stop thread and flush

**Context Buffer Strategy:**
- Circular buffer: 48 chunks (1.536s @ 32ms/chunk)
- `left_context_snapshot`: Captured when speech starts (immune to wraparound)
- `right_context`: Extracted from trailing silence chunks (reverse iteration)
- Enables better STT quality for short words (<100ms)
- Prevents hallucinations from silence padding

**Consecutive Speech Logic:**
- Requires 3 consecutive speech chunks to start segment
- Prevents false positives from transient noise
- Chunks 1-3: Buffered in context_buffer as unconfirmed
- Chunk 3: Triggers segment start, extracts last 3 chunks

**Intelligent Breakpoint Splitting:**
- When max_speech_duration_ms reached, search backward for last silence chunk
- Skips last 3 chunks to preserve right_context
- Splits at natural silence boundaries (no mid-word cuts)
- Preserves buffer continuity with 6-chunk overlap
- Fallback to hard cut if no silence found

**Chunk Structure:**
- All chunks: `{'audio', 'timestamp', 'chunk_id', 'is_speech'}`
- chunk_id: Sequential for segment chunks, None for non-segment silence
- is_speech: Reflects VAD result (True/False), not placement logic

**Input:** Raw audio dicts from `chunk_queue`
**Output:** Preliminary AudioSegments with context to `speech_queue`
**Side Effect:** Calls `windower.process_segment()` synchronously

**Design:** Dedicated thread prevents blocking audio capture callback

### 3. AdaptiveWindower (NO THREAD)

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

**Output:** Finalized AudioSegments with `chunk_ids=[id1, id2, ...]` to `speech_queue`

### 4. Recognizer (Enhanced with Context-Aware Recognition)

**Responsibility:** Convert audio to text using STT model with hardware-accelerated inference and timestamped token filtering

**Model:** Parakeet ONNX (nemo-parakeet-tdt-0.6b-v3, FP16 quantized)
**Adapter:** `TimestampedResultsAsrAdapter` with `.with_timestamps()` support

**Hardware Acceleration:**
- **DirectML GPU** (default on Windows): Automatic GPU selection and optimization
- **CPU fallback**: Multi-threaded execution for systems without GPU support
- **Configuration**: `recognition.inference` = "auto" (default) | "directml" | "cpu"

**GPU Selection Strategy:**
- Prefers discrete GPUs (NVIDIA GeForce/RTX, AMD Radeon RX) over integrated
- Uses DXGI native enumeration for accurate device_id mapping
- Filters out virtual GPUs (Microsoft Basic Render Driver)
- **Multi-GPU systems**: Automatically selects discrete GPU for best performance

**Session Optimization (Strategy Pattern):**
- **Integrated GPU** (Intel Iris/UHD, AMD Vega): Zero-copy shared memory (UMA), disable CPU arena
- **Discrete GPU** (NVIDIA, AMD Radeon RX): PCIe staging buffer, enable CPU arena
- **CPU**: Multi-threaded execution (auto-detect with caps: intra≤8, inter≤4)

**Model Warm-up:**
- Pre-compiles GPU shaders on startup (~500ms) to avoid first-inference delay
- Skipped for CPU mode (no shader compilation needed)

**Key Methods:**
- `recognize_window(segment)` - **ENHANCED:** Context-aware STT with timestamp filtering
- `_filter_tokens_by_timestamp()` - **NEW:** Filter hallucinated context tokens
- `process()` - Main loop reading from speech_queue

**Context-Aware Recognition Algorithm:**
```python
segment = speech_queue.get(timeout=0.1)

# Step 1: Concatenate context for recognition
full_audio = left_context + data + right_context

# Step 2: Recognize with timestamps
result: TimestampedResult = model.recognize(full_audio)
# Returns: {text, tokens, timestamps}

# Step 3: Calculate data boundaries
data_start = len(left_context) / sample_rate
data_end = data_start + len(data) / sample_rate

# Step 4: Filter tokens by timestamp
filtered_text = _filter_tokens_by_timestamp(
    result.text, result.tokens, result.timestamps,
    data_start, data_end
)
# Includes 0.3s tolerance for VAD warm-up delay
# Removes hallucinations like "yeah", "mm-hmm" from silence

# Step 5: Map AudioSegment.type to RecognitionResult.status
status_map = {
    'preliminary': 'preliminary',
    'finalized': 'final',
    'flush': 'flush'
}

result = RecognitionResult(
    text=filtered_text,
    start_time=segment.start_time,
    end_time=segment.end_time,
    status=status_map[segment.type],
    chunk_ids=segment.chunk_ids
)
text_queue.put(result)
```

**Benefits:**
- Better recognition for short words (50-128ms) due to acoustic context
- Eliminates hallucinations from silence padding via timestamp filtering
- Preserves original timing (data region only, not context)
- Fallback to full text when timestamps unavailable
- Maintains backward compatibility with empty contexts

**Input:** AudioSegments with context from `speech_queue`
**Output:** RecognitionResults with `status` field to `text_queue`

### 5. TextMatcher

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

**Input:** RecognitionResults from `text_queue`
**Output:** Direct GUI calls via `gui_window.update_partial()` and `gui_window.finalize_text()`

### 6. GuiWindow

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
| `chunk_queue` | AudioSource | SoundPreProcessor | `dict` {audio, timestamp, chunk_id} | 200 | Raw audio chunks (32ms frames) |
| `speech_queue` | SoundPreProcessor, AdaptiveWindower | Recognizer | `AudioSegment` | 200 | Audio segments (preliminary + finalized) |
| `text_queue` | Recognizer | TextMatcher | `RecognitionResult` | 50 | Recognition results |

---

## Control Flow

### Startup Sequence

```
1. Pipeline.__init__()
   ├─→ Load config from config/stt_config.json
   ├─→ Create queues (chunk, speech, text)
   ├─→ Hardware acceleration setup:
   │     ├─→ ExecutionProviderManager.detect_gpu_type() [DXGI enumeration]
   │     ├─→ SessionOptionsFactory.get_strategy(gpu_type)
   │     └─→ Strategy.configure_session_options(sess_options)
   ├─→ Load Parakeet model with hardware-optimized session
   ├─→ Create Tkinter GUI window
   ├─→ Create GuiWindow (helper for TextMatcher)
   ├─→ Create VAD (for SoundPreProcessor)
   ├─→ Create components:
   │     ├─→ AdaptiveWindower (no thread, writes to speech_queue)
   │     ├─→ AudioSource (writes raw dicts to chunk_queue)
   │     ├─→ SoundPreProcessor (reads chunk_queue, writes to speech_queue)
   │     ├─→ Recognizer (reads speech_queue)
   │     └─→ TextMatcher (with gui_window reference)
   └─→ Setup signal handlers (Ctrl+C)

2. Pipeline.start()
   ├─→ _warmup_model()                 → pre-compile GPU shaders (DirectML/CUDA only)
   ├─→ audio_source.start()           → starts sounddevice stream
   ├─→ sound_preprocessor.start()     → spawns daemon thread
   ├─→ recognizer.start()              → spawns daemon thread
   └─→ text_matcher.start()            → spawns daemon thread

3. Pipeline.run()
   └─→ root.mainloop()                 → Tkinter event loop
```

### Shutdown Sequence

```
1. User presses Ctrl+C or closes window

2. Pipeline.stop()
   ├─→ audio_source.stop()
   │     └─→ Stop microphone stream (sounddevice)
   │
   ├─→ sound_preprocessor.stop()
   │     ├─→ Set is_running=False → thread exits
   │     ├─→ _finalize_segment() → emit pending segment
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

## Configuration Tuning

The system configuration has been tuned through iterative testing to optimize preliminary segment quality, reduce false positives, and improve short-word recognition.

### Configuration Parameters (`config/stt_config.json`)

```json
{
  "audio": {
    "sample_rate": 16000,
    "chunk_duration": 0.032,
    "silence_energy_threshold": 3.0,
    "rms_normalization": {
      "target_rms": 0.07,
      "silence_threshold": 0.0001,
      "gain_smoothing": 0.7
    }
  },
  "vad": {
    "frame_duration_ms": 32,
    "threshold": 0.55
  },
  "windowing": {
    "window_duration": 3.0,
    "step_size": 1.0,
    "max_speech_duration_ms": 1500,
    "silence_timeout": 0.5
  },
  "recognition": {
    "model_name": "parakeet",
    "inference": "auto"
  }
}
```

### Parameter Reference

| Parameter | Value | Purpose | Tuning Notes |
|-----------|-------|---------|--------------|
| **VAD** ||||
| `threshold` | 0.55 | Speech probability threshold | Higher = fewer false positives, may miss soft speech |
| `frame_duration_ms` | 32 | Silero VAD optimal frame size | Fixed by model requirements |
| **Audio Processing** ||||
| `silence_energy_threshold` | 3.0 | Cumulative silence for finalization | Higher = more trailing silence (better right_context) |
| `target_rms` | 0.07 | RMS normalization target | Higher = better VAD response, may boost noise |
| `gain_smoothing` | 0.7 | Temporal smoothing factor (α) | Lower = faster gain ramp-up, may cause artifacts |
| **Windowing** ||||
| `window_duration` | 3.0 | Recognition window size (s) | Trade-off: context vs latency |
| `step_size` | 1.0 | Window step (overlap) | Smaller = more overlap, higher quality |
| `max_speech_duration_ms` | 1500 | Force split threshold | Triggers intelligent breakpoint search |
| `silence_timeout` | 0.5 | Windower flush timeout (s) | Finalizes pending text after silence |
| **Recognition** ||||
| `inference` | "auto" | Hardware selection | "auto" \| "directml" \| "cpu" |

---

## Design Principles

### Core Pipeline Flow

1. **AudioSource** → captures 32ms frames (non-blocking) → raw audio dict → `chunk_queue`
2. **SoundPreProcessor** → reads `chunk_queue` → RMS normalization + VAD + buffering → preliminary `AudioSegment` → `speech_queue`
3. **SoundPreProcessor** → calls **AdaptiveWindower** synchronously → aggregates into finalized `AudioSegment` → `speech_queue`
4. **Recognizer** → reads `speech_queue` → processes both preliminary and finalized AudioSegments → `RecognitionResult` → `text_queue`
5. **TextMatcher** → filters/processes text, routes to GuiWindow for display

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
- **TextMatcher:** Text processing and display routing
- **GuiWindow:** GUI updates

### 2. Open/Closed Principle (OCP)

System is open for extension, closed for modification:
- Can add SilenceObserver without modifying existing classes
- TextMatcher exposes `finalize_pending()` as extension point

### 3. Dependency Inversion Principle (DIP)

High-level modules don't depend on low-level details:
- SoundPreProcessor receives windower interface, not concrete class
- TextMatcher doesn't know about timeout implementation
- AudioSource has no knowledge of VAD or windowing logic

### 4. Type Safety

Strong typing throughout:
- Dataclasses replace dicts
- Type discriminators (`type: 'preliminary' | 'finalized'`)
- Type hints enable IDE/mypy checking

### 5. Hardware Abstraction

Strategy Pattern for hardware-specific optimizations:
- **SessionOptionsStrategy**: Abstract interface for session configuration
- **Concrete strategies**: IntegratedGPUStrategy, DiscreteGPUStrategy, CPUStrategy
- **Factory**: SessionOptionsFactory creates appropriate strategy
- **Manager**: ExecutionProviderManager handles detection and provider selection
- **Separation**: Hardware logic isolated from main pipeline (OCP/DIP compliance)

---

## Testing Strategy

### Test Pyramid

```
                    ┌───────────────┐
                    │  Integration  │
                    │     Tests     │
                    └───────────────┘
                   ┌─────────────────┐
                   │  Component      │
                   │  Integration    │
                   └─────────────────┘
                  ┌───────────────────┐
                  │   Unit Tests      │
                  │   (Mocked deps)   │
                  └───────────────────┘
```

## Conclusion

This architecture provides:

✅ **Real-time performance** - Sub-250ms latency for preliminary results
✅ **High accuracy** - 3s windows for quality results
✅ **Context-aware recognition** - Circular buffer strategy improves short-word quality
✅ **Intelligent segmentation** - Silence-based breakpoint splitting preserves speech integrity
✅ **Hallucination prevention** - Timestamped token filtering eliminates context artifacts
✅ **Hardware acceleration** - Automatic GPU detection with CPU fallback
✅ **Smart GPU selection** - Multi-GPU support with discrete GPU preference
✅ **Memory optimization** - Hardware-specific session configurations (UMA vs PCIe)
✅ **Clean separation** - Each component has single responsibility
✅ **Type safety** - Strongly-typed dataclasses throughout
✅ **Extensibility** - Open for enhancement without modification
✅ **Testability** - tests covering all major paths
✅ **Maintainability** - Clear architecture, well-documented
✅ **Tunable configuration** - Iteratively optimized parameters with trade-off guidance

### Key Innovations

**Context Buffer Strategy:**
- 48-chunk circular buffer (1.536s) captures pre/post-speech audio
- Left context snapshot immune to wraparound
- Right context extraction from trailing silence
- Enables better STT for short words (<100ms) without hallucinations

**Intelligent Breakpoint Splitting:**
- Backward search for natural silence boundaries
- Skips last 3 chunks to preserve right_context
- 6-chunk overlap maintains buffer continuity
- Eliminates mid-word cuts from max_speech_duration

**Timestamped Token Filtering:**
- Context concatenation: left_context + data + right_context
- Token-level timestamp alignment from TimestampedResultsAsrAdapter
- 0.3s tolerance for VAD warm-up delay
- Removes hallucinations ("yeah", "mm-hmm") from silence padding

**Consecutive Speech Logic:**
- 3-chunk confirmation prevents false positives
- Reduces transient noise triggering
- Preserves responsiveness while improving accuracy

The dual-path design (preliminary + finalized) balances responsiveness with accuracy, while the Strategy Pattern for hardware abstraction ensures optimal performance across different GPU architectures and CPU-only systems. The context-aware recognition improvements deliver better short-word quality and eliminate hallucinations without sacrificing real-time performance.
