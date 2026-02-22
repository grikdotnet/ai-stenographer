# Client-Server Architecture Migration Plan

## Overview

Transform the monolithic STT application into a distributed client-server architecture where:
- **Server**: Runs heavy ASR processing (VAD, Parakeet model, text matching) and exposes WebSocket API
- **Client**: Captures microphone audio, streams to server, displays results in GUI with text insertion

### User Requirements
- Audio capture: Client-side (each GUI captures local mic, streams PCM to server)
- Protocol: WebSocket for real-time bidirectional streaming
- Text insertion: Client-side (each client uses pynput locally)
- No backward compatibility needed (full migration to client-server)

---

## Architecture Summary

### Server Components
```
WebSocket Handler → ClientSession → Per-Session Pipeline:
  - WebSocketAudioReceiver (replaces AudioSource)
  - SoundPreProcessor (VAD + buffering)
  - Shared RecognizerPool (Parakeet model inference)
  - TextMatcher (overlap resolution)
  - RecognitionResultPublisher → WebSocket sender
```

**Threading Model:**
- WebSocket handler thread per client
- SoundPreProcessor thread per session
- Shared RecognizerPool (2-4 threads serving all clients)
- TextMatcher thread per session
- WebSocket sender thread per client

### Client Components
```
AudioSource → WebSocketClient → RecognitionResultPublisher:
  - AudioSource (mic capture, unchanged)
  - WebSocketClient (bidirectional WebSocket)
  - RecognitionResultPublisher (local subscriber registry)
  - ApplicationWindow (GUI, unchanged)
  - TextInsertionService (pynput, unchanged)
```

**Threading Model:**
- AudioSource callback thread (sounddevice)
- WebSocket sender thread (asyncio)
- WebSocket receiver thread (asyncio)
- GUI main thread (tkinter)

---

## WebSocket Protocol

### Message Types

**Client → Server:**

```json
// Audio chunk (every 32ms)
{
  "type": "audio_chunk",
  "session_id": "uuid",
  "audio": "base64-encoded float32 PCM",
  "timestamp": 1234567890.123,
  "chunk_id": 42
}

// Control command
{
  "type": "control_command",
  "session_id": "uuid",
  "command": "pause" | "resume" | "shutdown"
}
```

**Server → Client:**

```json
// Recognition result
{
  "type": "recognition_result",
  "session_id": "uuid",
  "text": "recognized text",
  "start_time": 1.5,
  "end_time": 3.2,
  "status": "preliminary" | "final" | "flush",
  "chunk_ids": [40, 41, 42],
  "confidence": 0.95,
  "token_confidences": [0.9, 0.98, ...],
  "audio_rms": 0.07,
  "confidence_variance": 0.02
}

// Session created (on connection)
{
  "type": "session_created",
  "session_id": "uuid",
  "server_config": {
    "sample_rate": 16000,
    "chunk_duration": 0.032
  }
}

// Error
{
  "type": "error",
  "session_id": "uuid",
  "error_code": "BACKPRESSURE" | "INVALID_AUDIO" | ...,
  "message": "Server queue full"
}
```

**Serialization:** JSON with base64-encoded audio (2.7KB per 32ms chunk = 168 Kbps overhead)

---

## Critical Files to Create/Modify

### Server Files (NEW)

**c:\workspaces\stt\server.py** - Server entry point
- WebSocket server using `websockets` library (asyncio)
- Manages SessionManager and RecognizerPool lifecycle
- Handles connection/disconnection events
- Routes messages to/from ClientSession instances

**c:\workspaces\stt\src\server\SessionManager.py** - Session lifecycle manager
- Thread-safe registry of active ClientSession instances
- `create_session(websocket)` - Initialize per-client pipeline components
- `get_session(session_id)` - Retrieve session by ID
- `remove_session(session_id)` - Cleanup threads and queues
- Session timeout handling (300s inactivity)

**c:\workspaces\stt\src\server\RecognizerPool.py** - Shared inference pool
- Loads Parakeet model once (1.25GB)
- Pool of 2-4 Recognizer threads serving all clients
- Shared `speech_queue` with session_id metadata
- Routes RecognitionResults back to per-session `text_queue`
- Prevents 1.25GB model duplication per client

**c:\workspaces\stt\src\server\ClientSession.py** - Session state dataclass
```python
@dataclass
class ClientSession:
    session_id: str
    websocket: WebSocket
    chunk_queue: queue.Queue
    speech_queue: queue.Queue
    text_queue: queue.Queue
    outbound_queue: queue.Queue
    sound_preprocessor: SoundPreProcessor
    text_matcher: TextMatcher
    publisher: RecognitionResultPublisher
    app_state: ApplicationState
    created_at: float
    last_activity: float
```

**c:\workspaces\stt\src\server\WebSocketAudioReceiver.py** - Replaces AudioSource
- Receives `audio_chunk` messages from WebSocket
- Decodes base64 → float32 numpy array
- Puts to `chunk_queue` (same interface as AudioSource)
- Validates chunk_id monotonicity

**c:\workspaces\stt\src\server\MessageSerializer.py** - JSON serialization
- `serialize_recognition_result(result: RecognitionResult) -> str`
- `deserialize_audio_chunk(msg: dict) -> dict` (audio: base64 → numpy)
- Handles all message type serialization/deserialization

**c:\workspaces\stt\config\server_config.json** - Server configuration
```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8765,
    "recognizer_pool_size": 2,
    "max_concurrent_sessions": 4,
    "session_timeout_seconds": 300
  }
}
```

### Client Files (NEW)

**c:\workspaces\stt\client.py** - Client entry point (replaces main.py)
- Initializes STTClient with server_url from config
- Creates AudioSource + WebSocketClient + GUI
- Manages client-side ApplicationState
- Handles window close → disconnect from server

**c:\workspaces\stt\src\client\WebSocketClient.py** - WebSocket communication
- Bidirectional WebSocket connection to server
- Sender loop: reads `chunk_queue` → serializes → sends `audio_chunk`
- Receiver loop: receives `recognition_result` → deserializes → publishes to local RecognitionResultPublisher
- Reconnection with exponential backoff (5 attempts, 1s initial delay)
- Synchronizes pause/resume state with server via `control_command`
- Registers as ApplicationState observer

**c:\workspaces\stt\src\client\AudioStreamEncoder.py** - Audio serialization
- Encodes numpy float32 → base64 string
- Adds chunk metadata (timestamp, chunk_id)
- Creates `audio_chunk` message dict

**c:\workspaces\stt\src\client\RecognitionResultDecoder.py** - Result deserialization
- Parses `recognition_result` JSON → RecognitionResult dataclass
- Validates required fields (text, start_time, end_time, status, chunk_ids)

**c:\workspaces\stt\config\client_config.json** - Client configuration
```json
{
  "client": {
    "server_url": "ws://localhost:8765",
    "reconnect_attempts": 5,
    "reconnect_backoff_ms": 1000,
    "audio_buffer_size": 200
  }
}
```

### Modified Files

**c:\workspaces\stt\src\types.py** - Add session metadata
```python
@dataclass
class AudioSegmentWithSession:
    """AudioSegment with session routing metadata for RecognizerPool."""
    segment: AudioSegment
    session_id: str
```

**c:\workspaces\stt\src\pipeline.py** → Delete (replaced by server.py and client.py)
- Current monolithic pipeline is fully replaced by client-server split
- No backward compatibility needed

**c:\workspaces\stt\main.py** → Delete (replaced by client.py)
- Entry point becomes client.py for GUI users
- server.py for running headless server

### Unchanged Files (Reused As-Is)

**Server side:**
- [src/sound/SoundPreProcessor.py](c:\workspaces\stt\src\sound\SoundPreProcessor.py) - Per-session instance
- [src/asr/VoiceActivityDetector.py](c:\workspaces\stt\src\asr\VoiceActivityDetector.py) - Used by SoundPreProcessor
- [src/asr/AdaptiveWindower.py](c:\workspaces\stt\src\asr\AdaptiveWindower.py) - Used by SoundPreProcessor
- [src/asr/Recognizer.py](c:\workspaces\stt\src\asr\Recognizer.py) - Used by RecognizerPool (minor routing changes)
- [src/postprocessing/TextMatcher.py](c:\workspaces\stt\src\postprocessing\TextMatcher.py) - Per-session instance
- [src/RecognitionResultPublisher.py](c:\workspaces\stt\src\RecognitionResultPublisher.py) - Per-session instance
- [src/ApplicationState.py](c:\workspaces\stt\src\ApplicationState.py) - Per-session instance

**Client side:**
- [src/sound/AudioSource.py](c:\workspaces\stt\src\sound\AudioSource.py) - Unchanged (mic capture)
- [src/gui/ApplicationWindow.py](c:\workspaces\stt\src\gui\ApplicationWindow.py) - Unchanged (GUI)
- [src/gui/TextDisplayWidget.py](c:\workspaces\stt\src\gui\TextDisplayWidget.py) - Unchanged (rendering)
- [src/gui/TextFormatter.py](c:\workspaces\stt\src\gui\TextFormatter.py) - Unchanged (subscribes to local publisher)
- [src/gui/TextInsertionService.py](c:\workspaces\stt\src\gui\TextInsertionService.py) - Unchanged (pynput)
- [src/quickentry/QuickEntryService.py](c:\workspaces\stt\src\quickentry\QuickEntryService.py) - Unchanged (hotkey)
- [src/controllers/*](c:\workspaces\stt\src\controllers) - All controllers unchanged

---

## Implementation Strategy

### Phase 1: Protocol Foundation (Week 1)
1. Add `AudioSegmentWithSession` to [src/types.py](c:\workspaces\stt\src\types.py:1)
2. Create [src/server/MessageSerializer.py](c:\workspaces\stt\src\server\MessageSerializer.py) with JSON serialization
3. Write `tests/test_message_serialization.py` (TDD approach)
4. Test serialization round-trip: RecognitionResult → JSON → RecognitionResult

### Phase 2: Server Core (Week 2)
1. Create [src/server/ClientSession.py](c:\workspaces\stt\src\server\ClientSession.py) dataclass
2. Write `tests/server/test_session_manager.py` (TDD)
3. Create [src/server/SessionManager.py](c:\workspaces\stt\src\server\SessionManager.py) with create/get/remove/cleanup
4. Write `tests/server/test_recognizer_pool.py` (TDD)
5. Create [src/server/RecognizerPool.py](c:\workspaces\stt\src\server\RecognizerPool.py) with shared queue routing
6. Write `tests/server/test_websocket_audio_receiver.py` (TDD)
7. Create [src/server/WebSocketAudioReceiver.py](c:\workspaces\stt\src\server\WebSocketAudioReceiver.py) (replaces AudioSource)

### Phase 3: Server Pipeline (Week 3)
1. Create [server.py](c:\workspaces\stt\server.py) entry point with asyncio WebSocket handler
2. Wire SessionManager + RecognizerPool + SoundPreProcessor/TextMatcher per session
3. Write `tests/server/test_server_integration.py` (single client connection)
4. Test server startup, session creation, and cleanup

### Phase 4: Client Core (Week 4)
1. Write `tests/client/test_websocket_client.py` (TDD)
2. Create [src/client/WebSocketClient.py](c:\workspaces\stt\src\client\WebSocketClient.py) with sender/receiver loops
3. Write `tests/client/test_audio_stream_encoder.py` (TDD)
4. Create [src/client/AudioStreamEncoder.py](c:\workspaces\stt\src\client\AudioStreamEncoder.py) (numpy → base64)
5. Write `tests/client/test_recognition_result_decoder.py` (TDD)
6. Create [src/client/RecognitionResultDecoder.py](c:\workspaces\stt\src\client\RecognitionResultDecoder.py) (JSON → dataclass)

### Phase 5: Client Pipeline (Week 5)
1. Create [client.py](c:\workspaces\stt\client.py) entry point (replaces main.py)
2. Wire AudioSource + WebSocketClient + ApplicationWindow + TextInsertionService
3. Test client startup, GUI rendering, and microphone capture
4. Write `tests/integration/test_audio_round_trip.py` (end-to-end: mic → server → GUI)

### Phase 6: Integration & Testing (Week 6)
1. Run real audio through client-server using `tests/fixtures/en.wav`
2. Test pause/resume synchronization (client ControlPanel → server → client sync)
3. Test reconnection with exponential backoff (kill server, restart, client reconnects)
4. Test concurrent clients (2-4 simultaneous connections)
5. Load testing: measure latency, queue depths, and throughput
6. Write `tests/migration/test_result_equivalence.py` (verify RecognitionResults match old pipeline)

### Phase 7: Configuration & Documentation (Week 7)
1. Create [config/server_config.json](c:\workspaces\stt\config\server_config.json) with defaults
2. Create [config/client_config.json](c:\workspaces\stt\config\client_config.json) with server_url
3. Update [ARCHITECTURE.md](c:\workspaces\stt\ARCHITECTURE.md) with client-server diagrams
4. Write SERVER_DEPLOYMENT.md (systemd service, Docker, firewall config)
5. Write CLIENT_SETUP.md (how to configure server_url)
6. Update [README.md](c:\workspaces\stt\README.md) with new usage instructions

### Phase 8: Cleanup (Week 8)
1. Delete [src/pipeline.py](c:\workspaces\stt\src\pipeline.py:1) (replaced by server.py/client.py)
2. Delete [main.py](c:\workspaces\stt\main.py:1) (replaced by client.py)
3. Update all test fixtures to use client-server mode
4. Verify all tests pass (pytest tests/)
5. Performance profiling (memory usage, CPU usage, latency)

---

## Testing Strategy (TDD Approach)

Per project requirements: **Tests written BEFORE implementation**.

### Unit Tests
- `tests/test_message_serialization.py` - JSON serialization/deserialization
- `tests/server/test_session_manager.py` - Session lifecycle
- `tests/server/test_recognizer_pool.py` - Shared pool routing
- `tests/server/test_websocket_audio_receiver.py` - Audio reception
- `tests/client/test_websocket_client.py` - WebSocket communication
- `tests/client/test_audio_stream_encoder.py` - Audio encoding
- `tests/client/test_recognition_result_decoder.py` - Result decoding

### Integration Tests (Real Audio)
Per project requirements: **Tests use real speech samples** (`tests/fixtures/en.wav`)

- `tests/integration/test_audio_round_trip.py`
  - Start server + client
  - Send `en.wav` audio → verify RecognitionResult text matches expected
- `tests/integration/test_pause_resume_sync.py`
  - Client pauses → verify server stops processing
  - Client resumes → verify server resumes
- `tests/integration/test_reconnection.py`
  - Kill server → verify client reconnects → verify no audio loss
- `tests/integration/test_concurrent_sessions.py`
  - 2-4 clients send different audio → verify no cross-talk

### Migration Equivalence Test
- `tests/migration/test_result_equivalence.py`
  - Run same audio through old pipeline (if preserved for testing) and new client-server
  - Verify RecognitionResults are identical (text, timing, chunk_ids)
  - Ensures no regressions in recognition quality

---

## SOLID Design Principles

### Single Responsibility Principle (SRP)
- **SessionManager**: Only manages session lifecycle (create/get/remove)
- **RecognizerPool**: Only handles model inference and routing
- **WebSocketAudioReceiver**: Only receives audio and puts to queue
- **MessageSerializer**: Only serializes/deserializes messages
- **WebSocketClient**: Only handles network communication (no GUI logic)

### Dependency Inversion Principle (DIP)
- WebSocketClient depends on `RecognitionResultPublisher` protocol (not concrete class)
- SessionManager depends on `AudioQueueProvider` protocol (not WebSocketAudioReceiver)
- RecognizerPool depends on `AsrModel` protocol (not Parakeet-specific)

### Open/Closed Principle (OCP)
- Add new message types without modifying WebSocketClient (message_handlers dict)
- Add new subscribers to RecognitionResultPublisher (observer pattern)

---

## Risk Mitigation

### Risk 1: Network Latency Degrades UX
**Mitigation:**
- Target LAN deployment (<5ms latency)
- Add latency metrics to `recognition_result` messages
- Show warning in GUI if latency > 100ms

### Risk 2: Server Crash Takes Down All Clients
**Mitigation:**
- Client auto-reconnect with exponential backoff (5 attempts, 1s initial)
- Server process supervisor (systemd, Docker restart policy)
- Health check endpoint (future: HTTP /health)

### Risk 3: Audio Desynchronization
**Mitigation:**
- Chunk IDs preserved end-to-end (client assigns, server echoes back)
- Timestamp validation (reject chunks older than 5 seconds)
- Out-of-order detection (log warning if chunk_id not monotonic)

### Risk 4: Memory Leak in Long-Running Sessions
**Mitigation:**
- Session timeout (300 seconds inactivity)
- Periodic queue draining (flush stale AudioSegments)
- Memory profiling during load tests

### Risk 5: Recognizer Pool Bottleneck
**Mitigation:**
- Configurable pool size (default 2, max 4)
- Queue depth monitoring (log warning if speech_queue > 80% full)
- Backpressure signaling to clients (`{"type": "backpressure"}`)

---

## Dependencies to Add

### Server Dependencies
```
websockets>=12.0  # Asyncio WebSocket server
```

### Client Dependencies
```
websockets>=12.0  # Asyncio WebSocket client
```

(All other dependencies unchanged: sounddevice, onnxruntime-directml, pynput, etc.)

---

## Expected Latency

**End-to-End Latency Breakdown:**
- Audio capture: 32ms (inherent)
- Network (LAN): <5ms one-way
- VAD processing: ~10ms
- Recognizer inference: 50-200ms (preliminary) / 300-500ms (finalized)
- **Total: ~100-600ms** (acceptable for real-time STT)

**Optimization opportunities:**
- Use WebSocket binary frames instead of JSON+base64 (10% bandwidth savings)
- Enable WebSocket compression (adds CPU overhead, test tradeoff)

---

## Verification Steps

After implementation, verify:
1. **Single client**: Send real audio (`en.wav`) → verify text recognition matches expected
2. **Multiple clients**: 2-4 clients send different audio → verify no cross-talk
3. **Pause/resume**: Client pauses → verify server stops → client resumes → verify no audio loss
4. **Reconnection**: Kill server → client reconnects → verify no data loss
5. **Latency**: Measure end-to-end latency (<600ms for finalized results)
6. **Memory**: Server runs 24h → verify no memory leaks (stable RSS)
7. **CPU**: Measure CPU usage (should be dominated by DirectML, not network I/O)
8. **Equivalence**: Run `tests/migration/test_result_equivalence.py` → verify results match old pipeline

---

## Out of Scope (Future Enhancements)

- Authentication/authorization (JWT tokens, API keys)
- TLS/WSS (encrypted WebSocket connections)
- Multi-language clients (JavaScript, Flutter)
- Binary WebSocket frames (replace base64)
- Server clustering (load balancer + multiple servers)
- REST API (for non-WebSocket clients)
- Prometheus metrics and Grafana dashboards
