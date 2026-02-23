# Client-Server Architecture Migration Plan

## Overview

Transform the monolithic STT application into a distributed client-server architecture where:
- **Server**: Runs heavy ASR processing (VAD, Parakeet model, text matching) and exposes WebSocket API
- **Client**: Captures microphone audio, streams to server, displays results in GUI with text insertion, commands the server (start/stop/pause)

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


