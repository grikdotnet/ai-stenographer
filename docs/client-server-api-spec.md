# Client-Server WebSocket API Specification

## Purpose

This document defines the wire protocol between:

- STT client process (audio capture, GUI, text insertion, voice agent)
- STT server process (VAD, preprocessing, recognizer, matcher)

Protocol goals:

- low-latency streaming audio upload
- structured recognition results
- deterministic validation and error handling

## Transport

- Protocol: WebSocket
- Encoding:
  - `audio_chunk`: binary WebSocket frame
  - all other messages: UTF-8 JSON text frames
- Time unit: seconds as `float`
- Audio format (v1): mono `float32` PCM, `16000` Hz

## Session Lifecycle

1. Client connects to WebSocket endpoint.
2. Server creates session and sends `session_created`.
3. Client verifies `protocol_version` matches expected value; disconnects if not.
4. Client starts sending `audio_chunk` frames.
5. Server streams `recognition_result` messages.
6. Either side can initiate shutdown (`control_command: shutdown` or socket close).
7. For server-initiated graceful shutdown paths, server sends `session_closed` before closing the WebSocket connection.

## Frame Types

## 1) Client -> Server: `audio_chunk` (binary frame)

Binary envelope:

- bytes `[0..3]`: unsigned 32-bit little-endian integer `header_len`
- bytes `[4..(4 + header_len - 1)]`: UTF-8 JSON header
- bytes `[4 + header_len..end]`: raw PCM bytes payload

Note: The JSON header is intentionally human-readable for debuggability. The per-frame overhead is ~150 bytes at 31 frames/sec (~4.7 KB/sec), negligible for local WebSocket. Binary optimization is deferred to v2.

Header JSON schema:

```json
{
  "type": "audio_chunk",
  "session_id": "uuid-string",
  "chunk_id": 42,
  "timestamp": 1735689600.123,
  "sample_rate": 16000,
  "num_samples": 512,
  "dtype": "float32",
  "channels": 1
}
```

Validation rules:

- `type` must be `audio_chunk`
- `session_id` must match active session (included in every frame as a defensive guard against client misconfiguration)
- `sample_rate` must be `16000`
- `dtype` must be `float32`
- `channels` must be `1`
- payload length must be exactly `num_samples * 4` bytes

Failure behavior:

- malformed frame: send `error` with `INVALID_AUDIO_FRAME`
- repeated protocol violations: server may close connection

## 2) Client -> Server: `control_command` (JSON text frame)

Schema:

```json
{
  "type": "control_command",
  "session_id": "uuid-string",
  "command": "shutdown",
  "request_id": "optional-correlation-id",
  "timestamp": 1735689601.001
}
```

Field constraints:

- `command` enum: `shutdown`
- `request_id` optional but recommended for tracing

Server response:

- `error` for invalid command/session

Shutdown drain semantics: when `command=shutdown`, the server stops accepting new audio chunks, drains all in-flight recognition (up to `max_window_duration`, typically 7 seconds), sends any remaining `recognition_result` messages, sends `session_closed`, then closes the WebSocket.

## 3) Server -> Client: `session_created` (JSON text frame)

Schema:

```json
{
  "type": "session_created",
  "session_id": "uuid-string",
  "protocol_version": "v1",
  "server_time": 1735689600.101,
  "server_config": {
    "sample_rate": 16000,
    "chunk_duration_sec": 0.032,
    "audio_dtype": "float32",
    "channels": 1
  }
}
```

Semantics:

- sent once after successful session creation
- client MUST disconnect with an error log if `protocol_version` doesn't match its expected value
- client should not stream audio before receiving this

## 4) Server -> Client: `recognition_result` (JSON text frame)

Schema:

```json
{
  "type": "recognition_result",
  "session_id": "uuid-string",
  "status": "partial",
  "text": "recognized text",
  "start_time": 12.544,
  "end_time": 13.888,
  "chunk_ids": [40, 41, 42],
  "utterance_id": 3,
  "confidence": 0.87,
  "token_confidences": [0.91, 0.82, 0.88],
  "audio_rms": 0.071,
  "confidence_variance": 0.0023
}
```

Field constraints:

- `status` enum: `partial | final`
- `chunk_ids` list of non-negative integers
- `utterance_id` non-negative integer; increments on each speech/silence boundary; constant within one utterance across all its partial results; optional field (omitted if unavailable)
- confidence fields in range `[0.0, 1.0]` when present; all confidence fields are optional

Mapping to local subscriber API:

- `status=partial` -> `on_partial_update`
- `status=final` -> `on_finalization`

## 5) Server -> Client: `session_closed` (JSON text frame)

Schema:

```json
{
  "type": "session_closed",
  "session_id": "uuid-string",
  "reason": "shutdown",
  "message": "optional human-readable detail"
}
```

`reason` enum:

- `shutdown` — explicit `control_command: shutdown` received
- `timeout` — session idle timeout exceeded
- `error` — fatal internal error forced session termination

Semantics:

- sent by the server before closing the WebSocket connection for server-initiated graceful shutdown paths (`shutdown`, `timeout`, `error`)
- client should stop sending audio and close its side of the connection upon receipt
- distinguishes graceful shutdown from unexpected connection drops

## 6) Server -> Client: `error` (JSON text frame)

Schema:

```json
{
  "type": "error",
  "session_id": "uuid-string",
  "error_code": "INVALID_AUDIO_FRAME",
  "message": "payload length does not match num_samples",
  "fatal": false
}
```

`error_code` enum (v1):

- `INVALID_AUDIO_FRAME`
- `UNKNOWN_MESSAGE_TYPE`
- `SESSION_NOT_FOUND`
- `BACKPRESSURE_DROP`
- `PROTOCOL_VIOLATION`
- `INTERNAL_ERROR`

Fatal handling:

- if `fatal=true`, the server closes the WebSocket connection immediately after sending the error
- the session is destroyed server-side; the client must create a new connection to start a new session

## Optional App-Level Keepalive

WebSocket ping/pong is transport-level and sufficient in most deployments.
If app-level heartbeat is needed, use:

```json
{ "type": "ping", "timestamp": 1735689605.123 }
{ "type": "pong", "timestamp": 1735689605.123 }
```

Pong echoes the original ping `timestamp` unchanged; client can compute RTT as `time.time() - original_timestamp`.

## Backpressure and Dropping Policy

- Client audio ingress queue full:
  - server may drop newest audio chunk
  - server should send `error` with `BACKPRESSURE_DROP`
- Result egress queue full:
  - server may drop partial results first
  - final results should be prioritized where possible

## Versioning

- This is protocol version `v1`.
- Backward-incompatible changes must bump version and be documented.
