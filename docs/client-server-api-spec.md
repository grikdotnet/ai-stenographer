# Client-Server WebSocket API Specification

## Purpose

This document defines the wire protocol between:

- STT client process (audio capture, GUI, text insertion, voice agent)
- STT server process (VAD, preprocessing, recognizer, matcher)

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
3. Server sends `server_state` snapshot (current server lifecycle state).
4. Client verifies `protocol_version` matches expected value; disconnects if not.
5. If `server_state=running`, client MAY send `audio_chunk` frames. Client MUST NOT send audio in any other state.
6. If `server_state=waiting_for_model`, client MAY send `list_models` to inspect available models, and optionally issue `download_model` command. The server will broadcast `server_state=running` when it becomes ready.
7. Server streams `recognition_result` messages.
8. Client may close its session gracefully with `control_command: close_session`, or disconnect by closing the socket. The server process is not affected.
9. For server-initiated graceful shutdown paths, server sends `session_closed` before closing the WebSocket connection.

Note: clients may receive broadcast messages (`server_state`, `download_progress`) at any time. The client MUST stop sending audio immediately if `server_state` transitions away from `running`.


## Server-Scoped vs Session-Scoped Messages

**Session-scoped messages** are delivered only to the connection that issued the request:
- `session_created`, `recognition_result`, `session_closed`, `error`, `model_list`

**Server-scoped messages** describe server-wide state and have no `session_id` field:
- `server_state` — connect-time snapshot + broadcast on every state change
- `download_progress` — broadcast to all active connections during a download

## Frame Types

## 1) Client -> Server: `audio_chunk` (binary frame)

Binary envelope:

- bytes `[0..3]`: unsigned 32-bit little-endian integer `header_len`
- bytes `[4..(4 + header_len - 1)]`: UTF-8 JSON header
- bytes `[4 + header_len..end]`: raw PCM bytes payload

The JSON header is human-readable.

Header JSON schema:

```json
{
  "type": "audio_chunk",
  "session_id": "uuid-string",
  "chunk_id": 42,
  "timestamp": 1735689600.123
}
```

Validation rules:

- `type` must be `audio_chunk`
- `session_id` must match active session (included in every frame as a defensive guard against client misconfiguration)

Failure behavior:

- malformed frame: send `error` with `INVALID_AUDIO_FRAME`
- repeated protocol violations: server may close connection

## 2) Client -> Server: `control_command` (JSON text frame)

Schema:

```json
{
  "type": "control_command",
  "session_id": "uuid-string",
  "command": "close_session | list_models | download_model",
  "model_name": "parakeet",
  "request_id": "optional-correlation-id",
  "timestamp": 1735689601.001
}
```

Field constraints:

- `command` enum
- `model_name` required when `command=download_model`; omitted otherwise
- `request_id` optional; meaningful for `list_models` and `download_model`; ignored for `close_session`

Per-command server response:

`close_session`: Drains session (see below); no direct response

Server stops accepting new audio chunks for this session, drains in-flight recognition (up to `max_window_duration`), sends any remaining `recognition_result` messages, sends `session_closed`, then closes the WebSocket connection. The server process and other active sessions are not affected.

`list_models`: Unicast `model_list` (echoes `request_id`)

`download_model` responses (exactly one):

Model already downloaded:
```json
{ "type": "model_status", "status": "ready", "request_id": "correlation-id" }
```

Download accepted and started (subsequent progress via `download_progress` broadcasts):
```json
{ "type": "model_status", "status": "downloading", "request_id": "correlation-id" }
```

Another download already running:
```json
{ "type": "error", "session_id": "uuid-string", "error_code": "DOWNLOAD_IN_PROGRESS", "message": "...", "fatal": false, "request_id": "correlation-id" }
```

Wrong model name:
```json
{ "type": "error", "session_id": "uuid-string", "error_code": "INVALID_MODEL_NAME", "message": "...", "fatal": false, "request_id": "correlation-id" }
```


## 3) Server -> Client: `session_created` (JSON text frame)

Schema:

```json
{
  "type": "session_created",
  "session_id": "uuid-string",
  "protocol_version": "v1",
  "server_time": 1735689600.101
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
  "token_confidences": [0.91, 0.82, 0.88]
}
```

Field constraints:

- `status` enum: `partial | final`
- `chunk_ids` list of non-negative integers
- `utterance_id` non-negative integer; increments on each speech/silence boundary; constant within one utterance across all its partial results
- `token_confidences` per-token confidence scores in range `[0.0, 1.0]`; may be empty list

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

- `close_session` — explicit `control_command: close_session` received
- `timeout` — session idle timeout exceeded
- `error` — fatal internal error forced session termination

Semantics:

- sent by the server before closing the WebSocket connection for graceful closure paths (`close_session`, `timeout`, `error`)
- client should stop sending audio and close its side of the connection upon receipt
- distinguishes graceful shutdown from unexpected connection drops

## 6) Server -> Client: `error` (JSON text frame)

Schema:

```json
{
  "type": "error",
  "session_id": "uuid-string",
  "error_code": "error code",
  "message": "audio payload bytes is not a multiple of 4",
  "fatal": false,
  "request_id": "optional-correlation-id"
}
```

Field constraints:

- `request_id` present only when the error is a direct response to a client command that included `request_id`; absent on server-initiated errors and errors not triggered by a command

`error_code` enum:

- `INVALID_AUDIO_FRAME`
- `UNKNOWN_MESSAGE_TYPE`
- `SESSION_ID_MISMATCH` — session_id in the frame does not match the active session
- `BACKPRESSURE_DROP`
- `PROTOCOL_VIOLATION`
- `INTERNAL_ERROR`
- `MODEL_NOT_READY` — audio received when `server_state` is not `running`; client must stop sending audio
- `DOWNLOAD_IN_PROGRESS` — `download_model` command received while another download is already running
- `INVALID_MODEL_NAME` — wrong model name provided in `download_model`

Fatal handling:

- if `fatal=true`, the server closes the WebSocket connection immediately after sending the error
- the session is destroyed server-side; the client must create a new connection to start a new session

## 7) Server -> Client: `server_state` (JSON text frame)

Schema:

```json
{
  "type": "server_state",
  "state": "starting|waiting_for_model|running|shutdown"
}
```

Field constraints:

- `state` enum: `starting | waiting_for_model | running | shutdown`

State meanings:

- `starting` — server process is initialising
- `waiting_for_model` — server is alive, but not accepting audio; waiting for a command to download a model 
- `running` — server process is up and ready to process audio
- `shutdown` — server is shutting down, closing all sessions 

Delivery: connect-time snapshot sent during session setup, after `session_created` + broadcast to all active connections on every state transition. No `request_id`.

## 8) Server -> Client: `model_list` (JSON text frame)

Schema:

```json
{
  "type": "model_list",
  "request_id": "optional-correlation-id",
  "models": [
    {
      "name": "parakeet",
      "display_name": "Parakeet TDT 0.6B v3",
      "size_description": "1.25 GB",
      "status": "downloaded"
    }
  ]
}
```

Field constraints:

- `request_id` echoed from the triggering `list_models` command; absent if the command carried no `request_id`
- model `status` enum: `downloaded | missing | downloading`

Delivery: unicast response to a `list_models` command; never broadcast.

## 9) Server -> Client: `download_progress` (JSON text frame)

Schema:

```json
{
  "type": "download_progress",
  "model_name": "parakeet",
  "progress": 0.45,
  "status": "downloading",
  "downloaded_bytes": 567000000,
  "total_bytes": 1250000000,
  "error_message": "optional — present only when status=error"
}
```

Field constraints:

- `status` enum: `downloading | complete | error`
- `progress` float in `[0.0, 1.0]`
- `error_message` present only when `status="error"`; this is the sole signal for async download failure — there is no separate error code for download failures

Delivery: broadcast to all active connections during a download; no `request_id`. The `error` message type is not used for async download failures; only `download_progress` with `status="error"` signals that a download job failed.

---

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

## Versioning and Compatibility

This is protocol version `v1`. Message types added in this revision: `server_state`, `model_list`, `download_progress`, and the new `list_models`/`download_model` command values.
