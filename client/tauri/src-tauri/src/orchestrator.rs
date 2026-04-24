/// Central coordinator that wires audio capture, WebSocket transport,
/// recognition fan-out, and text formatting into a running STT session.
///
/// Owns the lifecycle: connect -> start_audio -> pause/resume -> stop.
/// Emits events through an `EventEmitter` trait so the core logic is
/// testable without the Tauri runtime.
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tokio::time::Duration;

use crate::audio::{AudioSource, CpalAudioSource, FileAudioSource};
use crate::error::AppError;
use crate::formatting::{FinalizedUtterance, TextFormatter};
use crate::protocol::{AudioFrameEncoder, ServerMessage, ServerMessageDecoder};
use crate::recognition::{
    RecognitionFanOut, RecognitionResult, RecognitionStatus, RecognitionSubscriber,
};
use crate::state::{AppState, AppStateManager};
use crate::transport::WsClientTransport;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const EXPECTED_PROTOCOL_VERSION: &str = "v1";

// ---------------------------------------------------------------------------
// Event payloads (serialized to the Tauri frontend)
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Debug)]
pub struct ConnectionStatus {
    pub connected: bool,
    pub error: Option<String>,
}

#[derive(Clone, Serialize, Debug)]
pub struct TranscriptUpdate {
    pub action: String,
    pub utterances: Vec<FinalizedUtterance>,
    pub preliminary: String,
}

#[derive(Clone, Serialize, Debug)]
pub struct StateChanged {
    pub old: String,
    pub new: String,
}

#[derive(Clone, Serialize, Debug)]
pub struct InsertionStateChanged {
    pub enabled: bool,
}

#[derive(Clone, Serialize, Debug)]
pub struct ServerStateUpdate {
    pub state: String,
}

#[derive(Clone, Serialize, Debug)]
pub struct ModelInfoUpdate {
    pub name: String,
    pub display_name: String,
    pub size_description: String,
    pub status: String,
}

#[derive(Clone, Serialize, Debug)]
pub struct ModelListUpdate {
    pub models: Vec<ModelInfoUpdate>,
    pub request_id: Option<String>,
}

#[derive(Clone, Serialize, Debug)]
pub struct ModelStatusUpdate {
    pub status: String,
    pub request_id: Option<String>,
}

#[derive(Clone, Serialize, Debug)]
pub struct DownloadProgressUpdate {
    pub model_name: String,
    pub status: String,
    pub progress: Option<f64>,
    pub downloaded_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
    pub error_message: Option<String>,
}

// ---------------------------------------------------------------------------
// User-friendly error messages
// ---------------------------------------------------------------------------

/// Converts a low-level `AppError` into a short, user-readable connection error string.
///
/// Args:
///   err: The error returned by `connect()`.
///
/// Returns: A concise message suitable for display in the UI status banner.
pub fn friendly_connect_error(err: &AppError) -> String {
    let raw = err.to_string();
    let lower = raw.to_lowercase();
    if lower.contains("connection refused") || lower.contains("os error 10061") {
        "Cannot connect to server. Is the STT server running?".into()
    } else if lower.contains("timed out") {
        "Connection timed out. Check that the server address is correct.".into()
    } else if lower.contains("empty string") || lower.contains("invalid uri") {
        "No server URL configured. Set STT_SERVER_URL before starting.".into()
    } else if lower.contains("dns") || lower.contains("no such host") {
        "Server address not found. Check the server URL.".into()
    } else {
        "Connection failed. Check the server URL and try again.".into()
    }
}

// ---------------------------------------------------------------------------
// EventEmitter trait — abstracts Tauri's emit for testability
// ---------------------------------------------------------------------------

/// Abstraction for emitting named events with serializable payloads.
///
/// The Tauri implementation forwards to `AppHandle::emit()`; tests
/// can use `NoopEmitter` or a recording mock.
pub trait EventEmitter: Send + Sync + 'static {
    fn emit_serialized(&self, event: &str, payload: &str);
}

/// No-op emitter for headless / test contexts.
pub struct NoopEmitter;

impl EventEmitter for NoopEmitter {
    fn emit_serialized(&self, _event: &str, _payload: &str) {}
}

/// Emits an event through the provided `EventEmitter`, serializing the
/// payload to JSON. Silently drops serialization failures.
pub fn emit_event(emitter: &dyn EventEmitter, event: &str, payload: &impl Serialize) {
    if let Ok(json) = serde_json::to_string(payload) {
        emitter.emit_serialized(event, &json);
    }
}

fn emit_connection_status(
    connection_status: &Arc<Mutex<ConnectionStatus>>,
    emitter: &Arc<dyn EventEmitter>,
    connected: bool,
    error: Option<String>,
) {
    if let Ok(mut status) = connection_status.lock() {
        *status = ConnectionStatus {
            connected,
            error: error.clone(),
        };
    }
    emit_event(
        emitter.as_ref(),
        "stt://connection-status",
        &ConnectionStatus { connected, error },
    );
}

fn target_state_for_server_state(state_manager: &Arc<AppStateManager>, state: &str) -> AppState {
    match state {
        "running" => {
            if state_manager.current_state() == AppState::Paused {
                AppState::Paused
            } else {
                AppState::Running
            }
        }
        "shutdown" => AppState::Shutdown,
        "waiting_for_model" | "starting" => AppState::WaitingForServer,
        _ => AppState::WaitingForServer,
    }
}

fn set_state_if_needed(
    state_manager: &Arc<AppStateManager>,
    target: AppState,
) -> Result<(), AppError> {
    if state_manager.current_state() == target {
        return Ok(());
    }
    state_manager.set_state(target)
}

fn set_state_if_needed_or_log(
    state_manager: &Arc<AppStateManager>,
    target: AppState,
    reason: &str,
) {
    let current = state_manager.current_state();
    if current == target {
        tracing::trace!(
            "Skipping duplicate client state transition after {reason}: {current} -> {target}"
        );
        return;
    }
    if let Err(err) = state_manager.set_state(target) {
        tracing::warn!(
            "Failed client state transition after {reason}: {current} -> {target}: {err}"
        );
    }
}

fn should_surface_server_error_as_connection_error(error_code: &str, fatal: bool) -> bool {
    if fatal {
        return true;
    }
    !matches!(
        error_code,
        "MODEL_NOT_READY" | "DOWNLOAD_IN_PROGRESS" | "INVALID_MODEL_NAME"
    )
}

// ---------------------------------------------------------------------------
// ClientOrchestrator
// ---------------------------------------------------------------------------

/// Coordinates audio capture, server transport, recognition, and formatting.
///
/// Responsibilities:
/// - Creates and owns `WsClientTransport`, `TextFormatter`, `RecognitionFanOut`.
/// - Manages connect / start_audio / pause / resume / stop lifecycle.
/// - Dispatches server messages to the recognition pipeline.
/// - Emits frontend events via an `EventEmitter`.
pub struct ClientOrchestrator {
    server_url: String,
    input_file: Option<String>,
    state_manager: Arc<AppStateManager>,
    transport: Arc<tokio::sync::Mutex<WsClientTransport>>,
    formatter: Arc<TextFormatter>,
    fan_out: Arc<Mutex<RecognitionFanOut>>,
    emitter: Arc<dyn EventEmitter>,
    chunk_counter: Arc<AtomicI64>,
    audio_source: Option<Box<dyn AudioSource>>,
    connection_status: Arc<Mutex<ConnectionStatus>>,
}

impl ClientOrchestrator {
    /// Creates a new orchestrator wired to the given server URL.
    ///
    /// Args:
    ///   server_url: WebSocket URL of the STT server.
    ///   input_file: Optional WAV path to use instead of live microphone.
    ///   state_manager: Shared application state machine.
    pub fn new(
        server_url: String,
        input_file: Option<String>,
        state_manager: Arc<AppStateManager>,
    ) -> Self {
        let transport = Arc::new(tokio::sync::Mutex::new(WsClientTransport::new(
            server_url.clone(),
        )));
        let formatter = Arc::new(TextFormatter::new(None));
        let mut fan_out = RecognitionFanOut::new();
        fan_out.add_subscriber(Box::new(FormatterBridge(Arc::clone(&formatter))));

        Self {
            server_url,
            input_file,
            state_manager,
            transport,
            formatter,
            fan_out: Arc::new(Mutex::new(fan_out)),
            emitter: Arc::new(NoopEmitter),
            chunk_counter: Arc::new(AtomicI64::new(0)),
            audio_source: None,
            connection_status: Arc::new(Mutex::new(ConnectionStatus {
                connected: false,
                error: None,
            })),
        }
    }

    /// Sets the event emitter used for frontend notifications.
    pub fn set_emitter(&mut self, emitter: Arc<dyn EventEmitter>) {
        self.emitter = emitter;
    }

    /// Installs a callback invoked on every text change, replacing the current
    /// formatter and fan-out subscriber.
    ///
    /// Must be called before `connect()` so the new fan-out is captured by the
    /// message handler closure.
    ///
    /// Args:
    ///   cb: Zero-argument closure called whenever finalized or preliminary text changes.
    ///
    /// Returns: `Arc<TextFormatter>` pointing to the newly installed formatter, so
    ///          callers can query text from the same instance the callback is wired to.
    pub fn set_text_change_observer(&mut self, cb: Box<dyn Fn() + Send + Sync>) -> Arc<TextFormatter> {
        let formatter = Arc::new(TextFormatter::new(Some(cb)));
        let mut fan_out = RecognitionFanOut::new();
        fan_out.add_subscriber(Box::new(FormatterBridge(Arc::clone(&formatter))));
        self.formatter = formatter;
        self.fan_out = Arc::new(Mutex::new(fan_out));
        Arc::clone(&self.formatter)
    }

    /// Returns the configured server URL.
    pub fn server_url(&self) -> &str {
        &self.server_url
    }

    /// Returns the optional input file path.
    pub fn input_file(&self) -> Option<&str> {
        self.input_file.as_deref()
    }

    /// Returns the shared `TextFormatter` for reading accumulated text.
    pub fn formatter(&self) -> &Arc<TextFormatter> {
        &self.formatter
    }

    /// Injects a custom `AudioSource`, replacing the default cpal/file selection.
    ///
    /// Must be called before `start_audio()`. Intended for testing only.
    pub fn set_audio_source(&mut self, source: Box<dyn AudioSource>) {
        self.audio_source = Some(source);
    }

    /// Atomically returns the next chunk ID and increments the counter.
    pub fn next_chunk_id(&self) -> i64 {
        self.chunk_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Returns the latest connection status snapshot.
    pub fn current_connection_status(&self) -> ConnectionStatus {
        self.connection_status
            .lock()
            .map(|status| status.clone())
            .unwrap_or(ConnectionStatus {
                connected: false,
                error: Some("connection status unavailable".into()),
            })
    }

    /// Connects to the STT server over WebSocket.
    ///
    /// Algorithm:
    /// 1. Emit connecting status event.
    /// 2. Install message handler that decodes and dispatches server messages.
    /// 3. Call transport.connect() with a 10-second timeout.
    /// 4. On failure, emit error event and return Err.
    pub async fn connect(&mut self) -> Result<(), AppError> {
        emit_event(
            self.emitter.as_ref(),
            "stt://connection-status",
            &ConnectionStatus {
                connected: false,
                error: None,
            },
        );

        let fan_out = Arc::clone(&self.fan_out);
        let state_manager = Arc::clone(&self.state_manager);
        let transport_for_handler = Arc::clone(&self.transport);
        let emitter = Arc::clone(&self.emitter);
        let connection_status = Arc::clone(&self.connection_status);

        if let Ok(mut status) = self.connection_status.lock() {
            *status = ConnectionStatus {
                connected: false,
                error: None,
            };
        }

        {
            let transport = self.transport.lock().await;
            transport
                .set_message_handler(move |text: String| {
                    let msg = match ServerMessageDecoder::decode(&text) {
                        Ok(m) => m,
                        Err(e) => {
                            tracing::warn!("Failed to decode server message: {e}");
                            return;
                        }
                    };

                    match msg {
                        ServerMessage::SessionCreated {
                            session_id,
                            protocol_version,
                            ..
                        } => {
                            tracing::info!("SessionCreated received: session_id={session_id}");
                            if protocol_version != EXPECTED_PROTOCOL_VERSION {
                                let error_message =
                                    format!("Unsupported protocol version: {protocol_version}");
                                tracing::error!(
                                    "{error_message}"
                                );
                                emit_connection_status(
                                    &connection_status,
                                    &emitter,
                                    false,
                                    Some(error_message),
                                );
                                set_state_if_needed_or_log(
                                    &state_manager,
                                    AppState::Shutdown,
                                    "protocol version mismatch",
                                );
                                let transport_for_stop = Arc::clone(&transport_for_handler);
                                tokio::spawn(async move {
                                    transport_for_stop.lock().await.stop(true).await;
                                });
                                return;
                            }
                            if let Ok(transport) = transport_for_handler.try_lock() {
                                transport.set_session_id(session_id.clone());
                                tracing::info!("session_id set on transport");
                            } else {
                                tracing::error!("transport try_lock failed — session_id NOT set");
                            }
                            set_state_if_needed_or_log(
                                &state_manager,
                                AppState::WaitingForServer,
                                "session_created",
                            );
                            emit_connection_status(
                                &connection_status,
                                &emitter,
                                true,
                                None,
                            );
                        }
                        ServerMessage::RecognitionResult {
                            session_id,
                            status,
                            text,
                            start_time,
                            end_time,
                            chunk_ids,
                            utterance_id,
                            token_confidences,
                        } => {
                            let rr_status = match status.as_str() {
                                "partial" => RecognitionStatus::Partial,
                                "final" => RecognitionStatus::Final,
                                _ => return,
                            };
                            let rr = RecognitionResult {
                                session_id,
                                status: rr_status.clone(),
                                text,
                                start_time,
                                end_time,
                                chunk_ids,
                                utterance_id,
                                token_confidences,
                            };
                            if let Ok(fo) = fan_out.lock() {
                                match rr_status {
                                    RecognitionStatus::Partial => fo.on_partial(&rr),
                                    RecognitionStatus::Final => fo.on_finalization(&rr),
                                }
                            }
                        }
                        ServerMessage::Ping {} => {
                            tracing::trace!("Received ping from server");
                        }
                        ServerMessage::ServerState { state } => {
                            emit_event(
                                emitter.as_ref(),
                                "stt://server-state",
                                &ServerStateUpdate {
                                    state: state.clone(),
                                },
                            );
                            let target_state = target_state_for_server_state(&state_manager, &state);
                            set_state_if_needed_or_log(
                                &state_manager,
                                target_state,
                                "server_state update",
                            );
                        }
                        ServerMessage::SessionClosed { reason, .. } => {
                            tracing::info!("Session closed by server: {reason}");
                            set_state_if_needed_or_log(
                                &state_manager,
                                AppState::Shutdown,
                                "session_closed",
                            );
                        }
                        ServerMessage::Error {
                            error_code,
                            message,
                            fatal,
                            ..
                        } => {
                            if should_surface_server_error_as_connection_error(&error_code, fatal) {
                                tracing::error!(
                                    "Server error: code={error_code} fatal={fatal} message={message}"
                                );
                                emit_connection_status(
                                    &connection_status,
                                    &emitter,
                                    false,
                                    Some(message),
                                );
                            } else {
                                tracing::debug!(
                                    "Nonfatal server command/readiness error ignored for connection status: code={error_code} message={message}"
                                );
                            }
                        }
                        ServerMessage::ModelList { models, request_id } => {
                            emit_event(
                                emitter.as_ref(),
                                "stt://model-list",
                                &ModelListUpdate {
                                    models: models
                                        .into_iter()
                                        .map(|model| ModelInfoUpdate {
                                            name: model.name,
                                            display_name: model.display_name,
                                            size_description: model.size_description,
                                            status: model.status,
                                        })
                                        .collect(),
                                    request_id,
                                },
                            );
                        }
                        ServerMessage::ModelStatus { status, request_id } => {
                            emit_event(
                                emitter.as_ref(),
                                "stt://model-status",
                                &ModelStatusUpdate { status, request_id },
                            );
                        }
                        ServerMessage::DownloadProgress {
                            model_name,
                            status,
                            progress,
                            downloaded_bytes,
                            total_bytes,
                            error_message,
                        } => {
                            emit_event(
                                emitter.as_ref(),
                                "stt://download-progress",
                                &DownloadProgressUpdate {
                                    model_name,
                                    status,
                                    progress,
                                    downloaded_bytes,
                                    total_bytes,
                                    error_message,
                                },
                            );
                        }
                    }
                })
                .await;
        }

        let mut transport = self.transport.lock().await;
        if let Err(e) = transport.connect(CONNECT_TIMEOUT).await {
            tracing::error!("Connection failed (raw): {e}");
            let error_message = friendly_connect_error(&e);
            emit_connection_status(
                &self.connection_status,
                &self.emitter,
                false,
                Some(error_message),
            );
            return Err(e);
        }

        Ok(())
    }

    /// Starts capturing audio from the appropriate source.
    ///
    /// Algorithm:
    /// 1. Acquire the transport sender and session_id reference.
    /// 2. Build a callback that skips frames when state is not `Running`.
    /// 3. Use an injected source (from `set_audio_source`) if present; otherwise
    ///    select `FileAudioSource` (if `input_file` is set) or `CpalAudioSource`.
    /// 4. Start the source and store it.
    ///
    /// Must be called after `connect()` so the audio channel exists.
    pub fn start_audio(&mut self) -> Result<(), AppError> {
        let transport_guard = self.transport
            .try_lock()
            .map_err(|_| AppError::AudioError("transport locked during start_audio".into()))?;
        let audio_tx = transport_guard
            .audio_sender()
            .ok_or_else(|| AppError::AudioError("not connected — call connect() first".into()))?;
        let session_id = Arc::clone(&transport_guard.session_id);
        drop(transport_guard);

        let chunk_counter = Arc::clone(&self.chunk_counter);
        let state_manager = Arc::clone(&self.state_manager);

        let callback: Box<dyn Fn(Vec<f32>) + Send + 'static> = Box::new(move |samples| {
            let sid = session_id
                .lock()
                .ok()
                .and_then(|g| g.clone())
                .unwrap_or_default();
            if sid.is_empty() {
                return;
            }

            if state_manager.current_state() != AppState::Running {
                return;
            }

            let chunk_id = chunk_counter.fetch_add(1, Ordering::Relaxed);
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();

            let frame = AudioFrameEncoder::encode(&sid, chunk_id, timestamp, &samples);
            let _ = audio_tx.try_send(frame);
        });

        let mut source: Box<dyn AudioSource> = if let Some(s) = self.audio_source.take() {
            s
        } else if let Some(ref path) = self.input_file {
            Box::new(FileAudioSource::new(path.into(), true))
        } else {
            Box::new(CpalAudioSource::new())
        };

        source.start(callback)?;
        self.audio_source = Some(source);
        Ok(())
    }

    /// Stops audio capture and transitions to the Paused state.
    ///
    /// Releases the microphone by stopping the audio source, then marks
    /// the application as Paused so it can be resumed later.
    pub fn pause_audio(&mut self) -> Result<(), AppError> {
        if let Some(ref mut source) = self.audio_source {
            source.stop()?;
        }
        self.audio_source = None;
        self.state_manager.set_state(AppState::Paused)
    }

    /// Transitions to Running and restarts audio capture.
    ///
    /// Must be called after `pause_audio()`. Requires an active transport
    /// (i.e. `connect()` was called previously).
    pub fn resume_audio(&mut self) -> Result<(), AppError> {
        self.state_manager.set_state(AppState::Running)?;
        self.start_audio()
    }

    /// Clears accumulated text from the formatter.
    pub fn clear(&self) {
        self.formatter.clear();
    }

    /// Requests the current server model list.
    pub fn request_model_list(&self) -> Result<(), AppError> {
        let transport = self
            .transport
            .try_lock()
            .map_err(|_| AppError::ConnectionFailed("transport busy".into()))?;
        let session_id = transport
            .session_id
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
            .ok_or_else(|| AppError::ConnectionFailed("session not established".into()))?;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        transport.send_text(WsClientTransport::build_list_models_json(
            &session_id,
            timestamp,
            None,
        ))
    }

    /// Requests download of a specific model by name.
    pub fn request_model_download(&self, model_name: &str) -> Result<(), AppError> {
        let transport = self
            .transport
            .try_lock()
            .map_err(|_| AppError::ConnectionFailed("transport busy".into()))?;
        let session_id = transport
            .session_id
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
            .ok_or_else(|| AppError::ConnectionFailed("session not established".into()))?;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        transport.send_text(WsClientTransport::build_download_model_json(
            &session_id,
            timestamp,
            model_name,
            None,
        ))
    }

    /// Stops the orchestrator gracefully.
    ///
    /// Stops audio capture, transport, and transitions to Shutdown.
    pub async fn stop(&mut self, server_initiated: bool) {
        if let Some(ref mut source) = self.audio_source {
            let _ = source.stop();
        }
        self.audio_source = None;

        self.transport.lock().await.stop(server_initiated).await;

        let _ = self.state_manager.set_state(AppState::Shutdown);
    }

    /// Processes a decoded `ServerMessage` synchronously.
    ///
    /// Used by tests to verify message handling logic without a live WebSocket.
    /// The async message handler installed by `connect()` performs the same
    /// dispatch internally.
    pub fn handle_server_message(&self, msg: ServerMessage) -> Result<(), AppError> {
        match msg {
            ServerMessage::SessionCreated {
                session_id: _,
                protocol_version,
                ..
            } => {
                if protocol_version != EXPECTED_PROTOCOL_VERSION {
                    let message = format!("Unsupported protocol version: {protocol_version}");
                    emit_connection_status(
                        &self.connection_status,
                        &self.emitter,
                        false,
                        Some(message.clone()),
                    );
                    set_state_if_needed(&self.state_manager, AppState::Shutdown)?;
                    return Err(AppError::ProtocolError(format!(
                        "unsupported protocol version: {protocol_version}"
                    )));
                }
                set_state_if_needed(&self.state_manager, AppState::WaitingForServer)?;
                Ok(())
            }
            ServerMessage::RecognitionResult {
                session_id,
                status,
                text,
                start_time,
                end_time,
                chunk_ids,
                utterance_id,
                token_confidences,
            } => {
                let rr_status = match status.as_str() {
                    "partial" => RecognitionStatus::Partial,
                    "final" => RecognitionStatus::Final,
                    other => {
                        return Err(AppError::ProtocolError(format!(
                            "unknown recognition status: {other}"
                        )))
                    }
                };
                let rr = RecognitionResult {
                    session_id,
                    status: rr_status.clone(),
                    text,
                    start_time,
                    end_time,
                    chunk_ids,
                    utterance_id,
                    token_confidences,
                };
                self.dispatch_result(&rr);
                Ok(())
            }
            ServerMessage::Ping {} => Ok(()),
            ServerMessage::ServerState { state } => {
                emit_event(
                    self.emitter.as_ref(),
                    "stt://server-state",
                    &ServerStateUpdate {
                        state: state.clone(),
                    },
                );
                let target_state = target_state_for_server_state(&self.state_manager, &state);
                set_state_if_needed(&self.state_manager, target_state)?;
                Ok(())
            }
            ServerMessage::SessionClosed { reason, .. } => {
                tracing::info!("Session closed: {reason}");
                set_state_if_needed(&self.state_manager, AppState::Shutdown)?;
                Ok(())
            }
            ServerMessage::Error {
                error_code,
                message,
                fatal,
                ..
            } => {
                if should_surface_server_error_as_connection_error(&error_code, fatal) {
                    emit_connection_status(
                        &self.connection_status,
                        &self.emitter,
                        false,
                        Some(message.clone()),
                    );
                    Err(AppError::ProtocolError(format!("server error: {message}")))
                } else {
                    tracing::debug!(
                        "Nonfatal server command/readiness error ignored for connection status: code={error_code} message={message}"
                    );
                    Ok(())
                }
            }
            ServerMessage::ModelList { models, request_id } => {
                emit_event(
                    self.emitter.as_ref(),
                    "stt://model-list",
                    &ModelListUpdate {
                        models: models
                            .into_iter()
                            .map(|model| ModelInfoUpdate {
                                name: model.name,
                                display_name: model.display_name,
                                size_description: model.size_description,
                                status: model.status,
                            })
                            .collect(),
                        request_id,
                    },
                );
                Ok(())
            }
            ServerMessage::ModelStatus { status, request_id } => {
                emit_event(
                    self.emitter.as_ref(),
                    "stt://model-status",
                    &ModelStatusUpdate { status, request_id },
                );
                Ok(())
            }
            ServerMessage::DownloadProgress {
                model_name,
                status,
                progress,
                downloaded_bytes,
                total_bytes,
                error_message,
            } => {
                emit_event(
                    self.emitter.as_ref(),
                    "stt://download-progress",
                    &DownloadProgressUpdate {
                        model_name,
                        status,
                        progress,
                        downloaded_bytes,
                        total_bytes,
                        error_message,
                    },
                );
                Ok(())
            }
        }
    }

    /// Adds a `RecognitionSubscriber` to the fan-out.
    ///
    /// Args:
    ///   subscriber: Subscriber to receive partial and finalized recognition results.
    pub fn add_recognition_subscriber(&self, subscriber: Box<dyn RecognitionSubscriber>) {
        if let Ok(mut fo) = self.fan_out.lock() {
            fo.add_subscriber(subscriber);
        }
    }

    /// Dispatches a recognition result through the fan-out to all subscribers.
    pub fn dispatch_result(&self, result: &RecognitionResult) {
        if let Ok(fo) = self.fan_out.lock() {
            match result.status {
                RecognitionStatus::Partial => fo.on_partial(result),
                RecognitionStatus::Final => fo.on_finalization(result),
            }
        }
    }
}

/// Bridge that forwards `RecognitionSubscriber` calls to an `Arc<TextFormatter>`.
///
/// Needed because `TextFormatter` is behind an `Arc` and `RecognitionFanOut`
/// takes `Box<dyn RecognitionSubscriber>`.
struct FormatterBridge(Arc<TextFormatter>);

impl RecognitionSubscriber for FormatterBridge {
    fn on_partial(&self, result: &RecognitionResult) {
        self.0.on_partial(result);
    }

    fn on_finalization(&self, result: &RecognitionResult) {
        self.0.on_finalization(result);
    }
}
