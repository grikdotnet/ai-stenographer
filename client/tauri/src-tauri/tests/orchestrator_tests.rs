/// Tests for the `ClientOrchestrator` coordinator.
///
/// Since `tauri::AppHandle` cannot be created in unit tests, these tests
/// exercise the non-Tauri surface: construction, state wiring, formatter
/// access, audio source selection logic, connection failure paths, and
/// message-handler dispatch.
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use stt_tauri_client::audio::AudioSource;
use stt_tauri_client::error::AppError;
use stt_tauri_client::orchestrator::{
    friendly_connect_error, ClientOrchestrator, ConnectionStatus, EventEmitter, StateChanged,
    TranscriptUpdate,
};
use stt_tauri_client::protocol::ServerMessageDecoder;
use stt_tauri_client::recognition::{RecognitionResult, RecognitionStatus, RecognitionSubscriber};
use stt_tauri_client::state::{AppState, AppStateManager};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::tungstenite::Message;

struct SpyAudioSource {
    stopped: Arc<AtomicBool>,
    start_count: Arc<AtomicUsize>,
    stop_count: Arc<AtomicUsize>,
    callback: Arc<Mutex<Option<Box<dyn Fn(Vec<f32>) + Send + 'static>>>>,
}

impl SpyAudioSource {
    fn new(stopped: Arc<AtomicBool>) -> Self {
        Self::with_counters(
            stopped,
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(Mutex::new(None)),
        )
    }

    fn with_counters(
        stopped: Arc<AtomicBool>,
        start_count: Arc<AtomicUsize>,
        stop_count: Arc<AtomicUsize>,
        callback: Arc<Mutex<Option<Box<dyn Fn(Vec<f32>) + Send + 'static>>>>,
    ) -> Self {
        Self {
            stopped,
            start_count,
            stop_count,
            callback,
        }
    }
}

impl AudioSource for SpyAudioSource {
    fn start(&mut self, callback: Box<dyn Fn(Vec<f32>) + Send + 'static>) -> Result<(), AppError> {
        self.start_count.fetch_add(1, Ordering::SeqCst);
        *self.callback.lock().unwrap() = Some(callback);
        Ok(())
    }

    fn stop(&mut self) -> Result<(), AppError> {
        self.stopped.store(true, Ordering::SeqCst);
        self.stop_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct RecordingEmitter {
    events: Arc<Mutex<Vec<(String, String)>>>,
}

impl RecordingEmitter {
    fn new() -> (Self, Arc<Mutex<Vec<(String, String)>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                events: Arc::clone(&events),
            },
            events,
        )
    }
}

impl EventEmitter for RecordingEmitter {
    fn emit_serialized(&self, event: &str, payload: &str) {
        self.events
            .lock()
            .unwrap()
            .push((event.to_string(), payload.to_string()));
    }
}

async fn wait_until(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(condition(), "condition was not met before timeout");
}

async fn spawn_scripted_ws_server(
    messages: Vec<&'static str>,
    delay_between_messages: Duration,
) -> (String, mpsc::Receiver<Vec<u8>>, oneshot::Sender<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (audio_tx, audio_rx) = mpsc::channel(8);
    let (shutdown_tx, mut shutdown_rx) = oneshot::channel();

    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let ws_stream = tokio_tungstenite::accept_async(stream).await.unwrap();
        let (mut sink, mut stream) = ws_stream.split();

        for message in messages {
            sink.send(Message::Text(message.into())).await.unwrap();
            tokio::time::sleep(delay_between_messages).await;
        }

        loop {
            tokio::select! {
                _ = &mut shutdown_rx => break,
                next = stream.next() => {
                    match next {
                        Some(Ok(Message::Binary(frame))) => {
                            let _ = audio_tx.send(frame.to_vec()).await;
                        }
                        Some(Ok(Message::Close(_))) | None => break,
                        Some(Err(_)) => break,
                        _ => {}
                    }
                }
            }
        }
    });

    (format!("ws://{addr}"), audio_rx, shutdown_tx)
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn new_creates_orchestrator_without_panic() {
    let sm = Arc::new(AppStateManager::new());
    let orch = ClientOrchestrator::new("ws://127.0.0.1:9999".into(), None, sm.clone());
    assert_eq!(sm.current_state(), AppState::Starting);
    drop(orch);
}

#[test]
fn new_stores_server_url_and_input_file() {
    let sm = Arc::new(AppStateManager::new());
    let orch = ClientOrchestrator::new("ws://host:1234".into(), Some("audio.wav".into()), sm);
    assert_eq!(orch.server_url(), "ws://host:1234");
    assert_eq!(orch.input_file().as_deref(), Some("audio.wav"));
}

#[test]
fn replace_server_url_updates_orchestrator_target() {
    let sm = Arc::new(AppStateManager::new());
    let mut orch = ClientOrchestrator::new("ws://old:1".into(), None, sm);

    orch.replace_server_url("ws://new:2".into());

    assert_eq!(orch.server_url(), "ws://new:2");
}

#[test]
fn emit_connection_error_updates_status_snapshot() {
    let sm = Arc::new(AppStateManager::new());
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm);

    orch.emit_connection_error("terminal server failure".into());

    let status = orch.current_connection_status();
    assert!(!status.connected);
    assert_eq!(status.error.as_deref(), Some("terminal server failure"));
}

// ---------------------------------------------------------------------------
// Formatter access
// ---------------------------------------------------------------------------

#[test]
fn formatter_is_accessible_and_functional() {
    let sm = Arc::new(AppStateManager::new());
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm);
    let fmt = orch.formatter();

    assert!(fmt.get_utterances().is_empty());
    assert_eq!(fmt.get_preliminary_text(), "");

    let result = RecognitionResult {
        session_id: "s1".into(),
        status: RecognitionStatus::Final,
        text: "hello world".into(),
        start_time: 0.0,
        end_time: 1.0,
        chunk_ids: vec![],
        utterance_id: 1,
        token_confidences: vec![],
    };
    fmt.on_finalization(&result);
    let utterances = fmt.get_utterances();
    assert_eq!(utterances.len(), 1);
    assert_eq!(utterances[0].text, "hello world");
}

#[test]
fn formatter_registered_in_fan_out() {
    let sm = Arc::new(AppStateManager::new());
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm);

    let result = RecognitionResult {
        session_id: "s1".into(),
        status: RecognitionStatus::Partial,
        text: "partial stuff".into(),
        start_time: 0.0,
        end_time: 0.5,
        chunk_ids: vec![],
        utterance_id: 1,
        token_confidences: vec![],
    };

    orch.dispatch_result(&result);
    assert_eq!(orch.formatter().get_preliminary_text(), "partial stuff");
}

// ---------------------------------------------------------------------------
// Clear
// ---------------------------------------------------------------------------

#[test]
fn clear_resets_formatter_state() {
    let sm = Arc::new(AppStateManager::new());
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm);
    let fmt = orch.formatter();

    let result = RecognitionResult {
        session_id: "s1".into(),
        status: RecognitionStatus::Final,
        text: "accumulated".into(),
        start_time: 0.0,
        end_time: 1.0,
        chunk_ids: vec![],
        utterance_id: 1,
        token_confidences: vec![],
    };
    fmt.on_finalization(&result);
    assert!(!fmt.get_utterances().is_empty());

    orch.clear();
    assert!(fmt.get_utterances().is_empty());
    assert_eq!(fmt.get_preliminary_text(), "");
}

// ---------------------------------------------------------------------------
// State transitions (pause_audio / resume_audio)
// ---------------------------------------------------------------------------

#[test]
fn pause_audio_transitions_from_running_to_paused() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    sm.set_state(AppState::Running).unwrap();
    let mut orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());

    assert!(orch.pause_audio().is_ok());
    assert_eq!(sm.current_state(), AppState::Paused);
}

#[test]
fn pause_audio_does_not_stop_unstarted_audio_source() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    sm.set_state(AppState::Running).unwrap();
    let mut orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());

    let stopped = Arc::new(AtomicBool::new(false));
    orch.set_audio_source(Box::new(SpyAudioSource::new(Arc::clone(&stopped))));

    orch.pause_audio().unwrap();

    assert!(
        !stopped.load(Ordering::SeqCst),
        "audio source should not be stopped before capture starts"
    );
}

#[test]
fn resume_audio_transitions_from_paused_to_running() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    sm.set_state(AppState::Running).unwrap();
    sm.set_state(AppState::Paused).unwrap();
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());
    let msg = ServerMessageDecoder::decode(r#"{"type":"server_state","state":"running"}"#).unwrap();
    orch.handle_server_message(msg).unwrap();
    let mut orch = orch;

    // start_audio() needs a connected transport — it will fail, but the state
    // transition to Running happens before the audio start attempt when the
    // latest server readiness state is running.
    let _ = orch.resume_audio();
    assert_eq!(sm.current_state(), AppState::Running);
}

#[test]
fn pause_audio_from_starting_fails() {
    let sm = Arc::new(AppStateManager::new());
    let mut orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm);
    assert!(orch.pause_audio().is_err());
}

// ---------------------------------------------------------------------------
// Connection failure
// ---------------------------------------------------------------------------

#[tokio::test]
async fn connect_returns_error_when_server_unreachable() {
    let sm = Arc::new(AppStateManager::new());
    let mut orch = ClientOrchestrator::new("ws://127.0.0.1:19999".into(), None, sm);
    let result = orch.connect().await;
    assert!(result.is_err());
    match result.unwrap_err() {
        AppError::ConnectionFailed(_) => {}
        other => panic!("expected ConnectionFailed, got {other:?}"),
    }
}

#[tokio::test]
async fn failed_connect_updates_connection_status_snapshot() {
    let sm = Arc::new(AppStateManager::new());
    let mut orch = ClientOrchestrator::new("ws://127.0.0.1:19999".into(), None, sm);

    let _ = orch.connect().await;
    let status = orch.current_connection_status();

    assert!(!status.connected);
    assert!(
        status.error.as_deref().is_some(),
        "failed connection should persist an error message"
    );
}

// ---------------------------------------------------------------------------
// Message handler logic (tested via build_message_handler)
// ---------------------------------------------------------------------------

#[test]
fn handle_session_created_sets_waiting_for_server_state() {
    let sm = Arc::new(AppStateManager::new());
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());

    let json = r#"{"type":"session_created","session_id":"abc-123","protocol_version":"v1","server_time":1.0}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let handler_result = orch.handle_server_message(msg);
    assert!(handler_result.is_ok());
    assert_eq!(sm.current_state(), AppState::WaitingForServer);
}

#[test]
fn handle_session_created_is_noop_when_already_waiting_for_server() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());

    let json = r#"{"type":"session_created","session_id":"abc-123","protocol_version":"v1","server_time":1.0}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let handler_result = orch.handle_server_message(msg);
    assert!(handler_result.is_ok());
    assert_eq!(sm.current_state(), AppState::WaitingForServer);
}

#[test]
fn handle_session_created_rejects_wrong_protocol_version() {
    let sm = Arc::new(AppStateManager::new());
    let mut orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());
    let (emitter, events) = RecordingEmitter::new();
    orch.set_emitter(Arc::new(emitter));

    let json = r#"{"type":"session_created","session_id":"abc","protocol_version":"v99","server_time":1.0}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let result = orch.handle_server_message(msg);
    assert!(result.is_err());
    assert_eq!(sm.current_state(), AppState::Shutdown);
    let events = events.lock().unwrap();
    assert!(events.iter().any(|(event, payload)| {
        event == "stt://connection-status"
            && payload.contains("\"connected\":false")
            && payload.contains("Unsupported protocol version")
    }));
}

#[test]
fn handle_recognition_result_dispatches_partial_through_fan_out() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    sm.set_state(AppState::Running).unwrap();
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm);

    let json = r#"{"type":"recognition_result","session_id":"s1","status":"partial","text":"hello","start_time":0.0,"end_time":0.5,"chunk_ids":[],"utterance_id":1,"token_confidences":[]}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let result = orch.handle_server_message(msg);
    assert!(result.is_ok());
    assert_eq!(orch.formatter().get_preliminary_text(), "hello");
}

#[test]
fn handle_recognition_result_dispatches_final_through_fan_out() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    sm.set_state(AppState::Running).unwrap();
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm);

    let json = r#"{"type":"recognition_result","session_id":"s1","status":"final","text":"done","start_time":0.0,"end_time":1.0,"chunk_ids":[],"utterance_id":1,"token_confidences":[]}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let result = orch.handle_server_message(msg);
    assert!(result.is_ok());
    let utterances = orch.formatter().get_utterances();
    assert_eq!(utterances.len(), 1);
    assert_eq!(utterances[0].text, "done");
}

#[test]
fn handle_session_closed_transitions_to_shutdown() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    sm.set_state(AppState::Running).unwrap();
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());

    let json =
        r#"{"type":"session_closed","session_id":"s1","reason":"close_session","message":null}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let result = orch.handle_server_message(msg);
    assert!(result.is_ok());
    assert_eq!(sm.current_state(), AppState::Shutdown);
}

#[test]
fn handle_server_state_running_transitions_to_running() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());

    let json = r#"{"type":"server_state","state":"running"}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let result = orch.handle_server_message(msg);
    assert!(result.is_ok());
    assert_eq!(sm.current_state(), AppState::Running);
}

#[test]
fn handle_server_state_running_preserves_paused_state_without_error() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    sm.set_state(AppState::Running).unwrap();
    sm.set_state(AppState::Paused).unwrap();
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());

    let json = r#"{"type":"server_state","state":"running"}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let result = orch.handle_server_message(msg);
    assert!(result.is_ok());
    assert_eq!(sm.current_state(), AppState::Paused);
}

#[test]
fn handle_server_state_waiting_for_model_transitions_away_from_running() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    sm.set_state(AppState::Running).unwrap();
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());

    let json = r#"{"type":"server_state","state":"waiting_for_model"}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let result = orch.handle_server_message(msg);
    assert!(result.is_ok());
    assert_eq!(sm.current_state(), AppState::WaitingForServer);
}

#[test]
fn handle_server_state_waiting_for_model_is_noop_when_already_waiting() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());

    let json = r#"{"type":"server_state","state":"waiting_for_model"}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let result = orch.handle_server_message(msg);
    assert!(result.is_ok());
    assert_eq!(sm.current_state(), AppState::WaitingForServer);
}

#[test]
fn current_connection_status_includes_latest_server_state() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm);

    let json = r#"{"type":"server_state","state":"waiting_for_model"}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let result = orch.handle_server_message(msg);

    assert!(result.is_ok());
    assert_eq!(
        orch.current_connection_status().server_state.as_deref(),
        Some("waiting_for_model")
    );
}

// ---------------------------------------------------------------------------
// Readiness-gated audio lifecycle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn connect_waiting_for_model_does_not_start_audio_source() {
    let messages = vec![
        r#"{"type":"session_created","session_id":"s1","protocol_version":"v1","server_time":1.0}"#,
        r#"{"type":"server_state","state":"waiting_for_model"}"#,
    ];
    let (url, _audio_rx, shutdown_tx) =
        spawn_scripted_ws_server(messages, Duration::from_millis(20)).await;
    let sm = Arc::new(AppStateManager::new());
    let mut orch = ClientOrchestrator::new(url, None, Arc::clone(&sm));
    let stopped = Arc::new(AtomicBool::new(false));
    let start_count = Arc::new(AtomicUsize::new(0));
    let stop_count = Arc::new(AtomicUsize::new(0));
    orch.set_audio_source(Box::new(SpyAudioSource::with_counters(
        stopped,
        Arc::clone(&start_count),
        Arc::clone(&stop_count),
        Arc::new(Mutex::new(None)),
    )));

    orch.connect().await.unwrap();
    orch.start_audio().unwrap();
    wait_until(|| sm.current_state() == AppState::WaitingForServer).await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    assert_eq!(start_count.load(Ordering::SeqCst), 0);
    assert_eq!(stop_count.load(Ordering::SeqCst), 0);
    let _ = shutdown_tx.send(());
    orch.stop(true).await;
}

#[tokio::test]
async fn receive_loop_starts_audio_once_when_server_later_runs() {
    let messages = vec![
        r#"{"type":"session_created","session_id":"s1","protocol_version":"v1","server_time":1.0}"#,
        r#"{"type":"server_state","state":"waiting_for_model"}"#,
        r#"{"type":"server_state","state":"running"}"#,
        r#"{"type":"server_state","state":"running"}"#,
    ];
    let (url, mut audio_rx, shutdown_tx) =
        spawn_scripted_ws_server(messages, Duration::from_millis(20)).await;
    let sm = Arc::new(AppStateManager::new());
    let mut orch = ClientOrchestrator::new(url, None, Arc::clone(&sm));
    let stopped = Arc::new(AtomicBool::new(false));
    let start_count = Arc::new(AtomicUsize::new(0));
    let stop_count = Arc::new(AtomicUsize::new(0));
    let callback = Arc::new(Mutex::new(None));
    orch.set_audio_source(Box::new(SpyAudioSource::with_counters(
        stopped,
        Arc::clone(&start_count),
        Arc::clone(&stop_count),
        Arc::clone(&callback),
    )));

    orch.connect().await.unwrap();
    orch.start_audio().unwrap();
    wait_until(|| sm.current_state() == AppState::Running).await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    assert_eq!(start_count.load(Ordering::SeqCst), 1);
    assert_eq!(stop_count.load(Ordering::SeqCst), 0);
    if let Some(cb) = callback.lock().unwrap().as_ref() {
        cb(vec![0.25; 512]);
    }
    let frame = tokio::time::timeout(Duration::from_secs(1), audio_rx.recv())
        .await
        .unwrap()
        .expect("audio frame should be delivered after readiness");
    assert!(!frame.is_empty());
    let _ = shutdown_tx.send(());
    orch.stop(true).await;
}

#[tokio::test]
async fn server_state_transition_away_from_running_stops_active_capture() {
    let messages = vec![
        r#"{"type":"session_created","session_id":"s1","protocol_version":"v1","server_time":1.0}"#,
        r#"{"type":"server_state","state":"running"}"#,
        r#"{"type":"server_state","state":"waiting_for_model"}"#,
    ];
    let (url, _audio_rx, shutdown_tx) =
        spawn_scripted_ws_server(messages, Duration::from_millis(20)).await;
    let sm = Arc::new(AppStateManager::new());
    let mut orch = ClientOrchestrator::new(url, None, Arc::clone(&sm));
    let stopped = Arc::new(AtomicBool::new(false));
    let start_count = Arc::new(AtomicUsize::new(0));
    let stop_count = Arc::new(AtomicUsize::new(0));
    let callback = Arc::new(Mutex::new(None));
    orch.set_audio_source(Box::new(SpyAudioSource::with_counters(
        Arc::clone(&stopped),
        Arc::clone(&start_count),
        Arc::clone(&stop_count),
        Arc::clone(&callback),
    )));

    orch.connect().await.unwrap();
    orch.start_audio().unwrap();
    wait_until(|| stop_count.load(Ordering::SeqCst) == 1).await;

    assert_eq!(start_count.load(Ordering::SeqCst), 1);
    assert!(stopped.load(Ordering::SeqCst));
    let before = orch.next_chunk_id();
    if let Some(cb) = callback.lock().unwrap().as_ref() {
        cb(vec![0.0; 512]);
    }
    assert_eq!(
        orch.next_chunk_id(),
        before + 1,
        "callback must not send or allocate a chunk while client is not Running"
    );
    let _ = shutdown_tx.send(());
    orch.stop(true).await;
}

#[tokio::test]
async fn paused_client_does_not_auto_start_on_duplicate_running_state() {
    let messages = vec![
        r#"{"type":"session_created","session_id":"s1","protocol_version":"v1","server_time":1.0}"#,
        r#"{"type":"server_state","state":"running"}"#,
    ];
    let (url, _audio_rx, shutdown_tx) =
        spawn_scripted_ws_server(messages, Duration::from_millis(20)).await;
    let sm = Arc::new(AppStateManager::new());
    let mut orch = ClientOrchestrator::new(url, None, Arc::clone(&sm));
    let stopped = Arc::new(AtomicBool::new(false));
    let start_count = Arc::new(AtomicUsize::new(0));
    let stop_count = Arc::new(AtomicUsize::new(0));
    orch.set_audio_source(Box::new(SpyAudioSource::with_counters(
        stopped,
        Arc::clone(&start_count),
        Arc::clone(&stop_count),
        Arc::new(Mutex::new(None)),
    )));

    orch.connect().await.unwrap();
    orch.start_audio().unwrap();
    wait_until(|| start_count.load(Ordering::SeqCst) == 1).await;
    wait_until(|| sm.current_state() == AppState::Running).await;
    orch.pause_audio().unwrap();
    assert_eq!(sm.current_state(), AppState::Paused);

    let msg = ServerMessageDecoder::decode(r#"{"type":"server_state","state":"running"}"#).unwrap();
    orch.handle_server_message(msg).unwrap();

    assert_eq!(sm.current_state(), AppState::Paused);
    assert_eq!(start_count.load(Ordering::SeqCst), 1);
    assert_eq!(stop_count.load(Ordering::SeqCst), 1);
    let _ = shutdown_tx.send(());
    orch.stop(true).await;
}

#[tokio::test]
async fn resume_audio_does_not_bypass_server_readiness() {
    let messages = vec![
        r#"{"type":"session_created","session_id":"s1","protocol_version":"v1","server_time":1.0}"#,
        r#"{"type":"server_state","state":"running"}"#,
        r#"{"type":"server_state","state":"waiting_for_model"}"#,
    ];
    let (url, _audio_rx, shutdown_tx) =
        spawn_scripted_ws_server(messages, Duration::from_millis(20)).await;
    let sm = Arc::new(AppStateManager::new());
    let mut orch = ClientOrchestrator::new(url, None, Arc::clone(&sm));
    let stopped = Arc::new(AtomicBool::new(false));
    let start_count = Arc::new(AtomicUsize::new(0));
    let stop_count = Arc::new(AtomicUsize::new(0));
    orch.set_audio_source(Box::new(SpyAudioSource::with_counters(
        stopped,
        Arc::clone(&start_count),
        Arc::clone(&stop_count),
        Arc::new(Mutex::new(None)),
    )));

    orch.connect().await.unwrap();
    orch.start_audio().unwrap();
    wait_until(|| stop_count.load(Ordering::SeqCst) == 1).await;
    assert_eq!(sm.current_state(), AppState::WaitingForServer);

    orch.resume_audio().unwrap();

    assert_eq!(sm.current_state(), AppState::WaitingForServer);
    assert_eq!(start_count.load(Ordering::SeqCst), 1);
    assert_eq!(stop_count.load(Ordering::SeqCst), 1);
    let _ = shutdown_tx.send(());
    orch.stop(true).await;
}

#[tokio::test]
async fn failed_pause_while_waiting_does_not_disarm_readiness_capture() {
    let messages = vec![
        r#"{"type":"session_created","session_id":"s1","protocol_version":"v1","server_time":1.0}"#,
        r#"{"type":"server_state","state":"waiting_for_model"}"#,
    ];
    let (url, _audio_rx, shutdown_tx) =
        spawn_scripted_ws_server(messages, Duration::from_millis(20)).await;
    let sm = Arc::new(AppStateManager::new());
    let mut orch = ClientOrchestrator::new(url, None, Arc::clone(&sm));
    let stopped = Arc::new(AtomicBool::new(false));
    let start_count = Arc::new(AtomicUsize::new(0));
    let stop_count = Arc::new(AtomicUsize::new(0));
    orch.set_audio_source(Box::new(SpyAudioSource::with_counters(
        stopped,
        Arc::clone(&start_count),
        Arc::clone(&stop_count),
        Arc::new(Mutex::new(None)),
    )));

    orch.connect().await.unwrap();
    orch.start_audio().unwrap();
    wait_until(|| sm.current_state() == AppState::WaitingForServer).await;

    assert!(orch.pause_audio().is_err());
    let msg = ServerMessageDecoder::decode(r#"{"type":"server_state","state":"running"}"#).unwrap();
    orch.handle_server_message(msg).unwrap();

    assert_eq!(sm.current_state(), AppState::Running);
    assert_eq!(start_count.load(Ordering::SeqCst), 1);
    assert_eq!(stop_count.load(Ordering::SeqCst), 0);
    let _ = shutdown_tx.send(());
    orch.stop(true).await;
}

#[tokio::test]
async fn session_closed_stops_active_capture() {
    let messages = vec![
        r#"{"type":"session_created","session_id":"s1","protocol_version":"v1","server_time":1.0}"#,
        r#"{"type":"server_state","state":"running"}"#,
        r#"{"type":"session_closed","session_id":"s1","reason":"server_shutdown","message":null}"#,
    ];
    let (url, _audio_rx, shutdown_tx) =
        spawn_scripted_ws_server(messages, Duration::from_millis(20)).await;
    let sm = Arc::new(AppStateManager::new());
    let mut orch = ClientOrchestrator::new(url, None, Arc::clone(&sm));
    let stopped = Arc::new(AtomicBool::new(false));
    let start_count = Arc::new(AtomicUsize::new(0));
    let stop_count = Arc::new(AtomicUsize::new(0));
    orch.set_audio_source(Box::new(SpyAudioSource::with_counters(
        Arc::clone(&stopped),
        Arc::clone(&start_count),
        Arc::clone(&stop_count),
        Arc::new(Mutex::new(None)),
    )));

    orch.connect().await.unwrap();
    orch.start_audio().unwrap();
    wait_until(|| sm.current_state() == AppState::Shutdown).await;

    assert_eq!(start_count.load(Ordering::SeqCst), 1);
    assert_eq!(stop_count.load(Ordering::SeqCst), 1);
    assert!(stopped.load(Ordering::SeqCst));
    let _ = shutdown_tx.send(());
    orch.stop(true).await;
}

#[test]
fn handle_server_state_shutdown_transitions_to_shutdown() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());

    let json = r#"{"type":"server_state","state":"shutdown"}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let result = orch.handle_server_message(msg);
    assert!(result.is_ok());
    assert_eq!(sm.current_state(), AppState::Shutdown);
}

#[test]
fn handle_model_status_emits_model_status_event() {
    let sm = Arc::new(AppStateManager::new());
    let mut orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm);
    let (emitter, events) = RecordingEmitter::new();
    orch.set_emitter(Arc::new(emitter));

    let json = r#"{"type":"model_status","status":"downloading","request_id":"req-2"}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let result = orch.handle_server_message(msg);
    assert!(result.is_ok());
    let events = events.lock().unwrap();
    assert!(events.iter().any(|(event, payload)| {
        event == "stt://model-status"
            && payload.contains("\"status\":\"downloading\"")
            && payload.contains("\"request_id\":\"req-2\"")
    }));
}

#[test]
fn handle_error_message_returns_protocol_error() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    sm.set_state(AppState::Running).unwrap();
    let mut orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm);
    let (emitter, events) = RecordingEmitter::new();
    orch.set_emitter(Arc::new(emitter));

    let json = r#"{"type":"error","session_id":"s1","error_code":"BAD","message":"something broke","fatal":true}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let result = orch.handle_server_message(msg);
    match result {
        Err(AppError::ProtocolError(msg)) => assert!(msg.contains("something broke")),
        other => panic!("expected ProtocolError, got {other:?}"),
    }
    let events = events.lock().unwrap();
    assert!(events.iter().any(|(event, payload)| {
        event == "stt://connection-status"
            && payload.contains("\"connected\":false")
            && payload.contains("something broke")
    }));
}

#[test]
fn handle_nonfatal_model_not_ready_does_not_emit_connection_error() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    let mut orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());
    let (emitter, events) = RecordingEmitter::new();
    orch.set_emitter(Arc::new(emitter));

    let json = r#"{"type":"error","session_id":"s1","error_code":"MODEL_NOT_READY","message":"model is not ready","fatal":false}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let result = orch.handle_server_message(msg);

    assert!(result.is_ok());
    assert_eq!(sm.current_state(), AppState::WaitingForServer);
    let status = orch.current_connection_status();
    assert!(status.error.is_none());
    let events = events.lock().unwrap();
    assert!(!events
        .iter()
        .any(|(event, payload)| event == "stt://connection-status"
            && payload.contains("\"connected\":false")
            && payload.contains("model is not ready")));
}

#[test]
fn handle_nonfatal_download_in_progress_does_not_mark_connection_failed() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    sm.set_state(AppState::Running).unwrap();
    let mut orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());
    let (emitter, events) = RecordingEmitter::new();
    orch.set_emitter(Arc::new(emitter));

    let json = r#"{"type":"error","session_id":"s1","error_code":"DOWNLOAD_IN_PROGRESS","message":"download already running","fatal":false}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    let result = orch.handle_server_message(msg);

    assert!(result.is_ok());
    assert_eq!(sm.current_state(), AppState::Running);
    assert!(orch.current_connection_status().error.is_none());
    let events = events.lock().unwrap();
    assert!(!events
        .iter()
        .any(|(event, payload)| event == "stt://connection-status"
            && payload.contains("\"connected\":false")
            && payload.contains("download already running")));
}

#[test]
fn handle_ping_is_noop() {
    let sm = Arc::new(AppStateManager::new());
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm);

    let json = r#"{"type":"ping"}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    assert!(orch.handle_server_message(msg).is_ok());
}

// ---------------------------------------------------------------------------
// Chunk counter increments
// ---------------------------------------------------------------------------

#[test]
fn chunk_counter_starts_at_zero() {
    let sm = Arc::new(AppStateManager::new());
    let orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm);
    assert_eq!(orch.next_chunk_id(), 0);
    assert_eq!(orch.next_chunk_id(), 1);
    assert_eq!(orch.next_chunk_id(), 2);
}

// ---------------------------------------------------------------------------
// Event payload serde
// ---------------------------------------------------------------------------

#[test]
fn connection_status_serializes_correctly() {
    let status = ConnectionStatus {
        connected: true,
        error: None,
        server_state: Some("running".into()),
    };
    let json = serde_json::to_value(&status).unwrap();
    assert_eq!(json["connected"], true);
    assert!(json["error"].is_null());
    assert_eq!(json["server_state"], "running");

    let status2 = ConnectionStatus {
        connected: false,
        error: Some("timeout".into()),
        server_state: None,
    };
    let json2 = serde_json::to_value(&status2).unwrap();
    assert_eq!(json2["connected"], false);
    assert_eq!(json2["error"], "timeout");
    assert!(json2["server_state"].is_null());
}

#[test]
fn transcript_update_serializes_correctly() {
    let update = TranscriptUpdate {
        action: "update".into(),
        utterances: vec![],
        preliminary: "world".into(),
    };
    let json = serde_json::to_value(&update).unwrap();
    assert_eq!(json["action"], "update");
    assert!(json["utterances"].is_array());
    assert_eq!(json["preliminary"], "world");
}

#[test]
fn state_changed_serializes_correctly() {
    let sc = StateChanged {
        old: "Starting".into(),
        new: "Running".into(),
    };
    let json = serde_json::to_value(&sc).unwrap();
    assert_eq!(json["old"], "Starting");
    assert_eq!(json["new"], "Running");
}

// ---------------------------------------------------------------------------
// Stop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stop_transitions_to_shutdown_from_running() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    sm.set_state(AppState::Running).unwrap();
    let mut orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());

    orch.stop(false).await;
    assert_eq!(sm.current_state(), AppState::Shutdown);
}

#[tokio::test]
async fn stop_is_idempotent() {
    let sm = Arc::new(AppStateManager::new());
    sm.set_state(AppState::WaitingForServer).unwrap();
    sm.set_state(AppState::Running).unwrap();
    let mut orch = ClientOrchestrator::new("ws://localhost:0".into(), None, sm.clone());

    orch.stop(false).await;
    orch.stop(false).await;
    assert_eq!(sm.current_state(), AppState::Shutdown);
}

// ---------------------------------------------------------------------------
// Friendly connection error messages
// ---------------------------------------------------------------------------

#[test]
fn friendly_error_connection_refused_gives_server_not_running_hint() {
    let err = AppError::ConnectionFailed(
        "IO error: No connection could be made because the target machine actively refused it. (os error 10061)".into()
    );
    let msg = friendly_connect_error(&err);
    assert!(
        msg.contains("server") || msg.contains("running"),
        "expected server-not-running hint, got: {msg}"
    );
}

#[test]
fn friendly_error_connection_refused_linux_gives_server_not_running_hint() {
    let err = AppError::ConnectionFailed("IO error: Connection refused (os error 111)".into());
    let msg = friendly_connect_error(&err);
    assert!(
        msg.contains("server") || msg.contains("running"),
        "expected server-not-running hint, got: {msg}"
    );
}

#[test]
fn friendly_error_timeout_gives_address_hint() {
    let err = AppError::ConnectionFailed("connection timed out".into());
    let msg = friendly_connect_error(&err);
    assert!(
        msg.contains("timed out") || msg.contains("address"),
        "expected timeout hint, got: {msg}"
    );
}

#[test]
fn friendly_error_empty_url_gives_no_server_url_hint() {
    let err = AppError::ConnectionFailed("HTTP format error: empty string".into());
    let msg = friendly_connect_error(&err);
    assert!(
        msg.contains("URL") || msg.contains("url") || msg.contains("address"),
        "expected URL hint, got: {msg}"
    );
}

#[test]
fn friendly_error_non_connection_variant_returns_generic_message() {
    let err = AppError::AudioError("microphone not found".into());
    let msg = friendly_connect_error(&err);
    assert!(!msg.is_empty(), "expected non-empty fallback message");
}

#[test]
fn friendly_error_message_is_short_and_user_readable() {
    let err = AppError::ConnectionFailed(
        "IO error: No connection could be made because the target machine actively refused it. (os error 10061)".into()
    );
    let msg = friendly_connect_error(&err);
    assert!(msg.len() < 120, "message too long for UI: {msg}");
    assert!(
        !msg.contains("os error"),
        "raw OS error leaked into UI: {msg}"
    );
    assert!(
        !msg.contains("IO error"),
        "raw IO error leaked into UI: {msg}"
    );
}
