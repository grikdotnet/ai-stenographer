/// Async WebSocket transport for communicating with the STT server.
///
/// Manages connection lifecycle, audio frame sending with backpressure
/// (bounded channel, drop-on-full), and dispatching received messages
/// to a configurable callback.
use crate::error::AppError;
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::{mpsc, Mutex, Notify};
use tokio::time::Duration;
use tokio_tungstenite::tungstenite::Message;

type MessageHandler = Box<dyn Fn(String) + Send + Sync>;

/// Async WebSocket client transport with bounded backpressure.
///
/// Spawns three internal tasks on connect: a binary drain loop (audio frames),
/// a text drain loop (control commands), and a receive loop (incoming messages).
pub struct WsClientTransport {
    url: String,
    pub session_id: Arc<StdMutex<Option<String>>>,
    audio_tx: Option<mpsc::Sender<Vec<u8>>>,
    text_tx: Option<mpsc::Sender<String>>,
    message_handler: Arc<Mutex<Option<MessageHandler>>>,
    stop_notify: Arc<Notify>,
    connected: Arc<Mutex<bool>>,
}

impl WsClientTransport {
    /// Creates a new transport targeting the given WebSocket URL.
    pub fn new(url: String) -> Self {
        Self {
            url,
            session_id: Arc::new(StdMutex::new(None)),
            audio_tx: None,
            text_tx: None,
            message_handler: Arc::new(Mutex::new(None)),
            stop_notify: Arc::new(Notify::new()),
            connected: Arc::new(Mutex::new(false)),
        }
    }

    /// Returns the configured WebSocket URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Sets the session ID used for outgoing messages.
    pub fn set_session_id(&self, id: String) {
        if let Ok(mut guard) = self.session_id.lock() {
            *guard = Some(id);
        }
    }

    /// Registers a callback invoked for every text message from the server.
    pub async fn set_message_handler(&self, handler: impl Fn(String) + Send + Sync + 'static) {
        *self.message_handler.lock().await = Some(Box::new(handler));
    }

    /// Connects to the server with a timeout.
    ///
    /// Spawns internal drain and receive loops.
    ///
    /// Returns: `Ok(())` on success or `AppError::ConnectionFailed`.
    pub async fn connect(&mut self, timeout: Duration) -> Result<(), AppError> {
        let ws_stream = tokio::time::timeout(timeout, tokio_tungstenite::connect_async(&self.url))
            .await
            .map_err(|_| AppError::ConnectionFailed("connection timed out".into()))?
            .map_err(|e| AppError::ConnectionFailed(e.to_string()))?
            .0;

        let (ws_sink, ws_stream) = ws_stream.split();
        let ws_sink = Arc::new(Mutex::new(ws_sink));

        let (tx, rx) = mpsc::channel::<Vec<u8>>(20);
        self.audio_tx = Some(tx);

        let (text_tx, text_rx) = mpsc::channel::<String>(8);
        self.text_tx = Some(text_tx);

        *self.connected.lock().await = true;

        Self::spawn_drain_loop(Arc::clone(&ws_sink), rx);
        Self::spawn_text_drain_loop(Arc::clone(&ws_sink), text_rx);
        Self::spawn_receive_loop(
            ws_stream,
            Arc::clone(&self.message_handler),
            Arc::clone(&self.stop_notify),
        );

        Ok(())
    }

    /// Returns a clone of the audio sender channel for use from non-async threads.
    ///
    /// Returns `None` if `connect()` has not been called yet.
    /// The returned sender can be used from any thread (including cpal callbacks)
    /// without holding the transport mutex.
    pub fn audio_sender(&self) -> Option<mpsc::Sender<Vec<u8>>> {
        self.audio_tx.clone()
    }

    /// Sends an encoded audio frame through the bounded channel.
    ///
    /// On channel full, the **new** frame is dropped and a warning is logged
    /// (backpressure strategy: discard newest).
    pub fn send_audio(&self, frame: Vec<u8>) -> Result<(), AppError> {
        let tx = self
            .audio_tx
            .as_ref()
            .ok_or_else(|| AppError::ConnectionFailed("not connected".into()))?;

        if let Err(mpsc::error::TrySendError::Closed(_)) = tx.try_send(frame) {
            return Err(AppError::ConnectionFailed("channel closed".into()));
        }
        // TrySendError::Full is intentionally ignored — frame dropped as backpressure
        Ok(())
    }

    /// Sends a JSON text frame over the WebSocket.
    pub fn send_text(&self, text: String) -> Result<(), AppError> {
        let tx = self
            .text_tx
            .as_ref()
            .ok_or_else(|| AppError::ConnectionFailed("not connected".into()))?;

        if let Err(mpsc::error::TrySendError::Closed(_)) = tx.try_send(text) {
            return Err(AppError::ConnectionFailed("text channel closed".into()));
        }
        Ok(())
    }

    /// Stops the transport gracefully.
    ///
    /// If `server_initiated` is false, sends a close_session control command
    /// before closing.
    pub async fn stop(&self, server_initiated: bool) {
        if !server_initiated {
            let sid = self.session_id.lock().ok().and_then(|g| g.clone());
            if let Some(session_id) = sid {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();
                let close_json = Self::build_close_session_json(&session_id, now);
                let _ = self.send_text(close_json);
            }
        }
        self.stop_notify.notify_waiters();
        *self.connected.lock().await = false;
    }

    /// Builds the JSON string for a close_session control command.
    ///
    /// Args:
    ///   session_id: Current session identifier.
    ///   timestamp: Epoch seconds.
    ///
    /// Returns: Serialized JSON string.
    pub fn build_close_session_json(session_id: &str, timestamp: f64) -> String {
        json!({
            "type": "control_command",
            "command": "close_session",
            "session_id": session_id,
            "timestamp": timestamp,
        })
        .to_string()
    }

    /// Builds the JSON string for a list_models control command.
    pub fn build_list_models_json(
        session_id: &str,
        timestamp: f64,
        request_id: Option<&str>,
    ) -> String {
        let mut payload = json!({
            "type": "control_command",
            "command": "list_models",
            "session_id": session_id,
            "timestamp": timestamp,
        });
        if let Some(request_id) = request_id {
            payload["request_id"] = request_id.into();
        }
        payload.to_string()
    }

    /// Builds the JSON string for a download_model control command.
    pub fn build_download_model_json(
        session_id: &str,
        timestamp: f64,
        model_name: &str,
        request_id: Option<&str>,
    ) -> String {
        let mut payload = json!({
            "type": "control_command",
            "command": "download_model",
            "session_id": session_id,
            "timestamp": timestamp,
            "model_name": model_name,
        });
        if let Some(request_id) = request_id {
            payload["request_id"] = request_id.into();
        }
        payload.to_string()
    }

    /// Spawns the drain loop that reads audio frames from the mpsc channel
    /// and forwards them to the WebSocket as binary messages.
    fn spawn_drain_loop<S>(ws_sink: Arc<Mutex<S>>, mut rx: mpsc::Receiver<Vec<u8>>)
    where
        S: futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error>
            + Unpin
            + Send
            + 'static,
    {
        tokio::spawn(async move {
            while let Some(frame) = rx.recv().await {
                let mut sink = ws_sink.lock().await;
                if sink.send(Message::Binary(frame.into())).await.is_err() {
                    tracing::warn!("WebSocket sink error in drain loop — stopping");
                    break;
                }
            }
        });
    }

    /// Spawns the text drain loop that reads JSON control commands from the mpsc
    /// channel and forwards them to the WebSocket as text messages.
    fn spawn_text_drain_loop<S>(ws_sink: Arc<Mutex<S>>, mut rx: mpsc::Receiver<String>)
    where
        S: futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error>
            + Unpin
            + Send
            + 'static,
    {
        tokio::spawn(async move {
            while let Some(text) = rx.recv().await {
                let mut sink = ws_sink.lock().await;
                if sink.send(Message::Text(text.into())).await.is_err() {
                    tracing::warn!("WebSocket sink error in text drain loop — stopping");
                    break;
                }
            }
        });
    }

    /// Spawns the receive loop that reads messages from the WebSocket
    /// and dispatches text frames to the registered message handler.
    fn spawn_receive_loop<S>(
        mut ws_stream: S,
        handler: Arc<Mutex<Option<MessageHandler>>>,
        stop_notify: Arc<Notify>,
    ) where
        S: futures_util::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>>
            + Unpin
            + Send
            + 'static,
    {
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    msg = ws_stream.next() => {
                        match msg {
                            Some(Ok(Message::Text(text))) => {
                                tracing::debug!("WS recv text: {text}");
                                if let Some(h) = handler.lock().await.as_ref() {
                                    h(text.to_string());
                                }
                            }
                            Some(Ok(Message::Binary(b))) => {
                                tracing::debug!("WS recv binary: {} bytes", b.len());
                            }
                            Some(Ok(Message::Close(_))) | None => {
                                tracing::info!("WS receive loop: connection closed");
                                break;
                            }
                            Some(Err(e)) => {
                                tracing::warn!("WebSocket receive error: {e}");
                                break;
                            }
                            _ => {}
                        }
                    }
                    _ = stop_notify.notified() => break,
                }
            }
        });
    }
}
