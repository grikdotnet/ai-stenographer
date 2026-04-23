/// Wire protocol types for communication with the STT server.
///
/// Handles encoding of audio frames (binary) and decoding of server
/// JSON text messages. The binary frame layout follows the project's
/// wire protocol: 4-byte LE header length + UTF-8 JSON header + raw f32 PCM.
use crate::error::AppError;
use crate::recognition::{RecognitionResult, RecognitionStatus};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub display_name: String,
    pub size_description: String,
    pub status: String,
}

/// Tagged enum representing all possible server-to-client messages.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    SessionCreated {
        session_id: String,
        protocol_version: String,
        server_time: f64,
    },
    RecognitionResult {
        session_id: String,
        status: String,
        text: String,
        start_time: f64,
        end_time: f64,
        #[serde(default)]
        chunk_ids: Vec<i64>,
        #[serde(default)]
        utterance_id: i64,
        #[serde(default)]
        token_confidences: Vec<f64>,
    },
    SessionClosed {
        session_id: String,
        reason: String,
        message: Option<String>,
    },
    ServerState {
        state: String,
    },
    Error {
        session_id: String,
        error_code: String,
        message: String,
        #[serde(default)]
        fatal: bool,
        #[serde(default)]
        request_id: Option<String>,
    },
    ModelList {
        models: Vec<ModelInfo>,
        request_id: Option<String>,
    },
    ModelStatus {
        status: String,
        request_id: Option<String>,
    },
    DownloadProgress {
        model_name: String,
        status: String,
        progress: Option<f64>,
        downloaded_bytes: Option<u64>,
        total_bytes: Option<u64>,
        error_message: Option<String>,
    },
    Ping {},
}

impl TryFrom<ServerMessage> for RecognitionResult {
    type Error = AppError;

    /// Converts a `ServerMessage::RecognitionResult` variant into the
    /// domain `RecognitionResult` type.
    fn try_from(msg: ServerMessage) -> Result<Self, AppError> {
        match msg {
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
                let status = match status.as_str() {
                    "partial" => RecognitionStatus::Partial,
                    "final" => RecognitionStatus::Final,
                    other => {
                        return Err(AppError::ProtocolError(format!(
                            "unknown recognition status: {other}"
                        )))
                    }
                };
                Ok(RecognitionResult {
                    session_id,
                    status,
                    text,
                    start_time,
                    end_time,
                    chunk_ids,
                    utterance_id,
                    token_confidences,
                })
            }
            _ => Err(AppError::ProtocolError(
                "expected RecognitionResult variant".into(),
            )),
        }
    }
}

/// Encodes audio frames into the binary wire format.
///
/// Binary layout (client -> server):
///   bytes [0..4)             : u32 LE header length
///   bytes [4..4+header_len)  : UTF-8 JSON header
///   bytes [4+header_len..)   : raw f32 PCM payload (little-endian)
pub struct AudioFrameEncoder;

impl AudioFrameEncoder {
    /// Encodes an audio frame with metadata into a binary message.
    ///
    /// Args:
    ///   session_id: Current session identifier.
    ///   chunk_id: Sequential chunk number.
    ///   timestamp: Capture timestamp in seconds.
    ///   audio_data: PCM samples as f32 slice.
    ///
    /// Returns: Binary vec ready to send as a WebSocket binary frame.
    pub fn encode(
        session_id: &str,
        chunk_id: i64,
        timestamp: f64,
        audio_data: &[f32],
    ) -> Vec<u8> {
        let header = json!({
            "type": "audio_chunk",
            "session_id": session_id,
            "chunk_id": chunk_id,
            "timestamp": timestamp,
        });
        let header_bytes = serde_json::to_vec(&header).expect("JSON serialization cannot fail");
        let header_len = header_bytes.len() as u32;

        let pcm_byte_len = audio_data.len() * std::mem::size_of::<f32>();
        let mut buf = Vec::with_capacity(4 + header_bytes.len() + pcm_byte_len);

        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(&header_bytes);
        for &sample in audio_data {
            buf.extend_from_slice(&sample.to_le_bytes());
        }

        buf
    }
}

/// Decodes server JSON text messages into `ServerMessage`.
pub struct ServerMessageDecoder;

impl ServerMessageDecoder {
    /// Deserializes a JSON text frame from the server.
    ///
    /// Args:
    ///   text: Raw JSON string from the WebSocket text frame.
    ///
    /// Returns: Parsed `ServerMessage` or `AppError::ProtocolError`.
    pub fn decode(text: &str) -> Result<ServerMessage, AppError> {
        serde_json::from_str(text).map_err(|e| AppError::ProtocolError(e.to_string()))
    }
}
