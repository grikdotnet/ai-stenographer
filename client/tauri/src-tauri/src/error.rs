/// Unified error types for the STT Tauri client application.
///
/// Covers state machine violations, network failures, protocol
/// decoding problems, and audio subsystem issues.
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum AppError {
    #[error("invalid state transition from {from} to {to}")]
    InvalidTransition { from: String, to: String },

    #[error("connection failed: {0}")]
    ConnectionFailed(String),

    #[error("protocol error: {0}")]
    ProtocolError(String),

    #[error("audio error: {0}")]
    AudioError(String),

    #[error("keyboard error: {0}")]
    KeyboardError(String),
}
