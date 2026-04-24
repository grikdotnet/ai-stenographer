/// Library root for the STT Tauri client.
///
/// Re-exports core modules so integration tests and the binary
/// can share the same types.
pub mod audio;
pub mod cli;
pub mod commands;
pub mod error;
pub mod focus;
pub mod formatting;
pub mod hotkey;
pub mod insertion;
pub mod keyboard;
pub mod orchestrator;
pub mod protocol;
pub mod quickentry;
pub mod recognition;
pub mod state;
pub mod transport;
