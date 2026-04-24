/// Command-line argument definitions for the STT Tauri client.
///
/// Parses `--server-url` (required WebSocket endpoint) and an optional
/// `--input-file` for feeding a WAV file instead of live microphone input.
/// Uses `try_parse` so unknown Tauri-injected arguments don't cause a panic.
///
/// During `tauri dev`, Tauri 2 CLI inserts extra args before cargo's own `--`
/// separator, so `--server-url` is seen by cargo rather than the binary.
/// The `STT_SERVER_URL` env var bypasses this by not needing CLI forwarding at all.
use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(about = "STT desktop client — connects to the Python STT server via WebSocket")]
pub struct CliArgs {
    /// WebSocket URL of the STT server (e.g. ws://localhost:8765).
    /// Can also be set via the `STT_SERVER_URL` environment variable.
    #[arg(long, env = "STT_SERVER_URL")]
    pub server_url: Option<String>,

    /// Optional WAV file to use as audio input instead of the microphone.
    /// Can also be set via the `STT_INPUT_FILE` environment variable.
    #[arg(long, env = "STT_INPUT_FILE")]
    pub input_file: Option<String>,

    /// Run without the GUI — audio+WebSocket pipeline only, logs to stdout
    #[arg(long)]
    pub headless: bool,
}
