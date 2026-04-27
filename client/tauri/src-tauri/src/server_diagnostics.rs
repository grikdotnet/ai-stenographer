//! Bounded diagnostics captured from a Tauri-owned Python server process.
//!
//! This module intentionally stays pure: it owns diagnostic DTOs, bounded
//! output retention, and user-facing message formatting, but it does not spawn
//! processes or depend on Tauri runtime types.

use std::collections::VecDeque;
use std::process::ExitStatus;

const DEFAULT_MAX_LINES: usize = 50;

/// Identifies which process stream produced a diagnostic line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessOutputStream {
    Stdout,
    Stderr,
}

impl ProcessOutputStream {
    fn label(self) -> &'static str {
        match self {
            Self::Stdout => "stdout",
            Self::Stderr => "stderr",
        }
    }
}

/// One captured stdout/stderr line from a server process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessOutputLine {
    pub stream: ProcessOutputStream,
    pub text: String,
}

impl ProcessOutputLine {
    /// Creates a labeled process output line.
    ///
    /// Args:
    /// - `stream`: The process stream that produced the line.
    /// - `text`: The line text without a trailing newline.
    ///
    /// Returns:
    /// A process diagnostic line suitable for buffering and display.
    pub fn new(stream: ProcessOutputStream, text: impl Into<String>) -> Self {
        Self {
            stream,
            text: text.into(),
        }
    }

    fn format(&self) -> String {
        format!("[{}] {}", self.stream.label(), self.text)
    }
}

/// Immutable snapshot of retained diagnostics for one server process.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProcessDiagnosticsSnapshot {
    pub lines: Vec<ProcessOutputLine>,
}

impl ProcessDiagnosticsSnapshot {
    /// Creates a snapshot from already ordered process lines.
    ///
    /// Args:
    /// - `lines`: Lines ordered by arrival time, oldest to newest.
    ///
    /// Returns:
    /// A diagnostic snapshot.
    pub fn new(lines: Vec<ProcessOutputLine>) -> Self {
        Self { lines }
    }

    /// Returns true when no process output was retained.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    fn formatted_output(&self) -> Option<String> {
        if self.lines.is_empty() {
            return None;
        }
        Some(
            self.lines
                .iter()
                .map(ProcessOutputLine::format)
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }
}

/// Bounded in-memory output buffer for one server process generation.
pub struct ProcessDiagnosticsBuffer {
    max_lines: usize,
    lines: VecDeque<ProcessOutputLine>,
}

impl Default for ProcessDiagnosticsBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_LINES)
    }
}

impl ProcessDiagnosticsBuffer {
    /// Creates an empty bounded diagnostics buffer.
    ///
    /// Args:
    /// - `max_lines`: Maximum number of stdout/stderr lines to retain.
    ///
    /// Returns:
    /// A process diagnostics buffer.
    pub fn new(max_lines: usize) -> Self {
        Self {
            max_lines,
            lines: VecDeque::new(),
        }
    }

    /// Records one process output line.
    ///
    /// Args:
    /// - `stream`: The process stream that produced the line.
    /// - `text`: The line text without a trailing newline.
    pub fn push(&mut self, stream: ProcessOutputStream, text: impl Into<String>) {
        if self.max_lines == 0 {
            return;
        }
        if self.lines.len() == self.max_lines {
            self.lines.pop_front();
        }
        self.lines.push_back(ProcessOutputLine::new(stream, text));
    }

    /// Captures the currently retained output in arrival order.
    ///
    /// Returns:
    /// A snapshot ordered from oldest retained line to newest retained line.
    pub fn snapshot(&self) -> ProcessDiagnosticsSnapshot {
        ProcessDiagnosticsSnapshot::new(self.lines.iter().cloned().collect())
    }
}

/// Describes which owned-server lifecycle phase failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServerFailurePhase {
    Startup,
    Runtime,
}

/// Formats a startup failure reported by the owned Python process.
///
/// Args:
/// - `status`: Process exit status when it is available.
/// - `diagnostics`: Retained stdout/stderr for the failed process.
///
/// Returns:
/// A concise user-facing diagnostic message.
pub fn format_startup_exit_message(
    status: Option<ExitStatus>,
    diagnostics: &ProcessDiagnosticsSnapshot,
) -> String {
    format_owned_server_failure_message(ServerFailurePhase::Startup, None, status, diagnostics)
}

/// Formats a terminal runtime crash after the restart policy is exhausted.
///
/// Args:
/// - `crash_count`: Number of unstable crashes observed by the supervisor.
/// - `status`: Process exit status when it is available.
/// - `diagnostics`: Retained stdout/stderr for the failed process.
///
/// Returns:
/// A concise user-facing diagnostic message.
pub fn format_terminal_runtime_crash_message(
    crash_count: u8,
    status: Option<ExitStatus>,
    diagnostics: &ProcessDiagnosticsSnapshot,
) -> String {
    format_owned_server_failure_message(
        ServerFailurePhase::Runtime,
        Some(crash_count),
        status,
        diagnostics,
    )
}

fn format_owned_server_failure_message(
    phase: ServerFailurePhase,
    crash_count: Option<u8>,
    status: Option<ExitStatus>,
    diagnostics: &ProcessDiagnosticsSnapshot,
) -> String {
    let mut message = match phase {
        ServerFailurePhase::Startup => {
            "Python server failed during startup before reporting its WebSocket URL".to_string()
        }
        ServerFailurePhase::Runtime => match crash_count {
            Some(count) => {
                format!("Python server crashed during runtime after {count} unstable failures")
            }
            None => "Python server crashed during runtime".to_string(),
        },
    };
    if let Some(status) = status {
        message.push_str(&format!(" (exit status: {status})"));
    }
    if let Some(output) = diagnostics.formatted_output() {
        message.push_str("\nRecent server output:\n");
        message.push_str(&output);
    }
    message
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_buffer_retains_newest_50_labeled_lines_in_arrival_order() {
        let mut buffer = ProcessDiagnosticsBuffer::default();

        for i in 0..60 {
            let stream = if i % 2 == 0 {
                ProcessOutputStream::Stdout
            } else {
                ProcessOutputStream::Stderr
            };
            buffer.push(stream, format!("line-{i}"));
        }

        let snapshot = buffer.snapshot();
        assert_eq!(snapshot.lines.len(), 50);
        assert_eq!(
            snapshot.lines.first(),
            Some(&ProcessOutputLine::new(
                ProcessOutputStream::Stdout,
                "line-10"
            ))
        );
        assert_eq!(
            snapshot.lines.last(),
            Some(&ProcessOutputLine::new(
                ProcessOutputStream::Stderr,
                "line-59"
            ))
        );

        let rendered = snapshot.formatted_output().unwrap();
        assert!(rendered.contains("[stdout] line-10"));
        assert!(rendered.contains("[stderr] line-59"));
        assert!(!rendered.contains("line-9"));
    }

    #[test]
    fn startup_exit_message_includes_phase_status_and_recent_output() {
        let snapshot = ProcessDiagnosticsSnapshot::new(vec![ProcessOutputLine::new(
            ProcessOutputStream::Stderr,
            "Traceback: boom",
        )]);

        let message = format_startup_exit_message(None, &snapshot);

        assert!(message.contains("startup"));
        assert!(message.contains("before reporting its WebSocket URL"));
        assert!(message.contains("[stderr] Traceback: boom"));
    }

    #[test]
    fn runtime_crash_message_includes_crash_count_and_recent_output() {
        let snapshot = ProcessDiagnosticsSnapshot::new(vec![ProcessOutputLine::new(
            ProcessOutputStream::Stdout,
            "accepted connection",
        )]);

        let message = format_terminal_runtime_crash_message(2, None, &snapshot);

        assert!(message.contains("runtime"));
        assert!(message.contains("2 unstable failures"));
        assert!(message.contains("[stdout] accepted connection"));
    }
}
