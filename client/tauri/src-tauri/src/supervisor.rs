//! Process supervisor for a Tauri-owned Python STT server.
//!
//! This module is the client infrastructure boundary for development-time
//! Python process ownership: command resolution, child process startup,
//! stdout/stderr draining, normal shutdown, and unstable-crash retry policy.

use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, Mutex, Notify};

use crate::server_diagnostics::{
    format_startup_exit_message, ProcessDiagnosticsBuffer, ProcessDiagnosticsSnapshot,
    ProcessOutputStream,
};

const SERVER_URL_PREFIX: &str = "Server listening on ";
const UNSTABLE_FAILURES_BEFORE_TERMINAL: u8 = 2;
pub const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
pub const DEFAULT_STABILITY_RESET: Duration = Duration::from_secs(60);

/// Test/injection overrides; highest priority and bypass auto-detection.
#[derive(Clone, Debug, Default)]
pub struct CommandOverrides {
    pub python_executable: Option<PathBuf>,
    pub entry_point: Option<PathBuf>,
    pub current_dir: Option<PathBuf>,
}

/// Configuration for the Python server supervisor.
///
/// `exe_dir` is used for portable layout probing (next to `AI-Stenographer.exe`).
/// `dev_search_start` is used as the starting point for the development repo walk-up.
#[derive(Clone, Debug)]
pub struct ServerSupervisorConfig {
    pub exe_dir: PathBuf,
    pub dev_search_start: PathBuf,
    pub startup_timeout: Duration,
    pub stability_reset: Duration,
    pub overrides: CommandOverrides,
}

impl ServerSupervisorConfig {
    /// Creates a config given the directory of the running executable
    /// (for portable detection) and a development search start (for repo walk-up).
    pub fn new(exe_dir: impl Into<PathBuf>, dev_search_start: impl Into<PathBuf>) -> Self {
        Self {
            exe_dir: exe_dir.into(),
            dev_search_start: dev_search_start.into(),
            startup_timeout: DEFAULT_STARTUP_TIMEOUT,
            stability_reset: DEFAULT_STABILITY_RESET,
            overrides: CommandOverrides::default(),
        }
    }
}

/// Resolved command used to spawn the Python server.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerCommandSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub current_dir: PathBuf,
}

/// Runtime events emitted by the supervisor.
#[derive(Debug)]
pub enum ServerProcessEvent {
    Exited {
        status: Option<ExitStatus>,
        expected: bool,
        diagnostics: ProcessDiagnosticsSnapshot,
    },
}

/// Crash policy decision for an unstable server failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrashDecision {
    Restart,
    Terminal,
}

/// Errors produced by server supervision.
#[derive(Debug, Error)]
pub enum ServerSupervisorError {
    #[error("Python executable not found: {0}")]
    MissingPython(PathBuf),

    #[error("Python server entry point not found: {0}")]
    MissingEntryPoint(PathBuf),

    #[error("Python server runtime not found: portable python={portable_python}, portable entry={portable_entry}, dev search from={dev_candidate:?}")]
    MissingServerRuntime {
        portable_python: PathBuf,
        portable_entry: PathBuf,
        dev_candidate: Option<PathBuf>,
    },

    #[error("failed to spawn Python server: {0}")]
    Spawn(std::io::Error),

    #[error("Python server stdout was not captured")]
    MissingStdout,

    #[error("Python server startup timed out")]
    StartupTimeout,

    #[error("Python server startup was cancelled")]
    StartupCancelled,

    #[error("{}", format_startup_exit_message(*status, diagnostics))]
    StartupExited {
        status: Option<ExitStatus>,
        diagnostics: ProcessDiagnosticsSnapshot,
    },
}

/// Tauri-owned Python server supervisor.
pub struct ServerSupervisor {
    config: ServerSupervisorConfig,
    current_child: Arc<Mutex<Option<Arc<Mutex<Child>>>>>,
    shutdown: Arc<AtomicBool>,
    shutdown_notify: Arc<Notify>,
    events_tx: mpsc::UnboundedSender<ServerProcessEvent>,
    events_rx: Mutex<mpsc::UnboundedReceiver<ServerProcessEvent>>,
    unstable_failures: Arc<Mutex<u8>>,
    generation: Arc<AtomicU64>,
    current_diagnostics: Arc<Mutex<Option<Arc<Mutex<ProcessDiagnosticsBuffer>>>>>,
}

impl ServerSupervisor {
    /// Creates a supervisor with no running child process.
    pub fn new(config: ServerSupervisorConfig) -> Self {
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        Self {
            config,
            current_child: Arc::new(Mutex::new(None)),
            shutdown: Arc::new(AtomicBool::new(false)),
            shutdown_notify: Arc::new(Notify::new()),
            events_tx,
            events_rx: Mutex::new(events_rx),
            unstable_failures: Arc::new(Mutex::new(0)),
            generation: Arc::new(AtomicU64::new(0)),
            current_diagnostics: Arc::new(Mutex::new(None)),
        }
    }

    /// Resolves and validates the server command.
    ///
    /// Resolution order: overrides → portable layout next to `exe_dir` →
    /// development repo walk-up from `dev_search_start`.
    pub fn command_spec(&self) -> Result<ServerCommandSpec, ServerSupervisorError> {
        resolve_command_spec(
            &self.config.exe_dir,
            &self.config.dev_search_start,
            &self.config.overrides,
        )
    }

    /// Starts the Python server and returns the parsed WebSocket URL.
    ///
    /// Stdout and stderr are continuously drained after the URL line is found.
    pub async fn start(&self) -> Result<String, ServerSupervisorError> {
        self.shutdown.store(false, Ordering::SeqCst);
        let diagnostics = Arc::new(Mutex::new(ProcessDiagnosticsBuffer::default()));
        *self.current_diagnostics.lock().await = Some(Arc::clone(&diagnostics));
        let spec = self.command_spec()?;
        let mut command = Command::new(&spec.program);
        command
            .args(&spec.args)
            .current_dir(&spec.current_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // The Tauri client is built with windows_subsystem = "windows", so spawning a
        // console-subsystem python.exe would flash a console window. CREATE_NO_WINDOW
        // (0x08000000) suppresses it; stdout/stderr remain piped via the parent.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.as_std_mut().creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = command.spawn().map_err(ServerSupervisorError::Spawn)?;
        let stdout = child
            .stdout
            .take()
            .ok_or(ServerSupervisorError::MissingStdout)?;
        if let Some(stderr) = child.stderr.take() {
            spawn_stderr_drain(stderr, Arc::clone(&diagnostics));
        }

        let child = Arc::new(Mutex::new(child));
        *self.current_child.lock().await = Some(Arc::clone(&child));

        let startup = wait_for_server_url(
            stdout,
            Arc::clone(&child),
            Arc::clone(&self.shutdown_notify),
            Arc::clone(&diagnostics),
            self.config.startup_timeout,
        )
        .await;

        match startup {
            Ok(url) => {
                self.generation.fetch_add(1, Ordering::SeqCst);
                self.spawn_exit_monitor(Arc::clone(&child), Arc::clone(&diagnostics));
                Ok(url)
            }
            Err(err) => {
                terminate_child(&child).await;
                *self.current_child.lock().await = None;
                Err(err)
            }
        }
    }

    /// Marks the current child as connected and starts the stability reset timer.
    ///
    /// The timer captures the currently running child generation. If a later
    /// restart increments the generation before the timer fires, the reset is
    /// ignored so a stale connection marker cannot clear a newer crash count.
    pub async fn mark_connected(&self) {
        let generation = self.generation.load(Ordering::SeqCst);
        let failures = Arc::clone(&self.unstable_failures);
        let generation_counter = Arc::clone(&self.generation);
        let reset_after = self.config.stability_reset;
        tokio::spawn(async move {
            tokio::time::sleep(reset_after).await;
            if generation_counter.load(Ordering::SeqCst) == generation {
                *failures.lock().await = 0;
            }
        });
    }

    /// Records an unstable failure and returns the supervisor retry decision.
    pub async fn record_unstable_failure(&self) -> CrashDecision {
        let mut failures = self.unstable_failures.lock().await;
        *failures = failures.saturating_add(1);
        if *failures < UNSTABLE_FAILURES_BEFORE_TERMINAL {
            CrashDecision::Restart
        } else {
            CrashDecision::Terminal
        }
    }

    /// Returns the number of unstable failures recorded for the current run.
    ///
    /// Intended for diagnostic message formatting. Restart policy decisions
    /// should continue to flow through `record_unstable_failure()`.
    pub async fn unstable_failure_count(&self) -> u8 {
        *self.unstable_failures.lock().await
    }

    /// Waits for the next process event from the current owned child.
    pub async fn next_event(&self) -> Option<ServerProcessEvent> {
        self.events_rx.lock().await.recv().await
    }

    /// Terminates the current child through the normal shutdown path.
    pub async fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.shutdown_notify.notify_waiters();
        *self.unstable_failures.lock().await = 0;
        if let Some(child) = self.current_child.lock().await.take() {
            terminate_child(&child).await;
        }
    }

    /// Returns whether the supervisor currently has a live child process.
    ///
    /// Intended for diagnostics and process-lifecycle tests; production flow
    /// should react to `next_event()` instead.
    pub async fn has_running_child(&self) -> bool {
        if let Some(child) = self.current_child.lock().await.as_ref() {
            return !child_exited(child).await;
        }
        false
    }

    fn spawn_exit_monitor(
        &self,
        child: Arc<Mutex<Child>>,
        diagnostics: Arc<Mutex<ProcessDiagnosticsBuffer>>,
    ) {
        let shutdown = Arc::clone(&self.shutdown);
        let events = self.events_tx.clone();
        tokio::spawn(async move {
            loop {
                let status = {
                    let mut guard = child.lock().await;
                    match guard.try_wait() {
                        Ok(Some(status)) => Some(status),
                        Ok(None) => None,
                        Err(err) => {
                            tracing::warn!("Python server try_wait failed: {err}");
                            None
                        }
                    }
                };
                if let Some(status) = status {
                    let diagnostics = diagnostics.lock().await.snapshot();
                    let _ = events.send(ServerProcessEvent::Exited {
                        status: Some(status),
                        expected: shutdown.load(Ordering::SeqCst),
                        diagnostics,
                    });
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });
    }
}

/// Pure command-resolution helper. Does not call `current_exe()` or `current_dir()`.
///
/// Resolution order:
/// 1. Overrides — if either `python_executable` or `entry_point` is set, build
///    using overrides plus existing dev-style fallback for the other.
/// 2. Portable — `<exe_dir>/_internal/runtime/python.exe` plus `<exe_dir>/_internal/app/main.pyc`.
/// 3. Dev — `find_repo_root(dev_search_start)` for `venv/Scripts/python.exe` plus `main.py`.
/// 4. Otherwise — `MissingServerRuntime` with both checked candidates.
pub fn resolve_command_spec(
    exe_dir: &Path,
    dev_search_start: &Path,
    overrides: &CommandOverrides,
) -> Result<ServerCommandSpec, ServerSupervisorError> {
    if overrides.python_executable.is_some() || overrides.entry_point.is_some() {
        return resolve_with_overrides(exe_dir, dev_search_start, overrides);
    }

    let portable_python = exe_dir.join("_internal").join("runtime").join("python.exe");
    let portable_entry = exe_dir.join("_internal").join("app").join("main.pyc");
    if portable_python.exists() && portable_entry.exists() {
        return Ok(ServerCommandSpec {
            program: portable_python,
            args: vec![
                portable_entry.to_string_lossy().to_string(),
                "--host=127.0.0.1".to_string(),
            ],
            current_dir: exe_dir.to_path_buf(),
        });
    }

    if let Some(repo) = find_repo_root(dev_search_start) {
        let python = repo.join("venv").join("Scripts").join("python.exe");
        let entry = repo.join("main.py");
        return Ok(ServerCommandSpec {
            program: python,
            args: vec![
                entry.to_string_lossy().to_string(),
                "--host=127.0.0.1".to_string(),
            ],
            current_dir: repo,
        });
    }

    Err(ServerSupervisorError::MissingServerRuntime {
        portable_python,
        portable_entry,
        dev_candidate: Some(dev_search_start.to_path_buf()),
    })
}

fn resolve_with_overrides(
    exe_dir: &Path,
    dev_search_start: &Path,
    overrides: &CommandOverrides,
) -> Result<ServerCommandSpec, ServerSupervisorError> {
    let dev_root = find_repo_root(dev_search_start);
    let fallback_python = dev_root
        .as_ref()
        .map(|r| r.join("venv").join("Scripts").join("python.exe"))
        .unwrap_or_else(|| exe_dir.join("_internal").join("runtime").join("python.exe"));
    let fallback_entry = dev_root
        .as_ref()
        .map(|r| r.join("main.py"))
        .unwrap_or_else(|| exe_dir.join("_internal").join("app").join("main.pyc"));

    let python = overrides
        .python_executable
        .clone()
        .unwrap_or(fallback_python);
    let entry_point = overrides.entry_point.clone().unwrap_or(fallback_entry);

    if !python.exists() {
        return Err(ServerSupervisorError::MissingPython(python));
    }
    if !entry_point.exists() {
        return Err(ServerSupervisorError::MissingEntryPoint(entry_point));
    }

    let current_dir = overrides
        .current_dir
        .clone()
        .or_else(|| dev_root.clone())
        .or_else(|| entry_point.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| exe_dir.to_path_buf());

    Ok(ServerCommandSpec {
        program: python,
        args: vec![
            entry_point.to_string_lossy().to_string(),
            "--host=127.0.0.1".to_string(),
        ],
        current_dir,
    })
}

/// Finds a repository root by walking up from `start`.
pub fn find_repo_root(start: impl AsRef<Path>) -> Option<PathBuf> {
    let mut current = start.as_ref();
    loop {
        if current.join("main.py").exists()
            && current
                .join("venv")
                .join("Scripts")
                .join("python.exe")
                .exists()
        {
            return Some(current.to_path_buf());
        }
        current = current.parent()?;
    }
}

/// Parses the server URL from a stdout line.
pub fn parse_server_url_line(line: &str) -> Option<String> {
    let start = line.find(SERVER_URL_PREFIX)? + SERVER_URL_PREFIX.len();
    line[start..]
        .split_whitespace()
        .next()
        .filter(|url| url.starts_with("ws://") || url.starts_with("wss://"))
        .map(str::to_string)
}

async fn wait_for_server_url(
    stdout: tokio::process::ChildStdout,
    child: Arc<Mutex<Child>>,
    shutdown_notify: Arc<Notify>,
    diagnostics: Arc<Mutex<ProcessDiagnosticsBuffer>>,
    timeout: Duration,
) -> Result<String, ServerSupervisorError> {
    let mut lines = BufReader::new(stdout).lines();
    let read_url = async {
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    diagnostics
                        .lock()
                        .await
                        .push(ProcessOutputStream::Stdout, line.clone());
                    if let Some(url) = parse_server_url_line(&line) {
                        spawn_stdout_drain(lines, Arc::clone(&diagnostics));
                        return Ok(url);
                    }
                    tracing::info!(target: "stt::server", "{line}");
                }
                Ok(None) => {
                    return Err(startup_exited_error(&child, &diagnostics).await);
                }
                Err(err) => return Err(ServerSupervisorError::Spawn(err)),
            }
            if child_exited(&child).await {
                return Err(startup_exited_error(&child, &diagnostics).await);
            }
        }
    };

    tokio::select! {
        result = tokio::time::timeout(timeout, read_url) => {
            result.map_err(|_| ServerSupervisorError::StartupTimeout)?
        }
        _ = shutdown_notify.notified() => Err(ServerSupervisorError::StartupCancelled),
    }
}

fn spawn_stdout_drain(
    mut lines: tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
    diagnostics: Arc<Mutex<ProcessDiagnosticsBuffer>>,
) {
    tokio::spawn(async move {
        while let Ok(Some(line)) = lines.next_line().await {
            diagnostics
                .lock()
                .await
                .push(ProcessOutputStream::Stdout, line.clone());
            tracing::info!(target: "stt::server", "{line}");
        }
    });
}

fn spawn_stderr_drain(
    stderr: tokio::process::ChildStderr,
    diagnostics: Arc<Mutex<ProcessDiagnosticsBuffer>>,
) {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            diagnostics
                .lock()
                .await
                .push(ProcessOutputStream::Stderr, line.clone());
            tracing::warn!(target: "stt::server", "{line}");
        }
    });
}

async fn child_exited(child: &Arc<Mutex<Child>>) -> bool {
    child.lock().await.try_wait().ok().flatten().is_some()
}

async fn child_wait_status(child: &Arc<Mutex<Child>>) -> Option<ExitStatus> {
    child.lock().await.wait().await.ok()
}

async fn startup_exited_error(
    child: &Arc<Mutex<Child>>,
    diagnostics: &Arc<Mutex<ProcessDiagnosticsBuffer>>,
) -> ServerSupervisorError {
    ServerSupervisorError::StartupExited {
        status: child_wait_status(child).await,
        diagnostics: diagnostics.lock().await.snapshot(),
    }
}

async fn terminate_child(child: &Arc<Mutex<Child>>) {
    let mut guard = child.lock().await;
    if guard.try_wait().ok().flatten().is_none() {
        let _ = guard.start_kill();
        let _ = guard.wait().await;
    }
}
