#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;

use clap::Parser;
use tauri::{Emitter, Manager};
use tracing::info;

use stt_tauri_client::cli::CliArgs;
use stt_tauri_client::commands;
use stt_tauri_client::insertion::InsertionController;
use stt_tauri_client::orchestrator::{
    emit_event, ClientOrchestrator, ConnectionStatus, EventEmitter, StateChanged, TranscriptUpdate,
};
use stt_tauri_client::recognition::{RecognitionResult, RecognitionSubscriber};
use stt_tauri_client::server_diagnostics::format_terminal_runtime_crash_message;
use stt_tauri_client::state::{AppState, AppStateManager};
use stt_tauri_client::supervisor::{
    CrashDecision, ServerProcessEvent, ServerSupervisor, ServerSupervisorConfig,
};

/// Bridges the `EventEmitter` trait to a Tauri `AppHandle`.
struct TauriEmitter {
    handle: tauri::AppHandle,
}

impl EventEmitter for TauriEmitter {
    fn emit_serialized(&self, event: &str, payload: &str) {
        let _ = self.handle.emit(
            event,
            serde_json::value::RawValue::from_string(payload.to_owned()).ok(),
        );
    }
}

/// Wraps a Tauri `AppHandle` to show/hide the QuickEntry popup window.
///
/// Implements `PopupWindow` so `QuickEntryController` stays decoupled from Tauri types.
/// The popup window is created lazily on first `show()` via `WebviewWindowBuilder`.
#[cfg(windows)]
struct TauriPopupWindow {
    handle: tauri::AppHandle,
}

#[cfg(windows)]
impl stt_tauri_client::quickentry::PopupWindow for TauriPopupWindow {
    fn show(&self) {
        use tauri::{WebviewUrl, WebviewWindowBuilder};
        let existing = self.handle.get_webview_window("quickentry");
        let window = existing.unwrap_or_else(|| {
            WebviewWindowBuilder::new(
                &self.handle,
                "quickentry",
                WebviewUrl::App("/quickentry.html".into()),
            )
            .title("QuickEntry")
            .inner_size(720.0, 120.0)
            .center()
            .always_on_top(true)
            .decorations(false)
            .transparent(true)
            .skip_taskbar(true)
            .build()
            .expect("quickentry window")
        });
        let _ = window.show();
        let _ = window.set_focus();
    }

    fn hide(&self) {
        if let Some(window) = self.handle.get_webview_window("quickentry") {
            let _ = window.hide();
        }
    }
}

/// Bridges `Arc<QuickEntrySubscriber>` to `Box<dyn RecognitionSubscriber>`.
///
/// Required because `Arc<T>` does not automatically implement a trait even if `T` does.
#[cfg(windows)]
struct QuickEntrySubscriberBridge(Arc<stt_tauri_client::quickentry::QuickEntrySubscriber>);

#[cfg(windows)]
impl RecognitionSubscriber for QuickEntrySubscriberBridge {
    fn on_partial(&self, result: &RecognitionResult) {
        self.0.on_partial(result);
    }

    fn on_finalization(&self, result: &RecognitionResult) {
        self.0.on_finalization(result);
    }
}

/// Creates and manages the QuickEntryController on Windows.
///
/// Returns the subscriber Arc so the caller can wire it into the orchestrator's fan-out.
#[cfg(windows)]
fn setup_quickentry(
    app: &tauri::App,
    emitter: Arc<dyn EventEmitter>,
    state_manager: Arc<AppStateManager>,
) -> Option<Arc<stt_tauri_client::quickentry::QuickEntrySubscriber>> {
    use stt_tauri_client::focus::{FocusTracker, Win32FocusApi};
    use stt_tauri_client::hotkey::{GlobalHotkeyListener, HotkeyListener};
    use stt_tauri_client::keyboard::Win32KeyboardSimulator;
    use stt_tauri_client::quickentry::{handle_toggle, QuickEntryController, QuickEntrySubscriber};

    let focus_tracker = Arc::new(FocusTracker::new(Win32FocusApi));
    let keyboard: Arc<dyn stt_tauri_client::keyboard::KeyboardSimulator> =
        Arc::new(Win32KeyboardSimulator);
    let subscriber = Arc::new(QuickEntrySubscriber::new());
    let hotkey = Arc::new(GlobalHotkeyListener::new());
    let hotkey_for_start = Arc::clone(&hotkey);
    let popup_window: Arc<dyn stt_tauri_client::quickentry::PopupWindow> =
        Arc::new(TauriPopupWindow {
            handle: app.handle().clone(),
        });

    let controller = Arc::new(QuickEntryController::new(
        focus_tracker,
        keyboard,
        Arc::clone(&subscriber),
        emitter,
        hotkey,
        popup_window,
    ));

    let controller_for_submit = Arc::clone(&controller);
    let controller_for_cancel = Arc::clone(&controller);
    controller.set_popup_callbacks(
        Arc::new(move || controller_for_submit.submit()),
        Arc::new(move || controller_for_cancel.cancel()),
    );

    let controller_for_pause = Arc::clone(&controller);
    state_manager.add_observer(Box::new(move |_old, new| {
        if new == AppState::Paused && controller_for_pause.is_visible() {
            controller_for_pause.cancel();
        }
    }));

    let state_for_toggle = Arc::clone(&state_manager);
    let controller_for_toggle = Arc::clone(&controller);
    hotkey_for_start.start(Box::new(move || {
        handle_toggle(&controller_for_toggle, state_for_toggle.current_state());
    }));

    app.manage(controller);
    Some(subscriber)
}

#[cfg(not(windows))]
fn setup_quickentry(
    _app: &tauri::App,
    _emitter: Arc<dyn EventEmitter>,
    _state_manager: Arc<AppStateManager>,
) -> Option<Arc<stt_tauri_client::quickentry::QuickEntrySubscriber>> {
    None
}

fn create_owned_supervisor() -> Result<Arc<ServerSupervisor>, String> {
    let exe = std::env::current_exe()
        .map_err(|err| format!("failed to resolve current executable: {err}"))?;
    let exe_dir = exe
        .parent()
        .ok_or_else(|| "current executable has no parent directory".to_string())?
        .to_path_buf();
    let cwd = std::env::current_dir()
        .map_err(|err| format!("failed to resolve current directory: {err}"))?;
    let supervisor = ServerSupervisor::new(ServerSupervisorConfig::new(exe_dir, cwd));
    // Probe resolution eagerly so misconfiguration surfaces before Tauri starts.
    supervisor
        .command_spec()
        .map_err(|err| format!("Python server runtime resolution failed: {err}"))?;
    Ok(Arc::new(supervisor))
}

async fn emit_terminal_error(
    orchestrator: &Arc<tokio::sync::Mutex<ClientOrchestrator>>,
    message: String,
) {
    let mut guard = orchestrator.lock().await;
    guard.emit_connection_error(message);
    guard.stop(true).await;
}

async fn connect_and_start_audio(
    orchestrator: &Arc<tokio::sync::Mutex<ClientOrchestrator>>,
    supervisor: Option<Arc<ServerSupervisor>>,
) -> Result<(), String> {
    if let Some(supervisor) = supervisor {
        let server_url = supervisor
            .start()
            .await
            .map_err(|err| format!("Python server startup failed: {err}"))?;
        orchestrator.lock().await.replace_server_url(server_url);
    }

    let mut guard = orchestrator.lock().await;
    guard.connect().await.map_err(|err| err.to_string())?;
    guard.start_audio().map_err(|err| err.to_string())
}

async fn reconnect_owned_server_once(
    supervisor: Arc<ServerSupervisor>,
    orchestrator: Arc<tokio::sync::Mutex<ClientOrchestrator>>,
) -> Result<(), String> {
    {
        let mut guard = orchestrator.lock().await;
        guard.stop(true).await;
    }
    let server_url = supervisor
        .start()
        .await
        .map_err(|err| format!("Python server restart failed: {err}"))?;
    {
        let mut guard = orchestrator.lock().await;
        guard.replace_server_url(server_url);
        guard.connect().await.map_err(|err| err.to_string())?;
        guard.start_audio().map_err(|err| err.to_string())?;
    }
    Ok(())
}

async fn monitor_owned_server(
    supervisor: Arc<ServerSupervisor>,
    orchestrator: Arc<tokio::sync::Mutex<ClientOrchestrator>>,
) {
    while let Some(event) = supervisor.next_event().await {
        match event {
            ServerProcessEvent::Exited { expected: true, .. } => break,
            ServerProcessEvent::Exited {
                expected: false,
                status,
                diagnostics,
            } => {
                tracing::error!("Owned Python server exited unexpectedly: {status:?}");
                match supervisor.record_unstable_failure().await {
                    CrashDecision::Restart => {
                        if let Err(err) = reconnect_owned_server_once(
                            Arc::clone(&supervisor),
                            Arc::clone(&orchestrator),
                        )
                        .await
                        {
                            let _ = supervisor.record_unstable_failure().await;
                            let failure_count = supervisor.unstable_failure_count().await;
                            let message = format!(
                                "{}\nRestart failed: {err}",
                                format_terminal_runtime_crash_message(
                                    failure_count,
                                    status,
                                    &diagnostics
                                )
                            );
                            emit_terminal_error(&orchestrator, message).await;
                            break;
                        }
                    }
                    CrashDecision::Terminal => {
                        let failure_count = supervisor.unstable_failure_count().await;
                        emit_terminal_error(
                            &orchestrator,
                            format_terminal_runtime_crash_message(
                                failure_count,
                                status,
                                &diagnostics,
                            ),
                        )
                        .await;
                        break;
                    }
                }
            }
        }
    }
}

fn install_supervisor_connected_marker(
    orchestrator: &mut ClientOrchestrator,
    supervisor: Arc<ServerSupervisor>,
) {
    orchestrator.set_connected_callback(Arc::new(move || {
        let supervisor = Arc::clone(&supervisor);
        tokio::spawn(async move {
            supervisor.mark_connected().await;
        });
    }));
}

/// Runs the STT pipeline without any GUI.
///
/// Connects to the server, starts audio capture, and blocks until
/// Ctrl+C is received, then shuts down gracefully.
fn run_headless(
    server_url: Option<String>,
    input_file: Option<String>,
    state_manager: Arc<AppStateManager>,
) -> bool {
    use std::io::Write;
    use stt_tauri_client::orchestrator::NoopEmitter;

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async move {
        let owned_supervisor = if server_url.is_none() {
            match create_owned_supervisor() {
                Ok(supervisor) => Some(supervisor),
                Err(err) => {
                    tracing::error!("{err}");
                    return false;
                }
            }
        } else {
            None
        };
        let mut orchestrator = ClientOrchestrator::new(
            server_url.unwrap_or_default(),
            input_file,
            state_manager,
        );
        orchestrator.set_emitter(Arc::new(NoopEmitter));
        if let Some(supervisor) = owned_supervisor.as_ref() {
            install_supervisor_connected_marker(&mut orchestrator, Arc::clone(supervisor));
        }

        // set_text_change_observer creates a new formatter and returns it.
        // We clone the Arc first, then install the real callback via set_on_change,
        // so the closure holds the same formatter instance that receives results.
        let formatter_ref = orchestrator.set_text_change_observer(Box::new(|| {}));
        let formatter_for_cb = Arc::clone(&formatter_ref);
        formatter_ref.set_on_change(Box::new(move || {
            let utterances = formatter_for_cb.get_utterances();
            let pre = formatter_for_cb.get_preliminary_text();
            if !pre.is_empty() {
                print!("\r[partial] {pre}   ");
                let _ = std::io::stdout().flush();
            } else if let Some(last) = utterances.last() {
                println!("\n[final]   {}", last.text);
            }
        }));

        let orchestrator = Arc::new(tokio::sync::Mutex::new(orchestrator));
        if let Err(e) = connect_and_start_audio(&orchestrator, owned_supervisor.clone()).await {
            tracing::error!("{e}");
            return false;
        }
        info!("Connected to server and audio capture started");

        info!("Headless mode running - press Ctrl+C to stop");
        if let Some(supervisor) = owned_supervisor {
            loop {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        info!("Shutting down");
                        orchestrator.lock().await.stop(false).await;
                        supervisor.shutdown().await;
                        return true;
                    }
                    event = supervisor.next_event() => {
                        match event {
                            Some(ServerProcessEvent::Exited { expected: true, .. }) => return true,
                            Some(ServerProcessEvent::Exited {
                                expected: false,
                                status,
                                diagnostics,
                            }) => {
                                tracing::error!("Owned Python server exited unexpectedly: {status:?}");
                                match supervisor.record_unstable_failure().await {
                                    CrashDecision::Restart => {
                                        if let Err(err) = reconnect_owned_server_once(
                                            Arc::clone(&supervisor),
                                            Arc::clone(&orchestrator),
                                        ).await {
                                            let _ = supervisor.record_unstable_failure().await;
                                            let failure_count =
                                                supervisor.unstable_failure_count().await;
                                            tracing::error!(
                                                "{}\nRestart failed: {err}",
                                                format_terminal_runtime_crash_message(
                                                    failure_count,
                                                    status,
                                                    &diagnostics
                                                )
                                            );
                                            return false;
                                        }
                                    }
                                    CrashDecision::Terminal => {
                                        let failure_count =
                                            supervisor.unstable_failure_count().await;
                                        tracing::error!(
                                            "{}",
                                            format_terminal_runtime_crash_message(
                                                failure_count,
                                                status,
                                                &diagnostics
                                            )
                                        );
                                        return false;
                                    }
                                }
                            }
                            None => {
                                tracing::error!("supervisor event channel closed unexpectedly");
                                return false;
                            }
                        }
                    }
                }
            }
        }

        tokio::signal::ctrl_c().await.ok();
        info!("Shutting down");
        orchestrator.lock().await.stop(false).await;
        true
    })
}

/// Starts the Tauri shell for the STT desktop client.
///
/// Parses CLI arguments (tolerating unknown Tauri-injected flags),
/// initialises tracing, creates the shared `AppStateManager`,
/// wires up the `ClientOrchestrator`, and launches the Tauri event loop.
fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = CliArgs::try_parse();
    match &cli {
        Ok(args) => {
            info!(server_url = ?args.server_url, input_file = ?args.input_file, "CLI args parsed")
        }
        Err(e) => info!("CLI parse skipped (Tauri dev mode?): {e}"),
    }

    let state_manager = Arc::new(AppStateManager::new());
    let insertion_controller = Arc::new(InsertionController::new());

    let headless = cli.as_ref().map(|a| a.headless).unwrap_or(false);
    let server_url = cli.as_ref().ok().and_then(|a| a.server_url.clone());
    let input_file = cli.as_ref().ok().and_then(|a| a.input_file.clone());

    if headless {
        if !run_headless(server_url, input_file, state_manager) {
            std::process::exit(1);
        }
        return;
    }

    let (owned_supervisor, supervisor_bootstrap_error) = if server_url.is_none() {
        match create_owned_supervisor() {
            Ok(supervisor) => (Some(supervisor), None),
            Err(err) => {
                tracing::error!("{err}");
                (None, Some(err))
            }
        }
    } else {
        (None, None)
    };

    tauri::Builder::default()
        .manage(state_manager.clone())
        .manage(insertion_controller)
        .setup(move |app| {
            let handle = app.handle().clone();
            let emitter: Arc<dyn EventEmitter> = Arc::new(TauriEmitter { handle });

            let qe_subscriber =
                setup_quickentry(app, Arc::clone(&emitter), Arc::clone(&state_manager));

            let mut orchestrator_inner = ClientOrchestrator::new(
                server_url.unwrap_or_default(),
                input_file,
                Arc::clone(&state_manager),
            );
            orchestrator_inner.set_emitter(Arc::clone(&emitter));
            if let Some(supervisor) = owned_supervisor.as_ref() {
                install_supervisor_connected_marker(
                    &mut orchestrator_inner,
                    Arc::clone(supervisor),
                );
            }

            let formatter_ref = orchestrator_inner.set_text_change_observer(Box::new(|| {}));
            let formatter_for_cb = Arc::clone(&formatter_ref);
            let emitter_for_cb = Arc::clone(&emitter);
            formatter_ref.set_on_change(Box::new(move || {
                emit_event(
                    emitter_for_cb.as_ref(),
                    "stt://transcript-update",
                    &TranscriptUpdate {
                        action: "update".to_string(),
                        utterances: formatter_for_cb.get_utterances(),
                        preliminary: formatter_for_cb.get_preliminary_text(),
                    },
                );
            }));

            let emitter_for_state = Arc::clone(&emitter);
            state_manager.add_observer(Box::new(move |old, new| {
                emit_event(
                    emitter_for_state.as_ref(),
                    "stt://state-changed",
                    &StateChanged {
                        old: old.to_string(),
                        new: new.to_string(),
                    },
                );
            }));

            if let Some(sub) = qe_subscriber {
                orchestrator_inner
                    .add_recognition_subscriber(Box::new(QuickEntrySubscriberBridge(sub)));
            }

            let orchestrator = Arc::new(tokio::sync::Mutex::new(orchestrator_inner));
            app.manage(orchestrator.clone());

            if let Some(supervisor) = owned_supervisor.as_ref() {
                app.manage(Arc::clone(supervisor));
            }

            let startup_supervisor = owned_supervisor.clone();
            let monitor_supervisor = owned_supervisor.clone();
            let startup_emitter = Arc::clone(&emitter);
            tauri::async_runtime::spawn(async move {
                if let Some(err) = supervisor_bootstrap_error {
                    emit_event(
                        startup_emitter.as_ref(),
                        "stt://connection-status",
                        &ConnectionStatus {
                            connected: false,
                            error: Some(err),
                            server_state: None,
                        },
                    );
                    return;
                }

                if let Err(e) =
                    connect_and_start_audio(&orchestrator, startup_supervisor.clone()).await
                {
                    tracing::error!("{e}");
                    emit_event(
                        startup_emitter.as_ref(),
                        "stt://connection-status",
                        &ConnectionStatus {
                            connected: false,
                            error: Some(e),
                            server_state: None,
                        },
                    );
                    return;
                }

                if let Some(supervisor) = monitor_supervisor {
                    monitor_owned_server(supervisor, Arc::clone(&orchestrator)).await;
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_state,
            commands::get_connection_status,
            commands::pause,
            commands::resume,
            commands::clear,
            commands::list_models,
            commands::download_model,
            commands::toggle_insertion,
            commands::quickentry_submit,
            commands::quickentry_cancel,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let handle = window.app_handle().clone();
                tauri::async_runtime::spawn(async move {
                    let maybe_orch =
                        handle.try_state::<Arc<tokio::sync::Mutex<ClientOrchestrator>>>();
                    if let Some(state) = maybe_orch {
                        let orch: &Arc<tokio::sync::Mutex<ClientOrchestrator>> = &state;
                        let stop_fut = async { orch.lock().await.stop(false).await };
                        let _ =
                            tokio::time::timeout(std::time::Duration::from_secs(3), stop_fut).await;
                    }

                    if let Some(supervisor) = handle.try_state::<Arc<ServerSupervisor>>() {
                        supervisor.shutdown().await;
                    }

                    #[cfg(windows)]
                    {
                        use stt_tauri_client::quickentry::ProductionQuickEntryController;
                        if let Some(ctrl) =
                            handle.try_state::<Arc<ProductionQuickEntryController>>()
                        {
                            ctrl.stop_hotkey();
                        }
                    }

                    for w in handle.webview_windows().values() {
                        let _ = w.destroy();
                    }
                });
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
