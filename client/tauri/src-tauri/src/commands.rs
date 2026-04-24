/// Tauri IPC command handlers.
///
/// Each function is exposed to the frontend via `tauri::command`.
/// Commands return `Result<T, String>` because Tauri requires
/// string error types for IPC serialization.
use std::sync::Arc;
use tauri::State;
use tokio::sync::Mutex;

use crate::insertion::InsertionController;
use crate::orchestrator::{ClientOrchestrator, ConnectionStatus};
use crate::state::AppStateManager;

/// Returns the current application state as a display string.
#[tauri::command]
pub fn get_state(state_manager: State<'_, Arc<AppStateManager>>) -> String {
    state_manager.current_state().to_string()
}

/// Returns the latest connection status snapshot for startup recovery.
#[tauri::command]
pub async fn get_connection_status(
    orchestrator: State<'_, Arc<Mutex<ClientOrchestrator>>>,
) -> Result<ConnectionStatus, String> {
    Ok(orchestrator.lock().await.current_connection_status())
}

/// Stops audio capture and transitions to the Paused state.
#[tauri::command]
pub async fn pause(
    orchestrator: State<'_, Arc<Mutex<ClientOrchestrator>>>,
) -> Result<(), String> {
    orchestrator.lock().await.pause_audio().map_err(|e| e.to_string())
}

/// Restarts audio capture and transitions to the Running state.
#[tauri::command]
pub async fn resume(
    orchestrator: State<'_, Arc<Mutex<ClientOrchestrator>>>,
) -> Result<(), String> {
    orchestrator.lock().await.resume_audio().map_err(|e| e.to_string())
}

/// Clears the current text buffer by resetting the `TextFormatter` state.
#[tauri::command]
pub async fn clear(
    orchestrator: State<'_, Arc<Mutex<ClientOrchestrator>>>,
) -> Result<(), String> {
    orchestrator.lock().await.clear();
    Ok(())
}

/// Requests the current downloadable model list from the server.
#[tauri::command]
pub async fn list_models(
    orchestrator: State<'_, Arc<Mutex<ClientOrchestrator>>>,
) -> Result<(), String> {
    orchestrator
        .lock()
        .await
        .request_model_list()
        .map_err(|e| e.to_string())
}

/// Requests download of the named model from the server.
#[tauri::command]
pub async fn download_model(
    model_name: String,
    orchestrator: State<'_, Arc<Mutex<ClientOrchestrator>>>,
) -> Result<(), String> {
    orchestrator
        .lock()
        .await
        .request_model_download(&model_name)
        .map_err(|e| e.to_string())
}

/// Toggles keyboard insertion on/off and returns the new state.
#[tauri::command]
pub fn toggle_insertion(
    controller: State<'_, Arc<InsertionController>>,
) -> Result<bool, String> {
    Ok(controller.toggle())
}

/// Submits the quick-entry popup text into the previously-focused window.
#[cfg(windows)]
#[tauri::command]
pub fn quickentry_submit(
    controller: State<'_, Arc<crate::quickentry::ProductionQuickEntryController>>,
) -> Result<(), String> {
    controller.submit();
    Ok(())
}

/// Cancels the quick-entry popup without typing any text.
#[cfg(windows)]
#[tauri::command]
pub fn quickentry_cancel(
    controller: State<'_, Arc<crate::quickentry::ProductionQuickEntryController>>,
) -> Result<(), String> {
    controller.cancel();
    Ok(())
}

/// Submits the quick-entry popup (no-op on non-Windows).
#[cfg(not(windows))]
#[tauri::command]
pub fn quickentry_submit() -> Result<(), String> {
    Ok(())
}

/// Cancels the quick-entry popup (no-op on non-Windows).
#[cfg(not(windows))]
#[tauri::command]
pub fn quickentry_cancel() -> Result<(), String> {
    Ok(())
}
