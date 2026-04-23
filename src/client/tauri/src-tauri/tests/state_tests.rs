use stt_tauri_client::error::AppError;
use stt_tauri_client::state::{AppState, AppStateManager};
use std::sync::{Arc, Mutex};

#[test]
fn starting_to_waiting_for_server_succeeds() {
    let mgr = AppStateManager::new();
    assert!(mgr.set_state(AppState::WaitingForServer).is_ok());
    assert_eq!(mgr.current_state(), AppState::WaitingForServer);
}

#[test]
fn waiting_for_server_to_running_succeeds() {
    let mgr = AppStateManager::new();
    mgr.set_state(AppState::WaitingForServer).unwrap();
    assert!(mgr.set_state(AppState::Running).is_ok());
    assert_eq!(mgr.current_state(), AppState::Running);
}

#[test]
fn paused_to_running_succeeds() {
    let mgr = AppStateManager::new();
    mgr.set_state(AppState::WaitingForServer).unwrap();
    mgr.set_state(AppState::Running).unwrap();
    mgr.set_state(AppState::Paused).unwrap();
    assert!(mgr.set_state(AppState::Running).is_ok());
    assert_eq!(mgr.current_state(), AppState::Running);
}

#[test]
fn running_to_shutdown_succeeds() {
    let mgr = AppStateManager::new();
    mgr.set_state(AppState::WaitingForServer).unwrap();
    mgr.set_state(AppState::Running).unwrap();
    assert!(mgr.set_state(AppState::Shutdown).is_ok());
    assert_eq!(mgr.current_state(), AppState::Shutdown);
}

#[test]
fn paused_to_shutdown_succeeds() {
    let mgr = AppStateManager::new();
    mgr.set_state(AppState::WaitingForServer).unwrap();
    mgr.set_state(AppState::Running).unwrap();
    mgr.set_state(AppState::Paused).unwrap();
    assert!(mgr.set_state(AppState::Shutdown).is_ok());
    assert_eq!(mgr.current_state(), AppState::Shutdown);
}

#[test]
fn starting_to_shutdown_succeeds() {
    let mgr = AppStateManager::new();
    assert!(mgr.set_state(AppState::Shutdown).is_ok());
    assert_eq!(mgr.current_state(), AppState::Shutdown);
}

#[test]
fn shutdown_to_shutdown_is_silent_noop() {
    let mgr = AppStateManager::new();
    mgr.set_state(AppState::Shutdown).unwrap();
    assert!(mgr.set_state(AppState::Shutdown).is_ok());
    assert_eq!(mgr.current_state(), AppState::Shutdown);
}

#[test]
fn starting_to_paused_is_invalid() {
    let mgr = AppStateManager::new();
    let result = mgr.set_state(AppState::Paused);
    assert!(result.is_err());
    match result.unwrap_err() {
        AppError::InvalidTransition { from, to } => {
            assert_eq!(from, "Starting");
            assert_eq!(to, "Paused");
        }
        other => panic!("expected InvalidTransition, got {:?}", other),
    }
}

#[test]
fn running_to_starting_is_invalid() {
    let mgr = AppStateManager::new();
    mgr.set_state(AppState::WaitingForServer).unwrap();
    mgr.set_state(AppState::Running).unwrap();
    assert!(mgr.set_state(AppState::Starting).is_err());
}

#[test]
fn shutdown_to_running_is_invalid() {
    let mgr = AppStateManager::new();
    mgr.set_state(AppState::Shutdown).unwrap();
    assert!(mgr.set_state(AppState::Running).is_err());
}

#[test]
fn shutdown_to_paused_is_invalid() {
    let mgr = AppStateManager::new();
    mgr.set_state(AppState::Shutdown).unwrap();
    assert!(mgr.set_state(AppState::Paused).is_err());
}

#[test]
fn shutdown_to_starting_is_invalid() {
    let mgr = AppStateManager::new();
    mgr.set_state(AppState::Shutdown).unwrap();
    assert!(mgr.set_state(AppState::Starting).is_err());
}

#[test]
fn observer_notified_on_state_change() {
    let mgr = AppStateManager::new();
    let log: Arc<Mutex<Vec<(AppState, AppState)>>> = Arc::new(Mutex::new(Vec::new()));
    let log_clone = Arc::clone(&log);

    mgr.add_observer(Box::new(move |old, new| {
        log_clone.lock().unwrap().push((old, new));
    }));

    mgr.set_state(AppState::WaitingForServer).unwrap();

    let entries = log.lock().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0], (AppState::Starting, AppState::WaitingForServer));
}

#[test]
fn observer_not_called_on_failed_transition() {
    let mgr = AppStateManager::new();
    let call_count = Arc::new(Mutex::new(0u32));
    let count_clone = Arc::clone(&call_count);

    mgr.add_observer(Box::new(move |_, _| {
        *count_clone.lock().unwrap() += 1;
    }));

    let _ = mgr.set_state(AppState::Paused);

    assert_eq!(*call_count.lock().unwrap(), 0);
}

#[test]
fn multiple_observers_all_notified() {
    let mgr = AppStateManager::new();
    let count_a = Arc::new(Mutex::new(0u32));
    let count_b = Arc::new(Mutex::new(0u32));
    let a = Arc::clone(&count_a);
    let b = Arc::clone(&count_b);

    mgr.add_observer(Box::new(move |_, _| {
        *a.lock().unwrap() += 1;
    }));
    mgr.add_observer(Box::new(move |_, _| {
        *b.lock().unwrap() += 1;
    }));

    mgr.set_state(AppState::WaitingForServer).unwrap();

    assert_eq!(*count_a.lock().unwrap(), 1);
    assert_eq!(*count_b.lock().unwrap(), 1);
}

#[test]
fn current_state_returns_initial_state() {
    let mgr = AppStateManager::new();
    assert_eq!(mgr.current_state(), AppState::Starting);
}

#[test]
fn shutdown_to_shutdown_does_not_notify_observers() {
    let mgr = AppStateManager::new();
    mgr.set_state(AppState::Shutdown).unwrap();

    let call_count = Arc::new(Mutex::new(0u32));
    let count_clone = Arc::clone(&call_count);
    mgr.add_observer(Box::new(move |_, _| {
        *count_clone.lock().unwrap() += 1;
    }));

    mgr.set_state(AppState::Shutdown).unwrap();
    assert_eq!(*call_count.lock().unwrap(), 0);
}

#[test]
fn app_state_display_trait() {
    assert_eq!(format!("{}", AppState::Starting), "Starting");
    assert_eq!(format!("{}", AppState::WaitingForServer), "WaitingForServer");
    assert_eq!(format!("{}", AppState::Running), "Running");
    assert_eq!(format!("{}", AppState::Paused), "Paused");
    assert_eq!(format!("{}", AppState::Shutdown), "Shutdown");
}
