/// Application lifecycle state machine.
///
/// `AppState` represents the four possible lifecycle phases.
/// `AppStateManager` enforces a fixed transition table, provides
/// thread-safe mutation via `Mutex`, and notifies registered observers
/// **outside** the lock to avoid deadlocks (copy-on-notify pattern).
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::error::AppError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AppState {
    Starting,
    Running,
    Paused,
    Shutdown,
}

impl fmt::Display for AppState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            AppState::Starting => "Starting",
            AppState::Running => "Running",
            AppState::Paused => "Paused",
            AppState::Shutdown => "Shutdown",
        };
        write!(f, "{}", label)
    }
}

type ObserverFn = Arc<dyn Fn(AppState, AppState) + Send + Sync + 'static>;

struct Inner {
    state: AppState,
    observers: Vec<ObserverFn>,
}

/// Thread-safe state manager with observer support.
///
/// Responsibilities:
/// - Validates transitions against a fixed allowlist.
/// - Holds current state behind a `Mutex`.
/// - Notifies observers after a successful transition, outside the lock.
pub struct AppStateManager {
    inner: Mutex<Inner>,
}

impl AppStateManager {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                state: AppState::Starting,
                observers: Vec::new(),
            }),
        }
    }

    /// Returns the current application state.
    pub fn current_state(&self) -> AppState {
        self.inner.lock().unwrap().state
    }

    /// Attempts a state transition.
    ///
    /// Returns `Ok(())` on success or if `Shutdown->Shutdown` (silent no-op).
    /// Returns `Err(AppError::InvalidTransition)` for disallowed transitions.
    ///
    /// Algorithm:
    /// 1. Acquire lock, read current state.
    /// 2. If Shutdown->Shutdown, return Ok immediately without notifying.
    /// 3. Validate transition against the allowlist.
    /// 4. Update state, clone observer Arcs, release lock.
    /// 5. Notify each observer outside the lock.
    pub fn set_state(&self, new: AppState) -> Result<(), AppError> {
        let (old, observers) = {
            let mut guard = self.inner.lock().unwrap();
            let old = guard.state;

            if old == AppState::Shutdown && new == AppState::Shutdown {
                return Ok(());
            }

            if !Self::is_valid_transition(old, new) {
                return Err(AppError::InvalidTransition {
                    from: old.to_string(),
                    to: new.to_string(),
                });
            }

            guard.state = new;
            let snapshot: Vec<ObserverFn> = guard.observers.clone();
            (old, snapshot)
        };

        for obs in &observers {
            obs(old, new);
        }

        Ok(())
    }

    /// Registers an observer that will be called on every successful state transition.
    pub fn add_observer(&self, callback: Box<dyn Fn(AppState, AppState) + Send + Sync + 'static>) {
        self.inner
            .lock()
            .unwrap()
            .observers
            .push(Arc::from(callback));
    }

    fn is_valid_transition(from: AppState, to: AppState) -> bool {
        matches!(
            (from, to),
            (AppState::Starting, AppState::Running)
                | (AppState::Starting, AppState::Shutdown)
                | (AppState::Running, AppState::Paused)
                | (AppState::Running, AppState::Shutdown)
                | (AppState::Paused, AppState::Running)
                | (AppState::Paused, AppState::Shutdown)
        )
    }
}

impl Default for AppStateManager {
    fn default() -> Self {
        Self::new()
    }
}
