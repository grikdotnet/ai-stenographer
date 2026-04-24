/// Text insertion controller and recognition subscriber.
///
/// `TextInserter` implements `RecognitionSubscriber` and types finalized
/// text into the focused window via a `KeyboardSimulator`.
/// `InsertionController` owns the enabled flag and provides toggle/set
/// operations with optional state-change callbacks.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::keyboard::KeyboardSimulator;
use crate::recognition::{RecognitionResult, RecognitionSubscriber};

/// Types finalized recognition text into the focused window.
///
/// Implements `RecognitionSubscriber`. Partial results are ignored.
/// Finalized results are forwarded to the keyboard simulator only
/// when the shared `enabled` flag is true.
pub struct TextInserter {
    keyboard: Arc<dyn KeyboardSimulator>,
    enabled: Arc<AtomicBool>,
}

impl TextInserter {
    /// Creates a new TextInserter.
    ///
    /// Args:
    ///   keyboard: The keyboard simulator to use for typing.
    ///   enabled: Shared flag controlling whether insertion is active.
    pub fn new(keyboard: Arc<dyn KeyboardSimulator>, enabled: Arc<AtomicBool>) -> Self {
        Self { keyboard, enabled }
    }
}

impl RecognitionSubscriber for TextInserter {
    fn on_partial(&self, _result: &RecognitionResult) {}

    fn on_finalization(&self, result: &RecognitionResult) {
        if !self.enabled.load(Ordering::SeqCst) {
            return;
        }
        if let Err(e) = self.keyboard.type_text(&result.text) {
            tracing::error!("keyboard insertion failed: {e}");
        }
    }
}

type StateCallback = Box<dyn Fn(bool) + Send + Sync + 'static>;

/// Controls the insertion enabled/disabled state.
///
/// Owns the shared `AtomicBool` flag that `TextInserter` reads.
/// Provides toggle, set, and query operations. An optional callback
/// is invoked on every state change (for emitting Tauri events).
pub struct InsertionController {
    enabled: Arc<AtomicBool>,
    callback: Option<StateCallback>,
}

impl InsertionController {
    /// Creates a controller with insertion initially disabled and no callback.
    pub fn new() -> Self {
        Self {
            enabled: Arc::new(AtomicBool::new(false)),
            callback: None,
        }
    }

    /// Creates a controller with a state-change callback.
    ///
    /// Args:
    ///   callback: Called with the new enabled state on every change.
    pub fn with_callback(callback: StateCallback) -> Self {
        Self {
            enabled: Arc::new(AtomicBool::new(false)),
            callback: Some(callback),
        }
    }

    /// Returns a clone of the shared enabled flag for use by `TextInserter`.
    pub fn enabled_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.enabled)
    }

    /// Returns whether insertion is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Flips the enabled state and returns the new value.
    pub fn toggle(&self) -> bool {
        let previous = self.enabled.fetch_xor(true, Ordering::SeqCst);
        let new_state = !previous;
        if let Some(ref cb) = self.callback {
            cb(new_state);
        }
        new_state
    }

    /// Sets the enabled state to the given value.
    pub fn set_enabled(&self, value: bool) {
        self.enabled.store(value, Ordering::SeqCst);
        if let Some(ref cb) = self.callback {
            cb(value);
        }
    }
}

impl Default for InsertionController {
    fn default() -> Self {
        Self::new()
    }
}
