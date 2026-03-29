/// Quick-entry popup system for dictating text into any application.
///
/// `QuickEntrySubscriber` accumulates finalized recognition text.
/// `QuickEntryController` coordinates focus save/restore, subscriber
/// lifecycle, keyboard insertion, and event emission for the popup UI.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;

use crate::focus::{FocusTracker, WindowFocusApi};
use crate::hotkey::HotkeyListener;
use crate::keyboard::KeyboardSimulator;
use crate::orchestrator::EventEmitter;
use crate::recognition::{RecognitionResult, RecognitionSubscriber};
use crate::state::AppState;

type OnTextChangeFn = Box<dyn Fn(String) + Send + Sync>;

/// Visibility event payload emitted to the Tauri frontend.
#[derive(Clone, Serialize, Debug)]
struct QuickEntryVisibility {
    visible: bool,
}

/// Text update event payload emitted to the Tauri frontend.
#[derive(Clone, Serialize, Debug)]
struct QuickEntryTextUpdate {
    text: String,
}

/// Abstraction over OS-level popup window show/hide.
///
/// Decouples `QuickEntryController` from Tauri's `WebviewWindow` type,
/// preserving testability and SOLID separation.
pub trait PopupWindow: Send + Sync {
    fn show(&self);
    fn hide(&self);
}

/// No-op popup window for testing and non-Windows platforms.
pub struct NoOpPopupWindow;

impl PopupWindow for NoOpPopupWindow {
    fn show(&self) {}
    fn hide(&self) {}
}

/// Accumulates finalized recognition text for the quick-entry popup.
///
/// Thread-safe: all state is behind a Mutex or AtomicBool. Partial results are
/// ignored; finalized results are concatenated with spaces only when `active`.
/// An optional callback (`on_text_change`) is invoked after each accepted
/// finalization with the full accumulated text, allowing the controller to emit
/// frontend events.
pub struct QuickEntrySubscriber {
    text: Mutex<String>,
    on_text_change: Mutex<Option<OnTextChangeFn>>,
    active: AtomicBool,
}

impl QuickEntrySubscriber {
    pub fn new() -> Self {
        Self {
            text: Mutex::new(String::new()),
            on_text_change: Mutex::new(None),
            active: AtomicBool::new(false),
        }
    }

    /// Enables or disables recognition accumulation.
    ///
    /// Args:
    ///   active: `true` to start accepting finalizations; `false` to ignore them.
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::SeqCst);
    }

    /// Returns the accumulated finalized text.
    pub fn get_text(&self) -> String {
        self.text.lock().unwrap().clone()
    }

    /// Clears the accumulated text buffer.
    pub fn clear(&self) {
        self.text.lock().unwrap().clear();
    }

    /// Registers a callback invoked with the full accumulated text after each finalization.
    ///
    /// Args:
    ///   cb: Called with the current accumulated text string after every finalized result.
    pub fn set_on_text_change(&self, cb: OnTextChangeFn) {
        *self.on_text_change.lock().unwrap() = Some(cb);
    }
}

impl Default for QuickEntrySubscriber {
    fn default() -> Self {
        Self::new()
    }
}

impl RecognitionSubscriber for QuickEntrySubscriber {
    fn on_partial(&self, _result: &RecognitionResult) {}

    fn on_finalization(&self, result: &RecognitionResult) {
        if !self.active.load(Ordering::SeqCst) {
            return;
        }
        if result.text.is_empty() {
            return;
        }
        let accumulated = {
            let mut buf = self.text.lock().unwrap();
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(&result.text);
            buf.clone()
        };
        if let Some(cb) = self.on_text_change.lock().unwrap().as_ref() {
            cb(accumulated);
        }
    }
}

/// Emits a serialized event through the EventEmitter.
fn emit_event(emitter: &dyn EventEmitter, event: &str, payload: &impl Serialize) {
    if let Ok(json) = serde_json::to_string(payload) {
        emitter.emit_serialized(event, &json);
    }
}

/// Coordinates the quick-entry popup lifecycle.
///
/// Responsibilities:
/// - Saves/restores window focus around the popup.
/// - Shows/hides the OS-level popup window.
/// - Activates/deactivates the subscriber.
/// - Emits visibility events to the frontend.
/// - Registers/unregisters popup hotkeys (Enter, Escape) on each show/hide cycle.
/// - On submit: types accumulated text, restores focus, clears subscriber.
/// - On cancel: restores focus, clears subscriber without typing.
pub struct QuickEntryController<F: WindowFocusApi> {
    focus_tracker: Arc<FocusTracker<F>>,
    keyboard: Arc<dyn KeyboardSimulator>,
    subscriber: Arc<QuickEntrySubscriber>,
    emitter: Arc<dyn EventEmitter>,
    hotkey: Arc<dyn HotkeyListener>,
    popup_window: Arc<dyn PopupWindow>,
    submit_fn: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    cancel_fn: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    visible: AtomicBool,
}

impl<F: WindowFocusApi> QuickEntryController<F> {
    /// Creates a new QuickEntryController.
    ///
    /// Args:
    ///   focus_tracker: Saves/restores the previously-focused window.
    ///   keyboard: Types text into the focused window.
    ///   subscriber: Accumulates recognition text for the popup.
    ///   emitter: Emits events to the Tauri frontend.
    ///   hotkey: Global hotkey listener for Enter/Escape registration.
    ///   popup_window: OS-level window show/hide abstraction.
    pub fn new(
        focus_tracker: Arc<FocusTracker<F>>,
        keyboard: Arc<dyn KeyboardSimulator>,
        subscriber: Arc<QuickEntrySubscriber>,
        emitter: Arc<dyn EventEmitter>,
        hotkey: Arc<dyn HotkeyListener>,
        popup_window: Arc<dyn PopupWindow>,
    ) -> Self {
        let emitter_for_cb = Arc::clone(&emitter);
        subscriber.set_on_text_change(Box::new(move |text| {
            emit_event(
                emitter_for_cb.as_ref(),
                "stt://quickentry-text",
                &QuickEntryTextUpdate { text },
            );
        }));

        Self {
            focus_tracker,
            keyboard,
            subscriber,
            emitter,
            hotkey,
            popup_window,
            submit_fn: Mutex::new(None),
            cancel_fn: Mutex::new(None),
            visible: AtomicBool::new(false),
        }
    }

    /// Stores the submit and cancel callbacks used when registering popup hotkeys.
    ///
    /// Must be called after construction and before the first `show_popup()`.
    /// These callbacks are re-registered each time the popup opens.
    ///
    /// Args:
    ///   on_submit: Called when the user presses Enter in the popup.
    ///   on_cancel: Called when the user presses Escape in the popup.
    pub fn set_popup_callbacks(
        &self,
        on_submit: Arc<dyn Fn() + Send + Sync>,
        on_cancel: Arc<dyn Fn() + Send + Sync>,
    ) {
        *self.submit_fn.lock().unwrap() = Some(on_submit);
        *self.cancel_fn.lock().unwrap() = Some(on_cancel);
    }

    /// Returns whether the quick-entry popup is currently visible.
    pub fn is_visible(&self) -> bool {
        self.visible.load(Ordering::SeqCst)
    }

    /// Shows the quick-entry popup.
    ///
    /// Algorithm:
    /// 1. Activate subscriber (start accepting finalizations).
    /// 2. Save current foreground window focus.
    /// 3. Set visible flag.
    /// 4. Show the OS-level popup window.
    /// 5. Emit visibility true event.
    /// 6. Register popup hotkeys (Enter, Escape) with stored callbacks.
    pub fn show_popup(&self) {
        self.subscriber.set_active(true);
        self.focus_tracker.save();
        self.visible.store(true, Ordering::SeqCst);
        self.popup_window.show();

        emit_event(
            self.emitter.as_ref(),
            "stt://quickentry-visibility",
            &QuickEntryVisibility { visible: true },
        );

        let submit = self.submit_fn.lock().unwrap().clone();
        let cancel = self.cancel_fn.lock().unwrap().clone();
        if let (Some(s), Some(c)) = (submit, cancel) {
            self.hotkey.register_popup_hotkeys(
                Box::new(move || s()),
                Box::new(move || c()),
            );
        }
    }

    /// Submits the accumulated text: types it, hides the popup, restores focus.
    ///
    /// Algorithm:
    /// 1. Get accumulated text from subscriber.
    /// 2. Set visible to false.
    /// 3. Emit visibility false.
    /// 4. Unregister popup hotkeys.
    /// 5. Hide the OS-level popup window.
    /// 6. Restore focus to the original window.
    /// 7. Deactivate subscriber (stop accepting new finalizations).
    /// 8. Type the text via keyboard simulator.
    /// 9. Clear the subscriber.
    pub fn submit(&self) {
        let text = self.subscriber.get_text();

        self.visible.store(false, Ordering::SeqCst);
        emit_event(
            self.emitter.as_ref(),
            "stt://quickentry-visibility",
            &QuickEntryVisibility { visible: false },
        );
        self.hotkey.unregister_popup_hotkeys();
        self.popup_window.hide();
        self.focus_tracker.restore();

        self.subscriber.set_active(false);
        if !text.is_empty() {
            if let Err(e) = self.keyboard.type_text(&text) {
                tracing::error!("QuickEntry keyboard insertion failed: {e}");
            }
        }

        self.subscriber.clear();
    }

    /// Cancels the popup: hides it and restores focus without typing.
    ///
    /// Algorithm:
    /// 1. Set visible to false.
    /// 2. Emit visibility false.
    /// 3. Unregister popup hotkeys.
    /// 4. Hide the OS-level popup window.
    /// 5. Restore focus.
    /// 6. Deactivate subscriber and clear buffer.
    pub fn cancel(&self) {
        self.visible.store(false, Ordering::SeqCst);
        emit_event(
            self.emitter.as_ref(),
            "stt://quickentry-visibility",
            &QuickEntryVisibility { visible: false },
        );
        self.hotkey.unregister_popup_hotkeys();
        self.popup_window.hide();
        self.focus_tracker.restore();
        self.subscriber.set_active(false);
        self.subscriber.clear();
    }

    /// Stops the global hotkey listener, terminating the OS message-loop thread.
    ///
    /// Args: none
    /// Returns: nothing
    pub fn stop_hotkey(&self) {
        self.hotkey.stop();
    }
}

/// Handles the Ctrl+Space toggle hotkey for the quick-entry popup.
///
/// Rules:
/// - If the popup is visible: cancel it, regardless of app state.
/// - If the popup is not visible and state is `Running`: show it.
/// - Otherwise (Paused, Starting, Shutdown): do nothing — audio is inactive.
///
/// Args:
///   controller: The controller to show or cancel.
///   state: Current AppState from `AppStateManager::current_state()`.
pub fn handle_toggle<F: WindowFocusApi>(controller: &QuickEntryController<F>, state: AppState) {
    if controller.is_visible() {
        controller.cancel();
    } else if state == AppState::Running {
        controller.show_popup();
    }
}

/// Concrete controller type for production use on Windows.
#[cfg(windows)]
pub type ProductionQuickEntryController = QuickEntryController<crate::focus::Win32FocusApi>;
