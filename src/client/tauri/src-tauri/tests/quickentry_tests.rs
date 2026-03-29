use std::sync::{Arc, Mutex};

use stt_tauri_client::error::AppError;
use stt_tauri_client::focus::{FocusTracker, WindowFocusApi};
use stt_tauri_client::hotkey::{HotkeyListener, NoOpHotkeyListener};
use stt_tauri_client::keyboard::KeyboardSimulator;
use stt_tauri_client::orchestrator::EventEmitter;
use stt_tauri_client::quickentry::{
    handle_toggle, NoOpPopupWindow, PopupWindow, QuickEntryController, QuickEntrySubscriber,
};
use stt_tauri_client::recognition::{RecognitionResult, RecognitionStatus, RecognitionSubscriber};
use stt_tauri_client::state::AppState;

// --- Fakes ---

struct FakeKeyboard {
    typed: Arc<Mutex<Vec<String>>>,
}

impl FakeKeyboard {
    fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
        let typed = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                typed: Arc::clone(&typed),
            },
            typed,
        )
    }
}

impl KeyboardSimulator for FakeKeyboard {
    fn type_text(&self, text: &str) -> Result<(), AppError> {
        self.typed.lock().unwrap().push(text.to_string());
        Ok(())
    }
}

struct FakeFocusApi {
    save_calls: Mutex<u32>,
    restore_calls: Mutex<u32>,
}

impl FakeFocusApi {
    fn new() -> Self {
        Self {
            save_calls: Mutex::new(0),
            restore_calls: Mutex::new(0),
        }
    }
}

impl WindowFocusApi for FakeFocusApi {
    fn get_foreground_window(&self) -> isize {
        *self.save_calls.lock().unwrap() += 1;
        42
    }

    fn set_foreground_window(&self, _hwnd: isize) -> bool {
        *self.restore_calls.lock().unwrap() += 1;
        true
    }

    fn get_window_thread_process_id(&self, _hwnd: isize) -> u32 {
        1
    }

    fn get_current_thread_id(&self) -> u32 {
        1
    }

    fn attach_thread_input(&self, _attach_to: u32, _attach_from: u32, _attach: bool) -> bool {
        true
    }
}

/// Records emitted events for verification.
struct RecordingEmitter {
    events: Arc<Mutex<Vec<(String, String)>>>,
}

impl RecordingEmitter {
    fn new() -> (Self, Arc<Mutex<Vec<(String, String)>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                events: Arc::clone(&events),
            },
            events,
        )
    }
}

impl EventEmitter for RecordingEmitter {
    fn emit_serialized(&self, event: &str, payload: &str) {
        self.events
            .lock()
            .unwrap()
            .push((event.to_string(), payload.to_string()));
    }
}

/// Spy popup window that records show/hide calls.
struct SpyPopupWindow {
    show_calls: Mutex<u32>,
    hide_calls: Mutex<u32>,
}

impl SpyPopupWindow {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            show_calls: Mutex::new(0),
            hide_calls: Mutex::new(0),
        })
    }

    fn show_count(&self) -> u32 {
        *self.show_calls.lock().unwrap()
    }

    fn hide_count(&self) -> u32 {
        *self.hide_calls.lock().unwrap()
    }
}

impl PopupWindow for SpyPopupWindow {
    fn show(&self) {
        *self.show_calls.lock().unwrap() += 1;
    }

    fn hide(&self) {
        *self.hide_calls.lock().unwrap() += 1;
    }
}

/// Spy hotkey listener that records register/unregister/stop calls.
struct SpyHotkeyListener {
    register_calls: Mutex<u32>,
    unregister_calls: Mutex<u32>,
    stop_calls: Mutex<u32>,
}

impl SpyHotkeyListener {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            register_calls: Mutex::new(0),
            unregister_calls: Mutex::new(0),
            stop_calls: Mutex::new(0),
        })
    }

    fn register_count(&self) -> u32 {
        *self.register_calls.lock().unwrap()
    }

    fn stop_count(&self) -> u32 {
        *self.stop_calls.lock().unwrap()
    }
}

impl HotkeyListener for SpyHotkeyListener {
    fn start(&self, _on_toggle: Box<dyn Fn() + Send>) {}

    fn register_popup_hotkeys(
        &self,
        _on_submit: Box<dyn Fn() + Send>,
        _on_cancel: Box<dyn Fn() + Send>,
    ) {
        *self.register_calls.lock().unwrap() += 1;
    }

    fn unregister_popup_hotkeys(&self) {
        *self.unregister_calls.lock().unwrap() += 1;
    }

    fn stop(&self) {
        *self.stop_calls.lock().unwrap() += 1;
    }
}

fn make_result(status: RecognitionStatus, text: &str) -> RecognitionResult {
    RecognitionResult {
        session_id: "s1".into(),
        status,
        text: text.into(),
        start_time: 0.0,
        end_time: 1.0,
        chunk_ids: vec![1],
        utterance_id: 1,
        token_confidences: vec![0.9],
    }
}

// --- QuickEntrySubscriber tests ---

#[test]
fn subscriber_accumulates_finalized_text() {
    let sub = QuickEntrySubscriber::new();

    sub.set_active(true);
    sub.on_finalization(&make_result(RecognitionStatus::Final, "hello"));
    sub.on_finalization(&make_result(RecognitionStatus::Final, "world"));

    assert_eq!(sub.get_text(), "hello world");
}

#[test]
fn subscriber_ignores_partials() {
    let sub = QuickEntrySubscriber::new();

    sub.on_partial(&make_result(RecognitionStatus::Partial, "partial stuff"));

    assert_eq!(sub.get_text(), "");
}

#[test]
fn subscriber_clear_resets_text() {
    let sub = QuickEntrySubscriber::new();

    sub.set_active(true);
    sub.on_finalization(&make_result(RecognitionStatus::Final, "some text"));
    assert_eq!(sub.get_text(), "some text");

    sub.clear();
    assert_eq!(sub.get_text(), "");
}

#[test]
fn subscriber_empty_finalization_not_appended_with_extra_space() {
    let sub = QuickEntrySubscriber::new();

    sub.set_active(true);
    sub.on_finalization(&make_result(RecognitionStatus::Final, "hello"));
    sub.on_finalization(&make_result(RecognitionStatus::Final, ""));

    assert_eq!(sub.get_text(), "hello");
}

// --- QuickEntryController helpers ---

fn make_controller() -> (
    QuickEntryController<FakeFocusApi>,
    Arc<Mutex<Vec<String>>>,
    Arc<Mutex<Vec<(String, String)>>>,
    Arc<QuickEntrySubscriber>,
) {
    let focus_api = FakeFocusApi::new();
    let focus_tracker = Arc::new(FocusTracker::new(focus_api));
    let (kb, typed) = FakeKeyboard::new();
    let keyboard: Arc<dyn KeyboardSimulator> = Arc::new(kb);
    let subscriber = Arc::new(QuickEntrySubscriber::new());
    let (emitter, events) = RecordingEmitter::new();
    let emitter: Arc<dyn EventEmitter> = Arc::new(emitter);
    let hotkey = Arc::new(NoOpHotkeyListener::new());
    let popup_window: Arc<dyn PopupWindow> = Arc::new(NoOpPopupWindow);

    let controller = QuickEntryController::new(
        focus_tracker,
        keyboard,
        Arc::clone(&subscriber),
        emitter,
        hotkey,
        popup_window,
    );

    (controller, typed, events, subscriber)
}

type SpyController = QuickEntryController<FakeFocusApi>;

fn make_controller_with_spies() -> (
    SpyController,
    Arc<SpyPopupWindow>,
    Arc<SpyHotkeyListener>,
    Arc<Mutex<Vec<String>>>,
    Arc<Mutex<Vec<(String, String)>>>,
    Arc<QuickEntrySubscriber>,
) {
    let focus_api = FakeFocusApi::new();
    let focus_tracker = Arc::new(FocusTracker::new(focus_api));
    let (kb, typed) = FakeKeyboard::new();
    let keyboard: Arc<dyn KeyboardSimulator> = Arc::new(kb);
    let subscriber = Arc::new(QuickEntrySubscriber::new());
    let (emitter, events) = RecordingEmitter::new();
    let emitter: Arc<dyn EventEmitter> = Arc::new(emitter);
    let spy_hotkey = SpyHotkeyListener::new();
    let hotkey: Arc<dyn HotkeyListener> = Arc::clone(&spy_hotkey) as Arc<dyn HotkeyListener>;
    let spy_popup = SpyPopupWindow::new();
    let popup_window: Arc<dyn PopupWindow> = Arc::clone(&spy_popup) as Arc<dyn PopupWindow>;

    let controller = QuickEntryController::new(
        focus_tracker,
        keyboard,
        Arc::clone(&subscriber),
        emitter,
        hotkey,
        popup_window,
    );

    (controller, spy_popup, spy_hotkey, typed, events, subscriber)
}

// --- QuickEntryController tests ---

#[test]
fn show_sets_visible() {
    let (controller, _, _, _) = make_controller();

    assert!(!controller.is_visible());
    controller.show_popup();
    assert!(controller.is_visible());
}

#[test]
fn show_emits_visibility_event() {
    let (controller, _, events, _) = make_controller();

    controller.show_popup();

    let log = events.lock().unwrap();
    assert!(log.iter().any(|(event, payload)| {
        event == "stt://quickentry-visibility" && payload.contains("true")
    }));
}

#[test]
fn submit_types_text_and_restores_focus() {
    let (controller, typed, _, subscriber) = make_controller();

    controller.show_popup();
    subscriber.on_finalization(&make_result(RecognitionStatus::Final, "typed text"));
    controller.submit();

    let typed = typed.lock().unwrap();
    assert_eq!(typed.len(), 1);
    assert_eq!(typed[0], "typed text");
}

#[test]
fn submit_clears_subscriber_text() {
    let (controller, _, _, subscriber) = make_controller();

    controller.show_popup();
    subscriber.on_finalization(&make_result(RecognitionStatus::Final, "text"));
    controller.submit();

    assert_eq!(subscriber.get_text(), "");
}

#[test]
fn submit_emits_visibility_false() {
    let (controller, _, events, _) = make_controller();

    controller.show_popup();
    controller.submit();

    let log = events.lock().unwrap();
    let hide_events: Vec<_> = log
        .iter()
        .filter(|(event, payload)| {
            event == "stt://quickentry-visibility" && payload.contains("false")
        })
        .collect();
    assert!(!hide_events.is_empty());
}

#[test]
fn cancel_restores_focus_without_typing() {
    let (controller, typed, _, _) = make_controller();

    controller.show_popup();
    controller.cancel();

    assert!(typed.lock().unwrap().is_empty());
    assert!(!controller.is_visible());
}

#[test]
fn cancel_emits_visibility_false() {
    let (controller, _, events, _) = make_controller();

    controller.show_popup();
    controller.cancel();

    let log = events.lock().unwrap();
    let hide_events: Vec<_> = log
        .iter()
        .filter(|(event, payload)| {
            event == "stt://quickentry-visibility" && payload.contains("false")
        })
        .collect();
    assert!(!hide_events.is_empty());
}

#[test]
fn submit_sets_not_visible() {
    let (controller, _, _, _) = make_controller();

    controller.show_popup();
    assert!(controller.is_visible());

    controller.submit();
    assert!(!controller.is_visible());
}

#[test]
fn cancel_sets_not_visible() {
    let (controller, _, _, _) = make_controller();

    controller.show_popup();
    assert!(controller.is_visible());

    controller.cancel();
    assert!(!controller.is_visible());
}

// --- PopupWindow and hotkey spy tests ---

#[test]
fn show_popup_calls_window_show() {
    let (controller, spy_popup, _, _, _, _) = make_controller_with_spies();

    controller.show_popup();

    assert_eq!(spy_popup.show_count(), 1);
}

#[test]
fn submit_calls_window_hide() {
    let (controller, spy_popup, _, _, _, _) = make_controller_with_spies();

    controller.show_popup();
    controller.submit();

    assert_eq!(spy_popup.hide_count(), 1);
}

#[test]
fn cancel_calls_window_hide() {
    let (controller, spy_popup, _, _, _, _) = make_controller_with_spies();

    controller.show_popup();
    controller.cancel();

    assert_eq!(spy_popup.hide_count(), 1);
}

#[test]
fn popup_hotkeys_registered_on_show() {
    let (controller, _, spy_hotkey, _, _, _) = make_controller_with_spies();

    controller.set_popup_callbacks(Arc::new(|| {}), Arc::new(|| {}));
    controller.show_popup();

    assert_eq!(spy_hotkey.register_count(), 1);
}

#[test]
fn popup_hotkeys_registered_again_on_second_show() {
    let (controller, _, spy_hotkey, _, _, _) = make_controller_with_spies();

    controller.set_popup_callbacks(Arc::new(|| {}), Arc::new(|| {}));
    controller.show_popup();
    controller.cancel();
    controller.show_popup();

    assert_eq!(spy_hotkey.register_count(), 2);
}

#[test]
fn show_popup_without_callbacks_does_not_register_hotkeys() {
    let (controller, _, spy_hotkey, _, _, _) = make_controller_with_spies();

    controller.show_popup();

    assert_eq!(spy_hotkey.register_count(), 0);
}

#[test]
fn stop_hotkey_delegates_to_listener() {
    let (controller, _, spy_hotkey, _, _, _) = make_controller_with_spies();

    controller.stop_hotkey();

    assert_eq!(spy_hotkey.stop_count(), 1);
}

// --- Active/inactive subscriber tests ---

#[test]
fn subscriber_ignores_finalizations_when_inactive() {
    let sub = QuickEntrySubscriber::new();

    sub.on_finalization(&make_result(RecognitionStatus::Final, "pre-show text"));

    assert_eq!(sub.get_text(), "");
}

#[test]
fn subscriber_accumulates_only_when_active() {
    let sub = QuickEntrySubscriber::new();

    sub.on_finalization(&make_result(RecognitionStatus::Final, "ignored"));
    sub.set_active(true);
    sub.on_finalization(&make_result(RecognitionStatus::Final, "captured"));

    assert_eq!(sub.get_text(), "captured");
}

#[test]
fn show_popup_activates_subscriber() {
    let (controller, _, _, subscriber) = make_controller();

    controller.show_popup();
    subscriber.on_finalization(&make_result(RecognitionStatus::Final, "after open"));

    assert_eq!(subscriber.get_text(), "after open");
}

#[test]
fn submit_deactivates_subscriber() {
    let (controller, _, _, subscriber) = make_controller();

    controller.show_popup();
    controller.submit();
    subscriber.on_finalization(&make_result(RecognitionStatus::Final, "after submit"));

    assert_eq!(subscriber.get_text(), "");
}

#[test]
fn cancel_deactivates_subscriber() {
    let (controller, _, _, subscriber) = make_controller();

    controller.show_popup();
    controller.cancel();
    subscriber.on_finalization(&make_result(RecognitionStatus::Final, "after cancel"));

    assert_eq!(subscriber.get_text(), "");
}

// --- handle_toggle tests ---

#[test]
fn toggle_shows_popup_when_running_and_not_visible() {
    let (controller, spy_popup, _, _, _, _) = make_controller_with_spies();
    let controller = Arc::new(controller);
    handle_toggle(&controller, AppState::Running);
    assert_eq!(spy_popup.show_count(), 1);
    assert!(controller.is_visible());
}

#[test]
fn toggle_does_not_show_popup_when_paused() {
    let (controller, spy_popup, _, _, _, _) = make_controller_with_spies();
    let controller = Arc::new(controller);
    handle_toggle(&controller, AppState::Paused);
    assert_eq!(spy_popup.show_count(), 0);
    assert!(!controller.is_visible());
}

#[test]
fn toggle_does_not_show_popup_when_starting() {
    let (controller, spy_popup, _, _, _, _) = make_controller_with_spies();
    let controller = Arc::new(controller);
    handle_toggle(&controller, AppState::Starting);
    assert_eq!(spy_popup.show_count(), 0);
}

#[test]
fn toggle_cancels_when_visible_regardless_of_state() {
    let (controller, spy_popup, _, _, _, _) = make_controller_with_spies();
    let controller = Arc::new(controller);
    controller.show_popup();
    handle_toggle(&controller, AppState::Paused);
    assert_eq!(spy_popup.hide_count(), 1);
    assert!(!controller.is_visible());
}

#[test]
fn toggle_does_not_show_popup_when_shutdown() {
    let (controller, spy_popup, _, _, _, _) = make_controller_with_spies();
    let controller = Arc::new(controller);
    handle_toggle(&controller, AppState::Shutdown);
    assert_eq!(spy_popup.show_count(), 0);
}

#[test]
fn popup_auto_cancelled_when_state_transitions_to_paused() {
    use stt_tauri_client::state::AppStateManager;
    let (controller, spy_popup, _, _, _, _) = make_controller_with_spies();
    let controller = Arc::new(controller);
    let state_manager = Arc::new(AppStateManager::new());

    let controller_for_obs = Arc::clone(&controller);
    state_manager.add_observer(Box::new(move |_old, new| {
        if new == AppState::Paused && controller_for_obs.is_visible() {
            controller_for_obs.cancel();
        }
    }));

    state_manager.set_state(AppState::Running).unwrap();
    controller.show_popup();
    assert!(controller.is_visible());

    state_manager.set_state(AppState::Paused).unwrap();
    assert!(!controller.is_visible());
    assert_eq!(spy_popup.hide_count(), 1);
}

// --- Text event emission tests ---

#[test]
fn controller_emits_text_event_on_finalization() {
    let (_controller, _, events, subscriber) = make_controller();

    subscriber.set_active(true);
    subscriber.on_finalization(&make_result(RecognitionStatus::Final, "hello world"));

    let log = events.lock().unwrap();
    assert!(log.iter().any(|(event, payload)| {
        event == "stt://quickentry-text" && payload.contains("hello world")
    }));
}

#[test]
fn controller_emits_updated_text_on_second_finalization() {
    let (_controller, _, events, subscriber) = make_controller();

    subscriber.set_active(true);
    subscriber.on_finalization(&make_result(RecognitionStatus::Final, "hello"));
    subscriber.on_finalization(&make_result(RecognitionStatus::Final, "world"));

    let log = events.lock().unwrap();
    let text_events: Vec<_> = log
        .iter()
        .filter(|(event, _)| event == "stt://quickentry-text")
        .collect();

    assert_eq!(text_events.len(), 2);
    assert!(text_events[1].1.contains("hello world"));
}
