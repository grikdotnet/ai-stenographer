use stt_tauri_client::error::AppError;
use stt_tauri_client::insertion::{InsertionController, TextInserter};
use stt_tauri_client::keyboard::KeyboardSimulator;
use stt_tauri_client::recognition::{RecognitionResult, RecognitionStatus, RecognitionSubscriber};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Fake keyboard that records every typed string.
struct FakeKeyboardSimulator {
    recorded: Arc<Mutex<Vec<String>>>,
}

impl FakeKeyboardSimulator {
    fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
        let recorded = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                recorded: Arc::clone(&recorded),
            },
            recorded,
        )
    }
}

impl KeyboardSimulator for FakeKeyboardSimulator {
    fn type_text(&self, text: &str) -> Result<(), AppError> {
        self.recorded.lock().unwrap().push(text.to_string());
        Ok(())
    }
}

fn make_result(status: RecognitionStatus, text: &str) -> RecognitionResult {
    RecognitionResult {
        session_id: "sess-1".into(),
        status,
        text: text.into(),
        start_time: 0.0,
        end_time: 1.0,
        chunk_ids: vec![1],
        utterance_id: 1,
        token_confidences: vec![0.9],
    }
}

// --- TextInserter tests ---

#[test]
fn text_inserter_ignores_on_partial() {
    let (kb, recorded) = FakeKeyboardSimulator::new();
    let enabled = Arc::new(AtomicBool::new(true));
    let inserter = TextInserter::new(Arc::new(kb), Arc::clone(&enabled));

    let result = make_result(RecognitionStatus::Partial, "partial text");
    inserter.on_partial(&result);

    assert!(recorded.lock().unwrap().is_empty());
}

#[test]
fn text_inserter_types_on_finalization_when_enabled() {
    let (kb, recorded) = FakeKeyboardSimulator::new();
    let enabled = Arc::new(AtomicBool::new(true));
    let inserter = TextInserter::new(Arc::new(kb), Arc::clone(&enabled));

    let result = make_result(RecognitionStatus::Final, "hello world");
    inserter.on_finalization(&result);

    let log = recorded.lock().unwrap();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0], "hello world");
}

#[test]
fn text_inserter_skips_finalization_when_disabled() {
    let (kb, recorded) = FakeKeyboardSimulator::new();
    let enabled = Arc::new(AtomicBool::new(false));
    let inserter = TextInserter::new(Arc::new(kb), Arc::clone(&enabled));

    let result = make_result(RecognitionStatus::Final, "should not type");
    inserter.on_finalization(&result);

    assert!(recorded.lock().unwrap().is_empty());
}

#[test]
fn text_inserter_responds_to_runtime_enable_toggle() {
    let (kb, recorded) = FakeKeyboardSimulator::new();
    let enabled = Arc::new(AtomicBool::new(false));
    let inserter = TextInserter::new(Arc::new(kb), Arc::clone(&enabled));

    let result = make_result(RecognitionStatus::Final, "first");
    inserter.on_finalization(&result);
    assert!(recorded.lock().unwrap().is_empty());

    enabled.store(true, Ordering::SeqCst);

    let result2 = make_result(RecognitionStatus::Final, "second");
    inserter.on_finalization(&result2);

    let log = recorded.lock().unwrap();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0], "second");
}

// --- InsertionController tests ---

#[test]
fn controller_starts_disabled() {
    let controller = InsertionController::new();
    assert!(!controller.is_enabled());
}

#[test]
fn controller_toggle_flips_state() {
    let controller = InsertionController::new();
    assert!(!controller.is_enabled());

    controller.toggle();
    assert!(controller.is_enabled());

    controller.toggle();
    assert!(!controller.is_enabled());
}

#[test]
fn controller_toggle_returns_new_state() {
    let controller = InsertionController::new();

    let first = controller.toggle();
    assert!(first);

    let second = controller.toggle();
    assert!(!second);
}

#[test]
fn controller_set_enabled_overrides_state() {
    let controller = InsertionController::new();

    controller.set_enabled(true);
    assert!(controller.is_enabled());

    controller.set_enabled(false);
    assert!(!controller.is_enabled());
}

#[test]
fn controller_callback_notified_on_toggle() {
    let notifications = Arc::new(Mutex::new(Vec::new()));
    let notify_clone = Arc::clone(&notifications);

    let controller = InsertionController::with_callback(Box::new(move |enabled| {
        notify_clone.lock().unwrap().push(enabled);
    }));

    controller.toggle();
    controller.toggle();

    let log = notifications.lock().unwrap();
    assert_eq!(log.len(), 2);
    assert!(log[0]);
    assert!(!log[1]);
}

#[test]
fn controller_callback_notified_on_set_enabled() {
    let notifications = Arc::new(Mutex::new(Vec::new()));
    let notify_clone = Arc::clone(&notifications);

    let controller = InsertionController::with_callback(Box::new(move |enabled| {
        notify_clone.lock().unwrap().push(enabled);
    }));

    controller.set_enabled(true);
    controller.set_enabled(false);

    let log = notifications.lock().unwrap();
    assert_eq!(log.len(), 2);
    assert!(log[0]);
    assert!(!log[1]);
}

#[test]
fn controller_enabled_flag_shared_with_text_inserter() {
    let controller = InsertionController::new();
    let (kb, recorded) = FakeKeyboardSimulator::new();
    let inserter = TextInserter::new(Arc::new(kb), controller.enabled_flag());

    let result = make_result(RecognitionStatus::Final, "before toggle");
    inserter.on_finalization(&result);
    assert!(recorded.lock().unwrap().is_empty());

    controller.toggle();

    let result2 = make_result(RecognitionStatus::Final, "after toggle");
    inserter.on_finalization(&result2);

    let log = recorded.lock().unwrap();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0], "after toggle");
}
