use stt_tauri_client::keyboard::KeyboardSimulator;
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
    fn type_text(&self, text: &str) -> Result<(), stt_tauri_client::error::AppError> {
        self.recorded.lock().unwrap().push(text.to_string());
        Ok(())
    }
}

#[test]
fn fake_keyboard_records_typed_text() {
    let (kb, recorded) = FakeKeyboardSimulator::new();
    kb.type_text("hello world").unwrap();

    let log = recorded.lock().unwrap();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0], "hello world");
}

#[test]
fn fake_keyboard_records_multiple_calls() {
    let (kb, recorded) = FakeKeyboardSimulator::new();
    kb.type_text("first").unwrap();
    kb.type_text("second").unwrap();

    let log = recorded.lock().unwrap();
    assert_eq!(log.len(), 2);
    assert_eq!(log[0], "first");
    assert_eq!(log[1], "second");
}

#[test]
fn empty_string_is_accepted_without_error() {
    let (kb, recorded) = FakeKeyboardSimulator::new();
    kb.type_text("").unwrap();

    let log = recorded.lock().unwrap();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0], "");
}

#[test]
fn unicode_text_is_recorded() {
    let (kb, recorded) = FakeKeyboardSimulator::new();
    kb.type_text("cafe\u{0301} \u{1F600}").unwrap();

    let log = recorded.lock().unwrap();
    assert_eq!(log[0], "cafe\u{0301} \u{1F600}");
}

#[cfg(windows)]
mod win32_tests {
    use stt_tauri_client::keyboard::Win32KeyboardSimulator;
    use stt_tauri_client::keyboard::KeyboardSimulator;

    #[test]
    fn win32_simulator_handles_empty_string() {
        let sim = Win32KeyboardSimulator;
        assert!(sim.type_text("").is_ok());
    }
}
