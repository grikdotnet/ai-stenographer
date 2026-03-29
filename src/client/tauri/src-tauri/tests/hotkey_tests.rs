use stt_tauri_client::hotkey::{HotkeyListener, NoOpHotkeyListener};

#[test]
fn noop_listener_start_does_not_panic() {
    let listener = NoOpHotkeyListener::new();
    listener.start(Box::new(|| {}));
}

#[test]
fn noop_listener_register_popup_hotkeys_does_not_panic() {
    let listener = NoOpHotkeyListener::new();
    listener.register_popup_hotkeys(Box::new(|| {}), Box::new(|| {}));
}

#[test]
fn noop_listener_unregister_popup_hotkeys_does_not_panic() {
    let listener = NoOpHotkeyListener::new();
    listener.unregister_popup_hotkeys();
}

#[test]
fn noop_listener_stop_does_not_panic() {
    let listener = NoOpHotkeyListener::new();
    listener.stop();
}

#[test]
fn noop_listener_full_lifecycle_does_not_panic() {
    let listener = NoOpHotkeyListener::new();
    listener.start(Box::new(|| {}));
    listener.register_popup_hotkeys(Box::new(|| {}), Box::new(|| {}));
    listener.unregister_popup_hotkeys();
    listener.stop();
}
