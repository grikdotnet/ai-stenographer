use stt_tauri_client::focus::{FocusTracker, WindowFocusApi};
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};

/// Fake Win32 focus API that records all calls for verification.
struct FakeFocusApi {
    foreground_hwnd: AtomicIsize,
    set_foreground_result: AtomicBool,
    current_thread_id: u32,
    window_thread_id: u32,
    attach_calls: std::sync::Mutex<Vec<(u32, u32, bool)>>,
    set_foreground_calls: std::sync::Mutex<Vec<isize>>,
}

impl FakeFocusApi {
    fn new() -> Self {
        Self {
            foreground_hwnd: AtomicIsize::new(0),
            set_foreground_result: AtomicBool::new(true),
            current_thread_id: 100,
            window_thread_id: 200,
            attach_calls: std::sync::Mutex::new(Vec::new()),
            set_foreground_calls: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn with_foreground(self, hwnd: isize) -> Self {
        self.foreground_hwnd.store(hwnd, Ordering::SeqCst);
        self
    }

    fn with_set_foreground_failing(self) -> Self {
        self.set_foreground_result.store(false, Ordering::SeqCst);
        self
    }
}

impl WindowFocusApi for FakeFocusApi {
    fn get_foreground_window(&self) -> isize {
        self.foreground_hwnd.load(Ordering::SeqCst)
    }

    fn set_foreground_window(&self, hwnd: isize) -> bool {
        self.set_foreground_calls.lock().unwrap().push(hwnd);
        self.set_foreground_result.load(Ordering::SeqCst)
    }

    fn get_window_thread_process_id(&self, _hwnd: isize) -> u32 {
        self.window_thread_id
    }

    fn get_current_thread_id(&self) -> u32 {
        self.current_thread_id
    }

    fn attach_thread_input(&self, attach_to: u32, attach_from: u32, attach: bool) -> bool {
        self.attach_calls
            .lock()
            .unwrap()
            .push((attach_to, attach_from, attach));
        true
    }
}

// --- Tests ---

#[test]
fn save_captures_current_foreground_hwnd() {
    let api = FakeFocusApi::new().with_foreground(42);
    let tracker = FocusTracker::new(api);

    tracker.save();

    tracker.restore();
    // restore should have called set_foreground_window with the saved hwnd 42
}

#[test]
fn restore_calls_set_foreground_window_with_saved_hwnd() {
    let api = FakeFocusApi::new().with_foreground(999);
    let tracker = FocusTracker::new(api);

    tracker.save();
    tracker.restore();

    let calls = tracker.api().set_foreground_calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0], 999);
}

#[test]
fn restore_fallback_uses_attach_thread_input_when_set_foreground_fails() {
    let api = FakeFocusApi::new()
        .with_foreground(777)
        .with_set_foreground_failing();
    let tracker = FocusTracker::new(api);

    tracker.save();
    tracker.restore();

    let attach_calls = tracker.api().attach_calls.lock().unwrap();
    assert!(
        attach_calls.len() >= 2,
        "Expected at least attach + detach calls, got {}",
        attach_calls.len()
    );

    let (attach_to, attach_from, attach) = attach_calls[0];
    assert_eq!(attach_to, 200);
    assert_eq!(attach_from, 100);
    assert!(attach);
}

#[test]
fn detaches_thread_input_after_restore_attempt() {
    let api = FakeFocusApi::new()
        .with_foreground(777)
        .with_set_foreground_failing();
    let tracker = FocusTracker::new(api);

    tracker.save();
    tracker.restore();

    let attach_calls = tracker.api().attach_calls.lock().unwrap();
    let last = attach_calls.last().expect("expected detach call");
    assert!(!last.2, "Last attach call should be a detach (false)");
}

#[test]
fn restore_with_no_saved_hwnd_is_noop() {
    let api = FakeFocusApi::new();
    let tracker = FocusTracker::new(api);

    tracker.restore();

    let calls = tracker.api().set_foreground_calls.lock().unwrap();
    assert!(calls.is_empty());
}

#[test]
fn save_overwrites_previous_hwnd() {
    let api = FakeFocusApi::new().with_foreground(10);
    let tracker = FocusTracker::new(api);

    tracker.save();

    tracker.api().foreground_hwnd.store(20, Ordering::SeqCst);
    tracker.save();
    tracker.restore();

    let calls = tracker.api().set_foreground_calls.lock().unwrap();
    assert_eq!(calls.last(), Some(&20isize));
}
