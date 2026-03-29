/// Window focus tracking for save/restore of the previously-focused window.
///
/// Uses a `WindowFocusApi` trait to abstract Win32 calls, enabling
/// full testability without a real desktop environment.
use std::sync::Mutex;

/// Abstraction over Win32 window focus operations.
///
/// Enables test fakes to replace real OS calls.
pub trait WindowFocusApi: Send + Sync {
    /// Returns the HWND of the current foreground window (as isize).
    fn get_foreground_window(&self) -> isize;

    /// Attempts to bring the given window to the foreground.
    fn set_foreground_window(&self, hwnd: isize) -> bool;

    /// Returns the thread ID that created the given window.
    fn get_window_thread_process_id(&self, hwnd: isize) -> u32;

    /// Returns the current thread's ID.
    fn get_current_thread_id(&self) -> u32;

    /// Attaches or detaches the input processing of two threads.
    fn attach_thread_input(&self, attach_to: u32, attach_from: u32, attach: bool) -> bool;
}

/// Win32 implementation of `WindowFocusApi`.
#[cfg(windows)]
pub struct Win32FocusApi;

#[cfg(windows)]
impl WindowFocusApi for Win32FocusApi {
    fn get_foreground_window(&self) -> isize {
        use windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow;
        // SAFETY: GetForegroundWindow has no preconditions and never fails fatally.
        let hwnd = unsafe { GetForegroundWindow() };
        hwnd.0 as isize
    }

    fn set_foreground_window(&self, hwnd: isize) -> bool {
        use windows::Win32::Foundation::HWND;
        use windows::Win32::UI::WindowsAndMessaging::SetForegroundWindow;
        // SAFETY: SetForegroundWindow is best-effort; invalid HWND returns FALSE.
        unsafe { SetForegroundWindow(HWND(hwnd as *mut _)).as_bool() }
    }

    fn get_window_thread_process_id(&self, hwnd: isize) -> u32 {
        use windows::Win32::Foundation::HWND;
        use windows::Win32::UI::WindowsAndMessaging::GetWindowThreadProcessId;
        // SAFETY: Returns 0 for invalid HWND, no UB.
        unsafe { GetWindowThreadProcessId(HWND(hwnd as *mut _), None) }
    }

    fn get_current_thread_id(&self) -> u32 {
        use windows::Win32::System::Threading::GetCurrentThreadId;
        // SAFETY: No preconditions.
        unsafe { GetCurrentThreadId() }
    }

    fn attach_thread_input(&self, attach_to: u32, attach_from: u32, attach: bool) -> bool {
        use windows::Win32::System::Threading::AttachThreadInput;
        // SAFETY: AttachThreadInput is best-effort; mismatched IDs return FALSE.
        unsafe { AttachThreadInput(attach_to, attach_from, attach).as_bool() }
    }
}

/// Saves and restores window focus.
///
/// Responsibilities:
/// - `save()` captures the current foreground window HWND.
/// - `restore()` brings the saved window back to the foreground,
///   using AttachThreadInput as a fallback when SetForegroundWindow fails.
pub struct FocusTracker<F: WindowFocusApi> {
    api: F,
    saved_hwnd: Mutex<Option<isize>>,
}

impl<F: WindowFocusApi> FocusTracker<F> {
    /// Creates a new FocusTracker with the given platform API.
    pub fn new(api: F) -> Self {
        Self {
            api,
            saved_hwnd: Mutex::new(None),
        }
    }

    /// Returns a reference to the underlying API (for test inspection).
    pub fn api(&self) -> &F {
        &self.api
    }

    /// Saves the current foreground window's HWND.
    pub fn save(&self) {
        let hwnd = self.api.get_foreground_window();
        *self.saved_hwnd.lock().unwrap() = Some(hwnd);
    }

    /// Restores focus to the previously saved window.
    ///
    /// Algorithm:
    /// 1. Read saved HWND; if None, return immediately.
    /// 2. Try SetForegroundWindow directly.
    /// 3. If that fails, attach our thread's input to the target window's thread.
    /// 4. Retry SetForegroundWindow.
    /// 5. Detach thread input regardless of outcome.
    pub fn restore(&self) {
        let hwnd = match *self.saved_hwnd.lock().unwrap() {
            Some(h) => h,
            None => return,
        };

        if self.api.set_foreground_window(hwnd) {
            return;
        }

        tracing::debug!("SetForegroundWindow failed, trying AttachThreadInput fallback");

        let target_thread = self.api.get_window_thread_process_id(hwnd);
        let our_thread = self.api.get_current_thread_id();

        self.api.attach_thread_input(target_thread, our_thread, true);
        let _ = self.api.set_foreground_window(hwnd);
        self.api
            .attach_thread_input(target_thread, our_thread, false);
    }
}
