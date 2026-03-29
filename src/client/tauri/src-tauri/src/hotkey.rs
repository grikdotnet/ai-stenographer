/// Global hotkey listener for the STT client.
///
/// Provides `GlobalHotkeyListener` (Win32 RegisterHotKey/GetMessage loop on
/// a dedicated thread) and `NoOpHotkeyListener` for testing and non-Windows.
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Trait for a global hotkey listener.
///
/// Abstracts the OS-level hotkey registration so that tests can use
/// `NoOpHotkeyListener` without Win32 dependencies.
pub trait HotkeyListener: Send + Sync {
    /// Starts the listener thread and registers the toggle hotkey (Ctrl+Space).
    fn start(&self, on_toggle: Box<dyn Fn() + Send>);

    /// Registers Enter and Escape hotkeys for the quick-entry popup.
    fn register_popup_hotkeys(
        &self,
        on_submit: Box<dyn Fn() + Send>,
        on_cancel: Box<dyn Fn() + Send>,
    );

    /// Unregisters Enter and Escape hotkeys.
    fn unregister_popup_hotkeys(&self);

    /// Stops the listener thread.
    fn stop(&self);
}

/// No-op hotkey listener for testing and non-Windows platforms.
///
/// All methods are safe no-ops.
pub struct NoOpHotkeyListener;

impl NoOpHotkeyListener {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoOpHotkeyListener {
    fn default() -> Self {
        Self::new()
    }
}

impl HotkeyListener for NoOpHotkeyListener {
    fn start(&self, _on_toggle: Box<dyn Fn() + Send>) {}

    fn register_popup_hotkeys(
        &self,
        _on_submit: Box<dyn Fn() + Send>,
        _on_cancel: Box<dyn Fn() + Send>,
    ) {
    }

    fn unregister_popup_hotkeys(&self) {}

    fn stop(&self) {}
}

/// Win32 global hotkey listener using RegisterHotKey and a GetMessage loop.
///
/// Runs a dedicated OS thread that owns the hotkey registrations.
/// Communication from other threads uses PostThreadMessageW to send
/// custom WM_APP messages that trigger register/unregister actions.
///
/// Hotkey IDs:
/// - 9001: Ctrl+Space (toggle)
/// - 9002: Enter (submit)
/// - 9003: Escape (cancel)
#[cfg(windows)]
pub struct GlobalHotkeyListener {
    thread_id: AtomicU32,
    running: Arc<AtomicBool>,
    toggle_callback: Arc<Mutex<Option<Box<dyn Fn() + Send>>>>,
    submit_callback: Arc<Mutex<Option<Box<dyn Fn() + Send>>>>,
    cancel_callback: Arc<Mutex<Option<Box<dyn Fn() + Send>>>>,
    join_handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

#[cfg(windows)]
impl GlobalHotkeyListener {
    const HOTKEY_TOGGLE: i32 = 9001;
    const HOTKEY_SUBMIT: i32 = 9002;
    const HOTKEY_CANCEL: i32 = 9003;

    // WM_APP + 1 = register popup hotkeys, WM_APP + 2 = unregister
    const WM_REGISTER_POPUP: u32 = 0x8001;
    const WM_UNREGISTER_POPUP: u32 = 0x8002;

    pub fn new() -> Self {
        Self {
            thread_id: AtomicU32::new(0),
            running: Arc::new(AtomicBool::new(false)),
            toggle_callback: Arc::new(Mutex::new(None)),
            submit_callback: Arc::new(Mutex::new(None)),
            cancel_callback: Arc::new(Mutex::new(None)),
            join_handle: Mutex::new(None),
        }
    }
}

#[cfg(windows)]
impl HotkeyListener for GlobalHotkeyListener {
    /// Spawns the hotkey message loop thread and registers Ctrl+Space.
    ///
    /// Algorithm:
    /// 1. Store the toggle callback.
    /// 2. Spawn an OS thread that registers Ctrl+Space (hotkey 9001).
    /// 3. Enter a GetMessage loop dispatching WM_HOTKEY and WM_APP messages.
    /// 4. On WM_QUIT, unregister all hotkeys and exit the thread.
    fn start(&self, on_toggle: Box<dyn Fn() + Send>) {
        use windows::Win32::System::Threading::GetCurrentThreadId;
        use windows::Win32::UI::Input::KeyboardAndMouse::{
            RegisterHotKey, UnregisterHotKey, HOT_KEY_MODIFIERS, MOD_CONTROL, VK_ESCAPE,
            VK_RETURN, VK_SPACE,
        };
        use windows::Win32::UI::WindowsAndMessaging::{GetMessageW, MSG, WM_HOTKEY};

        *self.toggle_callback.lock().unwrap() = Some(on_toggle);
        self.running.store(true, Ordering::SeqCst);

        let running = Arc::clone(&self.running);
        let toggle_cb = Arc::clone(&self.toggle_callback);
        let submit_cb = Arc::clone(&self.submit_callback);
        let cancel_cb = Arc::clone(&self.cancel_callback);
        let thread_id_store = &self.thread_id as *const AtomicU32 as usize;

        let handle = std::thread::spawn(move || {
            // SAFETY: We cast the pointer back; the AtomicU32 outlives this thread
            // because stop() joins before dropping.
            let thread_id_ref = unsafe { &*(thread_id_store as *const AtomicU32) };

            // SAFETY: GetCurrentThreadId has no preconditions.
            let tid = unsafe { GetCurrentThreadId() };
            thread_id_ref.store(tid, Ordering::SeqCst);

            // SAFETY: RegisterHotKey with None HWND registers for the thread message queue.
            let _ = unsafe {
                RegisterHotKey(
                    None,
                    Self::HOTKEY_TOGGLE,
                    MOD_CONTROL,
                    VK_SPACE.0 as u32,
                )
            };

            let mut msg = MSG::default();
            loop {
                // SAFETY: GetMessageW blocks until a message arrives. Returns FALSE for WM_QUIT.
                let ret = unsafe { GetMessageW(&mut msg, None, 0, 0) };
                if !ret.as_bool() {
                    break;
                }

                match msg.message {
                    WM_HOTKEY => {
                        let id = msg.wParam.0 as i32;
                        match id {
                            Self::HOTKEY_TOGGLE => {
                                if let Ok(cb) = toggle_cb.lock() {
                                    if let Some(ref f) = *cb {
                                        f();
                                    }
                                }
                            }
                            Self::HOTKEY_SUBMIT => {
                                if let Ok(cb) = submit_cb.lock() {
                                    if let Some(ref f) = *cb {
                                        f();
                                    }
                                }
                            }
                            Self::HOTKEY_CANCEL => {
                                if let Ok(cb) = cancel_cb.lock() {
                                    if let Some(ref f) = *cb {
                                        f();
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    m if m == Self::WM_REGISTER_POPUP => {
                        let _ = unsafe {
                            RegisterHotKey(
                                None,
                                Self::HOTKEY_SUBMIT,
                                HOT_KEY_MODIFIERS(0),
                                VK_RETURN.0 as u32,
                            )
                        };
                        let _ = unsafe {
                            RegisterHotKey(
                                None,
                                Self::HOTKEY_CANCEL,
                                HOT_KEY_MODIFIERS(0),
                                VK_ESCAPE.0 as u32,
                            )
                        };
                    }
                    m if m == Self::WM_UNREGISTER_POPUP => {
                        let _ = unsafe { UnregisterHotKey(None, Self::HOTKEY_SUBMIT) };
                        let _ = unsafe { UnregisterHotKey(None, Self::HOTKEY_CANCEL) };
                    }
                    _ => {}
                }
            }

            let _ = unsafe { UnregisterHotKey(None, Self::HOTKEY_TOGGLE) };
            let _ = unsafe { UnregisterHotKey(None, Self::HOTKEY_SUBMIT) };
            let _ = unsafe { UnregisterHotKey(None, Self::HOTKEY_CANCEL) };
            running.store(false, Ordering::SeqCst);
        });

        *self.join_handle.lock().unwrap() = Some(handle);
    }

    fn register_popup_hotkeys(
        &self,
        on_submit: Box<dyn Fn() + Send>,
        on_cancel: Box<dyn Fn() + Send>,
    ) {
        use windows::Win32::Foundation::{LPARAM, WPARAM};
        use windows::Win32::UI::WindowsAndMessaging::PostThreadMessageW;

        *self.submit_callback.lock().unwrap() = Some(on_submit);
        *self.cancel_callback.lock().unwrap() = Some(on_cancel);

        let tid = self.thread_id.load(Ordering::SeqCst);
        if tid != 0 {
            // SAFETY: PostThreadMessageW to a valid thread ID; best-effort.
            unsafe {
                let _ = PostThreadMessageW(
                    tid,
                    Self::WM_REGISTER_POPUP,
                    WPARAM(0),
                    LPARAM(0),
                );
            }
        }
    }

    fn unregister_popup_hotkeys(&self) {
        use windows::Win32::Foundation::{LPARAM, WPARAM};
        use windows::Win32::UI::WindowsAndMessaging::PostThreadMessageW;

        let tid = self.thread_id.load(Ordering::SeqCst);
        if tid != 0 {
            // SAFETY: PostThreadMessageW to a valid thread ID; best-effort.
            unsafe {
                let _ = PostThreadMessageW(
                    tid,
                    Self::WM_UNREGISTER_POPUP,
                    WPARAM(0),
                    LPARAM(0),
                );
            }
        }
    }

    fn stop(&self) {
        use windows::Win32::Foundation::{LPARAM, WPARAM};
        use windows::Win32::UI::WindowsAndMessaging::{PostThreadMessageW, WM_QUIT};

        let tid = self.thread_id.load(Ordering::SeqCst);
        if tid != 0 {
            // SAFETY: Posting WM_QUIT causes GetMessageW to return FALSE, ending the loop.
            unsafe {
                let _ = PostThreadMessageW(tid, WM_QUIT, WPARAM(0), LPARAM(0));
            }
        }

        if let Some(handle) = self.join_handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

#[cfg(windows)]
impl Default for GlobalHotkeyListener {
    fn default() -> Self {
        Self::new()
    }
}
