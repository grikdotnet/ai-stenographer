/// Abstraction for typing text into the focused window.
///
/// Provides a platform-agnostic trait and platform-specific implementations:
/// `Win32KeyboardSimulator` on Windows (via SendInput with Unicode events),
/// `NoOpKeyboardSimulator` on other platforms.
use crate::error::AppError;

/// Sends keystrokes to the currently focused window.
pub trait KeyboardSimulator: Send + Sync {
    /// Types the given text string by simulating keyboard input.
    ///
    /// Args:
    ///   text: The text to type. May be empty (no-op) or contain Unicode.
    ///
    /// Returns:
    ///   Ok(()) on success, or AppError on platform failure.
    fn type_text(&self, text: &str) -> Result<(), AppError>;
}

/// Win32 implementation using SendInput with KEYBDINPUT Unicode events.
///
/// Algorithm:
/// 1. Encode each character as UTF-16 code units.
/// 2. For each code unit, create a KEYDOWN + KEYUP INPUT pair with KEYEVENTF_UNICODE.
/// 3. Call SendInput with the full batch of INPUT structs.
#[cfg(windows)]
pub struct Win32KeyboardSimulator;

#[cfg(windows)]
impl KeyboardSimulator for Win32KeyboardSimulator {
    fn type_text(&self, text: &str) -> Result<(), AppError> {
        use windows::Win32::UI::Input::KeyboardAndMouse::{
            SendInput, INPUT, INPUT_0, INPUT_TYPE, KEYBDINPUT, KEYEVENTF_KEYUP,
            KEYEVENTF_UNICODE, VIRTUAL_KEY,
        };

        if text.is_empty() {
            return Ok(());
        }

        let utf16_units: Vec<u16> = text.encode_utf16().collect();
        let mut inputs: Vec<INPUT> = Vec::with_capacity(utf16_units.len() * 2);

        for code_unit in &utf16_units {
            let keydown = INPUT {
                r#type: INPUT_TYPE(1), // INPUT_KEYBOARD
                Anonymous: INPUT_0 {
                    ki: KEYBDINPUT {
                        wVk: VIRTUAL_KEY(0),
                        wScan: *code_unit,
                        dwFlags: KEYEVENTF_UNICODE,
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            };

            let keyup = INPUT {
                r#type: INPUT_TYPE(1),
                Anonymous: INPUT_0 {
                    ki: KEYBDINPUT {
                        wVk: VIRTUAL_KEY(0),
                        wScan: *code_unit,
                        dwFlags: KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            };

            inputs.push(keydown);
            inputs.push(keyup);
        }

        let sent = unsafe { SendInput(&inputs, std::mem::size_of::<INPUT>() as i32) };

        if sent as usize != inputs.len() {
            return Err(AppError::KeyboardError(format!(
                "SendInput sent {} of {} events",
                sent,
                inputs.len()
            )));
        }

        Ok(())
    }
}

/// No-op stub for non-Windows platforms.
///
/// Always returns Ok(()) — exists so the crate compiles cross-platform.
#[cfg(not(windows))]
pub struct NoOpKeyboardSimulator;

#[cfg(not(windows))]
impl KeyboardSimulator for NoOpKeyboardSimulator {
    fn type_text(&self, _text: &str) -> Result<(), AppError> {
        Ok(())
    }
}
