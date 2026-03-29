/// Text formatter that accumulates recognized speech as a list of finalized utterances.
///
/// Implements `RecognitionSubscriber` to receive partial and finalized results.
/// Formatting decisions (paragraph breaks, spacing) are the responsibility of
/// the view layer. Manages duplicate suppression and notifies an optional
/// observer callback on every text change.
use std::sync::{Arc, Mutex};

use serde::Serialize;

use crate::recognition::{RecognitionResult, RecognitionSubscriber};

type OnChangeFn = Box<dyn Fn() + Send + Sync>;

/// A single finalized recognition result with timing metadata.
#[derive(Clone, Serialize, Debug)]
pub struct FinalizedUtterance {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// Internal mutable state for the formatter.
struct FormatterState {
    utterances: Vec<FinalizedUtterance>,
    preliminary_text: String,
    last_final_end_time: Option<f64>,
    last_final_text: Option<String>,
    last_final_utterance_id: Option<i64>,
}

/// Accumulates recognition results as a list of finalized utterances.
///
/// Holds its state behind `Arc<Mutex<...>>` so it can be both a
/// `RecognitionSubscriber` and queried for current text concurrently.
/// An optional callback is invoked whenever text changes (for UI refresh).
/// The callback can be set or replaced at any time via `set_on_change`.
/// Formatting decisions (paragraph breaks, spacing) are the responsibility of
/// the view layer.
pub struct TextFormatter {
    state: Arc<Mutex<FormatterState>>,
    on_change: Arc<Mutex<Option<OnChangeFn>>>,
}

impl TextFormatter {
    /// Creates a new formatter with an optional change-notification callback.
    pub fn new(on_change: Option<OnChangeFn>) -> Self {
        Self {
            state: Arc::new(Mutex::new(FormatterState {
                utterances: Vec::new(),
                preliminary_text: String::new(),
                last_final_end_time: None,
                last_final_text: None,
                last_final_utterance_id: None,
            })),
            on_change: Arc::new(Mutex::new(on_change)),
        }
    }

    /// Replaces the change-notification callback.
    ///
    /// Safe to call at any time, including after the formatter has been shared
    /// with subscribers. The new callback takes effect on the next text change.
    pub fn set_on_change(&self, cb: OnChangeFn) {
        if let Ok(mut guard) = self.on_change.lock() {
            *guard = Some(cb);
        }
    }

    /// Returns the accumulated finalized utterances.
    pub fn get_utterances(&self) -> Vec<FinalizedUtterance> {
        self.state.lock().unwrap().utterances.clone()
    }

    /// Returns the current preliminary (partial) text.
    pub fn get_preliminary_text(&self) -> String {
        self.state.lock().unwrap().preliminary_text.clone()
    }

    /// Resets all accumulated state and notifies the observer.
    pub fn clear(&self) {
        {
            let mut s = self.state.lock().unwrap();
            s.utterances.clear();
            s.preliminary_text.clear();
            s.last_final_end_time = None;
            s.last_final_text = None;
            s.last_final_utterance_id = None;
        }
        self.notify();
    }

    fn notify(&self) {
        if let Ok(guard) = self.on_change.lock() {
            if let Some(cb) = guard.as_ref() {
                cb();
            }
        }
    }
}

impl RecognitionSubscriber for TextFormatter {
    fn on_partial(&self, result: &RecognitionResult) {
        {
            let mut s = self.state.lock().unwrap();
            s.preliminary_text = result.text.clone();
        }
        self.notify();
    }

    /// Appends a finalized utterance with duplicate suppression.
    ///
    /// Algorithm:
    ///   1. Check for duplicate (same text + same utterance_id as previous final) — skip if so.
    ///   2. Push a new `FinalizedUtterance` with text and timestamps to the utterances vec.
    ///   3. Clear preliminary text and update tracking fields.
    ///   4. Notify observer.
    fn on_finalization(&self, result: &RecognitionResult) {
        {
            let mut s = self.state.lock().unwrap();

            if let (Some(ref prev_text), Some(prev_uid)) =
                (&s.last_final_text, s.last_final_utterance_id)
            {
                if *prev_text == result.text && prev_uid == result.utterance_id {
                    return;
                }
            }

            s.utterances.push(FinalizedUtterance {
                text: result.text.clone(),
                start_time: result.start_time,
                end_time: result.end_time,
            });
            s.preliminary_text.clear();
            s.last_final_end_time = Some(result.end_time);
            s.last_final_text = Some(result.text.clone());
            s.last_final_utterance_id = Some(result.utterance_id);
        }
        self.notify();
    }
}
