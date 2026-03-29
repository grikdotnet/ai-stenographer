/// Recognition types and fan-out dispatcher for the STT client.
///
/// Provides `RecognitionResult` (deserializable from the server wire protocol),
/// `RecognitionSubscriber` (observer trait), and `RecognitionFanOut` (broadcasts
/// results to multiple subscribers with panic isolation).
use serde::Deserialize;
use std::panic::{self, AssertUnwindSafe};

/// Status of a recognition result — partial (in-progress) or final.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecognitionStatus {
    Partial,
    Final,
}

/// A single recognition result from the STT server.
///
/// Fields mirror the server's `WsRecognitionResult` JSON payload.
/// The `type` field is ignored during deserialization (handled at the transport layer).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RecognitionResult {
    pub session_id: String,
    pub status: RecognitionStatus,
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
    pub chunk_ids: Vec<i64>,
    pub utterance_id: i64,
    pub token_confidences: Vec<f64>,
}

/// Observer trait for receiving recognition events.
pub trait RecognitionSubscriber: Send + Sync {
    /// Called when a partial (in-progress) recognition result arrives.
    fn on_partial(&self, result: &RecognitionResult);

    /// Called when a finalized recognition result arrives.
    fn on_finalization(&self, result: &RecognitionResult);
}

/// Broadcasts recognition events to multiple subscribers.
///
/// Each subscriber is called independently; a panic in one subscriber
/// is caught and does not prevent the remaining subscribers from receiving
/// the event.
pub struct RecognitionFanOut {
    subscribers: Vec<Box<dyn RecognitionSubscriber>>,
}

impl RecognitionFanOut {
    pub fn new() -> Self {
        Self {
            subscribers: Vec::new(),
        }
    }

    pub fn add_subscriber(&mut self, sub: Box<dyn RecognitionSubscriber>) {
        self.subscribers.push(sub);
    }
}

impl RecognitionSubscriber for RecognitionFanOut {
    fn on_partial(&self, result: &RecognitionResult) {
        for sub in &self.subscribers {
            let sub_ref = AssertUnwindSafe(sub);
            let result_ref = AssertUnwindSafe(result);
            let _ = panic::catch_unwind(move || {
                sub_ref.on_partial(&result_ref);
            });
        }
    }

    fn on_finalization(&self, result: &RecognitionResult) {
        for sub in &self.subscribers {
            let sub_ref = AssertUnwindSafe(sub);
            let result_ref = AssertUnwindSafe(result);
            let _ = panic::catch_unwind(move || {
                sub_ref.on_finalization(&result_ref);
            });
        }
    }
}

impl Default for RecognitionFanOut {
    fn default() -> Self {
        Self::new()
    }
}
