use stt_tauri_client::recognition::{
    RecognitionFanOut, RecognitionResult, RecognitionStatus, RecognitionSubscriber,
};
use std::sync::{Arc, Mutex};

/// Test helper that records every call it receives.
struct SpySubscriber {
    partials: Arc<Mutex<Vec<RecognitionResult>>>,
    finals: Arc<Mutex<Vec<RecognitionResult>>>,
}

impl SpySubscriber {
    fn new() -> (Self, Arc<Mutex<Vec<RecognitionResult>>>, Arc<Mutex<Vec<RecognitionResult>>>) {
        let partials = Arc::new(Mutex::new(Vec::new()));
        let finals = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                partials: Arc::clone(&partials),
                finals: Arc::clone(&finals),
            },
            partials,
            finals,
        )
    }
}

impl RecognitionSubscriber for SpySubscriber {
    fn on_partial(&self, result: &RecognitionResult) {
        self.partials.lock().unwrap().push(result.clone());
    }

    fn on_finalization(&self, result: &RecognitionResult) {
        self.finals.lock().unwrap().push(result.clone());
    }
}

/// Subscriber that panics on every call — used to verify isolation.
struct PanickingSubscriber;

impl RecognitionSubscriber for PanickingSubscriber {
    fn on_partial(&self, _result: &RecognitionResult) {
        panic!("boom from on_partial");
    }

    fn on_finalization(&self, _result: &RecognitionResult) {
        panic!("boom from on_finalization");
    }
}

fn make_result(status: RecognitionStatus, text: &str) -> RecognitionResult {
    RecognitionResult {
        session_id: "sess-1".into(),
        status,
        text: text.into(),
        start_time: 0.0,
        end_time: 1.0,
        chunk_ids: vec![1, 2],
        utterance_id: 1,
        token_confidences: vec![0.95, 0.88],
    }
}

#[test]
fn fanout_dispatches_on_partial_to_all_subscribers() {
    let (spy1, partials1, _finals1) = SpySubscriber::new();
    let (spy2, partials2, _finals2) = SpySubscriber::new();

    let mut fanout = RecognitionFanOut::new();
    fanout.add_subscriber(Box::new(spy1));
    fanout.add_subscriber(Box::new(spy2));

    let result = make_result(RecognitionStatus::Partial, "hello");
    fanout.on_partial(&result);

    assert_eq!(partials1.lock().unwrap().len(), 1);
    assert_eq!(partials2.lock().unwrap().len(), 1);
    assert_eq!(partials1.lock().unwrap()[0].text, "hello");
}

#[test]
fn fanout_dispatches_on_finalization_to_all_subscribers() {
    let (spy1, _partials1, finals1) = SpySubscriber::new();
    let (spy2, _partials2, finals2) = SpySubscriber::new();

    let mut fanout = RecognitionFanOut::new();
    fanout.add_subscriber(Box::new(spy1));
    fanout.add_subscriber(Box::new(spy2));

    let result = make_result(RecognitionStatus::Final, "world");
    fanout.on_finalization(&result);

    assert_eq!(finals1.lock().unwrap().len(), 1);
    assert_eq!(finals2.lock().unwrap().len(), 1);
    assert_eq!(finals1.lock().unwrap()[0].text, "world");
}

#[test]
fn panicking_subscriber_does_not_prevent_others_on_partial() {
    let (spy, partials, _finals) = SpySubscriber::new();

    let mut fanout = RecognitionFanOut::new();
    fanout.add_subscriber(Box::new(PanickingSubscriber));
    fanout.add_subscriber(Box::new(spy));

    let result = make_result(RecognitionStatus::Partial, "survive");
    fanout.on_partial(&result);

    assert_eq!(partials.lock().unwrap().len(), 1);
    assert_eq!(partials.lock().unwrap()[0].text, "survive");
}

#[test]
fn panicking_subscriber_does_not_prevent_others_on_finalization() {
    let (spy, _partials, finals) = SpySubscriber::new();

    let mut fanout = RecognitionFanOut::new();
    fanout.add_subscriber(Box::new(PanickingSubscriber));
    fanout.add_subscriber(Box::new(spy));

    let result = make_result(RecognitionStatus::Final, "survive");
    fanout.on_finalization(&result);

    assert_eq!(finals.lock().unwrap().len(), 1);
}

#[test]
fn empty_fanout_does_not_panic() {
    let fanout = RecognitionFanOut::new();
    let result = make_result(RecognitionStatus::Partial, "no subscribers");
    fanout.on_partial(&result);
    fanout.on_finalization(&result);
}

#[test]
fn recognition_result_default_collections() {
    let result = RecognitionResult {
        session_id: "s".into(),
        status: RecognitionStatus::Final,
        text: "hi".into(),
        start_time: 0.0,
        end_time: 0.5,
        chunk_ids: vec![],
        utterance_id: 0,
        token_confidences: vec![],
    };
    assert!(result.chunk_ids.is_empty());
    assert!(result.token_confidences.is_empty());
    assert_eq!(result.utterance_id, 0);
}

#[test]
fn recognition_result_deserializes_from_server_json() {
    let json = r#"{
        "type": "recognition_result",
        "session_id": "abc-123",
        "status": "partial",
        "text": "hello world",
        "start_time": 1.5,
        "end_time": 3.2,
        "chunk_ids": [1, 2, 3],
        "utterance_id": 1,
        "token_confidences": [0.9, 0.8]
    }"#;

    let result: RecognitionResult = serde_json::from_str(json).unwrap();
    assert_eq!(result.session_id, "abc-123");
    assert_eq!(result.status, RecognitionStatus::Partial);
    assert_eq!(result.text, "hello world");
    assert_eq!(result.start_time, 1.5);
    assert_eq!(result.end_time, 3.2);
    assert_eq!(result.chunk_ids, vec![1, 2, 3]);
    assert_eq!(result.utterance_id, 1);
    assert_eq!(result.token_confidences, vec![0.9, 0.8]);
}

#[test]
fn recognition_result_deserializes_final_status() {
    let json = r#"{
        "type": "recognition_result",
        "session_id": "s",
        "status": "final",
        "text": "done",
        "start_time": 0.0,
        "end_time": 1.0,
        "chunk_ids": [],
        "utterance_id": 2,
        "token_confidences": []
    }"#;

    let result: RecognitionResult = serde_json::from_str(json).unwrap();
    assert_eq!(result.status, RecognitionStatus::Final);
}
