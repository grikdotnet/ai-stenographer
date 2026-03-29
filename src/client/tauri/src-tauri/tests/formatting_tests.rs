use stt_tauri_client::formatting::TextFormatter;
use stt_tauri_client::recognition::{RecognitionResult, RecognitionStatus, RecognitionSubscriber};
use std::sync::{Arc, Mutex};

fn make_partial(text: &str, start: f64, end: f64, utterance_id: i64) -> RecognitionResult {
    RecognitionResult {
        session_id: "s".into(),
        status: RecognitionStatus::Partial,
        text: text.into(),
        start_time: start,
        end_time: end,
        chunk_ids: vec![],
        utterance_id,
        token_confidences: vec![],
    }
}

fn make_final(text: &str, start: f64, end: f64, utterance_id: i64) -> RecognitionResult {
    RecognitionResult {
        session_id: "s".into(),
        status: RecognitionStatus::Final,
        text: text.into(),
        start_time: start,
        end_time: end,
        chunk_ids: vec![],
        utterance_id,
        token_confidences: vec![],
    }
}

#[test]
fn partial_updates_preliminary_text() {
    let formatter = TextFormatter::new(None);
    formatter.on_partial(&make_partial("hello", 0.0, 1.0, 1));
    assert_eq!(formatter.get_preliminary_text(), "hello");
}

#[test]
fn finalization_appends_to_finalized_text() {
    let formatter = TextFormatter::new(None);
    formatter.on_finalization(&make_final("hello", 0.0, 1.0, 1));
    let utterances = formatter.get_utterances();
    assert_eq!(utterances.len(), 1);
    assert_eq!(utterances[0].text, "hello");
}

#[test]
fn multiple_finals_accumulates_utterances() {
    let formatter = TextFormatter::new(None);
    formatter.on_finalization(&make_final("hello", 0.0, 1.0, 1));
    formatter.on_finalization(&make_final("world", 1.0, 2.0, 2));
    let utterances = formatter.get_utterances();
    assert_eq!(utterances.len(), 2);
    assert_eq!(utterances[0].text, "hello");
    assert_eq!(utterances[1].text, "world");
}

#[test]
fn duplicate_suppressed_same_text_and_utterance_id() {
    let formatter = TextFormatter::new(None);
    formatter.on_finalization(&make_final("hello", 0.0, 1.0, 1));
    formatter.on_finalization(&make_final("hello", 0.0, 1.0, 1));
    assert_eq!(formatter.get_utterances().len(), 1);
}

#[test]
fn non_duplicate_with_different_utterance_id() {
    let formatter = TextFormatter::new(None);
    formatter.on_finalization(&make_final("hello", 0.0, 1.0, 1));
    formatter.on_finalization(&make_final("hello", 1.0, 2.0, 2));
    assert_eq!(formatter.get_utterances().len(), 2);
}

#[test]
fn clear_resets_all_state() {
    let formatter = TextFormatter::new(None);
    formatter.on_finalization(&make_final("hello", 0.0, 1.0, 1));
    formatter.on_partial(&make_partial("world", 1.0, 2.0, 2));
    formatter.clear();
    assert!(formatter.get_utterances().is_empty());
    assert_eq!(formatter.get_preliminary_text(), "");
}

#[test]
fn observer_callback_called_on_partial() {
    let call_count = Arc::new(Mutex::new(0u32));
    let count_clone = Arc::clone(&call_count);
    let callback = move || {
        *count_clone.lock().unwrap() += 1;
    };

    let formatter = TextFormatter::new(Some(Box::new(callback)));
    formatter.on_partial(&make_partial("hi", 0.0, 0.5, 1));
    assert_eq!(*call_count.lock().unwrap(), 1);
}

#[test]
fn observer_callback_called_on_finalization() {
    let call_count = Arc::new(Mutex::new(0u32));
    let count_clone = Arc::clone(&call_count);
    let callback = move || {
        *count_clone.lock().unwrap() += 1;
    };

    let formatter = TextFormatter::new(Some(Box::new(callback)));
    formatter.on_finalization(&make_final("hi", 0.0, 0.5, 1));
    assert_eq!(*call_count.lock().unwrap(), 1);
}

#[test]
fn clear_triggers_observer_callback() {
    let call_count = Arc::new(Mutex::new(0u32));
    let count_clone = Arc::clone(&call_count);
    let callback = move || {
        *count_clone.lock().unwrap() += 1;
    };

    let formatter = TextFormatter::new(Some(Box::new(callback)));
    formatter.on_finalization(&make_final("hello", 0.0, 1.0, 1));
    let before = *call_count.lock().unwrap();
    formatter.clear();
    assert_eq!(*call_count.lock().unwrap(), before + 1);
}

#[test]
fn partial_after_finalization_shows_new_partial_text() {
    let formatter = TextFormatter::new(None);
    formatter.on_finalization(&make_final("done", 0.0, 1.0, 1));
    formatter.on_partial(&make_partial("next", 1.0, 2.0, 2));
    assert_eq!(formatter.get_preliminary_text(), "next");
    assert_eq!(formatter.get_utterances().len(), 1);
    assert_eq!(formatter.get_utterances()[0].text, "done");
}

#[test]
fn finalization_stores_timestamps() {
    let formatter = TextFormatter::new(None);
    formatter.on_finalization(&make_final("hello", 1.0, 2.5, 1));
    let utterances = formatter.get_utterances();
    assert_eq!(utterances.len(), 1);
    assert_eq!(utterances[0].start_time, 1.0);
    assert_eq!(utterances[0].end_time, 2.5);
}
