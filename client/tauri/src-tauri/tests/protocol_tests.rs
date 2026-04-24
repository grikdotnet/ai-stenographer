use stt_tauri_client::protocol::{AudioFrameEncoder, ServerMessage, ServerMessageDecoder};
use stt_tauri_client::recognition::{RecognitionResult, RecognitionStatus};

// ---------------------------------------------------------------------------
// AudioFrameEncoder
// ---------------------------------------------------------------------------

#[test]
fn encode_audio_frame_produces_correct_binary_layout() {
    let audio: Vec<f32> = vec![0.5, -0.25, 1.0];
    let encoded = AudioFrameEncoder::encode("sess-1", 42, 1.5, &audio);

    let header_len = u32::from_le_bytes(encoded[0..4].try_into().unwrap()) as usize;
    assert!(header_len > 0);

    let header_json: serde_json::Value =
        serde_json::from_slice(&encoded[4..4 + header_len]).unwrap();
    assert_eq!(header_json["type"], "audio_chunk");
    assert_eq!(header_json["session_id"], "sess-1");
    assert_eq!(header_json["chunk_id"], 42);
    assert!((header_json["timestamp"].as_f64().unwrap() - 1.5).abs() < 1e-9);

    let pcm_bytes = &encoded[4 + header_len..];
    assert_eq!(pcm_bytes.len(), 3 * std::mem::size_of::<f32>());

    let mut samples = Vec::new();
    for chunk in pcm_bytes.chunks_exact(4) {
        samples.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    assert_eq!(samples, audio);
}

#[test]
fn encode_empty_audio_produces_header_only() {
    let encoded = AudioFrameEncoder::encode("s", 0, 0.0, &[]);

    let header_len = u32::from_le_bytes(encoded[0..4].try_into().unwrap()) as usize;
    assert_eq!(encoded.len(), 4 + header_len);
}

#[test]
fn encode_roundtrip_header_is_valid_json() {
    let encoded = AudioFrameEncoder::encode("abc", 7, 99.9, &[0.1, 0.2]);

    let header_len = u32::from_le_bytes(encoded[0..4].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&encoded[4..4 + header_len]).unwrap();

    assert_eq!(header["type"], "audio_chunk");
    assert_eq!(header["session_id"], "abc");
    assert_eq!(header["chunk_id"], 7);
}

// ---------------------------------------------------------------------------
// ServerMessageDecoder — valid messages
// ---------------------------------------------------------------------------

#[test]
fn decode_session_created() {
    let json = r#"{"type":"session_created","session_id":"s1","protocol_version":"v1","server_time":123.4}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    match msg {
        ServerMessage::SessionCreated {
            session_id,
            protocol_version,
            server_time,
        } => {
            assert_eq!(session_id, "s1");
            assert_eq!(protocol_version, "v1");
            assert!((server_time - 123.4).abs() < 1e-9);
        }
        other => panic!("expected SessionCreated, got {:?}", other),
    }
}

#[test]
fn decode_recognition_result_partial() {
    let json = r#"{
        "type": "recognition_result",
        "session_id": "s2",
        "status": "partial",
        "text": "hello",
        "start_time": 1.0,
        "end_time": 2.0
    }"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    match msg {
        ServerMessage::RecognitionResult {
            status, text, chunk_ids, utterance_id, token_confidences, ..
        } => {
            assert_eq!(status, "partial");
            assert_eq!(text, "hello");
            assert!(chunk_ids.is_empty());
            assert_eq!(utterance_id, 0);
            assert!(token_confidences.is_empty());
        }
        other => panic!("expected RecognitionResult, got {:?}", other),
    }
}

#[test]
fn decode_recognition_result_final_with_all_fields() {
    let json = r#"{
        "type": "recognition_result",
        "session_id": "s3",
        "status": "final",
        "text": "hello world",
        "start_time": 0.5,
        "end_time": 3.5,
        "chunk_ids": [1, 2, 3],
        "utterance_id": 7,
        "token_confidences": [0.9, 0.85]
    }"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    match msg {
        ServerMessage::RecognitionResult {
            status, text, chunk_ids, utterance_id, token_confidences, ..
        } => {
            assert_eq!(status, "final");
            assert_eq!(text, "hello world");
            assert_eq!(chunk_ids, vec![1, 2, 3]);
            assert_eq!(utterance_id, 7);
            assert_eq!(token_confidences, vec![0.9, 0.85]);
        }
        other => panic!("expected RecognitionResult, got {:?}", other),
    }
}

#[test]
fn decode_session_closed() {
    let json = r#"{"type":"session_closed","session_id":"s4","reason":"close_session","message":"bye"}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    match msg {
        ServerMessage::SessionClosed {
            session_id,
            reason,
            message,
        } => {
            assert_eq!(session_id, "s4");
            assert_eq!(reason, "close_session");
            assert_eq!(message, Some("bye".to_string()));
        }
        other => panic!("expected SessionClosed, got {:?}", other),
    }
}

#[test]
fn decode_session_closed_without_message() {
    let json = r#"{"type":"session_closed","session_id":"s5","reason":"timeout"}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    match msg {
        ServerMessage::SessionClosed { message, .. } => {
            assert_eq!(message, None);
        }
        other => panic!("expected SessionClosed, got {:?}", other),
    }
}

#[test]
fn decode_error_message() {
    let json = r#"{
        "type": "error",
        "session_id": "s6",
        "error_code": "RATE_LIMIT",
        "message": "slow down",
        "fatal": true,
        "request_id": "req-error-1"
    }"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    match msg {
        ServerMessage::Error {
            error_code,
            message,
            fatal,
            request_id,
            ..
        } => {
            assert_eq!(error_code, "RATE_LIMIT");
            assert_eq!(message, "slow down");
            assert!(fatal);
            assert_eq!(request_id.as_deref(), Some("req-error-1"));
        }
        other => panic!("expected Error, got {:?}", other),
    }
}

#[test]
fn decode_error_message_fatal_defaults_to_false() {
    let json = r#"{"type":"error","session_id":"s7","error_code":"E1","message":"oops"}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    match msg {
        ServerMessage::Error {
            fatal, request_id, ..
        } => {
            assert!(!fatal);
            assert_eq!(request_id, None);
        }
        other => panic!("expected Error, got {:?}", other),
    }
}

#[test]
fn decode_ping_message() {
    let json = r#"{"type":"ping"}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    assert!(matches!(msg, ServerMessage::Ping {}));
}

#[test]
fn decode_server_state_message() {
    let json = r#"{"type":"server_state","state":"waiting_for_model"}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    match msg {
        ServerMessage::ServerState { state } => assert_eq!(state, "waiting_for_model"),
        other => panic!("expected ServerState, got {:?}", other),
    }
}

#[test]
fn decode_model_list_message() {
    let json = r#"{
        "type":"model_list",
        "models":[{"name":"parakeet","display_name":"Parakeet","size_description":"1.25 GB","status":"missing"}],
        "request_id":"req-1"
    }"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    match msg {
        ServerMessage::ModelList { models, request_id } => {
            assert_eq!(models.len(), 1);
            assert_eq!(models[0].name, "parakeet");
            assert_eq!(request_id.as_deref(), Some("req-1"));
        }
        other => panic!("expected ModelList, got {:?}", other),
    }
}

#[test]
fn decode_model_status_message() {
    let json = r#"{"type":"model_status","status":"downloading","request_id":"req-2"}"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    match msg {
        ServerMessage::ModelStatus { status, request_id } => {
            assert_eq!(status, "downloading");
            assert_eq!(request_id.as_deref(), Some("req-2"));
        }
        other => panic!("expected ModelStatus, got {:?}", other),
    }
}

#[test]
fn decode_download_progress_message() {
    let json = r#"{
        "type":"download_progress",
        "model_name":"parakeet",
        "status":"downloading",
        "progress":0.5,
        "downloaded_bytes":512,
        "total_bytes":1024
    }"#;
    let msg = ServerMessageDecoder::decode(json).unwrap();

    match msg {
        ServerMessage::DownloadProgress {
            model_name,
            status,
            progress,
            downloaded_bytes,
            total_bytes,
            ..
        } => {
            assert_eq!(model_name, "parakeet");
            assert_eq!(status, "downloading");
            assert_eq!(progress, Some(0.5));
            assert_eq!(downloaded_bytes, Some(512));
            assert_eq!(total_bytes, Some(1024));
        }
        other => panic!("expected DownloadProgress, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// ServerMessageDecoder — error cases
// ---------------------------------------------------------------------------

#[test]
fn decode_unknown_type_returns_error() {
    let json = r#"{"type":"unknown_stuff","data":42}"#;
    let result = ServerMessageDecoder::decode(json);
    assert!(result.is_err());
}

#[test]
fn decode_invalid_json_returns_error() {
    let result = ServerMessageDecoder::decode("not json {{{");
    assert!(result.is_err());
}

#[test]
fn decode_missing_required_field_returns_error() {
    let json = r#"{"type":"session_created","session_id":"s1"}"#;
    let result = ServerMessageDecoder::decode(json);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// ServerMessage → RecognitionResult conversion
// ---------------------------------------------------------------------------

#[test]
fn recognition_result_conversion_partial() {
    let msg = ServerMessage::RecognitionResult {
        session_id: "s1".into(),
        status: "partial".into(),
        text: "hi".into(),
        start_time: 0.1,
        end_time: 0.5,
        chunk_ids: vec![1],
        utterance_id: 3,
        token_confidences: vec![0.95],
    };

    let result: RecognitionResult = msg.try_into().unwrap();
    assert_eq!(result.status, RecognitionStatus::Partial);
    assert_eq!(result.text, "hi");
    assert_eq!(result.session_id, "s1");
    assert_eq!(result.chunk_ids, vec![1]);
    assert_eq!(result.utterance_id, 3);
    assert_eq!(result.token_confidences, vec![0.95]);
}

#[test]
fn recognition_result_conversion_final() {
    let msg = ServerMessage::RecognitionResult {
        session_id: "s2".into(),
        status: "final".into(),
        text: "done".into(),
        start_time: 1.0,
        end_time: 2.0,
        chunk_ids: vec![],
        utterance_id: 0,
        token_confidences: vec![],
    };

    let result: RecognitionResult = msg.try_into().unwrap();
    assert_eq!(result.status, RecognitionStatus::Final);
}

#[test]
fn recognition_result_conversion_invalid_status() {
    let msg = ServerMessage::RecognitionResult {
        session_id: "s3".into(),
        status: "bogus".into(),
        text: "".into(),
        start_time: 0.0,
        end_time: 0.0,
        chunk_ids: vec![],
        utterance_id: 0,
        token_confidences: vec![],
    };

    let result: Result<RecognitionResult, _> = msg.try_into();
    assert!(result.is_err());
}

#[test]
fn recognition_result_conversion_wrong_variant() {
    let msg = ServerMessage::Ping {};
    let result: Result<RecognitionResult, _> = msg.try_into();
    assert!(result.is_err());
}
