use stt_tauri_client::transport::WsClientTransport;
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Backpressure: bounded channel drops new frames when full
// ---------------------------------------------------------------------------

#[tokio::test]
async fn channel_full_drops_new_frame() {
    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(2);

    tx.try_send(vec![1]).unwrap();
    tx.try_send(vec![2]).unwrap();

    let result = tx.try_send(vec![3]);
    assert!(result.is_err());

    assert_eq!(rx.recv().await.unwrap(), vec![1]);
    assert_eq!(rx.recv().await.unwrap(), vec![2]);
}

// ---------------------------------------------------------------------------
// Shutdown command JSON format
// ---------------------------------------------------------------------------

#[test]
fn shutdown_command_json_has_correct_structure() {
    let json = WsClientTransport::build_shutdown_json("sess-42", 1700000000.0);
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed["type"], "control_command");
    assert_eq!(parsed["command"], "shutdown");
    assert_eq!(parsed["session_id"], "sess-42");
    assert!((parsed["timestamp"].as_f64().unwrap() - 1700000000.0).abs() < 1e-3);
}

// ---------------------------------------------------------------------------
// Transport construction
// ---------------------------------------------------------------------------

#[test]
fn transport_new_stores_url() {
    let t = WsClientTransport::new("ws://localhost:8080".to_string());
    assert_eq!(t.url(), "ws://localhost:8080");
}

#[tokio::test]
async fn stop_without_connection_returns_promptly() {
    let transport = WsClientTransport::new("ws://localhost:8080".to_string());

    let result = tokio::time::timeout(
        tokio::time::Duration::from_millis(100),
        transport.stop(false),
    )
    .await;

    assert!(result.is_ok(), "stop() should not block when never connected");
}
