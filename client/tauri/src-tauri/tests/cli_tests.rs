use stt_tauri_client::cli::CliArgs;
use clap::Parser;

#[test]
fn parse_with_server_url_and_input_file() {
    let args = CliArgs::try_parse_from([
        "stt-client",
        "--server-url=ws://localhost:8765",
        "--input-file=test.wav",
    ])
    .unwrap();

    assert_eq!(args.server_url.as_deref(), Some("ws://localhost:8765"));
    assert_eq!(args.input_file.as_deref(), Some("test.wav"));
}

#[test]
fn parse_with_server_url_only() {
    let args = CliArgs::try_parse_from([
        "stt-client",
        "--server-url=ws://192.168.1.10:9090",
    ])
    .unwrap();

    assert_eq!(args.server_url.as_deref(), Some("ws://192.168.1.10:9090"));
    assert_eq!(args.input_file, None);
}

#[test]
fn missing_server_url_yields_none() {
    let args = CliArgs::try_parse_from(["stt-client"]).unwrap();
    assert_eq!(args.server_url, None);
}

#[test]
fn verify_parsed_values() {
    let args = CliArgs::try_parse_from([
        "stt-client",
        "--server-url=ws://example.com:1234",
        "--input-file=/path/to/audio.wav",
    ])
    .unwrap();

    assert_eq!(args.server_url.as_deref(), Some("ws://example.com:1234"));
    assert_eq!(args.input_file.as_deref(), Some("/path/to/audio.wav"));
}

#[test]
fn parse_with_space_separated_values() {
    let args = CliArgs::try_parse_from([
        "stt-client",
        "--server-url",
        "ws://host:8080",
        "--input-file",
        "recording.wav",
    ])
    .unwrap();

    assert_eq!(args.server_url.as_deref(), Some("ws://host:8080"));
    assert_eq!(args.input_file.as_deref(), Some("recording.wav"));
}
