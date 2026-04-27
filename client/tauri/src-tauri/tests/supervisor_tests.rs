use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use stt_tauri_client::server_diagnostics::ProcessOutputStream;
use stt_tauri_client::supervisor::{
    find_repo_root, parse_server_url_line, resolve_command_spec, CommandOverrides, CrashDecision,
    ServerProcessEvent, ServerSupervisor, ServerSupervisorConfig, ServerSupervisorError,
};

fn workspace_python() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("workspace root")
        .join("venv")
        .join("Scripts")
        .join("python.exe")
}

fn make_python_supervisor(script: &Path) -> ServerSupervisor {
    let parent = script.parent().unwrap();
    let mut config = ServerSupervisorConfig::new(parent, parent);
    config.overrides.python_executable = Some(workspace_python());
    config.overrides.entry_point = Some(script.to_path_buf());
    config.overrides.current_dir = Some(parent.to_path_buf());
    config.startup_timeout = Duration::from_millis(1500);
    config.stability_reset = Duration::from_millis(10);
    ServerSupervisor::new(config)
}

fn write_dev_layout(repo: &Path) {
    let python = repo.join("venv").join("Scripts").join("python.exe");
    std::fs::create_dir_all(python.parent().unwrap()).unwrap();
    std::fs::write(&python, "").unwrap();
    std::fs::write(repo.join("main.py"), "").unwrap();
}

fn write_portable_layout(exe_dir: &Path) {
    let python = exe_dir.join("_internal").join("runtime").join("python.exe");
    let entry = exe_dir.join("_internal").join("app").join("main.pyc");
    std::fs::create_dir_all(python.parent().unwrap()).unwrap();
    std::fs::create_dir_all(entry.parent().unwrap()).unwrap();
    std::fs::write(&python, "").unwrap();
    std::fs::write(&entry, "").unwrap();
}

#[test]
fn dev_fallback_resolves_python_and_main_when_only_dev_layout_is_present() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path();
    write_dev_layout(repo);
    let empty_exe_dir = tempfile::tempdir().unwrap();

    let spec = resolve_command_spec(
        empty_exe_dir.path(),
        repo,
        &CommandOverrides::default(),
    )
    .unwrap();

    assert_eq!(
        spec.program,
        repo.join("venv").join("Scripts").join("python.exe")
    );
    assert_eq!(spec.current_dir, repo);
    assert_eq!(spec.args[0], repo.join("main.py").to_string_lossy());
    assert!(spec.args.iter().any(|arg| arg == "--host=127.0.0.1"));
}

#[test]
fn portable_layout_resolves_when_files_are_present_next_to_exe() {
    let exe_dir = tempfile::tempdir().unwrap();
    write_portable_layout(exe_dir.path());
    let empty_dev = tempfile::tempdir().unwrap();

    let spec = resolve_command_spec(
        exe_dir.path(),
        empty_dev.path(),
        &CommandOverrides::default(),
    )
    .unwrap();

    assert_eq!(
        spec.program,
        exe_dir
            .path()
            .join("_internal")
            .join("runtime")
            .join("python.exe")
    );
    assert_eq!(spec.current_dir, exe_dir.path());
    assert_eq!(
        spec.args[0],
        exe_dir
            .path()
            .join("_internal")
            .join("app")
            .join("main.pyc")
            .to_string_lossy()
    );
    assert!(spec.args.iter().any(|arg| arg == "--host=127.0.0.1"));
}

#[test]
fn portable_wins_over_dev_when_both_layouts_are_valid() {
    let exe_dir = tempfile::tempdir().unwrap();
    write_portable_layout(exe_dir.path());
    let repo = tempfile::tempdir().unwrap();
    write_dev_layout(repo.path());

    let spec = resolve_command_spec(
        exe_dir.path(),
        repo.path(),
        &CommandOverrides::default(),
    )
    .unwrap();

    assert_eq!(spec.current_dir, exe_dir.path());
    assert_eq!(
        spec.program,
        exe_dir
            .path()
            .join("_internal")
            .join("runtime")
            .join("python.exe")
    );
}

#[test]
fn missing_runtime_returns_structured_error_with_both_candidates() {
    let exe_dir = tempfile::tempdir().unwrap();
    let dev = tempfile::tempdir().unwrap();

    let err =
        resolve_command_spec(exe_dir.path(), dev.path(), &CommandOverrides::default()).unwrap_err();

    match err {
        ServerSupervisorError::MissingServerRuntime {
            portable_python,
            portable_entry,
            dev_candidate,
        } => {
            assert_eq!(
                portable_python,
                exe_dir
                    .path()
                    .join("_internal")
                    .join("runtime")
                    .join("python.exe")
            );
            assert_eq!(
                portable_entry,
                exe_dir
                    .path()
                    .join("_internal")
                    .join("app")
                    .join("main.pyc")
            );
            assert_eq!(dev_candidate.as_deref(), Some(dev.path()));
        }
        other => panic!("expected MissingServerRuntime, got {other:?}"),
    }
}

#[test]
fn overrides_take_precedence_over_portable_layout() {
    let exe_dir = tempfile::tempdir().unwrap();
    write_portable_layout(exe_dir.path());
    let override_root = tempfile::tempdir().unwrap();
    let override_python = override_root.path().join("py.exe");
    let override_entry = override_root.path().join("entry.py");
    std::fs::write(&override_python, "").unwrap();
    std::fs::write(&override_entry, "").unwrap();

    let mut overrides = CommandOverrides::default();
    overrides.python_executable = Some(override_python.clone());
    overrides.entry_point = Some(override_entry.clone());
    overrides.current_dir = Some(override_root.path().to_path_buf());

    let spec = resolve_command_spec(exe_dir.path(), override_root.path(), &overrides).unwrap();

    assert_eq!(spec.program, override_python);
    assert_eq!(spec.args[0], override_entry.to_string_lossy());
    assert_eq!(spec.current_dir, override_root.path());
}

#[test]
fn override_with_missing_python_returns_missing_python_error() {
    let exe_dir = tempfile::tempdir().unwrap();
    let dev = tempfile::tempdir().unwrap();
    write_dev_layout(dev.path());

    let mut overrides = CommandOverrides::default();
    overrides.python_executable = Some(exe_dir.path().join("does-not-exist.exe"));

    let err = resolve_command_spec(exe_dir.path(), dev.path(), &overrides).unwrap_err();

    assert!(matches!(err, ServerSupervisorError::MissingPython(_)));
}

#[test]
fn override_with_missing_entrypoint_returns_missing_entry_error() {
    let exe_dir = tempfile::tempdir().unwrap();
    let dev = tempfile::tempdir().unwrap();
    write_dev_layout(dev.path());

    let mut overrides = CommandOverrides::default();
    overrides.entry_point = Some(exe_dir.path().join("missing.py"));

    let err = resolve_command_spec(exe_dir.path(), dev.path(), &overrides).unwrap_err();

    assert!(matches!(err, ServerSupervisorError::MissingEntryPoint(_)));
}

#[test]
fn find_repo_root_walks_up_to_development_layout() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path();
    let nested = repo.join("client").join("tauri").join("src-tauri");
    std::fs::create_dir_all(&nested).unwrap();
    write_dev_layout(repo);

    assert_eq!(find_repo_root(&nested).as_deref(), Some(repo));
}

#[test]
fn parses_server_url_from_stdout_line() {
    let line = "Server listening on ws://127.0.0.1:52431";

    assert_eq!(
        parse_server_url_line(line).as_deref(),
        Some("ws://127.0.0.1:52431")
    );
}

#[test]
fn parses_server_url_with_extra_output_after_url() {
    let line = "INFO Server listening on ws://localhost:9000 qr follows";

    assert_eq!(
        parse_server_url_line(line).as_deref(),
        Some("ws://localhost:9000")
    );
}

#[test]
fn ignores_lines_without_websocket_url() {
    assert_eq!(
        parse_server_url_line("Server listening on http://bad"),
        None
    );
    assert_eq!(parse_server_url_line("QR code line"), None);
}

#[tokio::test]
async fn first_unstable_failure_requests_restart() {
    let supervisor = ServerSupervisor::new(ServerSupervisorConfig::new("unused", "unused"));

    assert_eq!(
        supervisor.record_unstable_failure().await,
        CrashDecision::Restart
    );
}

#[tokio::test]
async fn second_unstable_failure_is_terminal() {
    let supervisor = ServerSupervisor::new(ServerSupervisorConfig::new("unused", "unused"));

    assert_eq!(
        supervisor.record_unstable_failure().await,
        CrashDecision::Restart
    );
    assert_eq!(
        supervisor.record_unstable_failure().await,
        CrashDecision::Terminal
    );
}

#[tokio::test]
async fn crash_counter_resets_after_stable_connected_runtime() {
    let mut config = ServerSupervisorConfig::new("unused", "unused");
    config.stability_reset = Duration::from_millis(10);
    let supervisor = ServerSupervisor::new(config);

    assert_eq!(
        supervisor.record_unstable_failure().await,
        CrashDecision::Restart
    );
    supervisor.mark_connected().await;
    tokio::time::sleep(Duration::from_millis(30)).await;

    assert_eq!(
        supervisor.record_unstable_failure().await,
        CrashDecision::Restart
    );
}

#[tokio::test]
async fn crash_counter_does_not_reset_before_mark_connected() {
    let mut config = ServerSupervisorConfig::new("unused", "unused");
    config.stability_reset = Duration::from_millis(10);
    let supervisor = ServerSupervisor::new(config);

    assert_eq!(
        supervisor.record_unstable_failure().await,
        CrashDecision::Restart
    );
    tokio::time::sleep(Duration::from_millis(30)).await;

    assert_eq!(
        supervisor.record_unstable_failure().await,
        CrashDecision::Terminal
    );
}

#[tokio::test]
async fn startup_times_out_and_terminates_child_when_url_is_missing() {
    if !workspace_python().exists() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let script = temp.path().join("main.py");
    std::fs::write(&script, "import time\nwhile True:\n    time.sleep(0.1)\n").unwrap();
    let supervisor = make_python_supervisor(&script);

    let err = supervisor.start().await.unwrap_err();

    assert!(matches!(err, ServerSupervisorError::StartupTimeout));
    assert!(!supervisor.has_running_child().await);
}

#[tokio::test]
async fn shutdown_during_startup_cancels_wait_and_terminates_child() {
    if !workspace_python().exists() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let script = temp.path().join("main.py");
    std::fs::write(&script, "import time\nwhile True:\n    time.sleep(0.1)\n").unwrap();
    let supervisor = Arc::new(make_python_supervisor(&script));
    let starter = Arc::clone(&supervisor);

    let start_task = tokio::spawn(async move { starter.start().await });
    tokio::time::sleep(Duration::from_millis(30)).await;
    supervisor.shutdown().await;
    let err = start_task.await.unwrap().unwrap_err();

    assert!(matches!(err, ServerSupervisorError::StartupCancelled));
    assert!(!supervisor.has_running_child().await);
}

#[tokio::test]
async fn startup_exit_before_url_reports_status_and_recent_stderr() {
    if !workspace_python().exists() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let script = temp.path().join("main.py");
    std::fs::write(
        &script,
        "import sys\nprint('startup exploded', file=sys.stderr, flush=True)\nsys.exit(13)\n",
    )
    .unwrap();
    let supervisor = make_python_supervisor(&script);

    let err = supervisor.start().await.unwrap_err();
    let message = err.to_string();

    match err {
        ServerSupervisorError::StartupExited {
            status,
            diagnostics,
        } => {
            assert_eq!(status.and_then(|s| s.code()), Some(13));
            assert!(diagnostics.lines.iter().any(|line| {
                line.stream == ProcessOutputStream::Stderr && line.text == "startup exploded"
            }));
            assert!(message.contains("startup"));
            assert!(message.contains("exit status"));
            assert!(message.contains("[stderr] startup exploded"));
        }
        other => panic!("expected StartupExited, got {other:?}"),
    }
    assert!(!supervisor.has_running_child().await);
}

#[tokio::test]
async fn stdout_continues_draining_after_url_line() {
    if !workspace_python().exists() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let script = temp.path().join("main.py");
    let marker = temp.path().join("drained.txt");
    let marker_literal = marker.to_string_lossy().replace('\\', "\\\\");
    std::fs::write(
        &script,
        format!(
             "import pathlib, sys, time\n\
             print('Server listening on ws://127.0.0.1:12345', flush=True)\n\
             for i in range(200):\n\
             \tprint('line', i, flush=True)\n\
             pathlib.Path(r'{marker_literal}').write_text('done')\n\
             time.sleep(0.05)\n"
        ),
    )
    .unwrap();
    let supervisor = make_python_supervisor(&script);

    let url = supervisor.start().await.unwrap();
    for _ in 0..40 {
        if marker.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    supervisor.shutdown().await;

    assert_eq!(url, "ws://127.0.0.1:12345");
    assert!(
        marker.exists(),
        "child should finish writing after URL line"
    );
}

#[tokio::test]
async fn runtime_exit_event_reports_status_and_recent_output() {
    if !workspace_python().exists() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let script = temp.path().join("main.py");
    std::fs::write(
        &script,
        "import sys, time\n\
         print('Server listening on ws://127.0.0.1:12345', flush=True)\n\
         print('runtime stdout', flush=True)\n\
         print('runtime stderr', file=sys.stderr, flush=True)\n\
         time.sleep(0.05)\n\
         sys.exit(7)\n",
    )
    .unwrap();
    let supervisor = make_python_supervisor(&script);

    assert_eq!(supervisor.start().await.unwrap(), "ws://127.0.0.1:12345");
    let event = tokio::time::timeout(Duration::from_secs(3), supervisor.next_event())
        .await
        .unwrap()
        .unwrap();

    match event {
        ServerProcessEvent::Exited {
            status,
            expected,
            diagnostics,
        } => {
            assert!(!expected);
            assert_eq!(status.and_then(|s| s.code()), Some(7));
            assert!(diagnostics.lines.iter().any(|line| {
                line.stream == ProcessOutputStream::Stdout && line.text == "runtime stdout"
            }));
            assert!(diagnostics.lines.iter().any(|line| {
                line.stream == ProcessOutputStream::Stderr && line.text == "runtime stderr"
            }));
        }
    }
}

#[tokio::test]
async fn restarted_process_uses_fresh_diagnostics_buffer() {
    if !workspace_python().exists() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let script = temp.path().join("main.py");
    let supervisor = make_python_supervisor(&script);

    std::fs::write(
        &script,
        "import sys\n\
         print('Server listening on ws://127.0.0.1:12345', flush=True)\n\
         print('first generation output', flush=True)\n\
         sys.exit(3)\n",
    )
    .unwrap();
    assert_eq!(supervisor.start().await.unwrap(), "ws://127.0.0.1:12345");
    let _ = tokio::time::timeout(Duration::from_secs(3), supervisor.next_event())
        .await
        .unwrap()
        .unwrap();

    std::fs::write(
        &script,
        "import sys\n\
         print('Server listening on ws://127.0.0.1:12346', flush=True)\n\
         print('second generation output', flush=True)\n\
         sys.exit(4)\n",
    )
    .unwrap();
    assert_eq!(supervisor.start().await.unwrap(), "ws://127.0.0.1:12346");
    let event = tokio::time::timeout(Duration::from_secs(3), supervisor.next_event())
        .await
        .unwrap()
        .unwrap();

    match event {
        ServerProcessEvent::Exited { diagnostics, .. } => {
            assert!(diagnostics
                .lines
                .iter()
                .any(|line| line.text == "second generation output"));
            assert!(!diagnostics
                .lines
                .iter()
                .any(|line| line.text == "first generation output"));
        }
    }
}
