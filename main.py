# main.py
import sys
import logging
from pathlib import Path
from src.PathResolver import PathResolver
from src.asr.ModelManager import ModelManager
from src.LoggingSetup import setup_logging


# ============================================================================
# RESOLVE PATHS AT MODULE LOAD TIME
# ============================================================================
if hasattr(sys.modules['__main__'], '__file__'):
    SCRIPT_PATH = Path(sys.modules['__main__'].__file__).resolve()
else:
    SCRIPT_PATH = Path(__file__).resolve()

path_resolver = PathResolver(SCRIPT_PATH)
PATHS = path_resolver.paths

APP_DIR = PATHS.app_dir
ROOT_DIR = PATHS.root_dir
MODELS_DIR = PATHS.models_dir
CONFIG_DIR = PATHS.config_dir
LOGS_DIR = PATHS.logs_dir

path_resolver.ensure_local_dir_structure()


def _spawn_client(server_url: str) -> "subprocess.Popen":
    """Spawn client.py as a subprocess with the given server WebSocket URL.

    Args:
        server_url: WebSocket URL passed as --server-url= argument.

    Returns:
        Running subprocess handle.
    """
    import subprocess
    client_script = Path(__file__).parent / "client.py"
    cmd = [sys.executable, str(client_script), f"--server-url={server_url}"]
    return subprocess.Popen(cmd)


def _make_watcher(server_app, proc) -> "threading.Thread":
    """Create and start a daemon thread that stops the server when the client exits.

    Args:
        server_app: ServerApp instance to stop on client exit.
        proc: subprocess.Popen handle to watch.

    Returns:
        Started daemon thread.
    """
    import threading

    def _watch() -> None:
        proc.wait()
        server_app.stop()
        proc.terminate()

    thread = threading.Thread(target=_watch, daemon=True, name="ClientWatcher")
    thread.start()
    return thread


def _main(argv: list[str], models_dir: Path, logs_dir: Path, config_path: str) -> None:
    """Core entry-point logic, extracted for testability.

    Algorithm:
        1. Parse flags from argv.
        2. Setup logging.
        3. Check for missing models; exit 1 with stderr message if any.
        4. Load ONNX model and create Recognizer.
        5. Create and start ServerApp.
        6. In --server-only mode: block on WsServer.join().
           In default mode: spawn client.py subprocess, start watcher thread,
           block on subprocess exit.

    Args:
        argv: Command-line arguments (typically sys.argv).
        models_dir: Path to models directory.
        logs_dir: Path to logs directory.
        config_path: Path to stt_config.json.
    """
    import json
    import queue as _queue
    import onnxruntime as rt
    import onnx_asr
    from src.asr.ExecutionProviderManager import ExecutionProviderManager
    from src.asr.SessionOptionsFactory import SessionOptionsFactory
    from src.asr.Recognizer import Recognizer
    from src.ServerApplicationState import ServerApplicationState
    from src.server.ServerApp import ServerApp

    verbose = "-v" in argv
    server_only = "--server-only" in argv

    is_frozen = getattr(sys, 'frozen', False)
    setup_logging(logs_dir, verbose=verbose, is_frozen=is_frozen)

    missing_models = ModelManager.get_missing_models(models_dir)
    if missing_models:
        print(
            f"Missing models: {', '.join(missing_models)}. "
            "Provision them before starting.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    exec_mgr = ExecutionProviderManager(config)
    providers = exec_mgr.build_provider_list()

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    gpu_type = exec_mgr.detect_gpu_type()
    factory = SessionOptionsFactory(config)
    factory.get_strategy(gpu_type).configure_session_options(sess_options)

    base_model = onnx_asr.load_model(
        "nemo-parakeet-tdt-0.6b-v3",
        str(models_dir / "parakeet"),
        quantization='fp16',
        providers=providers,
        sess_options=sess_options,
    )
    model = base_model.with_timestamps()
    logging.info("Model loaded.")

    recognizer_state = ServerApplicationState()
    recognizer = Recognizer(
        model=model,
        input_queue=_queue.Queue(),
        output_queue=_queue.Queue(),
        sample_rate=config['audio']['sample_rate'],
        app_state=recognizer_state,
        verbose=verbose,
    )

    vad_model_path = models_dir / "silero_vad" / "silero_vad.onnx"

    server_app = ServerApp(
        recognizer=recognizer,
        config=config,
        vad_model_path=vad_model_path,
        host="127.0.0.1",
        port=0,
    )
    server_app.start()

    if server_only:
        logging.info("Running in --server-only mode. Press Ctrl+C to stop.")
        server_app._ws_server.join()
    else:
        server_url = f"ws://127.0.0.1:{server_app.port}"
        proc = _spawn_client(server_url)
        _make_watcher(server_app, proc)
        proc.wait()


if __name__ == "__main__":
    server_app = None
    try:
        _main(
            argv=sys.argv,
            models_dir=MODELS_DIR,
            logs_dir=LOGS_DIR,
            config_path=str(path_resolver.get_config_path("stt_config.json")),
        )
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        logging.error(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)
