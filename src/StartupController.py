"""Application startup orchestration."""

import json
import logging
import queue as _queue
import subprocess
import sys
from pathlib import Path
from typing import Callable

import onnxruntime as rt

from src.ApplicationState import ApplicationState
from src.downloader.ModelDownloader import ModelDownloader
from src.downloader.ModelDownloadCliDialog import ModelDownloadCliDialog
from src.asr.ExecutionProviderManager import ExecutionProviderManager
from src.asr.ModelLoader import load_model as load_asr_model
from src.asr.ModelManager import ModelManager
from src.asr.Recognizer import Recognizer
from src.asr.SessionOptionsFactory import SessionOptionsFactory
from src.LoggingSetup import setup_logging
from src.PathResolver import PathResolver
from src.server.qr_display import print_qr_code
from src.server.ServerApp import ServerApp
from src.StartupArgs import StartupArgs


class TauriBinaryNotFoundError(RuntimeError):
    """Raised when the Tauri client binary is absent in default (non-server-only) mode."""


class MissingModelsError(RuntimeError):
    """Raised when required ASR models are absent and cannot be downloaded."""


class DownloadCancelledError(Exception):
    """Raised when model provisioning is declined or cancelled by the user."""


class StartupController:
    """Orchestrates the full application startup sequence.

    Encapsulates the ordered phases: environment validation, logging setup,
    model availability, model loading, server creation, and lifecycle management.
    Designed to be constructed once and run once.

    Args:
        args: Parsed and validated CLI arguments.
        path_resolver: Resolved filesystem paths for current distribution mode.
        model_manager: ModelManager instance used to check for missing models.
    """

    def __init__(
        self,
        args: StartupArgs,
        path_resolver: PathResolver,
        model_manager: ModelManager,
    ) -> None:
        self._args = args
        self._path_resolver = path_resolver
        self._paths = path_resolver.paths
        self._model_manager = model_manager
        self._model_downloader = ModelDownloader(self._paths.models_dir)

    def run(self) -> None:
        """Execute the full startup sequence.

        Algorithm:
            1. Validate environment.
            2. Set up logging.
            3. Ensure the Parakeet model is available.
            4. Load config.
            5. Create and start ServerApp.
            6. Create Recognizer from config.
            7. Attach recognizer to server.
            8. In default mode, spawn Tauri client.
            9. Block on server lifecycle; stop server in finally.
        """
        self._validate_environment()
        self._setup_logging()

        # Startup owns the user dialog for missing models.
        if not self._model_manager.model_exists():
            if self._args.server_only:
                if self._args.download_model:
                    self._download_server_only_model()
                # Only prompt when startup is attached to an interactive console.
                elif sys.stdin.isatty() and sys.stdout.isatty():
                    if not ModelDownloadCliDialog(["parakeet"]).confirm_download():
                        raise DownloadCancelledError("Model download was cancelled.")
                    self._download_server_only_model()
                else:
                    print(
                        "Missing models: parakeet. Non-interactive mode cannot prompt. "
                        "Re-run with --server-only --download-model to provision automatically.",
                        file=sys.stderr,
                    )
                    raise MissingModelsError("Missing models: parakeet")
            else:
                self._download_missing_models_via_client()
            if not self._model_manager.model_exists():
                raise MissingModelsError("Download failed. model still missing: parakeet")

        config_path = self._path_resolver.get_config_path("server_config.json")
        with open(config_path) as f:
            config = json.load(f)

        server_app = self._create_and_start_server(config)
        try:
            recognizer = self._create_recognizer(config, server_app.app_state)
            server_app.attach_recognizer(recognizer)
            if not self._args.server_only:
                self._spawn_tauri_client(f"ws://127.0.0.1:{server_app.port}")
            self._run_lifecycle(server_app)
        except KeyboardInterrupt:
            logging.info("Interrupted by user.")
        finally:
            server_app.stop()

    def _download_server_only_model(self) -> None:
        """Download the required server-only model in the current process.

        Raises:
            DownloadCancelledError: If the user interrupts the download.
            MissingModelsError: If the download fails.
        """
        try:
            self._model_downloader.download_parakeet(self._make_cli_progress_reporter())
        except KeyboardInterrupt as exc:
            print(file=sys.stderr)
            self._model_downloader.cleanup_partial_files(self._paths.models_dir)
            raise DownloadCancelledError("Model download was cancelled.") from exc
        except Exception as exc:
            self._model_downloader.cleanup_partial_files(self._paths.models_dir)
            raise MissingModelsError(f"Automatic download failed: {exc}") from exc

    def _download_missing_models_via_client(self) -> None:
        """Launch the download GUI client and wait for it to finish.

        Raises:
            DownloadCancelledError: If the download GUI exits with non-zero code.
        """
        proc = self._spawn_client_for_download()
        proc.wait()
        if proc.returncode != 0:
            raise DownloadCancelledError("Model was not downloaded.")

    def _validate_environment(self) -> None:
        """Check Tauri binary exists in default (non-server-only) mode.

        Raises TauriBinaryNotFoundError if the binary is missing.
        No-op when args.server_only is True.
        """
        if self._args.server_only:
            return
        binary = self._tauri_binary_path()
        if not binary.exists():
            port_hint = self._args.port or ""
            print(
                f"Error: Tauri client binary not found at:\n {binary}\n\n"
                "Build it first with:\n"
                "  cd src/client/tauri && npm run tauri:build\n\n"
                "Or run in server-only mode:\n"
                f"  python main.py --server-only [--port={port_hint}]",
                file=sys.stderr,
            )
            raise TauriBinaryNotFoundError(f"Tauri binary not found: {binary}")

    def _setup_logging(self) -> None:
        """Configure logging based on args.verbose and frozen status."""
        is_frozen = getattr(sys, "frozen", False)
        setup_logging(self._paths.logs_dir, verbose=self._args.verbose, is_frozen=is_frozen)

    def _create_and_start_server(self, config: dict) -> ServerApp:
        """Create and start ServerApp without a recognizer.

        Args:
            config: Parsed server_config.json dict.

        Returns:
            A started ServerApp instance.
        """
        vad_model_path = self._paths.models_dir / "silero_vad" / "silero_vad.onnx"
        app_state = ApplicationState()
        server_app = ServerApp(
            config=config,
            vad_model_path=vad_model_path,
            host="127.0.0.1",
            port=self._args.port,
            app_state=app_state,
        )
        server_app.start()
        return server_app

    def _create_recognizer(self, config: dict, app_state: ApplicationState) -> Recognizer:
        """Build the ONNX session and create a Recognizer.

        Args:
            config: Parsed server_config.json dict.
            app_state: Server ApplicationState to pass to Recognizer.

        Returns:
            A constructed (not started) Recognizer instance.

        Raises:
            ModelLoadError: If the ASR model file cannot be loaded.
        """
        exec_mgr = ExecutionProviderManager(config)
        providers = exec_mgr.build_provider_list()

        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        gpu_type = exec_mgr.detect_gpu_type()
        factory = SessionOptionsFactory(config)
        factory.get_strategy(gpu_type).configure_session_options(sess_options)

        model = load_asr_model(self._paths.models_dir, providers, sess_options)

        return Recognizer(
            model=model,
            input_queue=_queue.Queue(),
            output_queue=_queue.Queue(),
            sample_rate=config["audio"]["sample_rate"],
            app_state=app_state,
            verbose=self._args.verbose,
        )

    def _run_lifecycle(self, server_app: ServerApp) -> None:
        """Block until the server WsServer thread exits.

        In server-only mode, prints URL and QR code first.
        In both modes, blocks on the WsServer thread.
        """
        server_url = f"ws://127.0.0.1:{server_app.port}"
        if self._args.server_only:
            print(f"Server listening on {server_url}", flush=True)
            print_qr_code(server_url)
            logging.info("Running in --server-only mode. Press Ctrl+C to stop.")
        while (
            server_app._ws_server._thread is not None
            and server_app._ws_server._thread.is_alive()
        ):
            server_app._ws_server.join(timeout=0.5)

    def _spawn_tauri_client(self, server_url: str) -> "subprocess.Popen[bytes]":
        """Spawn the Tauri desktop client binary.

        Args:
            server_url: WebSocket URL passed as --server-url=.

        Returns:
            Running subprocess handle.
        """
        binary = self._tauri_binary_path()
        cmd = [str(binary), f"--server-url={server_url}"]
        if self._args.input_file is not None:
            cmd.append(f"--input-file={self._args.input_file}")
        return subprocess.Popen(cmd)

    def _spawn_client_for_download(self) -> "subprocess.Popen[bytes]":
        """Spawn download_models.py to show the model download GUI.

        Returns:
            Running subprocess handle.
        """
        script = self._paths.root_dir / "src" / "client" / "tk" / "download_models.py"
        return subprocess.Popen([sys.executable, str(script)])

    def _tauri_binary_path(self) -> Path:
        return (
            self._paths.root_dir
            / "src" / "client" / "tauri" / "src-tauri"
            / "target" / "release" / "stt-tauri-client.exe"
        )

    def _make_cli_progress_reporter(self) -> Callable[[float, int, int], None]:
        """Create a coarse-grained terminal progress reporter for headless downloads.

        Returns:
            Progress callback accepted by ModelDownloader.
        """
        last_reported = {"percent": -10}

        def report(progress: float, downloaded_bytes: int, total_bytes: int) -> None:
            percent = int(progress * 100)
            if percent >= 100 and last_reported["percent"] < 100:
                print("Downloading parakeet... 100%", flush=True)
                last_reported["percent"] = 100
                return
            if percent - last_reported["percent"] < 10:
                return
            downloaded_label = f"{downloaded_bytes / (1024 * 1024):.1f} MB"
            print(
                f"Downloading parakeet... {percent}% "
                f"({downloaded_label} downloaded)",
                flush=True,
            )
            last_reported["percent"] = percent

        return report
