"""Application startup orchestration."""

import json
import logging
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.ApplicationState import ApplicationState
from src.downloader.ModelDownloadCliDialog import ModelDownloadCliDialog
from src.asr.ModelManager import ModelManager
from src.asr.RecognizerFactory import IRecognizerFactory, RecognizerFactory
from src.LoggingSetup import setup_logging
from src.PathResolver import ResolvedPaths
from src.server.qr_display import print_qr_code
from src.server.ServerApp import ServerApp
from src.server.WsServer import DEFAULT_SERVER_HOST
from src.StartupArgs import StartupArgs

class TauriBinaryNotFoundError(RuntimeError):
    """Raised when the Tauri client binary is absent in default (non-server-only) mode."""


class MissingModelsError(RuntimeError):
    """Raised when required ASR models are absent and cannot be downloaded."""


class DownloadCancelledError(Exception):
    """Raised when model provisioning is declined or cancelled by the user."""


@dataclass(frozen=True)
class StartupDecisions:
    """Resolved startup actions for the current args + model availability."""

    run_cli_dialog: bool = False
    auto_download_model: bool = False
    create_recognizer: bool = False
    set_waiting_for_model: bool = False
    run_gui_client: bool = False


class StartupController:
    """Orchestrates the full application startup sequence.

    Encapsulates the ordered phases: environment validation, logging setup,
    model availability, model loading, server creation, and lifecycle management.
    Designed to be constructed once and run once.

    Args:
        args: Parsed and validated CLI arguments.
        paths: Resolved filesystem paths for current distribution mode.
        model_manager: ModelManager instance used to check for missing models.
    """

    def __init__(
        self,
        args: StartupArgs,
        paths: ResolvedPaths,
        model_manager: ModelManager,
    ) -> None:
        self._args = args
        self._paths = paths
        self._model_manager = model_manager
        self._model_registry = model_manager.model_registry
        self._asr_model = model_manager.get_asr_model()
        self._app_state = ApplicationState()
        self._recognizer_factory: IRecognizerFactory | None = None

    def run(self) -> None:
        """Execute the full startup sequence.

        Algorithm:
            1. Validate environment.
            2. Set up logging.
            3. Ensure the Parakeet model is available.
            4. Load config.
            5. Create the recognizer factory.
            6. Create and start ServerApp.
            7. Create a recognizer immediately when startup can run inference.
            8. Attach recognizer to server.
            9. In default mode, spawn Tauri client.
            10. Block on server lifecycle; stop server in finally.
        """
        self._validate_environment()
        self._setup_logging()

        model_exists = self._model_manager.model_exists()
        decisions = self._resolve_startup_decisions(model_exists)

        if decisions.set_waiting_for_model:
            self._app_state.set_state("waiting_for_model")

        if decisions.run_cli_dialog:
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                print(
                    f"Missing models: {self._asr_model.name}. Non-interactive mode cannot prompt. "
                    "Re-run with --server-only --download-model to provision automatically.",
                    file=sys.stderr,
                )
                raise MissingModelsError(f"Missing models: {self._asr_model.name}")

            if not ModelDownloadCliDialog([self._asr_model.name]).confirm_download():
                raise DownloadCancelledError("Model download was cancelled.")

        if decisions.auto_download_model:
            self._auto_download_model()

        if decisions.auto_download_model and not self._model_manager.model_exists():
            raise MissingModelsError("Download failed. Model still missing")

        config_path = self._paths.config_dir / "server_config.json"
        with open(config_path) as f:
            config = json.load(f)

        server_host = self._resolve_server_host(config)
        server_url = f"ws://{server_host}:{self._args.port}"
        recognizer_factory = RecognizerFactory(
            config=config,
            asr_model=self._asr_model,
            app_state=self._app_state,
            verbose=self._args.verbose,
        )
        self._recognizer_factory = recognizer_factory

        server_app = self._create_and_start_server(config, server_host, recognizer_factory)
        client_proc: subprocess.Popen[bytes] | None = None
        try:
            if decisions.create_recognizer:
                server_app.attach_recognizer(recognizer_factory.create_recognizer())
                self._app_state.set_state("running")

            server_url = f"ws://{server_host}:{server_app.port}"

            if decisions.run_gui_client:
                client_proc = self._spawn_tauri_client(server_url)
                self._start_client_exit_watcher(client_proc, server_app)

            self._run_lifecycle(server_app, server_url)
        except KeyboardInterrupt:
            logging.info("Interrupted by user.")
        finally:
            server_app.stop()

    def _resolve_startup_decisions(self, model_exists: bool) -> StartupDecisions:
        """Resolve startup actions from CLI args and model availability.

        Args:
            model_exists: Whether the required model is already available.

        Returns:
            StartupDecisions describing the actions to take in run().
        """
        if model_exists:
            return StartupDecisions(
                create_recognizer=True,
                run_gui_client=not self._args.server_only,
            )

        if self._args.server_only:
            if self._args.download_model:
                return StartupDecisions(
                    auto_download_model=True,
                    create_recognizer=True,
                    set_waiting_for_model=True,
                )

            return StartupDecisions(
                run_cli_dialog=True,
                auto_download_model=True,
                create_recognizer=True,
                set_waiting_for_model=True,
            )

        return StartupDecisions(
            set_waiting_for_model=True,
            run_gui_client=True,
        )

    def _resolve_server_host(self, config: dict) -> str:
        """Resolve the WebSocket host from CLI args, config, or default.

        Args:
            config: Parsed server configuration.

        Returns:
            The resolved host name or IP address.
        """
        if self._args.host is not None:
            return self._args.host
        return config.get("server", {}).get("host", DEFAULT_SERVER_HOST)

    def _auto_download_model(self) -> None:
        """Download the required server-only model in the current process.

        Raises:
            DownloadCancelledError: If the user interrupts the download.
            MissingModelsError: If the download fails.
        """
        try:
            self._asr_model.download(self._make_cli_progress_reporter())
        except KeyboardInterrupt as exc:
            print(file=sys.stderr)
            self._asr_model.cleanup_partial_files()
            raise DownloadCancelledError("Model download was cancelled.") from exc
        except Exception as exc:
            self._asr_model.cleanup_partial_files()
            raise MissingModelsError(f"Automatic download failed: {exc}") from exc

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
                "  cd client/tauri && npm run tauri:build\n\n"
                "Or run in server-only mode:\n"
                f"  python main.py --server-only [--port={port_hint}]",
                file=sys.stderr,
            )
            raise TauriBinaryNotFoundError(f"Tauri binary not found: {binary}")

    def _setup_logging(self) -> None:
        """Configure logging based on args.verbose and frozen status."""
        is_frozen = getattr(sys, "frozen", False)
        setup_logging(self._paths.logs_dir, verbose=self._args.verbose, is_frozen=is_frozen)

    def _create_and_start_server(
        self,
        config: dict,
        host: str,
        recognizer_factory: IRecognizerFactory,
    ) -> ServerApp:
        """Create and start ServerApp without a recognizer.

        Args:
            config: Parsed server_config.json dict.
            host: WebSocket host to bind.
            recognizer_factory: Factory used for eager or deferred recognizer creation.

        Returns:
            A started ServerApp instance.
        """
        server_app = ServerApp(
            config=config,
            model_registry=self._model_registry,
            host=host,
            port=self._args.port,
            app_state=self._app_state,
            recognizer_factory=recognizer_factory,
        )
        server_app.start()
        return server_app

    def _run_lifecycle(self, server_app: ServerApp, server_url: str) -> None:
        """Block until the server WsServer thread exits.

        In server-only mode, prints URL and QR code first.
        In both modes, blocks on the WsServer thread.
        """
        if self._args.server_only:
            print(f"Server listening on {server_url}", flush=True)
            print_qr_code(server_url)
            logging.info("Running in --server-only mode. Press Ctrl+C to stop.")
        while server_app.is_running():
            server_app.join(timeout=0.5)

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

    def _start_client_exit_watcher(
        self,
        client_proc: "subprocess.Popen[bytes]",
        server_app: ServerApp,
    ) -> None:
        """Start a daemon thread that stops the server when the GUI client exits.

        Args:
            client_proc: Spawned GUI client process.
            server_app: Server instance to stop after client exit.
        """

        def watch_client_exit() -> None:
            try:
                client_proc.wait()
            except Exception:
                logging.exception("ClientExitWatcher: failed while waiting for GUI client exit")
                return

            logging.info("GUI client exited; stopping server")
            server_app.stop()

        watcher = threading.Thread(
            target=watch_client_exit,
            daemon=True,
            name="ClientExitWatcher",
        )
        watcher.start()

    def _tauri_binary_path(self) -> Path:
        return (
            self._paths.root_dir
            / "client" / "tauri" / "src-tauri"
            / "target" / "release" / "stt-tauri-client.exe"
        )

    def _make_cli_progress_reporter(self) -> Callable[[float, int, int], None]:
        """Create a coarse-grained terminal progress reporter for headless downloads.

        Returns:
            Progress callback accepted by the shared model download flow.
        """
        last_reported_percent = -10
        model_name = self._asr_model.name

        def report(progress: float, downloaded_bytes: int, total_bytes: int) -> None:
            nonlocal last_reported_percent
            percent = int(progress * 100)
            if percent >= 100 and last_reported_percent < 100:
                print(f"Downloading {model_name}... 100%", flush=True)
                last_reported_percent = 100
                return
            if percent - last_reported_percent < 10:
                return
            downloaded_label = f"{downloaded_bytes / (1024 * 1024):.1f} MB"
            print(
                f"Downloading {model_name}... {percent}% "
                f"({downloaded_label} downloaded)",
                flush=True,
            )
            last_reported_percent = percent

        return report
