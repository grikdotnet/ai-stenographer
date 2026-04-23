"""Parsed and validated command-line arguments for application startup."""

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class StartupArgs:
    """Immutable value object holding parsed CLI arguments.

    Construct via from_argv(); do not instantiate directly.

    Attributes:
        verbose: If True, logging level is set to DEBUG.
        server_only: If True, skips Tauri client launch and runs headless.
        download_model: If True, auto-downloads missing models in server-only mode.
        host: Optional WebSocket host override; falls back to config/default when None.
        input_file: Optional path to a WAV file forwarded to the Tauri client.
        port: WebSocket port to bind on; 0 for OS-assigned.
    """

    verbose: bool
    server_only: bool
    download_model: bool
    host: str | None
    input_file: str | None
    port: int

    @classmethod
    def from_argv(cls, argv: list[str]) -> "StartupArgs":
        """Parse and validate raw argv into a StartupArgs instance.

        Parses: -v, --server-only, --download-model, --host=<value>,
        --input-file=<path>, --port=<n>.

        Args:
            argv: Raw argument list (typically sys.argv).

        Returns:
            Populated StartupArgs instance.

        Raises:
            SystemExit: With code 1 if arguments are invalid.
        """
        verbose = "-v" in argv
        server_only = "--server-only" in argv
        download_model = "--download-model" in argv
        host: str | None = next(
            (arg.split("=", 1)[1] for arg in argv if arg.startswith("--host=")),
            None,
        )
        input_file: str | None = next(
            (arg.split("=", 1)[1] for arg in argv if arg.startswith("--input-file=")),
            None,
        )
        if download_model and not server_only:
            print(
                "Error: --download-model is only supported together with --server-only",
                file=sys.stderr,
            )
            sys.exit(1)
        port_str = next(
            (arg.split("=", 1)[1] for arg in argv if arg.startswith("--port=")),
            "0",
        )
        try:
            port = int(port_str)
        except ValueError:
            print(f"Error: port must be an integer, got {port_str!r}", file=sys.stderr)
            sys.exit(1)
        if not (0 <= port <= 65535):
            print(f"Error: port must be 0-65535, got {port}", file=sys.stderr)
            sys.exit(1)
        return cls(
            verbose=verbose,
            server_only=server_only,
            download_model=download_model,
            host=host,
            input_file=input_file,
            port=port,
        )
