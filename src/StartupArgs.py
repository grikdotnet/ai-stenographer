"""Parsed and validated command-line arguments for application startup."""

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class StartupArgs:
    """Immutable value object holding parsed CLI arguments.

    Construct via from_argv(); do not instantiate directly.

    Attributes:
        verbose: If True, logging level is set to DEBUG.
        download_model: If True, auto-downloads missing models before serving.
        host: Optional WebSocket host override; falls back to config/default when None.
        port: WebSocket port to bind on; 0 for OS-assigned.
    """

    verbose: bool
    download_model: bool
    host: str | None
    port: int

    @classmethod
    def from_argv(cls, argv: list[str]) -> "StartupArgs":
        """Parse and validate raw argv into a StartupArgs instance.

        Parses: -v, --download-model, --host=<value>, --port=<n>.
        Accepts --server-only as a no-op compatibility flag.

        Args:
            argv: Raw argument list (typically sys.argv).

        Returns:
            Populated StartupArgs instance.

        Raises:
            SystemExit: With code 1 if arguments are invalid.
        """
        verbose = "-v" in argv
        download_model = "--download-model" in argv
        host: str | None = next(
            (arg.split("=", 1)[1] for arg in argv if arg.startswith("--host=")),
            None,
        )
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
            download_model=download_model,
            host=host,
            port=port,
        )
