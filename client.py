# client.py
"""Dedicated Tk client entry script.

Accepts --server-url=ws://host:port (required; passed by main.py after the
server binds its port).  Shows LoadingWindow while connecting, then runs the
Tk GUI main loop.  Exits with code 1 on connection loss.

Not intended for direct user invocation — running it without a server will
fail cleanly with a connection error.
"""

import asyncio
import json
import logging
import signal
import sys
import threading
from pathlib import Path


# ============================================================================
# RESOLVE PATHS AT MODULE LOAD TIME (same logic as main.py)
# ============================================================================
if hasattr(sys.modules['__main__'], '__file__'):
    _SCRIPT_PATH = Path(sys.modules['__main__'].__file__).resolve()
else:
    _SCRIPT_PATH = Path(__file__).resolve()

# Add repo root to path so ``src`` imports work when spawned as subprocess
_REPO_ROOT = _SCRIPT_PATH.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.PathResolver import PathResolver
from src.LoggingSetup import setup_logging

_path_resolver = PathResolver(_SCRIPT_PATH)
_PATHS = _path_resolver.paths
_LOGS_DIR = _PATHS.logs_dir
_path_resolver.ensure_local_dir_structure()


def _parse_args() -> tuple[str, bool]:
    """Parse CLI arguments.

    Returns:
        Tuple of (server_url, verbose).

    Raises:
        SystemExit: If --server-url is missing.
    """
    server_url: str | None = None
    verbose = "-v" in sys.argv

    for arg in sys.argv[1:]:
        if arg.startswith("--server-url="):
            server_url = arg.split("=", 1)[1]

    if server_url is None:
        print("ERROR: --server-url=ws://host:port is required.", file=sys.stderr)
        sys.exit(1)

    return server_url, verbose


async def _connect_and_receive_session(server_url: str) -> tuple:
    """Connect to the server and receive the session_created frame.

    Args:
        server_url: WebSocket server URL.

    Returns:
        Tuple of (websocket, session_created_json_str).

    Raises:
        ConnectionError: If the server is unreachable or sends unexpected data.
    """
    import websockets

    ws = await websockets.connect(server_url)
    first_message = await ws.recv()

    if not isinstance(first_message, str):
        await ws.close()
        raise ConnectionError("Expected text session_created frame; got binary.")

    obj = json.loads(first_message)
    if obj.get("type") != "session_created":
        await ws.close()
        raise ConnectionError(f"Expected session_created, got: {obj.get('type')}")

    protocol_version = obj.get("protocol_version")
    if protocol_version != "v1":
        await ws.close()
        raise ConnectionError(f"Unsupported protocol version: {protocol_version}")

    return ws, first_message


def _run_asyncio_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Run an asyncio event loop in the current thread until it is stopped.

    Args:
        loop: The event loop to run.
    """
    asyncio.set_event_loop(loop)
    loop.run_forever()


if __name__ == "__main__":
    client_app = None
    loop: asyncio.AbstractEventLoop | None = None

    try:
        server_url, verbose = _parse_args()
        is_frozen = getattr(sys, 'frozen', False)
        setup_logging(_LOGS_DIR, verbose=verbose, is_frozen=is_frozen)

        import json as _json_module
        from src.client.tk.gui.LoadingWindow import LoadingWindow

        image_path = _path_resolver.get_asset_path("stenographer.gif")
        loading_window = LoadingWindow(image_path, "Connecting to server...")

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=_run_asyncio_loop, args=(loop,), daemon=True, name="ClientAsyncLoop")
        loop_thread.start()

        loading_window.update_message("Connecting...")

        future = asyncio.run_coroutine_threadsafe(
            _connect_and_receive_session(server_url), loop
        )
        try:
            websocket, session_created_json = future.result(timeout=10.0)
        except Exception as exc:
            loading_window.close()
            logging.error("Client: failed to connect to server: %s", exc)
            sys.exit(1)

        loading_window.update_message("Starting...")

        from src.client.tk.ClientApp import ClientApp

        config_path = str(_path_resolver.get_config_path("stt_config.json"))
        with open(config_path) as f:
            config = _json_module.load(f)

        client_app = ClientApp.from_session_created(
            server_url=server_url,
            session_created_json=session_created_json,
            config=config,
            verbose=verbose,
        )

        client_app._transport._loop = loop

        start_future = asyncio.run_coroutine_threadsafe(
            client_app._transport.start(websocket), loop
        )
        start_future.result(timeout=5.0)

        loading_window.close()

        client_app.start()

        def _on_shutdown(old_state: str, new_state: str) -> None:
            if new_state == "shutdown":
                try:
                    root = client_app.get_root()
                    root.after(0, root.quit)
                except Exception:
                    pass

        client_app.app_state.register_gui_observer(_on_shutdown)

        root = client_app.get_root()

        def _on_window_close() -> None:
            client_app.stop()
            try:
                root.quit()
            except Exception:
                pass

        root.protocol("WM_DELETE_WINDOW", _on_window_close)
        signal.signal(signal.SIGINT, lambda *_: _on_window_close())

        root.mainloop()

        exit_code = 0
        if client_app.app_state.get_state() == "shutdown":
            exit_code = 1

        client_app.stop()
        sys.exit(exit_code)

    except KeyboardInterrupt:
        logging.info("Client interrupted.")
        if client_app is not None:
            client_app.stop()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as exc:
        logging.error("Client ERROR: %s: %s", type(exc).__name__, exc)
        import traceback
        logging.error(traceback.format_exc())
        if client_app is not None:
            client_app.stop()
        sys.exit(1)
    finally:
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
