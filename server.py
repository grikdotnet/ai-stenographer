#!/usr/bin/env python3
"""
AI Stenographer – WebSocket Server
Streams partial and final transcription results to WebSocket clients
using the existing STT pipeline architecture.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from threading import Thread
from typing import Set

import websockets
from websockets.server import WebSocketServerProtocol

# ---------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT_PATH.parent))

from src.protocols import TextRecognitionSubscriber
from src.types import RecognitionResult
from src.pipeline import STTPipeline
from src.PathResolver import PathResolver
from src.LoggingSetup import setup_logging

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

path_resolver = PathResolver(SCRIPT_PATH)
PATHS = path_resolver.paths
path_resolver.ensure_local_dir_structure()

setup_logging(PATHS.logs_dir, verbose=True, is_frozen=False)
logger = logging.getLogger("WebSocketServer")

# ---------------------------------------------------------------------
# WebSocket Hub (async-only networking)
# ---------------------------------------------------------------------

class WebSocketHub:
    def __init__(self):
        self.clients: Set[WebSocketServerProtocol] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    async def register(self, ws: WebSocketServerProtocol):
        self.clients.add(ws)
        logger.info(f"Client connected ({len(self.clients)} total)")

    async def unregister(self, ws: WebSocketServerProtocol):
        self.clients.discard(ws)
        logger.info(f"Client disconnected ({len(self.clients)} total)")

    async def broadcast(self, payload: dict):
        if not self.clients:
            return

        message = json.dumps(payload)
        await asyncio.gather(
            *[client.send(message) for client in self.clients],
            return_exceptions=True,
        )

    def send(self, payload: dict):
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self.broadcast(payload),
                self.loop,
            )

# ---------------------------------------------------------------------
# Observer subscriber (NO asyncio, NO threads)
# ---------------------------------------------------------------------

class WebSocketSubscriber(TextRecognitionSubscriber):
    def __init__(self, hub: WebSocketHub):
        self.hub = hub
        self.seq = 0  # monotonically increasing sequence number

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def on_partial_update(self, result: RecognitionResult) -> None:
        text = result.text.strip()
        if len(text) < 2:
            return  # filter junk partials

        self.hub.send({
            "type": "partial",
            "seq": self._next_seq(),
            "text": text,
            "start_time": result.start_time,
            "end_time": result.end_time,
        })

    def on_finalization(self, result: RecognitionResult) -> None:
        text = result.text.strip()
        if not text:
            return

        self.hub.send({
            "type": "final",
            "seq": self._next_seq(),
            "text": text,
            "start_time": result.start_time,
            "end_time": result.end_time,
        })

# ---------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------

hub = WebSocketHub()

async def handle_client(ws: WebSocketServerProtocol):
    await hub.register(ws)
    try:
        await ws.send(json.dumps({
            "type": "connected",
            "message": "Connected to AI Stenographer backend"
        }))
        async for _ in ws:
            pass  # server is push-only
    finally:
        await hub.unregister(ws)

async def run_ws_server(host: str, port: int):
    async with websockets.serve(handle_client, host, port):
        logger.info(f"WebSocket server running on ws://{host}:{port}")
        await asyncio.Future()  # run forever

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    logger.info("Starting WebSocket backend")

    # 1️⃣ Start asyncio loop for WebSocket server
    ws_loop = asyncio.new_event_loop()
    hub.set_loop(ws_loop)

    ws_thread = Thread(
        target=lambda: ws_loop.run_until_complete(
            run_ws_server("localhost", 8765)
        ),
        daemon=False,  # IMPORTANT: must NOT be daemon
    )
    ws_thread.start()

    logger.info("WebSocket thread started")

    # 2️⃣ Start STT pipeline (blocking call)
    pipeline = STTPipeline(
        model_path=str(PATHS.models_dir / "parakeet"),
        models_dir=PATHS.models_dir,
        config_path=str(path_resolver.get_config_path("stt_config.json")),
        verbose=True,
        window_duration=2.0,
        input_file=None,
    )

    # 3️⃣ Attach WebSocket subscriber
    subscriber = WebSocketSubscriber(hub)
    pipeline.text_recognition_publisher.subscribe(subscriber)

    logger.info("WebSocket subscriber registered")

    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        ws_loop.call_soon_threadsafe(ws_loop.stop)

if __name__ == "__main__":
    main()
