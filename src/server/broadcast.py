"""Queue protocol for the ServerApp broadcast queue.

_SHUTDOWN is the sentinel that SessionManager puts into the queue on server
shutdown, causing WsServer to close all sessions and stop serving.

Phase 4: typed model-status message classes will be added here.
"""

_SHUTDOWN = object()
