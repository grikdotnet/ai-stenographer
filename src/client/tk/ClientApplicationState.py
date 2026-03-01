"""Client-side application state with full state machine including paused.

Re-exports ApplicationState as ClientApplicationState so client code can import
from the canonical client-side location without duplicating the implementation.
ApplicationState is preserved unchanged for existing server-neutral tests.
"""

from src.ApplicationState import ApplicationState as ClientApplicationState

__all__ = ["ClientApplicationState"]
