import logging
from typing import Optional, TYPE_CHECKING
from src.types import RecognitionResult, DisplayInstructions
from src.protocols import TextRecognitionSubscriber

if TYPE_CHECKING:
    from src.client.tk.gui.TextDisplayWidget import TextDisplayWidget


def _format_chunk_ids(chunk_ids: list) -> str:
    """Format chunk IDs for logging (e.g., [45-47] or [])."""
    return f"[{chunk_ids[0]}-{chunk_ids[-1]}]" if chunk_ids else "[]"


class TextFormatter(TextRecognitionSubscriber):
    """Formatting logic for text display.

    TextFormatter is a stateful controller that:
    - Tracks formatting state
    - Calculates DisplayInstructions
    - Triggers TextDisplayWidget updates

    Implements TextRecognitionSubscriber protocol:
    - on_partial_update() → partial_update()
    - on_finalization() → finalization()

    Thread Safety:
        Methods are called from TextMatcher's background thread.
        Delegates to TextDisplayWidget which handles thread-safe GUI updates.
    """

    PARAGRAPH_PAUSE_THRESHOLD: float = 2.0  # seconds

    def __init__(self, display: 'TextDisplayWidget', verbose: bool = False) -> None:
        """Initialize formatter with display widget reference.

        Args:
            display: TextDisplayWidget for triggering GUI updates
            verbose: flag
        """
        self.display: 'TextDisplayWidget' = display
        self.verbose: bool = verbose
        self.current_preliminary: Optional[RecognitionResult] = None
        self.finalized_text: str = ""
        self.last_finalized_text: str = ""
        self.last_finalized_end_time: float = 0.0

    # TextRecognitionSubscriber protocol implementation
    def on_partial_update(self, result: RecognitionResult) -> None:
        """Handle preliminary recognition result (protocol method)."""
        self.partial_update(result)

    def on_finalization(self, result: RecognitionResult) -> None:
        """Handle finalized recognition result (protocol method).

        Args:
            result: Finalized RecognitionResult
        """
        self.finalization(result)

    # Implementation methods
    def partial_update(self, result: RecognitionResult) -> None:
        """Calculate partial update. Detect pauses and insert paragraph breaks.

        Args:
            result: Preliminary RecognitionResult to display
        """
        instructions = self._calculate_partial_update(result)
        self.display.apply_instructions(instructions)

    def finalization(self, result: RecognitionResult) -> None:
        """Calculate finalization.

        Args:
            result: Final RecognitionResult to finalize
        """
        instructions = self._calculate_finalization(result)
        if instructions:  # None = duplicate detected, skip display
            self.display.apply_instructions(instructions)

    def _calculate_partial_update(self, result: RecognitionResult) -> DisplayInstructions:
        """Calculate instructions for partial text update (replace semantics).

        Args:
            result: Preliminary RecognitionResult

        Returns:
            DisplayInstructions for rendering
        """
        # Replace current preliminary (not append)
        self.current_preliminary = result

        # Check for pause to add paragraph break
        if (self.last_finalized_end_time != 0
            and (result.start_time - self.last_finalized_end_time) >= self.PARAGRAPH_PAUSE_THRESHOLD
            and self.last_finalized_text != ""
            and self.finalized_text and not self.finalized_text.endswith('\n')):
            # Pause detected - insert paragraph break and re-render
            pause_duration = result.start_time - self.last_finalized_end_time
            if self.verbose:
                logging.debug(f"TextFormatter: pause detected ({pause_duration:.2f}s) - inserting paragraph break")

            self.finalized_text += "\n"

            # Get current preliminary result (singleton)
            remaining = [self.current_preliminary] if self.current_preliminary else []

            return DisplayInstructions(
                action='rerender_all',
                finalized_text=self.finalized_text,
                preliminary_segments=remaining,
                paragraph_break_inserted=True
            )

        # Reset last finalized text tracker because new preliminary text means no longer at boundary
        self.last_finalized_text = ""

        if self.verbose:
            logging.debug(f"TextFormatter: partial_update() text='{result.text}' "
                         f"chunk_ids={_format_chunk_ids(result.chunk_ids)}")

        # Trigger rerender_all to replace previous preliminary
        remaining = [self.current_preliminary] if self.current_preliminary else []
        return DisplayInstructions(
            action='rerender_all',
            finalized_text=self.finalized_text,
            preliminary_segments=remaining
        )

    def _calculate_finalization(self, result: RecognitionResult) -> Optional[DisplayInstructions]:
        """Calculate instructions for text finalization.

        Args:
            result: Final RecognitionResult

        Returns:
            DisplayInstructions for rendering, or None if duplicate detected
        """
        # Prevent duplicates
        if result.text == self.last_finalized_text:
            if self.verbose:
                logging.debug(f"TextFormatter: duplicate finalization prevented for text='{result.text}'")
            return None

        # Append to finalized text buffer
        if self.finalized_text:
            self.finalized_text += " " + result.text
        else:
            self.finalized_text = result.text

        if self.verbose:
            logging.debug(f"TextFormatter: finalization() text='{result.text}' "
                         f"chunk_ids={_format_chunk_ids(result.chunk_ids)}")
            logging.debug(f"TextFormatter: finalized_text buffer size: {len(self.finalized_text)} chars")

        # Track last finalized text and timing
        self.last_finalized_text = result.text
        self.last_finalized_end_time = result.end_time

        # Clear current preliminary
        self.current_preliminary = None

        # No remaining preliminary after finalization
        remaining = []

        if self.verbose:
            logging.debug(f"TextFormatter: finalization() cleared preliminary")

        return DisplayInstructions(
            action='finalize',
            finalized_text=self.finalized_text,
            preliminary_segments=remaining
        )
