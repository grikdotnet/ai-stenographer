import logging
from typing import List, Set, Optional, TYPE_CHECKING
from src.types import RecognitionResult, DisplayInstructions

if TYPE_CHECKING:
    from src.gui.TextDisplayWidget import TextDisplayWidget


def _format_chunk_ids(chunk_ids: list) -> str:
    """Format chunk IDs for logging (e.g., [45-47] or [])."""
    return f"[{chunk_ids[0]}-{chunk_ids[-1]}]" if chunk_ids else "[]"


class TextFormatter:
    """Formatting logic for text display.

    TextFormatter is a stateful controller that:
    - Tracks formatting state
    - Calculates DisplayInstructions
    - Triggers TextDisplayWidget updates
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
        self.preliminary_results: List[RecognitionResult] = []
        self.finalized_chunk_ids: Set[int] = set()
        self.finalized_text: str = ""
        self.last_finalized_text: str = ""
        self.last_finalized_end_time: float = 0.0

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
        """Calculate instructions for partial text update.

        Args:
            result: Preliminary RecognitionResult

        Returns:
            DisplayInstructions for rendering
        """
        self.preliminary_results.append(result)

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

            # Get remaining preliminary results
            remaining = self._get_remaining_preliminary_results()

            return DisplayInstructions(
                action='rerender_all',
                finalized_text=self.finalized_text,
                preliminary_segments=remaining,
                paragraph_break_inserted=True
            )

        # Determine if space needed and pre-format the text
        has_existing_preliminary = len(self.preliminary_results) > 1
        text_to_display = (" " + result.text) if has_existing_preliminary else result.text

        # Reset last finalized text tracker because new preliminary text means no longer at boundary
        self.last_finalized_text = ""

        if self.verbose:
            with_space = has_existing_preliminary
            logging.debug(f"TextFormatter: partial_update() text='{result.text}' "
                         f"chunk_ids={_format_chunk_ids(result.chunk_ids)} with_space={with_space}")

        return DisplayInstructions(
            action='update_partial',
            text_to_append=text_to_display  # Pre-formatted with space if needed
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

        self.finalized_chunk_ids.update(result.chunk_ids)

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

        # Get remaining preliminary results (filtered by chunk IDs)
        remaining = self._get_remaining_preliminary_results()

        if self.verbose:
            logging.debug(f"TextFormatter: finalization() remaining_count={len(remaining)}")

        return DisplayInstructions(
            action='finalize',
            finalized_text=self.finalized_text,
            preliminary_segments=remaining
        )

    def _get_remaining_preliminary_results(self) -> List[RecognitionResult]:
        """Get preliminary results that haven't been finalized yet.

        Filters by chunk ID overlap - if ANY chunk in result has been finalized,
        the entire result is excluded.

        Returns:
            List of RecognitionResults not yet finalized
        """
        return [
            r for r in self.preliminary_results
            if not any(cid in self.finalized_chunk_ids for cid in r.chunk_ids)
        ]
