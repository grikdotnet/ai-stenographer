"""Terminal prompt for confirming headless model downloads."""

from typing import Callable, TextIO


class ModelDownloadCliDialog:
    """Ask the user whether missing models should be downloaded now.

    Args:
        missing_models: Model names to include in the prompt.
        input_func: Input function used to read the user's response.
        output: Stream used for prompt text.
    """

    def __init__(
        self,
        missing_models: list[str],
        input_func: Callable[[str], str] = input,
        output: TextIO | None = None,
    ) -> None:
        self._missing_models = missing_models
        self._input_func = input_func
        self._output = output

    def confirm_download(self) -> bool:
        """Prompt until the user gives a valid yes/no answer.

        Returns:
            True when the user confirms the download, otherwise False.
            Terminal interruption is treated as cancellation.
        """
        self._write(
            "Missing models: "
            f"{', '.join(self._missing_models)}. "
            "Download now? [y/n]\n"
        )
        while True:
            try:
                response = self._input_func("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                self._write("\n")
                return False
            if response in {"y", "yes"}:
                return True
            if response in {"n", "no"}:
                return False
            self._write("Please enter 'y' or 'n'.\n")

    def _write(self, text: str) -> None:
        if self._output is None:
            print(text, end="")
            return
        self._output.write(text)
        self._output.flush()
