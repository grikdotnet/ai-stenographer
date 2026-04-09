"""Tests for terminal model download confirmation dialog."""

import io

from src.downloader.ModelDownloadCliDialog import ModelDownloadCliDialog


class TestModelDownloadCliDialog:
    def test_confirm_download_accepts_yes(self) -> None:
        output = io.StringIO()
        dialog = ModelDownloadCliDialog(
            ["parakeet"],
            input_func=lambda _prompt: "y",
            output=output,
        )

        assert dialog.confirm_download() is True

    def test_confirm_download_accepts_no(self) -> None:
        output = io.StringIO()
        dialog = ModelDownloadCliDialog(
            ["parakeet"],
            input_func=lambda _prompt: "n",
            output=output,
        )

        assert dialog.confirm_download() is False

    def test_confirm_download_reprompts_on_invalid_input(self) -> None:
        output = io.StringIO()
        responses = iter(["maybe", "yes"])
        dialog = ModelDownloadCliDialog(
            ["parakeet"],
            input_func=lambda _prompt: next(responses),
            output=output,
        )

        assert dialog.confirm_download() is True
        assert "Please enter 'y' or 'n'." in output.getvalue()

    def test_prompt_mentions_missing_model(self) -> None:
        output = io.StringIO()
        dialog = ModelDownloadCliDialog(
            ["parakeet"],
            input_func=lambda _prompt: "n",
            output=output,
        )

        dialog.confirm_download()
        assert "parakeet" in output.getvalue()

    def test_confirm_download_treats_ctrl_c_as_cancel(self) -> None:
        output = io.StringIO()

        def interrupted(_prompt: str) -> str:
            raise KeyboardInterrupt

        dialog = ModelDownloadCliDialog(
            ["parakeet"],
            input_func=interrupted,
            output=output,
        )

        assert dialog.confirm_download() is False
