"""Tests for download-models.py _main() entry-point function.

Strategy: import and call _main() directly with patched ModelManager and
show_download_dialog so no real network, files, or GUI are involved.

Tests cover:
- No missing models: exits 0 without showing dialog
- Download succeeds: exits 0
- Download cancelled/failed: exits 1
"""

import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from download_models import _main  # noqa: E402 — path setup must come first


_FAKE_MODELS_DIR = Path("/fake/models")


class TestNothingMissing:
    """When all models are present, exit 0 without opening dialog."""

    def test_exits_0(self) -> None:
        with (
            patch("download_models.ModelManager.get_missing_models", return_value=[]),
            patch("download_models.show_download_dialog") as mock_dialog,
            pytest.raises(SystemExit) as exc_info,
        ):
            _main(_FAKE_MODELS_DIR)

        assert exc_info.value.code == 0

    def test_dialog_not_shown(self) -> None:
        with (
            patch("download_models.ModelManager.get_missing_models", return_value=[]),
            patch("download_models.show_download_dialog") as mock_dialog,
            pytest.raises(SystemExit),
        ):
            _main(_FAKE_MODELS_DIR)

        mock_dialog.assert_not_called()


class TestDownloadSuccess:
    """When models are missing and user completes download, exit 0."""

    def test_exits_0_on_success(self) -> None:
        with (
            patch("download_models.ModelManager.get_missing_models", return_value=["parakeet"]),
            patch("download_models.show_download_dialog", return_value=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            _main(_FAKE_MODELS_DIR)

        assert exc_info.value.code == 0

    def test_dialog_called_with_missing_models_and_dir(self) -> None:
        with (
            patch("download_models.ModelManager.get_missing_models", return_value=["parakeet"]),
            patch("download_models.show_download_dialog", return_value=True) as mock_dialog,
            pytest.raises(SystemExit),
        ):
            _main(_FAKE_MODELS_DIR)

        mock_dialog.assert_called_once_with(None, ["parakeet"], _FAKE_MODELS_DIR)


class TestDownloadCancelledOrFailed:
    """When user cancels or download fails, exit 1."""

    def test_exits_1_on_cancel(self) -> None:
        with (
            patch("download_models.ModelManager.get_missing_models", return_value=["parakeet"]),
            patch("download_models.show_download_dialog", return_value=False),
            pytest.raises(SystemExit) as exc_info,
        ):
            _main(_FAKE_MODELS_DIR)

        assert exc_info.value.code == 1
