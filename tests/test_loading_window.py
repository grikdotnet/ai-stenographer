"""Tests for LoadingWindow class."""
import tkinter as tk
import pytest
from pathlib import Path
from src.LoadingWindow import LoadingWindow


@pytest.fixture
def image_path():
    """Path to test image."""
    return Path("stenographer.jpg")


def test_create_loading_window(image_path):
    """Test creating loading window with title and size."""
    window = LoadingWindow(image_path, "Loading models...")

    assert window.root is not None
    assert isinstance(window.root, tk.Tk)

    assert window.root.title() == "Loading AI Stenographer..."

    # Check geometry - width should be 700, height may vary based on content
    geometry = window.root.geometry()
    assert geometry.startswith("700x"), f"Width should be 700, got: {geometry}"

    window.close()


def test_show_image_existing_file(image_path):
    """Test displaying existing image file."""
    if not image_path.exists():
        pytest.skip("stenographer.jpg not found")

    window = LoadingWindow(image_path, "Loading...")

    assert window.image_label is not None
    assert window.photo_image is not None

    window.close()


def test_show_message(image_path):
    """Test displaying loading message below image."""
    message = "Loading Parakeet model..."
    window = LoadingWindow(image_path, message)

    assert window.message_label is not None

    assert window.message_label.cget("text") == message

    window.close()


def test_update_message(image_path):
    """Test updating loading message text."""
    try:
        window = LoadingWindow(image_path, "Initial message")

        new_message = "Loading Silero VAD..."
        window.update_message(new_message)

        assert window.message_label.cget("text") == new_message

        window.close()
    except tk.TclError as e:
        # Skip if Tk cannot create multiple windows (test environment issue)
        if "Can't find a usable tk.tcl" in str(e):
            pytest.skip("Tk installation issue in test environment")
        raise


def test_multiple_message_updates(image_path):
    """Test multiple sequential message updates."""
    window = LoadingWindow(image_path, "Step 1")

    messages = [
        "Checking models...",
        "Loading Parakeet model...",
        "Loading Silero VAD...",
        "Initializing pipeline...",
        "Ready!"
    ]

    for msg in messages:
        window.update_message(msg)
        assert window.message_label.cget("text") == msg

    window.close()


def test_close_window(image_path):
    """Test closing window properly."""
    window = LoadingWindow(image_path, "Loading...")

    assert window.root.winfo_exists()

    window.close()

    try:
        exists = window.root.winfo_exists()
        assert not exists
    except tk.TclError:
        # Expected - window destroyed
        pass


def test_window_not_resizable(image_path):
    """Test that loading window is not resizable."""
    window = LoadingWindow(image_path, "Loading...")

    assert not window.root.resizable()[0]  # width
    assert not window.root.resizable()[1]  # height

    window.close()


def test_window_centered_on_screen(image_path):
    """Test that loading window appears centered on screen."""
    window = LoadingWindow(image_path, "Loading...")

    # Get window position and size
    window.root.update_idletasks()
    x = window.root.winfo_x()
    y = window.root.winfo_y()
    window_width = 700  # Updated from 800 to 700
    window_height = 800

    # Get screen size
    screen_width = window.root.winfo_screenwidth()
    screen_height = window.root.winfo_screenheight()

    # Calculate expected centered position
    expected_x = (screen_width - window_width) // 2
    expected_y = (screen_height - window_height) // 2

    # Verify window is centered (allow margin for window manager and borderless adjustments)
    margin = 60  # Increased margin for borderless windows
    assert abs(x - expected_x) <= margin, f"X position {x} not centered (expected ~{expected_x})"
    assert abs(y - expected_y) <= margin, f"Y position {y} not centered (expected ~{expected_y})"

    window.close()


def test_image_centered_in_window(image_path):
    """Test that image is centered in window."""
    if not image_path.exists():
        pytest.skip("stenographer.jpg not found")

    window = LoadingWindow(image_path, "Loading...")

    # Verify image label exists
    assert window.image_label is not None

    # Get pack info to verify centering configuration
    pack_info = window.image_label.pack_info()
    assert pack_info is not None

    # Verify image is positioned in center (pack with pady creates vertical centering)
    # The image should have pady=20 as defined in LoadingWindow
    assert 'pady' in pack_info
    assert pack_info['pady'] != '0'  # Should have vertical padding for centering

    # Verify label is packed in parent container (not placed at edges)
    assert pack_info['in'] is not None

    window.close()
