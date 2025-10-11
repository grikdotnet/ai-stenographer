"""
LoadingWindow displays a loading screen with image and progress messages.
"""
import tkinter as tk
from pathlib import Path
from typing import Optional
from PIL import Image, ImageTk


class LoadingWindow:
    """Displays loading screen with image and progress messages.

    Shows stenographer.jpg image centered in an 800x800 window
    with loading status messages below. Window is modal and centered on screen.
    """

    def __init__(self, image_path: Path, initial_message: str) -> None:
        """Create loading window with image and initial message.

        Args:
            image_path: Path to image file to display (e.g., stenographer.jpg)
            initial_message: Initial loading message to show
        """
        self.root: tk.Tk = tk.Tk()
        self.root.title("Loading AI Stenographer...")
        self.root.overrideredirect(True)  # Remove title bar and borders
        self.root.geometry("700x800")  # 700px for image width, 800px for image + text
        self.root.resizable(False, False)

        self._center_window()

        # Create main frame with no padding or borders
        container = tk.Frame(self.root, bg="white", bd=0, highlightthickness=0)
        container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.image_label: tk.Label = self._create_image_label(container, image_path)

        self.message_label: tk.Label = tk.Label(
            container,
            text=initial_message,
            font=("Arial", 14),
            bg="white",
            fg="black"
        )
        self.message_label.pack(pady=0)

        # Force window to appear and process events
        self.root.update_idletasks()
        self.root.update()

    def _center_window(self) -> None:
        """Center window on screen."""
        self.root.update_idletasks()

        # Get window size (700px for image, 800px total for image + text + padding)
        window_width = 700
        window_height = 735

        # Get screen size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate position
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def _create_image_label(self, parent: tk.Frame, image_path: Path) -> tk.Label:
        """Create label with image or placeholder.

        Args:
            parent: Parent frame for label
            image_path: Path to image file

        Returns:
            Label widget with image or placeholder text
        """
        try:
            if image_path.exists():
                image = Image.open(image_path)

                self.photo_image = ImageTk.PhotoImage(image)

                label = tk.Label(parent, image=self.photo_image, bg="white", bd=0, highlightthickness=0)
                label.pack(side=tk.TOP, anchor='n', pady=0, padx=0)  # Pack at top with no padding

                return label
            else:
                # Image not found - show placeholder
                return self._create_placeholder_label(parent, "AI Stenograper")

        except Exception as e:
            # Error loading image - show placeholder
            return self._create_placeholder_label(parent, f"Error loading image: {e}")

    def _create_placeholder_label(self, parent: tk.Frame, text: str) -> tk.Label:
        """Create placeholder label when image cannot be loaded.

        Args:
            parent: Parent frame for label
            text: Placeholder text

        Returns:
            Label widget with placeholder text
        """
        label = tk.Label(
            parent,
            text=text,
            font=("Arial", 12),
            bg="white",
            fg="gray",
            width=60,
            height=20
        )
        label.pack(pady=20)
        return label

    def update_message(self, message: str) -> None:
        """Update loading message text.

        Args:
            message: New message to display
        """
        self.message_label.config(text=message)

        # Process GUI events to show update immediately
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        """Close and destroy the loading window."""
        try:
            self.root.quit()
            self.root.destroy()
        except tk.TclError:
            # Window already destroyed
            pass
