"""
LoadingWindow displays a loading screen with image and progress messages.
"""
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from typing import Optional, List, Dict, Callable
import threading
import os


# Color scheme matching ModelDownloadDialog
COLORS = {
    'bg_light': '#F0F3F5',          # Light blue-gray background
    'text_dark': '#333333',         # Dark text for titles/labels
    'text_gray': '#666666',         # Gray text for secondary info
    'progress_teal': '#1DAD3B',     # Teal/cyan for progress bars
    'progress_bg': '#E0E0E0',       # Light gray progress background
    'button_bg': '#FFFFFF',         # White button background
    'button_border': '#CCCCCC',     # Button border
    'button_active': '#E8E8E8',     # Button hover/active
    'success_green': '#4CAF50',     # Success checkmark
    'error_red': '#F44336'          # Error indicator
}

# Model size information
MODEL_SIZES = {
    'parakeet': '~1.2 GB',
    'silero_vad': '2 MB'
}

MODEL_NAMES = {
    'parakeet': 'Parakeet STT Model',
    'silero_vad': 'Silero VAD Model'
}


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes as human-readable string.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string (e.g., '1.5 GB', '256 MB')
    """
    if bytes_value < 1024:
        return f"{bytes_value} B"
    elif bytes_value < 1024 * 1024:
        return f"{bytes_value / 1024:.1f} KB"
    elif bytes_value < 1024 * 1024 * 1024:
        return f"{bytes_value / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_value / (1024 * 1024 * 1024):.2f} GB"


class LoadingWindow:
    """Displays loading screen with image and progress messages.

    Shows stenographer.gif image centered in an 800x800 window
    with loading status messages below. Window is modal and centered on screen.
    """

    def __init__(self, image_path: Path, initial_message: str) -> None:
        """Create loading window with image and initial message.

        Args:
            image_path: Path to image file to display (e.g., stenographer.gif)
            initial_message: Initial loading message to show
        """
        self.root: tk.Tk = tk.Tk()
        self.root.title("Loading AI Stenographer...")
        self.root.overrideredirect(True)  # Remove title bar and borders
        self.root.geometry("700x700")
        self.root.resizable(False, False)

        self._center_window()

        # Store image path for later recreation if needed
        self._image_path: Path = image_path

        # Create main frame with no padding or borders
        container = tk.Frame(self.root, bg="white", bd=0, highlightthickness=0)
        container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        self.photo_image: Optional[tk.PhotoImage] = None
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
                self.photo_image = tk.PhotoImage(file=str(image_path))

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

    def transform_to_download_dialog(self, missing_models: List[str], model_dir: Path) -> bool:
        """
        Transform loading window into model download dialog.

        Keeps the existing window and image visible, adds download UI below.
        This avoids the lag from destroying and recreating windows.

        Args:
            missing_models: List of model names to download ('parakeet', 'silero_vad')
            model_dir: Directory to download models to

        Returns:
            True if download succeeded, False if user cancelled
        """
        # Import ModelManager here to avoid circular dependencies
        from src.asr.ModelManager import ModelManager

        # Store the container frame reference
        container = None
        for child in self.root.winfo_children():
            if isinstance(child, tk.Frame):
                container = child
                break

        if not container:
            raise RuntimeError("Could not find container frame in LoadingWindow")

        # Restore window decorations (title bar, borders) so user can move window
        self.root.overrideredirect(False)

        # Clear the message label
        if self.message_label:
            self.message_label.destroy()
            self.message_label = None

        # Remove the image (download dialog doesn't need it)
        if self.image_label:
            self.image_label.destroy()
            self.image_label = None

        # Resize window to accommodate download UI (smaller without image)
        self.root.geometry("500x400")
        self._center_window_at_size(500, 400)

        # Update window title (now visible in title bar)
        self.root.title("Required Models Not Found")

        # Result tracking
        result = {'success': False}

        # Progress tracking
        status_labels: Dict[str, tk.Label] = {}
        progress_bars: Dict[str, ttk.Progressbar] = {}

        # Configure ttk style for progress bars
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Teal.Horizontal.TProgressbar",
            troughcolor=COLORS['progress_bg'],
            background=COLORS['progress_teal'],
            bordercolor=COLORS['progress_bg'],
            lightcolor=COLORS['progress_teal'],
            darkcolor=COLORS['progress_teal']
        )

        # Title label
        title_label = tk.Label(
            container,
            text="The following models need to be downloaded:",
            font=("Arial", 11),
            justify=tk.LEFT,
            bg="white",
            fg=COLORS['text_dark']
        )
        title_label.pack(anchor=tk.W, pady=(0, 5), padx=20)

        # Models frame
        models_frame = tk.Frame(container, bg="white")
        models_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15), padx=20)

        # Create progress bars for each model
        for model in missing_models:
            model_frame = tk.Frame(models_frame, bg="white")
            model_frame.pack(fill=tk.X, pady=5)

            # Model label
            model_label = tk.Label(
                model_frame,
                text=f"{MODEL_NAMES.get(model, model)} ({MODEL_SIZES.get(model, 'unknown size')})",
                font=("Arial", 9),
                justify=tk.LEFT,
                bg="white",
                fg=COLORS['text_dark']
            )
            model_label.pack(anchor=tk.W)

            # Progress bar
            progress_bar = ttk.Progressbar(
                model_frame,
                mode='determinate',
                length=400,
                style="Teal.Horizontal.TProgressbar"
            )
            progress_bar.pack(anchor=tk.W, fill=tk.X, pady=(2, 0))
            progress_bars[model] = progress_bar

            # Status label
            status_label = tk.Label(
                model_frame,
                text="",
                font=("Arial", 9),
                fg=COLORS['text_gray'],
                bg="white"
            )
            status_label.pack(anchor=tk.W, pady=(2, 0))
            status_labels[model] = status_label

        # License label
        license_label = tk.Label(
            container,
            text="The Parakeet-TDT-0.6B-v3 model by NVIDIA (CC BY 4.0)\n"+
                "   License: https://creativecommons.org/licenses/by/4.0/\n"+
                "The Silero VAD model by Silero Team (MIT)\n"+
                "   License: https://opensource.org/licenses/MIT.",
            font=("Arial", 9),
            fg=COLORS['text_dark'],
            bg="white",
            justify=tk.LEFT
        )
        license_label.pack(anchor=tk.W, pady=(0, 15), padx=20)

        # Button frame
        button_frame = tk.Frame(container, bg="white")
        button_frame.pack(side=tk.BOTTOM, pady=(10, 20))

        download_state = {'active': False, 'cancelled': False}

        def progress_callback(
            model_name: str,
            progress: float,
            status: str,
            downloaded_bytes: int,
            total_bytes: int
        ):
            """Updates progress UI from download thread."""
            def update_ui():
                try:
                    if model_name not in status_labels or model_name not in progress_bars:
                        return

                    # Update progress bar
                    progress_value = progress * 100
                    progress_bars[model_name]['value'] = progress_value

                    # Update status text
                    if status == 'downloading':
                        if total_bytes > 0:
                            status_text = f"Downloading... {format_bytes(downloaded_bytes)} / {format_bytes(total_bytes)} ({progress:.0%})"
                        else:
                            status_text = "Downloading... (preparing)"
                        status_labels[model_name].config(text=status_text, fg=COLORS['text_dark'])
                    elif status == 'complete':
                        status_labels[model_name].config(text="✓ Complete", fg=COLORS['success_green'])
                        progress_bars[model_name]['value'] = 100
                    elif status == 'error':
                        status_labels[model_name].config(text="✗ Error", fg=COLORS['error_red'])

                    # Force redraw
                    progress_bars[model_name].update_idletasks()
                    status_labels[model_name].update_idletasks()
                except Exception:
                    pass

            # Schedule update on main thread
            self.root.after(0, update_ui)

        def on_download():
            """Handles download button click."""
            download_btn.config(state=tk.DISABLED)
            exit_btn.config(state=tk.DISABLED)

            download_state['active'] = True

            def download_thread():
                """Runs download in background thread."""
                try:
                    if download_state['cancelled']:
                        download_state['active'] = False
                        return

                    success = ModelManager.download_models(model_dir=model_dir, progress_callback=progress_callback)

                    if download_state['cancelled']:
                        download_state['active'] = False
                        return

                    def finish_download():
                        download_state['active'] = False
                        if success:
                            result['success'] = True
                            self.root.quit()
                        else:
                            messagebox.showerror("Error", "Download failed. Please try again.", parent=self.root)
                            download_btn.config(state=tk.NORMAL)
                            exit_btn.config(state=tk.NORMAL)

                    self.root.after(0, finish_download)

                except Exception as exc:
                    def show_error():
                        download_state['active'] = False
                        messagebox.showerror("Error", f"Download failed: {exc}", parent=self.root)
                        download_btn.config(state=tk.NORMAL)
                        exit_btn.config(state=tk.NORMAL)

                    self.root.after(0, show_error)

            thread = threading.Thread(target=download_thread, daemon=True)
            thread.start()

        def on_exit():
            """Handles exit button click."""
            download_state['cancelled'] = True
            result['success'] = False

            if download_state['active']:
                os._exit(0)

            self.root.quit()

        def on_close():
            """Handles window close button."""
            download_state['cancelled'] = True
            result['success'] = False

            if download_state['active']:
                os._exit(0)

            self.root.quit()

        self.root.protocol("WM_DELETE_WINDOW", on_close)

        # Create buttons
        download_btn = tk.Button(
            button_frame,
            text="Download Models",
            command=on_download,
            width=15,
            font=("Arial", 10),
            bg=COLORS['button_bg'],
            fg=COLORS['text_dark'],
            activebackground=COLORS['button_active'],
            relief=tk.RAISED,
            bd=1
        )
        download_btn.pack(side=tk.LEFT, padx=(0, 10))

        exit_btn = tk.Button(
            button_frame,
            text="Exit",
            command=on_exit,
            width=15,
            font=("Arial", 10),
            bg=COLORS['button_bg'],
            fg=COLORS['text_dark'],
            activebackground=COLORS['button_active'],
            relief=tk.RAISED,
            bd=1
        )
        exit_btn.pack(side=tk.LEFT)

        # Store buttons for testing
        self._download_btn = download_btn
        self._exit_btn = exit_btn

        # Run mainloop
        self.root.mainloop()

        return result['success']

    def _center_window_at_size(self, width: int, height: int) -> None:
        """Center window on screen with specified size.

        Args:
            width: Window width in pixels
            height: Window height in pixels
        """
        self.root.update_idletasks()

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def transform_back_to_loading(self, message: str) -> None:
        """
        Transform download dialog back to loading screen.

        Removes download UI widgets and restores original layout with
        image + message label.

        Args:
            message: Message to display (e.g., "Models downloaded successfully")
        """
        # Find container frame
        container = None
        for child in self.root.winfo_children():
            if isinstance(child, tk.Frame):
                container = child
                break

        if not container:
            raise RuntimeError("Could not find container frame in LoadingWindow")

        # Destroy all widgets in container
        for widget in container.winfo_children():
            widget.destroy()

        # Restore borderless window
        self.root.overrideredirect(True)

        # Resize window back to original size
        self.root.geometry("700x800")
        self._center_window_at_size(700, 735)

        # Restore window title
        self.root.title("Loading AI Stenographer...")

        # Recreate image label (if we have the image path stored)
        if hasattr(self, '_image_path'):
            self.image_label = self._create_image_label(container, self._image_path)
        else:
            # No image path stored, create placeholder
            self.image_label = self._create_placeholder_label(container, "AI Stenographer")

        # Recreate message label
        self.message_label = tk.Label(
            container,
            text=message,
            font=("Arial", 14),
            bg="white",
            fg="black"
        )
        self.message_label.pack(pady=0)

        # Force update
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
