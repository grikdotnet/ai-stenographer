"""
ModelDownloadDialog provides GUI for downloading missing models.
"""
import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Dict
import threading
import os
from src.ModelManager import ModelManager


# Color scheme matching reference image
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
    'parakeet': '~1.5 GB',
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


def show_download_dialog(parent: tk.Tk, missing_models: List[str], model_dir=None) -> bool:
    """
    Shows modal dialog for downloading missing models.

    Args:
        parent: Parent Tk window (can be None or withdrawn, will create standalone window)
        missing_models: List of model names to download
        model_dir: Directory to download models to (defaults to ./models)

    Returns:
        True if download succeeded, False if user cancelled
    """
    # Create standalone dialog window instead of Toplevel
    # This fixes issues with withdrawn parents on Windows
    dialog = tk.Tk()
    dialog.title("Required Models Not Found")
    dialog.geometry("500x400")
    dialog.configure(bg=COLORS['bg_light'])

    result = {'success': False}

    # Progress tracking
    status_labels: Dict[str, tk.Label] = {}
    progress_bars: Dict[str, ttk.Progressbar] = {}

    # Configure ttk style for progress bars
    style = ttk.Style()
    style.theme_use('clam')  # Use 'clam' theme for better customization
    style.configure(
        "Teal.Horizontal.TProgressbar",
        troughcolor=COLORS['progress_bg'],
        background=COLORS['progress_teal'],
        bordercolor=COLORS['progress_bg'],
        lightcolor=COLORS['progress_teal'],
        darkcolor=COLORS['progress_teal']
    )

    # Main content frame
    content_frame = tk.Frame(dialog, padx=20, pady=20, bg=COLORS['bg_light'])
    content_frame.pack(fill=tk.BOTH, expand=True)

    title_label = tk.Label(
        content_frame,
        text="The following models need to be downloaded:",
        font=("Arial", 11),
        justify=tk.LEFT,
        bg=COLORS['bg_light'],
        fg=COLORS['text_dark']
    )
    title_label.pack(anchor=tk.W, pady=(0, 5))

    # Models frame
    models_frame = tk.Frame(content_frame, bg=COLORS['bg_light'])
    models_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

    # Create checkboxes and progress bars for each model
    for i, model in enumerate(missing_models):
        model_frame = tk.Frame(models_frame, bg=COLORS['bg_light'])
        model_frame.pack(fill=tk.X, pady=5)

        # Model label with checkbox
        model_label = tk.Label(
            model_frame,
            text=f"{MODEL_NAMES.get(model, model)} ({MODEL_SIZES.get(model, 'unknown size')})",
            font=("Arial", 9),
            justify=tk.LEFT,
            bg=COLORS['bg_light'],
            fg=COLORS['text_dark']
        )
        model_label.pack(anchor=tk.W)

        # Progress bar for this model
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
            bg=COLORS['bg_light']
        )
        status_label.pack(anchor=tk.W, pady=(2, 0))
        status_labels[model] = status_label

    # License reference text
    license_label = tk.Label(
        content_frame,
        text="The Parakeet-TDT-0.6B-v3 model by NVIDIA (CC BY 4.0)\n"+
            "   License: https://creativecommons.org/licenses/by/4.0/\n"+
            "The Silero VAD model by Silero Team (MIT)\n"+
            "   License: https://opensource.org/licenses/MIT.",
        font=("Arial", 9),
        fg=COLORS['text_dark'],
        bg=COLORS['bg_light'],
        justify=tk.LEFT
    )
    license_label.pack(anchor=tk.W, pady=(0, 15))

    button_frame = tk.Frame(content_frame, bg=COLORS['bg_light'])
    button_frame.pack(side=tk.BOTTOM, pady=(10, 0))

    download_state = {'active': False, 'cancelled': False}

    def on_download():
        """Handles download button click."""
        download_btn.config(state=tk.DISABLED)
        exit_btn.config(state=tk.DISABLED)

        download_state['active'] = True

        def progress_callback(
            model_name: str,
            progress: float,
            status: str,
            downloaded_bytes: int,
            total_bytes: int
        ):
            """Updates progress UI from download thread."""
            print(f"[CALLBACK] Model: {model_name}, Progress: {progress:.1%}, Status: {status}, Bytes: {downloaded_bytes}/{total_bytes}")

            def update_ui():
                try:
                    if model_name not in status_labels or model_name not in progress_bars:
                        print(f"[UPDATE_UI] Model {model_name} not found in widgets!")
                        return

                    # Update progress bar
                    progress_value = progress * 100  # 0-100 scale
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

                    # Force the widgets to redraw
                    progress_bars[model_name].update_idletasks()
                    status_labels[model_name].update_idletasks()
                except Exception as e:
                    import traceback
                    print(f"[UPDATE_UI] GUI update error: {e}")
                    traceback.print_exc()

            # Schedule update on main thread
            dialog.after(0, update_ui)

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
                    download_state['active'] = False  # Mark download as complete
                    if success:
                        result['success'] = True
                        dialog.quit()
                        dialog.destroy()
                    else:
                        messagebox.showerror("Error", "Download failed. Please try again.", parent=dialog)
                        download_btn.config(state=tk.NORMAL)
                        exit_btn.config(state=tk.NORMAL)

                dialog.after(0, finish_download)

            except Exception as exc:
                print(f"Download exception: {type(exc).__name__}: {exc}")
                import traceback
                traceback.print_exc()

                error_msg = str(exc)

                def show_error():
                    download_state['active'] = False  # Mark download as complete
                    messagebox.showerror("Error", f"Download failed: {error_msg}", parent=dialog)
                    download_btn.config(state=tk.NORMAL)
                    exit_btn.config(state=tk.NORMAL)

                dialog.after(0, show_error)

        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

    def on_exit():
        """
        Handles exit button click.

        Forces process termination if download is active to prevent
        pythonw.exe from hanging on blocking network operations.
        """
        download_state['cancelled'] = True
        result['success'] = False

        if download_state['active']:
            os._exit(0)

        dialog.quit()  # Exit mainloop
        dialog.destroy()

    # Handle window close (X button)
    def on_close():
        """
        Handles window close button.

        Forces process termination if download is active to prevent
        pythonw.exe from hanging on blocking network operations.
        """
        download_state['cancelled'] = True
        result['success'] = False

        # Force terminate if download is in progress
        if download_state['active']:
            os._exit(0)

        dialog.quit()
        dialog.destroy()

    dialog.protocol("WM_DELETE_WINDOW", on_close)

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

    # Center dialog on screen (not parent)
    dialog.update_idletasks()

    screen_width = dialog.winfo_screenwidth()
    screen_height = dialog.winfo_screenheight()

    dialog_width = dialog.winfo_width()
    dialog_height = dialog.winfo_height()
    x = (screen_width // 2) - (dialog_width // 2)
    y = (screen_height // 2) - (dialog_height // 2)

    dialog.geometry(f"+{x}+{y}")

    # Make sure dialog is visible and has focus
    dialog.deiconify()
    dialog.lift()
    dialog.focus_force()

    # Run the dialog's main loop (needed for .after() to work)
    dialog.mainloop()

    return result['success']
