"""
ModelDownloadDialog provides GUI for downloading missing models.
"""
import tkinter as tk
from tkinter import messagebox
from typing import List, Dict
import threading
import os
from src.ModelManager import ModelManager


# Model size information
MODEL_SIZES = {
    'parakeet': '~1.5 GB',
    'silero_vad': '2 MB'
}

MODEL_NAMES = {
    'parakeet': 'Parakeet STT Model',
    'silero_vad': 'Silero VAD Model'
}


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

    # Result tracking
    result = {'success': False}

    # Progress tracking (text-only, no animated progress bars)
    status_labels: Dict[str, tk.Label] = {}

    # Main content frame
    content_frame = tk.Frame(dialog, padx=20, pady=20)
    content_frame.pack(fill=tk.BOTH, expand=True)

    # Title message
    title_label = tk.Label(
        content_frame,
        text="The following models need to be downloaded:",
        font=("Arial", 11),
        justify=tk.LEFT
    )
    title_label.pack(anchor=tk.W, pady=(0, 5))

    # Models frame
    models_frame = tk.Frame(content_frame)
    models_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

    # Create checkboxes and progress bars for each model
    for i, model in enumerate(missing_models):
        model_frame = tk.Frame(models_frame)
        model_frame.pack(fill=tk.X, pady=5)

        # Model label with checkbox
        model_label = tk.Label(
            model_frame,
            text=f"☐ {MODEL_NAMES.get(model, model)} ({MODEL_SIZES.get(model, 'unknown size')})",
            font=("Arial", 10),
            justify=tk.LEFT
        )
        model_label.pack(anchor=tk.W)

        # Status label (text-only progress)
        status_label = tk.Label(
            model_frame,
            text="",
            font=("Arial", 9),
            fg="gray"
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
        fg="gray",
        justify=tk.LEFT
    )
    license_label.pack(anchor=tk.W, pady=(0, 15))


    # Buttons frame
    button_frame = tk.Frame(content_frame)
    button_frame.pack(side=tk.BOTTOM, pady=(10, 0))

    # Track if download is in progress and if cancelled
    download_state = {'active': False, 'cancelled': False}

    def on_download():
        """Handles download button click."""
        # Disable download button
        download_btn.config(state=tk.DISABLED)
        exit_btn.config(state=tk.DISABLED)

        # Mark download as active
        download_state['active'] = True

        def progress_callback(model_name: str, progress: float, status: str):
            """Updates progress UI from download thread."""
            def update_ui():
                if model_name in status_labels:
                    # Update status (text-only)
                    if status == 'downloading':
                        status_labels[model_name].config(
                            text="Downloading... (this may take time)",
                            fg="blue"
                        )
                    elif status == 'complete':
                        status_labels[model_name].config(text="✓ Complete", fg="green")
                    elif status == 'error':
                        status_labels[model_name].config(text="✗ Error", fg="red")

            dialog.after(0, update_ui)

        def download_thread():
            """Runs download in background thread."""
            try:
                # Check if cancelled before starting download
                if download_state['cancelled']:
                    download_state['active'] = False
                    return

                success = ModelManager.download_models(model_dir=model_dir, progress_callback=progress_callback)

                # Check if cancelled after download
                if download_state['cancelled']:
                    download_state['active'] = False
                    return

                def finish_download():
                    download_state['active'] = False  # Mark download as complete
                    if success:
                        result['success'] = True
                        dialog.quit()  # Exit mainloop
                        dialog.destroy()
                    else:
                        messagebox.showerror("Error", "Download failed. Please try again.", parent=dialog)
                        download_btn.config(state=tk.NORMAL)
                        exit_btn.config(state=tk.NORMAL)

                dialog.after(0, finish_download)

            except Exception as e:
                print(f"Download exception: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

                def show_error():
                    download_state['active'] = False  # Mark download as complete
                    messagebox.showerror("Error", f"Download failed: {str(e)}", parent=dialog)
                    download_btn.config(state=tk.NORMAL)
                    exit_btn.config(state=tk.NORMAL)

                dialog.after(0, show_error)

        # Start download thread
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

        # Force terminate if download is in progress
        if download_state['active']:
            # huggingface_hub downloads are blocking and cannot be cancelled gracefully
            # Use os._exit() to immediately terminate the process
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
            # huggingface_hub downloads are blocking and cannot be cancelled gracefully
            # Use os._exit() to immediately terminate the process
            os._exit(0)

        dialog.quit()  # Exit mainloop
        dialog.destroy()

    dialog.protocol("WM_DELETE_WINDOW", on_close)

    # Create buttons
    download_btn = tk.Button(
        button_frame,
        text="Download Models",
        command=on_download,
        width=15,
        font=("Arial", 10)
    )
    download_btn.pack(side=tk.LEFT, padx=(0, 10))

    exit_btn = tk.Button(
        button_frame,
        text="Exit",
        command=on_exit,
        width=15,
        font=("Arial", 10)
    )
    exit_btn.pack(side=tk.LEFT)

    # Center dialog on screen (not parent, since parent is withdrawn)
    dialog.update_idletasks()

    # Get screen dimensions
    screen_width = dialog.winfo_screenwidth()
    screen_height = dialog.winfo_screenheight()

    # Calculate center position
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
