"""
ModelDownloadDialog provides GUI for downloading missing models.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict
import threading
from src.GuiFactory import GuiFactory
from src.ModelManager import ModelManager


# Model size information
MODEL_SIZES = {
    'parakeet': '~1.5GB',
    'silero_vad': '~0.2GB'
}

MODEL_NAMES = {
    'parakeet': 'Parakeet STT Model',
    'silero_vad': 'Silero VAD Model'
}


def show_download_dialog(parent: tk.Tk, missing_models: List[str]) -> bool:
    """
    Shows modal dialog for downloading missing models.

    Args:
        parent: Parent Tk window (can be None or withdrawn, will create standalone window)
        missing_models: List of model names to download

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

    # Progress tracking
    progress_bars: Dict[str, ttk.Progressbar] = {}
    status_labels: Dict[str, tk.Label] = {}

    # Calculate total size
    total_size = sum(
        float(MODEL_SIZES[model].replace('~', '').replace('GB', ''))
        for model in missing_models
    )

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

    # Total size label
    size_label = tk.Label(
        content_frame,
        text=f"Total size: ~{total_size:.1f}GB",
        font=("Arial", 10),
        fg="gray"
    )
    size_label.pack(anchor=tk.W, pady=(0, 15))

    # Models frame
    models_frame = tk.Frame(content_frame)
    models_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

    # Create checkboxes and progress bars for each model
    for i, model in enumerate(missing_models):
        model_frame = tk.Frame(models_frame)
        model_frame.pack(fill=tk.X, pady=5)

        # Model label
        model_label = tk.Label(
            model_frame,
            text=f"☐ {MODEL_NAMES.get(model, model)} ({MODEL_SIZES.get(model, 'unknown size')})",
            font=("Arial", 10),
            justify=tk.LEFT
        )
        model_label.pack(anchor=tk.W)

        # Progress bar (initially hidden) - use indeterminate mode since HF doesn't provide progress
        progress_bar = GuiFactory.create_progress_bar(
            model_frame,
            length=400,
            mode='indeterminate'
        )
        progress_bars[model] = progress_bar

        # Status label
        status_label = tk.Label(
            model_frame,
            text="",
            font=("Arial", 9),
            fg="gray"
        )
        status_labels[model] = status_label

    # Buttons frame
    button_frame = tk.Frame(content_frame)
    button_frame.pack(side=tk.BOTTOM, pady=(10, 0))

    # Track if download is in progress
    downloading = {'active': False}

    def on_download():
        """Handles download button click."""
        # Disable download button
        download_btn.config(state=tk.DISABLED)
        exit_btn.config(state=tk.DISABLED)

        # Mark download as active
        downloading['active'] = True

        # Show progress bars
        for model in missing_models:
            progress_bars[model].pack(fill=tk.X, pady=(5, 0))
            status_labels[model].pack(anchor=tk.W)

        def progress_callback(model_name: str, progress: float, status: str):
            """Updates progress UI from download thread."""
            def update_ui():
                if model_name in progress_bars:
                    # Update status
                    if status == 'downloading':
                        # Start indeterminate animation
                        progress_bars[model_name].start(10)
                        status_labels[model_name].config(text="Downloading... (check terminal for progress)", fg="blue")
                    elif status == 'complete':
                        # Stop animation and show completion
                        progress_bars[model_name].stop()
                        progress_bars[model_name]['value'] = 100
                        status_labels[model_name].config(text="✓ Complete", fg="green")
                    elif status == 'error':
                        # Stop animation and show error
                        progress_bars[model_name].stop()
                        status_labels[model_name].config(text="✗ Error", fg="red")

            dialog.after(0, update_ui)

        def download_thread():
            """Runs download in background thread."""
            try:
                success = ModelManager.download_models(progress_callback=progress_callback)

                def finish_download():
                    if success:
                        result['success'] = True
                        messagebox.showinfo("Success", "Models downloaded successfully!", parent=dialog)
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
                    messagebox.showerror("Error", f"Download failed: {str(e)}", parent=dialog)
                    download_btn.config(state=tk.NORMAL)
                    exit_btn.config(state=tk.NORMAL)

                dialog.after(0, show_error)

        # Start download thread
        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

    def on_exit():
        """Handles exit button click."""
        result['success'] = False
        dialog.quit()  # Exit mainloop
        dialog.destroy()

    # Handle window close (X button)
    def on_close():
        """Handles window close button."""
        result['success'] = False
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
