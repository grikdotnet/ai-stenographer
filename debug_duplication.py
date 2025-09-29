#!/usr/bin/env python3
"""Debug script to reproduce the duplication issue."""

import queue
import sys
import os
import tkinter as tk
from tkinter import scrolledtext

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.TwoStageDisplayHandler import TwoStageDisplayHandler
from src.GuiWindow import create_stt_window

def main():
    # Create GUI window
    root, text_widget = create_stt_window()
    root.withdraw()  # Hide window during debug

    # Create queues
    final_queue = queue.Queue()
    partial_queue = queue.Queue()

    # Create display handler
    display = TwoStageDisplayHandler(final_queue, partial_queue, text_widget, root)

    print("Testing duplication scenario...")

    # Simulate what happens when the same word "Hello" is recognized multiple times
    # This pattern often happens in real speech recognition

    print("1. First partial: 'Hello'")
    display.update_preliminary_text('Hello')
    root.update()
    content = text_widget.get("1.0", tk.END).rstrip('\n')
    print(f"   Widget content: '{content}'")

    print("2. Second partial: 'Hello' (same text)")
    display.update_preliminary_text('Hello')
    root.update()
    content = text_widget.get("1.0", tk.END).rstrip('\n')
    print(f"   Widget content: '{content}'")

    print("3. Third partial: 'Hello' (same text again)")
    display.update_preliminary_text('Hello')
    root.update()
    content = text_widget.get("1.0", tk.END).rstrip('\n')
    print(f"   Widget content: '{content}'")

    print("4. Finalize: 'Hello'")
    display.finalize_text('Hello')
    root.update()
    content = text_widget.get("1.0", tk.END).rstrip('\n')
    print(f"   Widget content: '{content}'")

    print("\nTesting position corruption scenario...")

    # Simulate position corruption
    print("5. Manually corrupt position")
    display.preliminary_start_pos = "999.0"
    print(f"   Corrupted position: {display.preliminary_start_pos}")

    print("6. Add partial text with corrupted position: 'World'")
    display.update_preliminary_text('World')
    root.update()
    content = text_widget.get("1.0", tk.END).rstrip('\n')
    print(f"   Widget content: '{content}'")
    print(f"   Position after recovery: {display.preliminary_start_pos}")

    print("\nTesting rapid queue processing scenario...")

    # Clear the widget
    text_widget.delete("1.0", tk.END)
    display.preliminary_start_pos = "1.0"

    # Simulate what happens in the real pipeline with multiple rapid partial updates
    print("7. Simulating real speech recognition pattern:")

    # Multiple partial updates for the same word (like when recognition is uncertain)
    partial_queue.put({'text': 'Hello'})
    partial_queue.put({'text': 'Hello'})
    partial_queue.put({'text': 'Hello'})
    partial_queue.put({'text': 'Hello'})

    # Process them manually (like process_queues would)
    processed_count = 0
    while not partial_queue.empty() and processed_count < 10:
        try:
            partial_data = partial_queue.get(timeout=0.01)
            text = partial_data['text']
            print(f"   Processing partial: '{text}'")
            display.update_preliminary_text(text)
            root.update()
            content = text_widget.get("1.0", tk.END).rstrip('\n')
            print(f"   Widget content after update: '{content}'")
            processed_count += 1
        except queue.Empty:
            break

    final_content = text_widget.get("1.0", tk.END).rstrip('\n')
    print(f"Final widget content: '{final_content}'")

    # Count occurrences
    hello_count = final_content.count('Hello')
    print(f"Number of 'Hello' occurrences: {hello_count}")

    if hello_count > 1:
        print("[ERROR] DUPLICATION DETECTED!")
    else:
        print("[OK] No duplication")

    print("\nTesting with threading simulation...")

    # Clear and reset
    text_widget.delete("1.0", tk.END)
    display.preliminary_start_pos = "1.0"

    # Test with root.after() calls (simulating background thread)
    def delayed_update(text, delay_ms=0):
        root.after(delay_ms, lambda: display.update_preliminary_text(text))

    print("8. Testing with root.after() delays:")
    delayed_update('Hello', 0)
    delayed_update('Hello', 1)
    delayed_update('Hello', 2)
    delayed_update('Hello', 3)

    # Process all pending GUI updates
    for _ in range(50):
        root.update()
        root.update_idletasks()

    threading_content = text_widget.get("1.0", tk.END).rstrip('\n')
    print(f"Threading test content: '{threading_content}'")
    threading_count = threading_content.count('Hello')
    print(f"Number of 'Hello' occurrences in threading test: {threading_count}")

    if threading_count > 1:
        print("[ERROR] THREADING DUPLICATION DETECTED!")
    else:
        print("[OK] No threading duplication")

    print("\nDone.")
    root.destroy()

if __name__ == '__main__':
    main()