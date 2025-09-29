#!/usr/bin/env python3
"""Debug script to test TextMatcher -> TwoStageDisplayHandler interaction."""

import queue
import sys
import os
import tkinter as tk
from tkinter import scrolledtext
import threading
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.TwoStageDisplayHandler import TwoStageDisplayHandler
from src.TextMatcher import TextMatcher
from src.GuiWindow import create_stt_window

def main():
    # Create GUI window
    root, text_widget = create_stt_window()
    root.withdraw()  # Hide window during debug

    # Create all the queues like in the real pipeline
    text_queue = queue.Queue()
    final_queue = queue.Queue()
    partial_queue = queue.Queue()

    # Create components
    text_matcher = TextMatcher(text_queue, final_queue, partial_queue)
    display = TwoStageDisplayHandler(final_queue, partial_queue, text_widget, root)

    print("Testing TextMatcher -> TwoStageDisplayHandler flow...")

    # Scenario 1: Simulate the same text being recognized multiple times
    # (This is what happens when STT keeps recognizing the same word)
    print("\nScenario 1: Same text recognized multiple times")

    # Input multiple instances of the same text to TextMatcher
    text_matcher.process_text({'text': 'Hello', 'timestamp': 1.0})
    text_matcher.process_text({'text': 'Hello', 'timestamp': 1.1})
    text_matcher.process_text({'text': 'Hello', 'timestamp': 1.2})
    text_matcher.process_text({'text': 'Hello', 'timestamp': 1.3})

    # Process the queues manually to see what gets queued
    print("Processing partial queue:")
    processed_partials = []
    while not partial_queue.empty():
        try:
            partial_data = partial_queue.get(timeout=0.01)
            processed_partials.append(partial_data['text'])
            print(f"  Partial: '{partial_data['text']}'")
            display.update_preliminary_text(partial_data['text'])
            root.update()
            content = text_widget.get("1.0", tk.END).rstrip('\n')
            print(f"  Widget after partial: '{content}'")
        except queue.Empty:
            break

    print("Processing final queue:")
    while not final_queue.empty():
        try:
            final_data = final_queue.get(timeout=0.01)
            print(f"  Final: '{final_data['text']}'")
            display.finalize_text(final_data['text'])
            root.update()
            content = text_widget.get("1.0", tk.END).rstrip('\n')
            print(f"  Widget after final: '{content}'")
        except queue.Empty:
            break

    final_content = text_widget.get("1.0", tk.END).rstrip('\n')
    hello_count = final_content.count('Hello')
    print(f"Final content: '{final_content}'")
    print(f"Hello count: {hello_count}")

    if hello_count > 1:
        print("[ERROR] TextMatcher flow causes duplication!")
    else:
        print("[OK] TextMatcher flow looks good")

    # Scenario 2: Test with actual process_queues() method
    print("\nScenario 2: Using process_queues() method")

    # Clear widget and reset
    text_widget.delete("1.0", tk.END)
    display.preliminary_start_pos = "1.0"

    # Add multiple same texts to partial queue directly
    partial_queue.put({'text': 'Hello'})
    partial_queue.put({'text': 'Hello'})
    partial_queue.put({'text': 'Hello'})
    partial_queue.put({'text': 'Hello'})

    # Start display handler like in the real pipeline
    display.is_running = True
    display_thread = threading.Thread(target=display.process_queues, daemon=True)
    display_thread.start()

    # Let it run briefly
    time.sleep(0.1)

    # Process GUI updates
    for _ in range(20):
        root.update()
        root.update_idletasks()
        time.sleep(0.01)

    # Stop the thread
    display.is_running = False
    display_thread.join(timeout=0.2)

    process_content = text_widget.get("1.0", tk.END).rstrip('\n')
    process_hello_count = process_content.count('Hello')
    print(f"Process queues content: '{process_content}'")
    print(f"Process queues Hello count: {process_hello_count}")

    if process_hello_count > 1:
        print("[ERROR] process_queues() causes duplication!")
    else:
        print("[OK] process_queues() looks good")

    # Scenario 3: Test mixed sequence like real speech
    print("\nScenario 3: Mixed sequence (real speech pattern)")

    # Clear widget and reset everything
    text_widget.delete("1.0", tk.END)
    display.preliminary_start_pos = "1.0"

    # Clear any remaining queue items
    while not partial_queue.empty():
        partial_queue.get_nowait()
    while not final_queue.empty():
        final_queue.get_nowait()

    # Simulate real pattern: partial -> partial -> final -> partial -> final
    print("Adding mixed sequence to queues...")
    partial_queue.put({'text': 'Hello'})
    partial_queue.put({'text': 'Hello world'})
    final_queue.put({'text': 'Hello'})
    partial_queue.put({'text': 'there'})
    final_queue.put({'text': 'there'})

    # Start processing again
    display.is_running = True
    display_thread = threading.Thread(target=display.process_queues, daemon=True)
    display_thread.start()

    time.sleep(0.1)

    for _ in range(30):
        root.update()
        root.update_idletasks()
        time.sleep(0.01)

    display.is_running = False
    display_thread.join(timeout=0.2)

    mixed_content = text_widget.get("1.0", tk.END).rstrip('\n')
    print(f"Mixed sequence content: '{mixed_content}'")

    # Check for any word duplications
    words = mixed_content.split()
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    duplicates_found = False
    for word, count in word_counts.items():
        if count > 1:
            print(f"[ERROR] Word '{word}' appears {count} times!")
            duplicates_found = True

    if not duplicates_found:
        print("[OK] No word duplications in mixed sequence")

    print("\nDone.")
    root.destroy()

if __name__ == '__main__':
    main()