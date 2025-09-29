#!/usr/bin/env python3
"""Debug script to test finalize_text duplication issue."""

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
    root.withdraw()

    # Create display handler (dummy queues since we're calling methods directly)
    import queue
    final_queue = queue.Queue()
    partial_queue = queue.Queue()
    display = TwoStageDisplayHandler(final_queue, partial_queue, text_widget, root)

    print("Testing finalize_text duplication issue...")

    # Scenario: Finalize text multiple times (like when TextMatcher sends duplicates)
    print("\n1. Multiple finalize_text calls with same text:")

    for i in range(4):
        print(f"   Call {i+1}: finalize_text('Hello')")
        display.finalize_text('Hello')
        root.update()
        content = text_widget.get("1.0", tk.END).rstrip('\n')
        print(f"   Widget content: '{content}'")
        print(f"   Position: {display.preliminary_start_pos}")

    final_content = text_widget.get("1.0", tk.END).rstrip('\n')
    hello_count = final_content.count('Hello')
    print(f"\nFinal content: '{final_content}'")
    print(f"Hello count: {hello_count}")

    if hello_count > 1:
        print("[ERROR] Multiple finalize_text calls cause duplication!")
    else:
        print("[OK] Multiple finalize_text calls work correctly")

    # Scenario 2: Position corruption during finalize
    print("\n2. Testing position corruption during finalize:")

    # Clear and start fresh
    text_widget.delete("1.0", tk.END)
    display.preliminary_start_pos = "1.0"

    # Add some preliminary text first
    display.update_preliminary_text('preliminary text')
    root.update()
    print(f"   After preliminary: '{text_widget.get('1.0', tk.END).rstrip()}'")
    print(f"   Position: {display.preliminary_start_pos}")

    # Now corrupt the position and finalize
    display.preliminary_start_pos = "999.0"  # Invalid position
    print(f"   Corrupted position to: {display.preliminary_start_pos}")

    display.finalize_text('Hello')
    root.update()
    content = text_widget.get("1.0", tk.END).rstrip('\n')
    print(f"   After finalize with corrupted position: '{content}'")
    print(f"   New position: {display.preliminary_start_pos}")

    # This should trigger the bug - finalize again
    display.finalize_text('World')
    root.update()
    content = text_widget.get("1.0", tk.END).rstrip('\n')
    print(f"   After second finalize: '{content}'")

    # Check for duplications
    words = content.split()
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    print(f"\nWord counts: {word_counts}")
    duplicates = [word for word, count in word_counts.items() if count > 1]
    if duplicates:
        print(f"[ERROR] Position corruption causes duplication of: {duplicates}")
    else:
        print("[OK] Position corruption handled correctly")

    print("\nDone.")
    root.destroy()

if __name__ == '__main__':
    main()