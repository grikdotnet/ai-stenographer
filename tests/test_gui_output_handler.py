import unittest
import queue
import tkinter as tk
from src.TwoStageDisplayHandler import TwoStageDisplayHandler

class MockTextWidget:
    def __init__(self):
        self.content = ""
        self.marks = {"preliminary_start": 0, "preliminary_end": 0}
        self.tags = {}  # position -> tag mapping
        self.tag_ranges = []  # list of (start, end, tag) tuples

    def insert(self, position, text, tag=None):
        if isinstance(position, str):
            # Handle mark names
            if position in self.marks:
                pos = self.marks[position]
            elif position == "END":
                pos = len(self.content)
            else:
                pos = int(position.split('.')[1]) if '.' in position else int(position)
        else:
            pos = position

        # Insert text at position
        self.content = self.content[:pos] + text + self.content[pos:]

        # Update marks that come after insertion point (but not exactly at insertion point)
        for mark_name, mark_pos in self.marks.items():
            if mark_pos > pos:
                self.marks[mark_name] = mark_pos + len(text)

        # Track tag for inserted text
        if tag:
            self.tag_ranges.append((pos, pos + len(text), tag))

    def delete(self, start_mark, end_mark):
        start_pos = self.marks[start_mark] if start_mark in self.marks else 0
        end_pos = self.marks[end_mark] if end_mark in self.marks else len(self.content)

        if start_pos < end_pos:
            deleted_length = end_pos - start_pos
            # Remove text
            self.content = self.content[:start_pos] + self.content[end_pos:]

            # Update marks after deletion point
            for mark_name, mark_pos in self.marks.items():
                if mark_pos > end_pos:
                    self.marks[mark_name] = mark_pos - deleted_length
                elif mark_pos > start_pos:
                    self.marks[mark_name] = start_pos

            # Remove affected tag ranges
            self.tag_ranges = [
                (start, end, tag) for start, end, tag in self.tag_ranges
                if not (start < end_pos and end > start_pos)
            ]

    def mark_set(self, mark_name, position):
        if isinstance(position, str):
            if position == "END" or position == tk.END:
                pos = len(self.content)
            elif position in self.marks:
                pos = self.marks[position]
            else:
                pos = int(position.split('.')[1]) if '.' in position else int(position)
        else:
            pos = position

        self.marks[mark_name] = pos

    def index(self, mark_name):
        if mark_name == "END" or mark_name == tk.END:
            return str(len(self.content))
        return str(self.marks.get(mark_name, 0))

    def get_text_with_tag(self, tag):
        result = ""
        for start, end, tag_name in self.tag_ranges:
            if tag_name == tag:
                result += self.content[start:end]
        return result

    def tag_configure(self, tag_name, **kwargs):
        pass

    def see(self, position):
        pass


class TestTwoStageDisplay(unittest.TestCase):

    def setUp(self):
        self.final_queue = queue.Queue()
        self.partial_queue = queue.Queue()
        self.mock_text_widget = MockTextWidget()
        self.display = TwoStageDisplayHandler(
            final_queue=self.final_queue,
            partial_queue=self.partial_queue,
            text_widget=self.mock_text_widget
        )

    def test_realistic_speech_recognition_flow(self):
        # Step 1: Initial partial text "hello world ho"
        self.display.update_preliminary_text("hello world ho")

        # Verify step 1
        self.assertEqual(self.mock_text_widget.content, "hello world ho")
        self.assertEqual(self.mock_text_widget.get_text_with_tag("preliminary"), "hello world ho")
        self.assertEqual(self.mock_text_widget.marks["preliminary_start"], 0)
        self.assertEqual(self.mock_text_widget.marks["preliminary_end"], 14)

        # Step 2: Finalize "hello world"
        self.display.finalize_text("hello world")

        # Verify step 2
        self.assertEqual(self.mock_text_widget.content, "hello world ")
        self.assertEqual(self.mock_text_widget.get_text_with_tag("final"), "hello world ")
        self.assertEqual(self.mock_text_widget.get_text_with_tag("preliminary"), "")
        self.assertEqual(self.mock_text_widget.marks["preliminary_start"], 12)
        self.assertEqual(self.mock_text_widget.marks["preliminary_end"], 12)

        # Step 3: Add partial text "how are you do"
        self.display.update_preliminary_text("how are you do")

        # Verify step 3
        self.assertEqual(self.mock_text_widget.content, "hello world how are you do")
        self.assertEqual(self.mock_text_widget.get_text_with_tag("final"), "hello world ")
        self.assertEqual(self.mock_text_widget.get_text_with_tag("preliminary"), "how are you do")
        self.assertEqual(self.mock_text_widget.marks["preliminary_start"], 12)
        self.assertEqual(self.mock_text_widget.marks["preliminary_end"], 26)

        # Step 4: Finalize "how are you"
        self.display.finalize_text("how are you")

        # Verify step 4
        self.assertEqual(self.mock_text_widget.content, "hello world how are you ")
        self.assertEqual(self.mock_text_widget.get_text_with_tag("final"), "hello world how are you ")
        self.assertEqual(self.mock_text_widget.get_text_with_tag("preliminary"), "")
        self.assertEqual(self.mock_text_widget.marks["preliminary_start"], 24)
        self.assertEqual(self.mock_text_widget.marks["preliminary_end"], 24)

        # Step 5: Add partial text "doing great today"
        self.display.update_preliminary_text("doing great today")

        # Verify step 5
        self.assertEqual(self.mock_text_widget.content, "hello world how are you doing great today")
        self.assertEqual(self.mock_text_widget.get_text_with_tag("final"), "hello world how are you ")
        self.assertEqual(self.mock_text_widget.get_text_with_tag("preliminary"), "doing great today")
        self.assertEqual(self.mock_text_widget.marks["preliminary_start"], 24)
        self.assertEqual(self.mock_text_widget.marks["preliminary_end"], 41)

        # Step 6: Finalize "doing great today"
        self.display.finalize_text("doing great today")

        # Verify step 6
        self.assertEqual(self.mock_text_widget.content, "hello world how are you doing great today ")
        self.assertEqual(self.mock_text_widget.get_text_with_tag("final"), "hello world how are you doing great today ")
        self.assertEqual(self.mock_text_widget.get_text_with_tag("preliminary"), "")
        self.assertEqual(self.mock_text_widget.marks["preliminary_start"], 42)
        self.assertEqual(self.mock_text_widget.marks["preliminary_end"], 42)


if __name__ == '__main__':
    unittest.main()