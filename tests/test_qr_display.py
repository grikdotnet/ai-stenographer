"""Tests for qr_display module.

Tests verify:
- print_qr_code outputs lines containing block Unicode characters
- Output width is non-trivial and consistent across calls
- Works with the ws://127.0.0.1:<port> URL format used in server-only mode
"""

import io
import sys
import pytest

from src.server.qr_display import print_qr_code


BLOCK_CHARS = {"█", "▀", "▄", " "}


def capture_output(url: str) -> list[str]:
    buf = io.StringIO()
    print_qr_code(url, out=buf)
    return buf.getvalue().splitlines()


class TestPrintQrCode:
    def test_outputs_block_characters(self):
        lines = capture_output("ws://127.0.0.1:12345")
        all_chars = set("".join(lines))
        assert all_chars <= BLOCK_CHARS

    def test_outputs_multiple_lines(self):
        lines = capture_output("ws://127.0.0.1:12345")
        assert len(lines) > 5

    def test_all_lines_same_width(self):
        lines = capture_output("ws://127.0.0.1:12345")
        widths = {len(line) for line in lines if line}
        assert len(widths) == 1

    def test_output_width_reasonable(self):
        lines = capture_output("ws://127.0.0.1:12345")
        width = len(lines[0])
        assert 20 < width < 120

    def test_different_urls_produce_different_output(self):
        lines_a = capture_output("ws://127.0.0.1:11111")
        lines_b = capture_output("ws://127.0.0.1:22222")
        assert lines_a != lines_b

    def test_same_url_produces_deterministic_output(self):
        url = "ws://127.0.0.1:9876"
        assert capture_output(url) == capture_output(url)

    def test_default_out_is_stdout(self, monkeypatch):
        buf = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buf)
        print_qr_code("ws://127.0.0.1:12345")
        assert len(buf.getvalue().splitlines()) > 5
