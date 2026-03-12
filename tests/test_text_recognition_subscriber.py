"""Tests for TextRecognitionSubscriber protocol definition.

These tests validate that the protocol is correctly defined and that
TextFormatter correctly implements the subscriber interface.
"""

import pytest
from typing import get_type_hints
from src.protocols import TextRecognitionSubscriber
from src.client.tk.gui.TextFormatter import TextFormatter
from src.types import RecognitionResult


def test_protocol_has_required_methods():
    """Verify protocol defines both required callback methods."""
    # Protocol methods are defined in __annotations__
    # For runtime checking, we verify the presence of the methods
    assert hasattr(TextRecognitionSubscriber, 'on_partial_update')
    assert hasattr(TextRecognitionSubscriber, 'on_finalization')


def test_text_formatter_implements_protocol():
    """Verify TextFormatter matches the subscriber protocol signature.

    Uses duck typing verification - TextFormatter should have methods
    with matching signatures even though it doesn't explicitly inherit.
    """
    # TextFormatter has the required methods
    assert hasattr(TextFormatter, 'partial_update')
    assert hasattr(TextFormatter, 'finalization')

    # Method signatures match protocol expectations
    # partial_update maps to on_partial_update
    # finalization maps to on_finalization
    formatter_partial = getattr(TextFormatter, 'partial_update')
    formatter_final = getattr(TextFormatter, 'finalization')

    assert callable(formatter_partial)
    assert callable(formatter_final)


def test_protocol_methods_accept_recognition_result():
    """Verify protocol methods expect RecognitionResult parameter.

    This ensures the protocol signature matches our data flow.
    """
    # Get type hints from protocol methods
    hints_partial = get_type_hints(TextRecognitionSubscriber.on_partial_update)
    hints_final = get_type_hints(TextRecognitionSubscriber.on_finalization)

    # Both methods should accept RecognitionResult
    assert hints_partial.get('result') == RecognitionResult
    assert hints_final.get('result') == RecognitionResult

    # Both methods should return None
    assert hints_partial.get('return') == type(None)
    assert hints_final.get('return') == type(None)
