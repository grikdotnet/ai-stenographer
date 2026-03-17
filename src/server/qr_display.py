"""QR code renderer for terminal output.

Renders a URL as a QR code using Unicode half-block characters, printing
two rows of QR modules per terminal line to keep the output compact.

Block character mapping (top_bit, bottom_bit):
  (0, 0) → space   (1, 1) → █   (1, 0) → ▀   (0, 1) → ▄
"""

import sys
from typing import TextIO

import qrcode
import qrcode.constants


_HALF_BLOCKS = {(False, False): " ", (True, True): "█", (True, False): "▀", (False, True): "▄"}

# QR quiet zone (blank border) is 4 modules; we add 1 extra cell on each side.
_BORDER = 1


def print_qr_code(url: str, out: TextIO | None = None) -> None:
    """Render a QR code for url to out using Unicode half-block characters.

    Args:
        url: The URL to encode.
        out: Output stream; defaults to sys.stdout.
    """
    matrix = _build_matrix(url)
    _render(matrix, out if out is not None else sys.stdout)


def _build_matrix(url: str) -> list[list[bool]]:
    """Generate the boolean QR module matrix for url.

    Args:
        url: The URL to encode.

    Returns:
        2-D list of booleans; True = dark module.
    """
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=_BORDER,
    )
    qr.add_data(url)
    qr.make(fit=True)
    return qr.get_matrix()


def _render(matrix: list[list[bool]], out: TextIO) -> None:
    """Write the matrix to out as half-block Unicode lines.

    Args:
        matrix: 2-D boolean QR module matrix.
        out: Output stream.
    """
    height = len(matrix)
    width = len(matrix[0]) if matrix else 0

    for row in range(0, height, 2):
        line_chars = []
        for col in range(width):
            top = matrix[row][col]
            bottom = matrix[row + 1][col] if row + 1 < height else False
            line_chars.append(_HALF_BLOCKS[(top, bottom)])
        out.write("".join(line_chars) + "\n")
