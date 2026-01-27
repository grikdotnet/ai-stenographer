import ctypes
import logging
import platform
from ctypes import POINTER, Structure, byref, c_int
from ctypes.wintypes import BOOL, DWORD, HWND, LONG, UINT
from typing import Optional, Tuple


# Windows API constants
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000

# Accent states for SetWindowCompositionAttribute
ACCENT_DISABLED = 0
ACCENT_ENABLE_GRADIENT = 1
ACCENT_ENABLE_TRANSPARENTGRADIENT = 2
ACCENT_ENABLE_BLURBEHIND = 3
ACCENT_ENABLE_ACRYLICBLURBEHIND = 4  # Windows 10 1803+

# Window composition attribute
WCA_ACCENT_POLICY = 19

# DWM window attributes (Windows 11)
DWMWA_WINDOW_CORNER_PREFERENCE = 33

# Corner preference values
DWMWCP_DEFAULT = 0
DWMWCP_DONOTROUND = 1
DWMWCP_ROUND = 2
DWMWCP_ROUNDSMALL = 3


class ACCENT_POLICY(Structure):
    """Structure for accent policy used by SetWindowCompositionAttribute."""
    _fields_ = [
        ('AccentState', DWORD),
        ('AccentFlags', DWORD),
        ('GradientColor', DWORD),
        ('AnimationId', DWORD),
    ]


class WINDOWCOMPOSITIONATTRIBDATA(Structure):
    """Structure for window composition attribute data."""
    _fields_ = [
        ('Attribute', DWORD),
        ('Data', POINTER(c_int)),  # Pointer to data (cast from ACCENT_POLICY)
        ('SizeOfData', ctypes.c_ulong),
    ]


class MARGINS(Structure):
    """Structure for DWM margins."""
    _fields_ = [
        ('cxLeftWidth', c_int),
        ('cxRightWidth', c_int),
        ('cyTopHeight', c_int),
        ('cyBottomHeight', c_int),
    ]


def _get_windows_version() -> Tuple[int, int, int]:
    """Get Windows version as tuple (major, minor, build)."""
    version = platform.version()
    try:
        parts = version.split('.')
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except (IndexError, ValueError):
        return (10, 0, 0)


def _rgba_to_abgr(r: int, g: int, b: int, a: int) -> int:
    """ Convert RGBA to ABGR format used by Windows API. """
    return (a << 24) | (b << 16) | (g << 8) | r


def apply_rounded_corners_win11(hwnd: int) -> bool:
    """
    Apply smooth rounded corners using Windows 11 DwmSetWindowAttribute.

    This provides anti-aliased corners, unlike SetWindowRgn.

    Args:
        hwnd: Window handle

    Returns:
        True if successful, False otherwise
    """
    try:
        dwmapi = ctypes.windll.dwmapi

        # Set corner preference to rounded
        preference = ctypes.c_int(DWMWCP_ROUND)
        result = dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_WINDOW_CORNER_PREFERENCE,
            ctypes.byref(preference),
            ctypes.sizeof(preference)
        )

        if result == 0:  # S_OK
            logging.debug("windows_effects: Applied Windows 11 rounded corners")
            return True
        else:
            logging.debug(f"windows_effects: DwmSetWindowAttribute returned {result}")
            return False

    except Exception as e:
        logging.warning(f"windows_effects: apply_rounded_corners_win11 failed: {e}")
        return False


def apply_rounded_corners_legacy(hwnd: int, width: int, height: int, radius: int) -> bool:
    """
    Apply rounded corners using SetWindowRgn (legacy, has aliased edges).

    Args:
        hwnd: Window handle
        width: Window width
        height: Window height
        radius: Corner radius in pixels

    Returns:
        True if successful, False otherwise
    """
    try:
        gdi32 = ctypes.windll.gdi32
        user32 = ctypes.windll.user32

        hrgn = gdi32.CreateRoundRectRgn(
            0, 0,
            width + 1, height + 1,
            radius * 2, radius * 2
        )

        if not hrgn:
            logging.warning("windows_effects: CreateRoundRectRgn failed")
            return False

        result = user32.SetWindowRgn(hwnd, hrgn, True)

        if not result:
            gdi32.DeleteObject(hrgn)
            logging.warning("windows_effects: SetWindowRgn failed")
            return False

        logging.debug(f"windows_effects: Applied legacy rounded corners (radius={radius})")
        return True

    except Exception as e:
        logging.warning(f"windows_effects: apply_rounded_corners_legacy failed: {e}")
        return False


def apply_rounded_corners(hwnd: int, width: int, height: int, radius: int) -> bool:
    """
    Apply rounded corners to a window.

    On Windows 11 (build 22000+), uses DwmSetWindowAttribute for smooth corners.
    On older Windows, falls back to SetWindowRgn (aliased edges).

    Args:
        hwnd: Window handle
        width: Window width
        height: Window height
        radius: Corner radius in pixels

    Returns:
        True if successful, False otherwise
    """
    version = _get_windows_version()

    # Windows 11 (build 22000+) supports native rounded corners
    if version[2] >= 22000:
        if apply_rounded_corners_win11(hwnd):
            return True
        # Fall back to legacy if Win11 API fails
        logging.debug("windows_effects: Win11 corners failed, trying legacy")

    return apply_rounded_corners_legacy(hwnd, width, height, radius)


def apply_acrylic_effect(
    hwnd: int,
    tint_color: Tuple[int, int, int] = (45, 45, 45),
    opacity: int = 200,
    use_acrylic: bool = True
) -> bool:
    """
    Apply blur/acrylic effect to window background.

    The effect blurs what's behind the window. With acrylic mode,
    the tint_color and opacity create a frosted glass appearance.

    Args:
        hwnd: Window handle
        tint_color: RGB tuple for tint color
        opacity: Opacity 0-255
        use_acrylic: If True, use acrylic (frosted glass). If False, use simple blur.
    """
    try:
        user32 = ctypes.windll.user32

        SetWindowCompositionAttribute = user32.SetWindowCompositionAttribute
        SetWindowCompositionAttribute.argtypes = [HWND, POINTER(WINDOWCOMPOSITIONATTRIBDATA)]
        SetWindowCompositionAttribute.restype = BOOL

        version = _get_windows_version()

        # Choose accent state based on Windows version and preference
        if use_acrylic and version[2] >= 17134:
            accent_state = ACCENT_ENABLE_ACRYLICBLURBEHIND
        else:
            accent_state = ACCENT_ENABLE_BLURBEHIND

        r, g, b = tint_color
        gradient_color = _rgba_to_abgr(r, g, b, opacity)

        accent = ACCENT_POLICY()
        accent.AccentState = accent_state
        accent.AccentFlags = 2  # ACCENT_FLAG_DRAW_ALL
        accent.GradientColor = gradient_color
        accent.AnimationId = 0

        data = WINDOWCOMPOSITIONATTRIBDATA()
        data.Attribute = WCA_ACCENT_POLICY
        data.Data = ctypes.cast(ctypes.pointer(accent), POINTER(c_int))
        data.SizeOfData = ctypes.sizeof(accent)

        result = SetWindowCompositionAttribute(hwnd, byref(data))

        if result:
            logging.debug(f"windows_effects: Applied blur effect (state={accent_state})")
            return True
        else:
            logging.warning("windows_effects: SetWindowCompositionAttribute returned False")
            return False

    except Exception as e:
        logging.warning(f"windows_effects: apply_acrylic_effect failed: {e}")
        return False


def apply_drop_shadow(hwnd: int) -> bool:
    """
    Apply drop shadow effect using DwmExtendFrameIntoClientArea.

    Uses small margins to enable the DWM blur effect while keeping
    the window clickable. Zero margins work with acrylic effect.
    """
    try:
        dwmapi = ctypes.windll.dwmapi

        # Use small margins (1px) to enable DWM effects without breaking clicks
        margins = MARGINS()
        margins.cxLeftWidth = 1
        margins.cxRightWidth = 1
        margins.cyTopHeight = 1
        margins.cyBottomHeight = 1

        result = dwmapi.DwmExtendFrameIntoClientArea(hwnd, byref(margins))

        if result == 0:  # S_OK
            logging.debug("windows_effects: Applied DWM frame extension")
            return True
        else:
            logging.warning(f"windows_effects: DwmExtendFrameIntoClientArea returned {result}")
            return False

    except Exception as e:
        logging.warning(f"windows_effects: apply_drop_shadow failed: {e}")
        return False


def set_layered_window(hwnd: int) -> bool:
    """
    Set window as layered for transparency support.

    Args:
        hwnd: Window handle

    Returns:
        True if successful, False otherwise
    """
    try:
        user32 = ctypes.windll.user32

        # Get current extended style
        current_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)

        # Add layered style
        new_style = current_style | WS_EX_LAYERED
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)

        logging.debug("windows_effects: Set layered window style")
        return True

    except Exception as e:
        logging.warning(f"windows_effects: set_layered_window failed: {e}")
        return False


def get_hwnd_from_tkinter(window, use_ancestor: bool = False) -> Optional[int]:
    """
    Get Windows HWND from a tkinter window.

    Args:
        window: tkinter Tk or Toplevel window
        use_ancestor: If True, return ancestor HWND (needed for acrylic effects).
                      If False, return winfo_id (needed for SetWindowRgn).

    Returns:
        Window handle as integer, or None if failed
    """
    try:
        # Ensure window is created and updated
        window.update()

        # Get HWND via winfo_id - this is the direct window handle
        hwnd = window.winfo_id()

        if use_ancestor:
            user32 = ctypes.windll.user32
            ancestor_hwnd = user32.GetAncestor(hwnd, 2)  # GA_ROOT
            if ancestor_hwnd:
                return ancestor_hwnd

        return hwnd

    except Exception as e:
        logging.warning(f"windows_effects: get_hwnd_from_tkinter failed: {e}")
        return None


def apply_all_effects(
    window,
    width: int,
    height: int,
    corner_radius: int = 16,
    tint_color: Tuple[int, int, int] = (45, 45, 45),
    opacity: int = 200
) -> dict:
    """
    Apply modern UI effects to a tkinter window.

    Args:
        window: tkinter Tk or Toplevel window
        width: Window width
        height: Window height
        corner_radius: Corner radius in pixels
        tint_color: RGB tuple for acrylic tint
        opacity: Acrylic opacity 0-255

    Returns:
        Dict with success status for each effect
    """
    results = {
        'hwnd_direct': None,
        'hwnd_ancestor': None,
        'rounded_corners': False,
        'acrylic': False,
        'shadow': False,
    }

    # Get both HWNDs - different effects need different handles
    hwnd_direct = get_hwnd_from_tkinter(window, use_ancestor=False)
    hwnd_ancestor = get_hwnd_from_tkinter(window, use_ancestor=True)

    if not hwnd_direct or not hwnd_ancestor:
        logging.warning("windows_effects: Could not get window handles")
        return results

    results['hwnd_direct'] = hwnd_direct
    results['hwnd_ancestor'] = hwnd_ancestor

    # Apply effects - all use ancestor HWND for tkinter compatibility
    # Rounded corners must be on ancestor for overrideredirect windows
    results['rounded_corners'] = apply_rounded_corners(hwnd_ancestor, width, height, corner_radius)

    # Skip acrylic - tkinter doesn't support per-pixel alpha needed for it to work
    # The opaque background would cover the effect anyway
    results['acrylic'] = False

    # Shadow also needs ancestor
    results['shadow'] = apply_drop_shadow(hwnd_ancestor)

    logging.info(f"windows_effects: Applied effects - {results}")
    return results
