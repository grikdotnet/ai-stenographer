#!/usr/bin/env python3
"""
MSIX Store Assets Generator

Generates required icon suite for Microsoft Store MSIX package from source image.
Creates PNG assets with proper dimensions and aspect ratios.

Usage:
    python generate_store_assets.py [--source SOURCE_IMAGE] [--output OUTPUT_DIR]

Assets generated:
    - Square44x44Logo.png (44x44) - Taskbar icon
    - Square150x150Logo.png (150x150) - Medium tile
    - Wide310x150Logo.png (310x150) - Wide tile
    - StoreLogo.png (50x50) - Store listing
    - SplashScreen.png (620x300) - App launch screen

Source: c:\\workspaces\\stt\\stenographer.jpg (700x700)
Output: c:\\workspaces\\stt\\msix\\Assets\\

Author: Grigori Kochanov
Date: January 13, 2026
"""

import sys
from pathlib import Path
from typing import Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Pillow library not installed")
    print("Install with: pip install Pillow")
    sys.exit(1)


# MSIX asset specifications
ASSETS = {
    "Square44x44Logo.png": (44, 44),
    "Square150x150Logo.png": (150, 150),
    "Wide310x150Logo.png": (310, 150),
    "StoreLogo.png": (50, 50),
    "SplashScreen.png": (620, 300),
}

# Default paths
DEFAULT_SOURCE = Path(__file__).parent.parent / "stenographer.jpg"
DEFAULT_OUTPUT = Path(__file__).parent / "Assets"

# Crop region for logo (right side of image with "AI Stenographer" text)
# stenographer.jpg is actually 700x700, logo is roughly at x=350-700, y=50-350
LOGO_CROP_REGION = (350, 50, 700, 350)  # (left, top, right, bottom)


def crop_and_resize(
    source_img: Image.Image,
    target_width: int,
    target_height: int,
    use_logo_region: bool = True
) -> Image.Image:
    """
    Crop and resize image to target dimensions while maintaining quality.

    Logic:
    1. If use_logo_region is True, crop to logo area first (right side with text)
    2. Calculate aspect ratios
    3. Center crop to match target aspect ratio
    4. Resize with high-quality resampling (Lanczos)

    Args:
        source_img: Source PIL Image
        target_width: Target width in pixels
        target_height: Target height in pixels
        use_logo_region: Whether to use logo crop region (default: True)

    Returns:
        Resized PIL Image with target dimensions
    """
    # Start with logo region if requested
    if use_logo_region:
        img = source_img.crop(LOGO_CROP_REGION)
    else:
        img = source_img.copy()

    img_width, img_height = img.size
    target_ratio = target_width / target_height
    img_ratio = img_width / img_height

    # Calculate crop box to match target aspect ratio (center crop)
    if img_ratio > target_ratio:
        # Image is wider than target, crop sides
        new_width = int(img_height * target_ratio)
        left = (img_width - new_width) // 2
        crop_box = (left, 0, left + new_width, img_height)
    else:
        # Image is taller than target, crop top/bottom
        new_height = int(img_width / target_ratio)
        top = (img_height - new_height) // 2
        crop_box = (0, top, img_width, top + new_height)

    cropped = img.crop(crop_box)

    # Resize with high-quality Lanczos resampling
    resized = cropped.resize(
        (target_width, target_height),
        Image.Resampling.LANCZOS
    )

    return resized


def generate_assets(
    source_path: Path,
    output_dir: Path,
    verbose: bool = True
) -> dict[str, Path]:
    """
    Generate all required MSIX assets from source image.

    Args:
        source_path: Path to source image (stenographer.jpg)
        output_dir: Directory to save generated assets
        verbose: Print progress messages

    Returns:
        Dictionary mapping asset filename to output path

    Raises:
        FileNotFoundError: If source image doesn't exist
        IOError: If unable to create output directory or save images
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source image not found: {source_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading source image: {source_path}")

    source_img = Image.open(source_path)

    if verbose:
        print(f"Source image size: {source_img.size[0]}x{source_img.size[1]}")
        print(f"Output directory: {output_dir}")
        print(f"\nGenerating {len(ASSETS)} assets...")

    generated_files = {}

    for filename, (width, height) in ASSETS.items():
        output_path = output_dir / filename

        if verbose:
            print(f"  Creating {filename} ({width}x{height})...", end=" ")

        # Generate asset
        asset_img = crop_and_resize(source_img, width, height)

        # Save as PNG with optimization
        asset_img.save(output_path, "PNG", optimize=True)

        generated_files[filename] = output_path

        if verbose:
            file_size = output_path.stat().st_size
            print(f"✓ ({file_size:,} bytes)")

    if verbose:
        total_size = sum(p.stat().st_size for p in generated_files.values())
        print(f"\nGeneration complete!")
        print(f"Total size: {total_size:,} bytes ({total_size / 1024:.1f} KB)")
        print(f"\nAssets saved to: {output_dir}")

    return generated_files


def verify_assets(output_dir: Path) -> bool:
    """
    Verify all required assets exist with correct dimensions.

    Args:
        output_dir: Directory containing generated assets

    Returns:
        True if all assets valid, False otherwise
    """
    print("\nVerifying generated assets...")

    all_valid = True

    for filename, (expected_width, expected_height) in ASSETS.items():
        asset_path = output_dir / filename

        if not asset_path.exists():
            print(f"  ✗ {filename}: Missing")
            all_valid = False
            continue

        try:
            with Image.open(asset_path) as img:
                actual_width, actual_height = img.size

                if actual_width != expected_width or actual_height != expected_height:
                    print(f"  ✗ {filename}: Wrong size ({actual_width}x{actual_height}, "
                          f"expected {expected_width}x{expected_height})")
                    all_valid = False
                else:
                    print(f"  ✓ {filename}: Valid ({actual_width}x{actual_height})")

        except Exception as e:
            print(f"  ✗ {filename}: Error reading file - {e}")
            all_valid = False

    if all_valid:
        print("\nAll assets verified successfully! ✓")
    else:
        print("\nSome assets failed verification ✗")

    return all_valid


def main():
    """Main entry point for asset generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate MSIX store assets from source image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_store_assets.py
  python generate_store_assets.py --source custom.jpg
  python generate_store_assets.py --output ./custom_assets/
  python generate_store_assets.py --source custom.jpg --output ./assets/ --verify
        """
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source image path (default: {DEFAULT_SOURCE})"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify generated assets after creation"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    try:
        # Generate assets
        generated = generate_assets(
            args.source,
            args.output,
            verbose=not args.quiet
        )

        # Verify if requested
        if args.verify:
            if not verify_assets(args.output):
                sys.exit(1)

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error: Failed to generate assets - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error - {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
