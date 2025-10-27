"""Convert Parakeet ONNX models from FP32 to FP16.

This script converts the encoder and decoder_joint models from 32-bit floating
point to 16-bit floating point format, reducing model size by ~50% and potentially
improving inference speed on GPU hardware.

Usage:
    python convert_to_fp16.py [--input-dir DIR] [--output-suffix SUFFIX]

Options:
    --input-dir: Directory containing the models (default: ./models/parakeet)
    --output-suffix: Suffix for converted files (default: .fp16.onnx)

The script converts:
    - encoder-model.onnx → encoder-model.fp16.onnx
    - decoder_joint-model.onnx → decoder_joint-model.fp16.onnx

Note: The encoder-model.onnx.data file will be automatically handled during
conversion and saved as encoder-model.fp16.onnx.data
"""

import argparse
import logging
from pathlib import Path
import sys

try:
    import onnx
    from onnxconverter_common import float16
except ImportError as e:
    print("Error: Required packages not installed.")
    print("Please run: pip install onnx onnxconverter-common")
    sys.exit(1)


def convert_model_to_fp16(
    model_path: Path,
    output_path: Path,
    keep_io_types: bool = True
) -> None:
    """Convert a single ONNX model from FP32 to FP16.

    Args:
        model_path: Path to input FP32 model
        output_path: Path to save FP16 model
        keep_io_types: If True, keep input/output tensors as FP32 (default: True)

    The keep_io_types=True ensures compatibility with existing preprocessing
    that expects float32 inputs, while converting internal weights to FP16.
    """
    logging.info(f"Loading model from {model_path}")

    # Load the FP32 model (with external data if present)
    model = onnx.load(str(model_path), load_external_data=True)

    # Get model size before conversion (including external data)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    data_file = Path(str(model_path) + ".data")
    if data_file.exists():
        data_size_mb = data_file.stat().st_size / (1024 * 1024)
        model_size_mb += data_size_mb
        logging.info(f"Original model size: {model_size_mb:.2f} MB (model: {model_path.stat().st_size / (1024 * 1024):.2f} MB + data: {data_size_mb:.2f} MB)")
    else:
        logging.info(f"Original model size: {model_size_mb:.2f} MB")

    # Convert to FP16
    logging.info("Converting to FP16...")
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        disable_shape_infer=True  # Required to preserve weights in models with external data
    )

    # Save the converted FP16 model
    logging.info(f"Saving FP16 model to {output_path}")
    onnx.save(model_fp16, str(output_path))

    # Get model size after conversion (including external data)
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    output_data_file = Path(str(output_path) + ".data")
    if output_data_file.exists():
        output_data_size_mb = output_data_file.stat().st_size / (1024 * 1024)
        output_size_mb += output_data_size_mb
        logging.info(f"Converted model size: {output_size_mb:.2f} MB (model: {output_path.stat().st_size / (1024 * 1024):.2f} MB + data: {output_data_size_mb:.2f} MB)")
    else:
        logging.info(f"Converted model size: {output_size_mb:.2f} MB")
    logging.info(f"Size reduction: {model_size_mb - output_size_mb:.2f} MB ({(1 - output_size_mb/model_size_mb)*100:.1f}%)")


def convert_parakeet_models(
    input_dir: Path,
    output_suffix: str = ".fp16.onnx"
) -> None:
    """Convert all Parakeet models to FP16.

    Args:
        input_dir: Directory containing Parakeet models
        output_suffix: Suffix for output files (default: .fp16.onnx)

    Converts:
        - encoder-model.onnx
        - decoder_joint-model.onnx

    Note: External data files (.onnx.data) are handled automatically by onnx.save()
    """
    models_to_convert = [
        'encoder-model.onnx',
        'decoder_joint-model.onnx'
    ]

    for model_name in models_to_convert:
        input_path = input_dir / model_name

        # Check if model exists
        if not input_path.exists():
            logging.warning(f"Model not found: {input_path}")
            continue

        # Generate output filename
        base_name = model_name.replace('.onnx', '')
        output_name = base_name + output_suffix
        output_path = input_dir / output_name

        try:
            convert_model_to_fp16(input_path, output_path)
            logging.info(f"✓ Successfully converted {model_name}")
        except Exception as e:
            logging.error(f"✗ Failed to convert {model_name}: {e}")
            raise


def main():
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description='Convert Parakeet ONNX models from FP32 to FP16',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('./models/parakeet'),
        help='Directory containing the models (default: ./models/parakeet)'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='.fp16.onnx',
        help='Suffix for converted files (default: .fp16.onnx)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )

    # Validate input directory
    if not args.input_dir.exists():
        logging.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Perform conversion
    try:
        logging.info(f"Converting models in {args.input_dir}")
        convert_parakeet_models(args.input_dir, args.output_suffix)
        logging.info("Conversion completed successfully!")
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
