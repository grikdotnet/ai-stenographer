"""Fix FP16 ONNX models by converting Cast(to=FLOAT) operations to Cast(to=FLOAT16).

This script post-processes FP16-converted ONNX models to fix type mismatches where
Cast operations still target FLOAT (FP32) instead of FLOAT16, causing errors like:
    Type parameter (T) of Optype (Add) bound to different types
    (tensor(float) and tensor(float16))

Usage:
    python fix_fp16_casts.py [--input-dir DIR]

The script fixes:
    - encoder-model.fp16.onnx
    - decoder_joint-model.fp16.onnx
"""

import argparse
import logging
from pathlib import Path
import sys

try:
    import onnx
except ImportError:
    print("Error: onnx package not installed.")
    print("Please run: pip install onnx")
    sys.exit(1)


def fix_fp16_cast_operations(model_path: Path, output_path: Path) -> int:
    """Fix Cast operations in FP16 model to target FLOAT16 instead of FLOAT.

    Also adds Cast operations at FP16 inputs to allow FP32 inputs for compatibility.

    Args:
        model_path: Path to input FP16 model with Cast issues
        output_path: Path to save fixed model

    Returns:
        Number of Cast operations fixed
    """
    logging.info(f"Loading model from {model_path}")
    model = onnx.load(str(model_path))

    fixed_count = 0
    FLOAT = 1
    FLOAT16 = 10

    # Identify graph outputs to preserve their Cast operations
    graph_output_names = {out.name for out in model.graph.output}

    # Fix existing Cast operations (but not those feeding graph outputs)
    for node in model.graph.node:
        if node.op_type == 'Cast':
            # Check if this Cast feeds a graph output
            feeds_output = any(out_name in graph_output_names for out_name in node.output)

            if not feeds_output:
                for attr in node.attribute:
                    if attr.name == 'to' and attr.i == FLOAT:
                        logging.debug(f"Fixing Cast node: {node.name or node.output[0]}")
                        attr.i = FLOAT16
                        fixed_count += 1

    # Convert FP16 inputs to FP32 and add Cast nodes
    inputs_to_fix = []
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == FLOAT16:
            inputs_to_fix.append(inp)

    if inputs_to_fix:
        logging.info(f"Converting {len(inputs_to_fix)} FP16 input(s) to accept FP32")

        for inp in inputs_to_fix:
            original_name = inp.name
            cast_output_name = f"{original_name}_fp16_internal"

            logging.debug(f"  Converting input '{original_name}' to FP32 with Cast to FP16")

            # Change input type to FP32
            inp.type.tensor_type.elem_type = FLOAT

            # Create Cast node: FP32 input -> FP16 for internal use
            cast_node = onnx.helper.make_node(
                'Cast',
                inputs=[original_name],
                outputs=[cast_output_name],
                to=FLOAT16,
                name=f"cast_input_{original_name}"
            )

            # Update all nodes that use this input to use the Cast output
            for node in model.graph.node:
                for i, input_name in enumerate(node.input):
                    if input_name == original_name:
                        node.input[i] = cast_output_name

            # Insert Cast node at the beginning
            model.graph.node.insert(0, cast_node)

            fixed_count += 1

    if fixed_count > 0:
        logging.info(f"Fixed {fixed_count} Cast operation(s)")
        logging.info(f"Saving fixed model to {output_path}")
        onnx.save(model, str(output_path))
    else:
        logging.info("No Cast operations needed fixing")

    return fixed_count


def fix_parakeet_fp16_models(input_dir: Path) -> None:
    """Fix FP16 Cast operations in all Parakeet FP16 models.

    Args:
        input_dir: Directory containing the FP16 models
    """
    models_to_fix = [
        'encoder-model.fp16.onnx',
        'decoder_joint-model.fp16.onnx'
    ]

    total_fixed = 0

    for model_name in models_to_fix:
        model_path = input_dir / model_name

        if not model_path.exists():
            logging.warning(f"Model not found: {model_path}")
            continue

        # Fix in place
        try:
            fixed = fix_fp16_cast_operations(model_path, model_path)
            total_fixed += fixed
            if fixed > 0:
                logging.info(f"✓ Successfully fixed {model_name}")
            else:
                logging.info(f"○ No fixes needed for {model_name}")
        except Exception as e:
            logging.error(f"✗ Failed to fix {model_name}: {e}")
            raise

    if total_fixed > 0:
        logging.info(f"\nTotal Cast operations fixed: {total_fixed}")
    else:
        logging.info("\nAll models are already correct")


def main():
    """Main entry point for the fix script."""
    parser = argparse.ArgumentParser(
        description='Fix Cast operations in FP16 ONNX models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('./models/parakeet'),
        help='Directory containing the FP16 models (default: ./models/parakeet)'
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

    # Perform fixes
    try:
        logging.info(f"Fixing FP16 models in {args.input_dir}")
        fix_parakeet_fp16_models(args.input_dir)
        logging.info("Fix completed successfully!")
    except Exception as e:
        logging.error(f"Fix failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
