#!/usr/bin/env python
"""CPU profiling for STT pipeline.

This script profiles the STT pipeline to identify CPU-intensive components.
Uses cProfile and provides detailed breakdown of function calls and timings.

Usage:
    python profile_cpu.py [--duration=10] [--output=profile.txt]

Options:
    --duration=N    Run profiling for N seconds (default: 10)
    --output=FILE   Save profiling results to FILE (default: profile.txt)
"""

import cProfile
import pstats
import io
import sys
import time
import argparse
import logging
from src.pipeline import STTPipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def profile_pipeline(duration: int = 10, output_file: str = 'profile.txt') -> None:
    """Profile the STT pipeline for specified duration.

    Args:
        duration: How long to run profiling in seconds
        output_file: Where to save profiling results
    """
    logging.info(f"Starting CPU profiling for {duration} seconds...")
    logging.info("Speak into your microphone to generate activity.")

    profiler = cProfile.Profile()
    pipeline = None

    try:
        # Start profiling
        profiler.enable()

        # Create and start pipeline
        pipeline = STTPipeline(verbose=False)
        pipeline.start()

        # Let it run for specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(0.1)

        # Stop profiling
        profiler.disable()

        # Stop pipeline
        pipeline.stop()

        logging.info(f"Profiling complete. Analyzing results...")

        # Analyze results
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)

        # Sort by cumulative time spent in function
        stats.sort_stats('cumulative')

        # Write detailed report
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CPU PROFILING RESULTS - Top functions by cumulative time\n")
            f.write("=" * 80 + "\n\n")

            # Redirect stats output to file
            stats.stream = f
            stats.print_stats(50)  # Top 50 functions

            f.write("\n" + "=" * 80 + "\n")
            f.write("CPU PROFILING RESULTS - Top functions by total time\n")
            f.write("=" * 80 + "\n\n")

            stats.sort_stats('tottime')
            stats.print_stats(50)

            f.write("\n" + "=" * 80 + "\n")
            f.write("CPU PROFILING RESULTS - Callers of expensive functions\n")
            f.write("=" * 80 + "\n\n")

            # Show callers for expensive operations
            stats.print_callers(20)

        logging.info(f"Results saved to: {output_file}")

        # Print summary to console
        print("\n" + "=" * 80)
        print("TOP 20 FUNCTIONS BY CUMULATIVE TIME:")
        print("=" * 80)
        stats.stream = sys.stdout
        stats.sort_stats('cumulative')
        stats.print_stats(20)

        print("\n" + "=" * 80)
        print("TOP 20 FUNCTIONS BY TOTAL TIME:")
        print("=" * 80)
        stats.sort_stats('tottime')
        stats.print_stats(20)

    except KeyboardInterrupt:
        logging.info("Profiling interrupted by user")
        if pipeline:
            pipeline.stop()
    except Exception as e:
        logging.error(f"Error during profiling: {e}")
        if pipeline:
            pipeline.stop()
        raise


def main():
    """Parse arguments and run profiling."""
    parser = argparse.ArgumentParser(
        description='Profile CPU usage in STT pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='Duration to profile in seconds (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='profile.txt',
        help='Output file for profiling results (default: profile.txt)'
    )

    args = parser.parse_args()

    profile_pipeline(duration=args.duration, output_file=args.output)


if __name__ == '__main__':
    main()
