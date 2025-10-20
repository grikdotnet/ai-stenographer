#!/usr/bin/env python
"""Real-time CPU monitoring for STT pipeline threads.

This script monitors CPU usage per thread in real-time, showing which
components consume the most CPU resources during operation.

Usage:
    python monitor_cpu.py [--interval=0.5]

Options:
    --interval=N    Update interval in seconds (default: 0.5)
"""

import psutil
import threading
import time
import argparse
import logging
import sys
from collections import defaultdict
from src.pipeline import STTPipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class CPUMonitor:
    """Monitor CPU usage per thread in the pipeline."""

    def __init__(self, interval: float = 0.5):
        """Initialize CPU monitor.

        Args:
            interval: Update interval in seconds
        """
        self.interval = interval
        self.running = False
        self.thread = None
        self.process = psutil.Process()

    def start(self) -> None:
        """Start monitoring in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _monitor_loop(self) -> None:
        """Main monitoring loop - runs in background thread."""
        logging.info("CPU monitoring started. Press Ctrl+C to stop.")
        print("\n" + "=" * 100)
        print(f"{'Thread Name':<30} {'CPU %':>10} {'User Time':>12} {'Sys Time':>12} {'Status':<20}")
        print("=" * 100)

        thread_times = {}  # Track previous times for delta calculation

        while self.running:
            try:
                # Get process-wide CPU usage
                process_cpu = self.process.cpu_percent(interval=None)

                # Get per-thread CPU times
                threads = self.process.threads()
                thread_info = []

                for thread in threads:
                    thread_id = thread.id

                    # Calculate CPU time delta
                    prev_user, prev_sys = thread_times.get(thread_id, (0, 0))
                    user_delta = thread.user_time - prev_user
                    sys_delta = thread.system_time - prev_sys
                    total_delta = user_delta + sys_delta

                    # Store current times
                    thread_times[thread_id] = (thread.user_time, thread.system_time)

                    # Calculate CPU percentage (delta / interval * 100)
                    cpu_pct = (total_delta / self.interval) * 100 if self.interval > 0 else 0

                    # Get thread name from Python threading module
                    thread_name = "Unknown"
                    for t in threading.enumerate():
                        if t.ident == thread_id:
                            thread_name = t.name
                            break

                    thread_info.append({
                        'id': thread_id,
                        'name': thread_name,
                        'cpu_pct': cpu_pct,
                        'user_time': thread.user_time,
                        'sys_time': thread.system_time,
                        'user_delta': user_delta,
                        'sys_delta': sys_delta
                    })

                # Sort by CPU percentage (highest first)
                thread_info.sort(key=lambda x: x['cpu_pct'], reverse=True)

                # Clear screen (simple version)
                print("\033[H\033[J", end="")  # ANSI escape codes for clear screen

                # Print header
                print("=" * 100)
                print(f"Process CPU: {process_cpu:6.2f}% | Threads: {len(threads)} | Interval: {self.interval}s")
                print("=" * 100)
                print(f"{'Thread Name':<30} {'CPU %':>10} {'User Δ':>12} {'Sys Δ':>12} {'Total Time':>12}")
                print("=" * 100)

                # Print thread info
                for info in thread_info:
                    if info['cpu_pct'] > 0.1:  # Only show threads with >0.1% CPU
                        total_time = info['user_time'] + info['sys_time']
                        print(f"{info['name']:<30} {info['cpu_pct']:>9.2f}% "
                              f"{info['user_delta']:>11.3f}s {info['sys_delta']:>11.3f}s "
                              f"{total_time:>11.2f}s")

                print("=" * 100)
                print("\nTop CPU consumers:")
                print("-" * 100)

                # Group by component name (extract from thread names)
                component_cpu = defaultdict(float)
                for info in thread_info:
                    name = info['name']
                    # Extract component name (e.g., "Recognizer" from "Thread-Recognizer")
                    if '-' in name:
                        component = name.split('-', 1)[1]
                    else:
                        component = name
                    component_cpu[component] += info['cpu_pct']

                # Sort and display
                sorted_components = sorted(component_cpu.items(), key=lambda x: x[1], reverse=True)
                for component, cpu_pct in sorted_components[:10]:
                    if cpu_pct > 0.1:
                        print(f"  {component:<30} {cpu_pct:>9.2f}%")

                print("\n(Press Ctrl+C to stop)")

                time.sleep(self.interval)

            except Exception as e:
                logging.error(f"Error in monitoring: {e}")
                time.sleep(self.interval)


def main():
    """Parse arguments and run monitoring."""
    parser = argparse.ArgumentParser(
        description='Monitor CPU usage per thread in STT pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=0.5,
        help='Update interval in seconds (default: 0.5)'
    )

    args = parser.parse_args()

    # Create monitor
    monitor = CPUMonitor(interval=args.interval)
    pipeline = None

    try:
        # Start pipeline
        logging.info("Starting STT pipeline...")
        pipeline = STTPipeline(verbose=False)
        pipeline.start()

        # Start monitoring
        monitor.start()

        # Keep running until Ctrl+C
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        logging.info("\nStopping...")
    finally:
        monitor.stop()
        if pipeline:
            pipeline.stop()


if __name__ == '__main__':
    main()
