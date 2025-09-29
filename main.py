# main.py
import sys
from src.pipeline import STTPipeline

if __name__ == "__main__":
    verbose = "-v" in sys.argv

    # Parse windowing parameters (optional)
    window_duration = 2.0  # Default: 2 seconds
    step_duration = 1.0    # Default: 1 second (50% overlap)

    for arg in sys.argv:
        if arg.startswith("--window="):
            window_duration = float(arg.split("=")[1])
        elif arg.startswith("--step="):
            step_duration = float(arg.split("=")[1])

    pipeline = STTPipeline(
        model_path="./models/parakeet",
        verbose=verbose,
        window_duration=window_duration,
        step_duration=step_duration
    )
    pipeline.run()