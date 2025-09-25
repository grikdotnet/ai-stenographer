# main.py
from src.pipeline import STTPipeline

if __name__ == "__main__":
    pipeline = STTPipeline(model_path="./models/parakeet")
    pipeline.run()