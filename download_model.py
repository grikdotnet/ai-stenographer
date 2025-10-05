# download_model.py
"""Download required models for STT pipeline.

Downloads both Parakeet STT model and Silero VAD model to the models/ directory.
"""

from huggingface_hub import snapshot_download
import urllib.request
from pathlib import Path
import os


def download_parakeet():
    """Download Parakeet STT model from HuggingFace."""
    print("\n=== Downloading Parakeet STT model ===")
    local_model_path = "./models/parakeet"

    snapshot_download(
        repo_id="istupakov/parakeet-tdt-0.6b-v3-onnx",
        local_dir=local_model_path,
        local_dir_use_symlinks=False  # Important for Windows
    )

    print(f"Parakeet model downloaded to: {local_model_path}")
    print("Files:")
    for file in os.listdir(local_model_path):
        print(f"  - {file}")


def download_silero_vad():
    """Download Silero VAD ONNX model from HuggingFace."""
    print("\n=== Downloading Silero VAD model ===")

    from huggingface_hub import hf_hub_download

    models_dir = Path("./models/silero_vad")
    model_path = models_dir / "silero_vad.onnx"

    # Create directory
    models_dir.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        print(f"Model already exists at {model_path}")
        print(f"Size: {model_path.stat().st_size / 1024:.1f} KB")
        return

    print(f"Downloading from HuggingFace: onnx-community/silero-vad")
    print(f"Saving to: {model_path}")

    try:
        # Download from HuggingFace
        downloaded_path = hf_hub_download(
            repo_id="onnx-community/silero-vad",
            filename="onnx/model_q4f16.onnx",
            cache_dir=models_dir
        )

        # Copy to expected location
        import shutil
        shutil.copy(downloaded_path, model_path)

        print(f"Download complete!")
        print(f"Size: {model_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"Download failed: {e}")
        if model_path.exists():
            model_path.unlink()  # Clean up partial download
        raise


if __name__ == "__main__":
    print("Downloading models for STT pipeline...")

    download_parakeet()
    download_silero_vad()

    print("\n=== All models downloaded successfully! ===")