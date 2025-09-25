# download_model.py
from huggingface_hub import snapshot_download

print("Downloading model locally...")
local_model_path = "./models/parakeet"

snapshot_download(
    repo_id="istupakov/parakeet-tdt-0.6b-v3-onnx",
    local_dir=local_model_path,
    local_dir_use_symlinks=False  # Important for Windows
)

print(f"Model downloaded to: {local_model_path}")
print("Files:")
import os
for file in os.listdir(local_model_path):
    print(f"  - {file}")