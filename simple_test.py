# stt_local.py
import sounddevice as sd
import numpy as np
import onnx_asr

# Load from local path
print("Loading model from local folder...")
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", "./models/parakeet")

print("Say something (3 seconds)...")
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1)
sd.wait()

print("Processing...")
text = model.recognize(audio.squeeze())
print(f"You said: {text}")