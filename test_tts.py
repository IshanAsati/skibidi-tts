import torch

print("=== System Check ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n=== F5-TTS Test ===")
from f5_tts.api import F5TTS

f5tts = F5TTS(device="cuda")
print("F5-TTS loaded successfully!")
print(f"Mel spec: {f5tts.mel_spec_type}, Sample rate: {f5tts.target_sample_rate}")

print("\n=== XTTS v2 Test ===")
from TTS.api import TTS

print("XTTS v2 imports successful!")
print("All systems operational!")
