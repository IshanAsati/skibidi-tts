import os
import torch
from TTS.api import TTS

print("=== XTTS v2 Inference Test ===")

# Initialize XTTS v2
print("Loading XTTS v2 model...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

print("Model loaded!")

# Find a test audio file
ref_audio = "data/processed/wavs/test_ref.wav"
if not os.path.exists(ref_audio):
    # Use F5-TTS output as reference
    ref_audio = "outputs/test_f5_output.wav"
    if not os.path.exists(ref_audio):
        print("No reference audio available - skipping generation test")
        print("XTTS v2 model loading: SUCCESS!")
        exit(0)

print(f"Using reference audio: {ref_audio}")

# Generate speech
text = "Hello, this is a test of the XTTS v2 voice cloning system. It sounds pretty good, doesn't it?"

print(f"Generating: {text}")

wav = tts.tts(
    text=text,
    speaker_wav=ref_audio,
    language="en",
)

tts.tts_to_file(
    text=text,
    speaker_wav=ref_audio,
    language="en",
    file_path="outputs/test_xtts_output.wav",
)

print(f"\nGenerated audio saved to: outputs/test_xtts_output.wav")
print(f"Duration: {len(wav) / 24000:.2f} seconds")
print("XTTS v2 inference: SUCCESS!")
