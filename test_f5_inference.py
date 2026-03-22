import os
import torch
from f5_tts.api import F5TTS
import numpy as np
import soundfile as sf

print("=== F5-TTS Zero-shot Inference Test ===")

# Initialize F5-TTS
f5tts = F5TTS(device="cuda")

# Find built-in example audio
example_dir = os.path.join(
    os.path.dirname(__file__), "F5-TTS", "src", "f5_tts", "infer", "examples", "basic"
)
ref_audio = os.path.join(example_dir, "basic_ref_en.wav")

if os.path.exists(ref_audio):
    print(f"Using reference audio: {ref_audio}")
else:
    print("Creating test reference audio...")
    ref_audio = "data/processed/wavs/test_ref.wav"
    os.makedirs("data/processed/wavs", exist_ok=True)
    ref_audio = None

if ref_audio and os.path.exists(ref_audio):
    ref_text = "Some call me nature, others call me mother nature."
    gen_text = "I don't really care what you call me. I've been a silent spectator, watching species evolve."

    print(f"Reference text: {ref_text}")
    print(f"Generating text: {gen_text}")

    wav, sr, spec = f5tts.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text,
        file_wave="outputs/test_f5_output.wav",
    )

    print(f"\nGenerated audio saved to: outputs/test_f5_output.wav")
    print(f"Duration: {len(wav) / sr:.2f} seconds")
    print("F5-TTS inference: SUCCESS!")
else:
    print("No reference audio found - skipping generation test")
    print("F5-TTS model loading: SUCCESS!")
