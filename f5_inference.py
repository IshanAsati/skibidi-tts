import torch
from f5_tts.api import F5TTS
import argparse
import os


def generate_speech(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    output_path: str = "output_f5.wav",
    model: str = "F5TTS_v1_Base",
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading F5-TTS model on {device}...")
    f5tts = F5TTS(model=model, device=device)

    print(f"Generating speech...")
    wav, sr, spec = f5tts.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text,
        file_wave=output_path,
    )

    print(f"Saved to: {output_path}")
    return wav, sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F5-TTS Zero-shot Voice Cloning")
    parser.add_argument(
        "--ref-audio", "-r", required=True, help="Reference audio file (.wav)"
    )
    parser.add_argument(
        "--ref-text", "-t", required=True, help="Text spoken in reference audio"
    )
    parser.add_argument("--gen-text", "-g", required=True, help="Text to generate")
    parser.add_argument(
        "--output", "-o", default="output_f5.wav", help="Output audio file"
    )
    parser.add_argument("--model", "-m", default="F5TTS_v1_Base", help="Model name")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    args = parser.parse_args()

    device = "cpu" if args.cpu else None
    generate_speech(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        gen_text=args.gen_text,
        output_path=args.output,
        model=args.model,
        device=device,
    )
