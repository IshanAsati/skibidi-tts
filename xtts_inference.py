import torch
from TTS.api import TTS
import argparse
import os


def generate_speech_xtts(
    text: str,
    speaker_wav: str,
    language: str = "en",
    output_path: str = "output_xtts.wav",
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    fine_tuned_path: str = None,
):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Loading XTTS v2 model on {device}...")

    if fine_tuned_path:
        tts = TTS(model_path=fine_tuned_path, gpu=(device == "cuda"))
    else:
        tts = TTS(model_name=model_name, gpu=(device == "cuda"))

    print(f"Generating speech...")
    wav = tts.tts(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
    )

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=output_path,
    )

    print(f"Saved to: {output_path}")
    return wav


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XTTS v2 Voice Cloning")
    parser.add_argument("--text", "-t", required=True, help="Text to synthesize")
    parser.add_argument(
        "--speaker-wav", "-s", required=True, help="Reference speaker audio (.wav)"
    )
    parser.add_argument(
        "--language", "-l", default="en", help="Language code (default: en)"
    )
    parser.add_argument(
        "--output", "-o", default="output_xtts.wav", help="Output audio file"
    )
    parser.add_argument(
        "--fine-tuned", "-f", default=None, help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Model name (for non-fine-tuned)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    args = parser.parse_args()

    device = "cpu" if args.cpu else None

    generate_speech_xtts(
        text=args.text,
        speaker_wav=args.speaker_wav,
        language=args.language,
        output_path=args.output,
        model_name=args.model,
        fine_tuned_path=args.fine_tuned,
    )
