from pydub import AudioSegment
import os
import argparse


def preprocess_audio(input_path, output_path, target_sr=22050, normalize=True):
    audio = AudioSegment.from_file(input_path)

    if audio.channels != 1:
        audio = audio.set_channels(1)

    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)

    if normalize:
        change_in_dBFS = -20.0 - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)

    audio.export(output_path, format="wav", bitrate="16bit")
    print(f"Processed: {output_path}")


def preprocess_directory(input_dir, output_dir, target_sr=22050):
    os.makedirs(output_dir, exist_ok=True)

    extensions = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(extensions):
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.wav")

            try:
                preprocess_audio(input_path, output_path, target_sr)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio files for TTS")
    parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    parser.add_argument(
        "--output", "-o", required=True, help="Output file or directory"
    )
    parser.add_argument(
        "--sr",
        "-s",
        type=int,
        default=22050,
        help="Target sample rate (default: 22050)",
    )

    args = parser.parse_args()

    if os.path.isdir(args.input):
        preprocess_directory(args.input, args.output, args.sr)
    else:
        preprocess_audio(args.input, args.output, args.sr)
