# TTS Setup: F5-TTS + XTTS v2

A complete voice cloning and text-to-speech system using F5-TTS for zero-shot inference and XTTS v2 for fine-tuned voice synthesis.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 6GB | RTX 4050+ 6GB+ |
| RAM | 8GB | 16GB+ |
| Storage | 10GB | 20GB+ |
| OS | Windows 10+ | Windows 11 |

## Overview

This setup provides two TTS systems:

| Model | Use Case | Training Required | Speed |
|-------|----------|-------------------|-------|
| **F5-TTS** | Zero-shot voice cloning | No | Fast (~5 sec inference) |
| **XTTS v2** | Fine-tuned voice synthesis | Yes (Colab) | Slower but higher quality |

## Quick Start

```bash
# Activate environment
conda activate tts

# F5-TTS zero-shot
python f5_inference.py -r ref.wav -t "Text in audio" -g "Text to generate"

# XTTS v2 inference
python xtts_inference.py -t "Text to synthesize" -s ref.wav -l en
```

## Environment Setup

### Requirements
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- NVIDIA GPU with CUDA 12.4+
- FFmpeg

### Installation

```bash
# Create environment
conda create -n tts python=3.11 -y
conda activate tts
conda install ffmpeg -y

# Install PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Clone and install F5-TTS
git clone https://github.com/SWivid/F5-TTS.git
pip install -e ./F5-TTS

# Install coqui-tts (XTTS v2)
pip install coqui-tts

# Install compatible transformers version
pip install "transformers>=4.57,<5.0"
```

## Project Structure

```
skibidi/
├── PLAN.md                    # Setup plan and notes
├── README.md                  # This file
├── f5_inference.py            # F5-TTS zero-shot inference
├── xtts_inference.py          # XTTS v2 inference
├── preprocess_audio.py       # Audio preprocessing
├── test_tts.py               # System test script
├── F5-TTS/                   # F5-TTS source code
├── data/
│   ├── raw/                  # Original audio files
│   └── processed/            # Preprocessed audio
│       └── wavs/             # TTS-ready audio (22050Hz, mono)
└── outputs/                   # Generated audio files
```

## Usage

### 1. Audio Preprocessing

Before using audio for voice cloning, preprocess it to the correct format.

**Single file:**
```bash
python preprocess_audio.py -i audio.mp3 -o processed.wav
```

**Directory batch:**
```bash
python preprocess_audio.py -i data/raw/ -o data/processed/wavs/
```

**Options:**
- `--sr, -s`: Target sample rate (default: 22050)
- Automatically normalizes to mono, 16-bit WAV

### 2. F5-TTS (Zero-shot Voice Cloning)

F5-TTS requires no training - just provide a reference audio sample.

**Basic usage:**
```bash
python f5_inference.py \
    -r reference.wav \
    -t "Text that is spoken in the reference audio" \
    -g "Text you want to generate in that voice"
```

**Output file:**
```bash
python f5_inference.py \
    -r reference.wav \
    -t "Hello, how are you today?" \
    -g "This is a test of the voice cloning system" \
    -o my_output.wav
```

**Programmatic usage:**
```python
from f5_inference import generate_speech

wav, sr = generate_speech(
    ref_audio="reference.wav",
    ref_text="Hello, how are you today?",
    gen_text="This is a test of the voice cloning system",
    output_path="output.wav"
)
```

### 3. XTTS v2 (Fine-tuned Voice)

XTTS v2 requires fine-tuning for best results. Use Google Colab for training.

#### Fine-tuning on Google Colab

1. **Open the Colab notebook:**
   ```
   https://colab.research.google.com/github/idiap/coqui-ai-TTS/blob/dev/TTS/demos/xtts_ft_demo/XTTS_finetune_colab.ipynb
   ```

2. **Prepare your audio data:**
   - Format: WAV, 16-bit, 22050Hz, mono
   - Length: 10-30 seconds per sample
   - Total: 30-60 minutes of audio recommended
   - Place in Google Drive: `/drive/MyDrive/xtts_finetune/wavs/`

3. **Run the notebook:**
   - Connect to T4 GPU (Runtime → Change runtime type → GPU)
   - Mount Google Drive
   - Upload preprocessed audio
   - Run training cells
   - Download fine-tuned model

**Recommended settings for T4 (16GB VRAM):**
```yaml
batch_size: 4
grad_accum_steps: 8
learning_rate: 5e-6
epochs: 10-20
```

#### XTTS v2 Inference

**Using pre-trained model:**
```bash
python xtts_inference.py \
    -t "Hello, this is a test of the XTTS voice cloning system." \
    -s reference.wav \
    -l en
```

**Using fine-tuned model:**
```bash
python xtts_inference.py \
    -t "Hello, this is my cloned voice." \
    -s reference.wav \
    -l en \
    -f path/to/fine_tuned_model
```

**Programmatic usage:**
```python
from xtts_inference import generate_speech_xtts

wav = generate_speech_xtts(
    text="Hello, this is a test.",
    speaker_wav="reference.wav",
    language="en",
    output_path="output.wav"
)
```

## Supported Languages

### F5-TTS
- English (primary)
- Chinese (primary)
- French, Italian, Hindi, Japanese, Russian, Spanish, Finnish, German, Polish, Portuguese, Korean

### XTTS v2
- English, Spanish, French, German, Polish, Portuguese, Italian, Japanese, Chinese, Korean, Hungarian, Russian, Arabic, Dutch, Turkish, Czech, Hindi, Vietnamese

## Troubleshooting

### Out of Memory (OOM) on RTX 4050

**F5-TTS:**
- Should work fine (~2GB VRAM for inference)

**XTTS v2:**
- Use smaller batch sizes
- Enable gradient accumulation
- Fine-tune on Colab instead

### CUDA Not Found

```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, reinstall PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Import Errors

**coqui-tts:**
```bash
pip install "transformers>=4.57,<5.0"
```

**F5-TTS:**
```bash
pip install -e ./F5-TTS
```

### Windows Long Path Error

Run as Administrator and enable long paths:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

## Performance

### Inference Speed (RTX 4050)

| Model | Audio Length | Time |
|-------|--------------|------|
| F5-TTS | ~10 sec | ~5 sec |
| XTTS v2 | ~10 sec | ~15-30 sec |

### Fine-tuning Time (Google Colab T4)

| Dataset | Epochs | Time |
|---------|--------|------|
| 30 min audio | 10 | ~30-45 min |
| 60 min audio | 10 | ~60-90 min |

## API Reference

### F5-TTS

```python
from f5_tts.api import F5TTS

f5tts = F5TTS(device="cuda")

wav, sr, spec = f5tts.infer(
    ref_file="reference.wav",
    ref_text="Text in reference audio",
    gen_text="Text to generate",
    file_wave="output.wav",
    nfe_step=32,        # Inference steps (higher = better quality)
    speed=1.0,          # Speech speed
    seed=None           # Random seed for reproducibility
)
```

### XTTS v2

```python
from TTS.api import TTS

# Pre-trained model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Fine-tuned model
tts = TTS(model_path="/path/to/fine_tuned_model", gpu=True)

wav = tts.tts(
    text="Text to synthesize",
    speaker_wav="reference.wav",
    language="en"
)
```

## License

- **F5-TTS**: MIT License
- **XTTS v2 / coqui-tts**: MPL 2.0

## Resources

- [F5-TTS GitHub](https://github.com/SWivid/F5-TTS)
- [F5-TTS Paper](https://arxiv.org/abs/2410.06885)
- [coqui-tts GitHub](https://github.com/idiap/coqui-ai-TTS)
- [XTTS v2 Colab Notebook](https://colab.research.google.com/github/idiap/coqui-ai-TTS/blob/dev/TTS/demos/xtts_ft_demo/XTTS_finetune_colab.ipynb)
- [AllTalk TTS (Windows-friendly GUI)](https://github.com/erew123/alltalk_tts)

## Acknowledgments

- F5-TTS by [SWivid](https://github.com/SWivid/F5-TTS)
- XTTS v2 by [Coqui](https://github.com/coqui-ai/TTS)
- Community fork by [idiap](https://github.com/idiap/coqui-ai-TTS)
