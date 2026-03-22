# TTS Setup: F5-TTS + XTTS v2 for RTX 4050

## Overview
Dual TTS system with hybrid local/cloud training:
- **F5-TTS**: Zero-shot voice cloning (run locally)
- **XTTS v2**: Fine-tuning (train on Google Colab, infer locally)

## Hardware
| Component | Spec |
|-----------|------|
| GPU | RTX 4050 (6GB VRAM) |
| RAM | 16GB+ recommended |
| Storage | 20GB+ free space |

## Architecture
```
┌─────────────────────────────────────────────┐
│  Local (RTX 4050)        │  Google Colab    │
├──────────────────────────┼──────────────────┤
│  F5-TTS zero-shot        │  XTTS v2         │
│  XTTS v2 inference       │  Fine-tuning     │
│  Audio preprocessing     │  (T4 GPU, 16GB)  │
└──────────────────────────┴──────────────────┘
```

---

## Phase 1: Local Environment Setup

### 1.1 Create conda environment
```bash
conda create -n tts python=3.11 -y
conda activate tts
conda install ffmpeg -y
```

### 1.2 Install PyTorch (CUDA 12.4)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 1.3 Verify CUDA
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

---

## Phase 2: F5-TTS (Zero-shot Inference)

### 2.1 Install
```bash
pip install f5-tts
```

### 2.2 F5-TTS Dependencies
| Package | Constraint |
|---------|------------|
| Python | >=3.10 |
| PyTorch | >=2.0.0 |
| numpy | <=1.26.4 |
| pydantic | <=2.10.6 |
| accelerate | >=0.33.0 |
| bitsandbytes | >0.37.0 |
| ema_pytorch | >=0.5.2 |
| gradio | >=6.0.0 |
| hydra-core | >=1.3.0 |

### 2.3 Inference Script (`f5_inference.py`)
```python
from f5_tts import F5TTS

tts = F5TTS(device="cuda")

# Zero-shot voice cloning
audio = tts.generate(
    ref_audio="path/to/reference.wav",
    ref_text="Text spoken in reference audio",
    gen_text="Text to generate in cloned voice",
)
```

### 2.4 Launch Gradio UI (optional)
```bash
f5-tts_infer-gradio --port 7860 --host 0.0.0.0
```

### 2.5 VRAM Usage
- Inference: ~2GB VRAM (confirmed)
- RTX 4050 (6GB) is well above minimum

---

## Phase 3: XTTS v2 Fine-tuning (Google Colab)

### 3.1 Install coqui-tts (NOT TTS)
```bash
# CRITICAL: Use coqui-tts fork (original repo is unmaintained)
pip install coqui-tts
```

### 3.2 coqui-tts Dependencies
| Package | Constraint |
|---------|------------|
| Python | >=3.9, <=3.12 |
| PyTorch | 2.5+ (works with 2.5, 2.6, 2.7) |
| transformers | <4.52 (Windows audio fix) |
| CUDA | 11.8, 12.1, 12.4, 12.6, 12.8 |

### 3.3 If upgrading from TTS
```bash
pip uninstall TTS trainer coqpit
pip cache purge
pip install coqui-tts
```

### 3.4 Upload audio to Google Drive
```
/drive/MyDrive/
└── xtts_finetune/
    └── wavs/
        ├── sample1.wav
        ├── sample2.wav
        └── ...
```

### 3.5 Audio Requirements
- Format: WAV, 16-bit, 22050Hz, mono
- Length: 10-30 seconds per sample
- Quality: Clear speech, minimal background noise
- Quantity: 30-60 minutes total

### 3.6 Official Colab Notebook (idiap fork)
URL: https://colab.research.google.com/github/idiap/coqui-ai-TTS/blob/dev/TTS/demos/xtts_ft_demo/XTTS_finetune_colab.ipynb

Steps:
1. Connect to T4 GPU (Runtime → Change runtime type → GPU)
2. Mount Google Drive
3. Upload preprocessed audio to `wavs/` folder
4. Run training cells
5. Download fine-tuned model (~2GB)

### 3.7 Fine-tuning Config (for T4 16GB VRAM)
```yaml
batch_size: 4          # T4 can handle 4
grad_accum_steps: 8    # Adjust for effective batch size
learning_rate: 5e-6
epochs: 10-20
```

### 3.8 Local Fine-tuning (RTX 4050 - SLOW)
```yaml
batch_size: 1          # 6GB VRAM limit
grad_accum_steps: 32-64
learning_rate: 5e-6
epochs: 10-20
```

---

## Phase 4: Audio Preprocessing

### 4.1 Preprocess Script (`preprocess_audio.py`)
```python
from pydub import AudioSegment
import os

def preprocess(input_path, output_path, target_sr=22050):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(target_sr).set_channels(1)
    audio.export(output_path, format="wav", bitrate="16")
```

### 4.2 Run preprocessing
```bash
python preprocess_audio.py
```

---

## Phase 5: Local Inference (After Fine-tuning)

### 5.1 Load Fine-tuned Model (`xtts_inference.py`)
```python
from TTS.api import TTS

tts = TTS(model_path="/path/to/xtts_finetuned", gpu=True)

# Generate speech
wav = tts.tts(
    text="Your text here",
    speaker_wav="path/to/reference.wav",
    language="en"
)
```

---

## File Structure
```
skibidi/
├── PLAN.md
├── f5_inference.py          # F5-TTS zero-shot
├── xtts_inference.py        # XTTS v2 inference
├── preprocess_audio.py      # Audio preprocessing
└── data/
    └── raw/                 # Original audio files
    └── processed/           # Preprocessed audio
        └── wavs/
```

---

## Training Time Estimates
| Dataset | GPU | Epochs | Time |
|---------|-----|--------|------|
| 30 min | T4 | 10 | ~30-45 min |
| 60 min | T4 | 10 | ~60-90 min |
| 30 min | RTX 4050 | 10 | ~4-6 hours |

---

## Key Notes

1. **Package name**: Use `coqui-tts` NOT `TTS` (original repo is dead)
2. **Windows audio bug**: Fixed in `coqui-tts>=0.26.2`
3. **Clean install**: Remove TTS/trainer/coqpit before installing coqui-tts
4. **Python version**: 3.11 recommended (3.12 works with fork)
5. **CUDA Sysmem Fallback**: Update NVIDIA drivers for Windows VRAM extension
6. **Colab sessions**: ~8 hour limit - save checkpoints

---

## Workflow Summary

1. **Local**: Preprocess audio, run F5-TTS zero-shot
2. **Colab**: Upload data, fine-tune XTTS v2 (batch_size=4)
3. **Local**: Download model, run inference
4. **Optional**: Build UI on top of API
