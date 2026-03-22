# AGENTS.md - TTS Project Agent Guidelines

> **Note:** LSP errors in `F5-TTS/` submodule are false positives. The LSP doesn't resolve the `tts` conda environment. Code works correctly when run with: `conda activate tts`

## Overview
This repository contains F5-TTS and XTTS v2 TTS systems for voice cloning and speech synthesis.

## Project Structure
```
skibidi/
├── f5_inference.py           # F5-TTS zero-shot inference
├── xtts_inference.py         # XTTS v2 inference
├── preprocess_audio.py       # Audio preprocessing
├── test_tts.py              # System tests
├── test_f5_inference.py      # F5-TTS specific tests
├── test_xtts_inference.py    # XTTS specific tests
├── F5-TTS/                  # F5-TTS submodule (do not modify)
└── data/                    # Audio data directories
```

## Environment
- **Conda env**: `tts`
- **Python**: 3.11
- **PyTorch**: 2.6.0+cu124
- **CUDA**: 12.4

## Build/Lint/Test Commands

### Activate Environment
```bash
conda activate tts
```

### Run All Tests
```bash
C:/Users/Ishan/anaconda3/envs/tts/python.exe test_tts.py
```

### Run Single Test
```bash
C:/Users/Ishan/anaconda3/envs/tts/python.exe test_f5_inference.py
C:/Users/Ishan/anaconda3/envs/tts/python.exe test_xtts_inference.py
```

### Quick Import Check
```bash
conda run -n tts python -c "from f5_tts.api import F5TTS; from TTS.api import TTS"
```

### Audio Generation Tests
```bash
# F5-TTS
C:/Users/Ishan/anaconda3/envs/tts/python.exe f5_inference.py -r ref.wav -t "Text" -g "Generated text"

# XTTS
C:/Users/Ishan/anaconda3/envs/tts/python.exe xtts_inference.py -t "Text" -s ref.wav -l en
```

## Code Style Guidelines

### Python Version & Conventions
- Target: Python 3.11+
- Use type hints where beneficial
- No trailing whitespace
- Line length: 100 characters max

### Imports
```python
# Standard library first
import os
import sys
from typing import Optional

# Third-party
import torch
import numpy as np

# Local
from f5_tts.api import F5TTS
from TTS.api import TTS
```

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `generate_speech` |
| Classes | PascalCase | `F5TTSInference` |
| Constants | UPPER_SNAKE | `MAX_SAMPLE_RATE` |
| Variables | snake_case | `output_path` |
| Arguments | snake_case | `ref_audio` |
| Private | _leading_underscore | `_internal_state` |

### Type Hints
```python
def generate_speech(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    output_path: Optional[str] = None,
) -> tuple[np.ndarray, int]:
    ...
```

### Docstrings
```python
def preprocess_audio(input_path: str, output_path: str, target_sr: int = 22050) -> None:
    """Preprocess audio file to TTS-ready format.
    
    Args:
        input_path: Path to input audio file.
        output_path: Path to save processed audio.
        target_sr: Target sample rate (default 22050Hz).
    
    Returns:
        None
    """
```

### Error Handling
```python
try:
    audio = AudioSegment.from_file(input_path)
except FileNotFoundError:
    print(f"Error: File not found: {input_path}")
    raise
except Exception as e:
    print(f"Error processing {input_path}: {e}")
    raise
```

### File Paths
- Use forward slashes (`/`) for cross-platform compatibility
- Use `os.path.join()` for path construction
- Use absolute paths when possible or document relative paths

### Logging/Output
- Use `print()` for user-facing messages
- Include status indicators: `Loading...`, `Processing...`, `Done`
- Print file paths when saving outputs

## Key Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| f5-tts | 1.1.17 | Zero-shot TTS |
| coqui-tts | 0.27.5 | XTTS v2 inference |
| torch | 2.6.0+cu124 | GPU acceleration |
| transformers | 4.57.x | Model loading |
| pydub | - | Audio processing |
| soundfile | - | Audio I/O |

## GPU Usage
- Check CUDA: `torch.cuda.is_available()`
- Device selection: `"cuda" if torch.cuda.is_available() else "cpu"`
- Clear cache: `torch.cuda.empty_cache()`

## Audio Format Requirements
- TTS-ready: WAV, 16-bit, 22050Hz, mono
- Reference audio: Any common format (auto-converted)

## Common Tasks

### Add New TTS Model
1. Create new inference script following `f5_inference.py` pattern
2. Add model initialization in `__init__`
3. Add CLI argument parsing
4. Add test script

### Modify Audio Preprocessing
1. Edit `preprocess_audio.py`
2. Test with various formats: WAV, MP3, FLAC, M4A
3. Verify sample rate conversion

### Debug Model Loading
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```
